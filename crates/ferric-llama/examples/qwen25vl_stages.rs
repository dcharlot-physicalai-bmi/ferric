//! **Qwen2.5-VL (MiMo-Embodied-7B) against its authors' code, stage by stage** — the Ferric half of
//! `scripts/vl_conformance.sh`.
//!
//! Runs the whole path from the image FILE: Ferric's own preprocessing, the vision tower, the window
//! un-permutation, the splice into the prompt, Ferric's own mRoPE index, and the language model to
//! logits — straight from the authors' safetensors directory (no conversion step). It prints Ferric's
//! values at exactly the rows and columns the fixture recorded and does not judge; the gate compares,
//! so the same output serves the clean run and every negative control.
//!
//!   cargo run -p ferric-llama --release --example qwen25vl_stages -- <checkpoint-dir> <image.ppm> <fixture.json>
//!
//! Output, one record per line:
//! ```text
//! GRID <t> <h> <w>
//! WINDOW <merged token index in window order> …
//! POS <t|h|w> <position> …                        Ferric's rope index, per token
//! PIXELS <row> <sum> <ssq> <v_col0> …              sum/ssq over the WHOLE row
//! STAGE <stage> <row> <sum> <ssq> <v_col0> …
//! ROW <t> <argmax> <sum> <ssq> <v_id0> …           logits, as lm_logits prints them
//! ```
//! `FERRIC_VL_TOWER_ONLY=1` stops after the tower (no POS/ROW records).
//!
//! Controls: `FERRIC_VL_NEG` (tower: see `qwen25vl_vision::Neg`) and `FERRIC_VL_LM_NEG` (here):
//! `pos1d` (every token takes a text position), `advance_tokens` (after the image the position
//! advances by its token count, not by max(h, w) / merge), `imrope` (the interleaved sector rule).
use ferric_llama::qwen25vl_vision::VisionTower;
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_llama::qwen3vl_image::{preprocess, PreprocCfg, KEYS_DEFAULT};
use ferric_llama::qwen3vl_rope::rope_index;
use ferric_load::hf::HfCheckpoint;
use ferric_tensor::image::read_ppm;
use ferric_tensor::Tensor;
use serde_json::Value;
use std::sync::Arc;

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array().expect("array").iter().map(|x| x.as_u64().expect("uint") as u32).collect()
}

fn print_rows(tag: &str, x: &[f32], cols_n: usize, cols: &[u32]) {
    for (r, row) in x.chunks_exact(cols_n).enumerate() {
        let (mut s, mut q) = (0f64, 0f64);
        for &v in row { s += v as f64; q += (v as f64) * (v as f64); }
        let vals: Vec<String> = cols.iter().map(|&c| format!("{:e}", row[c as usize])).collect();
        println!("{tag} {r} {s:e} {q:e} {}", vals.join(" "));
    }
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let (dir, img_path, fx_path) = (&a[1], &a[2], &a[3]);
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(fx_path).expect("fixture")).expect("json");
    let ids = u32s(&fx["ids"]);
    let types: Vec<u8> = u32s(&fx["types"]).into_iter().map(|x| x as u8).collect();
    let lm_neg = std::env::var("FERRIC_VL_LM_NEG").unwrap_or_default();
    if !matches!(lm_neg.as_str(), "" | "pos1d" | "advance_tokens" | "imrope") {
        panic!("FERRIC_VL_LM_NEG={lm_neg} is not a control (pos1d|advance_tokens|imrope)");
    }

    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    eprintln!("adapter: {} [{:?}]", ctx.adapter_name, ctx.backend);

    // ---- preprocessing, from the image FILE -------------------------------------------------------
    let pcfg = PreprocCfg::load(dir).expect("preprocessor_config.json");
    let img = read_ppm(&std::fs::read(img_path).expect("image")).expect("P6 ppm");
    let (px, plan) = preprocess(&img, &pcfg, KEYS_DEFAULT).expect("preprocess");
    let (gh, gw) = (plan.grid_h, plan.grid_w);
    println!("GRID 1 {gh} {gw}");
    let row_w = 3 * pcfg.temporal_patch * pcfg.patch * pcfg.patch;
    print_rows("PIXELS", &px, row_w, &u32s(&fx["pixels"]["cols"]));

    // ---- the tower ---------------------------------------------------------------------------------
    let tower = VisionTower::load(&ctx, dir).expect("vision tower");
    let merge = tower.cfg.merge;
    let (order, _) = ferric_llama::qwen25vl_vision::window_index(gh, gw, merge, tower.cfg.window_merged());
    println!("WINDOW {}", order.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
    let pxt = Tensor::from_vec(&ctx, &px, &[gh * gw, row_w]);
    let mut taps = Vec::new();
    let image_rows = tower.encode_patches(&pxt, gh, gw, &mut taps).expect("encode");
    for (name, t) in &taps {
        let Some(rec) = fx["stages"].get(name) else { continue };
        let v = t.to_vec().await;
        print_rows(&format!("STAGE {name}"), &v, t.shape[1], &u32s(&rec["cols"]));
    }
    drop(taps);
    drop(tower); // 2.7 GB of f32 tower weights, not needed beside the language model

    // A tower control cannot move anything before the tower's output, and the language model is most of
    // the run time — so a gate running only tower controls can stop here.
    if std::env::var("FERRIC_VL_TOWER_ONLY").is_ok() { return; }

    // ---- positions: Ferric's own rope index on the authors' token types -----------------------------
    let t_n = ids.len();
    let ri = rope_index(&types, &[(1, gh, gw)], merge).expect("rope_index");
    let (mut pt, mut ph, mut pw) = (ri.t.clone(), ri.h.clone(), ri.w.clone());
    match lm_neg.as_str() {
        "pos1d" => { for v in [&mut pt, &mut ph, &mut pw] { *v = (0..t_n as i64).collect(); } }
        "advance_tokens" => {
            // Shift every text token after the image by (tokens - max(h,w)/merge).
            let n_img = gh * gw / (merge * merge);
            let extra = (n_img - gh.max(gw) / merge) as i64;
            let end = types.iter().rposition(|&x| x == 1).expect("an image run") + 1;
            for v in [&mut pt, &mut ph, &mut pw] { for x in &mut v[end..] { *x += extra; } }
        }
        _ => {}
    }
    for (tag, v) in [("t", &pt), ("h", &ph), ("w", &pw)] {
        println!("POS {tag} {}", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
    }
    let mut mrope: Vec<u32> = Vec::with_capacity(4 * t_n);
    for v in [&pt, &ph, &pw] { mrope.extend(v.iter().map(|&x| x as u32)); }
    mrope.extend(std::iter::repeat_n(0u32, t_n));

    // ---- the language model, from the same directory ------------------------------------------------
    let hf = HfCheckpoint::open(dir).expect("HF checkpoint");
    let mut model = Qwen3::load(&ctx, &hf).expect("Qwen3::load");
    if lm_neg == "imrope" { model.cfg.mrope_interleaved = true; }
    let img_start = types.iter().position(|&x| x == 1).expect("an image run");
    let spliced = model.splice_image_embeds(&model.embed_tokens(&ids), img_start, &image_rows);
    let mut cache = Cache::new(&model.cfg);
    let h = model.forward_embeds_mm(&spliced, &mut cache, &mrope, &[], img_start, &mut Vec::new());
    let lg = model.logits_from_normed(&h).to_vec().await;
    let sample = u32s(&fx["sample_ids"]);
    let v = lg.len() / t_n;
    for t in 0..t_n {
        let r = &lg[t * v..(t + 1) * v];
        let (mut best, mut bv, mut s, mut q) = (0usize, f32::NEG_INFINITY, 0f64, 0f64);
        for (i, &x) in r.iter().enumerate() {
            if x > bv { bv = x; best = i; }
            s += x as f64; q += (x as f64) * (x as f64);
        }
        let vals: Vec<String> = sample.iter().map(|&i| format!("{:.5}", r[i as usize])).collect();
        println!("ROW {t} {best} {s:.4} {q:.4} {}", vals.join(" "));
    }
}
