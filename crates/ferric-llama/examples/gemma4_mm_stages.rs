//! **Gemma 4 image and audio inputs against the authors' code, stage by stage** — the Ferric half of
//! `scripts/gemma4_mm_conformance.sh`.
//!
//! Runs the whole path from the FILE: Ferric's own preprocessing (the authors' resize, bit-exact; or their
//! log-mel front end), the tower, the projection into the text width, the splice over the placeholder run,
//! and the language model to logits. It prints Ferric's values at exactly the rows and columns the fixture
//! recorded and does not judge; the gate compares, so one output serves the clean run and every control.
//!
//!   cargo run -p ferric-llama --release --example gemma4_mm_stages -- <tower> <text.gguf> <file> <fixture.json>
//!
//! `<tower>` is the authors' checkpoint DIRECTORY or a ggml-org `mmproj-*.gguf`; `<file>` a P6 PPM (image
//! fixtures) or a 16 kHz WAV (audio fixtures). Output, one record per line:
//! ```text
//! GRID <rows> <cols>                               image: the patch grid
//! FRAMES <n> <valid>                               audio: mel frames, valid frames
//! FEAT <row> <sum> <ssq> <v_col0> …                audio: the log-mel rows
//! STAGE <stage> <row> <sum> <ssq> <v_col0> …       at the fixture's rows and columns
//! ROW <t> <argmax> <sum> <ssq> <v_id0> …           logits of the multimodal prompt
//! TROW <t> <argmax> <sum> <ssq> <v_id0> …          logits of the same question with no image
//! ```
//! `FERRIC_G4_TOWER_ONLY=1` stops after the tower. `FERRIC_G4_LEVELS_OUT=<path>` writes the resized image's
//! 8-bit levels (HWC) there, for the gate to hash against the authors'.
//!
//! Controls: `FERRIC_G4V_NEG` (vision tower, see `gemma4_vision::Neg`), `FERRIC_G4A_NEG` (audio tower,
//! `gemma4_audio::Neg`), and `FERRIC_G4_SPLICE_NEG` (here): `ple_mm_id` (the per-layer lookup reads the
//! placeholder's own id, not PAD), `scale_soft` (the soft rows are multiplied by sqrt(d) like text rows),
//! `shift` (the soft rows land one position late).
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_llama::gemma4::{Cache, Gemma4};
use ferric_llama::gemma4_mm::Src;
use ferric_tensor::Tensor;
use serde_json::Value;
use std::sync::Arc;

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array().expect("array").iter().map(|x| x.as_u64().expect("uint") as u32).collect()
}

/// Print the fixture's rows (`rows`) of a `[n, w]` host matrix: sum, sum of squares, and `cols`.
fn print_rows(tag: &str, x: &[f32], w: usize, rows: &[u32], cols: &[u32]) {
    for &r in rows {
        let row = &x[r as usize * w..(r as usize + 1) * w];
        let (mut s, mut q) = (0f64, 0f64);
        for &v in row { s += v as f64; q += (v as f64) * (v as f64); }
        let vals: Vec<String> = cols.iter().map(|&c| format!("{:e}", row[c as usize])).collect();
        println!("{tag} {r} {s:e} {q:e} {}", vals.join(" "));
    }
}

fn print_logits(tag: &str, lg: &[f32], t_n: usize, sample: &[u32]) {
    let v = lg.len() / t_n;
    for t in 0..t_n {
        let r = &lg[t * v..(t + 1) * v];
        let (mut best, mut bv, mut s, mut q) = (0usize, f32::NEG_INFINITY, 0f64, 0f64);
        for (i, &x) in r.iter().enumerate() {
            if x > bv { bv = x; best = i; }
            s += x as f64; q += (x as f64) * (x as f64);
        }
        let vals: Vec<String> = sample.iter().map(|&i| format!("{:.6}", r[i as usize])).collect();
        println!("{tag} {t} {best} {s:.6} {q:.6} {}", vals.join(" "));
    }
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 { eprintln!("usage: gemma4_mm_stages <tower dir|mmproj.gguf> <text.gguf> <file> <fixture.json>"); std::process::exit(2); }
    let (tower_path, text_path, file, fx_path) = (&a[1], &a[2], &a[3], &a[4]);
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(fx_path).expect("fixture")).expect("json");
    let kind = fx["kind"].as_str().expect("fixture kind").to_string();
    let splice_neg = std::env::var("FERRIC_G4_SPLICE_NEG").unwrap_or_default();
    if !matches!(splice_neg.as_str(), "" | "ple_mm_id" | "scale_soft" | "shift") {
        panic!("FERRIC_G4_SPLICE_NEG={splice_neg} is not a control (ple_mm_id|scale_soft|shift)");
    }

    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    eprintln!("adapter: {} [{:?}]", ctx.adapter_name, ctx.backend);
    let src = Src::open(tower_path).expect("tower source");
    eprintln!("tower weights: {}", if src.is_gguf() { "mmproj GGUF" } else { "the authors' safetensors" });

    let stage_out = |name: &str, t: &Tensor, v: &[f32]| {
        if let Some(rec) = fx["stages"].get(name) {
            print_rows(&format!("STAGE {name}"), v, t.shape[1], &u32s(&rec["rows"]), &u32s(&rec["cols"]));
        }
    };

    // ---- the tower, from the FILE ------------------------------------------------------------------
    let (soft, mm_tok): (Tensor, u32) = if kind == "image" {
        let pc = match &src { Src::Hf { .. } => ferric_llama::gemma4_vision::PreCfg::load(tower_path).expect("processor_config.json"),
                              Src::Gguf(_) => Default::default() };
        let img = ferric_tensor::image::read_ppm(&std::fs::read(file).expect("image")).expect("P6 ppm");
        let p = ferric_llama::gemma4_vision::preprocess(&img, &pc).expect("preprocess");
        println!("GRID {} {}", p.gh, p.gw);
        if let Ok(out) = std::env::var("FERRIC_G4_LEVELS_OUT") { std::fs::write(out, &p.resized.px).expect("levels out"); }
        let tower = ferric_llama::gemma4_vision::VisionTower::load(&ctx, &src).expect("vision tower");
        if tower.neg != ferric_llama::gemma4_vision::Neg::None { eprintln!("⚠ vision control: {:?}", tower.neg); }
        let mut taps = Vec::new();
        let t0 = std::time::Instant::now();
        let soft = tower.encode(&p, Some(&mut taps)).expect("encode");
        let _ = soft.to_vec().await;
        eprintln!("vision tower: {} patches -> {} soft tokens in {:.1} ms", p.n(), soft.shape[0], t0.elapsed().as_secs_f64() * 1e3);
        for (name, t) in &taps { stage_out(name, t, &t.to_vec().await); }
        (soft, fx["image_token_id"].as_u64().expect("image_token_id") as u32)
    } else {
        panic!("audio fixtures: not wired yet");
    };
    if std::env::var("FERRIC_G4_TOWER_ONLY").is_ok() { return; }

    // ---- the splice and the language model ---------------------------------------------------------
    let g = GgufFile::open(text_path).expect("text GGUF");
    let pad = match g.metadata().get("tokenizer.ggml.padding_token_id") { Some(Meta::U(x)) => *x as u32, _ => 0 };
    let model = Gemma4::load(&ctx, &g).expect("Gemma4::load");
    drop(g);
    let ids = u32s(&fx["ids"]);
    let at: Vec<usize> = ids.iter().enumerate().filter(|(_, x)| **x == mm_tok).map(|(i, _)| i).collect();
    assert_eq!(at.len(), soft.shape[0], "{} placeholder tokens for {} soft rows", at.len(), soft.shape[0]);
    assert!(at.windows(2).all(|w| w[1] == w[0] + 1), "one contiguous placeholder run");
    let start = at[0] + (splice_neg == "shift") as usize;
    let d = model.cfg.d;
    let soft = if splice_neg == "scale_soft" { soft.mul(&soft.scalar((d as f32).sqrt())) } else { soft };
    let emb = model.embed(&ids);
    let spliced = ferric_llama::gemma4_mm::splice_rows(&emb, start, &soft);
    let ple_ids: Vec<u32> = ids.iter().map(|&x| if x == mm_tok && splice_neg != "ple_mm_id" { pad } else { x }).collect();
    let mut cache = Cache::new(&model.cfg);
    let h = model.forward_hidden_embeds(&ple_ids, spliced, &mut cache);
    let lg = model.logits(h).to_vec().await;
    let sample = u32s(&fx["sample_ids"]);
    print_logits("ROW", &lg, ids.len(), &sample);
    if let Some(tids) = fx.get("text_ids") {
        let tids = u32s(tids);
        let mut c2 = Cache::new(&model.cfg);
        let lg = model.forward(&tids, &mut c2).to_vec().await;
        print_logits("TROW", &lg, tids.len(), &sample);
    }
}
