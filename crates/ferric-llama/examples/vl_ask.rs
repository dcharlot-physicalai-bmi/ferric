//! **Ask MiMo-Embodied-7B about an image, on Ferric** — Qwen2.5-VL end to end, generating.
//!
//!   cargo run -p ferric-llama --release --example vl_ask -- <checkpoint-dir> <image.ppm> "<question>" [max_new] [fixture.json]
//!
//! Reads the authors' safetensors directly (no conversion), builds the prompt the authors' chat
//! template builds, runs the image through Ferric's preprocessing and vision tower, and decodes greedily
//! with the KV cache. The prefill is what `scripts/vl_conformance.sh` verifies against the authors' code;
//! the decode steps continue it with one token and one mRoPE position each.
//!
//! ⛔ A decode step's position is NOT its index in the cache. After an image the sequence position runs
//! behind the token count (an image of `n` tokens advances it by `max(h, w) / merge`), so step `k`
//! rotates by `prompt_len + delta + k`, where `delta = max(position) + 1 - prompt_len` — negative here.
//! Rotating by the cache index instead is fluent and wrong, which is why the positions are passed
//! explicitly rather than derived.
//!
//! With a fixture (from `refgen/qwen25vl_ref.py`) whose question matches, the prompt ids built here are
//! checked against the authors' processor's ids before anything runs; with a `--greedy` fixture, the
//! generated ids are then printed as `MATCH <n>/<len>` against the continuation the authors' code
//! confirmed (the gate reads that line). `FERRIC_VL_LM_NEG=decode_cachepos` rotates decode steps by the
//! cache index instead — the control for the note above.
use ferric_llama::qwen25vl_vision::VisionTower;
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_llama::qwen3vl_image::{preprocess, PreprocCfg, KEYS_DEFAULT};
use ferric_llama::qwen3vl_rope::rope_index;
use ferric_load::hf::HfCheckpoint;
use ferric_tensor::image::read_ppm;
use ferric_tensor::Tensor;
use ferric_tokenizer::{Bpe, Pre};
use std::sync::Arc;
use std::time::Instant;

// The special tokens the chat template writes, by id (config.json names the vision ones).
const IM_START: u32 = 151644;
const IM_END: u32 = 151645;
const ENDOFTEXT: u32 = 151643;
// The system turn MiMo's `chat_template.json` inserts when the conversation has none.
const SYSTEM: &str = "You are MiMo, an AI assistant developed by Xiaomi.";

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let (dir, img_path, question) = (&a[1], &a[2], &a[3]);
    let max_new: usize = a.get(4).map(|s| s.parse().expect("max_new")).unwrap_or(48);
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json"))
        .expect("config.json")).expect("json");
    let id = |k: &str| cfg[k].as_u64().unwrap_or_else(|| panic!("config.json: {k}")) as u32;
    let (vs, ve, img_tok) = (id("vision_start_token_id"), id("vision_end_token_id"), id("image_token_id"));
    let tok = Bpe::from_tokenizer_json_with_pre(&std::fs::read(format!("{dir}/tokenizer.json"))
        .expect("tokenizer.json"), Pre::Qwen2).expect("tokenizer");

    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    let t0 = Instant::now();
    let pcfg = PreprocCfg::load(dir).expect("preprocessor_config.json");
    let img = read_ppm(&std::fs::read(img_path).expect("image")).expect("P6 ppm");
    let (px, plan) = preprocess(&img, &pcfg, KEYS_DEFAULT).expect("preprocess");
    let n_img = plan.tokens(pcfg.merge);

    // <|im_start|>system\n…<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|>*n<|vision_end|>…
    let mut ids = vec![IM_START];
    ids.extend(tok.encode(&format!("system\n{SYSTEM}")));
    ids.push(IM_END);
    ids.extend(tok.encode("\n"));
    ids.push(IM_START);
    ids.extend(tok.encode("user\n"));
    ids.push(vs);
    ids.extend(std::iter::repeat_n(img_tok, n_img));
    ids.push(ve);
    ids.extend(tok.encode(question));
    ids.push(IM_END);
    ids.extend(tok.encode("\n"));
    ids.push(IM_START);
    ids.extend(tok.encode("assistant\n"));
    let fx: Option<serde_json::Value> = a.get(5).map(|p| serde_json::from_str(&std::fs::read_to_string(p)
        .expect("fixture")).expect("json"));
    let u32v = |v: &serde_json::Value| -> Vec<u32> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect() };
    if let Some(f) = fx.as_ref().filter(|f| f["question"].as_str() == Some(question.as_str())) {
        let want = u32v(if f["prompt_ids"].is_array() { &f["prompt_ids"] } else { &f["ids"] });
        assert_eq!(ids, want, "the prompt differs from the authors' processor's ids");
        eprintln!("prompt ids: identical to the authors' processor ({} tokens)", ids.len());
    }
    let types: Vec<u8> = ids.iter().map(|&x| (x == img_tok) as u8).collect();
    let img_start = ids.iter().position(|&x| x == img_tok).unwrap();

    let tower = VisionTower::load(&ctx, dir).expect("vision tower");
    let (gh, gw) = (plan.grid_h, plan.grid_w);
    let pxt = Tensor::from_vec(&ctx, &px, &[gh * gw, 3 * pcfg.temporal_patch * pcfg.patch * pcfg.patch]);
    let rows = tower.encode_patches(&pxt, gh, gw, &mut Vec::new()).expect("encode");
    drop(tower);
    let ri = rope_index(&types, &[(1, gh, gw)], pcfg.merge).expect("rope_index");
    let t_n = ids.len();
    let mut mrope: Vec<u32> = Vec::with_capacity(4 * t_n);
    for v in [&ri.t, &ri.h, &ri.w] { mrope.extend(v.iter().map(|&x| x as u32)); }
    mrope.extend(std::iter::repeat_n(0u32, t_n));

    let hf = HfCheckpoint::open(dir).expect("checkpoint");
    let model = Qwen3::load(&ctx, &hf).expect("load");
    eprintln!("loaded in {:.1} s; image {}x{} -> {} tokens; prompt {} tokens",
              t0.elapsed().as_secs_f64(), img.w, img.h, n_img, t_n);

    let t1 = Instant::now();
    let mut cache = Cache::new(&model.cfg);
    let x = model.splice_image_embeds(&model.embed_tokens(&ids), img_start, &rows);
    let h = model.forward_embeds_mm(&x, &mut cache, &mrope, &[], img_start, &mut Vec::new());
    let last = h.narrow(0, t_n - 1, 1).contiguous();
    let mut next = argmax(&model.logits_from_normed(&last).to_vec().await);
    let prefill = t1.elapsed().as_secs_f64();

    let t2 = Instant::now();
    let mut out = Vec::new();
    let neg = std::env::var("FERRIC_VL_LM_NEG").unwrap_or_default();
    let mut pos = if neg == "decode_cachepos" { t_n as i64 } else { t_n as i64 + ri.delta };
    while out.len() < max_new && next != IM_END && next != ENDOFTEXT {
        out.push(next);
        let p = pos as u32;
        let h = model.forward_embeds_mm(&model.embed_tokens(&[next]), &mut cache, &[p, p, p, 0], &[], 0,
                                        &mut Vec::new());
        next = argmax(&model.logits_from_normed(&h).to_vec().await);
        pos += 1;
    }
    let dec = t2.elapsed().as_secs_f64();
    println!("{}", tok.decode(&out));
    eprintln!("ids: {out:?}");
    if let Some(want) = fx.as_ref().filter(|f| f["continuation"].is_array()).map(|f| u32v(&f["continuation"])) {
        let n = out.iter().zip(&want).take_while(|(a, b)| a == b).count();
        println!("MATCH {n}/{}", want.len());
    }
    eprintln!("prefill {t_n} tokens {prefill:.2} s; decode {} tokens {dec:.2} s ({:.1} tok/s)",
              out.len(), out.len() as f64 / dec.max(1e-9));
}
