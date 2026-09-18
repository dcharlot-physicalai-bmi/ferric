//! **Qwen3-VL's text path, straight from the published checkpoint, against published logits.**
//!
//! Ferric registers `qwen3vl` but had never RUN one: no GGUF of it exists on the machine that wrote
//! the port, so every check so far was of the *rule* (multimodal RoPE, mutation-tested against an
//! independent reference) and none was of a forward pass. This closes that.
//!
//! `HfCheckpoint` presents `config.json` + `model.safetensors` under GGUF names, so this drives the
//! SAME `Qwen3::load` a GGUF drives — no conversion step, and no llama.cpp converter.
//!
//! ## The oracle, and what it can and cannot settle
//!
//! `qwen3vl_ref.bin` is the last-token logit vector from **HuggingFace `transformers`**, the
//! implementation these weights were published against. ⚠ **Ferric is NOT expected to match it
//! exactly.** Ferric follows llama.cpp's interleaved-mRoPE sector rule, which leaves rotary sectors
//! 61 and 62 unrotated where HF rotates them by the temporal position — a real, reproduced
//! divergence between the two references (~4e-4 rad at position 1000, far less at these positions).
//! So the pass condition is: **the same argmax, and a small bounded deviation** — not equality.
//! A perfect match would mean the divergence analysis is wrong and is worth investigating, which is
//! why this prints the deviation rather than only asserting on it.
//!
//!   cargo run -p ferric-llama --example qwen3vl_hf --release -- <checkpoint-dir> <ref.bin>
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_load::hf::HfCheckpoint;
use std::sync::Arc;

fn read_f32s(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    assert_eq!(b.len(), 4 + n * 4, "{path}: header says {n} floats");
    b[4..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let mut a = std::env::args().skip(1);
    let dir = a.next().expect("usage: qwen3vl_hf <checkpoint-dir> <ref.bin> [ids.csv]");
    let refp = a.next().expect("usage: qwen3vl_hf <checkpoint-dir> <ref.bin> [ids.csv]");
    // ⛔ An optional id list, because the DEFAULT PROMPT CANNOT SEE THE THING THIS PORT IS ABOUT.
    // The llama.cpp/HF sector divergence scales with position: ~1.6e-6 rad at position 4, ~4.2e-4 at
    // 1024. A 5-token prompt puts it two orders of magnitude under f32 noise, so agreement there is
    // evidence about the weights and the wiring — and says nothing either way about the sector rule.
    // Pass a long id list to reach the operating point where the difference is observable.
    let idsp = a.next();
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    println!("adapter   : {} [{:?}]", ctx.adapter_name, ctx.backend);

    let hf = HfCheckpoint::open(&dir).expect("open HF checkpoint");
    println!("checkpoint: {dir}\n  model_type {} · {} tensors under GGUF names", hf.arch, hf.names().count());

    let model = Qwen3::load(&ctx, &hf).expect("Qwen3::load on a qwen3vl checkpoint");
    // "The capital of France is" — the same ids the reference was run on, unless one is supplied.
    let ids: Vec<u32> = match &idsp {
        Some(f) => std::fs::read_to_string(f).unwrap_or_else(|e| panic!("read {f}: {e}"))
            .trim().split(',').map(|t| t.trim().parse().expect("id")).collect(),
        None => vec![785, 6722, 315, 9625, 374],
    };
    println!("  prompt    : {} tokens (last position {})", ids.len(), ids.len() - 1);
    let mut cache = Cache::new(&model.cfg);
    let all = model.forward_cached(&ids, &mut cache).to_vec().await;
    let want = read_f32s(&refp);
    // ⚠ `forward_cached` returns [T, n_vocab] — every position, not just the last. The reference is
    // the LAST token's row; comparing the flat buffers instead just fails on length, but taking the
    // FIRST n_vocab would compare position 0 against position 4 and look like a numerics bug.
    assert_eq!(all.len() % want.len(), 0, "logit buffer {} is not a multiple of {}", all.len(), want.len());
    let t = all.len() / want.len();
    assert_eq!(t, ids.len(), "expected one logit row per token");
    let logits = &all[(t - 1) * want.len()..];

    let (mut worst, mut scale) = (0f32, 0f32);
    for (g, w) in logits.iter().zip(&want) {
        worst = worst.max((g - w).abs());
        scale = scale.max(w.abs());
    }
    let am_g = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let am_w = want.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    println!("\n  HF argmax     {am_w}");
    println!("  Ferric argmax {am_g}");
    println!("  max |Δ| = {worst:.4e} on |logit| up to {scale:.4e}  (rel {:.2e})", worst / scale);

    assert_eq!(am_g, am_w, "the predicted token must match the published implementation");
    // ⚠ A bound, not equality — see the header. Loose enough for the known sector divergence and f32
    // accumulation over 28 layers, tight enough that a transposed or swapped weight cannot pass:
    // those produce O(1) logit changes, not O(1e-2).
    assert!(worst / scale < 1e-2, "deviation {worst:.3e} exceeds what the known divergence explains");
    println!("\n✅ text path matches the published implementation to {:.2e} relative", worst / scale);
}
