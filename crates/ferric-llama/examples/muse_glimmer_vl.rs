//! **Muse Glimmer, vision + language, end to end** — an image and a question in, a caption out.
//!
//! This is the test that actually validates the vision tower. The encoder on its own can only be
//! checked for shape and finiteness, and a 1024×6656 block of finite numbers is not evidence of a
//! correct ViT. Splicing those rows into the text sequence and reading the caption is: if the 2-D
//! RoPE halves are swapped, the window permutation is wrong, the pixel shuffle is transposed, or the
//! adapter is mis-ordered, the model describes *something*, fluently, and it is not this image.
//!
//! Reference for the same image, from `llama-mtmd-cli` (needs `--jinja`):
//!   "abstract 3D glassy pill/capsule shapes in blue purple gradient with reflection"
//!
//! ## How the image enters the sequence
//!
//! The chat template renders an image part as a single `<|patch|>` token. That one token is replaced
//! by the encoder's `n_out` embedding ROWS — so the sequence is text rows, then image rows, then more
//! text rows, and the model sees one continuous stream. Nothing downstream knows an image happened,
//! which is why `forward_embeds` needed to exist: `forward_cached` can only take token ids.
//!
//!   cargo run -p ferric-llama --example muse_glimmer_vl --release -- \
//!       <model.gguf> <mmproj.gguf> <image.ppm> <ids.txt> <patch_index> [n]
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::glimmer_vision::VisionTower;
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tensor::Tensor;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let model = a.get(1).expect("usage: <model.gguf> <mmproj.gguf> <image.ppm> <ids.txt> <patch_idx> [n]");
    let mmproj = a.get(2).expect("mmproj");
    let imgp = a.get(3).expect("image.ppm");
    let idsp = a.get(4).expect("ids.txt (comma-separated, from the rendered chat template)");
    let patch_at: usize = a.get(5).expect("index of the <|patch|> token").parse().expect("index");
    let n_gen: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(48);

    let ctx = Arc::new(Context::new().await.unwrap());
    let g = GgufFile::open(model).expect("open model");
    let tokens: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokens"),
    };

    let ids: Vec<u32> = std::fs::read_to_string(idsp).expect("ids").trim()
        .split(',').map(|s| s.trim().parse().expect("id")).collect();
    assert!(patch_at < ids.len(), "patch index {patch_at} outside a {}-token prompt", ids.len());

    // ENCODE FIRST, THEN LOAD THE LLM. The vision tower's ~1.4 GB of weights are dead the moment
    // encode returns, and holding them alongside a 17 GB text model plus a 1076-row prefill gets the
    // process SIGKILLed (exit 137, no message). The image rows own their own buffer, so dropping the
    // tower here is safe; `flush` makes the release actually happen before the next allocation burst.
    let t0 = std::time::Instant::now();
    let vt = VisionTower::load(&ctx, mmproj).expect("load mmproj");
    println!("vision {} layers d={} grid {}×{} -> {} tokens (loaded {:.2?})",
             vt.n_layer, vt.d, vt.grid, vt.grid, vt.n_out, t0.elapsed());
    let (n_out, proj_dim) = (vt.n_out, vt.proj_dim);
    let t0 = std::time::Instant::now();
    let img_rows = vt.encode_ppm(&std::fs::read(imgp).expect("read image")).expect("encode");
    println!("  vision encode {:.2?} -> [{n_out}, {proj_dim}]", t0.elapsed());
    drop(vt);
    ctx.flush();

    let t0 = std::time::Instant::now();
    let m = Qwen3::load(&ctx, &g).expect("load model");
    println!("  text {} layers d={} (loaded {:.2?})", m.cfg.n_layer, m.cfg.n_embd, t0.elapsed());
    assert_eq!(proj_dim, m.cfg.n_embd,
               "the projector must emit the LLM's embedding width: {proj_dim} vs {}", m.cfg.n_embd);

    // Splice: text rows before the marker, the image rows, then text rows after. The marker token
    // itself is DROPPED — it is a placeholder for the image, not a token the model should see.
    let pre = m.embed_tokens(&ids[..patch_at]);
    let post = m.embed_tokens(&ids[patch_at + 1..]);
    let seq = pre.cat(&img_rows, 0).cat(&post, 0);
    let t_total = seq.shape[0];
    assert_eq!(t_total, ids.len() - 1 + n_out, "spliced length must be text - 1 + image rows");
    println!("  sequence: {} text - 1 marker + {n_out} image = {t_total} rows", ids.len());

    let vn = m.cfg.n_vocab;
    let am = |l: &[f32]| l[l.len() - vn..].iter().enumerate()
        .fold((0usize, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32;

    // CHUNKED PREFILL. One 1076-row forward through a 52-layer 6656-wide model allocates every
    // layer's activations at full sequence length at once and gets SIGKILLed (exit 137, no message).
    // The cache already carries state across calls, so feeding the rows in slices is the same
    // computation with bounded peak memory — only the LAST chunk's logits are needed.
    let t0 = std::time::Instant::now();
    let mut cache = Cache::new(&m.cfg);
    // ⚠ Chunking DOES NOT WORK on this model and the default is therefore the whole sequence.
    // Muse Glimmer's local layers use `causal_attention_win`, which assumes the query block covers
    // the whole history; with t=256 against a cache that has grown to 512 it reshapes to the wrong
    // numel and panics. The non-windowed path is fine (`chunked_attention` delegates to
    // `causal_attention` exactly when q covers the history), so this is a gap in the WINDOWED path,
    // not in chunked prefill as an idea. Left as an env knob so the fix can be tested against it.
    let chunk: usize = std::env::var("FERRIC_PREFILL_CHUNK").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(t_total);
    let mut logits: Vec<f32> = Vec::new();
    let mut off = 0usize;
    while off < t_total {
        let len = chunk.min(t_total - off);
        let part = seq.narrow(0, off, len).contiguous();
        logits = m.forward_embeds(&part, &mut cache).to_vec().await;
        off += len;
    }
    let last = &logits[logits.len() - vn..];
    let bad = last.iter().filter(|v| !v.is_finite()).count();
    let (mn, mx) = last.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    println!("  prefill {:.2?} · first-step logits: {bad} non-finite, min {mn:.3} max {mx:.3}", t0.elapsed());
    assert_eq!(bad, 0, "non-finite logits after the image splice");

    let mut next = am(&logits);
    let mut out = String::new();
    for _ in 0..n_gen {
        if matches!(g.metadata.get("tokenizer.ggml.eos_token_id"), Some(Meta::U(e)) if *e as u32 == next) { break; }
        out.push_str(&tokens[next as usize]);
        let l = m.forward_cached(&[next], &mut cache).to_vec().await;
        next = am(&l);
    }
    // GPT-2 byte-level back to raw bytes.
    let dec: String = {
        let mut mp = std::collections::HashMap::new();
        let mut n = 0u32;
        for b in 0u32..256 {
            let p = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
            let c = if p { b } else { let c = 256 + n; n += 1; c };
            mp.insert(char::from_u32(c).unwrap(), b as u8);
        }
        String::from_utf8_lossy(&out.chars().filter_map(|c| mp.get(&c).copied()).collect::<Vec<u8>>()).into_owned()
    };
    println!("\n--- Ferric ---\n{dec}\n");
    println!("--- reference (llama-mtmd-cli) ---");
    println!("abstract 3D glassy pill/capsule shapes in blue purple gradient with reflection\n");
    println!("Same subject = the whole chain is right: patchify, 2-D RoPE, window slicing,");
    println!("pixel shuffle, adapter, and the splice. Fluent text about something else means one");
    println!("of those is wrong, and which words differ says roughly which.");
}
