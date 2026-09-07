//! **Run a real Hy4-preview checkpoint.** Greedy generation from a prompt, straight through
//! `Hyv4::load` and `Hyv4::decode`.
//!
//! Everything else in this repo that touches hyv4 runs on a synthetic checkpoint or on a few
//! range-read blocks. This is the whole model: 78 blocks, 256 experts, 213.66 GiB.
//!
//! ⭐ `GgufFile` reads each tensor on demand rather than holding the file, so peak HOST memory is
//! one tensor — `ferric_gguf::parse` takes a `Vec<u8>` and would need the whole 213.66 GiB in RAM
//! *before* any of it reached the GPU. On a 256 GiB machine that is the difference between running
//! and thrashing.
//!
//! ⚠ It bypasses `arch::resolve`, which refuses `hyv4` because the row is `Status::Untried`. That
//! refusal is correct — a server must not serve a model whose output nobody has seen — and this
//! example is how somebody sees it. What it prints is evidence for changing that row, not a claim
//! that the row is already wrong.
//!
//! ```text
//! cargo run --release -p ferric-llama --example hyv4_run -- <model.gguf> "prompt" [n_tokens]
//! ```

use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::hyv4::{Hyv4, Hyv4Cache};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let mut a = std::env::args().skip(1);
    let path = match a.next() { Some(p) => p, None => { eprintln!("usage: hyv4_run <model.gguf> [prompt] [n_gen] [budget_gib]\n\
        \x20 budget_gib: stream block weights under this many GiB. OMIT IT AND THE MODEL LOADS\n\
        \x20 RESIDENT, which for Hy4-preview needs more RAM than a 256 GiB machine has — the\n\
        \x20 process is then killed by the OS with NO message at all."); return } };
    let prompt = a.next().unwrap_or_else(|| "The capital of France is".to_string());
    let n_gen: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(24);
    // Fourth arg: a streaming budget in GiB for BLOCK weights. Without it the model is loaded
    // resident, which for Hy4-preview needs more than a 256 GiB machine has.
    let budget_gib: Option<f64> = a.next().and_then(|s| s.parse().ok());

    let t0 = Instant::now();
    let g = match GgufFile::open(&path) { Ok(g) => g, Err(e) => { eprintln!("open {path}: {e}"); return } };
    println!("opened {path} in {:.1}s — {} shard(s)", t0.elapsed().as_secs_f64(), g.shard_count());

    let Ok(ctx) = pollster::block_on(Context::new()) else { eprintln!("no GPU context"); return };
    let ctx = Arc::new(ctx);
    println!("adapter: {} [{:?}]  max binding {:.1} GiB",
             ctx.adapter_name, ctx.backend, ctx.max_binding as f64 / 1073741824.0);

    let t1 = Instant::now();
    let m = match budget_gib {
        Some(gib) => {
            let b = (gib * 1073741824.0) as u64;
            println!("streaming: {gib:.1} GiB budget for block weights");
            match Hyv4::load_streaming(&ctx, &path, b) {
                Ok(m) => m,
                Err(e) => { eprintln!("STREAMING LOAD FAILED: {e}"); std::process::exit(1) }
            }
        }
        None => {
            // ⚠ SAY SO BEFORE TRYING. A resident load of a 213.66 GiB checkpoint is killed by the
            // OS, not by this program: no panic, no error, no exit code — the run simply stops
            // after the adapter line and the log ends mid-sentence. That is indistinguishable
            // from a hang or a crash unless something announced the attempt first.
            let gib = std::fs::metadata(&path).map(|m| m.len() as f64 / 1073741824.0).unwrap_or(0.0);
            println!("resident load of {gib:.1} GiB (no budget given). If this run stops right \
                      here with no error, the OS killed it — pass a budget in GiB as the 4th \
                      argument to stream instead.");
            match Hyv4::load(&ctx, &g) {
                Ok(m) => m,
                Err(e) => { eprintln!("LOAD FAILED: {e}"); std::process::exit(1) }
            }
        }
    };
    println!("loaded {} blocks, d={}, {} experts, vocab {} in {:.1}s{}",
             m.cfg.n_layer, m.cfg.d, m.cfg.n_expert, m.cfg.n_vocab, t1.elapsed().as_secs_f64(),
             if m.stream().is_some() { " (streamed)" } else { " (resident)" });

    // The tokenizer travels in the checkpoint. ⚠ `tokenizer.ggml.pre` is a SEPARATE question from
    // `tokenizer.ggml.model`: hyv4 declares model=gpt2 and pre=hyv4, and reading only the first
    // gives it GPT-2's splitting rule — the exact bug ferric-serve documents for the Qwen family.
    let tokens: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => { eprintln!("checkpoint has no tokenizer.ggml.tokens"); return }
    };
    let vocab: std::collections::HashMap<String, u32> =
        tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
        Some(Meta::Arr(a)) => a.iter().filter_map(|m| if let Meta::Str(s) = m {
            s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None }).collect(),
        _ => Vec::new(),
    };
    let pre = ferric_tokenizer::Pre::from_gguf(
        match g.metadata.get("tokenizer.ggml.pre") { Some(Meta::Str(p)) => Some(p.as_str()), _ => None });
    println!("tokenizer: {} tokens, {} merges, pre={:?}", tokens.len(), merges.len(), pre);
    let tok = ferric_tokenizer::Bpe::new_with_pre(vocab, &merges, pre);
    let ids = tok.encode(&prompt);
    println!("prompt {prompt:?} -> {} tokens {:?}", ids.len(), &ids[..ids.len().min(12)]);

    // ⭐ The comparable quantity. llama.cpp's eval-callback prints `result_output` as first-three,
    // last-three and a SUM over the row; the sum is sensitive to every element, so it is what
    // `scripts/hyv4_vs_reference.sh` gates on. Printing it here is what makes a real-weights
    // comparison against Tencent's implementation possible at all — the synthetic checkpoint is the
    // only thing that has ever been compared.
    let mut cache = Hyv4Cache::new(&m).expect("cache");
    let mut out: Vec<u32> = Vec::new();
    let t2 = Instant::now();
    let mut logits = pollster::block_on(m.decode(&ids, &mut cache).to_vec());
    let prefill = t2.elapsed().as_secs_f64();
    {
        let row = &logits[logits.len() - m.cfg.n_vocab..];
        let (mn, mx) = row.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        println!("prefill last-row logits: sum {:.6}  min {mn:.6}  max {mx:.6}  first3 {:?}",
                 row.iter().map(|v| *v as f64).sum::<f64>(),
                 &row[..3].iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>());
    }
    for i in 0..n_gen {
        let row = &logits[logits.len() - m.cfg.n_vocab..];
        let next = row.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0 as u32;
        out.push(next);
        if i + 1 == n_gen { break }
        logits = pollster::block_on(m.decode(&[next], &mut cache).to_vec());
    }
    let total = t2.elapsed().as_secs_f64();
    if let Some(st) = m.stream() {
        println!("\ntier: {} block rebuilds, {:.2} GiB resident for block weights",
                 st.rebuilds(), st.resident_bytes() as f64 / 1073741824.0);
    }
    // ⚠ Report the unit that carries information at THIS speed. A streamed 770B rebuilds 78 blocks
    // per token, so "0.00 tok/s" is what tok/s degenerates to and says nothing; seconds per token
    // is the number a reader can act on. Print both only when tok/s is actually legible.
    let decode_s = total - prefill;
    let per_tok = decode_s / (n_gen.saturating_sub(1)).max(1) as f64;
    println!("\nprefill {} tokens in {prefill:.2}s ({:.2}s/token); {} generated in {decode_s:.2}s ({per_tok:.1}s/token{})",
             ids.len(), prefill / ids.len().max(1) as f64, n_gen.saturating_sub(1),
             if per_tok < 1.0 { format!(", {:.2} tok/s", 1.0 / per_tok) } else { String::new() });
    println!("\n{}{}", prompt, tok.decode(&out));
}
