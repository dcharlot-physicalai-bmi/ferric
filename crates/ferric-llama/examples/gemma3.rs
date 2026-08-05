//! **Gemma-3 runs** — a second architecture, and the RoPE fallback path, on a real checkpoint.
//!
//! Everything else in this directory validates against Qwen/Llama. Gemma-3 is deliberately different in
//! four ways that all have to be right at once:
//!
//!   - **QK-norm** on q and k, which is what sends this model down the *non-fused* RoPE path (see
//!     `qwen3.rs`: the q|k RoPE fusion requires q and k to still be adjacent, and QK-norm normalises them
//!     separately first). So this is the regression test for that fallback.
//!   - **Alternating attention** — one global layer every 6, the rest local, with different RoPE θ.
//!   - **SentencePiece**, not BPE. Gemma's GGUF carries `tokenizer.ggml.model = "llama"` and **zero**
//!     merges; Qwen's carries `"gpt2"` and 151,387 merges.
//!   - **√d embedding scale** and post-attention / post-FFN norms.
//!
//! ## This example exists because of a mistake worth not repeating
//!
//! Gemma-3 was briefly recorded here as broken, emitting `"GfGfGfGf"`. It was not broken. The test was:
//! it built a **BPE** tokenizer from a vocabulary with no merge table, so the prompt never survived
//! tokenisation. Feeding a model garbage and reading garbage back proves nothing about the model.
//!
//! Two further harness errors were stacked underneath: an instruction-tuned checkpoint was given a bare
//! prompt instead of its chat template, and decoding ran past the stop token. Fixed, it answers "Paris".
//!
//! The lesson is the general one — a red result indicts the harness until the harness is cleared.
//!
//!   cargo run -p ferric-llama --example gemma3 --release
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tokenizer::Spm;
use std::sync::Arc;

/// Encode Gemma's chat template, keeping `<control tokens>` as single ids rather than letting the
/// SentencePiece model split them into pieces.
fn encode_chat(spm: &Spm, user: &str) -> Vec<u32> {
    let mut ids = vec![spm.id_of("<bos>").unwrap_or(2)];
    let tmpl = format!("<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n");
    let mut rest = tmpl.as_str();
    let mut first = true;
    while let Some(lt) = rest.find('<') {
        if lt > 0 { ids.extend(spm.encode_piece(&rest[..lt], first)); first = false; }
        match rest[lt..].find('>') {
            Some(gt) => {
                let tok = &rest[lt..lt + gt + 1];
                match spm.id_of(tok) {
                    Some(id) => ids.push(id),
                    // Not a control token after all — encode it as text rather than dropping it.
                    None => ids.extend(spm.encode_piece(tok, false)),
                }
                rest = &rest[lt + gt + 1..];
            }
            None => break,
        }
    }
    if !rest.is_empty() { ids.extend(spm.encode_piece(rest, first)); }
    ids
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let path = format!("{home}/.cache/ferric/hub/ggml-org_gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf");
    let Ok(g) = GgufFile::open(&path) else {
        println!("skipped — no gemma-3-1b-it checkpoint at {path}");
        return;
    };

    let tokens: Vec<String> = match g.metadata().get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokenizer.ggml.tokens"),
    };
    let scores: Vec<f32> = match g.metadata().get("tokenizer.ggml.scores") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::F(f) = m { *f as f32 } else { 0.0 }).collect(),
        _ => vec![0.0; tokens.len()],
    };
    let n_merges = match g.metadata().get("tokenizer.ggml.merges") { Some(Meta::Arr(a)) => a.len(), _ => 0 };
    let tok_model = match g.metadata().get("tokenizer.ggml.model") { Some(Meta::Str(s)) => s.clone(), _ => "?".into() };

    println!("Gemma-3 1B — a second architecture and the non-fused RoPE path\n");
    println!("  tokenizer.ggml.model = {tok_model:?}, {} tokens, {n_merges} merges", tokens.len());
    // The assertion that would have caught the original mistake immediately.
    assert_eq!(n_merges, 0, "this checkpoint has BPE merges — it is not the SentencePiece model this example assumes");

    let spm = Spm::new(tokens, scores);
    let m = Qwen3::load(&ctx, &g).unwrap();
    println!("  qk_norm = {} -> RoPE takes the NON-fused two-dispatch path", m.cfg.has_qk_norm);
    assert!(m.cfg.has_qk_norm, "Gemma-3 should report QK-norm; without it this is not testing the fallback");

    let eos: Vec<u32> = ["<end_of_turn>", "<eos>"].iter().filter_map(|t| spm.id_of(t)).collect();
    let mut all = encode_chat(&spm, "What is the capital of France? Answer in one word.");
    println!("  prompt encodes to {} tokens\n", all.len());

    let mut cache = Cache::new(&m.cfg);
    let vn = m.cfg.n_vocab;
    let mut out = String::new();
    let mut fed = 0usize;
    for _ in 0..16 {
        let l = m.forward_cached(&all[fed..], &mut cache).to_vec().await;
        fed = all.len();
        let b = l[l.len() - vn..].iter().enumerate()
            .fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0 as u32;
        // Stop at EOS. Decoding past it is how the original test turned a correct answer into "junk".
        if eos.contains(&b) { break; }
        all.push(b);
        out.push_str(&spm.decode(&[b]));
    }

    let answer = out.trim();
    println!("  answer: {answer:?}");
    assert!(
        answer.contains("Paris"),
        "Gemma-3 answered {answer:?}, expected it to contain \"Paris\" — either the model path is wrong \
         or this harness is (check the tokenizer, the chat template, and the stop token, in that order)"
    );

    println!("\n  ✅ Gemma-3 answers correctly and stops at its own EOS. This covers the RoPE fallback");
    println!("     (QK-norm keeps q and k from being one span, so the fused single-dispatch path must not");
    println!("     be taken), alternating local/global attention, the √d embedding scale, and a");
    println!("     SentencePiece vocabulary with no merge table.");
}
