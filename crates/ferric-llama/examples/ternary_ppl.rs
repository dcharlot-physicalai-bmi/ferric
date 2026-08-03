//! Ferric ternary encoder — END-TO-END model validation. Wraps the real Qwen2.5-0.5B GGUF in a GgufSource
//! that ternarizes every projection weight on the fly (group-wise multi-plane ternary), loads the FULL model
//! through the unchanged Qwen3 path, and measures PERPLEXITY vs the original — the real "does ternarize-by-
//! Ferric keep the model working" number. (Calibration-free multi-plane; GPTQ-with-real-activations is next.)
//!   cargo run -p ferric-llama --example ternary_ppl --release
use ferric_core::Context;
use ferric_gguf::{quant_q8_0, GgufFile, GgufSource, Meta, TensorInfo};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

const GS: usize = 128;
const Q8_0: u32 = 8; // ggml type id

// group-wise multi-plane ternary → dequantized reconstruction (BitNet/PrismML style)
fn ternarize(w: &[f32], planes: usize) -> Vec<f32> {
    let mut resid = w.to_vec();
    let mut recon = vec![0f32; w.len()];
    for _ in 0..planes {
        for g in 0..(w.len() + GS - 1) / GS {
            let (lo, hi) = (g * GS, ((g + 1) * GS).min(w.len()));
            let grp = &resid[lo..hi];
            let ma = grp.iter().map(|x| x.abs()).sum::<f32>() / grp.len() as f32;
            let d = 0.7 * ma;
            let (mut ss, mut sc) = (0f32, 0usize);
            for &x in grp { if x.abs() > d { ss += x.abs(); sc += 1; } }
            let a = if sc > 0 { ss / sc as f32 } else { 0.0 };
            for k in lo..hi { let t = if resid[k].abs() > d { resid[k].signum() } else { 0.0 }; recon[k] += a * t; resid[k] -= a * t; }
        }
    }
    recon
}

fn is_proj(n: &str) -> bool {
    ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
     "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"].iter().any(|s| n.ends_with(s))
}

// GgufSource wrapper: returns ternarized weights for projections, delegates everything else.
struct TGguf<'a> { inner: &'a GgufFile, planes: usize }
impl GgufSource for TGguf<'_> {
    fn metadata(&self) -> &HashMap<String, Meta> { self.inner.metadata() }
    fn tensor(&self, n: &str) -> Option<&TensorInfo> { self.inner.tensor(n) }
    fn raw(&self, n: &str) -> Result<Vec<u8>, String> {
        if is_proj(n) && self.inner.tensor(n).map(|t| t.ggml_type) == Some(Q8_0) {
            Ok(quant_q8_0(&ternarize(&self.inner.dequant(n)?, self.planes)))
        } else { self.inner.raw(n) }
    }
    fn dequant(&self, n: &str) -> Result<Vec<f32>, String> {
        if is_proj(n) { Ok(ternarize(&self.inner.dequant(n)?, self.planes)) } else { self.inner.dequant(n) }
    }
}

async fn perplexity(ctx: &Arc<Context>, m: &Qwen3, ids: &[u32]) -> f64 {
    let c = &m.cfg;
    let v = m.forward_cached(ids, &mut Cache::new(c)).to_vec().await; // [n*vocab]
    let (mut nll, mut cnt) = (0f64, 0usize);
    for i in 0..ids.len() - 1 {
        let row = &v[i * c.n_vocab..(i + 1) * c.n_vocab];
        let mx = row.iter().cloned().fold(f32::MIN, f32::max);
        let lse = mx + row.iter().map(|&x| (x - mx).exp()).sum::<f32>().ln();
        nll += (lse - row[ids[i + 1] as usize]) as f64;
        cnt += 1;
    }
    (nll / cnt as f64).exp()
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let path = format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf");
    let g = GgufFile::open(&path).unwrap();

    // tokenizer from the GGUF
    let tokens: Vec<String> = match g.metadata().get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(), _ => panic!() };
    let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata().get("tokenizer.ggml.merges") {
        Some(Meta::Arr(a)) => a.iter().filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None }).collect(), _ => panic!() };
    let bpe = Bpe::new(vocab, &merges);
    let text = "The history of artificial intelligence began in antiquity, with myths and stories of \
artificial beings endowed with intelligence by master craftsmen. Modern machine learning studies algorithms \
that improve automatically through experience and by the use of data, building models from a training set in \
order to make predictions or decisions without being explicitly programmed to do so.";
    let ids = bpe.encode(text);
    println!("eval: {} tokens\n", ids.len());

    let orig = Qwen3::load(&ctx, &g).unwrap();
    let ppl_orig = perplexity(&ctx, &orig, &ids).await;
    println!("  original (Q8_0)                    perplexity = {ppl_orig:.3}");
    drop(orig);

    for planes in [1usize, 2, 3] {
        let tern = Qwen3::load(&ctx, &TGguf { inner: &g, planes }).unwrap();
        let ppl = perplexity(&ctx, &tern, &ids).await;
        let bpw = 1.6 * planes as f32;
        println!("  ternarized-by-Ferric ({planes} plane{})   perplexity = {ppl:.3}   ({:.1} bpw, {:+.1}% vs orig)",
                 if planes > 1 { "s" } else { " " }, bpw, 100.0 * (ppl / ppl_orig - 1.0));
    }
    println!("\n✅ End-to-end: Ferric ternarizes every projection of a real Qwen2.5-0.5B and runs the full model.");
    println!("   Calibration-free multi-plane; the perplexity gap shows what GPTQ-with-real-activations must close next.");
}
