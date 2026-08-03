//! Ferric ternary encoder — LAYER SENSITIVITY SWEEP. Using the on-the-fly ternarizing GgufSource, measure
//! which parts of a real Qwen2.5-0.5B survive calibration-free ternary and which collapse it — isolating the
//! damage from the catastrophic all-ternary result, and finding the practical mixed-precision envelope.
//!   cargo run -p ferric-llama --example ternary_sweep --release
use ferric_core::Context;
use ferric_gguf::{quant_q8_0, GgufFile, GgufSource, Meta, TensorInfo};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

const GS: usize = 128;
const Q8_0: u32 = 8;

fn ternarize(w: &[f32], planes: usize) -> Vec<f32> {
    let mut resid = w.to_vec(); let mut recon = vec![0f32; w.len()];
    for _ in 0..planes {
        for g in 0..(w.len() + GS - 1) / GS {
            let (lo, hi) = (g * GS, ((g + 1) * GS).min(w.len()));
            let grp = &resid[lo..hi];
            let d = 0.7 * grp.iter().map(|x| x.abs()).sum::<f32>() / grp.len() as f32;
            let (mut ss, mut sc) = (0f32, 0usize);
            for &x in grp { if x.abs() > d { ss += x.abs(); sc += 1; } }
            let a = if sc > 0 { ss / sc as f32 } else { 0.0 };
            for k in lo..hi { let t = if resid[k].abs() > d { resid[k].signum() } else { 0.0 }; recon[k] += a * t; resid[k] -= a * t; }
        }
    }
    recon
}
fn layer_of(n: &str) -> Option<usize> { n.strip_prefix("blk.")?.split_once('.').and_then(|(a, _)| a.parse().ok()) }
fn is_proj(n: &str) -> bool { ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"].iter().any(|s| n.contains(s)) && n.ends_with(".weight") }
fn is_ffn(n: &str) -> bool { n.contains("ffn_") && n.ends_with(".weight") }
fn is_attn(n: &str) -> bool { n.contains("attn_") && n.ends_with(".weight") }

struct TGguf<'a> { inner: &'a GgufFile, target: Box<dyn Fn(&str) -> bool + 'a> }
impl GgufSource for TGguf<'_> {
    fn metadata(&self) -> &HashMap<String, Meta> { self.inner.metadata() }
    fn tensor(&self, n: &str) -> Option<&TensorInfo> { self.inner.tensor(n) }
    fn raw(&self, n: &str) -> Result<Vec<u8>, String> {
        if (self.target)(n) && self.inner.tensor(n).map(|t| t.ggml_type) == Some(Q8_0) {
            Ok(quant_q8_0(&ternarize(&self.inner.dequant(n)?, 1)))
        } else { self.inner.raw(n) }
    }
    fn dequant(&self, n: &str) -> Result<Vec<f32>, String> {
        if (self.target)(n) { Ok(ternarize(&self.inner.dequant(n)?, 1)) } else { self.inner.dequant(n) }
    }
}

async fn ppl(ctx: &Arc<Context>, m: &Qwen3, ids: &[u32]) -> f64 {
    let c = &m.cfg;
    let v = m.forward_cached(ids, &mut Cache::new(c)).to_vec().await;
    let (mut nll, mut cnt) = (0f64, 0usize);
    for i in 0..ids.len() - 1 {
        let row = &v[i * c.n_vocab..(i + 1) * c.n_vocab];
        let mx = row.iter().cloned().fold(f32::MIN, f32::max);
        let lse = mx + row.iter().map(|&x| (x - mx).exp()).sum::<f32>().ln();
        nll += (lse - row[ids[i + 1] as usize]) as f64; cnt += 1;
    }
    (nll / cnt as f64).exp()
}
// fraction of projection params covered by a target predicate
fn frac(g: &GgufFile, tgt: &dyn Fn(&str) -> bool) -> f64 {
    let (mut t, mut a) = (0u64, 0u64);
    for ti in &g.tensors { if is_proj(&ti.name) { let n: u64 = ti.dims.iter().product(); a += n; if tgt(&ti.name) { t += n; } } }
    t as f64 / a as f64
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let g = GgufFile::open(format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf")).unwrap();
    let tokens: Vec<String> = match g.metadata().get("tokenizer.ggml.tokens") { Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(), _ => panic!() };
    let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata().get("tokenizer.ggml.merges") { Some(Meta::Arr(a)) => a.iter().filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None }).collect(), _ => panic!() };
    let bpe = Bpe::new(vocab, &merges);
    let text = "The history of artificial intelligence began in antiquity, with myths and stories of artificial \
beings endowed with intelligence by master craftsmen. Modern machine learning studies algorithms that improve \
automatically through experience and by the use of data, building models from a training set in order to make \
predictions without being explicitly programmed to do so.";
    let ids = bpe.encode(text);
    let nl = 24usize;

    let base = ppl(&ctx, &Qwen3::load(&ctx, &g).unwrap(), &ids).await;
    println!("original Q8_0 perplexity = {base:.3}  ({} eval tokens)\n", ids.len());
    println!("  {:<40} {:>10}  {:>7}  {:>9}", "calibration-free ternary target", "ppl", "×base", "%params");

    let configs: Vec<(&str, Box<dyn Fn(&str) -> bool>)> = vec![
        ("attention only (all layers)", Box::new(|n: &str| is_attn(n))),
        ("FFN only (all layers)", Box::new(|n: &str| is_ffn(n))),
        ("FFN, middle layers 6..18", Box::new(|n: &str| is_ffn(n) && layer_of(n).map_or(false, |l| (6..18).contains(&l)))),
        ("all proj, keep first/last 2 layers fp", Box::new(|n: &str| is_proj(n) && layer_of(n).map_or(false, |l| (2..nl - 2).contains(&l)))),
        ("all projections (baseline)", Box::new(|n: &str| is_proj(n))),
    ];
    for (label, tgt) in &configs {
        let f = frac(&g, tgt.as_ref());
        let m = Qwen3::load(&ctx, &TGguf { inner: &g, target: Box::new(move |n: &str| tgt(n)) }).unwrap();
        let p = ppl(&ctx, &m, &ids).await;
        println!("  {:<40} {:>10.1}  {:>6.0}×  {:>7.0}%", label, p, p / base, 100.0 * f);
    }
    println!("\nWhich parts survive calibration-free ternary (ppl near base) vs collapse it — the mixed-precision");
    println!("envelope, and which layer type is the culprit. (Full ternary needs GPTQ-real-acts/QAT, as shown.)");
}
