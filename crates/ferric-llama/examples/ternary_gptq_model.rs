//! Ferric ternary encoder — REAL-ACTIVATION GPTQ on a full model. Captures each linear's real inputs (the
//! qwen3 capture hook), builds per-layer input Hessians, GPTQ-ternarizes every projection using them, serves
//! the result through the ternarizing GgufSource, and measures perplexity vs the original. MAXLAY (env) GPTQs
//! the first N layers and naive-ternarizes the rest (MAXLAY=0 → all naive baseline; large → all GPTQ).
//!   MAXLAY=2 cargo run -p ferric-llama --example ternary_gptq_model --release   (quick pipeline check)
use ferric_core::Context;
use ferric_gguf::{quant_q8_0, GgufFile, GgufSource, Meta, TensorInfo};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

const GS: usize = 128;
const Q8_0: u32 = 8;

fn ternarize(w: &[f32]) -> Vec<f32> { // 1 group-wise plane
    let mut out = vec![0f32; w.len()];
    for g in 0..(w.len() + GS - 1) / GS {
        let (lo, hi) = (g * GS, ((g + 1) * GS).min(w.len()));
        let grp = &w[lo..hi];
        let d = 0.7 * grp.iter().map(|x| x.abs()).sum::<f32>() / grp.len() as f32;
        let (mut ss, mut sc) = (0f32, 0usize);
        for &x in grp { if x.abs() > d { ss += x.abs(); sc += 1; } }
        let a = if sc > 0 { ss / sc as f32 } else { 0.0 };
        for k in lo..hi { out[k] = if w[k].abs() > d { if w[k] > 0.0 { a } else { -a } } else { 0.0 }; }
    }
    out
}
// Cholesky factor Lh of H^-1 (Hinv=Lh Lhᵀ) from acts [T,C] (row-major): H = ΣₜxₜxₜT + damp
fn lh_from_acts(acts: &[f32], t: usize, c: usize) -> Vec<f32> {
    let mut h = vec![0f32; c * c];
    for tk in 0..t { let row = &acts[tk * c..tk * c + c]; for i in 0..c { let xi = row[i]; if xi != 0.0 { let hi = &mut h[i * c..i * c + c]; for j in 0..=i { hi[j] += xi * row[j]; } } } }
    for i in 0..c { for j in 0..i { h[j * c + i] = h[i * c + j]; } }
    let damp = 0.1 * (0..c).map(|i| h[i * c + i]).sum::<f32>() / c as f32;
    for i in 0..c { h[i * c + i] += damp.max(1e-6); }
    let mut l = vec![0f32; c * c];
    for i in 0..c { for j in 0..=i { let mut s = h[i * c + j]; for k in 0..j { s -= l[i * c + k] * l[j * c + k]; } if i == j { l[i * c + i] = s.max(1e-12).sqrt(); } else { l[i * c + j] = s / l[j * c + j]; } } }
    let mut li = vec![0f32; c * c];
    for i in 0..c { li[i * c + i] = 1.0 / l[i * c + i]; for j in 0..i { let mut s = 0.0; for k in j..i { s -= l[i * c + k] * li[k * c + j]; } li[i * c + j] = s / l[i * c + i]; } }
    let mut hinv = vec![0f32; c * c];
    for i in 0..c { for j in i..c { let mut s = 0.0; for k in j..c { s += li[k * c + i] * li[k * c + j]; } hinv[i * c + j] = s; hinv[j * c + i] = s; } }
    let mut lh = vec![0f32; c * c];
    for i in 0..c { for j in 0..=i { let mut s = hinv[i * c + j]; for k in 0..j { s -= lh[i * c + k] * lh[j * c + k]; } if i == j { lh[i * c + i] = s.max(1e-20).sqrt(); } else { lh[i * c + j] = s / lh[j * c + j]; } } }
    lh
}
// GPTQ-ternarize a weight [r,c] (row-major, c=input dim) with Cholesky factor lh → reconstruction
fn gptq_tern(w: &[f32], r: usize, c: usize, lh: &[f32]) -> Vec<f32> {
    let mut alpha = vec![0f32; r]; let mut delta = vec![0f32; r];
    for ro in 0..r { let row = &w[ro * c..ro * c + c]; let ma = row.iter().map(|x| x.abs()).sum::<f32>() / c as f32; delta[ro] = 0.7 * ma;
        let (mut ss, mut sc) = (0f32, 0usize); for &x in row { if x.abs() > delta[ro] { ss += x.abs(); sc += 1; } } alpha[ro] = if sc > 0 { ss / sc as f32 } else { 0.0 }; }
    let qz = |x: f32, ro: usize| if x.abs() > delta[ro] { if x > 0.0 { alpha[ro] } else { -alpha[ro] } } else { 0.0 };
    let mut wc = w.to_vec(); let mut q = vec![0f32; r * c];
    for j in 0..c {
        let djj = lh[j * c + j]; let mut e = vec![0f32; r];
        for ro in 0..r { let wj = wc[ro * c + j]; let qq = qz(wj, ro); q[ro * c + j] = qq; e[ro] = (wj - qq) / djj; }
        for k in (j + 1)..c { let u = lh[k * c + j]; if u == 0.0 { continue; } for ro in 0..r { wc[ro * c + k] -= e[ro] * u; } }
    }
    q
}

struct MapGguf<'a> { inner: &'a GgufFile, recon: HashMap<String, Vec<f32>> }
impl GgufSource for MapGguf<'_> {
    fn metadata(&self) -> &HashMap<String, Meta> { self.inner.metadata() }
    fn tensor(&self, n: &str) -> Option<&TensorInfo> { self.inner.tensor(n) }
    fn raw(&self, n: &str) -> Result<Vec<u8>, String> {
        if let Some(r) = self.recon.get(n) { if self.inner.tensor(n).map(|t| t.ggml_type) == Some(Q8_0) { return Ok(quant_q8_0(r)); } }
        self.inner.raw(n)
    }
    fn dequant(&self, n: &str) -> Result<Vec<f32>, String> {
        if let Some(r) = self.recon.get(n) { Ok(r.clone()) } else { self.inner.dequant(n) }
    }
}

async fn ppl(m: &Qwen3, ids: &[u32]) -> f64 {
    let c = &m.cfg;
    let v = m.forward_cached(ids, &mut Cache::new(c)).to_vec().await;
    let (mut nll, mut cnt) = (0f64, 0usize);
    for i in 0..ids.len() - 1 { let row = &v[i * c.n_vocab..(i + 1) * c.n_vocab];
        let mx = row.iter().cloned().fold(f32::MIN, f32::max);
        let lse = mx + row.iter().map(|&x| (x - mx).exp()).sum::<f32>().ln();
        nll += (lse - row[ids[i + 1] as usize]) as f64; cnt += 1; }
    (nll / cnt as f64).exp()
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let maxlay: usize = std::env::var("MAXLAY").ok().and_then(|s| s.parse().ok()).unwrap_or(24);
    let g = GgufFile::open(format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf")).unwrap();
    let toks: Vec<String> = match g.metadata().get("tokenizer.ggml.tokens") { Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(), _ => panic!() };
    let vocab: HashMap<String, u32> = toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata().get("tokenizer.ggml.merges") { Some(Meta::Arr(a)) => a.iter().filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None }).collect(), _ => panic!() };
    let bpe = Bpe::new(vocab, &merges);
    let para = "Artificial intelligence is the simulation of human intelligence by machines. Machine learning, a subset of AI, enables systems to learn from data. Deep neural networks process information through layers of interconnected nodes. Language models predict the next token given a context of previous tokens. Quantization reduces the memory footprint of a model by representing its weights with fewer bits, trading a small amount of accuracy for large gains in efficiency and speed. ";
    let calib = bpe.encode(&para.repeat(6));
    let eval = bpe.encode("The history of artificial intelligence began in antiquity with myths of artificial beings. Modern machine learning studies algorithms that improve automatically through experience and data.");
    println!("MAXLAY={maxlay}  calib {} tokens  eval {} tokens\n", calib.len(), eval.len());

    // 1) capture real activations
    let m0 = Qwen3::load(&ctx, &g).unwrap();
    let base = ppl(&m0, &eval).await;
    m0.set_capture(true);
    let _ = m0.forward_cached(&calib, &mut Cache::new(&m0.cfg)).to_vec().await;
    let cap = m0.take_capture();
    let mut acts: HashMap<String, (Vec<f32>, usize, usize)> = HashMap::new();
    for (name, t) in &cap { let v = pollster::block_on(t.to_vec()); acts.insert(name.clone(), (v, t.shape[0], t.shape[1])); }
    drop(m0);
    println!("captured {} activation sets; original eval ppl = {base:.3}\n", acts.len());

    // 2) GPTQ (layers < maxlay) / naive (rest) each projection
    let t0 = std::time::Instant::now();
    let mut recon: HashMap<String, Vec<f32>> = HashMap::new();
    let map: &[(&str, &str)] = &[("attn_q", "qkv"), ("attn_k", "qkv"), ("attn_v", "qkv"), ("attn_output", "wo"), ("ffn_gate", "ffn_gu"), ("ffn_up", "ffn_gu"), ("ffn_down", "ffn_down")];
    let nl = m_cfg_layers(&g);
    for l in 0..nl {
        // precompute Lh for the 4 input types of this layer (only if GPTQ'ing it)
        let mut lhs: HashMap<&str, Vec<f32>> = HashMap::new();
        if l < maxlay { for key in ["qkv", "wo", "ffn_gu", "ffn_down"] {
            if let Some((a, t, c)) = acts.get(&format!("l{l}.{key}")) { lhs.insert(key, lh_from_acts(a, *t, *c)); }
        }}
        for (proj, key) in map {
            let name = format!("blk.{l}.{proj}.weight");
            if g.tensor(&name).map(|t| t.ggml_type) != Some(Q8_0) { continue; }
            let ti = g.tensor(&name).unwrap();
            let (r, c) = (ti.dims[1] as usize, ti.dims[0] as usize);
            let w = g.dequant(&name).unwrap();
            let q = if l < maxlay { if let Some(lh) = lhs.get(*key) { gptq_tern(&w, r, c, lh) } else { ternarize(&w) } } else { ternarize(&w) };
            recon.insert(name, q);
        }
        if l < maxlay { println!("  layer {l} GPTQ'd  ({:?} elapsed)", t0.elapsed()); }
    }

    // 3) load ternary model + perplexity
    let mt = Qwen3::load(&ctx, &MapGguf { inner: &g, recon }).unwrap();
    let p = ppl(&mt, &eval).await;
    println!("\n  original                 ppl {base:.3}");
    println!("  ternary (GPTQ {maxlay} layers, naive rest)  ppl {p:.3}   ({:.0}× base)", p / base);
    println!("\n✅ Real-activation GPTQ ran end-to-end (capture → Hessian → GPTQ-ternarize → full model).");
}

fn m_cfg_layers(g: &GgufFile) -> usize {
    match g.metadata().get("qwen2.block_count").or_else(|| g.metadata().get("qwen3.block_count")) { Some(Meta::U(v)) => *v as usize, _ => 24 }
}
