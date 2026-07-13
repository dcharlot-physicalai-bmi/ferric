//! A small, deterministic Llama/SmolVLA-style demo LM — the SAME pure-Rust forward pass running on
//! any fabric. Weights are pseudo-random but fixed, so the result is reproducible and the GPU output
//! can be checked against the CPU reference in the same process (native OR browser WebGPU).
//!
//! Architecture: embed → N × [ RMSNorm → RoPE causal multi-head attention → +res → RMSNorm → SwiGLU
//! → +res ] → final RMSNorm → LM head → logits. This is exactly what `tiny_llm`/`generate` exercise,
//! packaged as a library so `ferric-web` runs the identical model in the browser.

use crate::{cpu, matmul_cpu, Context};

pub const VOCAB: usize = 32;
pub const D: usize = 64;
pub const H: usize = 4;
pub const DH: usize = 16;
pub const HFF: usize = 128;
pub const N_LAYERS: usize = 3;
pub const BASE: f32 = 10000.0;
pub const EPS: f32 = 1e-5;

fn fill(n: usize, s: f32) -> Vec<f32> {
    (0..n).map(|i| (((i as f32 * 12.9898 + s).sin() * 43758.5453).fract()) * 0.2 - 0.1).collect()
}
fn emul(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x * y).collect() }

struct L { wn1: Vec<f32>, wn2: Vec<f32>, wq: Vec<f32>, wk: Vec<f32>, wv: Vec<f32>, wo: Vec<f32>, wg: Vec<f32>, wu: Vec<f32>, wd: Vec<f32> }
struct W { emb: Vec<f32>, wln: Vec<f32>, wlm: Vec<f32>, layers: Vec<L> }
fn weights() -> W {
    let layers = (0..N_LAYERS).map(|i| {
        let s = 10.0 * (i as f32 + 1.0);
        L {
            wn1: fill(D, s + 1.0).iter().map(|v| v + 1.0).collect(),
            wn2: fill(D, s + 2.0).iter().map(|v| v + 1.0).collect(),
            wq: fill(D * D, s + 3.0), wk: fill(D * D, s + 4.0), wv: fill(D * D, s + 5.0), wo: fill(D * D, s + 6.0),
            wg: fill(D * HFF, s + 7.0), wu: fill(D * HFF, s + 8.0), wd: fill(HFF * D, s + 9.0),
        }
    }).collect();
    W { emb: fill(VOCAB * D, 100.0), wln: fill(D, 200.0).iter().map(|v| v + 1.0).collect(), wlm: fill(D * VOCAB, 300.0), layers }
}

/// GPU forward → logits [T, VOCAB]. Runs entirely on whatever fabric `ctx` bound (native GPU / WebGPU).
pub async fn logits(ctx: &Context, ids: &[u32]) -> crate::Result<Vec<f32>> {
    let w = weights();
    let t = ids.len() as u32;
    let (d, hff) = (D as u32, HFF as u32);
    let up = |v: &Vec<f32>, s: &[usize]| ctx.tensor(v, s);
    let embt = up(&w.emb, &[VOCAB, D]);
    let mut x = ctx.gather0(&embt, ids, D);
    for l in &w.layers {
        let rms1 = ctx.rmsnorm_t(&x, &up(&l.wn1, &[D]), t, d, EPS);
        let q = ctx.rope_t(&ctx.mm(&rms1, &up(&l.wq, &[D, D]), t, d, d), t, H as u32, DH as u32, BASE);
        let k = ctx.rope_t(&ctx.mm(&rms1, &up(&l.wk, &[D, D]), t, d, d), t, H as u32, DH as u32, BASE);
        let v = ctx.mm(&rms1, &up(&l.wv, &[D, D]), t, d, d);
        let attn = ctx.mha_causal_t(&q, &k, &v, t, H as u32, H as u32, DH as u32);
        x = ctx.add_t(&x, &ctx.mm(&attn, &up(&l.wo, &[D, D]), t, d, d));
        let rms2 = ctx.rmsnorm_t(&x, &up(&l.wn2, &[D]), t, d, EPS);
        let g = ctx.mm(&rms2, &up(&l.wg, &[D, HFF]), t, d, hff);
        let u2 = ctx.mm(&rms2, &up(&l.wu, &[D, HFF]), t, d, hff);
        let act = ctx.mul_t(&g, &ctx.sigmoid_t(&g));
        x = ctx.add_t(&x, &ctx.mm(&ctx.mul_t(&act, &u2), &up(&l.wd, &[HFF, D]), t, hff, d));
    }
    let x = ctx.rmsnorm_t(&x, &up(&w.wln, &[D]), t, d, EPS);
    ctx.to_vec(&ctx.mm(&x, &up(&w.wlm, &[D, VOCAB]), t, d, VOCAB as u32)).await
}

/// CPU reference forward → logits [T, VOCAB]. Same math in plain Rust (runs in wasm too).
pub fn logits_cpu(ids: &[u32]) -> Vec<f32> {
    let w = weights();
    let t = ids.len();
    let mut x: Vec<f32> = ids.iter().flat_map(|&tk| w.emb[tk as usize * D..(tk as usize + 1) * D].to_vec()).collect();
    for l in &w.layers {
        let rms1 = cpu::rmsnorm(&x, &l.wn1, t, D, EPS);
        let q = cpu::rope(&matmul_cpu(&rms1, &l.wq, t, D, D), t, H, DH, BASE);
        let k = cpu::rope(&matmul_cpu(&rms1, &l.wk, t, D, D), t, H, DH, BASE);
        let v = matmul_cpu(&rms1, &l.wv, t, D, D);
        let attn = cpu::mha_causal(&q, &k, &v, t, H, H, DH);
        x = cpu::add(&x, &matmul_cpu(&attn, &l.wo, t, D, D));
        let rms2 = cpu::rmsnorm(&x, &l.wn2, t, D, EPS);
        let g = matmul_cpu(&rms2, &l.wg, t, D, HFF);
        let u2 = matmul_cpu(&rms2, &l.wu, t, D, HFF);
        x = cpu::add(&x, &matmul_cpu(&emul(&emul(&g, &cpu::sigmoid(&g)), &u2), &l.wd, t, HFF, D));
    }
    let x = cpu::rmsnorm(&x, &w.wln, t, D, EPS);
    matmul_cpu(&x, &w.wlm, t, D, VOCAB)
}

/// Greedy autoregressive generation on the GPU: argmax the last token, feed it back, repeat.
pub async fn generate(ctx: &Context, prompt: &[u32], steps: usize) -> crate::Result<Vec<u32>> {
    let mut seq = prompt.to_vec();
    let mut out = Vec::new();
    for _ in 0..steps {
        let lg = logits(ctx, &seq).await?;
        let last = &lg[(seq.len() - 1) * VOCAB..];
        let next = last.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        out.push(next);
        seq.push(next);
    }
    Ok(out)
}
