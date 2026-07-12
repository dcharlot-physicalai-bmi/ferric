//! Autoregressive generation with a KV cache. The same tiny Llama/SmolVLA-style LM as `tiny_llm`,
//! but decoded ONE token at a time: each step projects only the new token, appends its K/V to a
//! per-layer cache, and attends the single query against the whole cache (`mha_decode`). RoPE uses
//! the token's absolute position. Prefill and decode are the SAME step (prompt tokens are teacher-
//! forced, generated tokens are argmax-fed).
//!
//! Correctness: every step's logits are checked against a full non-cached recompute over the whole
//! prefix so far — proving the KV cache is exact, not just plausible.
use ferric_core::{cpu, matmul_cpu, max_abs_diff, Context, Tensor};

fn fill(n: usize, s: f32) -> Vec<f32> {
    (0..n).map(|i| (((i as f32 * 12.9898 + s).sin() * 43758.5453).fract()) * 0.2 - 0.1).collect()
}
fn emul(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x * y).collect() }

struct Cfg { h: usize, dh: usize, d: usize, hff: usize, base: f32, eps: f32 }
struct Layer { wn1: Vec<f32>, wn2: Vec<f32>, wq: Vec<f32>, wk: Vec<f32>, wv: Vec<f32>, wo: Vec<f32>, wg: Vec<f32>, wu: Vec<f32>, wd: Vec<f32> }
impl Layer {
    fn new(d: usize, hff: usize, s: f32) -> Self {
        Layer {
            wn1: fill(d, s + 1.0).iter().map(|v| v + 1.0).collect(),
            wn2: fill(d, s + 2.0).iter().map(|v| v + 1.0).collect(),
            wq: fill(d * d, s + 3.0), wk: fill(d * d, s + 4.0), wv: fill(d * d, s + 5.0), wo: fill(d * d, s + 6.0),
            wg: fill(d * hff, s + 7.0), wu: fill(d * hff, s + 8.0), wd: fill(hff * d, s + 9.0),
        }
    }
}

// ---- full non-cached CPU forward over a whole sequence (the reference) ----
fn layer_cpu(x: &[f32], l: &Layer, c: &Cfg, t: usize) -> Vec<f32> {
    let (d, hff) = (c.d, c.hff);
    let rms1 = cpu::rmsnorm(x, &l.wn1, t, d, c.eps);
    let q = cpu::rope(&matmul_cpu(&rms1, &l.wq, t, d, d), t, c.h, c.dh, c.base);
    let k = cpu::rope(&matmul_cpu(&rms1, &l.wk, t, d, d), t, c.h, c.dh, c.base);
    let v = matmul_cpu(&rms1, &l.wv, t, d, d);
    let attn = cpu::mha_causal(&q, &k, &v, t, c.h, c.h, c.dh);
    let x2 = cpu::add(x, &matmul_cpu(&attn, &l.wo, t, d, d));
    let rms2 = cpu::rmsnorm(&x2, &l.wn2, t, d, c.eps);
    let g = matmul_cpu(&rms2, &l.wg, t, d, hff);
    let up = matmul_cpu(&rms2, &l.wu, t, d, hff);
    let down = matmul_cpu(&emul(&emul(&g, &cpu::sigmoid(&g)), &up), &l.wd, t, hff, d);
    cpu::add(&x2, &down)
}
fn forward_cpu(ids: &[u32], emb: &[f32], layers: &[Layer], wln: &[f32], wlm: &[f32], c: &Cfg, vocab: usize) -> Vec<f32> {
    let t = ids.len();
    let mut x: Vec<f32> = ids.iter().flat_map(|&tk| emb[tk as usize * c.d..(tk as usize + 1) * c.d].to_vec()).collect();
    for l in layers { x = layer_cpu(&x, l, c, t); }
    let x = cpu::rmsnorm(&x, wln, t, c.d, c.eps);
    matmul_cpu(&x, wlm, t, c.d, vocab)[(t - 1) * vocab..].to_vec() // logits for the last token
}

// ---- one incremental GPU decode step for a layer (mutates the layer's K/V cache) ----
async fn step_gpu(ctx: &Context, x1: &Tensor, l: &Layer, kc: &mut Vec<f32>, vc: &mut Vec<f32>, c: &Cfg, pos: u32) -> Tensor {
    let d = c.d as u32;
    let up = |v: &Vec<f32>, s: &[usize]| ctx.tensor(v, s);
    let rms1 = ctx.rmsnorm_t(x1, &up(&l.wn1, &[c.d]), 1, d, c.eps);
    let q1 = ctx.rope_off_t(&ctx.mm(&rms1, &up(&l.wq, &[c.d, c.d]), 1, d, d), 1, c.h as u32, c.dh as u32, c.base, pos);
    let knew = ctx.rope_off_t(&ctx.mm(&rms1, &up(&l.wk, &[c.d, c.d]), 1, d, d), 1, c.h as u32, c.dh as u32, c.base, pos);
    let vnew = ctx.mm(&rms1, &up(&l.wv, &[c.d, c.d]), 1, d, d);
    kc.extend(ctx.to_vec(&knew).await.unwrap());   // append to the KV cache
    vc.extend(ctx.to_vec(&vnew).await.unwrap());
    let s = (kc.len() / c.d) as u32;
    let attn = ctx.mha_decode_t(&q1, &up(kc, &[s as usize, c.d]), &up(vc, &[s as usize, c.d]), c.h as u32, c.h as u32, c.dh as u32, s);
    let x2 = ctx.add_t(x1, &ctx.mm(&attn, &up(&l.wo, &[c.d, c.d]), 1, d, d));
    let rms2 = ctx.rmsnorm_t(&x2, &up(&l.wn2, &[c.d]), 1, d, c.eps);
    let g = ctx.mm(&rms2, &up(&l.wg, &[c.d, c.hff]), 1, d, c.hff as u32);
    let u2 = ctx.mm(&rms2, &up(&l.wu, &[c.d, c.hff]), 1, d, c.hff as u32);
    let down = ctx.mm(&ctx.mul_t(&ctx.mul_t(&g, &ctx.sigmoid_t(&g)), &u2), &up(&l.wd, &[c.hff, c.d]), 1, c.hff as u32, d);
    ctx.add_t(&x2, &down)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Context::new().await.unwrap();
    let c = Cfg { h: 4, dh: 16, d: 64, hff: 128, base: 10000.0, eps: 1e-5 };
    let (vocab, n_layers) = (32usize, 3usize);
    let emb = fill(vocab * c.d, 100.0);
    let wln: Vec<f32> = fill(c.d, 200.0).iter().map(|v| v + 1.0).collect();
    let wlm = fill(c.d * vocab, 300.0);
    let layers: Vec<Layer> = (0..n_layers).map(|i| Layer::new(c.d, c.hff, 10.0 * (i as f32 + 1.0))).collect();

    let prompt: Vec<u32> = vec![3, 14, 1, 15];
    let gen_steps = 5usize;
    let mut kcache = vec![Vec::<f32>::new(); n_layers];
    let mut vcache = vec![Vec::<f32>::new(); n_layers];
    let mut seq = prompt.clone();
    let mut generated = Vec::new();
    let mut worst = 0f32;

    for pos in 0..prompt.len() + gen_steps {
        let tok = seq[pos];
        let mut x = ctx.tensor(&emb[tok as usize * c.d..(tok as usize + 1) * c.d], &[1, c.d]);
        for (li, l) in layers.iter().enumerate() {
            x = step_gpu(&ctx, &x, l, &mut kcache[li], &mut vcache[li], &c, pos as u32).await;
        }
        let x = ctx.rmsnorm_t(&x, &ctx.tensor(&wln, &[c.d]), 1, c.d as u32, c.eps);
        let logits = ctx.to_vec(&ctx.mm(&x, &ctx.tensor(&wlm, &[c.d, vocab]), 1, c.d as u32, vocab as u32)).await.unwrap();

        // reference: full non-cached recompute over the prefix seq[0..=pos]
        let refl = forward_cpu(&seq[..=pos], &emb, &layers, &wln, &wlm, &c, vocab);
        worst = worst.max(max_abs_diff(&logits, &refl));

        let next = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        if pos + 1 >= prompt.len() {
            generated.push(next);
            if pos + 1 < seq.len() { } else { seq.push(next); } // feed the generated token back
        }
    }

    println!("Ferric autoregressive generation · {:?} · {n_layers} layers, KV cache", ctx.backend);
    println!("  prompt {prompt:?} → generated {generated:?}");
    println!("  max|cached-decode logits - full-recompute| across all steps = {worst:.3e}");
    assert!(worst < 2e-3, "kv-cache decode mismatch {worst}");
    println!("✅ KV-cache autoregressive decode is EXACT vs full recompute — Ferric generates tokens on-GPU");
}
