//! A complete, runnable Llama/SmolVLA-style language model assembled from Ferric kernels:
//!
//!   token ids → embedding (Gather) → N×[ RMSNorm → RoPE causal-MHA → +res → RMSNorm → SwiGLU → +res ]
//!             → final RMSNorm → LM head → logits → argmax next-token
//!
//! Runs entirely on-GPU (one readback for the logits), same source native + browser. Validated
//! end-to-end against a plain-Rust CPU reference — this is a full transformer forward pass, our stack.
use ferric_core::{cpu, matmul_cpu, max_abs_diff, Context, Tensor};

fn fill(n: usize, s: f32) -> Vec<f32> {
    (0..n).map(|i| (((i as f32 * 12.9898 + s).sin() * 43758.5453).fract()) * 0.2 - 0.1).collect()
}
fn emul(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x * y).collect() }

struct Cfg { t: usize, h: usize, dh: usize, d: usize, hff: usize, base: f32, eps: f32 }
// One decoder layer's weights (deterministic, seeded per layer).
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

fn layer_gpu(ctx: &Context, x: &Tensor, l: &Layer, c: &Cfg) -> Tensor {
    let (t, d, hff) = (c.t as u32, c.d as u32, c.hff as u32);
    let up = |v: &Vec<f32>, s: &[usize]| ctx.tensor(v, s);
    let rms1 = ctx.rmsnorm_t(x, &up(&l.wn1, &[c.d]), t, d, c.eps);
    let q = ctx.rope_t(&ctx.mm(&rms1, &up(&l.wq, &[c.d, c.d]), t, d, d), t, c.h as u32, c.dh as u32, c.base);
    let k = ctx.rope_t(&ctx.mm(&rms1, &up(&l.wk, &[c.d, c.d]), t, d, d), t, c.h as u32, c.dh as u32, c.base);
    let v = ctx.mm(&rms1, &up(&l.wv, &[c.d, c.d]), t, d, d);
    let attn = ctx.mha_causal_t(&q, &k, &v, t, c.h as u32, c.h as u32, c.dh as u32);
    let x2 = ctx.add_t(x, &ctx.mm(&attn, &up(&l.wo, &[c.d, c.d]), t, d, d));
    let rms2 = ctx.rmsnorm_t(&x2, &up(&l.wn2, &[c.d]), t, d, c.eps);
    let g = ctx.mm(&rms2, &up(&l.wg, &[c.d, c.hff]), t, d, hff);
    let up2 = ctx.mm(&rms2, &up(&l.wu, &[c.d, c.hff]), t, d, hff);
    let silu = ctx.mul_t(&g, &ctx.sigmoid_t(&g));
    let down = ctx.mm(&ctx.mul_t(&silu, &up2), &up(&l.wd, &[c.hff, c.d]), t, hff, d);
    ctx.add_t(&x2, &down)
}
fn layer_cpu(x: &[f32], l: &Layer, c: &Cfg) -> Vec<f32> {
    let (t, d, hff) = (c.t, c.d, c.hff);
    let rms1 = cpu::rmsnorm(x, &l.wn1, t, d, c.eps);
    let q = cpu::rope(&matmul_cpu(&rms1, &l.wq, t, d, d), t, c.h, c.dh, c.base);
    let k = cpu::rope(&matmul_cpu(&rms1, &l.wk, t, d, d), t, c.h, c.dh, c.base);
    let v = matmul_cpu(&rms1, &l.wv, t, d, d);
    let attn = cpu::mha_causal(&q, &k, &v, t, c.h, c.h, c.dh);
    let x2 = cpu::add(x, &matmul_cpu(&attn, &l.wo, t, d, d));
    let rms2 = cpu::rmsnorm(&x2, &l.wn2, t, d, c.eps);
    let g = matmul_cpu(&rms2, &l.wg, t, d, hff);
    let up2 = matmul_cpu(&rms2, &l.wu, t, d, hff);
    let silu = emul(&g, &cpu::sigmoid(&g));
    let down = matmul_cpu(&emul(&silu, &up2), &l.wd, t, hff, d);
    cpu::add(&x2, &down)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Context::new().await.unwrap();
    let c = Cfg { t: 6, h: 4, dh: 16, d: 64, hff: 128, base: 10000.0, eps: 1e-5 };
    let (vocab, n_layers) = (32usize, 3usize);
    let emb = fill(vocab * c.d, 100.0);
    let wln: Vec<f32> = fill(c.d, 200.0).iter().map(|v| v + 1.0).collect(); // final norm
    let wlm = fill(c.d * vocab, 300.0);                                     // LM head
    let layers: Vec<Layer> = (0..n_layers).map(|i| Layer::new(c.d, c.hff, 10.0 * (i as f32 + 1.0))).collect();
    let ids: Vec<u32> = vec![3, 14, 1, 15, 9, 2];

    // ---- GPU forward ----
    let temb = ctx.tensor(&emb, &[vocab, c.d]);
    let mut x = ctx.gather0(&temb, &ids, c.d); // [T, D] token embeddings
    for l in &layers { x = layer_gpu(&ctx, &x, l, &c); }
    let x = ctx.rmsnorm_t(&x, &ctx.tensor(&wln, &[c.d]), c.t as u32, c.d as u32, c.eps);
    let logits = ctx.mm(&x, &ctx.tensor(&wlm, &[c.d, vocab]), c.t as u32, c.d as u32, vocab as u32);
    let g_logits = ctx.to_vec(&logits).await.unwrap();

    // ---- CPU reference ----
    let mut xc: Vec<f32> = ids.iter().flat_map(|&t| emb[t as usize * c.d..(t as usize + 1) * c.d].to_vec()).collect();
    for l in &layers { xc = layer_cpu(&xc, l, &c); }
    let xc = cpu::rmsnorm(&xc, &wln, c.t, c.d, c.eps);
    let c_logits = matmul_cpu(&xc, &wlm, c.t, c.d, vocab);

    let diff = max_abs_diff(&g_logits, &c_logits);
    let last = &g_logits[(c.t - 1) * vocab..];
    let next = last.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    println!("Ferric tiny-LLM · {:?} · {n_layers} layers · vocab {vocab} · D {} · {} tokens", ctx.backend, c.d, c.t);
    println!("  input ids {ids:?} → predicted next token id = {next}");
    println!("  max|gpu - cpu| over all {}×{vocab} logits = {diff:.3e}", c.t);
    assert!(diff < 2e-3, "llm logits mismatch {diff}");
    println!("✅ A full multi-layer transformer LM forward pass runs on-GPU in Ferric — matches CPU reference");
}
