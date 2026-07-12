// A full single-head pre-norm transformer block, built as a Ferric graph and run ENTIRELY on the GPU
// (buffers chain op→op; one readback at the very end), validated against a pure-Rust CPU reference.
// This is the L2/L3 substrate: a real model subgraph executing on the fabric we own.
use ferric_core::{cpu, matmul_cpu, max_abs_diff, Context};

fn main() { pollster::block_on(run()); }

fn gen(n: usize, a: usize, b: usize, s: f32) -> Vec<f32> {
    (0..n).map(|i| ((i * a % b) as f32 - (b as f32) / 2.0) * s).collect()
}

async fn run() {
    let ctx = Context::new().await.expect("ctx");
    println!("Ferric transformer block · {:?} · {}", ctx.backend, ctx.adapter_name);
    let (t, d, h) = (16usize, 64usize, 256usize);
    let eps = 1e-5f32;
    let scale = 1.0 / (d as f32).sqrt();

    // deterministic inputs + weights
    let x = gen(t * d, 7, 23, 0.1);
    let (wq, wk, wv, wo) = (gen(d * d, 3, 17, 0.05), gen(d * d, 5, 19, 0.05), gen(d * d, 11, 13, 0.05), gen(d * d, 7, 29, 0.05));
    let (ln1w, ln1b) = (gen(d, 1, 7, 0.01).iter().map(|v| 1.0 + v).collect::<Vec<_>>(), gen(d, 2, 11, 0.02));
    let (ln2w, ln2b) = (gen(d, 3, 5, 0.01).iter().map(|v| 1.0 + v).collect::<Vec<_>>(), gen(d, 4, 13, 0.02));
    let (w1, w2) = (gen(d * h, 5, 31, 0.03), gen(h * d, 7, 23, 0.03));

    // ---- GPU: build the block as a graph, everything stays on device until to_vec ----
    let (u, uu) = (|v: &[f32], s: &[usize]| ctx.tensor(v, s), t as u32);
    let xt = u(&x, &[t, d]);
    let (wqt, wkt, wvt, wot) = (u(&wq, &[d, d]), u(&wk, &[d, d]), u(&wv, &[d, d]), u(&wo, &[d, d]));
    let (l1w, l1b, l2w, l2b) = (u(&ln1w, &[d]), u(&ln1b, &[d]), u(&ln2w, &[d]), u(&ln2b, &[d]));
    let (w1t, w2t) = (u(&w1, &[d, h]), u(&w2, &[h, d]));
    let (du, hu) = (d as u32, h as u32);

    let hn = ctx.layernorm_t(&xt, &l1w, &l1b, uu, du, eps);
    let q = ctx.mm(&hn, &wqt, uu, du, du);
    let k = ctx.mm(&hn, &wkt, uu, du, du);
    let v = ctx.mm(&hn, &wvt, uu, du, du);
    let a = ctx.attention_t(&q, &k, &v, uu, uu, du, du, scale);
    let o = ctx.mm(&a, &wot, uu, du, du);
    let x1 = ctx.add_t(&xt, &o);
    let h2 = ctx.layernorm_t(&x1, &l2w, &l2b, uu, du, eps);
    let m1 = ctx.mm(&h2, &w1t, uu, du, hu);
    let s = ctx.silu_t(&m1);
    let m2 = ctx.mm(&s, &w2t, uu, hu, du);
    let xo = ctx.add_t(&x1, &m2);
    let gpu = ctx.to_vec(&xo).await.expect("readback");

    // ---- CPU reference (same math, plain Rust) ----
    let hn_c = cpu::layernorm(&x, &ln1w, &ln1b, t, d, eps);
    let q_c = matmul_cpu(&hn_c, &wq, t, d, d);
    let k_c = matmul_cpu(&hn_c, &wk, t, d, d);
    let v_c = matmul_cpu(&hn_c, &wv, t, d, d);
    let a_c = cpu::attention(&q_c, &k_c, &v_c, t, t, d, d, scale);
    let o_c = matmul_cpu(&a_c, &wo, t, d, d);
    let x1_c = cpu::add(&x, &o_c);
    let h2_c = cpu::layernorm(&x1_c, &ln2w, &ln2b, t, d, eps);
    let m1_c = matmul_cpu(&h2_c, &w1, t, d, h);
    let s_c = cpu::silu(&m1_c);
    let m2_c = matmul_cpu(&s_c, &w2, t, h, d);
    let cpu_out = cpu::add(&x1_c, &m2_c);

    let diff = max_abs_diff(&gpu, &cpu_out);
    println!("block [{}x{}], 12 ops on-GPU, 1 readback → max|gpu-cpu| = {:.3e}", t, d, diff);
    println!("out[:6] = {:?}", &gpu[..6]);
    assert!(diff < 1e-4, "block diverged: {diff}");
    println!("✅ TRANSFORMER BLOCK VALIDATED — runs entirely on {:?}, matches CPU", ctx.backend);
}
