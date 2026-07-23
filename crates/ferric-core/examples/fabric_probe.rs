//! fabric_probe — which kernels are bit-identical across GPU fabrics?
//!
//! Runs each transformer-critical kernel on deterministic integer-PRNG inputs
//! (no libm anywhere in input generation) and prints an FNV-1a hash of the
//! output *bit patterns*. Run on two machines (Metal vs Vulkan) and diff the
//! lines: matching hash ⇒ that kernel is bit-identical on both fabrics.
//! This is the measurement behind extending Ferrite's verified-behavior
//! envelope from matmul chains to whole transformer forwards.
//!
//!   cargo run --release -p ferric-core --example fabric_probe

use ferric_core::Context;

/// xorshift64* → f32 in [-0.5, 0.5) via mantissa bitcast. Pure integer ops —
/// bit-exact on every platform, every libm. (Same generator as Ferrite's
/// matmul-chain engine.)
fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            f32::from_bits(0x3F80_0000 | (s >> 41) as u32) - 1.5
        })
        .collect()
}

/// Bit-exact CPU replica of the WGSL det_sqrt barriered sequence (Rust f32
/// ops are IEEE round-to-nearest with no fast-math — the reference semantics).
fn det_sqrt_cpu(y: f32) -> f32 {
    if y <= 0.0 {
        return 0.0;
    }
    let hy = 0.5 * y;
    let mut x = f32::from_bits(0x5F37_59DFu32.wrapping_sub(y.to_bits() >> 1));
    for _ in 0..3 {
        let t = x * x;
        let u = hy * t;
        let w = 1.5 - u;
        x = x * w;
    }
    y * x
}

fn fnv(bits: impl IntoIterator<Item = f32>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in bits {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let ctx = Context::new().await.expect("gpu context");
    println!("fabric: {:?} ({})", ctx.backend, ctx.adapter_name);

    const T: usize = 12;
    const D: usize = 64;
    const H: usize = 4;
    const DH: usize = 16;
    let (t, d) = (T as u32, D as u32);

    let x = det(T * D, 1);
    let w = det(D * D, 2);
    let wn = det(D, 3).iter().map(|v| v + 1.0).collect::<Vec<_>>();

    let xt = ctx.tensor(&x, &[T, D]);
    let wt = ctx.tensor(&w, &[D, D]);
    let wnt = ctx.tensor(&wn, &[D]);

    // 1. matmul — the proven bit-identical path (control).
    let mm = ctx.mm(&xt, &wt, t, d, d);
    println!("mm        {:016x}", fnv(ctx.to_vec(&mm).await.unwrap()));

    // 2. rmsnorm — probes 1/sqrt(mean+eps) and division.
    let rms = ctx.rmsnorm_t(&xt, &wnt, t, d, 1e-5);
    println!("rmsnorm   {:016x}", fnv(ctx.to_vec(&rms).await.unwrap()));

    // 3. sqrt alone — plus a forensic diff against a bit-exact CPU replica of
    // the same barriered Newton sequence (Rust IEEE ops, no fast-math): tells
    // us WHICH fabric deviates from plain sequential rounding, and where.
    let x_pos: Vec<f32> = x.iter().map(|v| v * v + 0.25).collect(); // positive, exact ops
    let xpt = ctx.tensor(&x_pos, &[T, D]);
    let sq = ctx.sqrt_t(&xpt);
    let sq_gpu = ctx.to_vec(&sq).await.unwrap();
    println!("sqrt      {:016x}", fnv(sq_gpu.iter().copied()));
    let mut bad = 0usize;
    for (i, (&y, &g)) in x_pos.iter().zip(&sq_gpu).enumerate() {
        let c = det_sqrt_cpu(y);
        if c.to_bits() != g.to_bits() {
            if bad == 0 {
                println!(
                    "  first dev: i={i} y={:08x} cpu={:08x} gpu={:08x}",
                    y.to_bits(),
                    c.to_bits(),
                    g.to_bits()
                );
            }
            bad += 1;
        }
    }
    println!("  sqrt gpu-vs-cpu deviations: {bad}/{}", sq_gpu.len());

    // 4. rope — probes exp/cos/sin (frequency + rotation).
    let rope = ctx.rope_t(&mm, t, H as u32, DH as u32, 10000.0);
    println!("rope      {:016x}", fnv(ctx.to_vec(&rope).await.unwrap()));

    // 5. causal MHA — probes streaming-softmax exp and division.
    let q = ctx.mm(&xt, &wt, t, d, d);
    let k = ctx.mm(&rms, &wt, t, d, d);
    let v = ctx.mm(&xt, &wt, t, d, d);
    let mha = ctx.mha_causal_t(&q, &k, &v, t, H as u32, H as u32, DH as u32);
    println!("mha       {:016x}", fnv(ctx.to_vec(&mha).await.unwrap()));

    // 6. sigmoid — probes exp + division elementwise.
    let sg = ctx.sigmoid_t(&xt);
    println!("sigmoid   {:016x}", fnv(ctx.to_vec(&sg).await.unwrap()));

    // 6b/6c. layernorm + softmax — the storage-chain kernels outside the
    // demo path, probed so their parity is measured, not assumed.
    let bias = det(D, 4);
    let bt = ctx.tensor(&bias, &[D]);
    let ln = ctx.layernorm_t(&xt, &wnt, &bt, t, d, 1e-5);
    println!("layernorm {:016x}", fnv(ctx.to_vec(&ln).await.unwrap()));
    let sm = ctx.softmax_t(&xt, t, d);
    println!("softmax   {:016x}", fnv(ctx.to_vec(&sm).await.unwrap()));
    let rt = ctx.rmsnorm_tree_t(&xt, &wnt, t, d, 1e-5);
    println!("rms-tree  {:016x}", fnv(ctx.to_vec(&rt).await.unwrap()));
    // the heterogeneous row: SAME algorithm on the parallel CPU — the digest
    // must equal the GPU row's on every substrate.
    let rc = ferric_core::rmsnorm_tree_cpu(&x, &wn, T, D, 1e-5);
    println!("rms-tcpu  {:016x}", fnv(rc));
    let lt = ctx.layernorm_tree_t(&xt, &wnt, &bt, t, d, 1e-5);
    println!("ln-tree   {:016x}", fnv(ctx.to_vec(&lt).await.unwrap()));
    let lc = ferric_core::layernorm_tree_cpu(&x, &wn, &bias, T, D, 1e-5);
    println!("ln-tcpu   {:016x}", fnv(lc));
    let st = ctx.softmax_tree_t(&xt, t, d);
    println!("sm-tree   {:016x}", fnv(ctx.to_vec(&st).await.unwrap()));
    let sc = ferric_core::softmax_tree_cpu(&x, T, D);
    println!("sm-tcpu   {:016x}", fnv(sc));

    // 7. the full demo LM forward — the composite target.
    let ids: Vec<u32> = (0..T as u32).map(|i| (i * 7 + 1) % 32).collect();
    let lg = ferric_core::demo::logits(&ctx, &ids).await.unwrap();
    println!("demo-lm   {:016x}", fnv(lg));
}
