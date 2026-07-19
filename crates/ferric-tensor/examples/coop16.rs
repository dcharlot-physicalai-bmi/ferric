//! NVIDIA/Intel tensor-core GEMM via 16×16 f16-input cooperative matrix. a=0.01,b=0.02 ⇒ K·2e-4
//! (within f16 tolerance), plus a vs-naive check. This is the path that replaces the all-zeros
//! 8×8-f32 coop kernel on Vulkan.
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;
use std::time::Instant;
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("{:?} · {} · coop16_ok={}", ctx.backend, ctx.adapter_name, ctx.coop16_ok());
    if !ctx.coop16_ok() { println!("⏭  not a coop16 fabric (Vulkan + coop + f16)"); return; }
    let mut ok = true;
    for (m, k, n) in [(16usize, 16usize, 16usize), (512, 512, 512), (64, 2048, 2048), (2048, 5120, 5120)] {
        let a = Tensor::from_vec(&ctx, &vec![0.01f32; m * k], &[m, k]);
        let b = Tensor::from_vec(&ctx, &vec![0.02f32; k * n], &[k, n]);
        let c = a.matmul_coop16(&b).to_vec().await;
        let expect = k as f32 * 0.01 * 0.02;
        let e = c.iter().map(|v| (v - expect).abs()).fold(0f32, f32::max) / expect;
        let p = e < 0.05; ok &= p;
        let bench = |f: &dyn Fn() -> Tensor| { let mut l = None; let t = Instant::now();
            for _ in 0..20 { l = Some(f()); } let _ = pollster::block_on(l.unwrap().to_vec()); t.elapsed().as_secs_f64() / 20.0 };
        let _ = a.matmul_coop16(&b).to_vec().await;
        let ct = bench(&|| a.matmul_coop16(&b));
        let flop = 2.0 * m as f64 * k as f64 * n as f64;
        println!("  {} [{m:>4}×{k:>4}]·[{k}×{n}]  c[0]={:.4} exp {:.4}  relΔ={:.2e}  {:>7.0} GFLOP/s",
            if p { "✅" } else { "❌ WRONG" }, c[0], expect, e, flop / ct / 1e9);
    }
    // Correctness vs the naive f32 matmul (f16 inputs → expect ~1e-3 relative).
    let a = Tensor::from_vec(&ctx, &(0..256 * 256).map(|i| ((i as f32 * 0.001).sin()) * 0.1).collect::<Vec<_>>(), &[256, 256]);
    let b = Tensor::from_vec(&ctx, &(0..256 * 256).map(|i| ((i as f32 * 0.002).sin()) * 0.1).collect::<Vec<_>>(), &[256, 256]);
    let coop = a.matmul_coop16(&b).to_vec().await;
    let naive = a.matmul_naive(&b).to_vec().await;
    let sc = naive.iter().map(|v| v.abs()).fold(1e-3, f32::max);
    let e = coop.iter().zip(&naive).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) / sc;
    println!("  vs naive 256³: relΔ={:.2e} ({})", e, if e < 0.05 { "f16-close ✅" } else { "❌" });
    println!("{}", if ok { "✅ NVIDIA tensor-core coop16 produces correct results" } else { "❌ still wrong" });
    assert!(ok);
}
