//! **Training on the tensor units** — a real Var/Adam training loop with `FERRIC_METAL4` on vs off.
//! Every GEMM in the step routes resident when opted in: the two forward matmuls and the four
//! backward ones (dA = g·Bᵀ, dB = Aᵀ·g per layer — the transposes are materialized by autograd,
//! then the plain matmul routes). Loss must FALL under both paths (fp16 inputs change the
//! trajectory slightly, not the outcome), and the resident path must be faster per step.
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;
use std::time::Instant;

fn seq(n: usize, s: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 * 0.7 + s).sin()) * 0.05).collect()
}

fn main() {
    pollster::block_on(run());
}
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("adapter: {} ({:?})", ctx.adapter_name, ctx.backend);
    // Two regimes, honestly labeled: at (512,1024) the step is FRAMEWORK-bound (elementwise ops,
    // transposes, Adam, per-op dispatch — the GEMMs are ~20% of the step), so the tensor units only
    // shave that slice. At (1024,2048) the six GEMMs dominate and the speedup shows.
    for (b, d) in [(512usize, 1024usize), (1024, 2048)] {
        run_size(&ctx, b, d).await;
    }
}

async fn run_size(ctx: &Arc<ferric_core::Context>, b: usize, d: usize) {
    println!("
=== batch {b}, width {d} (6 GEMMs/step ≈ {:.1} GFLOP) ===", 6.0 * 2.0 * (b * d * d) as f64 / 1e9);
    let xt = Tensor::from_vec(ctx, &seq(b * d, 1.0), &[b, d]);
    let yt = Tensor::from_vec(ctx, &seq(b * d, 9.0), &[b, d]);

    let mut results = Vec::new();
    for m4 in [false, true] {
        if m4 {
            std::env::set_var("FERRIC_METAL4", "1");
        } else {
            std::env::remove_var("FERRIC_METAL4");
        }
        let mut params = vec![
            Tensor::from_vec(ctx, &seq(d * d, 2.0), &[d, d]),
            Tensor::from_vec(ctx, &seq(d * d, 5.0), &[d, d]),
        ];
        let mut adam = Adam::new(&params, 1e-4);
        let step = |params: &mut Vec<Tensor>, adam: &mut Adam| {
            let xv = Var::leaf(xt.clone());
            let yv = Var::leaf(yt.clone());
            let p: Vec<Var> = params.iter().map(|t| Var::leaf(t.clone())).collect();
            let h = xv.matmul(&p[0]).relu();
            let out = h.matmul(&p[1]);
            let e = out.sub(&yv);
            // mean over the BATCH dim only (per-sample MSE summed over features): with a full
            // mean the gradient g = 2e/(b·d) ~ 1e-7 sits below fp16's normal range and the
            // tensor units' fp16 inputs crush it (slower convergence — observed, not theoretical).
            // Keeping gradients O(1e-4) is the loss-scaling discipline every fp16 trainer uses.
            let loss = e.mul(&e).mean(&[0]).sum_all();
            loss.backward();
            let grads: Vec<Tensor> = p.iter().map(|v| v.grad().unwrap()).collect();
            adam.step(params, &grads);
            loss
        };
        // warm (shader/pipeline/shape caches), then measure
        let first = pollster::block_on(step(&mut params, &mut adam).value().to_vec())[0];
        let t0 = Instant::now();
        let steps = 15;
        let mut last = first;
        for _ in 0..steps {
            last = pollster::block_on(step(&mut params, &mut adam).value().to_vec())[0];
        }
        let per = t0.elapsed().as_secs_f64() / steps as f64;
        let label = if m4 { "Metal4 resident" } else { "portable WGSL " };
        println!("  {label}: {:.2} ms/step   loss {first:.5} → {last:.5}", per * 1e3);
        assert!(last < first * 0.9, "loss must fall ({label}): {first} → {last}");
        results.push((per, last));
    }
    let speedup = results[0].0 / results[1].0;
    println!("\n  training-step speedup on the tensor units: {speedup:.1}x");
    assert!(
        (results[0].1 - results[1].1).abs() < 0.2 * results[0].1.abs().max(1e-3),
        "both paths must converge to the same neighbourhood: {} vs {}",
        results[0].1,
        results[1].1
    );
    if 6 * 2 * b * d * d > 20_000_000_000 {
        assert!(speedup > 1.5, "tensor units should clearly win once the GEMMs dominate");
    }
    println!("✅ loss falls on both paths, tensor units {speedup:.1}x faster per step");
}
