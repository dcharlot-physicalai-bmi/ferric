//! Kernel fusion: an elementwise expression compiled to ONE WGSL kernel matches the same expression
//! composed op-by-op — but as a single dispatch with no intermediate buffers. Validates SwiGLU
//! (silu(g)·u) and a 3-input expression, and times fused vs unfused on a big tensor.
use ferric_core::Context;
use ferric_tensor::fuse::{eval, E};
use ferric_tensor::Tensor;
use std::sync::Arc;
use std::time::Instant;

fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.017 + s).sin())).collect() }
fn maxdiff(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let n = 1 << 20; // 1M elements
    let (g, u, c) = (
        Tensor::from_vec(&ctx, &seq(n, 1.0), &[n]),
        Tensor::from_vec(&ctx, &seq(n, 2.0), &[n]),
        Tensor::from_vec(&ctx, &seq(n, 3.0), &[n]),
    );
    let mut ok = true;

    // SwiGLU gate: silu(g)·u  — fused (1 kernel) vs composed (2 ops + a temp)
    let fused = eval(&[&g, &u], &E::input(0).silu().mul(&E::input(1))).to_vec().await;
    let composed = g.silu().mul(&u).to_vec().await;
    let d1 = maxdiff(&fused, &composed); ok &= d1 < 1e-5;
    println!("  {} fused SwiGLU (silu(g)·u) == composed        max|Δ| = {:.2e}  [1 kernel vs 2]", if d1 < 1e-5 { "✅" } else { "❌" }, d1);

    // 3-input: (g + u) · relu(c) — fused (1 kernel) vs composed (3 ops + 2 temps)
    let e = E::input(0).add(&E::input(1)).mul(&E::input(2).relu());
    let fused3 = eval(&[&g, &u, &c], &e).to_vec().await;
    let comp3 = g.add(&u).mul(&c.relu()).to_vec().await;
    let d2 = maxdiff(&fused3, &comp3); ok &= d2 < 1e-5;
    println!("  {} fused (g+u)·relu(c) == composed             max|Δ| = {:.2e}  [1 kernel vs 3]", if d2 < 1e-5 { "✅" } else { "❌" }, d2);

    // timing: fused vs unfused SwiGLU over the 1M tensor (warm)
    for _ in 0..3 { let _ = eval(&[&g, &u], &E::input(0).silu().mul(&E::input(1))).to_vec().await; }
    let t0 = Instant::now();
    for _ in 0..20 { let _ = eval(&[&g, &u], &E::input(0).silu().mul(&E::input(1))).to_vec().await; }
    let tf = t0.elapsed().as_secs_f64() / 20.0;
    let t1 = Instant::now();
    for _ in 0..20 { let _ = g.silu().mul(&u).to_vec().await; }
    let tu = t1.elapsed().as_secs_f64() / 20.0;
    println!("  timing (1M elems): fused {:.0}µs  vs  unfused {:.0}µs  ({:.2}× fewer dispatches worth)", tf * 1e6, tu * 1e6, tu / tf);

    // matmul-epilogue fusion: silu(x·Wᵀ) — the FFN gate — in ONE kernel vs linear + silu (2 kernels)
    let (rows, ind, outd) = (32usize, 48, 64);
    let xa = Tensor::from_vec(&ctx, &seq(rows * ind, 4.0), &[rows, ind]);
    let wg = Tensor::from_vec(&ctx, &seq(outd * ind, 5.0), &[outd, ind]);
    let fe = xa.matmul_bt_act(&wg, 2).to_vec().await; // 2 = silu
    let ce = ferric_tensor::nn::linear_hf(&xa, &wg).silu().to_vec().await;
    let d3 = maxdiff(&fe, &ce); ok &= d3 < 1e-5;
    println!("  {} fused silu(x·Wᵀ) (FFN gate) == linear+silu  max|Δ| = {:.2e}  [1 kernel vs 2]", if d3 < 1e-5 { "✅" } else { "❌" }, d3);

    println!("{}", if ok { "✅ Kernel fusion (runtime WGSL codegen) is exact — one dispatch, no intermediate buffers" } else { "❌ fusion mismatch" });
    assert!(ok);
}
