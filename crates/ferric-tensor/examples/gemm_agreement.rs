//! **Do the two GEMM kernels agree BIT FOR BIT?**
//!
//! `matmul` picks between `MATMUL_WGSL` (naive) and `TILED_MATMUL_WGSL` (register-blocked tiled) by
//! reading `GEMM_CACHE`, which `autotune_matmul` fills by TIMING the two and keeping the faster. So
//! the kernel that runs a given shape is decided by a wall-clock race, and the cache is THREAD-LOCAL
//! — an untuned thread gets naive.
//!
//! If the two kernels disagree numerically by even one ULP, then Ferric's logits depend on who won a
//! race on a loaded machine, and "bit-identical across fabrics" would be a claim about a coin toss.
//! Nothing in the tree asserted this: `matmul_tiled` appears only in `lib.rs` and `bench.rs`.
//!
//!   cargo run -p ferric-tensor --example gemm_agreement --release
use ferric_tensor::Tensor;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }

async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    println!("adapter: {} [{:?}]\n", ctx.adapter_name, ctx.backend);

    // deterministic, non-degenerate inputs: a constant or a ramp can hide reassociation entirely,
    // because summing equal terms is order-independent in floating point.
    let mk = |n: usize, seed: u64| -> Vec<f32> {
        (0..n).map(|i| {
            let x = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(seed);
            ((x >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        }).collect()
    };

    let shapes = [(64usize, 64usize, 64usize), (128, 256, 128), (256, 2048, 256),
                  (512, 1024, 512), (37, 53, 71), (1, 4096, 4096), (320, 4096, 320)];
    println!("{:>18} {:>12} {:>14} {:>12} {:>10}", "shape (m,k,n)", "elements", "differing", "max |Δ|", "max ULP");
    let mut worst_ulp = 0u32;
    let mut any_diff = false;
    for (m, k, n) in shapes {
        let a = Tensor::from_vec(&ctx, &mk(m * k, 1), &[m, k]);
        let b = Tensor::from_vec(&ctx, &mk(k * n, 2), &[k, n]);
        let naive = a.matmul_naive(&b).to_vec().await;
        let tiled = a.matmul_tiled(&b).to_vec().await;
        assert_eq!(naive.len(), tiled.len());
        let (mut ndiff, mut maxd, mut maxulp) = (0usize, 0f32, 0u32);
        for (x, y) in naive.iter().zip(&tiled) {
            if x.to_bits() != y.to_bits() {
                ndiff += 1;
                maxd = maxd.max((x - y).abs());
                // ULP distance for same-sign finite floats
                let (bx, by) = (x.to_bits() as i64, y.to_bits() as i64);
                maxulp = maxulp.max((bx - by).unsigned_abs() as u32);
            }
        }
        if ndiff > 0 { any_diff = true; }
        worst_ulp = worst_ulp.max(maxulp);
        println!("{:>18} {:>12} {:>14} {:>12.3e} {:>10}",
                 format!("({m},{k},{n})"), naive.len(),
                 format!("{ndiff} ({:.2}%)", 100.0 * ndiff as f32 / naive.len() as f32), maxd, maxulp);
    }
    println!();
    if any_diff {
        println!("⛔ THE TWO GEMM KERNELS DISAGREE (worst {worst_ulp} ULP).");
        println!("   `matmul` selects between them by a WALL-CLOCK RACE in `autotune_matmul`, cached");
        println!("   THREAD-LOCALLY. So the same model, same machine, same weights can produce");
        println!("   different logits depending on which kernel won — and an untuned thread differs");
        println!("   from a tuned one. Any bit-identity claim has to pin the kernel, not the fabric.");
    } else {
        println!("✅ the two GEMM kernels are bit-identical on every shape tested, so the autotuner's");
        println!("   wall-clock choice cannot move the numerics. The determinism claim survives it.");
    }
}
