//! **How accurate are the rope angles the GPU computes?** Rotates a unit vector (x1 = 1, x2 = 0 in every
//! pair, so the output IS cos and sin) at every position and compares with float64 truth, for the
//! kernel's on-device `exp`/`log`/`cos`/`sin` and for a host-built float32 cos/sin table (the way the
//! reference implementations build theirs).
//!
//! The third column is the one a conformance gate sees: how far the GPU's angles sit from a reference
//! that builds its table on the host in float32.
//!
//!   rope_precision [base] [head_dim] [max_pos]
use std::sync::Arc;
use ferric_tensor::Tensor;

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let base: f32 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(10000.0);
    let dh: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let tmax: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(32768);
    let half = dh / 2;
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    // positions 0..tmax in chunks (one head, so the tensor is [t, dh])
    let mut bands = [0f64; 8];
    let mut bands_tab = [0f64; 8];
    let mut bands_vs = [0f64; 8];
    let band = |p: usize| (p.max(1) as f64).log2().floor().max(8.0) as usize / 2 - 4;
    let chunk = 4096;
    let mut worst = (0f64, 0usize, 0usize);
    for start in (0..tmax).step_by(chunk) {
        let t = chunk.min(tmax - start);
        let mut x = vec![0f32; t * dh];
        for r in 0..t { for c in 0..half { x[r * dh + c] = 1.0; } }
        let xt = Tensor::from_vec(&ctx, &x, &[t, dh]);
        let gpu = xt.rope(1, dh, base, start).to_vec().await;
        // the reference way: inv_freq in f32 (1 / base^(2c/d)), angle = f32(pos) * inv, cos/sin correctly rounded
        let inv32: Vec<f32> = (0..half).map(|c| 1.0 / base.powf((2 * c) as f32 / dh as f32)).collect();
        let (mut ct, mut st) = (vec![0f32; t * dh], vec![0f32; t * dh]);
        for r in 0..t { for c in 0..half {
            let ang = (start + r) as f32 * inv32[c];
            ct[r * dh + c] = (ang as f64).cos() as f32; st[r * dh + c] = (ang as f64).sin() as f32;
        } }
        let tab = xt.apply_rope_costable(&Tensor::from_vec(&ctx, &ct, &[t, dh]), &Tensor::from_vec(&ctx, &st, &[t, dh]), 1, dh)
            .to_vec().await;
        for r in 0..t {
            let p = start + r;
            for c in 0..half {
                let inv64 = 1.0 / (base as f64).powf((2 * c) as f64 / dh as f64);
                let (tc, ts) = ((p as f64 * inv64).cos(), (p as f64 * inv64).sin());
                let e = (gpu[r * dh + c] as f64 - tc).abs().max((gpu[r * dh + c + half] as f64 - ts).abs());
                let et = (tab[r * dh + c] as f64 - tc).abs().max((tab[r * dh + c + half] as f64 - ts).abs());
                let b = band(p).min(7);
                bands[b] = bands[b].max(e); bands_tab[b] = bands_tab[b].max(et);
                let ev = ((gpu[r * dh + c] - tab[r * dh + c]) as f64).abs().max(((gpu[r * dh + c + half] - tab[r * dh + c + half]) as f64).abs());
                bands_vs[b] = bands_vs[b].max(ev);
                if e > worst.0 { worst = (e, p, c); }
            }
        }
    }
    println!("base {base}, head_dim {dh}, positions 0..{tmax}, adapter {}", ctx.adapter_name);
    println!("  positions        on-device exp/cos    host f32 table     on-device vs host table   (max |error|; first two vs float64 truth)");
    for b in 0..8 {
        let lo = if b == 0 { 0 } else { 1usize << (2 * b + 8) };
        let hi = 1usize << (2 * b + 10);
        if lo >= tmax { break; }
        println!("  {lo:>7}..{:<7}   {:>12.2e}        {:>12.2e}        {:>12.2e}", hi.min(tmax), bands[b], bands_tab[b], bands_vs[b]);
    }
    println!("  worst on-device: {:.2e} at position {}, pair {}", worst.0, worst.1, worst.2);
}
