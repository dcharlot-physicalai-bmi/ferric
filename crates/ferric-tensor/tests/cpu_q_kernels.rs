//! **The CPU fabric's kernels against an independent float64 reading of ggml's formats.**
//!
//! Every weight format `cpu_q` multiplies natively, through BOTH kernel families (NEON+DotProd where the
//! machine has it, and the portable scalar kernels every other target compiles), with int8 and with f32
//! activations, over tile shapes that reach every microkernel the driver dispatches (1, 3 and 6
//! activation rows x 8 weight rows: the 4x1, 2x1, 2x4, 1x4 and 1x1 tiles).
//!
//! What "agrees" means, stated per arm:
//!  - int8 activations: the kernel must equal `Σ w·q(x)` — the f64 reference applied to `x` rounded
//!    the way the kernels are specified to round it — to 5e-7 of `Σ|w·q(x)|`. The rounding itself is
//!    a SEPARATE, reported number (the int8 activation error), bounded only loosely.
//!  - f32 activations (and F32/F16/BF16 weights): `Σ w·x` to the same bound.
//!
//! Negative controls: for each format one plausible WRONG reading (`common/ggml_ref.rs`) must sit >= 20x
//! outside this test's own bound — a test that cannot tell them apart proves nothing.
#[path = "common/ggml_ref.rs"]
mod ggml_ref;

use ferric_tensor::cpu_q::{self, Opts, QWeight, WType};
use ggml_ref as r;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn range(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.unit() }
}

fn f16b(v: f64) -> [u8; 2] { half::f16::from_f64(v).to_bits().to_le_bytes() }

/// `rows` synthetic rows of format `t`: random codes/bits/scale bytes, with the f16 super-scales
/// overwritten by sane magnitudes (random bytes there would be inf/nan or 1e4).
fn synth(t: u32, rows: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let (qk, bb) = r::block(t);
    let mut out = Vec::with_capacity(rows * k / qk * bb);
    for _ in 0..rows * k / qk {
        let mut b: Vec<u8> = (0..bb).map(|_| rng.next() as u8).collect();
        let sc = |rng: &mut Rng| { let s = rng.range(1e-3, 4e-2); if rng.next() & 1 == 1 { -s } else { s } };
        match t {
            r::F32 => b.copy_from_slice(&(rng.range(-0.2, 0.2) as f32).to_le_bytes()),
            r::F16 => b.copy_from_slice(&f16b(rng.range(-0.2, 0.2))),
            r::BF16 => b.copy_from_slice(&half::bf16::from_f64(rng.range(-0.2, 0.2)).to_bits().to_le_bytes()),
            r::Q4_0 | r::Q5_0 | r::Q8_0 => b[0..2].copy_from_slice(&f16b(sc(rng))),
            r::Q4_1 | r::Q5_1 => { b[0..2].copy_from_slice(&f16b(sc(rng))); b[2..4].copy_from_slice(&f16b(rng.range(-0.1, 0.1))); }
            r::Q4_K | r::Q5_K => { b[0..2].copy_from_slice(&f16b(rng.range(1e-4, 3e-3))); b[2..4].copy_from_slice(&f16b(rng.range(1e-4, 3e-3))); }
            r::Q6_K => b[208..210].copy_from_slice(&f16b(sc(rng) * 0.1)),
            _ => unreachable!(),
        }
        out.extend(b);
    }
    out
}

fn acts(m: usize, k: usize, rng: &mut Rng) -> Vec<f32> {
    let mut x: Vec<f32> = (0..m * k).map(|_| rng.range(-2.0, 2.0) as f32).collect();
    // Outliers, as real activations carry: they set a block's scale and squeeze everything else.
    for i in (0..x.len()).step_by(97) { x[i] *= 6.0; }
    x
}

struct Worst { right: f64, wrong: f64, act: f64 }

/// One format through one kernel arm: the worst right-reference error, the worst wrong-reference error
/// (what this test's own max-error criterion would see if the kernel read the format that way), and the
/// worst int8 activation error.
fn check(t: u32, opts: Opts, rng: &mut Rng) -> Worst {
    let ty = WType::from_ggml(t).unwrap();
    let (qk, _) = r::block(t);
    let k = if qk == 256 { 512 } else if qk == 32 { 256 } else { 37 * 8 + 3 };
    let rows = 8;
    let data = synth(t, rows, k, rng);
    let rb = data.len() / rows;
    let w = QWeight::new(ty, rows, k, data.clone()).unwrap();
    let mut worst = Worst { right: 0.0, wrong: 0.0, act: 0.0 };
    for &m in &[1usize, 3, 6] {
        let x = acts(m, k, rng);
        let mut y = vec![0f32; m * rows];
        cpu_q::matmul_opts(&mut [(&w, &mut y)], &x, m, opts);
        for a in 0..m {
            let xa = &x[a * k..(a + 1) * k];
            let exact: Vec<f64> = xa.iter().map(|&v| v as f64).collect();
            let seen = match (r::act_block(t), opts.f32act) {
                (Some(b), false) => {
                    let (codes, scales, q) = r::quantize_act(xa, b);
                    // The library's quantizer must BE the specified one, code for code, scale for scale.
                    let (lc, ls) = cpu_q::quantized_activation(ty, xa, opts).expect("an int8 arm quantizes");
                    assert_eq!(lc, codes, "{}: the library's activation codes differ from their specification", r::name(t));
                    assert_eq!(ls, scales, "{}: the library's activation scales differ from their specification", r::name(t));
                    q
                }
                _ => exact.clone(),
            };
            for row in 0..rows {
                let wr = &data[row * rb..(row + 1) * rb];
                let (want, scale) = r::dot(&r::dequant(t, wr, k), &seen);
                let (bad, _) = r::dot(&r::dequant_wrong(t, wr, k), &seen);
                let (ideal, _) = r::dot(&r::dequant(t, wr, k), &exact);
                let got = y[a * rows + row] as f64;
                worst.right = worst.right.max((got - want).abs() / scale);
                worst.wrong = worst.wrong.max((got - bad).abs() / scale);
                worst.act = worst.act.max((want - ideal).abs() / scale);
            }
        }
    }
    worst
}

fn arms() -> Vec<(&'static str, Opts)> {
    let mut v = vec![("scalar/int8", Opts { neon: false, f32act: false }), ("scalar/f32act", Opts { neon: false, f32act: true })];
    if cfg!(target_arch = "aarch64") && cpu_q::kernel_family() == "neon+dotprod" {
        v.push(("neon/int8", Opts { neon: true, f32act: false }));
        v.push(("neon/f32act", Opts { neon: true, f32act: true }));
    }
    v
}

#[test]
fn every_format_every_arm_matches_the_f64_reference() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    // Measured: every arm, every format <= 4.7e-8 (f32 accumulation over K = 256..512). 5e-7 leaves 10x
    // for other seeds without admitting a real defect: the wrong readings sit at 1e-4 .. 1e-1.
    let tol = 5e-7;
    let mut failures = Vec::new();
    for (arm, opts) in arms() {
        for &t in &r::ALL {
            let w = check(t, opts, &mut rng);
            let ok_right = w.right <= tol;
            // The control holds when this test would reject the wrong reading by >= 20x its own bound.
            let ok_ctrl = w.wrong >= 20.0 * tol;
            println!("{arm:<14} {:<5} right {:.2e}  wrong-reading {:.2e} ({:>9.0}x)  int8-activation error {:.2e}",
                     r::name(t), w.right, w.wrong, w.wrong / w.right.max(1e-12), w.act);
            if !ok_right { failures.push(format!("{arm} {}: {:.3e} > {tol:e}", r::name(t), w.right)); }
            if !ok_ctrl { failures.push(format!("{arm} {}: wrong reading only {:.3e} away — the control cannot see it", r::name(t), w.wrong)); }
            // The int8 rounding is a specified approximation; anything near 1% of |w·x| is a defect.
            if w.act > 2e-2 { failures.push(format!("{arm} {}: int8 activation error {:.3e}", r::name(t), w.act)); }
        }
    }
    assert!(failures.is_empty(), "CPU kernel conformance failed:\n  {}", failures.join("\n  "));
}

#[test]
fn the_pool_runs_every_item_once_and_flattens_nesting() {
    let pool = cpu_q::pool();
    let n = 10_007;
    let hits: Vec<std::sync::atomic::AtomicU32> = (0..n).map(|_| std::sync::atomic::AtomicU32::new(0)).collect();
    pool.for_each(n, |i| {
        hits[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // A job that submits a job must run it inline, not deadlock on the pool it occupies.
        if i % 1000 == 0 { pool.for_each(3, |_| {}); }
    });
    assert!(hits.iter().all(|h| h.load(std::sync::atomic::Ordering::Relaxed) == 1), "an item ran twice or never");
    // Many small jobs back to back: the epoch handshake must not lose or double one.
    let total = std::sync::atomic::AtomicUsize::new(0);
    for _ in 0..2000 {
        pool.run(|_, _| { total.fetch_add(1, std::sync::atomic::Ordering::Relaxed); });
    }
    assert_eq!(total.load(std::sync::atomic::Ordering::Relaxed), 2000 * pool.threads());
}

#[test]
fn a_fused_projection_splits_into_exact_row_ranges() {
    let mut rng = Rng(7);
    let (rows, k) = (12, 256);
    let data = synth(r::Q4_K, rows, k, &mut rng);
    let w = QWeight::new(WType::Q4_K, rows, k, data.clone()).unwrap();
    let parts = QWeight::new(WType::Q4_K, rows, k, data).unwrap().split_rows(&[6, 4, 2]).unwrap();
    let x = acts(2, k, &mut rng);
    let mut whole = vec![0f32; 2 * rows];
    cpu_q::matmul(&w, &x, 2, &mut whole);
    let mut off = 0;
    for p in &parts {
        let mut y = vec![0f32; 2 * p.rows];
        cpu_q::matmul(p, &x, 2, &mut y);
        for a in 0..2 { for rr in 0..p.rows { assert_eq!(y[a * p.rows + rr], whole[a * rows + off + rr]); } }
        off += p.rows;
    }
}
