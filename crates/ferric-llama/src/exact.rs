//! **Exact verification of the algebraic identities the architecture rests on.**
//!
//! Three claims in this crate are *identities*, not approximations, and the f32 tests check them
//! to a tolerance -- 2e-4 for the hyper-connection closed form, 2e-5 for MLA absorption. A
//! tolerance says "close"; it cannot distinguish an identity from an approximation that happens
//! to be good at the tested point, and 2e-4 is about 1700 f32 ulps of slack for a bug to hide in.
//!
//! ## Why a finite field, and why random points constitute a proof
//!
//! Each identity is a POLYNOMIAL in its inputs once the gates and attention weights are treated as
//! given coefficients -- which they are: `H^(m)_i = emb + Σ q^(l)_i out^(l)` holds for *any* `q`,
//! sigmoid or not. Its coefficients are all `±1`, so it is a polynomial identity over the integers,
//! and an integer polynomial identity holds over ℚ exactly when it holds over GF(p) for any prime
//! `p` larger than its coefficients. Schwartz–Zippel: a non-zero polynomial of total degree `D`
//! vanishes at a uniformly random point of `GF(p)^n` with probability at most `D/p`. With
//! `p = 2^61 − 1` and `D ≤ 3`, one exact agreement bounds the probability that the identity is
//! false at `1.3e-18`; the eight independent trials below drive it below `1e-140`. That is a
//! probabilistic proof with a stated bound, which is more than any tolerance can say.
//!
//! The field is also why this needs no dependency. Ferric builds only from `vendor/`, and big
//! rationals are not in it; `u64` residues mod a Mersenne prime never overflow a `u128` product,
//! never grow, and never need a gcd.
//!
//! ## What the finite field cannot do, and what is done instead
//!
//! A field has no metric, so it cannot measure how far the GPU's f32 answer is from the exact one.
//! For that, `gpu_hyper_connection_step_error_is_f32_rounding_only` restricts the inputs to 11-bit
//! dyadics (`k/1024`) and uses the GPU's own gates read back as f32: one sublayer's exact result
//! is then a 37-bit integer over a power-of-two denominator, which f64 represents EXACTLY. So the
//! oracle for a single step is exact with no big-number arithmetic at all. Depth beyond one would
//! need 61+ bits and is left to the f32 test, whose tolerance the measured per-step error now
//! justifies.
//!
//! ## What is written independently of what
//!
//! The recursion is written from the SPEC's definitions of the read and write halves; the closed
//! form from `hc.rs`'s derivation. Neither calls the other and neither calls `hc.rs`, so the
//! identity test is a cross-check between definition and derivation. The GPU step test then takes
//! the GPU's own gates as coefficients, so its residual is f32 rounding in the reduce and the
//! write-back and nothing else.

// ─────────────────────────── GF(2^61 − 1) ───────────────────────────

/// A Mersenne prime: `2^61 − 1`. Products of two residues fit in `u128` with room to spare.
const P: u64 = (1u64 << 61) - 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct F(u64);

impl F {
    fn add(self, o: F) -> F { let s = self.0 + o.0; F(if s >= P { s - P } else { s }) }
    fn mul(self, o: F) -> F { F(((self.0 as u128 * o.0 as u128) % P as u128) as u64) }
    const ZERO: F = F(0);
}

/// Deterministic field elements, uniform enough over `GF(p)` for the bound quoted above.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 3
    }
    fn f(&mut self) -> F { F(self.next() % P) }
    fn vec(&mut self, n: usize) -> Vec<F> { (0..n).map(|_| self.f()).collect() }
    fn mat(&mut self, r: usize, c: usize) -> Vec<Vec<F>> { (0..r).map(|_| self.vec(c)).collect() }
}

fn dot(a: &[F], b: &[F]) -> F { a.iter().zip(b).fold(F::ZERO, |s, (x, y)| s.add(x.mul(*y))) }
fn matvec(m: &[Vec<F>], v: &[F]) -> Vec<F> { m.iter().map(|row| dot(row, v)).collect() }
fn matvec_t(m: &[Vec<F>], v: &[F]) -> Vec<F> {
    let cols = m[0].len();
    (0..cols).map(|c| m.iter().zip(v).fold(F::ZERO, |s, (row, vi)| s.add(row[c].mul(*vi)))).collect()
}
fn axpy(y: &mut [F], a: F, x: &[F]) { for (yi, xi) in y.iter_mut().zip(x) { *yi = yi.add(a.mul(*xi)) } }

/// **The hyper-connection recursion equals its closed form, exactly, over GF(p).**
///
/// Recursion, from the spec: `H_i ← emb` for every stream; per sublayer `x = Σ_i p_i H_i`, then
/// `H_i ← H_i + q_i · out`. Closed form, from `hc.rs`: `H^(m)_i = emb + Σ_{l≤m} q^(l)_i out^(l)`
/// and `x^(m) = (Σ_i p^(m)_i) emb + Σ_{l<m} ⟨p^(m), q^(l)⟩ out^(l)`. Written separately, compared
/// with `==` after every sublayer, at eight random points.
#[test]
fn hyper_connection_recursion_equals_closed_form_exactly() {
    const HC: usize = 4;
    const D: usize = 8;
    const L: usize = 6;
    for trial in 0..8u64 {
        let mut rng = Rng(0x9e3779b97f4a7c15 ^ trial);
        let emb = rng.vec(D);
        let mut h: Vec<Vec<F>> = (0..HC).map(|_| emb.clone()).collect();
        let mut outs: Vec<Vec<F>> = Vec::new();
        let mut qs: Vec<Vec<F>> = Vec::new();

        for m in 0..L {
            let p = rng.vec(HC);
            let q = rng.vec(HC);

            // recursion, read half
            let mut x = vec![F::ZERO; D];
            for i in 0..HC { axpy(&mut x, p[i], &h[i]) }

            // closed form for x^(m)
            let psum = p.iter().fold(F::ZERO, |s, v| s.add(*v));
            let mut x_cf: Vec<F> = emb.iter().map(|e| psum.mul(*e)).collect();
            for (o, ql) in outs.iter().zip(&qs) { axpy(&mut x_cf, dot(&p, ql), o) }
            assert!(x == x_cf, "trial {trial} sublayer {m}: reduce is not the closed form");

            // an arbitrary sublayer output; the identity is indifferent to what produced it
            let out = rng.vec(D);

            // recursion, write half
            for i in 0..HC { axpy(&mut h[i], q[i], &out) }
            outs.push(out);
            qs.push(q);

            // closed form for H^(m)
            for i in 0..HC {
                let mut h_cf = emb.clone();
                for (o, ql) in outs.iter().zip(&qs) { axpy(&mut h_cf, ql[i], o) }
                assert!(h[i] == h_cf, "trial {trial} sublayer {m} stream {i}: state is not the closed form");
            }
        }
    }
}

/// **Both MLA absorption folds are exact identities over GF(p).**
///
/// `q_nope · (W_K c) = (W_Kᵀ q_nope) · c` for every head and position, and
/// `Σ_j P_j (W_V c_j) = W_V (Σ_j P_j c_j)` for ANY weights `P` -- the softmax is not needed for
/// the identity, so `P` is random and unnormalised on purpose.
#[test]
fn mla_absorption_folds_are_exact_identities() {
    const H: usize = 3;
    const NOPE: usize = 6;
    const VH: usize = 8;
    const R: usize = 10;
    const S: usize = 5;
    for trial in 0..8u64 {
        let mut rng = Rng(0xd1b54a32d192ed03 ^ trial);
        for _h in 0..H {
            let wk = rng.mat(NOPE, R);
            let wv = rng.mat(VH, R);
            let q_nope = rng.vec(NOPE);
            let cs: Vec<Vec<F>> = (0..S).map(|_| rng.vec(R)).collect();
            let pw = rng.vec(S);

            let q_abs = matvec_t(&wk, &q_nope);
            for (j, c) in cs.iter().enumerate() {
                let expanded = dot(&q_nope, &matvec(&wk, c));
                let absorbed = dot(&q_abs, c);
                assert!(expanded == absorbed, "trial {trial} pos {j}: key fold is not exact");
            }

            let mut expanded = vec![F::ZERO; VH];
            for (j, c) in cs.iter().enumerate() { axpy(&mut expanded, pw[j], &matvec(&wv, c)) }
            let mut o_lat = vec![F::ZERO; R];
            for (j, c) in cs.iter().enumerate() { axpy(&mut o_lat, pw[j], c) }
            let absorbed = matvec(&wv, &o_lat);
            assert!(expanded == absorbed, "trial {trial}: value fold is not exact");
        }
    }
}

// ───────────────────── the GPU against an exact oracle ─────────────────────

/// **One GPU hyper-connection step against an EXACT f64 oracle: the error is f32 rounding, and it
/// is measured in ulps rather than bounded by a number chosen to pass.**
///
/// Inputs are 11-bit dyadics `k/1024`, `|k| ≤ 1024`. The GPU's gates are read back as f32 (24-bit
/// mantissas, exponents near zero) and taken as GIVEN. Then `p_i · H_i` has at most 35 significant
/// bits, a sum of four at most 37, and `H_i + q_i · out` at most 36 -- all below f64's 53, so the
/// f64 reference is exact, not approximate. The residual is the GPU's f32 arithmetic in the reduce
/// and the write-back, and nothing else.
///
/// Error is measured against the operand scale Σ|terms| -- see the note at the measurement.
/// Bounds: 4 ulps for the reduce (four products and three adds), 2 for the write-back (one product,
/// one add). If either fails, the interesting question is what non-rounding error appeared.
#[test]
fn gpu_hyper_connection_step_error_is_f32_rounding_only() {
    use crate::hc::{Hc, HcConfig, HcGate};
    use ferric_core::Context;
    use ferric_tensor::Tensor;
    use std::sync::Arc;

    let Ok(ctx) = pollster::block_on(Context::new()) else { eprintln!("no GPU context — skipping"); return };
    let ctx = Arc::new(ctx);
    const HC: usize = 4;
    const D: usize = 8;
    const T: usize = 3;

    let mut seed = 0x2545f4914f6cdd1du64;
    let mut dyadic = |n: usize| -> Vec<f32> {
        (0..n).map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((seed >> 33) % 2049) as i64 - 1024) as f32 / 1024.0
        }).collect()
    };
    // Every value below is exactly representable in f64; the check is that the ARITHMETIC stays
    // exact, which the bit budgets in the doc comment guarantee for one step.
    let emb = dyadic(T * D);
    let fw: Vec<f32> = dyadic(2 * HC * HC * D).iter().map(|v| v * 0.15).collect();
    let bs = dyadic(2 * HC);
    let out = dyadic(T * D);

    let hcm = Hc::new(HcConfig { hc: HC, d: D, eps_hc: 1e-6, eps_rms: 1e-5, magnitude: 2.0 });
    let h0 = hcm.replicate(&Tensor::from_vec(&ctx, &emb, &[T, D]));
    let gate = HcGate {
        fn_w: Tensor::from_vec(&ctx, &fw, &[2 * HC, HC * D]),
        base: Tensor::from_vec(&ctx, &bs, &[2 * HC]),
        scale: [0.7, 1.3],
    };
    let (p_t, q_t) = hcm.gates(&h0, &gate);
    let p = pollster::block_on(p_t.to_vec());
    let q = pollster::block_on(q_t.to_vec());
    // The gates must be off the sigmoid rails or the products are trivially exact and the test
    // measures nothing about rounding.
    assert!(p.iter().chain(&q).any(|v| v.fract() != 0.0), "gates are all integers; rounding is not exercised");

    let x = pollster::block_on(hcm.reduce(&h0, &p_t).to_vec());
    let h1 = pollster::block_on(hcm.post(&Tensor::from_vec(&ctx, &out, &[T, D]), &h0, &q_t).to_vec());

    // ⛔ Error is measured against the OPERAND scale Σ|terms|, not against |exact|. A rounded
    // product feeding an add carries error proportional to the PRODUCT's magnitude; when the add
    // cancels -- `emb + q·out` with opposite signs -- the exact result shrinks and a half-ulp of the
    // operands becomes tens of ulps of the result. The first version of this test divided by
    // |exact| and reported 21.7 ulps on a GPU that had done nothing wrong. Forward error of a
    // rounded expression is bounded relative to the magnitudes that went INTO it.
    let ulps_of = |got: f32, exact: f64, scale: f64| -> f64 {
        let scale = if scale == 0.0 { 1.0 } else { scale };
        ((got as f64 - exact).abs() / scale) / f32::EPSILON as f64
    };
    let (mut worst_reduce, mut worst_post) = (0.0f64, 0.0f64);
    let mut worst_case = (0.0f64, 0.0f64, 0.0f64, 0.0f32); // (emb, q·out, exact, got) at the worst post
    for t in 0..T {
        for e in 0..D {
            // exact reduce: streams are all `emb` after replicate, so x = Σ_i p_i · emb
            let (mut x_exact, mut x_scale) = (0.0f64, 0.0f64);
            for i in 0..HC {
                let term = p[t * HC + i] as f64 * emb[t * D + e] as f64;
                x_exact += term; x_scale += term.abs();
            }
            worst_reduce = worst_reduce.max(ulps_of(x[t * D + e], x_exact, x_scale));
            for i in 0..HC {
                let (a, b) = (emb[t * D + e] as f64, q[t * HC + i] as f64 * out[t * D + e] as f64);
                let h_exact = a + b;
                let u = ulps_of(h1[(t * HC + i) * D + e], h_exact, a.abs() + b.abs());
                if u > worst_post { worst_post = u; worst_case = (a, b, h_exact, h1[(t * HC + i) * D + e]) }
            }
        }
    }
    eprintln!("GPU hyper-connection step vs exact f64 oracle (operand-scaled):");
    eprintln!("  reduce (4 mul + 3 add):  worst {worst_reduce:.3} ulps");
    eprintln!("  post   (1 mul + 1 add):  worst {worst_post:.3} ulps   at emb={:.6} q·out={:.6} exact={:.3e} got={:.3e}",
              worst_case.0, worst_case.1, worst_case.2, worst_case.3);
    let worst = worst_reduce.max(worst_post);
    assert!(worst_reduce <= 4.0, "reduce error {worst_reduce:.2} ulps exceeds 7 roundings' budget");
    assert!(worst_post <= 2.0, "write-back error {worst_post:.2} ulps exceeds 2 roundings' budget");
    assert!(worst > 0.0, "zero error is not plausible for f32 arithmetic with non-integer gates; the comparison is vacuous");
}
