//! **Hyper-connections** — the residual stream, widened and made learnable.
//!
//! Tencent's Hy4 (`hyv4`) replaces `x = x + f(norm(x))` with `hc = 4` parallel streams of width
//! `d`. Before every sublayer a gate vector `p ∈ R⁴` contracts the four streams down to the one
//! `d`-vector the sublayer actually sees; after it, a gate vector `q ∈ R⁴` broadcasts that one
//! output back into all four, each with its own gain. Both vectors come from a single projection of
//! the RMS-normalised concatenation of the streams, so the residual TOPOLOGY is chosen per token,
//! per layer, from the token's own state.
//!
//! ## Why four streams buy something
//!
//! Because the width connection is the identity — hyv4 has no learned 4×4 stream-mixing term and
//! ships no tensor for one — the recursion unrolls exactly:
//!
//! ```text
//!   H⁽ᵐ⁾ᵢ  =  emb  +  Σ_{l≤m}  q⁽ˡ⁾ᵢ · out⁽ˡ⁾
//!   x⁽ᵐ⁾   =  (Σᵢ p⁽ᵐ⁾ᵢ) · emb  +  Σ_{l<m}  ⟨p⁽ᵐ⁾, q⁽ˡ⁾⟩ · out⁽ˡ⁾
//! ```
//!
//! So the model is a **DenseNet over sublayer outputs** whose L×L coefficient matrix is factorised
//! as a 4-dimensional inner product. That is the whole trade: dense depth-wise connectivity for
//! `4·d` of state instead of `L·d`. It is also the strongest available test — [`Hc::pre`] and
//! [`Hc::post`] have a closed form, and `closed_form_matches_the_recursion` checks it rather than
//! checking that the code agrees with itself.
//!
//! ## What a port gets wrong without erroring
//!
//! Every one of these keeps all the shapes:
//!
//! * ⚠ **`mag·σ(z) + ε`, not `mag·(σ(z) + ε)`.** With `mag = 2` they differ by `1e-6`, invisible
//!   at `q ≈ 1` and a clean factor of two at the floor — which is where the test looks.
//! * ⚠ **`res + q·out`, not `(res + out)·q`.**
//! * ⚠ **`p` has no magnitude and `q` does.** `p ∈ (ε, 1+ε)`, `q ∈ (ε, mag+ε)`.
//! * ⚠ **One joint RMS norm over all `hc·d` values**, not four norms of `d`. And its epsilon is the
//!   model's `attention.layer_norm_rms_epsilon` (1e-5), NOT `hyper_connection.epsilon` (1e-6),
//!   which is only ever the additive floor on the gates.
//! * ⚠ **The gates weight the RAW streams.** The normalised flattening exists only to produce the
//!   eight coefficients; it never reaches the reduce.
//! * ⚠ **The eight coefficients split as a contiguous block** `[0..hc]` for `p` and `[hc..2hc]` for
//!   `q` — never interleaved.
//! * ⚠ **The flattening is stream-major**, `[stream0 | stream1 | …]`. Element-major interleaving
//!   permutes the projection's input axis without changing its length.
//! * ⚠ **`q` is computed BEFORE the sublayer**, from the same projection as `p`. Recomputing it
//!   from the post-sublayer state is a different model.

use ferric_tensor::Tensor;

/// Model-level hyper-connection constants, all read from GGUF KV.
#[derive(Debug, Clone, Copy)]
pub struct HcConfig {
    /// `hyper_connection.count` — the number of residual streams.
    pub hc: usize,
    /// `embedding_length` — the width of one stream, and of everything the sublayers see.
    pub d: usize,
    /// `hyper_connection.epsilon` — added to both gates AFTER the sigmoid and after the magnitude.
    pub eps_hc: f32,
    /// `attention.layer_norm_rms_epsilon` — used by the joint norm. Not `eps_hc`.
    pub eps_rms: f32,
    /// `hyper_connection.magnitude` — scales the write gate only.
    pub magnitude: f32,
}

/// One sublayer's gate weights: `hc_attn_*` or `hc_ffn_*`.
pub struct HcGate {
    /// `hc_*_fn.weight` — GGUF `ne = [hc·d, 2·hc]`, so `[2·hc, hc·d]` here.
    pub fn_w: Tensor,
    /// `hc_*_base.weight` — `[2·hc]`, the pre-sigmoid bias. First `hc` bias `p`, last `hc` bias `q`.
    pub base: Tensor,
    /// `hc_*_scale.weight` — `[2]`, read to the host. `[0]` scales `p`'s logits, `[1]` scales `q`'s.
    pub scale: [f32; 2],
}

/// The final collapse, `output_hc_*` — structurally [`Hc::pre`]'s read half with no write half:
/// `hc` rows instead of `2·hc`, one scalar, no magnitude.
pub struct HcHead {
    /// `output_hc_fn.weight` — GGUF `ne = [hc·d, hc]`, so `[hc, hc·d]` here.
    pub fn_w: Tensor,
    /// `output_hc_base.weight` — `[hc]`.
    pub base: Tensor,
    /// `output_hc_scale.weight` — `[1]`.
    pub scale: f32,
}

pub struct Hc {
    pub cfg: HcConfig,
}

impl Hc {
    pub fn new(cfg: HcConfig) -> Self { Self { cfg } }

    /// Per-token residual state in floats, against a single-stream transformer of the same width.
    /// Exactly `hc`× — there is no compression anywhere in the scheme.
    pub fn residual_floats_per_token(&self) -> usize { self.cfg.hc * self.cfg.d }

    /// Enter the stack: replicate the embedding into every stream, bit-identically.
    ///
    /// Not scaled by `1/hc`, not by `1/sqrt(hc)`, not zero-padded, and there is no per-stream learned
    /// initialiser. `hyv4` also writes no embedding-scale KV, so there is no `sqrt(d)` here either.
    pub fn replicate(&self, emb: &Tensor) -> Tensor {
        let (t, d, hc) = (emb.shape[0], self.cfg.d, self.cfg.hc);
        emb.reshape(&[t, 1, d]).broadcast_to(&[t, hc, d]).contiguous()
    }

    /// The eight (or four) gate logits: one joint ungained RMS norm of the stream-major flattening,
    /// then one projection. `h` is `[T, hc, d]`; the result is `[T, rows]`.
    fn logits(&self, h: &Tensor, fn_w: &Tensor) -> Tensor {
        let (t, hc, d) = (h.shape[0], self.cfg.hc, self.cfg.d);
        h.reshape(&[t, hc * d]).rmsnorm_weightless(self.cfg.eps_rms).matmul_bt(fn_w)
    }

    /// `σ(m·scale + bias) + ε`, optionally scaled by the magnitude BEFORE the epsilon.
    fn gate(&self, m: &Tensor, lo: usize, scale: f32, base: &Tensor, magnitude: Option<f32>) -> Tensor {
        let (t, hc) = (m.shape[0], self.cfg.hc);
        let bias = base.reshape(&[1, base.shape[0]]).broadcast_to(&[t, base.shape[0]]);
        let z = m
            .narrow(1, lo, hc)
            .contiguous()
            .mul(&m.scalar(scale))
            .add(&bias.narrow(1, lo, hc).contiguous());
        // The magnitude multiplies the sigmoid and the epsilon is added after. Reversing them
        // changes the floor by a factor of `magnitude`, which is what the test measures.
        let g = match magnitude { Some(mag) => z.sigmoid().mul(&z.scalar(mag)), None => z.sigmoid() };
        g.add(&z.scalar(self.cfg.eps_hc))
    }

    /// **Read half.** Returns `(x, q)`: the single `[T, d]` vector the sublayer consumes, and the
    /// write gate `[T, hc]` that [`Hc::post`] will need afterwards.
    ///
    /// `q` is produced here, from the same projection and the same normalised state as `p`, because
    /// that is when the information it depends on exists — not after the sublayer has run.
    pub fn pre(&self, h: &Tensor, g: &HcGate) -> (Tensor, Tensor) {
        let (p, q) = self.gates(h, g);
        (self.reduce(h, &p), q)
    }

    /// Both gate vectors, `(p, q)`, each `[T, hc]`. Exposed because the read gate is the model's
    /// own statement about which streams matter for this token, and because a test that can only
    /// see `x` cannot separate a wrong `p` from a wrong reduce.
    pub fn gates(&self, h: &Tensor, g: &HcGate) -> (Tensor, Tensor) {
        let m = self.logits(h, &g.fn_w);
        let p = self.gate(&m, 0, g.scale[0], &g.base, None);
        let q = self.gate(&m, self.cfg.hc, g.scale[1], &g.base, Some(self.cfg.magnitude));
        (p, q)
    }

    /// `x = Σᵢ pᵢ · Hᵢ` — a plain weighted sum of the RAW streams. No `1/hc`, and no
    /// renormalisation by `Σp`, which is free to land anywhere in `(0, hc)`.
    pub fn reduce(&self, h: &Tensor, p: &Tensor) -> Tensor {
        let (t, hc, d) = (h.shape[0], self.cfg.hc, self.cfg.d);
        h.mul(&p.reshape(&[t, hc, 1]).broadcast_to(&[t, hc, d])).sum(&[1], false)
    }

    /// **Write half.** One `[T, d]` sublayer output into all `hc` streams, each with its own gain.
    ///
    /// `res` must be the streams as they stood when [`Hc::pre`] read them. Every stream receives a
    /// strictly positive multiple of every output, because `q > ε > 0` always — no stream can skip a
    /// sublayer, which is what makes the closed form a plain sum rather than a gated one.
    pub fn post(&self, out: &Tensor, res: &Tensor, q: &Tensor) -> Tensor {
        let (t, hc, d) = (res.shape[0], self.cfg.hc, self.cfg.d);
        let gain = q.reshape(&[t, hc, 1]).broadcast_to(&[t, hc, d]);
        res.add(&gain.mul(&out.reshape(&[t, 1, d]).broadcast_to(&[t, hc, d])))
    }

    /// **Leave the stack.** A learned, input-conditioned weighted sum down to one `[T, d]` vector.
    ///
    /// Not a sum, not a mean, and not "take stream 0". The model's `output_norm` is applied to the
    /// result of this, after the collapse — never to the streams.
    pub fn collapse(&self, h: &Tensor, head: &HcHead) -> Tensor {
        let (t, hc, d) = (h.shape[0], self.cfg.hc, self.cfg.d);
        let m = self.logits(h, &head.fn_w);
        let bias = head.base.reshape(&[1, hc]).broadcast_to(&[t, hc]);
        let p = m.mul(&m.scalar(head.scale)).add(&bias).sigmoid().add(&m.scalar(self.cfg.eps_hc));
        h.mul(&p.reshape(&[t, hc, 1]).broadcast_to(&[t, hc, d])).sum(&[1], false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::Context;
    use std::sync::Arc;

    const HC: usize = 4;
    const D: usize = 8;
    const T: usize = 3;

    fn cfg() -> HcConfig {
        HcConfig { hc: HC, d: D, eps_hc: 1e-6, eps_rms: 1e-5, magnitude: 2.0 }
    }

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // ⛔ `>> 33` here gave [-1, 0) -- every input negative -- until 2026-09-04. See dsa.rs::rnd.
        ((*s >> 32) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn rnd(ctx: &Arc<Context>, shape: &[usize], seed: u64) -> (Tensor, Vec<f32>) {
        let n: usize = shape.iter().product();
        let mut s = seed;
        let v: Vec<f32> = (0..n).map(|_| lcg(&mut s)).collect();
        (Tensor::from_vec(ctx, &v, shape), v)
    }

    /// Random projection weights scaled down so the gate logits stay off the sigmoid's rails.
    fn g_scaled(ctx: &Arc<Context>, seed: u64, k: f32) -> Tensor {
        let mut s = seed;
        let v: Vec<f32> = (0..2 * HC * HC * D).map(|_| lcg(&mut s) * k).collect();
        Tensor::from_vec(ctx, &v, &[2 * HC, HC * D])
    }

    fn gate(ctx: &Arc<Context>, seed: u64) -> HcGate {
        HcGate {
            fn_w: rnd(ctx, &[2 * HC, HC * D], seed).0,
            base: rnd(ctx, &[2 * HC], seed ^ 0x5bf0).0,
            scale: [0.7, 1.3],
        }
    }

    fn get(t: &Tensor) -> Vec<f32> { pollster::block_on(t.to_vec()) }

    macro_rules! ctx_or_skip {
        () => {
            match pollster::block_on(Context::new()) {
                Ok(c) => Arc::new(c),
                Err(_) => { eprintln!("no GPU context — skipping"); return }
            }
        };
    }

    /// ⛔ The generator is two-signed and spans its range. Until 2026-09-04 it was uniform in
    /// [-1, 0) -- a `>> 33` where `>> 32` was meant -- and every test in this module ran on
    /// negative-only inputs without anything noticing. A fixture needs a guard like any other claim.
    #[test]
    fn the_fixture_generator_is_two_signed() {
        let ctx = ctx_or_skip!();
        let v = { let mut s = 777u64; (0..1024).map(|_| lcg(&mut s)).collect::<Vec<f32>>() };
        let (mx, mn) = v.iter().fold((f32::MIN, f32::MAX), |(a, b), x| (a.max(*x), b.min(*x)));
        assert!(mx > 0.5 && mn < -0.5, "generator does not span both signs: max {mx}, min {mn}");
        assert!(v.iter().filter(|x| **x > 0.0).count() * 4 > v.len(), "fewer than a quarter of the draws are positive");
    }

    /// **The whole scheme has a closed form, and this is it.**
    ///
    /// Because the width connection is the identity, `hc_pre`/`hc_post` composed over any number of
    /// sublayers must satisfy `H⁽ᵐ⁾ᵢ = emb + Σ_{l≤m} q⁽ˡ⁾ᵢ·out⁽ˡ⁾` and
    /// `x⁽ᵐ⁾ = (Σᵢ p⁽ᵐ⁾ᵢ)·emb + Σ_{l<m} ⟨p⁽ᵐ⁾, q⁽ˡ⁾⟩·out⁽ˡ⁾`.
    ///
    /// This is a real oracle rather than a self-consistency check: the right-hand sides are computed
    /// on the host from `p` and `q` alone and never touch [`Hc::reduce`] or [`Hc::post`]. A swapped
    /// `(res + out)·q`, a `1/hc` in the reduce, a renormalised `p`, or a `q` recomputed after the
    /// sublayer all break it while leaving every shape intact. The identity itself is proved exactly
    /// over GF(2⁶¹−1) in `exact.rs`; this test is about the GPU's f32 arithmetic against it.
    ///
    /// ## The tolerance is DERIVED, not chosen
    ///
    /// This used to accept `2e-4 · max(|want|, 1)` — about 1700 ulps of room, picked to pass. The
    /// bound now comes from counting roundings against each element's OPERAND SCALE, the quantity
    /// forward error is actually proportional to (dividing by `|want|` instead reports tens of ulps
    /// on a correct GPU the moment two terms cancel — see `exact.rs`):
    ///
    /// * state after `m` write-backs: each is `fl(H + fl(q·out))`, two roundings, each ≤ ½ ulp of
    ///   a magnitude bounded by `S_H = |emb| + Σ_l |q_l·out_l|`, so `|err_H| ≤ m·ε·S_H`;
    /// * the reduce: four products and three adds (≤ 3.5 ulps of `Σᵢ|pᵢHᵢ|`) on top of the
    ///   propagated state error, so `|err_x| ≤ (m + 4)·ε·S_x` with `S_x = Σᵢ pᵢ·S_H,ᵢ`.
    ///
    /// The host reference is computed in f64, whose own rounding is 2²⁹ times below f32's and is
    /// ignored. The GPU's gates are read back and used as given, so no gate error enters. A safety
    /// factor of 2 is applied, and the worst observed/bound ratio is printed; a FLOOR on that ratio
    /// fails the test if the bound ever becomes loose enough to do no work — the old 2e-4 would have
    /// failed it by a factor of ten.
    #[test]
    fn closed_form_matches_the_recursion() {
        let ctx = ctx_or_skip!();
        let hcm = Hc::new(cfg());
        let (emb_t, emb) = rnd(&ctx, &[T, D], 11);
        let mut h = hcm.replicate(&emb_t);

        // Every stream starts as a bit-identical copy of the embedding.
        let h0 = get(&h);
        for t in 0..T { for i in 0..HC { for e in 0..D {
            assert_eq!(h0[(t * HC + i) * D + e], emb[t * D + e], "stream {i} is not a copy of emb");
        }}}

        const EPS: f64 = f32::EPSILON as f64;
        const SAFETY: f64 = 2.0;
        let mut outs: Vec<Vec<f32>> = Vec::new();   // out⁽ˡ⁾ per sublayer
        let mut qs: Vec<Vec<f32>> = Vec::new();     // q⁽ˡ⁾ per sublayer, [T, hc]
        let (mut worst_x, mut worst_h) = (0.0f64, 0.0f64); // observed / bound

        for l in 0..6 {
            let g = gate(&ctx, 100 + l as u64);
            let (p, q) = hcm.gates(&h, &g);
            let (pv, qv) = (get(&p), get(&q));
            let x = get(&hcm.reduce(&h, &p));
            let m = outs.len(); // write-backs applied so far

            // x⁽ᵐ⁾ = (Σᵢ pᵢ)·emb + Σ_{l<m} ⟨p, q⁽ˡ⁾⟩·out⁽ˡ⁾, in f64, with per-element operand scale
            for t in 0..T {
                let psum: f64 = (0..HC).map(|i| pv[t * HC + i] as f64).sum();
                for e in 0..D {
                    let mut want = psum * emb[t * D + e] as f64;
                    let mut s_x = 0.0f64;                        // Σᵢ pᵢ·S_H,ᵢ
                    for i in 0..HC {
                        let mut s_h = (emb[t * D + e] as f64).abs();
                        for (o, qq) in outs.iter().zip(&qs) { s_h += (qq[t * HC + i] as f64 * o[t * D + e] as f64).abs() }
                        s_x += pv[t * HC + i] as f64 * s_h;
                    }
                    for (o, qq) in outs.iter().zip(&qs) {
                        let dot: f64 = (0..HC).map(|i| pv[t * HC + i] as f64 * qq[t * HC + i] as f64).sum();
                        want += dot * o[t * D + e] as f64;
                    }
                    let got = x[t * D + e] as f64;
                    let bound = SAFETY * (m as f64 + 4.0) * EPS * s_x;
                    let err = (got - want).abs();
                    assert!(err <= bound,
                            "sublayer {l} token {t} elem {e}: reduce gave {got}, closed form {want}, \
                             error {err:.3e} exceeds derived bound {bound:.3e} ({:.1} ulps of the operand scale)",
                            err / (EPS * s_x));
                    if bound > 0.0 { worst_x = worst_x.max(err / bound) }
                }
            }

            // Run an arbitrary "sublayer" — the scheme is indifferent to what produced `out`.
            let (out_t, out) = rnd(&ctx, &[T, D], 900 + l as u64);
            h = hcm.post(&out_t, &h, &q);
            outs.push(out);
            qs.push(qv);
            let m = outs.len();

            // H⁽ᵐ⁾ᵢ = emb + Σ_{l≤m} q⁽ˡ⁾ᵢ·out⁽ˡ⁾
            let hv = get(&h);
            for t in 0..T { for i in 0..HC { for e in 0..D {
                let mut want = emb[t * D + e] as f64;
                let mut s_h = (emb[t * D + e] as f64).abs();
                for (o, qq) in outs.iter().zip(&qs) {
                    let term = qq[t * HC + i] as f64 * o[t * D + e] as f64;
                    want += term; s_h += term.abs();
                }
                let got = hv[(t * HC + i) * D + e] as f64;
                let bound = SAFETY * (m as f64) * EPS * s_h;
                let err = (got - want).abs();
                assert!(err <= bound,
                        "after sublayer {l}, stream {i} token {t} elem {e}: {got} vs {want}, \
                         error {err:.3e} exceeds derived bound {bound:.3e} ({:.1} ulps of the operand scale)",
                        err / (EPS * s_h));
                if bound > 0.0 { worst_h = worst_h.max(err / bound) }
            }}}
        }
        eprintln!("hc closed form vs GPU over 6 sublayers: worst observed/bound  reduce {worst_x:.3}  state {worst_h:.3}");
        // Guard the guard. If the bound were loose enough that the observed error never came within
        // 1% of it, the assertion above would be decorative. The old 2e-4 lands near 6e-4 here.
        assert!(worst_x > 0.01 && worst_h > 0.01,
                "derived bound is >100x looser than observed ({worst_x:.4}, {worst_h:.4}); it is not doing work");
        assert!(worst_x > 0.0 && worst_h > 0.0, "zero error is not plausible for f32; the comparison is vacuous");
    }

    /// After exactly ONE sublayer every stream has moved by a scalar multiple of the same vector,
    /// so the three deltas from stream 0 are pairwise colinear. Any per-stream mixing — a 4×4 width
    /// connection, a stream-dependent output, a transposed `q` — destroys this immediately.
    #[test]
    fn one_sublayer_leaves_the_stream_deltas_colinear() {
        let ctx = ctx_or_skip!();
        let hcm = Hc::new(cfg());
        let (emb_t, _) = rnd(&ctx, &[T, D], 21);
        let h = hcm.replicate(&emb_t);
        // ⚠ The gate weights must be gentle enough to stay inside the sigmoid's active range. With
        // full-scale random weights over 32 inputs the logits reach ±5, every q saturates at
        // magnitude+eps, the four gains become equal and the streams never diverge — which trips the
        // guard below rather than passing vacuously.
        let soft = HcGate { fn_w: g_scaled(&ctx, 7, 0.15), base: rnd(&ctx, &[2 * HC], 8).0, scale: [0.7, 1.3] };
        let (_, q) = hcm.gates(&h, &soft);
        let qv = get(&q);
        let (out_t, _) = rnd(&ctx, &[T, D], 22);
        let hv = get(&hcm.post(&out_t, &h, &q));

        for t in 0..T {
            assert!((qv[t * HC + 1] - qv[t * HC]).abs() > 1e-3,
                    "token {t}: the write gains are equal, so this test cannot see stream mixing");
            let base: Vec<f32> = (0..D).map(|e| hv[(t * HC + 1) * D + e] - hv[(t * HC) * D + e]).collect();
            let bn: f32 = base.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(bn > 1e-4, "streams must actually diverge, or the test proves nothing");
            for i in 2..HC {
                let dv: Vec<f32> = (0..D).map(|e| hv[(t * HC + i) * D + e] - hv[(t * HC) * D + e]).collect();
                let dn: f32 = dv.iter().map(|v| v * v).sum::<f32>().sqrt();
                let dot: f32 = base.iter().zip(&dv).map(|(a, b)| a * b).sum();
                let cos = dot / (bn * dn);
                assert!(cos.abs() > 1.0 - 1e-3, "stream {i} delta is not colinear with stream 1's (cos {cos})");
            }
        }
    }

    /// ⚠ `mag·σ(z) + ε` versus `mag·(σ(z) + ε)`. At `q ≈ 1` they differ by 1e-6 and no test would
    /// see it; at the floor, where `σ(z) → 0`, they differ by exactly the magnitude. So drive the
    /// logits hard negative and read the floor.
    #[test]
    fn the_write_gate_floor_is_eps_not_magnitude_times_eps() {
        let ctx = ctx_or_skip!();
        let c = cfg();
        let hcm = Hc::new(c);
        let (emb_t, _) = rnd(&ctx, &[T, D], 31);
        let h = hcm.replicate(&emb_t);
        // A zero projection with a large negative bias sends both sigmoids to 0 for every token.
        let g = HcGate {
            fn_w: Tensor::from_vec(&ctx, &vec![0.0; 2 * HC * HC * D], &[2 * HC, HC * D]),
            base: Tensor::from_vec(&ctx, &vec![-40.0; 2 * HC], &[2 * HC]),
            scale: [1.0, 1.0],
        };
        let (p, q) = hcm.gates(&h, &g);
        for v in get(&q) {
            assert!((v - c.eps_hc).abs() < 1e-9,
                    "write-gate floor is {v}; eps_hc is {} and magnitude*eps_hc would be {}",
                    c.eps_hc, c.magnitude * c.eps_hc);
        }
        for v in get(&p) {
            assert!((v - c.eps_hc).abs() < 1e-9, "read-gate floor is {v}, want eps_hc {}", c.eps_hc);
        }
    }

    /// The magnitude belongs to the write gate alone: `p ∈ (ε, 1+ε)` while `q ∈ (ε, mag+ε)`.
    /// Applying it to both, or to neither, keeps every shape.
    #[test]
    fn only_the_write_gate_carries_the_magnitude() {
        let ctx = ctx_or_skip!();
        let c = cfg();
        let hcm = Hc::new(c);
        let (emb_t, _) = rnd(&ctx, &[T, D], 41);
        let h = hcm.replicate(&emb_t);
        let g = HcGate {
            fn_w: Tensor::from_vec(&ctx, &vec![0.0; 2 * HC * HC * D], &[2 * HC, HC * D]),
            base: Tensor::from_vec(&ctx, &vec![40.0; 2 * HC], &[2 * HC]),
            scale: [1.0, 1.0],
        };
        let (p, q) = hcm.gates(&h, &g);
        for v in get(&p) { assert!((v - (1.0 + c.eps_hc)).abs() < 1e-5, "p saturates at {v}, want 1+eps") }
        for v in get(&q) {
            assert!((v - (c.magnitude + c.eps_hc)).abs() < 1e-5,
                    "q saturates at {v}, want magnitude+eps = {}", c.magnitude + c.eps_hc);
        }
    }

    /// **One joint norm over `hc·d`, not `hc` norms of `d`.** Rescaling a single stream is invisible
    /// to a per-stream norm — it divides the scaling straight back out — and must NOT be invisible
    /// here. This also pins the flattening order: a host-side stream-major flatten reproduces the
    /// gates exactly, which an element-major interleave would not.
    #[test]
    fn the_norm_is_joint_over_all_streams() {
        let ctx = ctx_or_skip!();
        let c = cfg();
        let hcm = Hc::new(c);
        // ⚠ The projection is scaled down on purpose. With full-scale weights over hc*d inputs
        // every logit reaches ±5, every sigmoid pins to 0 or 1, and the comparison below stops
        // being able to see WHICH logit fed WHICH gate — it passed a deliberate mutation that read
        // p and q from the same half of the split. A saturated gate is a constant, and a test of a
        // constant tests nothing.
        let g = HcGate { fn_w: g_scaled(&ctx, 55, 0.15), base: rnd(&ctx, &[2 * HC], 56).0, scale: [0.7, 1.3] };
        let (fw, bs) = (get(&g.fn_w), get(&g.base));

        let (h_t, hv) = rnd(&ctx, &[T, HC, D], 61);
        let (p, q) = hcm.gates(&h_t, &g);
        let (pv, qv) = (get(&p), get(&q));

        // Host reference: stream-major flatten, ONE ungained RMS norm over all hc*d, one projection.
        for t in 0..T {
            let flat: Vec<f32> = (0..HC * D).map(|k| hv[t * HC * D + k]).collect();
            let ms: f32 = flat.iter().map(|v| v * v).sum::<f32>() / (HC * D) as f32;
            let inv = 1.0 / (ms + c.eps_rms).sqrt();
            for j in 0..2 * HC {
                let m: f32 = (0..HC * D).map(|k| fw[j * HC * D + k] * flat[k] * inv).sum();
                let (lo, sc, mag) = if j < HC { (0, g.scale[0], 1.0) } else { (HC, g.scale[1], c.magnitude) };
                let z = m * sc + bs[j];
                let want = mag / (1.0 + (-z).exp()) + c.eps_hc;
                let got = if j < HC { pv[t * HC + j] } else { qv[t * HC + (j - HC)] };
                let _ = lo;
                assert!((got - want).abs() < 2e-5, "token {t} gate {j}: {got} vs host {want}");
                // Guard the guard: a saturated sigmoid makes the line above a tautology.
                let unit = want / mag;
                assert!(unit > 0.02 && unit < 0.98,
                        "token {t} gate {j} is saturated at {unit}; this comparison cannot see a swap");
            }
        }

        // Discriminator: scale stream 0 by 10. A per-stream norm cancels it exactly; a joint one
        // cannot, because the other three streams did not change.
        let mut scaled = hv.clone();
        for t in 0..T { for e in 0..D { scaled[t * HC * D + e] *= 10.0 } }
        let h2 = Tensor::from_vec(&ctx, &scaled, &[T, HC, D]);
        let (p2, _) = hcm.gates(&h2, &g);
        let moved = get(&p2).iter().zip(&pv).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(moved > 1e-3, "rescaling one stream moved the gates by {moved}; the norm is per-stream");
    }

    /// The collapse is a learned weighted sum, not a mean and not "take stream 0".
    #[test]
    fn the_collapse_is_learned_not_a_mean() {
        let ctx = ctx_or_skip!();
        let hcm = Hc::new(cfg());
        let (h_t, hv) = rnd(&ctx, &[T, HC, D], 71);
        let head = HcHead { fn_w: rnd(&ctx, &[HC, HC * D], 72).0, base: rnd(&ctx, &[HC], 73).0, scale: 0.9 };
        let y = get(&hcm.collapse(&h_t, &head));
        let mean: Vec<f32> = (0..T * D).map(|k| {
            let (t, e) = (k / D, k % D);
            (0..HC).map(|i| hv[(t * HC + i) * D + e]).sum::<f32>() / HC as f32
        }).collect();
        let s0: Vec<f32> = (0..T * D).map(|k| hv[(k / D * HC) * D + k % D]).collect();
        let far = |a: &[f32]| y.iter().zip(a).map(|(x, z)| (x - z).abs()).fold(0.0f32, f32::max);
        assert!(far(&mean) > 1e-3, "collapse coincides with a plain mean");
        assert!(far(&s0) > 1e-3, "collapse coincides with taking stream 0");
        assert_eq!(y.len(), T * D);
    }

    /// The trade, stated as a number: `hc`× the residual state, exactly, at every block. There is no
    /// compression anywhere in the scheme, and the KV cache is untouched because attention only ever
    /// sees the collapsed `d`-vector.
    #[test]
    fn residual_state_is_exactly_hc_times_a_single_stream() {
        let hcm = Hc::new(HcConfig { hc: 4, d: 6144, eps_hc: 1e-6, eps_rms: 1e-5, magnitude: 2.0 });
        assert_eq!(hcm.residual_floats_per_token(), 24576);
        assert_eq!(hcm.residual_floats_per_token() / 6144, 4);
    }
}
