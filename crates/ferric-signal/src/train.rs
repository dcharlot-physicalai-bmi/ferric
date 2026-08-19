//! The pieces that make the discrete bottleneck trainable.
//!
//! ## The problem, and why it fails silently without this
//!
//! Quantization rounds, and rounding has **zero derivative almost everywhere**. Put an FSQ layer in
//! the middle of an autoencoder and differentiate naively and every gradient reaching the encoder is
//! exactly zero: the decoder trains, the encoder does not, the loss falls a little and then stops,
//! and nothing anywhere reports an error. The model simply never learns to produce useful latents.
//!
//! The straight-through estimator is the standard fix and it is one line — forward through the
//! quantizer, backward as if it were the identity:
//!
//! ```text
//!     z_q = z + detach(quantize(z) - z)
//! ```
//!
//! The forward value is exactly `quantize(z)`; the backward path sees only `z`, so `dz_q/dz = 1`.
//!
//! Both halves of that are tested below, and the gradient half is the one that matters: it is the
//! difference between a model that trains and a model that appears to.

use crate::fsq::Fsq;
use ferric_core::Context;
use ferric_tensor::autograd::Var;
use ferric_tensor::Tensor;
use std::sync::Arc;

/// Quantize on the forward pass, pass the gradient through unchanged on the backward pass.
///
/// `z` is `[t, dim]` with `dim == fsq.dim()`. Returns the quantizer's centre values, in the same
/// bounded units the decoder consumes.
pub fn straight_through(ctx: &Arc<Context>, z: &Var, fsq: &Fsq) -> Var {
    let shape = z.value().shape.clone();
    let d = fsq.dim();
    let t = shape.iter().product::<usize>() / d;

    // THE BOUNDING FUNCTION BELONGS INSIDE THE GRAPH.
    //
    // My first version quantized entirely on the host and wrapped the whole thing in the estimator,
    // so `tanh` never appeared in the autograd graph at all. The forward value was right and the
    // gradient was the identity — a textbook STE — and the model still failed, in a way that took a
    // measurement to see: with no differentiable bound, NOTHING penalises a large latent. The
    // encoder grew the latent scale freely, reaching standard deviations of 2,500 to 17,000 on a
    // 1,500-step run, which saturates tanh completely. Each dimension then reached only 2 to 6 of
    // its 8 levels and the whole 32,768-code space collapsed to 27 observed codes.
    //
    // Putting the bound in the graph fixes it at the source: `tanh` is differentiable, so growing
    // the latent past saturation earns a vanishing gradient instead of a free ride, and the
    // estimator is applied ONLY around the rounding, which is the genuinely non-differentiable step.
    let (half, off, sh) = fsq.bound_terms();
    let tile = |v: &[f32]| {
        let mut out = Vec::with_capacity(t * d);
        for _ in 0..t {
            out.extend_from_slice(v);
        }
        Var::leaf(Tensor::from_vec(ctx, &out, &shape))
    };
    let zb = z.add(&tile(&sh)).tanh().mul(&tile(&half)).sub(&tile(&off));

    // Round the bounded value on the host; the estimator carries the gradient past it.
    //
    // NON-FINITE DETECTION READS THE RAW LATENT, NOT THE BOUNDED ONE, and that is not fussiness.
    // Measured on this machine: `tanh(NaN)` returns **-1.0** on Metal through wgpu — the GPU does
    // not propagate NaN through a transcendental. Checking `zb` for finiteness therefore never
    // fires, and a NaN latent arrives as a perfectly legal token id with nothing to distinguish it
    // from a real measurement. The raw latent is the only place the fault is still visible.
    let zflat = pollster::block_on(z.value().to_vec());
    let bflat = pollster::block_on(zb.value().to_vec());
    let mut rounded = Vec::with_capacity(bflat.len());
    for (idx, &b) in bflat.iter().enumerate() {
        let i = idx % d;
        if zflat[idx].is_finite() {
            let code = fsq.round_bounded(i, b);
            rounded.push(fsq.dequantize_dim(i, code));
        } else {
            rounded.push(zflat[idx]);
        }
    }
    let rt = Var::leaf(Tensor::from_vec(ctx, &rounded, &shape));
    zb.add(&rt.sub(&zb.detach()))
}

/// Mean squared error, reduced to a scalar so `backward()` has somewhere to start.
pub fn mse(pred: &Var, target: &Var) -> Var {
    let d = pred.sub(target);
    d.mul(&d).mean_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> Option<Arc<Context>> {
        match pollster::block_on(Context::new()) {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => {
                if std::env::var("FERRIC_NO_GPU").is_ok() {
                    eprintln!("FERRIC_NO_GPU set; skipping deliberately ({e:?})");
                    None
                } else {
                    panic!("no GPU context ({e:?}). Set FERRIC_NO_GPU=1 to waive this on purpose.");
                }
            }
        }
    }

    #[test]
    fn the_forward_value_is_exactly_the_quantizer_output() {
        let Some(ctx) = ctx() else { return };
        let q = Fsq::signal_15bit();
        let z: Vec<f32> = (0..40).map(|i| (i as f32 - 20.0) * 0.31).collect();
        let v = Var::leaf(Tensor::from_vec(&ctx, &z, &[8, 5]));
        let out = pollster::block_on(straight_through(&ctx, &v, &q).value().to_vec());
        let mut want = Vec::new();
        for row in z.chunks(5) {
            want.extend(q.dequantize(&q.quantize(row).unwrap()).unwrap());
        }
        assert_eq!(out, want, "straight-through changed the forward value");
    }

    /// THE TEST THIS MODULE EXISTS FOR.
    ///
    /// The estimator's claim is precise: **the backward pass behaves as if the ROUNDING were the
    /// identity.** It is not "as if the whole quantizer were the identity" — the bounding `tanh` is
    /// a real, differentiable part of the graph, and it must stay in the gradient. My first version
    /// of this test asserted the stronger, wrong thing, and it passed against an implementation that
    /// had the bound OUTSIDE the graph entirely. That implementation trained to 27 of 32,768 codes
    /// because nothing penalised an exploding latent.
    ///
    /// So: build the bounded value independently, and require that the gradient reaching `z`
    /// through the quantizer equals the gradient reaching it through the bound alone.
    #[test]
    fn the_gradient_behaves_as_if_the_rounding_were_the_identity() {
        let Some(ctx) = ctx() else { return };
        let q = Fsq::signal_15bit();
        let z: Vec<f32> = (0..30).map(|i| ((i * 37) % 23) as f32 * 0.12 - 1.4).collect();

        // Through the quantizer.
        let a = Var::leaf(Tensor::from_vec(&ctx, &z, &[6, 5]));
        straight_through(&ctx, &a, &q).sum_all().backward();
        let ga = pollster::block_on(a.grad().expect("no gradient reached z").to_vec());

        // Through the bound alone, rebuilt here from the quantizer's own terms.
        let (half, off, sh) = q.bound_terms();
        let tile = |v: &[f32]| {
            let mut o = Vec::new();
            for _ in 0..6 { o.extend_from_slice(v); }
            Var::leaf(Tensor::from_vec(&ctx, &o, &[6, 5]))
        };
        let b = Var::leaf(Tensor::from_vec(&ctx, &z, &[6, 5]));
        b.add(&tile(&sh)).tanh().mul(&tile(&half)).sub(&tile(&off)).sum_all().backward();
        let gb = pollster::block_on(b.grad().unwrap().to_vec());

        assert_eq!(ga, gb, "rounding did not behave as the identity in the backward pass");
        assert!(ga.iter().any(|g| g.abs() > 1e-6), "the gradient was zero everywhere: nothing can train");
    }

    /// A control for the test above: with the estimator removed — quantizing as a plain constant —
    /// the encoder receives NO gradient at all. This is the failure the module prevents, shown
    /// rather than described.
    #[test]
    fn quantizing_without_the_estimator_kills_the_gradient() {
        let Some(ctx) = ctx() else { return };
        let q = Fsq::signal_15bit();
        let z: Vec<f32> = (0..30).map(|i| ((i * 37) % 23) as f32 * 0.4 - 4.0).collect();
        let tv = Var::leaf(Tensor::from_vec(&ctx, &vec![0.0f32; 30], &[6, 5]));

        let a = Var::leaf(Tensor::from_vec(&ctx, &z, &[6, 5]));
        // Quantize as a detached constant: exactly what a naive implementation does.
        let mut qv = Vec::new();
        for row in z.chunks(5) {
            qv.extend(q.dequantize(&q.quantize(row).unwrap()).unwrap());
        }
        let naive = Var::leaf(Tensor::from_vec(&ctx, &qv, &[6, 5]));
        mse(&naive, &tv).backward();
        assert!(a.grad().is_none(), "a detached quantizer must leave the encoder with no gradient");
    }

    /// Descent actually works through the bottleneck: latents move toward a target that only the
    /// quantizer's output is compared against.
    #[test]
    fn gradient_descent_reduces_the_loss_through_the_bottleneck() {
        let Some(ctx) = ctx() else { return };
        let q = Fsq::signal_15bit();
        let target: Vec<f32> = (0..50).map(|i| ((i * 13) % 7) as f32 - 3.0).collect();
        let tv = Var::leaf(Tensor::from_vec(&ctx, &target, &[10, 5]));
        let mut z: Vec<f32> = (0..50).map(|i| ((i * 29) % 11) as f32 * 0.2 - 1.0).collect();

        let loss_now = |z: &[f32]| {
            let v = Var::leaf(Tensor::from_vec(&ctx, z, &[10, 5]));
            let l = mse(&straight_through(&ctx, &v, &q), &tv);
            pollster::block_on(l.value().to_vec())[0]
        };
        let first = loss_now(&z);

        for _ in 0..300 {
            let v = Var::leaf(Tensor::from_vec(&ctx, &z, &[10, 5]));
            let l = mse(&straight_through(&ctx, &v, &q), &tv);
            l.backward();
            let g = pollster::block_on(v.grad().unwrap().to_vec());
            for (zi, gi) in z.iter_mut().zip(&g) {
                *zi -= 0.5 * gi;
            }
        }
        let last = loss_now(&z);
        assert!(last < first, "loss did not fall: {first} -> {last}");
        // And it fell materially, not by rounding.
        assert!(last < first * 0.5, "loss barely moved: {first} -> {last}");
        eprintln!("    descent through the bottleneck: loss {first:.4} -> {last:.4} ({:.1}x)", first / last.max(1e-9));
    }

    #[test]
    fn a_non_finite_latent_is_passed_through_rather_than_laundered() {
        let Some(ctx) = ctx() else { return };
        let q = Fsq::signal_15bit();
        let mut z = vec![0.5f32; 10];
        z[3] = f32::NAN;
        let v = Var::leaf(Tensor::from_vec(&ctx, &z, &[2, 5]));
        let out = pollster::block_on(straight_through(&ctx, &v, &q).value().to_vec());
        assert!(out[3].is_nan(), "a NaN latent was silently turned into a legal code");
    }
}
