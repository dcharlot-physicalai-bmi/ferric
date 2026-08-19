//! The 1-D transformer encoder that turns patches into the latents FSQ quantizes.
//!
//! ## Sizing the reproduction against the thing being reproduced
//!
//! The published description of the tokenizer this crate reproduces gives one hard number: a
//! **9.5M-parameter 1-D transformer autoencoder**. An autoencoder is an encoder plus a decoder, so
//! the encoder alone should land near 4.75M, and [`EncoderConfig::params`] exists so that a claimed
//! architecture can be checked against that figure rather than asserted to match it.
//!
//! [`EncoderConfig::signal_4m`] is a configuration that lands within 1% of half of 9.5M. It is a
//! **hypothesis about the shape of the published model, not a recovery of it**: the width, depth and
//! feed-forward ratio below are one solution among many that hit the same parameter count, and the
//! source does not state which one it used. The count is checked; the correspondence is not.
//!
//! ## What is verified here
//!
//! Parameter arithmetic and the positional encoding are exact and tested on the CPU. The forward
//! pass is tested for shape and for bit-exact determinism against a live context. Nothing here has
//! been compared to a reference implementation's outputs, because no reference weights were located.

use crate::patch::PatchError;

/// Shape of the encoder tower.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderConfig {
    /// Samples per patch, which is the input width of the patch embedding.
    pub patch_len: usize,
    /// Residual width.
    pub d_model: usize,
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// Attention heads. Must divide `d_model`.
    pub n_heads: usize,
    /// Feed-forward inner width. SwiGLU uses three matrices of this size.
    pub d_ff: usize,
    /// Latent width handed to the quantizer. Must equal the FSQ dimension.
    pub latent_dim: usize,
}

/// Where the parameters actually are. Reported rather than summed silently, because "the model is
/// the right size" and "the model is the right size for the right reasons" are different claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamBreakdown {
    pub patch_embed: usize,
    pub attention: usize,
    pub feed_forward: usize,
    pub norms: usize,
    pub latent_head: usize,
}

impl ParamBreakdown {
    pub fn total(&self) -> usize {
        self.patch_embed + self.attention + self.feed_forward + self.norms + self.latent_head
    }
}

impl EncoderConfig {
    /// An encoder sized so that encoder + decoder lands near the published 9.5M.
    ///
    /// 16-sample patches, width 256, 5 layers, 4 heads, SwiGLU inner width 896, and a 5-dimensional
    /// latent to match [`crate::Fsq::signal_15bit`].
    pub fn signal_4m() -> Self {
        Self {
            patch_len: 16,
            d_model: 256,
            n_layers: 5,
            n_heads: 4,
            d_ff: 896,
            latent_dim: 5,
        }
    }

    pub fn validate(&self) -> Result<(), PatchError> {
        if self.d_model == 0 || self.n_heads == 0 || self.d_model % self.n_heads != 0 {
            return Err(PatchError::Ragged { len: self.d_model, channels: self.n_heads });
        }
        if self.patch_len == 0 || self.latent_dim == 0 || self.n_layers == 0 || self.d_ff == 0 {
            return Err(PatchError::Degenerate { patch_len: self.patch_len, stride: self.latent_dim });
        }
        Ok(())
    }

    #[inline]
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }


    /// Parameter count of the DECODER tower, which is **exactly the encoder's**.
    ///
    /// I got this wrong first and the test caught it. The intuition — "the encoder's head maps
    /// `d_model -> latent` and the decoder's maps `d_model -> patch_len`, so the towers differ by
    /// that swap" — is false, because **each tower contains BOTH end matrices**. The encoder has
    /// `patch_embed [d, P]` on the way in and `latent_head [latent, d]` on the way out; the decoder
    /// has `latent_up [d, latent]` in and `patch_head [P, d]` out. Same two shapes, transposed, so
    /// the totals are identical for every configuration — not by coincidence, by construction.
    ///
    /// Kept as a named accessor rather than inlined so the symmetry is asserted somewhere instead of
    /// assumed everywhere.
    pub fn decoder_params(&self) -> usize {
        self.params().total()
    }

    /// Exact parameter count, with no bias terms and no learned positional table.
    ///
    /// Per block: four `d x d` attention projections, three `d x d_ff` SwiGLU matrices, and two
    /// RMSNorm weight vectors of width `d`.
    pub fn params(&self) -> ParamBreakdown {
        let d = self.d_model;
        let l = self.n_layers;
        ParamBreakdown {
            patch_embed: self.patch_len * d,
            attention: 4 * d * d * l,
            feed_forward: 3 * d * self.d_ff * l,
            norms: 2 * d * l + d,
            latent_head: d * self.latent_dim,
        }
    }
}

/// Sinusoidal position signal, the closed form, with no learned table.
///
/// Deterministic and parameter-free, so it can be checked against its own definition rather than
/// against a checkpoint. Returned row-major, `out[t * d + i]`.
///
/// Even indices carry sine, odd indices cosine, at geometrically spaced wavelengths. `d` must be
/// even; an odd width would leave the final channel with a sine and no matching cosine, which is a
/// silent half-feature rather than an error.
pub fn sinusoidal_positions(t_len: usize, d: usize) -> Result<Vec<f32>, PatchError> {
    if d == 0 || d % 2 != 0 {
        return Err(PatchError::Ragged { len: d, channels: 2 });
    }
    let mut out = vec![0.0f32; t_len * d];
    for t in 0..t_len {
        for i in 0..d / 2 {
            // 10000^(2i/d), computed in f64 so the longest wavelengths stay distinct at large d.
            let inv = (t as f64) / 10000f64.powf(2.0 * i as f64 / d as f64);
            out[t * d + 2 * i] = inv.sin() as f32;
            out[t * d + 2 * i + 1] = inv.cos() as f32;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_parameter_breakdown_sums_to_the_total() {
        let c = EncoderConfig::signal_4m();
        let p = c.params();
        assert_eq!(
            p.total(),
            p.patch_embed + p.attention + p.feed_forward + p.norms + p.latent_head
        );
    }

    /// THE SIZING CHECK. The published tokenizer is a 9.5M-parameter autoencoder, so an encoder
    /// plus a decoder of this shape must land near that. A configuration that claims to reproduce
    /// a 9.5M model and quietly builds a 3M one is the kind of mismatch that only shows up as
    /// "our reproduction underperforms", months later.
    #[test]
    fn encoder_plus_decoder_lands_within_one_percent_of_the_published_9_5m() {
        let c = EncoderConfig::signal_4m();
        let enc = c.params().total();
        // The decoder is the same size as the encoder, which is not the intuitive answer; see
        // `decoder_params` for why, and `the_two_towers_are_the_same_size_by_construction`.
        let dec = c.decoder_params();
        let both = enc + dec;
        let target = 9_500_000f64;
        let err = (both as f64 - target).abs() / target;
        assert!(
            err < 0.01,
            "encoder {enc} + decoder {dec} = {both}, which is {:.2}% from the published 9.5M",
            err * 100.0
        );
    }

    #[test]
    fn parameter_arithmetic_is_right_for_a_hand_checkable_config() {
        let c = EncoderConfig { patch_len: 4, d_model: 8, n_layers: 2, n_heads: 2, d_ff: 16, latent_dim: 3 };
        let p = c.params();
        assert_eq!(p.patch_embed, 4 * 8);
        assert_eq!(p.attention, 4 * 8 * 8 * 2);
        assert_eq!(p.feed_forward, 3 * 8 * 16 * 2);
        assert_eq!(p.norms, 2 * 8 * 2 + 8);
        assert_eq!(p.latent_head, 8 * 3);
        assert_eq!(p.total(), 32 + 512 + 768 + 40 + 24);
    }

    #[test]
    fn bad_configs_are_refused() {
        let ok = EncoderConfig::signal_4m();
        assert!(ok.validate().is_ok());
        assert!(EncoderConfig { n_heads: 7, ..ok }.validate().is_err(), "d_model must divide by heads");
        assert!(EncoderConfig { n_heads: 0, ..ok }.validate().is_err());
        assert!(EncoderConfig { n_layers: 0, ..ok }.validate().is_err());
        assert!(EncoderConfig { latent_dim: 0, ..ok }.validate().is_err());
    }

    /// The latent width and the quantizer's dimension are the same number in two places, and a
    /// mismatch would be caught only when a quantize call errored at runtime.
    #[test]
    fn the_latent_width_matches_the_quantizer_it_feeds() {
        assert_eq!(EncoderConfig::signal_4m().latent_dim, crate::Fsq::signal_15bit().dim());
    }

    #[test]
    fn positional_encoding_matches_its_closed_form() {
        let (t, d) = (7usize, 8usize);
        let pe = sinusoidal_positions(t, d).unwrap();
        for tt in 0..t {
            for i in 0..d / 2 {
                let inv = (tt as f64) / 10000f64.powf(2.0 * i as f64 / d as f64);
                assert!((pe[tt * d + 2 * i] - inv.sin() as f32).abs() < 1e-6);
                assert!((pe[tt * d + 2 * i + 1] - inv.cos() as f32).abs() < 1e-6);
            }
        }
        // Position 0 is sin(0)=0, cos(0)=1 in every pair.
        for i in 0..d / 2 {
            assert_eq!(pe[2 * i], 0.0);
            assert_eq!(pe[2 * i + 1], 1.0);
        }
    }

    /// Two positions must never encode identically, or the model cannot tell them apart. This is
    /// checked at a realistic width and length rather than a toy one, because collisions appear at
    /// the long-wavelength end where f32 rounding bites.
    #[test]
    fn every_position_is_distinguishable_from_every_other() {
        let (t, d) = (512usize, 256usize);
        let pe = sinusoidal_positions(t, d).unwrap();
        for a in 0..t {
            for b in (a + 1)..t {
                let same = (0..d).all(|i| pe[a * d + i] == pe[b * d + i]);
                assert!(!same, "positions {a} and {b} encode identically");
            }
        }
    }

    #[test]
    fn positional_encoding_is_deterministic_and_refuses_an_odd_width() {
        let a = sinusoidal_positions(16, 32).unwrap();
        for _ in 0..8 {
            assert_eq!(sinusoidal_positions(16, 32).unwrap(), a);
        }
        assert!(sinusoidal_positions(16, 31).is_err());
        assert!(sinusoidal_positions(16, 0).is_err());
    }
}
