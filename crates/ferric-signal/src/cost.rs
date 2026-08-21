//! What one token actually costs, in operations and in bytes.
//!
//! ## Why this is not a joules figure, and why it is still the useful half
//!
//! This review located no joules-per-token figure published for any sensor-language or time-series
//! foundation model. Energy needs a meter, and a meter needs the run to happen on hardware that has
//! one; [`ferric_joule`] already draws that line properly, returning `None` rather than a zero when
//! no counter is readable. What does NOT need a meter is the operation count, which is exact
//! arithmetic over the configuration, and which is the term any energy figure is proportional to.
//!
//! ## The part that makes a single number wrong
//!
//! Attention is quadratic in the window. So the cost of a token is **not a property of the token** —
//! it is a property of the window the token was in. Quoting "X joules per token" without the window
//! length is not a tight figure with the context missing; it is an underspecified one. The
//! breakdown below separates the linear term from the quadratic term precisely so a reader can see
//! where the crossover is for their own window, and [`TokenCost::quadratic_share`] reports how much
//! of the bill the window length is responsible for.

use crate::encoder::EncoderConfig;

/// Cost of encoding one window, and of one token within it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenCost {
    /// Patches in the window this cost was computed for.
    pub window: usize,
    /// Multiply-accumulates that scale linearly with the window: projections and feed-forward.
    pub linear_macs: u64,
    /// Multiply-accumulates from attention, which scale with the SQUARE of the window.
    pub quadratic_macs: u64,
    /// Weight bytes read once per forward pass, at 4 bytes per parameter.
    pub weight_bytes: u64,
}

impl TokenCost {
    /// Two floating-point operations per multiply-accumulate.
    pub fn flops(&self) -> u64 {
        2 * (self.linear_macs + self.quadratic_macs)
    }

    pub fn flops_per_token(&self) -> f64 {
        if self.window == 0 { f64::NAN } else { self.flops() as f64 / self.window as f64 }
    }

    /// Fraction of the arithmetic that exists only because the window is long.
    ///
    /// At a short window this is negligible and a per-token figure is nearly honest. As it rises,
    /// a per-token figure quoted without its window becomes progressively more misleading.
    pub fn quadratic_share(&self) -> f64 {
        let t = self.linear_macs + self.quadratic_macs;
        if t == 0 { 0.0 } else { self.quadratic_macs as f64 / t as f64 }
    }

    /// Arithmetic intensity: FLOPs per byte of weights read. Below the hardware's ratio the pass is
    /// bandwidth-bound, which is where a small model on an edge part actually lives.
    pub fn flops_per_weight_byte(&self) -> f64 {
        if self.weight_bytes == 0 { f64::NAN } else { self.flops() as f64 / self.weight_bytes as f64 }
    }
}

impl EncoderConfig {
    /// Exact cost of one encoder forward pass over `window` patches.
    ///
    /// Counts matrix multiplies only. Norms, activations and the positional add are O(T·d) and
    /// contribute under a percent at every configuration in this crate; they are excluded and this
    /// sentence is the disclosure.
    pub fn cost(&self, window: usize) -> TokenCost {
        let (t, d, l) = (window as u64, self.d_model as u64, self.n_layers as u64);
        let ff = self.d_ff as u64;

        let embed = t * self.patch_len as u64 * d;
        let qkv = 3 * t * d * d * l;
        let out_proj = t * d * d * l;
        let ffn = 3 * t * d * ff * l;
        let head = t * d * self.latent_dim as u64;

        // scores = Q·Kᵀ and probs·V, each t² · d per layer.
        let attn = 2 * t * t * d * l;

        TokenCost {
            window,
            linear_macs: embed + qkv + out_proj + ffn + head,
            quadratic_macs: attn,
            weight_bytes: self.params().total() as u64 * 4,
        }
    }
}

/// What a vocabulary costs to carry, per token generated.
///
/// A decoder over a hybrid vocabulary touches its embedding table twice per position: once to look
/// a token up, and once at the output head to score every row. The lookup is one row; **the head is
/// the whole table**, and that is the term that grows with codebook size. At the edge this is not a
/// rounding error — the crate's own arithmetic-intensity measurement puts a small tokenizer at
/// ~8 FLOP per weight byte, far under the ~100–300 a modern part needs to be compute-bound, so
/// **bytes moved, not operations, set the bill**.
///
/// This exists because the resolution sweep in the README measured 243 codes and 32,768 codes
/// scoring identically on held-out accuracy at one corpus scale. Identical accuracy at 135x fewer
/// codes is an energy claim as much as a modelling one, and this makes it exact rather than
/// rhetorical.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VocabCost {
    pub rows: u64,
    pub d_model: u64,
    /// Bytes in the embedding table at 4 bytes per parameter.
    pub embed_bytes: u64,
    /// Bytes in the untied output head, which is the same shape.
    pub head_bytes: u64,
    /// Multiply-accumulates to score every row at one position.
    pub head_macs_per_token: u64,
}

impl VocabCost {
    /// Weight bytes that must be read to emit ONE token: the head in full, plus one embedding row.
    ///
    /// The head dominates by a factor of `rows`, which is exactly why codebook size is an energy
    /// decision.
    pub fn bytes_per_token(&self) -> u64 {
        self.head_bytes + self.d_model * 4
    }

    /// Ratio of this vocabulary's per-token traffic to another's. `> 1` means self is costlier.
    pub fn traffic_vs(&self, other: &VocabCost) -> f64 {
        self.bytes_per_token() as f64 / other.bytes_per_token().max(1) as f64
    }
}

/// Cost of carrying a vocabulary of `rows` rows at width `d_model`.
pub fn vocab_cost(rows: u32, d_model: usize) -> VocabCost {
    let (r, d) = (rows as u64, d_model as u64);
    VocabCost {
        rows: r,
        d_model: d,
        embed_bytes: r * d * 4,
        head_bytes: r * d * 4,
        head_macs_per_token: r * d,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE ENERGY FORM OF THE RESOLUTION FINDING. The README measures 243 codes and 32,768 codes
    /// scoring the same on held-out accuracy at one corpus scale. If accuracy is equal, the cost
    /// difference is the whole argument, and it is large.
    #[test]
    fn the_small_codebook_moves_far_fewer_bytes_per_token() {
        let words = 6u32;
        let markers = 3u32;
        let d = 64usize;
        let small = vocab_cost(words + 243 + markers, d);
        let large = vocab_cost(words + 32_768 + markers, d);
        assert!(small.bytes_per_token() < large.bytes_per_token());
        let ratio = large.traffic_vs(&small);
        assert!(
            ratio > 100.0,
            "expected the 32,768-code vocabulary to move over 100x the bytes per token, got {ratio:.0}x"
        );
    }

    #[test]
    fn the_head_dominates_a_single_row_lookup() {
        let v = vocab_cost(32_777, 64);
        // The lookup is one row; the head is every row. The ratio IS the vocabulary size.
        assert!(v.head_bytes / (v.d_model * 4) > 30_000);
        assert_eq!(v.bytes_per_token(), v.head_bytes + v.d_model * 4);
    }

    #[test]
    fn vocab_cost_arithmetic_is_hand_checkable() {
        let v = vocab_cost(10, 8);
        assert_eq!(v.embed_bytes, 10 * 8 * 4);
        assert_eq!(v.head_bytes, 10 * 8 * 4);
        assert_eq!(v.head_macs_per_token, 80);
        assert_eq!(v.bytes_per_token(), 320 + 32);
    }

    #[test]
    fn a_vocabulary_costs_the_same_as_itself() {
        let v = vocab_cost(1000, 64);
        assert_eq!(v.traffic_vs(&v), 1.0);
    }

    #[test]
    fn linear_terms_scale_linearly_and_attention_scales_quadratically() {
        let c = EncoderConfig::signal_4m();
        let a = c.cost(64);
        let b = c.cost(128);
        assert_eq!(b.linear_macs, 2 * a.linear_macs, "projections must be linear in the window");
        assert_eq!(b.quadratic_macs, 4 * a.quadratic_macs, "attention must be quadratic");
    }

    /// THE POINT OF THIS MODULE. A per-token cost is not a constant, so a per-token figure quoted
    /// without its window is underspecified. Demonstrated rather than asserted.
    #[test]
    fn cost_per_token_grows_with_the_window() {
        let c = EncoderConfig::signal_4m();
        let mut prev = 0.0;
        for w in [16usize, 64, 256, 1024, 4096] {
            let f = c.cost(w).flops_per_token();
            assert!(f > prev, "per-token cost did not grow from window {w}");
            prev = f;
        }
        // And the growth is material, not a rounding effect.
        let short = c.cost(16).flops_per_token();
        let long = c.cost(4096).flops_per_token();
        assert!(long > 3.0 * short, "expected the window to dominate by 4096; {long} vs {short}");
    }

    #[test]
    fn the_quadratic_share_is_negligible_when_short_and_dominant_when_long() {
        let c = EncoderConfig::signal_4m();
        assert!(c.cost(16).quadratic_share() < 0.02, "attention should be noise at 16 patches");
        assert!(c.cost(8192).quadratic_share() > 0.5, "attention should dominate at 8192");
    }

    #[test]
    fn cost_arithmetic_is_right_for_a_hand_checkable_config() {
        let c = EncoderConfig { patch_len: 4, d_model: 8, n_layers: 2, n_heads: 2, d_ff: 16, latent_dim: 3 };
        let k = c.cost(10);
        // embed 10*4*8, qkv 3*10*8*8*2, out 10*8*8*2, ffn 3*10*8*16*2, head 10*8*3
        assert_eq!(k.linear_macs, 320 + 3840 + 1280 + 7680 + 240);
        // attention 2 * 10 * 10 * 8 * 2
        assert_eq!(k.quadratic_macs, 3200);
        assert_eq!(k.flops(), 2 * (k.linear_macs + k.quadratic_macs));
        assert_eq!(k.weight_bytes, c.params().total() as u64 * 4);
    }

    #[test]
    fn a_zero_window_reports_nan_rather_than_zero() {
        let k = EncoderConfig::signal_4m().cost(0);
        assert!(k.flops_per_token().is_nan(), "an empty window has no per-token cost to report");
        assert_eq!(k.quadratic_share(), 0.0);
    }
}
