//! **Role-based quantization planning** — decide precision and calibration per tensor role, from
//! measurements rather than from a uniform default.
//!
//! Three results from Ferric's own experiments are encoded here, each of which contradicts the obvious
//! choice, and each of which is a test below.
//!
//! ## 1. `gate` is the fragile role — spend bits there
//!
//! `ferric-llama/examples/ternary_by_role.rs`. Qwen's FFN makes this controlled: `gate`, `up` and `down`
//! have **identical parameter counts**, so ternarizing one at a time varies role with the bit budget
//! fixed. Measured in NLL (nats/token, 6 disjoint chunks):
//!
//! ```text
//!   gate only  +5.849      up only  +4.561      down only  +4.237
//! ```
//!
//! `gate` and `up` have the same shape, the same fan-in, and read the *same* input — the only difference
//! is that `gate`'s output passes through SiLU — and `gate` is worse by 1.288 nats on **6/6 chunks**.
//!
//! Note this is the *opposite* of what a straight port of ds4's recipe would give: ds4 spends its extra
//! bits on `down`. Its stated reason (one-sided SwiGLU input) did not transfer, and `down` measured as the
//! **least** sensitive role here.
//!
//! ## 2. The imatrix must NOT be applied to `gate`
//!
//! `ferric-llama/examples/imatrix_ternary.rs`. Importance calibration helps `up` (−0.262 nats) and `down`
//! (−0.467, on 6/6) — right in the range ds4 publishes — and **catastrophically hurts `gate`: +4.156 nats
//! on 0/6 chunks**. Not a degenerate quantizer: importance skew was 23× max/median and the reconstruction
//! zero-fraction moved only 52.9% → 54.2%.
//!
//! The mechanism: importance weighting minimises error in `Wx` weighted by input energy, which is the
//! right proxy only when the output is consumed **linearly**. `up`'s is; `gate`'s is not — it passes
//! through SiLU, whose behaviour near zero decides which units gate *off*, so moving error toward
//! low-energy channels changes *which units cross zero*. Selective calibration beat both uniform choices:
//!
//! ```text
//!   all-FFN ternary:  plain +10.417   uniform-imatrix +10.028   SELECTIVE +9.553
//! ```
//!
//! ## 3. Never let one scale span two populations
//!
//! From colibri's int8-MTP failure: `eh_proj`'s two column halves differ ~20–30× in scale, and one
//! per-row int4 scale rounds the **entire** small half to exact zeros, collapsing draft acceptance to
//! 0–4%. Group-scaled int4 does not fail, so the bug is the *scale span*, not the bit width.
//! [`scale_span_check`] turns that into a computable guard.

use crate::imatrix::Imatrix;

/// What a tensor does, derived from its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// FFN gate — output passes through the activation and decides gating. The fragile one.
    Gate,
    Up,
    Down,
    AttnQkv,
    AttnOut,
    /// MoE router. Never quantized: it decides *which* experts run, and an error here changes the
    /// computation rather than perturbing it. Every production 2-bit engine keeps this at F16 or better.
    Router,
    /// Norms, biases, anything 1-D.
    Norm,
    Embed,
    Output,
    Other,
}

/// Classify by tensor name (GGUF conventions).
pub fn role_of(name: &str) -> Role {
    let n = name.rsplit('.').nth(1).unwrap_or(name); // "blk.0.ffn_gate.weight" -> "ffn_gate"
    match n {
        "ffn_gate" | "ffn_gate_exps" | "ffn_gate_shexp" => Role::Gate,
        "ffn_up" | "ffn_up_exps" | "ffn_up_shexp" => Role::Up,
        "ffn_down" | "ffn_down_exps" | "ffn_down_shexp" => Role::Down,
        "attn_q" | "attn_k" | "attn_v" | "attn_qkv" => Role::AttnQkv,
        "attn_output" => Role::AttnOut,
        "ffn_gate_inp" | "ffn_exp_probs_b" => Role::Router,
        "token_embd" => Role::Embed,
        "output" => Role::Output,
        _ if n.ends_with("_norm") || n.contains("norm") => Role::Norm,
        _ => Role::Other,
    }
}

/// Per-tensor decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decision {
    /// `None` means "do not quantize this tensor".
    pub bits: Option<u8>,
    /// Whether importance calibration should weight this tensor's quantization.
    pub use_imatrix: bool,
}

/// A quantization plan: base precision, plus the role-specific corrections that were measured.
#[derive(Debug, Clone, Copy)]
pub struct QuantPlan {
    /// Bits for ordinary quantizable roles.
    pub base_bits: u8,
    /// Extra bits for [`Role::Gate`], the measurably most fragile role. `0` disables the correction.
    pub gate_bonus_bits: u8,
    /// Apply the imatrix at all. When true it is still withheld from `gate` — see the module docs.
    pub imatrix: bool,
}

impl Default for QuantPlan {
    fn default() -> Self { Self { base_bits: 2, gate_bonus_bits: 2, imatrix: true } }
}

impl QuantPlan {
    pub fn decide(&self, name: &str) -> Decision {
        let role = role_of(name);
        match role {
            // Never quantized. The router decides which experts run; norms and biases are a rounding
            // error's worth of parameters and a large share of the damage if they move.
            Role::Router | Role::Norm => Decision { bits: None, use_imatrix: false },
            // Embedding and output head: quantizable, but not below 8 bits by default — colibri measured
            // that keeping them in fp16 is NOT the fix for per-row int4 damage, so they are not special,
            // merely not worth squeezing.
            Role::Embed | Role::Output => {
                Decision { bits: Some(self.base_bits.max(8)), use_imatrix: self.imatrix }
            }
            Role::Gate => Decision {
                bits: Some(self.base_bits.saturating_add(self.gate_bonus_bits)),
                // THE finding: calibration hurts this role badly (+4.156 nats, 0/6 chunks).
                use_imatrix: false,
            },
            Role::Up | Role::Down | Role::AttnQkv | Role::AttnOut => {
                Decision { bits: Some(self.base_bits), use_imatrix: self.imatrix }
            }
            Role::Other => Decision { bits: Some(self.base_bits), use_imatrix: self.imatrix },
        }
    }

    /// Importance vector for a tensor, or `None` when the plan withholds calibration from its role.
    pub fn importance(&self, name: &str, cols: usize, im: &Imatrix, capture: &str) -> Option<Vec<f32>> {
        self.decide(name).use_imatrix.then(|| im.get_or_uniform(capture, cols))
    }
}

/// Outcome of the scale-span guard.
#[derive(Debug, Clone, PartialEq)]
pub enum ScaleSpan {
    Ok { ratio: f64 },
    /// One shared scale would round an entire low-magnitude population to zero.
    Collapse { ratio: f64, limit: f64, zeroed_fraction: f64 },
}

impl ScaleSpan {
    pub fn is_ok(&self) -> bool { matches!(self, ScaleSpan::Ok { .. }) }
}

/// Would a shared scale destroy a sub-population of `row`?
///
/// A symmetric quantizer with `levels` positive steps sets each group's scale from that group's largest
/// magnitude: `s = absmax / levels`. Anything below `s/2` rounds to zero. So if a **scaling group**
/// contains a coherent sub-population whose magnitudes are more than `2 · levels` times smaller than the
/// group's peak, that population is annihilated — silently, because the model still runs and still emits
/// fluent text.
///
/// For int4 (`levels = 7`) the limit is **14×**, exactly the threshold colibri's dead MTP head crossed at
/// 20–30×. For ternary (`levels = 1`) it is **2×**, which is why ternary is far more sensitive to a
/// heterogeneous row than the bit count alone suggests.
///
/// `group` is the quantizer's scaling-group size — pass the row width to model a per-row scale. The check
/// is deliberately made **within each group independently**, because that is what group scaling means: a
/// span between two groups is irrelevant, since each carries its own scale. (Getting this wrong was the
/// first version of this function, and the group-scaling test caught it.)
///
/// Populations are detected at sub-block granularity rather than per element, so that naturally small
/// individual weights — which every tensor has and which quantize to zero harmlessly — are not mistaken
/// for a structured population.
pub fn scale_span_check(row: &[f32], group: usize, levels: u32) -> ScaleSpan {
    assert!(group > 0 && levels > 0);
    const SUBS: usize = 8;
    let limit = 2.0 * levels as f64;
    let (mut worst_ratio, mut worst_zeroed) = (1.0f64, 0.0f64);

    for g in row.chunks(group) {
        let sub = (g.len() / SUBS).max(1);
        let maxes: Vec<f64> = g
            .chunks(sub)
            .map(|c| c.iter().fold(0f64, |a, &x| a.max(x.abs() as f64)))
            .filter(|m| *m > 0.0)
            .collect();
        if maxes.len() < 2 { continue; }
        let (lo, hi) = maxes.iter().fold((f64::MAX, 0f64), |(a, b), &m| (a.min(m), b.max(m)));
        let ratio = hi / lo;
        if ratio > worst_ratio {
            worst_ratio = ratio;
            let step = hi / levels as f64;
            worst_zeroed = g.iter().filter(|x| (x.abs() as f64) < step * 0.5).count() as f64
                / g.len().max(1) as f64;
        }
    }

    if worst_ratio > limit {
        ScaleSpan::Collapse { ratio: worst_ratio, limit, zeroed_fraction: worst_zeroed }
    } else {
        ScaleSpan::Ok { ratio: worst_ratio }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_are_recognised_from_gguf_names() {
        assert_eq!(role_of("blk.0.ffn_gate.weight"), Role::Gate);
        assert_eq!(role_of("blk.12.ffn_up.weight"), Role::Up);
        assert_eq!(role_of("blk.3.ffn_down.weight"), Role::Down);
        assert_eq!(role_of("blk.0.attn_q.weight"), Role::AttnQkv);
        assert_eq!(role_of("blk.0.attn_output.weight"), Role::AttnOut);
        assert_eq!(role_of("blk.0.ffn_gate_inp.weight"), Role::Router);
        assert_eq!(role_of("blk.0.attn_norm.weight"), Role::Norm);
        assert_eq!(role_of("token_embd.weight"), Role::Embed);
        // MoE variants must not be mistaken for their dense namesakes.
        assert_eq!(role_of("blk.5.ffn_gate_exps.weight"), Role::Gate);
        assert_eq!(role_of("blk.5.ffn_down_exps.weight"), Role::Down);
    }

    #[test]
    fn gate_gets_more_bits_than_its_siblings() {
        // Measured: gate +5.849 nats vs up +4.561 at IDENTICAL parameter count, 6/6 chunks. If bits are
        // reallocated by role, they go here — the opposite of a straight port of ds4's recipe, which
        // spends them on `down`.
        let p = QuantPlan::default();
        let g = p.decide("blk.0.ffn_gate.weight").bits.unwrap();
        let u = p.decide("blk.0.ffn_up.weight").bits.unwrap();
        let d = p.decide("blk.0.ffn_down.weight").bits.unwrap();
        assert!(g > u && g > d, "gate {g} did not get more bits than up {u} / down {d}");
        assert_eq!(u, d, "up and down measured comparably and should be treated the same");
    }

    #[test]
    fn the_imatrix_is_withheld_from_gate_and_only_from_gate() {
        // Measured: imatrix helps up (-0.262) and down (-0.467, 6/6) and devastates gate (+4.156, 0/6).
        // Selective calibration beat both uniform choices (+9.553 vs +10.028 / +10.417).
        let p = QuantPlan::default();
        assert!(!p.decide("blk.0.ffn_gate.weight").use_imatrix, "gate must not be calibrated");
        for n in ["blk.0.ffn_up.weight", "blk.0.ffn_down.weight", "blk.0.attn_q.weight", "blk.0.attn_output.weight"] {
            assert!(p.decide(n).use_imatrix, "{n} should be calibrated");
        }
    }

    #[test]
    fn the_router_and_norms_are_never_quantized() {
        // The router decides WHICH experts run; an error there changes the computation rather than
        // perturbing it, which is why every production 2-bit engine leaves it alone.
        let p = QuantPlan::default();
        assert_eq!(p.decide("blk.0.ffn_gate_inp.weight").bits, None);
        assert_eq!(p.decide("blk.0.attn_norm.weight").bits, None);
        assert_eq!(p.decide("output_norm.weight").bits, None);
    }

    #[test]
    fn disabling_the_corrections_gives_a_uniform_plan() {
        // The measured corrections must be switchable, so a uniform baseline stays expressible — that is
        // how the findings were established in the first place.
        let p = QuantPlan { base_bits: 2, gate_bonus_bits: 0, imatrix: false };
        for n in ["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"] {
            let d = p.decide(n);
            assert_eq!(d.bits, Some(2));
            assert!(!d.use_imatrix);
        }
    }

    #[test]
    fn the_scale_span_guard_catches_the_dead_mtp_head() {
        // colibri's failure, reconstructed: two column halves 25x apart under a single per-row int4 scale
        // rounds the small half to exact zeros and collapses draft acceptance to 0-4%.
        let mut row = vec![0f32; 256];
        for (i, v) in row.iter_mut().enumerate() {
            *v = if i < 128 { 0.05 * ((i % 7) as f32 - 3.0) / 3.0 } else { 1.5 * ((i % 5) as f32 - 2.0) / 2.0 };
        }
        // A per-row scale means ONE group spanning the whole row.
        match scale_span_check(&row, 256, 7) {
            ScaleSpan::Collapse { ratio, limit, zeroed_fraction } => {
                assert!(ratio > 14.0, "ratio {ratio} should exceed the int4 limit {limit}");
                assert!(zeroed_fraction > 0.3, "expected a large zeroed population, got {zeroed_fraction:.2}");
            }
            other => panic!("guard missed the collapse: {other:?}"),
        }
    }

    #[test]
    fn group_scaling_rescues_what_a_per_row_scale_destroys() {
        // The real lesson: the bug is the SCALE SPAN, not the bit width. With a small enough group each
        // population gets its own scale and nothing is annihilated.
        let mut row = vec![0f32; 256];
        for (i, v) in row.iter_mut().enumerate() {
            *v = if i < 128 { 0.05 } else { 1.5 };
        }
        assert!(!scale_span_check(&row, 256, 7).is_ok(), "a per-row scale must be refused here");
        // Group scaling gives each population its own scale, so nothing is annihilated. This is the
        // actual lesson: the bug is the SCALE SPAN WITHIN A GROUP, not the bit width.
        assert!(scale_span_check(&row, 128, 7).is_ok(), "group scaling should be accepted");
        assert!(scale_span_check(&row, 64, 7).is_ok(), "finer group scaling should be accepted");
    }

    #[test]
    fn the_limit_scales_with_the_quantizer_and_ternary_is_the_strictest() {
        // 2*levels: int4 (7 levels) -> 14x, int8 (127) -> 254x, ternary (1) -> 2x. Ternary's extreme
        // sensitivity to a heterogeneous row is not obvious from the bit count alone.
        let mut row = vec![0f32; 128];
        for (i, v) in row.iter_mut().enumerate() { *v = if i < 64 { 0.2 } else { 1.0 }; } // 5x span
        assert!(scale_span_check(&row, 128, 7).is_ok(), "5x is under int4's 14x limit");
        assert!(scale_span_check(&row, 128, 127).is_ok(), "5x is far under int8's 254x limit");
        assert!(!scale_span_check(&row, 128, 1).is_ok(), "5x EXCEEDS ternary's 2x limit");
    }

    #[test]
    fn a_uniform_row_is_never_flagged() {
        let row = vec![0.3f32; 256];
        assert!(scale_span_check(&row, 32, 1).is_ok());
        assert!(scale_span_check(&row, 256, 1).is_ok(), "a uniform row is fine at any group size");
        assert!(scale_span_check(&[], 32, 7).is_ok(), "an empty row must not panic");
    }
}
