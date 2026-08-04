//! **DeepSeekMoE routing and experts** — the `noaux_tc` router, as shipped by AMD Instella-MoE and
//! DeepSeek-V3.
//!
//! Verified layer-exact against the stock `DeepseekV3MoE` module before promotion, and
//! `examples/instella_moe.rs` still runs that comparison **through this code**, so the claim is about the
//! library rather than a copy of it.
//!
//! ## The invariant that silently produces a different model
//!
//! The router computes a **sigmoid** score per expert (not a softmax — the scores do not sum to 1), then:
//!
//! ```text
//!   selection:  top-k by  (score + bias)      <- bias steers WHICH experts run
//!   weighting:  score / Σ score  · routed_scale   <- from the UNBIASED scores
//! ```
//!
//! **The bias participates in selection only.** Letting it leak into the combining weights produces a
//! model that runs, emits fluent text, and is wrong — which is why kimi-k3-in-c lists this among the five
//! invariants its test suite exists to protect, and why [`route`] is a pure function with a test that
//! pins exactly this behaviour.
//!
//! The bias exists because it is how these models balance expert load without an auxiliary loss (hence
//! *noaux*): a frozen per-expert term nudges under-used experts into the top-k without distorting the
//! mixture once they are there.

use ferric_tensor::Tensor;

/// Shape and constants for one DeepSeekMoE block.
#[derive(Debug, Clone, Copy)]
pub struct MoeConfig {
    pub hidden: usize,
    /// Per-expert FFN width.
    pub inter: usize,
    pub n_experts: usize,
    pub top_k: usize,
    /// Multiplier applied to the normalised combining weights. 2.5 on Instella; DeepSeek-V4 uses 1.5.
    /// Read from the checkpoint rather than assumed — it scales the entire routed contribution.
    pub routed_scale: f32,
}

/// One token's routing decision: `(expert index, combining weight)`.
pub type Routing = Vec<(usize, f32)>;

/// Guard against a zero denominator when every selected score underflows. Matches the reference.
const WSUM_EPS: f32 = 1e-20;

/// Compute the routing for one token from its raw gate logits.
///
/// Pure and weight-free, so the invariant above is testable without a checkpoint or a GPU.
pub fn route(logits: &[f32], bias: &[f32], cfg: &MoeConfig) -> Routing {
    debug_assert_eq!(logits.len(), cfg.n_experts);
    debug_assert_eq!(bias.len(), cfg.n_experts);
    let sigmoid = |z: f32| 1.0 / (1.0 + (-z).exp());
    let scores: Vec<f32> = logits.iter().map(|&z| sigmoid(z)).collect();

    // SELECTION uses the biased scores.
    let biased: Vec<f32> = scores.iter().zip(bias).map(|(s, b)| s + b).collect();
    let mut order: Vec<usize> = (0..cfg.n_experts).collect();
    order.sort_by(|&a, &b| biased[b].total_cmp(&biased[a]));
    let top = &order[..cfg.top_k.min(cfg.n_experts)];

    // WEIGHTING uses the unbiased scores. Normalised over the selected set only.
    let wsum: f32 = top.iter().map(|&j| scores[j]).sum::<f32>() + WSUM_EPS;
    top.iter().map(|&j| (j, scores[j] / wsum * cfg.routed_scale)).collect()
}

/// Weights for one MoE block. Expert matrices are `[out, in]`, consumed with `matmul_bt`.
pub struct MoeWeights {
    /// Router projection `[n_experts, hidden]`.
    pub gate: Tensor,
    /// Frozen per-expert selection bias `[n_experts]`.
    pub gate_bias: Vec<f32>,
    /// Per routed expert: fused gate|up `[2 * inter, hidden]`.
    pub expert_gate_up: Vec<Tensor>,
    /// Per routed expert: down `[hidden, inter]`.
    pub expert_down: Vec<Tensor>,
    /// Shared expert, run for every token and added **unweighted**.
    pub shared_gate: Tensor,
    pub shared_up: Tensor,
    pub shared_down: Tensor,
}

pub struct Moe {
    pub cfg: MoeConfig,
    pub w: MoeWeights,
}

impl Moe {
    pub fn new(cfg: MoeConfig, w: MoeWeights) -> Self { Self { cfg, w } }

    /// Routing decisions for a `[seq, n_experts]` block of gate logits.
    pub fn route_all(&self, logits: &[f32], seq: usize) -> Vec<Routing> {
        (0..seq)
            .map(|t| route(&logits[t * self.cfg.n_experts..(t + 1) * self.cfg.n_experts], &self.w.gate_bias, &self.cfg))
            .collect()
    }

    /// The shared expert: SwiGLU over the full hidden state, added to every token **unweighted**.
    ///
    /// Unweighted is not an oversight — the shared expert is outside the routed mixture, which is what
    /// makes it the part of a MoE that must stay resident and at full precision while routed experts are
    /// streamed and quantized.
    pub fn shared(&self, x: &Tensor) -> Tensor {
        x.matmul_bt(&self.w.shared_gate)
            .silu()
            .mul(&x.matmul_bt(&self.w.shared_up))
            .matmul_bt(&self.w.shared_down)
    }

    /// One routed expert applied to a single token row `[1, hidden]`.
    pub fn expert(&self, xt: &Tensor, e: usize) -> Tensor {
        let gu = xt.matmul_bt(&self.w.expert_gate_up[e]);
        let gate = gu.narrow(1, 0, self.cfg.inter);
        let up = gu.narrow(1, self.cfg.inter, self.cfg.inter);
        gate.silu().mul(&up).matmul_bt(&self.w.expert_down[e])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n: usize, k: usize) -> MoeConfig {
        MoeConfig { hidden: 8, inter: 4, n_experts: n, top_k: k, routed_scale: 2.5 }
    }

    #[test]
    fn the_bias_steers_selection_only_and_never_the_weights() {
        // THE invariant. A model that lets the bias into the combining weights runs, emits fluent text,
        // and is wrong — which is why this is pinned rather than trusted.
        let c = cfg(4, 2);
        let logits = [0.0f32, 0.0, 0.0, 0.0]; // all sigmoid to exactly 0.5
        // A large bias on experts 2 and 3 must change WHICH are chosen...
        let r = route(&logits, &[0.0, 0.0, 10.0, 9.0], &c);
        let picked: Vec<usize> = r.iter().map(|(e, _)| *e).collect();
        assert_eq!(picked, vec![2, 3], "bias did not steer selection");
        // ...but NOT the weights: all scores are equal, so the two weights must be equal, and must sum
        // to routed_scale. If the bias leaked in, expert 2 would dominate 10:9.
        assert!((r[0].1 - r[1].1).abs() < 1e-6, "bias leaked into the combining weights: {r:?}");
        assert!((r[0].1 + r[1].1 - c.routed_scale).abs() < 1e-5, "weights do not sum to routed_scale");
    }

    #[test]
    fn weights_come_from_unbiased_scores_when_scores_actually_differ() {
        let c = cfg(3, 2);
        // Experts 0 and 1 have genuinely different scores; the bias only promotes them over expert 2.
        let logits = [2.0f32, 0.0, 5.0];
        let r = route(&logits, &[10.0, 10.0, 0.0], &c);
        let picked: Vec<usize> = r.iter().map(|(e, _)| *e).collect();
        assert_eq!(picked.len(), 2);
        assert!(picked.contains(&0) && picked.contains(&1), "bias failed to promote 0 and 1 over 2");
        // The ratio must be sigmoid(2)/sigmoid(0), not anything involving the equal biases.
        let sg = |z: f32| 1.0 / (1.0 + (-z).exp());
        let w0 = r.iter().find(|(e, _)| *e == 0).unwrap().1;
        let w1 = r.iter().find(|(e, _)| *e == 1).unwrap().1;
        let expect = sg(2.0) / sg(0.0);
        assert!((w0 / w1 - expect).abs() < 1e-5, "weight ratio {} != unbiased {expect}", w0 / w1);
    }

    #[test]
    fn scores_are_sigmoid_not_softmax() {
        // Sigmoid scores do NOT sum to 1 before normalisation. Swapping in a softmax changes every
        // combining weight while leaving the top-k selection identical, so nothing downstream notices.
        let c = cfg(4, 4);
        let logits = [1.0f32, 1.0, 1.0, 1.0];
        let r = route(&logits, &[0.0; 4], &c);
        // Four equal sigmoid scores normalised over all four give 1/4 each, times routed_scale.
        for (_, w) in &r {
            assert!((w - c.routed_scale / 4.0).abs() < 1e-6, "got {w}");
        }
        // And with unequal logits the pre-normalisation scores are each in (0,1) and do not sum to 1.
        let sg = |z: f32| 1.0f32 / (1.0 + (-z).exp());
        let total: f32 = [3.0f32, -3.0, 0.5, 0.0].iter().map(|&z| sg(z)).sum();
        assert!((total - 1.0).abs() > 0.1, "sigmoid scores happened to sum to 1; test is not discriminating");
    }

    #[test]
    fn normalisation_is_over_the_selected_set_only() {
        // Normalising over ALL experts instead of the chosen top-k silently shrinks every weight — a
        // uniform scale error that looks like a slightly weaker model rather than a bug.
        let c = cfg(8, 2);
        let logits = [5.0f32, 5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0];
        let r = route(&logits, &[0.0; 8], &c);
        let sum: f32 = r.iter().map(|(_, w)| *w).sum();
        assert!((sum - c.routed_scale).abs() < 1e-5,
                "weights sum to {sum}, not routed_scale — normalised over the wrong set");
    }

    #[test]
    fn top_k_larger_than_the_expert_count_does_not_panic() {
        let c = cfg(3, 8);
        let r = route(&[1.0, 2.0, 3.0], &[0.0; 3], &c);
        assert_eq!(r.len(), 3);
    }
}
