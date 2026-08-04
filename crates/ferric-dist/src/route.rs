//! Route construction: find a chain of workers that covers every layer **exactly once**.
//!
//! There is no auto-balancing here, deliberately — spans are declared by each worker (`--layers A:B`) and
//! this only decides whether a declared set forms a valid chain. Balancing is a tuning decision made from
//! measured per-hop timings, and guessing at it would produce a route that is valid and slow, which is
//! harder to debug than one that is refused.
//!
//! The requirement is **exact adjacency**: gapless, non-overlapping, starting at layer 0, ending with a
//! worker that owns the output head. Every one of those conditions, if relaxed, produces a route that
//! runs and returns wrong numbers rather than failing:
//!
//! - a **gap** silently skips layers, and the output is a plausible continuation from a shallower model;
//! - an **overlap** applies layers twice;
//! - not starting at 0 feeds embeddings into the middle of the stack;
//! - no output head means nothing produces logits.

use crate::DistError;

/// A worker advertising the span it can serve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Registration {
    pub id: String,
    /// First layer this worker owns.
    pub layer_start: u32,
    /// One past the last layer this worker owns.
    pub layer_end: u32,
    /// Whether it also owns the final norm + output head.
    pub has_output: bool,
    /// Model fingerprint. Two workers holding different checkpoints must never end up in one chain, and
    /// nothing in the activations they exchange would reveal the mismatch.
    pub model: u64,
}

impl Registration {
    pub fn covers(&self) -> u32 { self.layer_end.saturating_sub(self.layer_start) }
}

/// An ordered chain of workers covering `0..n_layers`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Route {
    pub hops: Vec<Registration>,
    pub n_layers: u32,
    pub model: u64,
}

impl Route {
    /// Bytes of hidden state that cross the wire per hop per token.
    ///
    /// `n_hc` is the hyper-connection multiplier — 1 for an ordinary residual stream, 4 for architectures
    /// that carry several. It is a parameter rather than an assumption because getting it wrong
    /// under-counts traffic by 4× on exactly the models that need distributing.
    pub fn bytes_per_token_per_hop(n_embd: usize, n_hc: usize) -> usize { n_embd * n_hc * 4 }

    /// Total activation bytes moved for one token across the whole chain.
    pub fn bytes_per_token(&self, n_embd: usize, n_hc: usize) -> usize {
        self.hops.len().saturating_sub(1) * Self::bytes_per_token_per_hop(n_embd, n_hc)
    }
}

/// Find a chain serving `model`, or explain why there isn't one.
///
/// Depth-first over exact-adjacency successors. Candidates are tried longest-span-first, which is a
/// preference and not a correctness property: fewer hops means fewer round trips, and decode pays one
/// round trip per token.
///
/// # Why the model is a parameter rather than inferred
///
/// An earlier version searched every model present and returned the first chain it could complete. That
/// is a silent wrong answer waiting to happen: a fleet serving two checkpoints would hand back a valid
/// route for whichever one happened to sort first, and every layer of validation downstream would agree
/// it was fine. The caller knows which checkpoint it wants; asking is free.
pub fn plan_route(regs: &[Registration], n_layers: u32, model: u64) -> Result<Route, DistError> {
    if n_layers == 0 || regs.is_empty() { return Err(DistError::NoRoute); }
    let pool: Vec<&Registration> = regs
        .iter()
        .filter(|r| r.model == model && r.layer_end > r.layer_start && r.layer_end <= n_layers)
        .collect();
    let mut chain: Vec<usize> = Vec::new();
    let mut used = vec![false; pool.len()];
    if dfs(&pool, 0, n_layers, &mut used, &mut chain) {
        return Ok(Route {
            hops: chain.into_iter().map(|i| pool[i].clone()).collect(),
            n_layers,
            model,
        });
    }
    Err(DistError::NoRoute)
}

fn dfs(
    pool: &[&Registration],
    at: u32,
    n_layers: u32,
    used: &mut [bool],
    chain: &mut Vec<usize>,
) -> bool {
    if at == n_layers {
        // The chain is only complete if its last hop can actually emit logits.
        return chain.last().is_some_and(|&i| pool[i].has_output);
    }
    // Longest span first: fewer hops, fewer round trips.
    let mut cands: Vec<usize> = (0..pool.len())
        .filter(|&i| !used[i] && pool[i].layer_start == at)
        .collect();
    cands.sort_by_key(|&i| std::cmp::Reverse(pool[i].covers()));

    for i in cands {
        used[i] = true;
        chain.push(i);
        if dfs(pool, pool[i].layer_end, n_layers, used, chain) { return true; }
        chain.pop();
        used[i] = false;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reg(id: &str, s: u32, e: u32, out: bool) -> Registration {
        Registration { id: id.into(), layer_start: s, layer_end: e, has_output: out, model: 1 }
    }

    #[test]
    fn builds_a_gapless_chain() {
        let r = plan_route(&[reg("b", 16, 32, true), reg("a", 0, 16, false)], 32, 1).unwrap();
        assert_eq!(r.hops.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(), ["a", "b"]);
    }

    #[test]
    fn a_gap_is_refused_rather_than_skipped() {
        // Left unchecked this runs and returns a plausible continuation from a SHALLOWER model — the
        // worst failure mode available, because nothing errors.
        let e = plan_route(&[reg("a", 0, 10, false), reg("b", 12, 32, true)], 32, 1).unwrap_err();
        assert_eq!(e, DistError::NoRoute);
    }

    #[test]
    fn an_overlap_is_refused_rather_than_applied_twice() {
        let e = plan_route(&[reg("a", 0, 20, false), reg("b", 10, 32, true)], 32, 1).unwrap_err();
        assert_eq!(e, DistError::NoRoute);
    }

    #[test]
    fn a_chain_that_does_not_start_at_zero_is_refused() {
        let e = plan_route(&[reg("a", 4, 20, false), reg("b", 20, 32, true)], 32, 1).unwrap_err();
        assert_eq!(e, DistError::NoRoute);
    }

    #[test]
    fn a_chain_with_no_output_head_is_refused() {
        // Covers every layer and still cannot produce a token.
        let e = plan_route(&[reg("a", 0, 16, false), reg("b", 16, 32, false)], 32, 1).unwrap_err();
        assert_eq!(e, DistError::NoRoute);
    }

    #[test]
    fn workers_from_different_checkpoints_never_share_a_chain() {
        // Nothing in the activations they exchange would reveal the mismatch, so it has to be refused
        // structurally.
        let mut a = reg("a", 0, 16, false);
        let mut b = reg("b", 16, 32, true);
        a.model = 111;
        b.model = 222;
        assert_eq!(plan_route(&[a, b], 32, 111).unwrap_err(), DistError::NoRoute);
        // ...and asking for the OTHER model does not accidentally succeed either.
        let (a2, b2) = (reg("a", 0, 16, false), reg("b", 16, 32, true));
        let (mut a2, mut b2) = (a2, b2);
        a2.model = 111; b2.model = 222;
        assert_eq!(plan_route(&[a2, b2], 32, 222).unwrap_err(), DistError::NoRoute);
    }

    #[test]
    fn the_right_model_is_chosen_when_several_are_registered() {
        let (mut a2, mut b2) = (reg("a2", 0, 16, false), reg("b2", 16, 32, true));
        a2.model = 999;
        b2.model = 999;
        let regs = vec![reg("a", 0, 16, false), a2, b2, reg("b", 16, 32, true)];
        let r = plan_route(&regs, 32, 999).unwrap();
        assert_eq!(r.model, 999, "routed a model the caller did not ask for");
        assert!(r.hops.iter().all(|h| h.model == r.model), "chain mixed models: {:?}", r.hops);
    }

    #[test]
    fn prefers_fewer_hops_when_both_cover_the_layers() {
        // Decode pays one round trip per hop per token, so a 2-hop chain beats a 4-hop one at equal
        // coverage. A preference, not a correctness property — but a measurable one.
        let regs = vec![
            reg("wide", 0, 24, false),
            reg("n1", 0, 8, false),
            reg("n2", 8, 16, false),
            reg("n3", 16, 24, false),
            reg("tail", 24, 32, true),
        ];
        let r = plan_route(&regs, 32, 1).unwrap();
        assert_eq!(r.hops.len(), 2, "took the long way round: {:?}", r.hops.iter().map(|h| &h.id).collect::<Vec<_>>());
    }

    #[test]
    fn backtracks_out_of_a_dead_end() {
        // The greedy longest-first choice leads nowhere here; only backtracking finds the valid chain.
        let regs = vec![
            reg("greedy", 0, 20, false), // nothing starts at 20
            reg("a", 0, 16, false),
            reg("b", 16, 32, true),
        ];
        let r = plan_route(&regs, 32, 1).unwrap();
        assert_eq!(r.hops.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(), ["a", "b"]);
    }

    #[test]
    fn traffic_accounting_scales_with_hyper_connections() {
        // n_hc is a parameter because assuming 1 under-counts by 4x on exactly the architectures large
        // enough to need distributing.
        let r = plan_route(&[reg("a", 0, 16, false), reg("b", 16, 32, true)], 32, 1).unwrap();
        assert_eq!(Route::bytes_per_token_per_hop(6144, 1), 24_576);
        assert_eq!(Route::bytes_per_token_per_hop(6144, 4), 98_304);
        assert_eq!(r.bytes_per_token(6144, 4), 98_304, "2 hops = 1 wire crossing");
    }
}
