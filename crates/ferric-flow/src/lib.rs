//! # ferric-flow — incremental dataflow, native to Ferric
//!
//! The hard, valuable idea behind Pathway (and behind Materialize, RisingWave, and Feldera) is not
//! proprietary: it is **Differential Dataflow** (McSherry et al.), the `(data, diff)` change model in
//! which a result is refreshed by computing *only the change* — the minimum work — rather than
//! recomputing from scratch. Pathway wraps a modified subset of that Rust engine in a Python layer and
//! monetizes it open-core (BSL, paid Scale/Enterprise). The *method* is public and MIT-licensed.
//!
//! `ferric-flow` ingests that method into Ferric as our own pure-Rust reimplementation. It is the
//! substrate for the Institute's live-data / RAG / MCP layer — and, unlike Pathway's always-on
//! Python/Docker service, it is `std`-only and compiles to `wasm32`, so it runs on the *edge*
//! (Cloudflare Workers) and cross-fabric, alongside the rest of Ferric.
//!
//! ## The model
//! A collection is a multiset whose contents evolve over time. A **delta** `(record, diff)` inserts
//! (`diff > 0`) or retracts (`diff < 0`) a record; a *correction* to bad or late data is just a
//! retraction of the old value and an assertion of the new one. Operators consume delta batches and
//! emit delta batches, keeping only the indexed state they need:
//! - `map` / `filter` — stateless, per-record.
//! - [`Join`] — keeps an arrangement of each input; a delta joins against the *other* side's current
//!   arrangement, so work is `|delta| × matches`, never `|left| × |right|`.
//! - [`Reduce`] — keeps each group's multiset and last emitted value; only groups whose input changed
//!   are recomputed, and each emits a retraction of its old aggregate + an assertion of the new.
//!
//! ## The guarantee (phase-1 make-or-break, proven in the tests)
//! Feeding the same data as one big batch or as any sequence of small deltas yields a **bit-identical**
//! accumulated result — while the incremental path recomputes only the touched groups. This mirrors
//! Ferric's own "KV-cache: exact vs full recompute" milestone, one layer up the stack.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::Hash;

pub mod index;
pub use index::{Hnsw, LiveIndex};

/// Signed multiplicity of a record in a collection. `+1` asserts, `-1` retracts.
pub type Diff = i64;

/// Merge a delta batch: sum multiplicities per identical record, drop anything that nets to zero,
/// and return in a deterministic (sorted) order. Consolidation is what makes "insert then retract"
/// vanish and keeps every operator's output canonical.
pub fn consolidate<D: Ord>(mut v: Vec<(D, Diff)>) -> Vec<(D, Diff)> {
    v.sort();
    let mut out: Vec<(D, Diff)> = Vec::with_capacity(v.len());
    for (d, diff) in v {
        match out.last_mut() {
            Some(last) if last.0 == d => last.1 += diff,
            _ => out.push((d, diff)),
        }
    }
    out.retain(|(_, diff)| *diff != 0);
    out
}

/// Stateless `map`: apply `f` to each record, preserving its diff.
pub fn map_delta<A, B: Ord>(input: &[(A, Diff)], f: impl Fn(&A) -> B) -> Vec<(B, Diff)> {
    consolidate(input.iter().map(|(a, d)| (f(a), *d)).collect())
}

/// Stateless `filter`: keep records satisfying `p`, preserving their diff.
pub fn filter_delta<A: Clone + Ord>(input: &[(A, Diff)], p: impl Fn(&A) -> bool) -> Vec<(A, Diff)> {
    consolidate(input.iter().filter(|(a, _)| p(a)).cloned().collect())
}

/// Apply a delta batch into a running multiset index, removing keys that reach zero multiplicity.
fn apply_into<K: Eq + Hash + Clone, V: Ord + Clone>(
    idx: &mut HashMap<K, BTreeMap<V, Diff>>,
    deltas: &[((K, V), Diff)],
) {
    for ((k, v), d) in deltas {
        let m = idx.entry(k.clone()).or_default();
        let e = m.entry(v.clone()).or_insert(0);
        *e += d;
        if *e == 0 {
            m.remove(v);
        }
    }
    idx.retain(|_, m| !m.is_empty());
}

/// Incremental equi-join of two keyed collections `(K, V1)` and `(K, V2)` → `(K, V1, V2)`.
///
/// Each side keeps an arrangement (its current multiset, indexed by key). A step processes new deltas
/// on either side; the delta-join ordering (delta1 ⋈ old-right, then new-left ⋈ delta2) includes the
/// delta×delta cross term exactly once, so the output is exactly the change in the join.
#[derive(Default)]
pub struct Join<K: Eq + Hash + Clone + Ord, V1: Ord + Clone, V2: Ord + Clone> {
    left: HashMap<K, BTreeMap<V1, Diff>>,
    right: HashMap<K, BTreeMap<V2, Diff>>,
}

impl<K: Eq + Hash + Clone + Ord, V1: Ord + Clone, V2: Ord + Clone> Join<K, V1, V2> {
    pub fn new() -> Self {
        Self { left: HashMap::new(), right: HashMap::new() }
    }

    /// Process new deltas on the left and/or right inputs; return the resulting delta on the join.
    pub fn step(
        &mut self,
        d_left: Vec<((K, V1), Diff)>,
        d_right: Vec<((K, V2), Diff)>,
    ) -> Vec<((K, V1, V2), Diff)> {
        let d_left = consolidate(d_left);
        let d_right = consolidate(d_right);
        let mut out: Vec<((K, V1, V2), Diff)> = Vec::new();

        // 1) new left deltas ⋈ current (old) right arrangement
        for ((k, v1), dl) in &d_left {
            if let Some(rm) = self.right.get(k) {
                for (v2, dr) in rm {
                    out.push(((k.clone(), v1.clone(), v2.clone()), dl * dr));
                }
            }
        }
        // 2) fold left deltas into the left arrangement
        apply_into(&mut self.left, &d_left);
        // 3) new-left arrangement ⋈ new right deltas — picks up old-left×delta and delta×delta once
        for ((k, v2), dr) in &d_right {
            if let Some(lm) = self.left.get(k) {
                for (v1, dl) in lm {
                    out.push(((k.clone(), v1.clone(), v2.clone()), dl * dr));
                }
            }
        }
        // 4) fold right deltas into the right arrangement
        apply_into(&mut self.right, &d_right);

        consolidate(out)
    }
}

/// Incremental group-by reduce: keyed input `(K, V)` → one aggregate `(K, R)` per non-empty group.
///
/// `agg` maps a group's current multiset to its aggregate. Only groups whose input changed this step
/// are recomputed; each such group emits a retraction of its previous aggregate and an assertion of
/// the new one (nothing if unchanged). `recomputes` counts group recomputations — the work-saved meter.
pub struct Reduce<K, V, R, F>
where
    K: Eq + Hash + Clone + Ord,
    V: Ord + Clone,
    R: Clone + Ord,
    F: Fn(&BTreeMap<V, Diff>) -> R,
{
    state: HashMap<K, BTreeMap<V, Diff>>,
    emitted: HashMap<K, R>,
    agg: F,
    /// Number of per-group aggregate recomputations performed over this reducer's lifetime.
    pub recomputes: usize,
}

impl<K, V, R, F> Reduce<K, V, R, F>
where
    K: Eq + Hash + Clone + Ord,
    V: Ord + Clone,
    R: Clone + Ord,
    F: Fn(&BTreeMap<V, Diff>) -> R,
{
    pub fn new(agg: F) -> Self {
        Self { state: HashMap::new(), emitted: HashMap::new(), agg, recomputes: 0 }
    }

    pub fn step(&mut self, input: Vec<((K, V), Diff)>) -> Vec<((K, R), Diff)> {
        let input = consolidate(input);
        // which groups were touched
        let mut touched: BTreeSet<K> = BTreeSet::new();
        for ((k, _), _) in &input {
            touched.insert(k.clone());
        }
        apply_into(&mut self.state, &input);

        let mut out: Vec<((K, R), Diff)> = Vec::new();
        for k in touched {
            self.recomputes += 1;
            let new = self.state.get(&k).map(|m| (self.agg)(m)); // None if the group emptied out
            let old = self.emitted.get(&k).cloned();
            if new != old {
                if let Some(o) = old {
                    out.push(((k.clone(), o), -1));
                }
                if let Some(n) = &new {
                    out.push(((k.clone(), n.clone()), 1));
                }
            }
            match new {
                Some(n) => {
                    self.emitted.insert(k, n);
                }
                None => {
                    self.emitted.remove(&k);
                }
            }
        }
        consolidate(out)
    }
}

/// Accumulate a stream of output deltas into a canonical multiset (record → multiplicity), for
/// comparing an incremental run against a from-scratch reference.
pub fn accumulate<D: Ord + Clone>(batches: &[Vec<(D, Diff)>]) -> Vec<(D, Diff)> {
    let mut all: Vec<(D, Diff)> = Vec::new();
    for b in batches {
        all.extend(b.iter().cloned());
    }
    consolidate(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny deterministic LCG so the randomized stream is reproducible without a `rand` dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 16
        }
        fn range(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    // Reference: sum(amount) grouped by region, computed from scratch over accumulated inputs.
    fn full_join_reduce(
        orders: &[((u32, i64), Diff)],   // (customer, amount)
        custs: &[((u32, String), Diff)], // (customer, region)
    ) -> Vec<((String, i64), Diff)> {
        let orders = consolidate(orders.to_vec());
        let custs = consolidate(custs.to_vec());
        let mut region_sum: BTreeMap<String, i64> = BTreeMap::new();
        for ((c1, amt), do_) in &orders {
            for ((c2, region), dc) in &custs {
                if c1 == c2 {
                    *region_sum.entry(region.clone()).or_insert(0) += amt * do_ * dc;
                }
            }
        }
        let mut out: Vec<((String, i64), Diff)> = Vec::new();
        for (region, s) in region_sum {
            if s != 0 {
                out.push(((region, s), 1));
            }
        }
        consolidate(out)
    }

    #[test]
    fn reduce_incremental_equals_full_recompute_under_random_stream() {
        // A random insert/retract/correct stream on (region, amount); reduce = sum(amount) per region.
        // After every step the accumulated incremental output must equal a from-scratch groupby.
        let mut rng = Lcg(0xF3271C);
        let regions = ["north", "south", "east", "west"];
        let mut reducer = Reduce::new(|m: &BTreeMap<i64, Diff>| m.iter().map(|(v, d)| v * d).sum::<i64>());
        let mut accumulated_input: Vec<((String, i64), Diff)> = Vec::new();
        let mut out_batches: Vec<Vec<((String, i64), Diff)>> = Vec::new();

        for _ in 0..400 {
            // build a small random delta batch
            let mut batch: Vec<((String, i64), Diff)> = Vec::new();
            for _ in 0..(1 + rng.range(4)) {
                let region = regions[rng.range(regions.len() as u64) as usize].to_string();
                let amount = (rng.range(9) as i64) + 1;
                let diff: Diff = if rng.range(3) == 0 { -1 } else { 1 }; // sometimes a retraction
                batch.push(((region, amount), diff));
            }
            accumulated_input.extend(batch.iter().cloned());
            out_batches.push(reducer.step(batch));

            // from-scratch reference over everything seen so far
            let acc = consolidate(accumulated_input.clone());
            let mut ref_sum: BTreeMap<String, i64> = BTreeMap::new();
            for ((region, amount), d) in &acc {
                *ref_sum.entry(region.clone()).or_insert(0) += amount * d;
            }
            // Every region still present in the consolidated input is a NON-EMPTY group, so it emits
            // its sum even when that sum is 0 — matching reduce's semantics (emit per live group).
            let reference: Vec<((String, i64), Diff)> =
                consolidate(ref_sum.into_iter().map(|(r, s)| ((r, s), 1)).collect());

            assert_eq!(accumulate(&out_batches), reference, "incremental != full recompute");
        }
    }

    #[test]
    fn join_reduce_pipeline_batch_equals_incremental_and_saves_work() {
        // Pipeline: join(orders[customer,amount], customers[customer,region]) → sum by region.
        // Feed the SAME data (a) as one batch, (b) as many small deltas — final answers must match,
        // and the incremental reducer must recompute far fewer groups than a full recompute would.
        let mut rng = Lcg(0x9E37);
        let n_cust = 40u32;
        let regions = ["north", "south", "east", "west", "central"];

        // ground-truth data
        let mut orders: Vec<((u32, i64), Diff)> = Vec::new();
        let mut custs: Vec<((u32, String), Diff)> = Vec::new();
        for c in 0..n_cust {
            custs.push(((c, regions[(c as usize) % regions.len()].to_string()), 1));
        }
        for _ in 0..300 {
            let c = rng.range(n_cust as u64) as u32;
            let amt = (rng.range(20) as i64) + 1;
            orders.push(((c, amt), 1));
        }

        // (a) one-shot batch
        let reference = full_join_reduce(&orders, &custs);

        // (b) incremental: customers first (as a batch), then orders trickle in one at a time
        let mut join: Join<u32, i64, String> = Join::new();
        let mut reducer =
            Reduce::new(|m: &BTreeMap<i64, Diff>| m.iter().map(|(v, d)| v * d).sum::<i64>());
        let mut out_batches: Vec<Vec<((String, i64), Diff)>> = Vec::new();

        let cust_deltas: Vec<((u32, String), Diff)> = custs.clone();
        let j0 = join.step(vec![], cust_deltas); // no orders yet → join is empty
        assert!(j0.is_empty());
        out_batches.push(reducer.step(vec![]));

        for o in &orders {
            let jd = join.step(vec![o.clone()], vec![]);
            // reshape join output (customer, amount, region) → (region, amount) for the reducer
            let red_in: Vec<((String, i64), Diff)> =
                jd.iter().map(|((_, amt, region), d)| ((region.clone(), *amt), *d)).collect();
            out_batches.push(reducer.step(red_in));
        }

        assert_eq!(accumulate(&out_batches), reference, "incremental pipeline != one-shot batch");

        // work saved: incremental recomputes vs. a naive full-recompute-every-step baseline.
        // full baseline would recompute all live regions on every one of the 301 steps.
        let steps = 1 + orders.len();
        let full_baseline = steps * regions.len();
        assert!(
            reducer.recomputes < full_baseline,
            "expected incremental work {} < full baseline {}",
            reducer.recomputes,
            full_baseline
        );
        eprintln!(
            "ferric-flow: incremental group recomputes = {}, full-recompute baseline = {} ({:.1}x less work)",
            reducer.recomputes,
            full_baseline,
            full_baseline as f64 / reducer.recomputes as f64
        );
    }

    #[test]
    fn corrections_and_late_data_converge() {
        // A wrong value is written, then corrected (retract old, assert new), out of order.
        // The final aggregate must reflect only the corrected truth — Pathway's "handles corrected data".
        let mut reducer =
            Reduce::new(|m: &BTreeMap<i64, Diff>| m.iter().map(|(v, d)| v * d).sum::<i64>());
        let mut outs = Vec::new();
        outs.push(reducer.step(vec![(("acct".to_string(), 100), 1)])); // wrong: 100
        outs.push(reducer.step(vec![(("acct".to_string(), 5), 1)])); // add a correct 5 → 105
        // correction: that 100 was wrong, it should have been 10
        outs.push(reducer.step(vec![(("acct".to_string(), 100), -1), (("acct".to_string(), 10), 1)]));
        let final_state = accumulate(&outs);
        assert_eq!(final_state, vec![(("acct".to_string(), 15), 1)], "correction did not converge to 5+10");
    }

    #[test]
    fn insert_then_retract_nets_to_nothing() {
        let mut join: Join<u32, i64, String> = Join::new();
        let out1 = join.step(vec![((1, 7), 1)], vec![((1, "x".to_string()), 1)]);
        assert_eq!(out1, vec![((1, 7, "x".to_string()), 1)]);
        // retract both sides
        let out2 = join.step(vec![((1, 7), -1)], vec![((1, "x".to_string()), -1)]);
        assert_eq!(accumulate(&[out1, out2]), vec![], "join did not net to empty after full retraction");
    }
}
