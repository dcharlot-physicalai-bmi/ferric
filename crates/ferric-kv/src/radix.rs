//! Radix prefix index — find the longest already-computed prefix of an incoming request.
//!
//! Paged KV makes sharing *possible*; this makes it *automatic*. A tree over token sequences maps every
//! prefix that has been computed to the sequence holding it, so a new request arriving behind the same
//! 8,000-token system prompt reuses those blocks instead of prefilling them again.
//!
//! ## Why a radix tree and not a hash of the whole prompt
//!
//! A hash answers "have I seen exactly this?" — which is almost never true, because the tail differs on
//! every request. The question that matters is "how much of this have I seen?", and that is a
//! longest-prefix query. Ferric's on-disk `kvstore` hashes whole rendered prefixes and is the right tool
//! for resuming a *session*; this is the different problem of sharing between *concurrent* requests.
//!
//! ## Block granularity is not a detail
//!
//! Matches are reported in whole blocks. A prefix that agrees on 100 tokens with a block size of 16
//! yields 6 blocks (96 tokens), not 100 — the seventh block holds tokens that diverge, and handing it
//! over would share KV computed from *different* preceding tokens. Rounding up here is the subtle version
//! of the copy-on-write bug: everything looks valid and the model is quietly wrong.

use crate::BlockId;
use std::collections::HashMap;

/// What an incoming request can reuse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedPrefix {
    /// Tokens whose KV already exists — always a whole number of blocks.
    pub tokens: usize,
    /// The blocks holding them, in order.
    pub blocks: Vec<BlockId>,
    /// The sequence they came from, so a caller can bump its refcounts.
    pub seq: u64,
}

#[derive(Debug, Default)]
struct Node {
    children: HashMap<u32, usize>,
    /// Set on nodes that end a whole block: the block covering the tokens leading here.
    block: Option<BlockId>,
    seq: Option<u64>,
    /// Requests that matched through this node, for eviction ordering.
    hits: u32,
}

/// Prefix tree over token sequences.
#[derive(Debug)]
pub struct RadixIndex {
    nodes: Vec<Node>,
    block_tokens: usize,
    pub lookups: u64,
    pub tokens_reused: u64,
    pub tokens_requested: u64,
}

impl RadixIndex {
    pub fn new(block_tokens: usize) -> Self {
        assert!(block_tokens > 0);
        Self { nodes: vec![Node::default()], block_tokens, lookups: 0, tokens_reused: 0, tokens_requested: 0 }
    }

    pub fn len(&self) -> usize { self.nodes.len() - 1 }
    pub fn is_empty(&self) -> bool { self.nodes.len() == 1 }

    /// Fraction of requested prefill tokens served from the cache.
    pub fn reuse_rate(&self) -> f64 {
        if self.tokens_requested == 0 { return 0.0; }
        self.tokens_reused as f64 / self.tokens_requested as f64
    }

    /// Record that `seq` has computed KV for `tokens`, held in `blocks`.
    ///
    /// Only whole blocks are inserted. A partial tail is deliberately not indexed: it is still being
    /// written, and handing it to another request would share a block whose contents are not final.
    pub fn insert(&mut self, tokens: &[u32], blocks: &[BlockId], seq: u64) {
        let bt = self.block_tokens;
        let whole = (tokens.len() / bt).min(blocks.len());
        let mut cur = 0usize;
        for b in 0..whole {
            for &t in &tokens[b * bt..(b + 1) * bt] {
                cur = match self.nodes[cur].children.get(&t) {
                    Some(&n) => n,
                    None => {
                        self.nodes.push(Node::default());
                        let n = self.nodes.len() - 1;
                        self.nodes[cur].children.insert(t, n);
                        n
                    }
                };
            }
            // First writer wins: an existing entry is already shared and re-pointing it would orphan
            // whoever is holding it.
            if self.nodes[cur].block.is_none() {
                self.nodes[cur].block = Some(blocks[b]);
                self.nodes[cur].seq = Some(seq);
            }
        }
    }

    /// Longest indexed prefix of `tokens`.
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<SharedPrefix> {
        self.lookups += 1;
        self.tokens_requested += tokens.len() as u64;
        let bt = self.block_tokens;
        let mut cur = 0usize;
        let mut blocks = Vec::new();
        let mut seq = None;
        let mut matched = 0usize;

        'outer: for b in 0..(tokens.len() / bt) {
            let mut probe = cur;
            for &t in &tokens[b * bt..(b + 1) * bt] {
                match self.nodes[probe].children.get(&t) {
                    Some(&n) => probe = n,
                    // A divergence inside this block means the block cannot be reused at all — its KV was
                    // computed from different preceding tokens.
                    None => break 'outer,
                }
            }
            match self.nodes[probe].block {
                Some(blk) => {
                    blocks.push(blk);
                    seq = self.nodes[probe].seq;
                    matched += bt;
                    self.nodes[probe].hits += 1;
                    cur = probe;
                }
                None => break 'outer,
            }
        }

        if matched == 0 { return None; }
        self.tokens_reused += matched as u64;
        Some(SharedPrefix { tokens: matched, blocks, seq: seq.expect("matched implies a seq") })
    }

    /// Forget everything belonging to `seq` — call when its blocks are freed.
    ///
    /// Leaving stale entries is the dangling-pointer version of this design: a later request would be
    /// handed block ids that have since been reallocated to someone else.
    pub fn forget(&mut self, seq: u64) {
        for n in &mut self.nodes {
            if n.seq == Some(seq) {
                n.block = None;
                n.seq = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(range: std::ops::Range<u32>) -> Vec<u32> { range.collect() }

    #[test]
    fn a_shared_system_prompt_is_found_and_the_divergent_tail_is_not() {
        // The case this exists for: many requests behind one long system prompt.
        let bt = 16;
        let mut ix = RadixIndex::new(bt);
        let prompt = toks(0..64);
        ix.insert(&prompt, &[10, 11, 12, 13], 1);

        let mut req = prompt.clone();
        req.extend(toks(900..910)); // a different user turn
        let hit = ix.lookup(&req).expect("should reuse the system prompt");
        assert_eq!(hit.tokens, 64, "the whole shared prompt should match");
        assert_eq!(hit.blocks, vec![10, 11, 12, 13]);
        assert_eq!(hit.seq, 1);
    }

    #[test]
    fn a_block_that_diverges_partway_is_not_reused() {
        // THE subtle bug: rounding a partial match up to a whole block hands over KV computed from
        // DIFFERENT preceding tokens. Everything stays valid-looking and the model is quietly wrong.
        let bt = 16;
        let mut ix = RadixIndex::new(bt);
        ix.insert(&toks(0..32), &[7, 8], 1);

        let mut req = toks(0..16);      // first block identical
        req.extend(toks(16..24));       // second block agrees for 8...
        req.extend(toks(500..508));     // ...then diverges inside it
        let hit = ix.lookup(&req).unwrap();
        assert_eq!(hit.tokens, 16, "only the fully-matching block may be reused");
        assert_eq!(hit.blocks, vec![7]);
    }

    #[test]
    fn nothing_matches_when_the_very_first_token_differs() {
        let mut ix = RadixIndex::new(4);
        ix.insert(&toks(0..8), &[1, 2], 1);
        assert!(ix.lookup(&toks(100..108)).is_none());
    }

    #[test]
    fn a_partial_tail_is_never_indexed() {
        // It is still being written; sharing it would hand over a block whose contents are not final.
        let mut ix = RadixIndex::new(16);
        let t = toks(0..20);            // 1 whole block + 4
        ix.insert(&t, &[5, 6], 1);
        let hit = ix.lookup(&t).unwrap();
        assert_eq!(hit.tokens, 16, "only the whole block should be indexed");
        assert_eq!(hit.blocks, vec![5]);
    }

    #[test]
    fn forgetting_a_sequence_stops_its_blocks_being_handed_out() {
        // Stale entries are dangling pointers: the block ids get reallocated to someone else.
        let mut ix = RadixIndex::new(4);
        ix.insert(&toks(0..8), &[1, 2], 42);
        assert!(ix.lookup(&toks(0..8)).is_some());
        ix.forget(42);
        assert!(ix.lookup(&toks(0..8)).is_none(), "freed blocks were still offered");
    }

    #[test]
    fn the_first_writer_of_a_prefix_keeps_it() {
        // Re-pointing an existing entry would orphan whoever is already holding those blocks.
        let mut ix = RadixIndex::new(4);
        ix.insert(&toks(0..8), &[1, 2], 1);
        ix.insert(&toks(0..8), &[90, 91], 2);
        let hit = ix.lookup(&toks(0..8)).unwrap();
        assert_eq!(hit.blocks, vec![1, 2], "a later insert stole an already-shared prefix");
        assert_eq!(hit.seq, 1);
    }

    #[test]
    fn reuse_rate_reflects_a_realistic_agent_workload() {
        // Many turns behind one system prompt is the shape that makes prefix caching worth having.
        let bt = 16;
        let mut ix = RadixIndex::new(bt);
        let system = toks(0..512);
        ix.insert(&system, &(0..32).collect::<Vec<_>>(), 1);

        for turn in 0..20u32 {
            let mut req = system.clone();
            req.extend((0..40).map(|i| 10_000 + turn * 100 + i));
            let hit = ix.lookup(&req).expect("system prompt should always hit");
            assert_eq!(hit.tokens, 512);
        }
        assert!(ix.reuse_rate() > 0.9, "reuse rate {:.3} — prefix caching is not paying", ix.reuse_rate());
    }

    #[test]
    fn a_longer_indexed_prefix_wins_over_a_shorter_one() {
        let bt = 8;
        let mut ix = RadixIndex::new(bt);
        ix.insert(&toks(0..8), &[1], 1);
        ix.insert(&toks(0..24), &[1, 2, 3], 2);
        let hit = ix.lookup(&toks(0..24)).unwrap();
        assert_eq!(hit.tokens, 24, "the longest match should win, got {}", hit.tokens);
        assert_eq!(hit.blocks, vec![1, 2, 3]);
    }
}
