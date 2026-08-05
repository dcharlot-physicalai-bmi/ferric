//! **Prefix caching** — skip prefill for the part of a prompt that has already been computed.
//!
//! The single largest win in agent and chat serving, and the reason SGLang built RadixAttention. Twenty
//! turns behind one 8,000-token system prompt is twenty prefills of the same 8,000 tokens; with a prefix
//! cache it is one, and the other nineteen copy its KV.
//!
//! ## What is actually saved
//!
//! Not memory — compute. A shared prefix still occupies KV for each sequence here, because Ferric's
//! attention reads a contiguous view and paged attention would need its own kernel. What is skipped is
//! the *forward pass* over those tokens: on a 24-layer model an 8k prefix is on the order of a hundred
//! megabytes of device-to-device copy against **seconds** of prefill.
//!
//! `ferric-kv`'s [`PagedKv`](ferric_kv::PagedKv) is the memory-sharing half and is complete; wiring it in
//! needs a paged-attention kernel, which is a separate piece of work. This is the compute half, which is
//! the larger and more immediately available win.
//!
//! ## Matching is by whole chunks, and that is load-bearing
//!
//! [`ferric_kv::RadixIndex`] reports matches in whole chunks. A prompt agreeing on 100 tokens at a chunk
//! size of 16 reuses 96, not 100 — the seventh chunk's KV was computed from *different* preceding tokens,
//! and reusing it looks completely valid while making the model quietly wrong.

use crate::qwen3::{Cache, Cfg};
use ferric_core::Context;
use ferric_kv::RadixIndex;
use ferric_tensor::KvBuf;
use std::collections::HashMap;
use std::sync::Arc;

/// Tokens per match granule. Smaller matches more of a prompt but grows the index.
pub const CHUNK: usize = 16;

/// One cached sequence: its tokens and the KV they produced.
struct Entry {
    tokens: Vec<u32>,
    kv: Vec<(KvBuf, KvBuf)>,
}

/// What a lookup found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hit {
    /// Tokens whose KV was copied — always a whole number of chunks.
    pub tokens: usize,
    /// Which cached sequence it came from.
    pub seq: u64,
}

/// Cache of computed prefixes, keyed by token sequence.
pub struct PrefixCache {
    index: RadixIndex,
    entries: HashMap<u64, Entry>,
    next: u64,
    capacity: usize,
    pub hits: u64,
    pub misses: u64,
    pub tokens_saved: u64,
}

impl PrefixCache {
    /// `capacity` is the number of cached sequences retained.
    pub fn new(capacity: usize) -> Self {
        Self {
            index: RadixIndex::new(CHUNK),
            entries: HashMap::new(),
            next: 1,
            capacity: capacity.max(1),
            hits: 0,
            misses: 0,
            tokens_saved: 0,
        }
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    pub fn hit_rate(&self) -> f64 {
        let n = self.hits + self.misses;
        if n == 0 { return 0.0 } else { self.hits as f64 / n as f64 }
    }

    /// Remember `tokens` and the KV they produced.
    ///
    /// The cache clones the KV rather than borrowing it, because the caller's cache keeps growing and
    /// `KvBuf::append` writes in place — sharing the buffer would let the caller's next token overwrite
    /// what a later request is about to reuse.
    pub fn insert(&mut self, ctx: &Arc<Context>, tokens: &[u32], cache: &Cache) {
        let keep = (tokens.len() / CHUNK) * CHUNK;
        if keep == 0 { return; }
        if self.entries.len() >= self.capacity {
            // Evict the oldest. A frequency policy would be better, but an arbitrary one that is honest
            // beats a clever one that is untested; `hits` is recorded so this can be revisited on data.
            if let Some(&old) = self.entries.keys().min() {
                self.entries.remove(&old);
                self.index.forget(old);
            }
        }
        let id = self.next;
        self.next += 1;
        let kv: Vec<(KvBuf, KvBuf)> = cache
            .layers()
            .iter()
            .map(|(k, v)| (k.clone_prefix(ctx, keep), v.clone_prefix(ctx, keep)))
            .collect();
        // Chunk ids are positional here — this index is being used for its longest-prefix query, not for
        // block addressing, so the "block id" is just the chunk's ordinal.
        let chunks: Vec<u32> = (0..(keep / CHUNK) as u32).collect();
        self.index.insert(&tokens[..keep], &chunks, id);
        self.entries.insert(id, Entry { tokens: tokens[..keep].to_vec(), kv });
    }

    /// Seed `cache` with the longest cached prefix of `tokens`.
    ///
    /// Returns the hit, if any. The caller must then prefill only `tokens[hit.tokens..]` — and must set
    /// `cache.pos` from the returned count, which `seed` does.
    pub fn seed(&mut self, ctx: &Arc<Context>, tokens: &[u32], cache: &mut Cache) -> Option<Hit> {
        let m = match self.index.lookup(tokens) {
            Some(m) if m.tokens > 0 => m,
            _ => { self.misses += 1; return None }
        };
        let Some(e) = self.entries.get(&m.seq) else {
            // Indexed but evicted — a stale entry. Drop it so the next lookup does not pay for it again.
            self.index.forget(m.seq);
            self.misses += 1;
            return None;
        };
        // Belt and braces: the index says these tokens match, so verify it rather than trust it. A wrong
        // prefix here is undetectable downstream — the model simply attends to someone else's history.
        debug_assert_eq!(&e.tokens[..m.tokens], &tokens[..m.tokens], "radix returned a non-matching prefix");
        if e.tokens.len() < m.tokens || e.tokens[..m.tokens] != tokens[..m.tokens] {
            self.misses += 1;
            return None;
        }

        let seeded: Vec<(KvBuf, KvBuf)> = e
            .kv
            .iter()
            .map(|(k, v)| (k.clone_prefix(ctx, m.tokens), v.clone_prefix(ctx, m.tokens)))
            .collect();
        cache.set_layers(seeded);
        cache.pos = m.tokens;
        self.hits += 1;
        self.tokens_saved += m.tokens as u64;
        Some(Hit { tokens: m.tokens, seq: m.seq })
    }

    /// Fresh cache, seeded if possible. Returns it and the number of tokens already computed.
    pub fn cache_for(&mut self, ctx: &Arc<Context>, cfg: &Cfg, tokens: &[u32]) -> (Cache, usize) {
        let mut c = Cache::new(cfg);
        let n = self.seed(ctx, tokens, &mut c).map(|h| h.tokens).unwrap_or(0);
        (c, n)
    }
}
