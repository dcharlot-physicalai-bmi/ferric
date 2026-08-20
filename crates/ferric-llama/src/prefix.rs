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
//! the *forward pass* over those tokens, and the trade is a copy against a prefill.
//!
//! The sizes are arithmetic, **not a measurement taken here**: an 8k prefix on a 24-layer model with
//! 2 KV heads of width 128 in f16 is 24 · 8192 · 2 · 2 · 128 · 2 B ≈ 200 MB to copy. Whether that
//! beats re-prefilling the same 8k tokens is a per-device question this module does not answer;
//! `examples/prefix_tree.rs` measures the end-to-end effect on a real model.
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
use ferric_tensor::kvquant::KvStore;
use ferric_tensor::KvBuf;
use std::collections::HashMap;
use std::sync::Arc;

/// Tokens per match granule. Smaller matches more of a prompt but grows the index.
pub const CHUNK: usize = 16;

/// One cached sequence: its tokens and the KV they produced.
struct Entry {
    tokens: Vec<u32>,
    kv: EntryKv,
}

/// A cached prefix's KV, in whichever representation the cache that produced it uses.
///
/// Kept as one enum rather than two parallel vectors so an entry cannot be half-populated, and so the
/// mismatch case — seeding f32 rows into a quantized cache or the reverse — is a `match` arm that has
/// to be written rather than a `None` that can be ignored. Mis-seeding does not error downstream: the
/// model attends to whatever is in the buffer and produces fluent text.
enum EntryKv {
    F32(Vec<(KvBuf, KvBuf)>),
    Quant(Vec<(KvStore, KvStore)>),
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
        // Both arms clone rather than borrow, for the same reason: the caller's cache keeps growing
        // and BOTH stores append in place, so sharing a buffer would let the caller's next token
        // overwrite what a later request is about to reuse.
        let kv = match cache.kvq_fmt() {
            None => EntryKv::F32(cache.layers().iter()
                .map(|(k, v)| (k.clone_prefix(ctx, keep), v.clone_prefix(ctx, keep))).collect()),
            Some(_) => {
                // A token-GROUPED side cannot hand back an arbitrary prefix: the last group is a
                // quantized block spanning 32 tokens, so a prefix ending mid-group would have to split
                // it. `clone_prefix` returns `None` there rather than rounding down, and the honest
                // response is to not cache this entry at all — storing a silently shorter prefix would
                // make every later hit reuse fewer tokens than it reports.
                let mut rows = Vec::with_capacity(cache.layers_q().len());
                for (k, v) in cache.layers_q() {
                    match (k.clone_prefix(ctx, keep), v.clone_prefix(ctx, keep)) {
                        (Some(kp), Some(vp)) => rows.push((kp, vp)),
                        _ => return,
                    }
                }
                EntryKv::Quant(rows)
            }
        };
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

        // An entry produced by an f32 cache cannot seed a quantized one, or the reverse: the two
        // stores are different buffers with different layouts, and the attention path reads only the
        // one matching `fmt`. Treat a mismatch as a MISS rather than a panic — a process can legitimately
        // change FERRIC_KVQ between runs while an in-memory cache outlives the change, and refusing to
        // reuse is always safe where reusing the wrong representation is silently wrong.
        match (&e.kv, cache.kvq_fmt()) {
            (EntryKv::F32(kv), None) => {
                cache.set_layers(kv.iter()
                    .map(|(k, v)| (k.clone_prefix(ctx, m.tokens), v.clone_prefix(ctx, m.tokens))).collect());
            }
            (EntryKv::Quant(kv), Some(f)) if kv.first().is_none_or(|(k, _)| k.fmt() == f) => {
                let mut rows = Vec::with_capacity(kv.len());
                for (k, v) in kv {
                    match (k.clone_prefix(ctx, m.tokens), v.clone_prefix(ctx, m.tokens)) {
                        (Some(kp), Some(vp)) => rows.push((kp, vp)),
                        // Same reasoning as `insert`: refuse rather than seed a short prefix.
                        _ => { self.misses += 1; return None }
                    }
                }
                cache.set_layers_q(rows);
            }
            _ => { self.misses += 1; return None }
        }
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

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// RadixAttention-style prefix tree
// ═══════════════════════════════════════════════════════════════════════════════════════════════

use std::collections::BTreeMap;

/// Handle for one cached prefix's KV. The runtime keys its own storage by this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EntryId(pub u64);

/// What a lookup found: `len` tokens of the request are already computed, inside `entry`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Match {
    /// Tokens of the request whose KV already exists. Exact, not rounded. `0 < len ≤ entry_len`.
    pub len: usize,
    /// The cache entry holding them — fork this.
    pub entry: EntryId,
    /// That entry's own length. Greater than `len` when the request diverged inside it.
    pub entry_len: usize,
}

impl Match {
    /// Whether the entry is *exactly* the matched prefix, with nothing after it.
    pub fn is_exact(&self) -> bool { self.len == self.entry_len }

    /// Round the match down to a multiple of `granule`, for a runtime that can only fork on block
    /// boundaries. Returns `None` if nothing survives.
    ///
    /// Rounding **up** is the silent-corruption bug: the partial block's KV was computed from
    /// different preceding tokens, so the model attends to someone else's history and still reads
    /// fluent. Only down.
    pub fn floor_to(self, granule: usize) -> Option<Match> {
        assert!(granule > 0, "granule must be positive");
        let len = (self.len / granule) * granule;
        if len == 0 { None } else { Some(Match { len, ..self }) }
    }
}

/// How the runtime should obtain the matched KV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reuse {
    /// Copy the entry's first `rows` positions into a fresh cache. Always correct.
    Fork { entry: EntryId, rows: usize },
    /// The entry *is* the prefix and nobody else holds it, so its buffers can be taken over instead
    /// of copied. The caller must call [`PrefixTree::adopt`] to un-index it first — the sequence will
    /// append to those buffers, so leaving the tree pointing at them would hand a later request KV
    /// whose length no longer matches what the tree recorded.
    Adopt { entry: EntryId, rows: usize },
}

const ROOT: u32 = 0;

#[derive(Debug)]
struct Node {
    /// Tokens on the edge from the parent into this node. Empty only for the root.
    edge: Vec<u32>,
    /// Tokens from the root to *this* node, inclusive of `edge`. Absolute, so a split leaves the
    /// child's depth unchanged.
    depth: usize,
    parent: Option<u32>,
    /// `BTreeMap` rather than `HashMap`: representative selection walks children, and a hash order
    /// would make which entry a request forks from depend on the allocator.
    children: BTreeMap<u32, u32>,
    entry: Option<EntryId>,
}

#[derive(Debug)]
struct EntryMeta {
    node: u32,
    /// The entry's tokens. Kept so a match can be *verified* rather than trusted: a wrong prefix is
    /// undetectable downstream, because the model simply attends to another request's history.
    tokens: Vec<u32>,
    refs: u32,
    last_used: u64,
}

/// **A radix tree over cached prefixes**, reference counted, with an LRU bound in tokens — the
/// general form of the single-branch cache above.
///
/// [`PrefixCache`] answers "have I computed a prefix of this?" against a flat set of entries indexed
/// by [`ferric_kv::RadixIndex`], which is a *trie*: one node per token, no splitting, no reference
/// counting, no bound on memory, and matches rounded down to whole [`CHUNK`]s. That is enough for one
/// long system prompt followed by many turns. It is not enough for the shape an agent loop actually
/// produces, which is a **tree**: a root context, a branch per tool call, a branch per retry, each
/// branch spawning sub-branches that share the branch point but not each other.
///
/// This is that structure. The differences from the trie above are all load-bearing:
///
/// | | [`ferric_kv::RadixIndex`] | [`PrefixTree`] |
/// |---|---|---|
/// | node granularity | one per token | one per **run**; nodes split when requests diverge |
/// | match granularity | whole [`CHUNK`]s (loses up to 15 tokens) | exact tokens |
/// | eviction | none — `forget` is manual and O(nodes) | LRU, automatic |
/// | memory bound | none | **tokens**, which is what KV actually costs |
/// | live sequences | not modelled | reference counted; a pinned entry cannot be evicted |
///
/// Match granularity is exact here rather than block-rounded because Ferric's KV is contiguous and
/// [`ferric_tensor::KvBuf::clone_prefix`] takes an arbitrary row count. A paged runtime that can only
/// fork on block boundaries should floor the match itself — see [`Match::floor_to`].
///
/// ## Why the budget is in tokens
///
/// "Keep 64 entries" bounds nothing: one entry may be 32 tokens and the next 128k. KV is charged per
/// *position*, so the bound has to be per position. [`PrefixTree::tokens_used`] is the sum of the
/// lengths of the live entries — **not** the number of distinct tokens in the tree, because in Ferric
/// two sequences sharing an 8k prefix each hold their own contiguous 8k of KV. Sharing the storage
/// needs paged attention (`ferric_kv::PagedKv` plus a paged kernel), which does not exist yet; until
/// it does, counting the shared prefix once would under-report real memory by exactly the factor the
/// sharing was supposed to save.
///
/// ## What a runtime's `Cache` must support
///
/// Exactly one operation, and Ferric already has it:
///
/// - **fork(rows) → Cache** — a new cache holding a *copy* of the first `rows` positions, with `pos`
///   set to `rows`. For [`crate::qwen3::Cache`] that is
///   `set_layers(layers().map(|(k, v)| (k.clone_prefix(ctx, rows), v.clone_prefix(ctx, rows))))`
///   followed by `cache.pos = rows`.
/// - **truncate is not needed.** Forking at `rows` shorter than the entry subsumes it, and every match
///   this tree returns is a fork at `rows ≤ entry_len`.
/// - **copy-on-write is not available.** [`ferric_tensor::KvBuf::append`] writes in place through an
///   `Arc<wgpu::Buffer>`; two caches sharing one buffer would overwrite each other's history while
///   both still read plausible-looking KV. Forking copies for that reason.
///
/// A fork must reproduce **all** per-sequence state, not just KV. `qwen3::Cache` is only `KvBuf`s, so
/// Dense forks with the two lines above. The hybrid runtimes each carry more — `lfm2` a short-conv
/// window, `qwen35` a per-layer recurrent state, `gemma4` blocks that read another block's KV,
/// `deepseek2` latent KV with asymmetric head widths — and each needs its own `fork`, which is model
/// work this module deliberately does not do. **A runtime with no `fork` must simply not call this**;
/// a partial fork that copies KV and drops the recurrent state emits fluent text and is wrong.
///
/// ## Protocol
///
/// ```text
///   let m = tree.match_prefix(&req);          // pins m.entry so it cannot be evicted under you
///   let cache = match m {
///       Some(m) => runtime.fork(m.entry, m.len),   //  ← the only runtime-side operation
///       None    => Cache::new(cfg),
///   };
///   if let Some(m) = &m { tree.release(m.entry); } // the copy is done; the source may go
///   runtime.prefill(&req[m.len..], &mut cache);
///   let id = tree.insert(&full_tokens);       // returns PINNED — the sequence is still writing
///   ... generate ...
///   tree.release(id);                         // now evictable
///   for gone in tree.take_evicted() { runtime.free(gone); }
/// ```
///
/// Both pins are mandatory and both must be released. [`PrefixTree::pins`] exists so a server can
/// assert it has not leaked one — a leaked pin is not a crash, it is a cache that silently stops
/// evicting and then silently stops inserting.
#[derive(Debug)]
pub struct PrefixTree {
    nodes: Vec<Option<Node>>,
    free: Vec<u32>,
    entries: HashMap<EntryId, EntryMeta>,
    capacity_tokens: usize,
    tokens_used: usize,
    clock: u64,
    next_id: u64,
    evicted: Vec<EntryId>,
    /// Cumulative counters, for a server that has to explain its own hit rate.
    pub hits: u64,
    pub misses: u64,
    pub tokens_saved: u64,
    pub evictions: u64,
    pub refusals: u64,
}

impl PrefixTree {
    /// `capacity_tokens` is the KV budget, in positions. See the note on why it is not entries.
    pub fn new(capacity_tokens: usize) -> PrefixTree {
        assert!(capacity_tokens > 0, "a zero-token cache can never hold anything");
        PrefixTree {
            nodes: vec![Some(Node { edge: Vec::new(), depth: 0, parent: None, children: BTreeMap::new(), entry: None })],
            free: Vec::new(),
            entries: HashMap::new(),
            capacity_tokens,
            tokens_used: 0,
            clock: 0,
            next_id: 1,
            evicted: Vec::new(),
            hits: 0, misses: 0, tokens_saved: 0, evictions: 0, refusals: 0,
        }
    }

    // ---- accessors -------------------------------------------------------------------------

    pub fn capacity_tokens(&self) -> usize { self.capacity_tokens }
    /// Sum of the lengths of the live entries — the quantity the budget bounds.
    pub fn tokens_used(&self) -> usize { self.tokens_used }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    /// Live tree nodes, excluding the root. The radix property is that this stays far below the
    /// token count; a per-token trie would make them equal.
    pub fn nodes(&self) -> usize { self.nodes.iter().filter(|n| n.is_some()).count() - 1 }
    /// Outstanding references on `id`, or `None` if it is not live.
    pub fn refs(&self, id: EntryId) -> Option<u32> { self.entries.get(&id).map(|e| e.refs) }
    /// Total outstanding pins. A server should see this return to 0 when idle.
    pub fn pins(&self) -> u32 { self.entries.values().map(|e| e.refs).sum() }
    pub fn entry_len(&self, id: EntryId) -> Option<usize> { self.entries.get(&id).map(|e| e.tokens.len()) }
    pub fn hit_rate(&self) -> f64 {
        let n = self.hits + self.misses;
        if n == 0 { 0.0 } else { self.hits as f64 / n as f64 }
    }

    fn node(&self, id: u32) -> &Node { self.nodes[id as usize].as_ref().expect("live node") }
    fn node_mut(&mut self, id: u32) -> &mut Node { self.nodes[id as usize].as_mut().expect("live node") }
    fn tick(&mut self) -> u64 { self.clock += 1; self.clock }

    fn alloc(&mut self, edge: Vec<u32>, depth: usize, parent: u32) -> u32 {
        let n = Node { edge, depth, parent: Some(parent), children: BTreeMap::new(), entry: None };
        match self.free.pop() {
            Some(i) => { self.nodes[i as usize] = Some(n); i }
            None => { self.nodes.push(Some(n)); (self.nodes.len() - 1) as u32 }
        }
    }
    fn dealloc(&mut self, id: u32) { self.nodes[id as usize] = None; self.free.push(id); }

    // ---- lookup ----------------------------------------------------------------------------

    /// Longest matching prefix, **without** pinning, touching the LRU, or counting a hit.
    ///
    /// For metrics and assertions. A runtime must use [`PrefixTree::match_prefix`], because between a
    /// non-pinning lookup and the fork that follows it, an insert on another thread may evict the
    /// entry the caller is about to read.
    pub fn peek(&self, tokens: &[u32]) -> Option<Match> {
        let (len, node) = self.walk(tokens)?;
        let entry = self.representative(node);
        let meta = &self.entries[&entry];
        debug_assert_eq!(&meta.tokens[..len], &tokens[..len], "the tree returned a non-matching prefix");
        Some(Match { len, entry, entry_len: meta.tokens.len() })
    }

    /// Longest matching prefix, **pinning** the entry so it survives until [`PrefixTree::release`].
    ///
    /// The pin is the whole point: the fork reads the entry's device buffers, and an insert between
    /// the lookup and the fork could otherwise evict them.
    pub fn match_prefix(&mut self, tokens: &[u32]) -> Option<Match> {
        let Some(m) = self.peek(tokens) else { self.misses += 1; return None };
        let now = self.tick();
        let e = self.entries.get_mut(&m.entry).expect("peek returned a live entry");
        e.refs += 1;
        e.last_used = now;
        self.hits += 1;
        self.tokens_saved += m.len as u64;
        Some(m)
    }

    /// How to obtain the matched KV: copy it, or take it over.
    ///
    /// [`Reuse::Adopt`] is offered only when the entry is exactly the matched prefix **and** the
    /// caller's own pin is the only one — an entry a live sequence is still appending to must be
    /// copied, or the two sequences write the same buffer.
    pub fn plan(&self, m: &Match) -> Reuse {
        let sole = self.entries.get(&m.entry).is_some_and(|e| e.refs <= 1);
        if m.is_exact() && sole {
            Reuse::Adopt { entry: m.entry, rows: m.len }
        } else {
            Reuse::Fork { entry: m.entry, rows: m.len }
        }
    }

    /// Take an entry out of the index so the caller can own its buffers. Returns `false` — and
    /// changes nothing — whenever a copy is required instead, which is any of:
    ///
    /// - **anyone else holds it**: a live sequence is appending to those buffers;
    /// - **the match is partial**: the entry carries positions beyond the ones matched, and adopting
    ///   it would silently extend the request's prefix with tokens it does not contain;
    /// - **the match is stale**: `entry_len` no longer describes the live entry.
    ///
    /// It takes the [`Match`] rather than the [`EntryId`] precisely so the second check is not the
    /// caller's job. Not reported through [`PrefixTree::take_evicted`]: the caller is receiving the
    /// KV, not losing it, and freeing it there would be a double free.
    pub fn adopt(&mut self, m: &Match) -> bool {
        let Some(e) = self.entries.get(&m.entry) else { return false };
        if e.refs > 1 || e.tokens.len() != m.len { return false }
        self.drop_entry(m.entry);
        true
    }

    /// Walk down, returning `(matched tokens, deepest node reached)`.
    ///
    /// The node is returned rather than an entry because the stop point may be a *split* node that
    /// holds no entry of its own — every entry beneath it still agrees with the request for the whole
    /// matched length, which is what makes the representative below valid.
    fn walk(&self, tokens: &[u32]) -> Option<(usize, u32)> {
        let mut cur = ROOT;
        let mut i = 0usize;
        let mut best: Option<(usize, u32)> = None;
        while i < tokens.len() {
            let Some(&child) = self.node(cur).children.get(&tokens[i]) else { break };
            let edge = &self.node(child).edge;
            let c = edge.iter().zip(&tokens[i..]).take_while(|(a, b)| a == b).count();
            debug_assert!(c >= 1, "the child was keyed on tokens[i], so at least one token matches");
            best = Some((i + c, child));
            if c < edge.len() { break }          // diverged inside this run
            i += c;
            cur = child;
        }
        best
    }

    /// Any live entry in `node`'s subtree. Its own if it has one — that is the case where the match
    /// can be adopted rather than copied — otherwise the first descendant in child order.
    ///
    /// Total, because pruning maintains the invariant that every live node has an entry beneath it.
    fn representative(&self, node: u32) -> EntryId {
        let mut cur = node;
        loop {
            if let Some(e) = self.node(cur).entry { return e }
            let (_, &c) = self.node(cur).children.iter().next()
                .expect("invariant: a live node always has an entry in its subtree");
            cur = c;
        }
    }

    // ---- insert ----------------------------------------------------------------------------

    /// Index `tokens` and return a **pinned** handle: the caller is still writing that KV, so the
    /// entry must not be evicted until it calls [`PrefixTree::release`].
    ///
    /// Returns `None` when the entry cannot be admitted — longer than the whole budget, or the pinned
    /// entries alone already fill it. Refusing is deliberate: over-committing the bound to avoid an
    /// awkward return type is how a token budget becomes decorative.
    ///
    /// Re-inserting a sequence that is already indexed returns the **existing** id with one more
    /// reference rather than creating a second entry. A second entry would charge the same tokens
    /// twice and hand out two handles to identical KV.
    pub fn insert(&mut self, tokens: &[u32]) -> Option<EntryId> {
        if tokens.is_empty() { return None }

        // Duplicate first, so an exact re-insert never evicts anything to make room for KV that is
        // already there.
        if let Some((len, node)) = self.walk(tokens) {
            if len == tokens.len() && self.node(node).depth == len {
                if let Some(id) = self.node(node).entry {
                    let now = self.tick();
                    let e = self.entries.get_mut(&id).expect("indexed entry is live");
                    e.refs += 1;
                    e.last_used = now;
                    return Some(id)
                }
            }
        }

        let need = tokens.len();
        if need > self.capacity_tokens { self.refusals += 1; return None }
        let pinned: usize = self.entries.values().filter(|e| e.refs > 0).map(|e| e.tokens.len()).sum();
        if pinned + need > self.capacity_tokens { self.refusals += 1; return None }

        while self.tokens_used + need > self.capacity_tokens {
            let victim = self.entries.iter()
                .filter(|(_, e)| e.refs == 0)
                .min_by_key(|(id, e)| (e.last_used, **id))     // id breaks ties deterministically
                .map(|(id, _)| *id);
            let Some(v) = victim else { break };
            self.drop_entry(v);
            self.evicted.push(v);
            self.evictions += 1;
        }
        debug_assert!(self.tokens_used + need <= self.capacity_tokens, "the pinned check should have guaranteed room");

        let node = self.insert_path(tokens);
        let id = EntryId(self.next_id);
        self.next_id += 1;
        let now = self.tick();
        self.node_mut(node).entry = Some(id);
        self.entries.insert(id, EntryMeta { node, tokens: tokens.to_vec(), refs: 1, last_used: now });
        self.tokens_used += need;
        Some(id)
    }

    /// Walk down creating nodes, **splitting** any run the new sequence diverges inside, and return
    /// the node at depth `tokens.len()`.
    fn insert_path(&mut self, tokens: &[u32]) -> u32 {
        let mut cur = ROOT;
        let mut i = 0usize;
        loop {
            if i == tokens.len() { return cur }
            let head = tokens[i];
            let Some(child) = self.node(cur).children.get(&head).copied() else {
                let leaf = self.alloc(tokens[i..].to_vec(), tokens.len(), cur);
                self.node_mut(cur).children.insert(head, leaf);
                return leaf
            };
            let elen = self.node(child).edge.len();
            let c = self.node(child).edge.iter().zip(&tokens[i..]).take_while(|(a, b)| a == b).count();
            if c == elen { i += c; cur = child; continue }

            // Diverged mid-run. Split `child` into [prefix] -> [tail], which costs ONE new node
            // regardless of how long the run was — the reason this is a radix tree and not a trie.
            let mid_depth = i + c;
            let mid = self.alloc(tokens[i..i + c].to_vec(), mid_depth, cur);
            let tail: Vec<u32> = self.node(child).edge[c..].to_vec();
            let tail_head = tail[0];
            {
                let ch = self.node_mut(child);
                ch.edge = tail;                 // depth is absolute and does NOT change
                ch.parent = Some(mid);
            }
            self.node_mut(mid).children.insert(tail_head, child);
            self.node_mut(cur).children.insert(head, mid);

            if mid_depth == tokens.len() { return mid }
            let leaf = self.alloc(tokens[mid_depth..].to_vec(), tokens.len(), mid);
            let lhead = tokens[mid_depth];
            self.node_mut(mid).children.insert(lhead, leaf);
            return leaf
        }
    }

    // ---- eviction and lifetime ---------------------------------------------------------------

    /// Drop one more reference. Returns `false` if `id` is unknown or already unreferenced.
    pub fn release(&mut self, id: EntryId) -> bool {
        match self.entries.get_mut(&id) {
            Some(e) if e.refs > 0 => { e.refs -= 1; true }
            _ => false,
        }
    }

    /// One more reference, so a second holder can keep the entry alive.
    pub fn acquire(&mut self, id: EntryId) -> bool {
        match self.entries.get_mut(&id) {
            Some(e) => { e.refs += 1; let now = self.clock + 1; self.clock = now; e.last_used = now; true }
            None => false,
        }
    }

    /// Un-index `id` outright. Refuses while it is referenced — a live sequence is reading that KV.
    ///
    /// Not reported through [`PrefixTree::take_evicted`]: the caller asked, so it already knows.
    pub fn remove(&mut self, id: EntryId) -> bool {
        if self.entries.get(&id).is_none_or(|e| e.refs > 0) { return false }
        self.drop_entry(id);
        true
    }

    /// Entries evicted since the last call, so the runtime can free their KV.
    ///
    /// Drained rather than peeked, for the same reason `sched::Scheduler::take_retired` is: KV freed
    /// twice is a bug the engine cannot detect, and KV never freed is a leak that only shows up under
    /// sustained load.
    pub fn take_evicted(&mut self) -> Vec<EntryId> { std::mem::take(&mut self.evicted) }

    fn drop_entry(&mut self, id: EntryId) {
        let Some(meta) = self.entries.remove(&id) else { return };
        self.tokens_used -= meta.tokens.len();
        let node = meta.node;
        debug_assert_eq!(self.node(node).entry, Some(id), "entry/node link went out of sync");
        self.node_mut(node).entry = None;
        self.prune(node);
    }

    /// Delete nodes that no longer lead to any entry, then re-compress.
    ///
    /// Both halves matter. Leaving dead leaves would break the invariant `representative` relies on —
    /// a walk could reach a node with nothing beneath it and claim a match it cannot back. Leaving a
    /// one-child chain would be merely wasteful, but it also means the structure stops being a radix
    /// tree after the first eviction, which is the kind of quiet decay a node count catches and
    /// nothing else does.
    fn prune(&mut self, from: u32) {
        let mut cur = from;
        while cur != ROOT {
            let n = self.node(cur);
            if n.entry.is_some() || !n.children.is_empty() { break }
            let parent = n.parent.expect("only the root has no parent");
            let key = n.edge[0];
            self.node_mut(parent).children.remove(&key);
            self.dealloc(cur);
            cur = parent;
        }
        // `cur` is the deepest surviving node on the path; it may now have exactly one child.
        if cur == ROOT { return }
        let n = self.node(cur);
        if n.entry.is_some() || n.children.len() != 1 { return }
        let child = *n.children.values().next().expect("len == 1");
        let (cedge, cdepth, cchildren, centry) = {
            let c = self.node(child);
            (c.edge.clone(), c.depth, c.children.clone(), c.entry)
        };
        {
            let m = self.node_mut(cur);
            m.edge.extend_from_slice(&cedge);
            m.depth = cdepth;
            m.children = cchildren;
            m.entry = centry;
        }
        let grandkids: Vec<u32> = self.node(cur).children.values().copied().collect();
        for g in grandkids { self.node_mut(g).parent = Some(cur); }
        if let Some(e) = centry { self.entries.get_mut(&e).expect("live entry").node = cur; }
        self.dealloc(child);
    }

    /// The entry an LRU that **ignored references** would take next — the real victim selector with
    /// the `refs == 0` filter removed.
    ///
    /// Test-only, and it exists to keep the reference-counting half of the randomised workload from
    /// being vacuous: "the pinned entry survived" proves nothing on a step where nothing was coming
    /// for it. What has to be shown is that the pin was repeatedly *first in line*.
    #[cfg(test)]
    fn oldest_ignoring_refs(&self) -> Option<EntryId> {
        self.entries.iter().min_by_key(|(id, e)| (e.last_used, **id)).map(|(id, _)| *id)
    }

    /// Every structural invariant, checked from scratch. Test-only, and deliberately re-derived
    /// rather than incrementally maintained — a counter that drifts agrees with itself forever.
    #[cfg(test)]
    fn check(&self) {
        let mut seen_entries = 0usize;
        let mut total_tokens = 0usize;
        let mut stack = vec![(ROOT, 0usize)];
        while let Some((id, expect_depth)) = stack.pop() {
            let n = self.node(id);
            assert_eq!(n.depth, expect_depth + n.edge.len(), "node {id} depth is out of step with its edge");
            if id != ROOT {
                assert!(!n.edge.is_empty(), "node {id} has an empty edge but is not the root");
                assert!(
                    n.entry.is_some() || n.children.len() >= 2,
                    "node {id} is neither an entry nor a branch — the tree is no longer compressed",
                );
            }
            if let Some(e) = n.entry {
                seen_entries += 1;
                let meta = self.entries.get(&e).expect("node points at a live entry");
                assert_eq!(meta.node, id, "entry {e:?} and node {id} disagree");
                assert_eq!(meta.tokens.len(), n.depth, "entry length must equal its node's depth");
                total_tokens += meta.tokens.len();
            }
            for (k, &c) in &n.children {
                assert_eq!(self.node(c).edge[0], *k, "child key does not match its edge head");
                assert_eq!(self.node(c).parent, Some(id), "child parent link is wrong");
                stack.push((c, n.depth));
            }
        }
        assert_eq!(seen_entries, self.entries.len(), "an entry is not reachable from the root");
        assert_eq!(total_tokens, self.tokens_used, "tokens_used drifted from the live entries");
        assert!(self.tokens_used <= self.capacity_tokens, "the token budget was exceeded");
        // Every entry's stored tokens must actually spell out the path to its node.
        for (id, meta) in &self.entries {
            let m = self.peek(&meta.tokens).expect("an indexed entry must match itself");
            assert_eq!(m.len, meta.tokens.len(), "entry {id:?} does not fully match itself");
        }
    }
}

#[cfg(test)]
mod tree_tests {
    use super::*;

    /// Insert and immediately release, for the many tests that do not care about the owner's pin.
    fn put(t: &mut PrefixTree, tokens: &[u32]) -> EntryId {
        let id = t.insert(tokens).expect("insert should have been admitted");
        t.release(id);
        id
    }

    fn lcp(a: &[u32], b: &[u32]) -> usize { a.iter().zip(b).take_while(|(x, y)| x == y).count() }

    // ---- matching ---------------------------------------------------------------------------

    #[test]
    fn a_divergence_at_the_very_first_token_matches_nothing() {
        let mut t = PrefixTree::new(1000);
        put(&mut t, &[1, 2, 3, 4]);
        assert_eq!(t.peek(&[9, 2, 3, 4]), None, "a different first token shares nothing");
        assert_eq!(t.match_prefix(&[9, 2, 3, 4]), None);
        assert_eq!(t.misses, 1);
        t.check();
    }

    #[test]
    fn a_divergence_exactly_at_a_node_boundary_matches_the_whole_node() {
        // The off-by-one that matters: the cached entry ends at exactly the point the request turns
        // away. Reporting len-1 wastes a token of prefill; reporting len+1 hands over KV computed
        // from a token the request does not contain, and the model reads it as fluent.
        let mut t = PrefixTree::new(1000);
        let a = put(&mut t, &[1, 2, 3, 4]);
        put(&mut t, &[1, 2, 3, 4, 5, 6]);          // makes depth 4 a real node boundary

        let m = t.peek(&[1, 2, 3, 4, 9, 9]).expect("the first four tokens are cached");
        assert_eq!(m.len, 4, "matched {} tokens, expected exactly the node", m.len);
        assert_eq!(m.entry, a, "the boundary node's own entry is the one to fork");
        assert_eq!(m.entry_len, 4);
        assert!(m.is_exact(), "the entry ends exactly where the request diverges");
        t.check();
    }

    #[test]
    fn inserting_a_divergence_at_a_node_boundary_adds_a_child_and_splits_nothing() {
        let mut t = PrefixTree::new(1000);
        put(&mut t, &[1, 2, 3, 4]);
        let before = t.nodes();
        put(&mut t, &[1, 2, 3, 4, 5, 6]);          // diverges exactly where the first entry ends
        assert_eq!(t.nodes(), before + 1, "a boundary divergence needs one new node, not a split");
        assert_eq!(t.tokens_used(), 4 + 6);
        t.check();
    }

    #[test]
    fn a_request_that_is_a_strict_prefix_of_a_cached_entry_matches_all_of_itself() {
        // Nothing is left to prefill. The fork is shorter than the entry, which is exactly why the
        // runtime needs fork(rows) and not "hand me the cache".
        let mut t = PrefixTree::new(1000);
        let a = put(&mut t, &[1, 2, 3, 4, 5, 6, 7, 8]);
        let m = t.peek(&[1, 2, 3]).expect("three tokens are cached inside the entry");
        assert_eq!(m.len, 3);
        assert_eq!(m.entry, a);
        assert_eq!(m.entry_len, 8, "the entry is longer than the match");
        assert!(!m.is_exact(), "forking is mandatory here — the entry holds five extra positions");
        assert_eq!(t.plan(&m), Reuse::Fork { entry: a, rows: 3 });
        t.check();
    }

    #[test]
    fn a_cached_entry_that_is_a_strict_prefix_of_the_request_matches_all_of_the_entry() {
        let mut t = PrefixTree::new(1000);
        let a = put(&mut t, &[1, 2, 3]);
        let m = t.peek(&[1, 2, 3, 4, 5]).expect("the entry is a prefix of the request");
        assert_eq!(m.len, 3, "the whole entry, and not one token more");
        assert_eq!(m.entry, a);
        assert!(m.is_exact());
        t.check();
    }

    #[test]
    fn the_longest_of_several_cached_prefixes_wins() {
        let mut t = PrefixTree::new(1000);
        put(&mut t, &[1, 2]);
        let long = put(&mut t, &[1, 2, 3, 4]);
        let m = t.peek(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(m.len, 4, "matched {} — a shorter cached prefix was preferred", m.len);
        assert_eq!(m.entry, long);
        t.check();
    }

    // ---- splitting --------------------------------------------------------------------------

    #[test]
    fn a_divergence_inside_a_run_splits_one_node_and_not_one_per_token() {
        // The radix property. A per-token trie would need 8 + 2 = 10 nodes for these two sequences;
        // a compressed tree needs 3, and that ratio is the whole reason this structure is used.
        let mut t = PrefixTree::new(1000);
        put(&mut t, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(t.nodes(), 1, "one run, one node");
        put(&mut t, &[1, 2, 3, 99, 100]);
        assert_eq!(t.nodes(), 3, "split node + two tails; a trie would have 10");
        t.check();
    }

    #[test]
    fn a_split_does_not_double_count_tokens() {
        // The split node covers tokens that BOTH entries already paid for. Charging it again would
        // make the budget shrink every time two requests diverged.
        let mut t = PrefixTree::new(1000);
        put(&mut t, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(t.tokens_used(), 8);
        put(&mut t, &[1, 2, 3, 99, 100]);
        assert_eq!(t.tokens_used(), 13, "8 + 5; the shared [1,2,3] must not be charged a third time");
        assert_eq!(t.len(), 2, "a split creates a NODE, never an entry");
        t.check();
    }

    #[test]
    fn a_match_landing_on_a_split_point_forks_an_entry_that_really_holds_it() {
        // The split node has no KV of its own — it was never a request. The match must still be
        // backed, by an entry underneath it whose first `len` tokens are the ones matched.
        let mut t = PrefixTree::new(1000);
        let long = put(&mut t, &[1, 2, 3, 4, 5, 6, 7, 8]);
        put(&mut t, &[1, 2, 3, 99, 100]);

        let m = t.peek(&[1, 2, 3, 55]).expect("the shared run is cached");
        assert_eq!(m.len, 3, "only the shared run");
        assert!(!m.is_exact(), "no entry is exactly 3 tokens long, so this must be a fork");
        // Named outright rather than as `m.entry`, which would make half the assertion a copy of the
        // answer. The split node's children are keyed 4 and 99 in a `BTreeMap`, so `representative`
        // descends the 4 branch — the choice is by token order and not by allocation order, and that
        // is the property being pinned here.
        assert_eq!(m.entry, long, "the representative must be chosen in child-key order, deterministically");
        assert_eq!(t.plan(&m), Reuse::Fork { entry: long, rows: 3 });
        assert_eq!(t.entry_len(long), Some(8), "the entry backing the match holds all 3 matched tokens and 5 more");
        t.check();
    }

    #[test]
    fn re_inserting_the_same_tokens_reuses_the_entry_instead_of_charging_twice() {
        let mut t = PrefixTree::new(1000);
        let a = t.insert(&[1, 2, 3, 4]).unwrap();
        let nodes = t.nodes();
        let b = t.insert(&[1, 2, 3, 4]).unwrap();
        assert_eq!(a, b, "an identical sequence must not get a second handle to identical KV");
        assert_eq!(t.tokens_used(), 4, "the same four tokens were charged twice");
        assert_eq!(t.len(), 1);
        assert_eq!(t.nodes(), nodes);
        assert_eq!(t.refs(a), Some(2), "both holders must be counted");
        t.check();
    }

    // ---- reference counting and eviction -----------------------------------------------------

    #[test]
    fn a_referenced_entry_is_never_evicted() {
        // The failure this prevents is a use-after-free on device memory: the runtime is mid-fork,
        // reading the entry's buffers, when an unrelated insert drops them.
        let mut t = PrefixTree::new(20);
        let a = t.insert(&[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]).unwrap();  // stays pinned
        let b = put(&mut t, &[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]);
        assert_eq!(t.tokens_used(), 20);

        let c = put(&mut t, &[30, 31, 32, 33, 34, 35, 36, 37, 38, 39]);
        assert_eq!(t.take_evicted(), vec![b], "the unpinned entry is the one that goes");
        assert_eq!(t.refs(a), Some(1), "the pinned entry survived");
        assert!(t.entry_len(a).is_some() && t.entry_len(c).is_some());
        assert_eq!(t.entry_len(b), None);
        t.check();
    }

    #[test]
    fn insert_refuses_rather_than_overcommitting_when_pins_fill_the_budget() {
        // Over-committing to avoid an awkward return type is how a token budget becomes decorative.
        let mut t = PrefixTree::new(20);
        let a = t.insert(&[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]).unwrap();
        let b = t.insert(&[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]).unwrap();
        assert_eq!(t.pins(), 2);

        assert_eq!(t.insert(&[30, 31, 32, 33, 34, 35, 36, 37, 38, 39]), None, "there is no room to make");
        assert_eq!(t.refusals, 1);
        assert!(t.take_evicted().is_empty(), "a refusal must not have destroyed anything first");
        assert_eq!(t.tokens_used(), 20);
        assert!(t.entry_len(a).is_some() && t.entry_len(b).is_some());

        assert_eq!(t.insert(&(0..21).collect::<Vec<u32>>()), None, "longer than the whole budget");
        assert_eq!(t.refusals, 2);
        t.check();
    }

    #[test]
    fn eviction_is_least_recently_used_under_the_token_budget() {
        let mut t = PrefixTree::new(30);
        let a = put(&mut t, &[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
        let b = put(&mut t, &[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]);
        let c = put(&mut t, &[30, 31, 32, 33, 34, 35, 36, 37, 38, 39]);
        assert_eq!(t.tokens_used(), 30);

        // Touch the OLDEST. If a match did not count as a use, this is the one that would be dropped.
        let m = t.match_prefix(&[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]).unwrap();
        assert_eq!(m.entry, a);
        t.release(a);

        put(&mut t, &[40, 41, 42, 43, 44, 45, 46, 47, 48, 49]);
        assert_eq!(t.take_evicted(), vec![b], "the least recently used entry was not the one evicted");
        assert!(t.entry_len(a).is_some(), "the entry that was just matched must survive");
        assert!(t.entry_len(c).is_some());
        assert_eq!(t.tokens_used(), 30);
        t.check();
    }

    /// Fill a 30-token budget with three 10-token entries a, b, c *in that order*, run `touch` on the
    /// tree, then insert a fourth entry that forces exactly one eviction. Returns `(a, b, c, evicted)`.
    ///
    /// The budget is exactly full on entry to `touch`, so the fourth insert must drop precisely one
    /// entry and the identity of that entry is the whole observable.
    fn victim_after(touch: impl FnOnce(&mut PrefixTree, EntryId, EntryId, EntryId)) -> (EntryId, EntryId, EntryId, Vec<EntryId>, PrefixTree) {
        let mut t = PrefixTree::new(30);
        let a = put(&mut t, &[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
        let b = put(&mut t, &[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]);
        let c = put(&mut t, &[30, 31, 32, 33, 34, 35, 36, 37, 38, 39]);
        assert_eq!(t.tokens_used(), 30, "the budget must be exactly full or nothing is evicted at all");
        touch(&mut t, a, b, c);
        assert_eq!(t.pins(), 0, "the touch must hand every reference back, or the pin decides the victim");
        put(&mut t, &[40, 41, 42, 43, 44, 45, 46, 47, 48, 49]);
        let evicted = t.take_evicted();
        (a, b, c, evicted, t)
    }

    #[test]
    fn acquire_refreshes_recency_so_the_entry_it_touched_is_not_the_next_victim() {
        // `acquire` bumps the LRU clock as well as the reference count, and the reference count
        // alone can never show it: a held entry is unevictable *while* it is held, so the refresh
        // has exactly one observable consequence, and it only appears AFTER the release — in which
        // entry the next insert chooses to drop.
        //
        // So the sequence has to make insert-order and acquire-order name different victims:
        //
        //     insert   a, b, c   ->  last_used  a < b < c   (oldest first)
        //     acquire(a)         ->  last_used  b < c < a   (a is now the newest)
        //     release(a)         ->  a is evictable again, but no longer the oldest
        //     insert d           ->  evicts b.  Without the refresh it evicts a.
        //
        // A control is what makes it a measurement rather than an assertion of the obvious: the same
        // scenario with no touch at all must evict `a`, so the expected victim really does move.
        let (a, ..,  no_touch, _) = victim_after(|_, _, _, _| {});
        assert_eq!(no_touch, vec![a], "control: untouched, the oldest-inserted entry is the victim");

        let (a, b, c, evicted, t) = victim_after(|t, a, _, _| {
            assert!(t.acquire(a), "acquire on a live entry reports success");
            assert_eq!(t.refs(a), Some(1), "acquire must take a reference as well as refresh");
            assert!(t.release(a), "and give it straight back, so only recency differs from the control");
            assert_eq!(t.refs(a), Some(0));
        });
        assert_eq!(evicted, vec![b],
                   "acquire did not refresh recency — the entry it touched was evicted as if untouched");
        assert!(t.entry_len(a).is_some(), "the re-held entry must outlive the two inserted after it");
        assert!(t.entry_len(c).is_some(), "c was never the oldest and had no business being dropped");
        assert_eq!(t.tokens_used(), 30);
        t.check();
    }

    #[test]
    fn acquire_refreshes_recency_on_an_entry_that_is_already_held() {
        // The refresh must not be conditional on the 0 -> 1 transition. A second holder arriving on
        // an entry the first holder has been sitting on for a long time is precisely the agent case:
        // a branch pinned by a decoding sequence, joined by a new request that forks from it. If only
        // the first `acquire` refreshed, that entry would age while it was being actively re-used and
        // would then be evicted the moment the last holder let go.
        let (a, b, c, evicted, t) = victim_after(|t, a, b, c| {
            assert!(t.acquire(a));                       // the long-lived holder: a is newest
            for (id, toks) in [(b, [20u32, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
                               (c, [30, 31, 32, 33, 34, 35, 36, 37, 38, 39])] {
                let m = t.match_prefix(&toks).expect("the entry is cached");
                assert_eq!(m.entry, id);
                assert!(t.release(id));                  // b then c are newer than a again
            }
            assert!(t.acquire(a), "a second holder arrives while the first is still there");
            assert_eq!(t.refs(a), Some(2));
            assert!(t.release(a));
            assert!(t.release(a));
        });
        assert_eq!(evicted, vec![b],
                   "the second acquire did not refresh — the refresh is conditional on the refcount");
        assert!(t.entry_len(a).is_some());
        assert!(t.entry_len(c).is_some());
        t.check();
    }

    #[test]
    fn eviction_frees_exactly_enough_and_no_more() {
        let mut t = PrefixTree::new(30);
        put(&mut t, &[10, 11, 12, 13, 14]);                              // 5
        put(&mut t, &[20, 21, 22, 23, 24]);                              // 5
        put(&mut t, &[30, 31, 32, 33, 34, 35, 36, 37, 38, 39]);          // 10
        assert_eq!(t.tokens_used(), 20);
        put(&mut t, &[40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]);  // 12 -> needs 2 freed
        assert_eq!(t.take_evicted().len(), 1, "freeing 5 was enough; a second eviction is waste");
        assert_eq!(t.tokens_used(), 27);
        t.check();
    }

    #[test]
    fn evictions_are_reported_once_and_only_once() {
        // Reported twice, the engine double-frees device memory; never reported, it leaks.
        let mut t = PrefixTree::new(10);
        put(&mut t, &[1, 2, 3, 4, 5]);
        put(&mut t, &[10, 11, 12, 13, 14]);
        put(&mut t, &[20, 21, 22, 23, 24]);
        assert_eq!(t.take_evicted().len(), 1);
        assert!(t.take_evicted().is_empty(), "the same eviction was reported twice");
        t.check();
    }

    #[test]
    fn releasing_more_than_was_acquired_does_not_underflow() {
        let mut t = PrefixTree::new(100);
        let a = t.insert(&[1, 2, 3]).unwrap();
        assert!(t.release(a));
        assert!(!t.release(a), "an unreferenced entry cannot be released again");
        assert_eq!(t.refs(a), Some(0));
        assert!(!t.release(EntryId(9999)), "an unknown id is not an error, just false");
        t.check();
    }

    #[test]
    fn remove_refuses_while_the_entry_is_held() {
        let mut t = PrefixTree::new(100);
        let a = t.insert(&[1, 2, 3]).unwrap();
        assert!(!t.remove(a), "a live sequence is reading that KV");
        t.release(a);
        assert!(t.remove(a));
        assert!(t.is_empty());
        assert!(t.take_evicted().is_empty(), "an explicit removal is not an eviction to report");
        t.check();
    }

    // ---- structure hygiene -------------------------------------------------------------------

    #[test]
    fn eviction_re_compresses_the_tree_instead_of_leaving_a_chain() {
        // Without the merge the structure stops being a radix tree after the first eviction: the
        // split node survives its second branch and the tree slowly decays into a trie.
        let mut t = PrefixTree::new(1000);
        let a = put(&mut t, &[1, 2, 3, 4, 5, 6]);
        let b = put(&mut t, &[1, 2, 3, 7, 8]);
        assert_eq!(t.nodes(), 3, "split + two tails");

        assert!(t.remove(b));
        assert_eq!(t.nodes(), 1, "the split node and the surviving tail must merge back into one run");
        let m = t.peek(&[1, 2, 3, 4, 5, 6]).expect("the survivor is still findable");
        assert_eq!(m.len, 6);
        assert_eq!(m.entry, a, "the merge moved the entry to a different node and lost track of it");
        t.check();
    }

    #[test]
    fn evicting_everything_leaves_an_empty_tree_and_not_a_skeleton() {
        let mut t = PrefixTree::new(1000);
        let ids: Vec<EntryId> = (0..8u32)
            .map(|i| put(&mut t, &[1, 2, 3, i, i + 50, i + 60]))
            .collect();
        assert!(t.nodes() > 1);
        for id in ids { assert!(t.remove(id)); }
        assert_eq!(t.nodes(), 0, "dead nodes were left behind — a leak the token budget cannot see");
        assert_eq!(t.tokens_used(), 0);
        assert_eq!(t.peek(&[1, 2, 3, 0, 50, 60]), None);
        t.check();
    }

    // ---- the runtime-facing API ---------------------------------------------------------------

    #[test]
    fn adopt_is_offered_only_when_nobody_else_holds_the_entry() {
        // Adopting an entry a live sequence is still appending to would give two sequences one set of
        // buffers. `KvBuf::append` writes in place, so they would overwrite each other and both keep
        // reading plausible KV.
        let mut t = PrefixTree::new(100);
        let owner = t.insert(&[1, 2, 3, 4]).unwrap();          // the writer still holds it
        let m = t.match_prefix(&[1, 2, 3, 4]).unwrap();
        assert!(m.is_exact());
        assert_eq!(t.refs(owner), Some(2));
        assert_eq!(t.plan(&m), Reuse::Fork { entry: owner, rows: 4 }, "a second holder forces a copy");
        assert!(!t.adopt(&m), "adopting under a live writer is the copy-on-write bug");

        t.release(owner);                                       // the writer finished
        assert_eq!(t.plan(&m), Reuse::Adopt { entry: owner, rows: 4 });
        assert!(t.adopt(&m));
        assert!(t.is_empty(), "an adopted entry must leave the index — the caller now mutates it");
        assert_eq!(t.tokens_used(), 0);
        assert!(t.take_evicted().is_empty(), "adoption hands KV over; freeing it would be a double free");
        t.check();
    }

    #[test]
    fn a_partial_match_is_never_adoptable_however_it_is_referenced() {
        let mut t = PrefixTree::new(100);
        let a = put(&mut t, &[1, 2, 3, 4, 5, 6]);
        let m = t.match_prefix(&[1, 2, 3, 9]).unwrap();
        assert_eq!(m.len, 3);
        assert_eq!(t.refs(a), Some(1), "only our own pin");
        assert_eq!(t.plan(&m), Reuse::Fork { entry: a, rows: 3 }, "adopting would silently extend the prefix to 6");
        // And `adopt` re-checks rather than trusting the caller to have consulted `plan`: the entry
        // holds six positions, so handing it over would give the request three tokens of someone
        // else's continuation, which reads as fluent output.
        assert!(!t.adopt(&m), "a partial match must never be adoptable");
        assert!(t.entry_len(a).is_some(), "a refused adoption must not have removed the entry");
        t.check();
    }

    #[test]
    fn adopting_a_stale_match_is_refused() {
        // A Match is a plain Copy value the caller can hold across other operations and can build by
        // hand. Ids are monotonic, so a Match naming an evicted entry resolves to nothing — but a
        // Match whose `entry_len` disagrees with the live entry is not caught by anything except this
        // check, and it is the difference between forking 2 positions and handing over 4.
        let mut t = PrefixTree::new(100);
        let a = put(&mut t, &[1, 2, 3, 4]);
        let stale = Match { len: 2, entry: a, entry_len: 2 };   // claims the entry is 2 long; it is 4
        assert!(stale.is_exact(), "the forged match looks adoptable on its face");
        assert!(!t.adopt(&stale), "adopt trusted entry_len instead of checking the live entry");
        assert!(!t.adopt(&Match { len: 4, entry: EntryId(999), entry_len: 4 }), "unknown entry");
        assert_eq!(t.entry_len(a), Some(4));
        t.check();
    }

    #[test]
    fn flooring_a_match_to_a_block_boundary_only_ever_rounds_down() {
        // For a paged runtime that cannot fork mid-block. Rounding UP is the silent-corruption bug:
        // the partial block's KV was computed from different preceding tokens.
        let m = Match { len: 100, entry: EntryId(7), entry_len: 200 };
        let f = m.floor_to(16).expect("96 tokens survive");
        assert_eq!(f.len, 96);
        assert_eq!(f.entry, EntryId(7));
        assert_eq!(f.entry_len, 200);
        assert_eq!(m.floor_to(128), None, "nothing survives a granule larger than the match");
        assert_eq!(m.floor_to(1).unwrap().len, 100, "a granule of 1 is the token-exact case");
        assert_eq!(Match { len: 16, entry: EntryId(1), entry_len: 16 }.floor_to(16).unwrap().len, 16);
    }

    #[test]
    fn pins_are_visible_so_a_server_can_prove_it_did_not_leak_one() {
        let mut t = PrefixTree::new(100);
        let a = t.insert(&[1, 2, 3, 4]).unwrap();
        assert_eq!(t.pins(), 1);
        let m = t.match_prefix(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(t.pins(), 2);
        t.release(m.entry);
        t.release(a);
        assert_eq!(t.pins(), 0, "an idle server holding pins is a cache that has stopped evicting");
    }

    #[test]
    fn an_empty_request_matches_nothing_and_inserts_nothing() {
        let mut t = PrefixTree::new(100);
        assert_eq!(t.insert(&[]), None);
        assert!(t.is_empty());
        put(&mut t, &[1, 2, 3]);
        assert_eq!(t.peek(&[]), None);
        t.check();
    }

    // ---- differential test against a brute-force reference ------------------------------------

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
    }

    #[test]
    fn matches_agree_with_brute_force_over_a_randomised_branching_workload() {
        // Reasoning about a radix tree's split points is exactly where this gets silently wrong, so
        // the answer is checked against a structure that cannot be wrong: a flat list scanned with a
        // longest-common-prefix loop. Both are driven by the same op stream, including evictions.
        //
        // The workload is the shape an agent loop makes — a few root contexts that themselves share
        // prefixes, each extended by a short tail from a tiny alphabet, so divergence lands inside a
        // run far more often than at a node boundary.
        let roots: [&[u32]; 4] = [
            &[1, 1, 1, 1, 1, 1],
            &[1, 1, 1, 2, 2, 2],
            &[1, 1, 3, 3, 3, 3],
            &[4, 4, 4, 4, 4, 4],
        ];
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let mut t = PrefixTree::new(240);
        let mut live: Vec<(EntryId, Vec<u32>)> = Vec::new();
        let mut checked_mid_run = 0usize;
        // A long-lived pin, standing in for a sequence that is still decoding. It is held for 60 steps,
        // which is long enough for the rest of the cache to turn over: each window is measured below
        // and every one of them evicted 53 to 60 other entries against a 240-token bound. Surviving is
        // not by itself evidence — a step where nothing was coming for the entry proves nothing — so
        // `steps_pin_was_lru` counts the steps where the pin was what an LRU *ignoring references*
        // would have taken first, and that count is what the assertion at the bottom is on.
        let mut long_pin: Option<(EntryId, usize)> = None;
        let mut long_pin_survivals = 0usize;
        let mut steps_pin_was_lru = 0usize;
        let mut pin_ev_start = 0u64;
        let mut pin_window_evs: Vec<u64> = Vec::new();
        // Occupancy against the bound, sampled after every step. `hw` is the high-water mark and
        // `saturated_steps` the number of steps that ended with the budget exactly full.
        let mut hw = 0usize;
        let mut saturated_steps = 0usize;

        for step in 0..4000 {
            match long_pin {
                Some((id, until)) => {
                    assert!(t.entry_len(id).is_some(), "step {step}: a held reference was evicted anyway");
                    long_pin_survivals += 1;
                    if t.oldest_ignoring_refs() == Some(id) { steps_pin_was_lru += 1; }
                    if step >= until {
                        let w = t.evictions - pin_ev_start;
                        pin_window_evs.push(w);
                        assert!(t.release(id)); long_pin = None;
                    }
                }
                None if step % 137 == 0 && !live.is_empty() => {
                    let id = live[step % live.len()].0;
                    assert!(t.acquire(id));
                    pin_ev_start = t.evictions;
                    long_pin = Some((id, step + 60));
                }
                None => {}
            }

            let mut req: Vec<u32> = roots[rng.below(roots.len())].to_vec();
            for _ in 0..=rng.below(8) { req.push(7 + rng.below(3) as u32); }

            // --- the reference ---
            let want = live.iter().map(|(_, s)| lcp(s, &req)).max().filter(|&l| l > 0);

            let peeked = t.peek(&req);
            let got = t.match_prefix(&req);          // the real protocol: this PINS m.entry
            assert_eq!(peeked, got, "step {step}: peek and match_prefix disagreed");
            assert_eq!(got.map(|m| m.len), want, "step {step}: disagreed with brute force on {req:?}");
            if let Some(m) = got {
                let held = &live.iter().find(|(i, _)| *i == m.entry)
                    .unwrap_or_else(|| panic!("step {step}: match named entry {:?}, which is not live", m.entry)).1;
                assert_eq!(lcp(held, &req), m.len, "step {step}: entry {:?} does not hold the matched prefix", m.entry);
                assert_eq!(held.len(), m.entry_len, "step {step}: entry_len is wrong");
                assert!(m.len <= m.entry_len);
                if m.len < m.entry_len && m.len < req.len() { checked_mid_run += 1; }
            }

            // Insert while the match is still pinned, which is the order a runtime forks in — so the
            // entry being read must survive the eviction the insert may trigger.
            let ins = t.insert(&req);
            if let Some(m) = got {
                assert!(t.entry_len(m.entry).is_some(), "step {step}: the pinned source was evicted mid-fork");
                assert!(t.release(m.entry));
            }
            if let Some(id) = ins {
                if !live.iter().any(|(i, _)| *i == id) { live.push((id, req.clone())); }
                assert!(t.release(id));
            }
            for gone in t.take_evicted() { live.retain(|(i, _)| *i != gone); }
            assert_eq!(t.pins(), long_pin.is_some() as u32, "step {step}: a pin leaked");

            let held: usize = live.iter().map(|(_, s)| s.len()).sum();
            hw = hw.max(t.tokens_used());
            if t.tokens_used() == t.capacity_tokens() { saturated_steps += 1; }
            assert_eq!(held, t.tokens_used(), "step {step}: token accounting drifted from the live set");
            assert!(t.tokens_used() <= t.capacity_tokens(), "step {step}: the budget was exceeded");
            // Radix property: every node is an entry or a ≥2-way branch, so nodes ≤ 2·entries − 1.
            assert!(t.nodes() <= 2 * t.len(), "step {step}: {} nodes for {} entries — the tree decayed", t.nodes(), t.len());
            if step % 97 == 0 { t.check(); }
        }

        t.check();

        // ---- the workload's own receipt ------------------------------------------------------
        //
        // Everything below is asserted rather than narrated. The comment that used to sit here
        // recorded "3987 hits / 13 misses, 3778 evictions, 2490 mid-run, 23 entries in 38 nodes and
        // exactly 240 tokens"; re-derived 2026-08-16, not one of those seven numbers reproduced, and
        // the load-bearing one was false — the tree does NOT settle at the bound. A comment stating a
        // measured fact that nothing re-checks is a false receipt, so these are assertions now: if the
        // structure changes, this test goes red and the numbers get re-derived, which is the point.
        //
        // The run is deterministic: the seed is fixed, the LRU victim is chosen with an id tie-break,
        // and node children live in a `BTreeMap`, so nothing depends on `HashMap` iteration order.
        // Confirmed identical across five separate processes (each of which reseeds `RandomState`).
        //
        // ⚠ If you are here because one of these failed: re-derive them, do not relax them. Every
        // number is a consequence of the tree's behaviour on a fixed input, and a change to any of
        // them is a change to what the tree does.
        assert_eq!((t.hits, t.misses), (3989, 11),
                   "4000 lookups, 99.7% of them hits — the workload has to share almost everything, \
                    or the eviction path below is never the thing being tested");
        assert_eq!(t.hits + t.misses, 4000, "one lookup per step and no more");
        assert_eq!(t.tokens_saved, 28961, "tokens whose prefill was skipped");
        assert_eq!(t.evictions, 3780, "the budget is turned over ~16x during the run");
        assert_eq!(t.refusals, 0,
                   "no request here is longer than the budget, so the refusal path is NOT covered by \
                    this test — see insert_refuses_rather_than_overcommitting_when_pins_fill_the_budget");
        assert_eq!(checked_mid_run, 2499,
                   "2499 of the 3989 hits stopped inside a run rather than at a node boundary, which \
                    is the case reasoning gets wrong, so it must be the common one and not a corner");

        // The bound. It is *reached* — 240 is the high-water mark and 387 of the 4000 steps end with
        // the budget exactly full — but the tree does NOT rest there. Eviction frees whole entries and
        // stops the instant the incoming one fits, so the resting occupancy is the bound minus
        // whatever slack the last admitted entry left: 22 entries totalling 230 of 240 tokens here.
        assert_eq!(hw, 240, "the workload never filled the 240-token budget, so eviction was incidental");
        assert_eq!(saturated_steps, 387, "steps that ended with the budget exactly full");
        assert_eq!((t.len(), t.nodes(), t.tokens_used()), (22, 37, 230),
                   "final state: 22 entries in 37 nodes holding 230 of 240 tokens — NOT saturated at rest");

        // The reference-counting half. `long_pin_survivals` alone would be weak: it counts steps under
        // a pin whether or not anything was coming for that entry. `steps_pin_was_lru` counts the ones
        // where the pinned entry was exactly what an LRU ignoring references would have taken first,
        // and each of the 28 completed windows evicted 53-60 *other* entries around it.
        assert_eq!(long_pin_survivals, 1706, "steps spent holding the long-lived pin");
        assert_eq!(steps_pin_was_lru, 517,
                   "on 517 steps the pinned entry was the outright LRU victim and was passed over; if \
                    this reaches 0 the pin is never at risk and the survival assertion means nothing");
        assert_eq!(pin_window_evs.len(), 28, "completed pin windows (the 29th is still held at step 4000)");
        assert_eq!(
            (pin_window_evs.iter().copied().min(), pin_window_evs.iter().copied().max()),
            (Some(53), Some(60)),
            "every 60-step pin window must outlast a full turnover of the rest of the cache",
        );
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use crate::qwen3::Cfg;
    use ferric_tensor::kvquant::KvqFmt;
    use ferric_tensor::Tensor;

    /// `tree_tests` covers the radix structure and needs no GPU. `PrefixCache` itself — insert, seed,
    /// and the representation match between them — had NO test until this module: the 5.17x figure in
    /// its commit message came from an example. These are the paths a wrong answer is silent in, since
    /// seeding the wrong KV makes the model attend to someone else's history and still write fluently.
    fn ctx() -> Option<Arc<Context>> {
        match pollster::block_on(ferric_core::Context::new()) {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => { eprintln!("no GPU context ({e:?}); this test did NOT run"); None }
        }
    }

    fn cfg(n_layer: usize) -> Cfg {
        Cfg {
            n_embd: 128, n_layer, n_head: 2, n_head_kv: 1, head_dim: 64, n_ff: 256,
            n_vocab: 1024, eps: 1e-6, rope_base: 1e4, has_qk_norm: false, qkv_bias: false,
            is_gemma: false, embd_scale: 1.0, sliding_window: 0, sliding_pattern: 0,
            gemma2: false, attn_softcap: 0.0, final_softcap: 0.0,
            swa: vec![false; n_layer], logit_scale: 1.0, post_norms: false, nope_global: false,
            rope_interleaved: false, post_norm_eps: 1e-6, embd_rmsnorm: false,
            yarn_factor: 1.0, yarn_orig_ctx: 0,
        }
    }

    /// Fill a quantized cache with `rows` of synthetic K/V so it can be inserted.
    fn quantized_cache(ctx: &Arc<Context>, c: &Cfg, rows: usize, fmt: KvqFmt, seed: f32) -> Cache {
        let width = 64usize;
        let mk = |o: f32| {
            let v: Vec<f32> = (0..rows * width).map(|i| ((i as f32) * 0.019 + o).sin()).collect();
            KvStore::Block(ferric_tensor::kvquant::QKvCache::from_tensor(
                ctx, &Tensor::from_vec(ctx, &v, &[rows, width]), fmt))
        };
        let mut cache = Cache::with_kvq(c, Some(fmt));
        cache.set_layers_q((0..c.n_layer).map(|l| (mk(seed + l as f32), mk(seed + 100.0 + l as f32))).collect());
        cache.pos = rows;
        cache
    }

    /// A quantized prefix round-trips: inserted from one cache, seeded into another.
    #[test]
    fn a_quantized_prefix_is_stored_and_seeded_back() {
        let Some(ctx) = ctx() else { return };
        let c = cfg(3);
        let tokens: Vec<u32> = (0..CHUNK as u32 * 2).collect();

        let src = quantized_cache(&ctx, &c, tokens.len(), KvqFmt::Q8_0, 1.0);
        let want: Vec<Vec<f32>> = src.layers_q().iter()
            .map(|(k, _)| pollster::block_on(k.dequantize(&ctx).to_vec())).collect();

        let mut pc = PrefixCache::new(4);
        pc.insert(&ctx, &tokens, &src);

        let mut dst = Cache::with_kvq(&c, Some(KvqFmt::Q8_0));
        let hit = pc.seed(&ctx, &tokens, &mut dst).expect("the prefix it just stored must be found");
        assert_eq!(hit.tokens, tokens.len(), "the whole chunk-aligned prefix");
        assert_eq!(dst.pos, tokens.len(), "seed must set pos, or the next token ropes at 0");
        assert_eq!(dst.kvq_fmt(), Some(KvqFmt::Q8_0));

        for (il, (k, _)) in dst.layers_q().iter().enumerate() {
            assert_eq!(k.len(), tokens.len(), "layer {il}: row count");
            let got = pollster::block_on(k.dequantize(&ctx).to_vec());
            let d = got.iter().zip(&want[il]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert_eq!(d, 0.0, "layer {il}: seeded rows differ from the source by {d:e}");
        }
    }

    /// A quantized entry must NOT seed an f32 cache, or the reverse.
    ///
    /// The two stores are different buffers with different layouts and the attention path reads only
    /// the one matching `fmt`, so a cross-seed leaves the reader looking at an empty vector — finite
    /// attention over nothing, fluent output, no error. A miss is the safe answer: refusing to reuse
    /// costs a prefill, reusing the wrong representation costs correctness.
    #[test]
    fn a_prefix_never_seeds_a_cache_of_the_other_representation() {
        let Some(ctx) = ctx() else { return };
        let c = cfg(2);
        let tokens: Vec<u32> = (0..CHUNK as u32 * 2).collect();

        let mut pc = PrefixCache::new(4);
        pc.insert(&ctx, &tokens, &quantized_cache(&ctx, &c, tokens.len(), KvqFmt::Q8_0, 1.0));

        // quantized entry -> f32 cache
        let mut f32_cache = Cache::with_kvq(&c, None);
        assert!(pc.seed(&ctx, &tokens, &mut f32_cache).is_none(),
                "a quantized entry seeded an f32 cache: its f32 vector stays empty and every layer \
                 would attend to nothing while still producing text");
        assert_eq!(f32_cache.pos, 0, "a miss must not move pos");

        // quantized entry -> DIFFERENT quantized format
        let mut other = Cache::with_kvq(&c, Some(KvqFmt::Q4_0));
        assert!(pc.seed(&ctx, &tokens, &mut other).is_none(),
                "a q8_0 entry seeded a q4_0 cache: one buffer cannot hold two block layouts and the \
                 seeded rows would reconstruct wrong");
        assert_eq!(other.pos, 0, "a miss must not move pos");
    }
}
