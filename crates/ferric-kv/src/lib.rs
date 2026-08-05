//! # ferric-kv — paged KV with radix prefix sharing
//!
//! The two ideas that make modern LLM serving work, as pure bookkeeping:
//!
//! - **Paged KV** (vLLM's PagedAttention): a sequence's KV lives in fixed-size *blocks* that need not be
//!   contiguous, addressed through a per-sequence block table. Kills fragmentation, removes the
//!   grow-and-copy on every token, and — the part that actually matters — lets two sequences *share* the
//!   blocks of a common prefix.
//! - **Radix prefix sharing** (SGLang's RadixAttention): a tree over token sequences mapping prefixes to
//!   the blocks that already hold their KV. A new request that shares 8,000 tokens of system prompt with
//!   a live one reuses those blocks instead of recomputing them.
//!
//! Ferric had neither. Its cache was one contiguous buffer per session, which cannot share anything and
//! reallocates as it grows.
//!
//! ## Why this crate holds no tensors
//!
//! Everything here is integer bookkeeping: block ids, reference counts, a prefix tree. The storage those
//! ids describe belongs to the caller — a GPU buffer natively, a JS array in a browser, a `Vec` in a
//! test. That separation is deliberate and it is what lets the interesting failure modes (a block freed
//! while still shared, a prefix match that crosses a block boundary, a failed append that half-extends a
//! table) be unit tests that run in microseconds on any target.
//!
//! ## The invariant everything else rests on
//!
//! **A block is freed exactly when its last referent releases it, and a shared block is never written
//! in place.** Violate the first and a live sequence reads another's KV; violate the second and one
//! sequence's continuation corrupts another's history. Both produce fluent, wrong output rather than an
//! error, so both are asserted directly rather than inferred.
//!
//! The second half is achieved *structurally* rather than with copy-on-write: [`PagedKv::fork`] shares
//! whole blocks only, which makes writing into a shared block unreachable. See its docs — the COW
//! machinery was written first and removed once its test proved to assert nothing.

#![forbid(unsafe_code)]

mod radix;
pub use radix::{RadixIndex, SharedPrefix};

/// A physical block id — an index into whatever storage the caller owns.
pub type BlockId = u32;

/// Tokens per block. 16 is what vLLM defaults to and the reason is worth keeping: larger blocks waste
/// more on the last partial block of every sequence, smaller ones make the block table longer and the
/// prefix match coarser. It is a caller choice here because the right value depends on typical sequence
/// length, not on this crate.
pub const DEFAULT_BLOCK_TOKENS: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvError {
    /// No free block. The caller must evict or refuse — this crate will not silently overwrite one.
    OutOfBlocks { capacity: usize },
    UnknownSequence(u64),
    /// A logical position past what the sequence has allocated.
    OutOfRange { seq: u64, pos: usize, len: usize },
}

impl core::fmt::Display for KvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            KvError::OutOfBlocks { capacity } => write!(f, "no free KV block ({capacity} total)"),
            KvError::UnknownSequence(s) => write!(f, "unknown sequence {s}"),
            KvError::OutOfRange { seq, pos, len } => {
                write!(f, "sequence {seq}: position {pos} beyond its {len} tokens")
            }
        }
    }
}

impl std::error::Error for KvError {}

/// Fixed-capacity pool of KV blocks with reference counting.
#[derive(Debug)]
pub struct BlockPool {
    refs: Vec<u32>,
    free: Vec<BlockId>,
    block_tokens: usize,
}

impl BlockPool {
    pub fn new(capacity: usize, block_tokens: usize) -> Self {
        assert!(capacity > 0 && block_tokens > 0);
        Self {
            refs: vec![0; capacity],
            // Reverse so allocation hands out low ids first — nothing depends on it, but it makes a
            // failing test's block ids readable.
            free: (0..capacity as BlockId).rev().collect(),
            block_tokens,
        }
    }

    pub fn capacity(&self) -> usize { self.refs.len() }
    pub fn free_blocks(&self) -> usize { self.free.len() }
    pub fn used_blocks(&self) -> usize { self.capacity() - self.free_blocks() }
    pub fn block_tokens(&self) -> usize { self.block_tokens }
    pub fn refcount(&self, b: BlockId) -> u32 { self.refs[b as usize] }

    fn alloc(&mut self) -> Result<BlockId, KvError> {
        let b = self.free.pop().ok_or(KvError::OutOfBlocks { capacity: self.capacity() })?;
        self.refs[b as usize] = 1;
        Ok(b)
    }

    fn retain(&mut self, b: BlockId) { self.refs[b as usize] += 1; }

    /// Drop one reference. Returns `true` if the block became free.
    fn release(&mut self, b: BlockId) -> bool {
        let r = &mut self.refs[b as usize];
        debug_assert!(*r > 0, "releasing block {b} that is already free");
        *r -= 1;
        if *r == 0 {
            self.free.push(b);
            true
        } else {
            false
        }
    }
}

/// One sequence's logical→physical mapping.
#[derive(Debug, Clone, Default)]
pub struct BlockTable {
    blocks: Vec<BlockId>,
    /// Tokens actually written. The last block is usually partial.
    len: usize,
}

impl BlockTable {
    pub fn blocks(&self) -> &[BlockId] { &self.blocks }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Physical location of logical token `pos`, as `(block, offset)`.
    pub fn locate(&self, pos: usize, block_tokens: usize) -> Option<(BlockId, usize)> {
        (pos < self.len).then(|| (self.blocks[pos / block_tokens], pos % block_tokens))
    }
}

/// Paged KV across many sequences, with shared prefixes.
#[derive(Debug)]
pub struct PagedKv {
    pool: BlockPool,
    seqs: std::collections::HashMap<u64, BlockTable>,
    next_seq: u64,
    pub shared_blocks_granted: u64,
}

impl PagedKv {
    pub fn new(capacity_blocks: usize, block_tokens: usize) -> Self {
        Self {
            pool: BlockPool::new(capacity_blocks, block_tokens),
            seqs: std::collections::HashMap::new(),
            next_seq: 1,
            shared_blocks_granted: 0,
        }
    }

    pub fn pool(&self) -> &BlockPool { &self.pool }
    pub fn table(&self, seq: u64) -> Option<&BlockTable> { self.seqs.get(&seq) }
    pub fn block_tokens(&self) -> usize { self.pool.block_tokens }

    pub fn new_sequence(&mut self) -> u64 {
        let id = self.next_seq;
        self.next_seq += 1;
        self.seqs.insert(id, BlockTable::default());
        id
    }

    /// Start a sequence that **shares** `prefix`'s blocks.
    ///
    /// Only whole blocks are shared: a partially-filled block cannot be, because the sharer would write
    /// into the other sequence's live tail. `shared_tokens` reports how many tokens were actually reused,
    /// which is what a caller skips recomputing.
    pub fn fork(&mut self, parent: u64, tokens: usize) -> Result<(u64, usize), KvError> {
        let bt = self.pool.block_tokens;
        let p = self.seqs.get(&parent).ok_or(KvError::UnknownSequence(parent))?;
        let usable = tokens.min(p.len);
        let whole = usable / bt; // partial tail block is NOT shared
        let blocks: Vec<BlockId> = p.blocks[..whole].to_vec();
        for &b in &blocks { self.pool.retain(b); }
        self.shared_blocks_granted += blocks.len() as u64;
        let id = self.next_seq;
        self.next_seq += 1;
        self.seqs.insert(id, BlockTable { blocks, len: whole * bt });
        Ok((id, whole * bt))
    }

    /// Make room for `n` more tokens on `seq`.
    ///
    /// # Why there is no copy-on-write here
    ///
    /// vLLM needs COW because it can share a partially-filled block; extending one then requires
    /// privatising it first, or one sequence's continuation overwrites another's history in place.
    ///
    /// [`PagedKv::fork`] shares **whole blocks only**, and that single rule makes the case unreachable: a
    /// sequence that shares blocks always has a length that is a multiple of the block size, so its next
    /// append starts a *fresh* block and never writes into a shared one. The cost is at most
    /// `block_tokens − 1` tokens of extra prefill per fork — 15 out of thousands — against an entire
    /// class of silent cross-sequence corruption.
    ///
    /// This was written with COW first. It was removed after its test turned out to assert nothing and
    /// the path proved unreachable: dead safety machinery is worse than none, because it implies a
    /// hazard is handled.
    pub fn append(&mut self, seq: u64, n: usize) -> Result<(), KvError> {
        let bt = self.pool.block_tokens;
        let cur_len = self.seqs.get(&seq).ok_or(KvError::UnknownSequence(seq))?.len;
        debug_assert!(
            cur_len % bt == 0
                || self.seqs[&seq].blocks.last().is_none_or(|b| self.pool.refcount(*b) == 1),
            "a shared block is partial — fork's whole-block rule has been broken and COW is now required"
        );
        let target = cur_len + n;
        let needed = target.div_ceil(bt);
        while self.seqs[&seq].blocks.len() < needed {
            let b = match self.pool.alloc() {
                Ok(b) => b,
                Err(e) => {
                    // Undo the blocks this call added, so a failed append leaves the sequence exactly as
                    // it was. A partially-extended table would be worse than the failure.
                    let t = self.seqs.get_mut(&seq).unwrap();
                    while t.blocks.len() > cur_len.div_ceil(bt) {
                        let b = t.blocks.pop().unwrap();
                        self.pool.release(b);
                    }
                    return Err(e);
                }
            };
            self.seqs.get_mut(&seq).unwrap().blocks.push(b);
        }
        self.seqs.get_mut(&seq).unwrap().len = target;
        Ok(())
    }

    /// Release a sequence. Blocks shared with others survive; the rest return to the pool.
    pub fn free(&mut self, seq: u64) -> usize {
        let Some(t) = self.seqs.remove(&seq) else { return 0 };
        t.blocks.iter().filter(|&&b| self.pool.release(b)).count()
    }

    /// Physical slot for a logical position, for a caller writing or reading KV.
    pub fn locate(&self, seq: u64, pos: usize) -> Result<(BlockId, usize), KvError> {
        let t = self.seqs.get(&seq).ok_or(KvError::UnknownSequence(seq))?;
        t.locate(pos, self.pool.block_tokens)
            .ok_or(KvError::OutOfRange { seq, pos, len: t.len })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_are_recycled_only_when_the_last_referent_lets_go() {
        // Free a block while another sequence still reads it and that sequence starts reading someone
        // else's KV — fluent, wrong output, no error. This is the invariant everything else rests on.
        let mut kv = PagedKv::new(8, 4);
        let a = kv.new_sequence();
        kv.append(a, 8).unwrap();                       // 2 whole blocks
        let (b, shared) = kv.fork(a, 8).unwrap();
        assert_eq!(shared, 8, "both blocks are whole and should be shared");
        assert_eq!(kv.pool().used_blocks(), 2, "forking must not copy");
        for blk in kv.table(a).unwrap().blocks() {
            assert_eq!(kv.pool().refcount(*blk), 2);
        }
        assert_eq!(kv.free(a), 0, "nothing may be freed while the fork holds it");
        assert_eq!(kv.pool().used_blocks(), 2);
        assert_eq!(kv.free(b), 2, "the last referent releases both");
        assert_eq!(kv.pool().free_blocks(), 8);
    }

    #[test]
    fn a_partial_tail_block_is_never_shared() {
        // Sharing a half-written block would let the forked sequence write into the parent's live tail.
        let mut kv = PagedKv::new(8, 4);
        let a = kv.new_sequence();
        kv.append(a, 6).unwrap(); // 1 whole block + 2 tokens
        let (_, shared) = kv.fork(a, 6).unwrap();
        assert_eq!(shared, 4, "only the whole block may be shared, not the partial tail");
    }

    #[test]
    fn a_sharing_sequence_never_writes_into_a_shared_block() {
        // This is what makes copy-on-write unnecessary, and it is worth pinning because the whole design
        // rests on it: because fork shares only WHOLE blocks, a sharing sequence's length is always a
        // multiple of the block size, so its next append starts a fresh block.
        let bt = 4;
        let mut kv = PagedKv::new(16, bt);
        let a = kv.new_sequence();
        kv.append(a, 10).unwrap();                       // 2 whole blocks + a partial tail
        let (b, shared) = kv.fork(a, 10).unwrap();
        assert_eq!(shared % bt, 0, "fork handed over a partial block");
        assert_eq!(kv.table(b).unwrap().len() % bt, 0, "a sharing sequence must start block-aligned");

        let before: Vec<_> = kv.table(b).unwrap().blocks().to_vec();
        kv.append(b, 1).unwrap();
        let after = kv.table(b).unwrap().blocks();
        assert_eq!(&after[..before.len()], &before[..], "an append rewrote a shared block");
        assert_eq!(after.len(), before.len() + 1, "the append should have started a fresh block");
        assert_eq!(kv.pool().refcount(*after.last().unwrap()), 1, "the new tail must be private");
        // And every block still shared with the parent is untouched and still shared.
        for blk in &before { assert_eq!(kv.pool().refcount(*blk), 2); }
    }

    #[test]
    fn running_out_of_blocks_leaves_the_sequence_exactly_as_it_was() {
        // A partially-extended table after a failed append is worse than the failure: the caller believes
        // it has room it does not have.
        let mut kv = PagedKv::new(3, 4);
        let a = kv.new_sequence();
        kv.append(a, 8).unwrap();
        let before = kv.table(a).unwrap().clone();
        let e = kv.append(a, 100).unwrap_err();
        assert!(matches!(e, KvError::OutOfBlocks { .. }));
        let after = kv.table(a).unwrap();
        assert_eq!(after.len(), before.len(), "length moved after a failed append");
        assert_eq!(after.blocks(), before.blocks(), "blocks moved after a failed append");
        assert_eq!(kv.pool().used_blocks(), 2, "a failed append leaked blocks");
    }

    #[test]
    fn locate_maps_logical_positions_to_the_right_slot() {
        let mut kv = PagedKv::new(8, 4);
        let a = kv.new_sequence();
        kv.append(a, 10).unwrap();
        let b = kv.table(a).unwrap().blocks().to_vec();
        assert_eq!(kv.locate(a, 0).unwrap(), (b[0], 0));
        assert_eq!(kv.locate(a, 3).unwrap(), (b[0], 3));
        assert_eq!(kv.locate(a, 4).unwrap(), (b[1], 0));
        assert_eq!(kv.locate(a, 9).unwrap(), (b[2], 1));
        assert!(matches!(kv.locate(a, 10), Err(KvError::OutOfRange { .. })), "past the end must fail");
    }

    #[test]
    fn sharing_actually_saves_memory() {
        // The point of all this. Ten requests behind one long system prompt should cost roughly one
        // prompt's worth of blocks, not ten.
        let (bt, prompt) = (16usize, 512usize);
        let mut kv = PagedKv::new(1024, bt);
        let base = kv.new_sequence();
        kv.append(base, prompt).unwrap();
        let solo = kv.pool().used_blocks();

        for _ in 0..10 {
            let (c, shared) = kv.fork(base, prompt).unwrap();
            assert_eq!(shared, prompt, "the whole prompt should be shareable");
            kv.append(c, 8).unwrap(); // each generates a little
        }
        let total = kv.pool().used_blocks();
        // 10 forks add only their own tails, not 10 copies of the prompt.
        assert!(total < solo * 2, "sharing saved nothing: {solo} -> {total} blocks for 11 sequences");
        assert!(kv.shared_blocks_granted >= 10 * (prompt / bt) as u64);
    }
}
