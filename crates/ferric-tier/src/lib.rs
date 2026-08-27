//! # ferric-tier — the memory hierarchy
//!
//! Run a model larger than memory by streaming weights from a backing store, **where placement decides
//! speed but never results**.
//!
//! Ferric already guarantees bit-reproducible output across *fabrics* (CPU / Metal / Vulkan / WebGPU).
//! This crate extends the same guarantee along a second axis — the memory hierarchy:
//!
//! ```text
//!         deterministic across FABRIC    (ferric-tensor)
//!                       x
//!         deterministic across PLACEMENT (this crate)
//! ```
//!
//! Three independent 2026 projects — kimi-k3-in-c, colibri, and ds4/DwarfStar — converged on this
//! capability and all advertise placement-invariance in their READMEs ("byte-identical output at every
//! budget"). None of them *enforces* it. Ferric does: see `tests/placement_invariance.rs`, which runs the
//! same logical read sequence at a dozen memory budgets and asserts the bytes are identical.
//!
//! ## Two access patterns, two policies — this is the load-bearing design decision
//!
//! It is tempting to write one LRU and use it for everything. That is wrong, and wrong in a way that is
//! invisible until you measure a hit rate of exactly zero:
//!
//! - **Layers are accessed cyclically** (0, 1, ... N-1, 0, 1, ...) — once per token, forever. This is the
//!   textbook LRU pathology: with fewer slots than layers, the scan returns to layer 0 at exactly the
//!   moment layer 0 has become least-recently-used and was evicted. **Hit rate is 0 no matter how much
//!   memory you add.** [`LayerCache`] therefore uses a *pinned prefix + ring*, which yields a
//!   deterministic `npin/n_layers` hit rate where every extra byte buys its fair share.
//! - **Experts are accessed data-dependently** — which ones fire depends on the token. Here recency and
//!   frequency genuinely predict reuse, so [`ExpertCache`] uses hotness-LFU with an LRU tiebreak.
//!
//! ## What "placement never changes results" actually requires
//!
//! Only that a fetch return the same bytes regardless of which tier served it. This crate guarantees that
//! structurally: nothing here quantizes, approximates, skips, or reorders as a function of the budget. The
//! budget selects *where bytes come from*, never *how they are combined*. Arithmetic is the caller's
//! concern, and Ferric's determinism rules apply there (see `docs/FLAGS.md`).
//!
//! ## Everything is testable without a checkpoint
//!
//! I/O lives entirely behind [`Backing`]. The policy core is integer arithmetic. So the interesting
//! failure modes — cyclic-LRU collapse, budget arithmetic, the partial-read corruption trap — are unit
//! tests that run in milliseconds on any machine, including wasm.

#![forbid(unsafe_code)]

mod expert;
/// Splitting work across a heterogeneous fabric: which missing experts cross the bus,
/// and which execute in place. Pure arithmetic over measured bandwidths, so it runs on wasm too.
mod fabric;
/// Real file I/O. Not on wasm, which has no filesystem — the browser supplies its own [`Backing`]
/// (fetch, OPFS, or an in-memory buffer).
#[cfg(not(target_arch = "wasm32"))]
mod file;
/// Disk-backed KV session checkpoints. Filesystem, so not on wasm.
#[cfg(not(target_arch = "wasm32"))]
pub mod kvstore;
mod layer;
/// In-memory and asynchronously-staged backings. No filesystem, so these are the wasm/browser path.
mod staged;
mod plan;
/// Threads, so not on wasm — the browser gets the synchronous [`LayerCache`], which is the same policy
/// without the overlap.
#[cfg(not(target_arch = "wasm32"))]
mod prefetch;
/// Several full copies of one checkpoint, read in parallel and weighted by measured bandwidth.
/// Scoped threads, so not on wasm.
#[cfg(not(target_arch = "wasm32"))]
mod mirror;

pub use expert::{ExpertCache, ExpertStats};
pub use fabric::{additive_ceiling, makespan, split_n, EnergyModel, FabricProfile, Split};
#[cfg(not(target_arch = "wasm32"))]
pub use file::FileBacking;
pub use layer::{LayerCache, LayerStats};
pub use staged::{SliceBacking, StagedBacking};
pub use plan::{align_up, plan_layers, LayerDesc, LayerPlan, RING_SLOTS};
#[cfg(not(target_arch = "wasm32"))]
pub use prefetch::{PrefetchCache, PrefetchStats};
#[cfg(not(target_arch = "wasm32"))]
pub use mirror::{MirroredBacking, MIN_SPLIT_BYTES};

/// Where a weight was served from on a given fetch.
///
/// **Never observable in results — only in latency and in [`LayerStats`]/[`ExpertStats`].** It is
/// reported so callers can tune a budget, never so they can branch on it. Branching on `Tier` to change
/// arithmetic is exactly the bug this crate exists to make impossible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Held for the lifetime of the run; never evicted.
    Pinned,
    /// Present in a cache slot; may be evicted later.
    Cached,
    /// Fetched from the backing store on this call.
    Backing,
}

/// Identifies one streamable weight.
///
/// `expert: None` means a whole-layer weight (the dense trunk); `Some(e)` means one routed expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WeightId {
    pub layer: u32,
    pub expert: Option<u32>,
}

impl WeightId {
    pub fn layer(layer: u32) -> Self { Self { layer, expert: None } }
    pub fn expert(layer: u32, expert: u32) -> Self { Self { layer, expert: Some(expert) } }
}

/// The backing store: byte-exact random reads.
///
/// Implementations must be **side-effect free and deterministic** — the same `(offset, len)` must yield
/// the same bytes on every call, regardless of caching, readahead, or which thread asks. That is the
/// entire basis of placement-invariance; if a `Backing` is not deterministic, nothing above it can be.
///
/// A short read is an error, not a partial success. Returning `Ok` after filling only part of `dst`
/// would leave a slot holding a mix of two weights, which is the silent-corruption failure mode
/// [`LayerCache::bind`] is written to prevent.
pub trait Backing {
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), TierError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TierError {
    /// The backing store could not supply the requested range.
    Io(String),
    /// A read returned fewer bytes than requested. Kept distinct from `Io` because it is the specific
    /// condition that corrupts a cache slot if mishandled.
    ShortRead { want: usize, got: usize },
    /// The budget cannot hold even the minimum working set. Refusing up front beats failing mid-token:
    /// a cache smaller than one step's working set would evict an entry that is still in use.
    BudgetTooSmall { need: u64, have: u64 },
    /// A weight id outside the configured shape.
    OutOfRange(WeightId),
    /// Two devices of a mirror returned different bytes for the same range.
    ///
    /// Kept distinct from `Io` for the same reason `ShortRead` is: this is the specific condition
    /// that produces a buffer of the right length holding bytes from two different versions of the
    /// checkpoint. Every component `Backing` honoured its contract; the composite did not.
    MirrorMismatch(String),
    /// A synchronous read asked for bytes that were never staged.
    ///
    /// Only reachable on the asynchronous path ([`StagedBacking`]): a browser cannot block a sync read on
    /// a fetch, so an un-prefetched range must fail loudly rather than stall or return zeros.
    NotStaged { offset: u64, len: usize },
}

impl core::fmt::Display for TierError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TierError::Io(m) => write!(f, "backing store error: {m}"),
            TierError::ShortRead { want, got } => write!(f, "short read: wanted {want} bytes, got {got}"),
            TierError::BudgetTooSmall { need, have } => {
                write!(f, "budget too small: need {need} bytes, have {have}")
            }
            TierError::OutOfRange(id) => write!(f, "weight out of range: {id:?}"),
            TierError::MirrorMismatch(m) => write!(f, "mirror mismatch: {m}"),
            TierError::NotStaged { offset, len } => write!(
                f, "bytes not staged: [{offset}, {}) — the prefetch missed this range", offset + *len as u64),
        }
    }
}

impl std::error::Error for TierError {}
