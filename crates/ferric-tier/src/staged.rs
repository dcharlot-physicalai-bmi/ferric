//! Backings that need no filesystem: an in-memory slice, and a **staged** one for environments where
//! I/O is asynchronous and reads are not.
//!
//! ## The browser problem, and why staging is the answer rather than a workaround
//!
//! [`crate::Backing::read_at`] is synchronous. Browser `fetch` is asynchronous, and a wasm main thread
//! **cannot block on a future**. Those are irreconcilable in general — you cannot turn an async fetch
//! into a sync read.
//!
//! They are reconcilable *here*, because of the one property this whole crate is built around: **the
//! access order is known in advance.** A transformer walks layers 0, 1, … N−1 and then repeats. So the
//! bytes for step *k+1* can be fetched while step *k* is still computing, and by the time the synchronous
//! read happens they are already in memory.
//!
//! That is the same one-ahead prefetch [`crate::PrefetchCache`] does with a thread. In a browser it is
//! not an optimisation — it is the only way the read can be synchronous at all. [`StagedBacking`] is the
//! seam: `stage()` is called from async code, `read_at()` from sync code, and a read of un-staged bytes
//! is a loud error rather than a stall or a wrong answer.

use crate::{Backing, TierError};
use std::collections::BTreeMap;
use std::sync::Mutex;

/// A backing over bytes already in memory.
///
/// The simplest thing that works everywhere, including wasm: fetch the whole file once, then stream
/// *within* it. That does not reduce peak memory, so it is not the interesting case — but it is the
/// correct fallback when a server does not honour range requests, and it makes the tier usable from a
/// browser in one line.
#[derive(Debug)]
pub struct SliceBacking {
    bytes: Vec<u8>,
}

impl SliceBacking {
    pub fn new(bytes: Vec<u8>) -> Self { Self { bytes } }
    pub fn len(&self) -> u64 { self.bytes.len() as u64 }
    pub fn is_empty(&self) -> bool { self.bytes.is_empty() }
}

impl Backing for SliceBacking {
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), TierError> {
        let end = offset
            .checked_add(dst.len() as u64)
            .ok_or_else(|| TierError::Io("read range overflows u64".into()))?;
        if end > self.len() {
            return Err(TierError::ShortRead {
                want: dst.len(),
                got: self.len().saturating_sub(offset) as usize,
            });
        }
        dst.copy_from_slice(&self.bytes[offset as usize..end as usize]);
        Ok(())
    }
}

/// A backing fed asynchronously and read synchronously.
///
/// Ranges are inserted by [`StagedBacking::stage`] (from async code — a `fetch`, an XHR, a worker
/// message) and served by `read_at` (from the synchronous forward pass). A read that is not covered
/// fails with [`TierError::NotStaged`], which names the range so the caller can see exactly what its
/// prefetch missed.
///
/// **Failing loudly is the whole design.** The alternatives are worse in ways that are hard to diagnose:
/// blocking is impossible on a wasm main thread, and returning zeros would produce a model that runs and
/// is quietly wrong — the same silent-corruption class this crate guards everywhere else.
///
/// Interior state is behind a `Mutex` rather than a `RefCell` so the type is `Sync` and can be handed to
/// the same APIs a file backing uses. On wasm that lock never contends; the uniformity is worth more than
/// the nanoseconds.
#[derive(Debug, Default)]
pub struct StagedBacking {
    /// `offset -> bytes`. A `BTreeMap` so a read can find the range that *contains* it via a range
    /// query, rather than requiring the caller to read back exactly the extents it staged.
    ranges: Mutex<BTreeMap<u64, Vec<u8>>>,
    resident: Mutex<u64>,
    /// Bytes ever staged, for accounting.
    staged_total: Mutex<u64>,
    /// Highest residency observed. A budget claim has to be judged against the PEAK; reporting only the
    /// current figure lets a caller release everything and quote the trough.
    peak: Mutex<u64>,
}

impl StagedBacking {
    pub fn new() -> Self { Self::default() }

    /// Make `bytes` readable at `offset`. Call from async code, before the sync read that needs them.
    pub fn stage(&self, offset: u64, bytes: Vec<u8>) {
        let n = bytes.len() as u64;
        let prev = self.ranges.lock().unwrap().insert(offset, bytes);
        let mut res = self.resident.lock().unwrap();
        *res += n;
        if let Some(p) = prev { *res -= p.len() as u64; }
        let mut pk = self.peak.lock().unwrap();
        if *res > *pk { *pk = *res; }
        *self.staged_total.lock().unwrap() += n;
    }

    /// Bytes ever staged — total fetched traffic, which is the figure that matters over a metered link.
    pub fn staged_total(&self) -> u64 { *self.staged_total.lock().unwrap() }

    /// Drop a staged range. This is what keeps a browser session inside its memory budget: the tier
    /// decides *what* to hold, and the caller releases what the tier evicted.
    pub fn release(&self, offset: u64) {
        if let Some(v) = self.ranges.lock().unwrap().remove(&offset) {
            *self.resident.lock().unwrap() -= v.len() as u64;
        }
    }

    /// Bytes currently staged.
    pub fn resident_bytes(&self) -> u64 { *self.resident.lock().unwrap() }

    /// Highest residency ever observed — the figure a memory budget must be judged against.
    pub fn peak_bytes(&self) -> u64 { *self.peak.lock().unwrap() }

    pub fn is_staged(&self, offset: u64, len: usize) -> bool {
        self.copy_into(offset, &mut vec![0u8; len]).is_ok()
    }

    /// Copy `[offset, offset+dst.len())` out of the staged ranges, **stitching across adjacent ones**.
    ///
    /// Stitching matters because the fetch granularity is not the read granularity: a browser fetching
    /// fixed 1 MB chunks has no idea where a tensor begins, so a read will routinely straddle two chunks.
    /// Requiring a read to sit inside one staged range would push that alignment problem onto every
    /// caller — and the first version did exactly that, which the cross-backing test caught.
    ///
    /// A *gap* between ranges is still an error: partial data is the failure mode this type exists to
    /// prevent.
    fn copy_into(&self, offset: u64, dst: &mut [u8]) -> Result<(), TierError> {
        let r = self.ranges.lock().unwrap();
        let mut pos = offset;
        let mut done = 0usize;
        while done < dst.len() {
            let (&start, buf) = r
                .range(..=pos)
                .next_back()
                .ok_or(TierError::NotStaged { offset, len: dst.len() })?;
            let local = (pos - start) as usize;
            if local >= buf.len() {
                return Err(TierError::NotStaged { offset, len: dst.len() });
            }
            let take = (buf.len() - local).min(dst.len() - done);
            dst[done..done + take].copy_from_slice(&buf[local..local + take]);
            done += take;
            pos += take as u64;
        }
        Ok(())
    }
}

impl Backing for StagedBacking {
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), TierError> {
        self.copy_into(offset, dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize) -> Vec<u8> { (0..n).map(|i| (i % 251) as u8).collect() }

    #[test]
    fn slice_backing_reads_exact_ranges_and_refuses_past_the_end() {
        let b = SliceBacking::new(data(1024));
        let mut got = vec![0u8; 32];
        b.read_at(100, &mut got).unwrap();
        assert_eq!(got, &data(1024)[100..132]);
        assert!(matches!(b.read_at(1000, &mut vec![0u8; 100]), Err(TierError::ShortRead { .. })));
    }

    #[test]
    fn a_read_of_unstaged_bytes_is_a_named_error_not_zeros() {
        // The failure this prevents: a browser prefetch that missed, serving zeros, producing a model
        // that runs and is quietly wrong. The error names the range so the miss is diagnosable.
        let s = StagedBacking::new();
        let e = s.read_at(4096, &mut vec![0u8; 16]).unwrap_err();
        match e {
            TierError::NotStaged { offset, len } => { assert_eq!(offset, 4096); assert_eq!(len, 16); }
            other => panic!("expected NotStaged, got {other:?}"),
        }
    }

    #[test]
    fn staged_bytes_are_served_and_a_read_may_span_a_subrange() {
        // A caller stages whole layer runs but the model reads individual tensors inside them, so a read
        // must resolve to the range that CONTAINS it — not only to an exact match.
        let s = StagedBacking::new();
        s.stage(1000, data(500));
        let mut got = vec![0u8; 64];
        s.read_at(1200, &mut got).unwrap();
        assert_eq!(got, &data(500)[200..264]);
        // A read that runs off the end of the staged range is NOT silently truncated.
        assert!(s.read_at(1400, &mut vec![0u8; 200]).is_err());
    }

    #[test]
    fn release_frees_the_range_and_the_accounting() {
        // This is what holds a browser session to its budget: the tier decides what to keep, the caller
        // releases what it dropped. If accounting drifted, the budget would be fiction.
        let s = StagedBacking::new();
        s.stage(0, data(4096));
        s.stage(4096, data(4096));
        assert_eq!(s.resident_bytes(), 8192);
        s.release(0);
        assert_eq!(s.resident_bytes(), 4096);
        assert!(s.read_at(0, &mut vec![0u8; 8]).is_err(), "released range still readable");
        assert!(s.read_at(4096, &mut vec![0u8; 8]).is_ok());
        s.release(4096);
        assert_eq!(s.resident_bytes(), 0);
        assert_eq!(s.peak_bytes(), 8192, "peak must survive the releases — it is what a budget is judged on");
        s.release(999_999); // releasing something never staged must not corrupt the count
        assert_eq!(s.resident_bytes(), 0);
    }

    #[test]
    fn restaging_the_same_offset_does_not_double_count() {
        let s = StagedBacking::new();
        s.stage(0, data(1000));
        s.stage(0, data(1000));
        assert_eq!(s.resident_bytes(), 1000, "re-staging inflated the resident total");
    }

    #[test]
    fn is_staged_answers_before_the_read_so_a_caller_can_fetch_first() {
        // The prefetch loop asks this to decide whether it must await a fetch — the whole point of the
        // async/sync seam.
        let s = StagedBacking::new();
        assert!(!s.is_staged(0, 16));
        s.stage(0, data(64));
        assert!(s.is_staged(0, 16));
        assert!(s.is_staged(48, 16));
        assert!(!s.is_staged(48, 32), "must not claim bytes past the staged range");
        // Adjacent ranges stitch; a GAP does not.
        s.stage(64, data(64));
        assert!(s.is_staged(48, 32), "adjacent staged ranges should stitch");
        s.stage(200, data(16));
        assert!(!s.is_staged(120, 90), "a gap between ranges must not be readable");
    }

    #[test]
    fn staged_delivers_the_same_bytes_a_slice_backing_would() {
        // Placement-invariance across BACKINGS, not just budgets: an identical logical read must give
        // identical bytes whether it came from memory or from a staged fetch.
        let all = data(8192);
        let mem = SliceBacking::new(all.clone());
        let staged = StagedBacking::new();
        for chunk in 0..4u64 {
            let (o, n) = (chunk * 2048, 2048usize);
            staged.stage(o, all[o as usize..o as usize + n].to_vec());
        }
        for (off, len) in [(0u64, 100usize), (2000, 500), (4095, 2), (6000, 192)] {
            let mut a = vec![0u8; len];
            let mut b = vec![0u8; len];
            mem.read_at(off, &mut a).unwrap();
            staged.read_at(off, &mut b).unwrap();
            assert_eq!(a, b, "backing changed the bytes at ({off}, {len})");
        }
    }
}
