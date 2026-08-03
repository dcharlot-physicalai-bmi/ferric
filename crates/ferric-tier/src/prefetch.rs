//! Overlapped layer streaming: read layer *L+1* while the caller is still using layer *L*.
//!
//! This is the single largest performance win available in a layer-streaming engine. kimi-k3-in-c
//! measured its I/O at 41–77% of wall clock and estimated real overlap to be worth ~16 s/token against
//! ~3 s/layer of arithmetic — and then **deliberately did not write it**, for a reason worth quoting:
//!
//! > it introduces a concurrent writer to the slot the kernels are reading
//!
//! That is a real hazard in C, where the prefetcher and the compute path both hold raw pointers into a
//! shared slab and nothing but discipline keeps them apart.
//!
//! **In Rust the hazard does not exist, because the buffer is moved rather than shared.** A read request
//! hands ownership of an empty `Vec<u8>` to the worker; the worker fills it and hands ownership back. At
//! no point do two threads hold the same buffer, so there is no concurrent writer *by construction* —
//! not by convention, and not by a lock. The borrow checker enforces what the C version could only
//! promise, which is why this crate can ship the overlap that motivated it.
//!
//! The access order makes the prediction exact rather than speculative: a transformer walks
//! `0, 1, ... N-1, 0, 1, ...` forever, so "the next layer" is known with certainty, not guessed. A
//! misprediction (the caller binds out of order) is still *correct* — the in-flight buffer is reclaimed
//! and the read redone — it just wastes one read's bandwidth.

use crate::{Backing, LayerDesc, LayerPlan, Tier, TierError};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

struct Req {
    layer: u32,
    offset: u64,
    len: usize,
    buf: Vec<u8>,
}

struct Done {
    layer: u32,
    /// **Always returned, even on failure.** If the worker dropped the buffer on an error path, every I/O
    /// error would also leak a slot and the ring would silently shrink to nothing over a long run.
    buf: Vec<u8>,
    result: Result<(), TierError>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefetchStats {
    pub binds: u64,
    pub pinned_hits: u64,
    /// Served from the buffer the caller most recently bound.
    pub ring_hits: u64,
    /// Served by a read that was **already in flight** when the bind arrived — the overlap working.
    pub prefetch_hits: u64,
    /// Had to issue and wait for a read on the calling thread.
    pub sync_reads: u64,
    /// Prefetched a layer the caller then did not ask for. Wasted bandwidth, never wrong results.
    pub mispredicts: u64,
}

impl PrefetchStats {
    /// Fraction of binds that did not block the caller on a fresh read.
    pub fn overlap_rate(&self) -> f64 {
        if self.binds == 0 { return 0.0; }
        (self.pinned_hits + self.ring_hits + self.prefetch_hits) as f64 / self.binds as f64
    }
}

/// Layer cache with a background reader.
///
/// Same policy as [`crate::LayerCache`] — pinned prefix plus a small ring — and the same invariant: the
/// bytes returned are identical regardless of which tier or thread produced them. `prefetch_identical_
/// to_synchronous` in this module asserts exactly that against the synchronous implementation.
pub struct PrefetchCache {
    plan: LayerPlan,
    layers: Vec<LayerDesc>,
    pinned: Vec<Vec<u8>>,
    pinned_filled: Vec<bool>,
    backing: Arc<dyn Backing + Send + Sync>,
    /// The buffer most recently handed to the caller, and which layer it holds.
    current: Option<(u32, Vec<u8>)>,
    /// The layer the worker is reading right now, if any.
    pending: Option<u32>,
    /// Free buffers. Exactly two exist for the lifetime of the cache: one the caller is using and one
    /// the worker is filling. A pool rather than a single `Option` because both can be free at once (at
    /// startup, and after a misprediction is drained), and a single slot silently drops the second —
    /// which leaves `issue` with nothing to read into and disables the overlap entirely.
    pool: Vec<Vec<u8>>,
    req_tx: Option<Sender<Req>>,
    done_rx: Receiver<Done>,
    worker: Option<JoinHandle<()>>,
    stats: PrefetchStats,
}

impl PrefetchCache {
    /// Build from a resolved plan. Spawns one reader thread.
    ///
    /// Two buffers total: one the caller is using, one the worker is filling. That is the minimum for
    /// overlap and — with a fixed access order — also the maximum that helps, since there is only ever
    /// one layer worth prefetching ahead.
    pub fn new(
        plan: LayerPlan,
        layers: Vec<LayerDesc>,
        backing: Arc<dyn Backing + Send + Sync>,
    ) -> Result<Self, TierError> {
        if !plan.fits(u64::MAX) {
            return Err(TierError::BudgetTooSmall { need: plan.spent, have: u64::MAX });
        }
        let pinned: Vec<Vec<u8>> =
            layers[..plan.npin].iter().map(|l| vec![0u8; l.bytes as usize]).collect();
        let pinned_filled = vec![false; plan.npin];
        let slot = plan.ring_slot as usize;

        let (req_tx, req_rx) = std::sync::mpsc::channel::<Req>();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<Done>();
        let wb = Arc::clone(&backing);
        let worker = std::thread::Builder::new()
            .name("ferric-tier-prefetch".into())
            .spawn(move || {
                // Ownership of `buf` transfers in with the request and out with the response. Nothing is
                // shared, so nothing needs a lock and nothing can race the compute path.
                while let Ok(Req { layer, offset, len, mut buf }) = req_rx.recv() {
                    if buf.len() < len { buf.resize(len, 0); }
                    let result = wb.read_at(offset, &mut buf[..len]);
                    if done_tx.send(Done { layer, buf, result }).is_err() {
                        break; // receiver gone: the cache is being dropped
                    }
                }
            })
            .map_err(|e| TierError::Io(format!("could not spawn prefetch thread: {e}")))?;

        Ok(Self {
            plan,
            layers,
            pinned,
            pinned_filled,
            backing,
            current: None,
            pending: None,
            pool: vec![vec![0u8; slot], vec![0u8; slot]],
            req_tx: Some(req_tx),
            done_rx,
            worker: Some(worker),
            stats: PrefetchStats::default(),
        })
    }

    pub fn plan(&self) -> &LayerPlan { &self.plan }
    pub fn stats(&self) -> PrefetchStats { self.stats }

    /// Fill the pinned prefix on the calling thread.
    pub fn prefill(&mut self) -> Result<(), TierError> {
        for li in 0..self.plan.npin {
            if !self.pinned_filled[li] {
                self.backing.read_at(self.layers[li].offset, &mut self.pinned[li])?;
                self.pinned_filled[li] = true;
            }
        }
        Ok(())
    }

    /// Reclaim whatever the worker is holding, discarding the data but keeping the buffer.
    fn drain_pending(&mut self) {
        if self.pending.take().is_some() {
            if let Ok(d) = self.done_rx.recv() {
                self.pool.push(d.buf);
                self.stats.mispredicts += 1;
            }
        }
    }

    /// Ask the worker to start reading `layer`, if we have a buffer free and the layer actually streams.
    fn issue(&mut self, layer: u32) {
        if self.pending.is_some() { return; }
        let li = layer as usize;
        if li < self.plan.npin || li >= self.layers.len() { return; } // pinned or out of range: nothing to read
        let Some(buf) = self.pool.pop() else { return };
        let d = self.layers[li];
        if let Some(tx) = &self.req_tx {
            if tx.send(Req { layer, offset: d.offset, len: d.bytes as usize, buf }).is_ok() {
                self.pending = Some(layer);
            } else {
                self.pool.push(vec![0u8; self.plan.ring_slot as usize]); // worker died; keep capacity
            }
        }
    }

    /// The layer this walk will want next. Cyclic, because that is what a transformer does.
    fn next_layer(&self, layer: u32) -> u32 {
        let n = self.layers.len() as u32;
        if n == 0 { 0 } else { (layer + 1) % n }
    }

    /// Bind a layer, then start reading the next one in the background.
    pub fn bind(&mut self, layer: u32) -> Result<(&[u8], Tier), TierError> {
        let li = layer as usize;
        let desc = *self
            .layers
            .get(li)
            .ok_or(TierError::OutOfRange(crate::WeightId::layer(layer)))?;
        self.stats.binds += 1;

        // --- pinned ---
        if li < self.plan.npin {
            if !self.pinned_filled[li] {
                self.backing.read_at(desc.offset, &mut self.pinned[li])?;
                self.pinned_filled[li] = true;
            } else {
                self.stats.pinned_hits += 1;
            }
            let nxt = self.next_layer(layer);
            self.issue(nxt);
            return Ok((&self.pinned[li], Tier::Pinned));
        }

        // --- already in hand ---
        if matches!(self.current, Some((l, _)) if l == layer) {
            self.stats.ring_hits += 1;
            let nxt = self.next_layer(layer);
            self.issue(nxt);
            let n = desc.bytes as usize;
            return Ok((&self.current.as_ref().unwrap().1[..n], Tier::Cached));
        }

        // --- in flight: this is the overlap paying off ---
        if self.pending == Some(layer) {
            self.pending = None;
            let d = self
                .done_rx
                .recv()
                .map_err(|_| TierError::Io("prefetch worker vanished".into()))?;
            // Recycle the buffer FIRST, so an I/O error cannot cost us a slot.
            let recycled = self.current.take().map(|(_, b)| b);
            match d.result {
                Ok(()) => {
                    self.current = Some((layer, d.buf));
                    if let Some(r) = recycled { self.pool.push(r); }
                    self.stats.prefetch_hits += 1;
                    let nxt = self.next_layer(layer);
                    self.issue(nxt);
                    let n = desc.bytes as usize;
                    return Ok((&self.current.as_ref().unwrap().1[..n], Tier::Cached));
                }
                Err(e) => {
                    // Both buffers survive the failure; only the data is discarded.
                    self.pool.push(d.buf);
                    if let Some(r) = recycled { self.pool.push(r); }
                    return Err(e);
                }
            }
        }

        // --- miss: the caller went somewhere we did not predict ---
        self.drain_pending();
        let mut buf = self
            .pool
            .pop()
            .or_else(|| self.current.take().map(|(_, b)| b))
            .unwrap_or_else(|| vec![0u8; self.plan.ring_slot as usize]);
        let n = desc.bytes as usize;
        if buf.len() < n { buf.resize(n, 0); }
        // Read into a buffer we exclusively own. It is not published as `current` until it succeeds, so a
        // failure cannot leave a slot claiming a layer whose bytes were only partly written — the same
        // trap LayerCache::bind guards, made structural here by ownership.
        match self.backing.read_at(desc.offset, &mut buf[..n]) {
            Ok(()) => {}
            Err(e) => { self.pool.push(buf); return Err(e); }
        }
        if let Some((_, old)) = self.current.take() { self.pool.push(old); }
        self.current = Some((layer, buf));
        self.stats.sync_reads += 1;
        let nxt = self.next_layer(layer);
        self.issue(nxt);
        Ok((&self.current.as_ref().unwrap().1[..n], Tier::Backing))
    }
}

impl Drop for PrefetchCache {
    fn drop(&mut self) {
        // Close the request channel so the worker's `recv` returns Err and its loop exits, then join.
        // Detaching instead would let a worker outlive the `Arc<dyn Backing>` it borrows in tests and
        // turn a clean shutdown into an intermittent one.
        self.req_tx.take();
        if let Some(h) = self.worker.take() { let _ = h.join(); }
    }
}

impl std::fmt::Debug for PrefetchCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefetchCache")
            .field("plan", &self.plan)
            .field("pending", &self.pending)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{plan_layers, LayerCache};
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Synth {
        reads: AtomicU64,
        delay: std::time::Duration,
    }
    impl Synth {
        fn new(delay_ms: u64) -> Arc<Self> {
            Arc::new(Self {
                reads: AtomicU64::new(0),
                delay: std::time::Duration::from_millis(delay_ms),
            })
        }
        fn byte_at(off: u64) -> u8 {
            let x = off.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            ((x >> 29) ^ x) as u8
        }
    }
    impl Backing for Synth {
        fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), TierError> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            if !self.delay.is_zero() { std::thread::sleep(self.delay); }
            for (i, b) in dst.iter_mut().enumerate() {
                *b = Synth::byte_at(offset + i as u64);
            }
            Ok(())
        }
    }

    fn shape(n: usize, bytes: u64) -> Vec<LayerDesc> {
        (0..n).map(|i| LayerDesc { offset: i as u64 * bytes, bytes }).collect()
    }

    #[test]
    fn prefetch_is_byte_identical_to_synchronous() {
        // The invariant that matters most: adding a thread must not change a single byte. If this ever
        // fails, the overlap is not free and must not ship.
        let layers = shape(24, 1024);
        let budget = 6 * 1024 + 2048;
        let b = Synth::new(0);

        let mut sync = LayerCache::with_budget(layers.clone(), budget, 0, 64).unwrap();
        let plan = plan_layers(&layers, budget, 0, 64);
        let mut pre = PrefetchCache::new(plan, layers, Arc::clone(&b) as Arc<dyn Backing + Send + Sync>).unwrap();

        for _tok in 0..3 {
            for l in 0..24u32 {
                let a = sync.bind(l, &*b).unwrap().0.to_vec();
                let z = pre.bind(l).unwrap().0.to_vec();
                assert_eq!(a, z, "layer {l}: prefetching changed the bytes");
            }
        }
    }

    #[test]
    fn the_overlap_actually_engages_on_a_cyclic_walk() {
        // Deterministic proof that reads are being issued ahead of demand: `prefetch_hits` counts binds
        // served by a read that was ALREADY IN FLIGHT. No wall clock involved, so this cannot flake.
        let layers = shape(16, 512);
        let b = Synth::new(0);
        let plan = plan_layers(&layers, 4 * 512 + 1024, 0, 64);
        let mut pre = PrefetchCache::new(plan, layers, Arc::clone(&b) as Arc<dyn Backing + Send + Sync>).unwrap();
        pre.prefill().unwrap();
        for _tok in 0..4 {
            for l in 0..16u32 { pre.bind(l).unwrap(); }
        }
        let s = pre.stats();
        assert!(s.prefetch_hits > 0, "no bind was served by an in-flight read: {s:?}");
        // On a pure cyclic walk every streamed layer should be predicted, so synchronous reads should be
        // rare — essentially just the first one after each wrap-around discontinuity.
        assert!(
            s.prefetch_hits > s.sync_reads,
            "prediction is not paying: {} prefetch hits vs {} sync reads",
            s.prefetch_hits, s.sync_reads
        );
        assert!(s.overlap_rate() > 0.9, "overlap rate {:.3} too low", s.overlap_rate());
    }

    #[test]
    fn out_of_order_access_is_correct_and_merely_wasteful() {
        // A misprediction must cost bandwidth, never correctness.
        let layers = shape(12, 256);
        let b = Synth::new(0);
        let plan = plan_layers(&layers, 2 * 256 + 512, 0, 64);
        let mut pre = PrefetchCache::new(plan, layers, Arc::clone(&b) as Arc<dyn Backing + Send + Sync>).unwrap();
        // Deliberately adversarial order: always jump away from whatever was predicted.
        let order = [11u32, 3, 9, 5, 11, 2, 8, 4, 10, 6];
        for &l in &order {
            let got = pre.bind(l).unwrap().0.to_vec();
            let want: Vec<u8> = (0..256u64).map(|i| Synth::byte_at(l as u64 * 256 + i)).collect();
            assert_eq!(got, want, "layer {l} wrong under adversarial access order");
        }
        assert!(pre.stats().mispredicts > 0, "the adversarial order should have mispredicted");
    }

    #[test]
    fn a_failing_read_returns_its_buffer_and_stays_usable() {
        // If a failed read dropped its buffer, every I/O error would also leak a slot and the ring would
        // silently shrink to nothing over a long run — a slow leak that looks like a performance
        // regression rather than a bug.
        struct Flaky { n: AtomicU64 }
        impl Backing for Flaky {
            fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), TierError> {
                let i = self.n.fetch_add(1, Ordering::SeqCst);
                if i % 3 == 1 { return Err(TierError::Io("flaky".into())); }
                for (k, b) in dst.iter_mut().enumerate() { *b = Synth::byte_at(offset + k as u64); }
                Ok(())
            }
        }
        let layers = shape(10, 128);
        let b: Arc<dyn Backing + Send + Sync> = Arc::new(Flaky { n: AtomicU64::new(0) });
        let plan = plan_layers(&layers, 128 + 256, 0, 64);
        let mut pre = PrefetchCache::new(plan, layers, b).unwrap();
        let mut ok = 0;
        for _ in 0..4 {
            for l in 0..10u32 {
                if let Ok((bytes, _)) = pre.bind(l) {
                    let want: Vec<u8> = (0..128u64).map(|i| Synth::byte_at(l as u64 * 128 + i)).collect();
                    assert_eq!(bytes, &want[..], "layer {l} served wrong bytes after an unrelated failure");
                    ok += 1;
                }
            }
        }
        assert!(ok > 10, "cache stopped serving after failures ({ok} successes)");
    }

    /// Overlap is worth **min(read, compute) per layer** — and with zero compute it is worth nothing.
    ///
    /// This test earns its place by having first FAILED in a way that looked like success. Binding in a
    /// tight loop with no work between binds produced 11 prefetch hits and **zero** wall-clock benefit
    /// (114 ms vs 118 ms): the caller issued a read and then immediately blocked on it, so the prefetch
    /// moved the wait without removing it. Hit counters said the machinery worked; the clock said it
    /// bought nothing. Both were true.
    ///
    /// Real inference does arithmetic between binds, and that arithmetic is what the read hides behind —
    /// kimi-k3-in-c sized the opportunity as ~16 s/token of I/O against ~3 s/layer of compute. So the
    /// honest model is:
    ///
    /// ```text
    ///   serial      = N * (read + compute)
    ///   overlapped  = N * max(read, compute) + read
    /// ```
    ///
    /// which is a ~1.8x win at read ≈ compute, and **1.0x when compute is 0**. A prefetcher benchmarked
    /// without compute measures its own overhead and nothing else.
    #[test]
    fn overlap_hides_reads_behind_compute() {
        const N: u32 = 10;
        const READ_MS: u64 = 6;
        const COMPUTE: std::time::Duration = std::time::Duration::from_millis(6);
        let layers = shape(N as usize, 4096);
        let budget = 8192; // one pinned layer, the rest stream

        // Median of 3, per Ferric's own benchmark rule: a single wall-clock sample is not a measurement.
        let mut serial = Vec::new();
        let mut over = Vec::new();
        let mut hits = 0;
        for _ in 0..3 {
            let b = Synth::new(READ_MS);
            let t0 = std::time::Instant::now();
            let mut sync = LayerCache::with_budget(layers.clone(), budget, 0, 64).unwrap();
            for l in 0..N {
                sync.bind(l, &*b).unwrap();
                std::thread::sleep(COMPUTE); // stand-in for the layer's arithmetic
            }
            serial.push(t0.elapsed());

            let b2 = Synth::new(READ_MS);
            let plan = plan_layers(&layers, budget, 0, 64);
            let mut pre =
                PrefetchCache::new(plan, layers.clone(), Arc::clone(&b2) as Arc<dyn Backing + Send + Sync>)
                    .unwrap();
            let t1 = std::time::Instant::now();
            for l in 0..N {
                pre.bind(l).unwrap();
                std::thread::sleep(COMPUTE);
            }
            over.push(t1.elapsed());
            hits = pre.stats().prefetch_hits;
        }
        serial.sort();
        over.sort();
        let (s, o) = (serial[1], over[1]);
        let speedup = s.as_secs_f64() / o.as_secs_f64();
        println!("serial {s:?} vs overlapped {o:?} = {speedup:.2}x ({hits} prefetch hits)");

        assert!(hits > 0, "no reads were issued ahead of demand");
        // Theory says ~1.8x here. Assert only that a clear majority of the read time was hidden, so the
        // test reports a real structural effect rather than tracking scheduler noise.
        assert!(
            speedup > 1.3,
            "overlap only reached {speedup:.2}x ({s:?} -> {o:?}); the reads are not being hidden"
        );
    }
}
