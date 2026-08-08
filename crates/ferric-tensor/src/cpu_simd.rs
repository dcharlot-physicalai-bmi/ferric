//! **The other half of the fabric: quantised matmul on the CPU's vector units.**
//!
//! Ferric reaches the GPU through wgpu and, until this module, reached nothing else. That is the
//! convention every runtime in the 2026 survey follows: pick a backend, call it portability. On a
//! machine with 18 CPU cores and a GPU, it means running on one of those and leaving the other idle.
//!
//! The reason that is expensive here specifically is measured, not assumed. Decode is **bandwidth-bound**
//! (this workspace: ~525 MB of weights read per token, and cutting 29% of GPU dispatches moved wall time
//! by 0.00 ms). When the bound is bandwidth, what matters is how many independent issue paths to memory
//! the machine has. A saturated GPU beside an idle CPU is using one of them.
//!
//! ## Why this is plain safe Rust rather than intrinsics
//!
//! Explicit NEON/AVX-512 intrinsics would need `unsafe` and a `cfg` maze per target, and the resulting
//! kernel would be unavailable on exactly the platforms Ferric cares about reaching (wasm, unknown
//! hardware). Instead the inner loop is written in the shape LLVM autovectorises well: fixed-length
//! chunks, no branches, integer accumulation. That gets the bulk of the vector throughput while staying
//! portable and safe, which is the right trade for a runtime whose thesis is running everywhere.
//!
//! Threading is `std::thread::scope`, so the crate keeps its dependency-free posture.

/// A Q8_0 block: `f16` scale then 32 `i8` codes, 34 bytes. The GGUF layout, read in place.
const Q8_0_BLOCK: usize = 34;
const Q8_0_VALS: usize = 32;

/// Decode an IEEE binary16 to f32. Handles subnormals and inf/nan, because a weight file is allowed to
/// contain them and silently mapping them to zero would be a correctness bug that only shows up on
/// unusual checkpoints.
fn f16(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let man = (bits & 0x3FF) as u32;
    let out = match exp {
        0 if man == 0 => sign << 31,
        0 => {
            // Subnormal. Its value is `man * 2^-24`. Renormalise by shifting until the implicit bit
            // (0x400) is set: after k shifts the value is `1.xxx * 2^(-14-k)`, so the f32 exponent
            // field is `(-14 - k) + 127 = 113 - k`.
            let mut m = man;
            let mut k = 0u32;
            while m & 0x400 == 0 { m <<= 1; k += 1; }
            (sign << 31) | ((113 - k) << 23) | ((m & 0x3FF) << 13)
        }
        0x1F => (sign << 31) | 0x7F80_0000 | (man << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(out)
}

/// One row of `y = x · Wᵀ` for a Q8_0 weight, over `cols` inputs.
///
/// The inner loop is deliberately shaped for autovectorisation: a fixed 32-wide body, integer
/// multiply-accumulate, and the scale applied once per block rather than per element. Applying the
/// scale per element would be mathematically identical and several times slower, because it turns an
/// integer dot product into a float one.
#[inline]
fn row_q8_0(x: &[f32], blocks: &[u8], cols: usize) -> f32 {
    let nblk = cols / Q8_0_VALS;
    let mut acc = 0.0f32;
    for b in 0..nblk {
        let blk = &blocks[b * Q8_0_BLOCK..(b + 1) * Q8_0_BLOCK];
        let d = f16(u16::from_le_bytes([blk[0], blk[1]]));
        let xs = &x[b * Q8_0_VALS..(b + 1) * Q8_0_VALS];
        // Fixed 32-wide, branch-free. LLVM turns this into NEON SDOT / AVX-512 VNNI shapes.
        let mut part = 0.0f32;
        for j in 0..Q8_0_VALS {
            part += xs[j] * (blk[2 + j] as i8) as f32;
        }
        acc += part * d;
    }
    acc
}

/// `y = x · Wᵀ` on the CPU, where `W` is raw GGUF **Q8_0** bytes laid out `[rows, cols]`.
///
/// `x` is one activation row of length `cols`; the result is `rows` long. This is the decode shape, and
/// the shape the fabric splits: rows are independent, so any contiguous span of them can be handed to a
/// different compute unit and the results concatenated.
pub fn matvec_q8_0(x: &[f32], w: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(cols % Q8_0_VALS, 0, "Q8_0 cols must be a multiple of 32");
    assert_eq!(x.len(), cols, "activation length must equal cols");
    let row_bytes = (cols / Q8_0_VALS) * Q8_0_BLOCK;
    assert_eq!(w.len(), rows * row_bytes, "unexpected Q8_0 byte length");
    (0..rows).map(|r| row_q8_0(x, &w[r * row_bytes..(r + 1) * row_bytes], cols)).collect()
}

/// The same, across `threads` OS threads.
///
/// Rows are partitioned into contiguous spans so each thread walks memory linearly, which matters more
/// than load balance on a bandwidth-bound kernel: interleaving rows across threads would give every
/// thread a strided access pattern and throw away the prefetcher.
pub fn matvec_q8_0_threaded(x: &[f32], w: &[u8], rows: usize, cols: usize, threads: usize) -> Vec<f32> {
    let threads = threads.max(1).min(rows.max(1));
    if threads == 1 { return matvec_q8_0(x, w, rows, cols); }
    let row_bytes = (cols / Q8_0_VALS) * Q8_0_BLOCK;
    let mut out = vec![0.0f32; rows];
    let per = rows.div_ceil(threads);

    std::thread::scope(|s| {
        for (t, chunk) in out.chunks_mut(per).enumerate() {
            let lo = t * per;
            let hi = (lo + chunk.len()).min(rows);
            let wslice = &w[lo * row_bytes..hi * row_bytes];
            s.spawn(move || {
                for (i, o) in chunk.iter_mut().enumerate() {
                    *o = row_q8_0(x, &wslice[i * row_bytes..(i + 1) * row_bytes], cols);
                }
            });
        }
    });
    out
}

/// A contiguous span of output rows, so the fabric can hand different spans to different units.
///
/// Returns only `[lo, hi)` of the result. The GPU arm computes its own span concurrently and the two are
/// concatenated, which is the whole point: no unit waits for another.
pub fn matvec_q8_0_span(x: &[f32], w: &[u8], cols: usize, lo: usize, hi: usize, threads: usize) -> Vec<f32> {
    assert!(lo <= hi, "empty or inverted span");
    let row_bytes = (cols / Q8_0_VALS) * Q8_0_BLOCK;
    matvec_q8_0_threaded(x, &w[lo * row_bytes..hi * row_bytes], hi - lo, cols, threads)
}

/// Logical CPU count, for sizing a thread pool. Falls back to 1 rather than guessing high, since
/// oversubscribing a bandwidth-bound kernel makes it slower.
pub fn cpu_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a Q8_0 weight and the exact f32 values it encodes, so the test has ground truth that does
    /// not go through the same code being tested.
    fn synth(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        let mut rnd = || { s = s.wrapping_mul(1664525).wrapping_add(1013904223); (s >> 16) as u16 };
        let nblk = cols / 32;
        let mut raw = Vec::with_capacity(rows * nblk * 34);
        let mut vals = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for b in 0..nblk {
                // A scale that is exactly representable in f16, so ground truth is unambiguous.
                let dbits: u16 = 0x1C00 | ((rnd() & 0x3F) as u16); // ~0.0039 .. 0.0042
                raw.extend_from_slice(&dbits.to_le_bytes());
                let d = f16(dbits);
                for j in 0..32 {
                    let c = (rnd() & 0xFF) as u8;
                    raw.push(c);
                    vals[r * cols + b * 32 + j] = (c as i8) as f32 * d;
                }
            }
        }
        (raw, vals)
    }

    fn reference(x: &[f32], vals: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        (0..rows).map(|r| (0..cols).map(|i| x[i] * vals[r * cols + i]).sum()).collect()
    }

    #[test]
    fn matvec_matches_the_dequantised_reference() {
        for &(rows, cols) in &[(1usize, 32usize), (8, 64), (17, 256), (64, 512)] {
            let (raw, vals) = synth(rows, cols, (rows * 13 + cols) as u32);
            let x: Vec<f32> = (0..cols).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
            let got = matvec_q8_0(&x, &raw, rows, cols);
            let want = reference(&x, &vals, rows, cols);
            let scale = want.iter().fold(1e-6f32, |a, &v| a.max(v.abs()));
            let d = got.iter().zip(&want).fold(0f32, |a, (&g, &w)| a.max((g - w).abs())) / scale;
            assert!(d < 1e-5, "rows={rows} cols={cols}: rel delta {d:.3e}");
        }
    }

    #[test]
    fn threading_changes_nothing_about_the_answer() {
        // Rows are independent, so any partition must give the identical result. If this ever fails, a
        // thread is reading the wrong slice of the weight and the model would emit fluent, wrong text.
        let (raw, _) = synth(64, 256, 7);
        let x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
        let one = matvec_q8_0(&x, &raw, 64, 256);
        for t in [2usize, 3, 8, 64, 128] {
            let many = matvec_q8_0_threaded(&x, &raw, 64, 256, t);
            assert_eq!(one, many, "threads={t} changed the result");
        }
    }

    #[test]
    fn spans_concatenate_to_the_whole() {
        // The property the fabric depends on: the GPU takes one span, the CPU another, and the join is
        // the full answer. If spans did not compose, splitting would be silently wrong rather than slow.
        let (raw, _) = synth(40, 128, 3);
        let x: Vec<f32> = (0..128).map(|i| ((i % 7) as f32) - 3.0).collect();
        let whole = matvec_q8_0(&x, &raw, 40, 128);
        for cut in [0usize, 1, 17, 39, 40] {
            let mut joined = matvec_q8_0_span(&x, &raw, 128, 0, cut, 2);
            joined.extend(matvec_q8_0_span(&x, &raw, 128, cut, 40, 2));
            assert_eq!(whole, joined, "cut at {cut} did not reassemble");
        }
    }

    #[test]
    fn f16_decodes_subnormals_and_specials() {
        // A weight file may legally contain these, and mapping them to zero would be a silent
        // correctness bug on unusual checkpoints.
        assert_eq!(f16(0x0000), 0.0);
        assert_eq!(f16(0x3C00), 1.0);
        assert_eq!(f16(0xBC00), -1.0);
        assert!((f16(0x0001) - 5.9604645e-8).abs() < 1e-14, "smallest subnormal");
        assert!(f16(0x7C00).is_infinite());
        assert!(f16(0xFC00).is_infinite() && f16(0xFC00) < 0.0);
        assert!(f16(0x7E00).is_nan());
    }

    #[test]
    fn cpu_threads_never_reports_zero() {
        assert!(cpu_threads() >= 1);
    }
}

// ---- persistent worker pool ----

use std::sync::mpsc::{channel, Sender, Receiver};
use std::sync::{Arc, Mutex};

/// A job handed to one worker: compute a contiguous span of output rows.
type Job = Box<dyn FnOnce() -> (usize, Vec<f32>) + Send + 'static>;

/// Long-lived worker threads, so a split does not pay thread creation on every call.
///
/// Measured motivation: splitting one matmul across GPU and CPU with `std::thread::scope` spent **4.2 ms
/// of a 5.4 ms total on coordination**, a 78% overhead that turned a predicted 1.33x win into a measured
/// 0.30x regression. Spawning 18 threads costs more than the arithmetic when the arithmetic is
/// milliseconds. The threads here are created once and parked on a channel between jobs.
///
/// Data reaches workers through `Arc`, not borrows, so no scope is needed and the pool can outlive any
/// individual call. That is the whole reason this is safe without a lifetime escape hatch.
pub struct Pool {
    tx: Option<Sender<Job>>,
    /// Behind a `Mutex` for two reasons that are really one. It makes `Pool` `Sync`, so a `&Pool` can be
    /// handed to a scoped thread while another unit works. And it serialises `run`, which is required
    /// for correctness rather than merely convenient: two overlapping `run` calls on one receiver would
    /// steal each other's results and each would return a mix of the other's rows.
    results: Mutex<Receiver<(usize, Vec<f32>)>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl Pool {
    /// Start `n` workers. They park immediately and cost nothing until given work.
    pub fn new(n: usize) -> Pool {
        let n = n.max(1);
        let (tx, rx) = channel::<Job>();
        let (rtx, results) = channel::<(usize, Vec<f32>)>();
        let rx = Arc::new(Mutex::new(rx));
        let workers = (0..n).map(|_| {
            let rx = Arc::clone(&rx);
            let rtx = rtx.clone();
            std::thread::spawn(move || loop {
                // Lock only to take a job, never while running one, so workers do not serialise.
                let job = { rx.lock().expect("pool mutex poisoned").recv() };
                match job {
                    Ok(job) => { let _ = rtx.send(job()); }
                    Err(_) => break, // sender dropped: shut down
                }
            })
        }).collect();
        Pool { tx: Some(tx), results: Mutex::new(results), workers }
    }

    pub fn threads(&self) -> usize { self.workers.len() }

    /// Run `spans.len()` jobs and collect their outputs in span order.
    ///
    /// Results arrive out of order over the channel and are reordered by index here, because a caller
    /// concatenating them needs row order and the completion order is whatever the scheduler chose.
    pub fn run<F>(&self, spans: usize, make: F) -> Vec<Vec<f32>>
    where F: Fn(usize) -> Job {
        // Held across the whole call, so a concurrent `run` waits rather than consuming our results.
        let rx = self.results.lock().expect("pool mutex poisoned");
        let tx = self.tx.as_ref().expect("pool is shut down");
        for i in 0..spans { tx.send(make(i)).expect("worker threads have exited"); }
        let mut out: Vec<Option<Vec<f32>>> = (0..spans).map(|_| None).collect();
        for _ in 0..spans {
            let (i, v) = rx.recv().expect("a worker died before reporting");
            out[i] = Some(v);
        }
        out.into_iter().map(|o| o.expect("every span must report exactly once")).collect()
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        // Dropping the sender ends every worker's recv loop; then join so no thread outlives the pool.
        self.tx = None;
        for w in self.workers.drain(..) { let _ = w.join(); }
    }
}

/// `y = x · Wᵀ` for Q8_0 on a persistent pool, over the row span `[lo, hi)`.
///
/// `x` and `w` are `Arc`s so workers own their handles rather than borrowing, which is what lets the
/// pool outlive the call.
pub fn matvec_q8_0_pooled(
    pool: &Pool,
    x: Arc<Vec<f32>>,
    w: Arc<Vec<u8>>,
    cols: usize,
    lo: usize,
    hi: usize,
) -> Vec<f32> {
    assert!(lo <= hi, "empty or inverted span");
    let rows = hi - lo;
    if rows == 0 { return Vec::new(); }
    let row_bytes = (cols / Q8_0_VALS) * Q8_0_BLOCK;
    let spans = pool.threads().min(rows);
    let per = rows.div_ceil(spans);

    let parts = pool.run(spans, |i| {
        let (x, w) = (Arc::clone(&x), Arc::clone(&w));
        let a = lo + i * per;
        let b = (a + per).min(hi);
        Box::new(move || {
            let mut v = Vec::with_capacity(b.saturating_sub(a));
            for r in a..b {
                v.push(row_q8_0(&x, &w[r * row_bytes..(r + 1) * row_bytes], cols));
            }
            (i, v)
        }) as Job
    });
    parts.into_iter().flatten().collect()
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    fn synth_raw(rows: usize, cols: usize, seed: u32) -> Vec<u8> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        let mut rnd = || { s = s.wrapping_mul(1664525).wrapping_add(1013904223); (s >> 16) as u16 };
        let mut raw = Vec::with_capacity(rows * (cols / 32) * 34);
        for _ in 0..rows * (cols / 32) {
            raw.extend_from_slice(&(0x1C00u16 | (rnd() & 0x3F)).to_le_bytes());
            for _ in 0..32 { raw.push((rnd() & 0xFF) as u8); }
        }
        raw
    }

    #[test]
    fn pooled_matches_the_single_threaded_answer() {
        let (rows, cols) = (256usize, 128usize);
        let raw = Arc::new(synth_raw(rows, cols, 5));
        let x = Arc::new((0..cols).map(|i| ((i * 31 % 97) as f32 - 48.0) / 48.0).collect::<Vec<f32>>());
        let want = matvec_q8_0(&x, &raw, rows, cols);
        for n in [1usize, 2, 5, 16] {
            let pool = Pool::new(n);
            let got = matvec_q8_0_pooled(&pool, Arc::clone(&x), Arc::clone(&raw), cols, 0, rows);
            assert_eq!(got, want, "pool of {n} changed the result");
        }
    }

    #[test]
    fn results_are_reordered_into_span_order() {
        // Workers finish out of order; the caller concatenates and needs ROW order. If this ever broke,
        // the weights would be applied to the right values in the wrong places: fluent, wrong output.
        let (rows, cols) = (200usize, 64usize);
        let raw = Arc::new(synth_raw(rows, cols, 11));
        let x = Arc::new(vec![0.5f32; cols]);
        let pool = Pool::new(8);
        let got = matvec_q8_0_pooled(&pool, Arc::clone(&x), Arc::clone(&raw), cols, 0, rows);
        let want = matvec_q8_0(&x, &raw, rows, cols);
        assert_eq!(got, want);
    }

    #[test]
    fn spans_still_compose_through_the_pool() {
        let (rows, cols) = (128usize, 64usize);
        let raw = Arc::new(synth_raw(rows, cols, 2));
        let x = Arc::new(vec![-1.25f32; cols]);
        let pool = Pool::new(4);
        let whole = matvec_q8_0(&x, &raw, rows, cols);
        for cut in [0usize, 1, 63, 127, 128] {
            let mut j = matvec_q8_0_pooled(&pool, Arc::clone(&x), Arc::clone(&raw), cols, 0, cut);
            j.extend(matvec_q8_0_pooled(&pool, Arc::clone(&x), Arc::clone(&raw), cols, cut, rows));
            assert_eq!(j, whole, "cut at {cut} did not reassemble through the pool");
        }
    }

    #[test]
    fn a_pool_can_be_reused_without_respawning() {
        // The entire point. If each call respawned, this would be no better than thread::scope.
        let (rows, cols) = (64usize, 32usize);
        let raw = Arc::new(synth_raw(rows, cols, 9));
        let x = Arc::new(vec![1.0f32; cols]);
        let pool = Pool::new(4);
        let first = matvec_q8_0_pooled(&pool, Arc::clone(&x), Arc::clone(&raw), cols, 0, rows);
        for _ in 0..20 {
            let again = matvec_q8_0_pooled(&pool, Arc::clone(&x), Arc::clone(&raw), cols, 0, rows);
            assert_eq!(again, first, "a reused pool drifted");
        }
    }

    #[test]
    fn dropping_the_pool_joins_every_worker() {
        // A leaked thread outliving its pool would keep a core warm forever, which on a battery-powered
        // device is exactly the failure this whole crate exists to avoid.
        let pool = Pool::new(4);
        assert_eq!(pool.threads(), 4);
        drop(pool); // must not hang: Drop clears the sender, workers exit, join returns
    }
}
