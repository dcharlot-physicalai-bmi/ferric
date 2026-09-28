//! **The CPU fabric: GGUF weights multiplied in their stored block format on the CPU's vector units.**
//!
//! Most of the world's local inference runs on CPU cores — every laptop without a usable GPU, every
//! server without one — and until this module Ferric had no path there: `cpu_simd.rs` held one Q8_0
//! kernel wired into nothing. This is the substrate the Dense runtime's CPU tier
//! (`ferric_llama::qwen3::cpu`) runs on: a persistent worker pool, the activation quantizers, and a
//! dot-product kernel per weight format, dispatched by one matmul driver.
//!
//! ## The arithmetic (and where it differs from a GPU path)
//!
//! Quantized weights are NOT dequantized to f32. The activation row is quantized once per matmul to
//! int8 blocks (`d = amax/127`, 32-wide for the `*_0`/`*_1` formats, 256-wide with 16-element sums for
//! the k-quants), and each weight block is dotted in integers — the scheme llama.cpp's CPU backend uses,
//! because an int8 dot product is 4x the lanes of an f32 one and decode is bandwidth-bound anyway. It is
//! a DIFFERENT computation from Ferric's GPU path (which multiplies dequantized weights by f32
//! activations): the int8 rounding of the activations is a second, small quantization. `FERRIC_CPU_F32ACT=1`
//! keeps activations in f32 (weights dequantized row by row) — the arm that should agree with the GPU
//! to accumulation order, and what separates "the kernel is wrong" from "int8 activations cost this".
//!
//! F32 / F16 / BF16 weights always take f32 activations: those files exist to verify the math against
//! the model's authors, and an int8 activation would put a quantization error inside that comparison.
//!
//! ## Kernels
//!
//! Every format has a portable scalar kernel (x86, wasm, ARM without dotprod) and an aarch64 NEON
//! kernel using `SDOT` (FEAT_DotProd: every Apple M-series core, Graviton 2+, Cortex-A76+). The NEON
//! kernels are generic over a tile of `NR` weight rows x `NA` activation rows, so the same code serves
//! decode (one activation row, several weight rows in flight) and prefill (several activation rows
//! reusing each unpacked weight block).
//!
//! ## Threads
//!
//! A persistent pool, not `std::thread::scope`: a 24-layer model issues ~120 matmuls per decoded token,
//! and spawning threads per matmul costs more than the matmul. Workers spin on an epoch counter for
//! [`SPIN_US`] (the gap between two ops of one token is microseconds) and then park, so an idle model
//! does not hold cores. On macOS the workers ask for `QOS_CLASS_USER_INTERACTIVE`, which is how a
//! process asks the scheduler for performance cores (there is no affinity API).

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

// =====================================================================================================
// 1. Worker pool
// =====================================================================================================

/// How long an idle worker spins before parking, in microseconds (`FERRIC_CPU_SPIN_US` overrides).
///
/// Two ops of one decoded token are microseconds apart, and an unpark on macOS costs tens of
/// microseconds, so parking between ops would dominate a 0.5B model's step. Parking after a
/// millisecond of nothing means an idle model costs no cores.
pub const SPIN_US: u64 = 1000;

type JobFn<'a> = dyn Fn(usize) + Sync + 'a;

struct Shared {
    /// Bumped once per job; a worker that sees it change tries to join that job.
    epoch: AtomicUsize,
    /// The epoch of the last job whose items are all DONE. A worker that arrives for a closed job
    /// backs off without touching it — its closure may no longer exist.
    closed: AtomicUsize,
    /// Workers currently inside a job (between joining and leaving). The submitter waits for this to
    /// reach zero after closing, so a job's closure outlives every worker that saw it.
    active: AtomicUsize,
    /// The job in flight: a pointer to a `&JobFn` on the submitting thread's stack.
    job: AtomicPtr<&'static JobFn<'static>>,
    /// Next item to hand out, items in the job, items finished.
    next: AtomicUsize,
    total: AtomicUsize,
    done: AtomicUsize,
    quit: AtomicBool,
    parked: Vec<AtomicBool>,
    spin_us: u64,
}

/// A fixed set of worker threads that run one job's ITEMS at a time, the submitting thread included.
///
/// ⛔ A job is complete when its items are, not when every worker has checked in. The first version
/// was a barrier (the submitter waited for all n-1 workers to arrive and leave), and on a shared
/// machine — load 40-60 on 18 cores — every descheduled worker stalled every one of the ~100 matmuls
/// of a decoded token: 6 tok/s where the kernels alone allow hundreds. Here a worker that never gets
/// the CPU simply takes no items; the submitter waits only for workers already holding one.
pub struct Pool {
    n: usize,
    shared: Arc<Shared>,
    threads: Vec<std::thread::Thread>,
    /// Serialises submitters: two model threads sharing one pool take turns.
    submit: Mutex<()>,
}

thread_local! {
    /// Set on pool workers and on a submitter while its job runs: a job that itself submits runs the
    /// inner job inline instead of deadlocking on the pool it is occupying.
    static IN_POOL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn pthread_set_qos_class_self_np(qos: u32, relative_priority: i32) -> i32;
    fn sysctlbyname(name: *const std::ffi::c_char, oldp: *mut std::ffi::c_void, oldlenp: *mut usize,
                    newp: *mut std::ffi::c_void, newlen: usize) -> i32;
}

/// Ask for the performance cores. macOS has no affinity API; QoS is the documented way to tell the
/// scheduler a thread is latency-critical, and USER_INTERACTIVE is what keeps it off the efficiency
/// cores. A no-op elsewhere.
fn prefer_performance_cores() {
    #[cfg(target_os = "macos")]
    unsafe {
        const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

#[cfg(target_os = "macos")]
fn sysctl_u32(name: &str) -> Option<u32> {
    let c = std::ffi::CString::new(name).ok()?;
    let mut v: u32 = 0;
    let mut len = std::mem::size_of::<u32>();
    let r = unsafe { sysctlbyname(c.as_ptr(), &mut v as *mut u32 as *mut _, &mut len, std::ptr::null_mut(), 0) };
    (r == 0 && len == 4).then_some(v)
}

#[cfg(target_os = "macos")]
fn sysctl_str(name: &str) -> Option<String> {
    let c = std::ffi::CString::new(name).ok()?;
    let mut buf = [0u8; 64];
    let mut len = buf.len();
    let r = unsafe { sysctlbyname(c.as_ptr(), buf.as_mut_ptr() as *mut _, &mut len, std::ptr::null_mut(), 0) };
    if r != 0 { return None; }
    let s = &buf[..len.min(buf.len())];
    Some(String::from_utf8_lossy(s.split(|&b| b == 0).next().unwrap_or(&[])).into_owned())
}

/// The machine's non-efficiency cores, by the OS's own classification.
///
/// On Apple silicon that is every `hw.perflevelN` not named "Efficiency": the M3 Ultra's 24
/// Performance cores (not its 8 Efficiency cores), the M5 Max's 6 Super + 12 Performance cores. The
/// sum, not the first level, because the M5 generation's second level is a performance tier. Elsewhere,
/// `available_parallelism` (Linux does not distinguish, and SMT siblings are counted).
pub fn performance_cores() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Some(levels) = sysctl_u32("hw.nperflevels") {
            let mut n = 0usize;
            for l in 0..levels {
                let name = sysctl_str(&format!("hw.perflevel{l}.name")).unwrap_or_default();
                if name.eq_ignore_ascii_case("efficiency") { continue; }
                n += sysctl_u32(&format!("hw.perflevel{l}.physicalcpu")).unwrap_or(0) as usize;
            }
            if n > 0 { return n; }
        }
    }
    std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1)
}

impl Pool {
    /// A pool of `n` threads in total: `n - 1` workers plus whichever thread submits.
    ///
    /// On `wasm32-unknown-unknown` there are no threads; the pool is size 1 and every job runs inline.
    pub fn new(n: usize) -> Pool {
        let n = if cfg!(target_arch = "wasm32") { 1 } else { n.max(1) };
        let spin_us = std::env::var("FERRIC_CPU_SPIN_US").ok().and_then(|v| v.parse().ok()).unwrap_or(SPIN_US);
        let shared = Arc::new(Shared {
            epoch: AtomicUsize::new(0),
            closed: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            job: AtomicPtr::new(std::ptr::null_mut()),
            next: AtomicUsize::new(0),
            total: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            quit: AtomicBool::new(false),
            parked: (0..n).map(|_| AtomicBool::new(false)).collect(),
            spin_us,
        });
        let mut threads = Vec::with_capacity(n.saturating_sub(1));
        for ith in 1..n {
            let sh = Arc::clone(&shared);
            let h = std::thread::Builder::new()
                .name(format!("ferric-cpu-{ith}"))
                .spawn(move || worker(sh, ith))
                .expect("spawn CPU fabric worker");
            threads.push(h.thread().clone());
        }
        Pool { n, shared, threads, submit: Mutex::new(()) }
    }

    /// Threads a job can be split across (the submitter included).
    pub fn threads(&self) -> usize { self.n }

    /// Run `f(item)` for every `item` in `0..n_items` across the pool, items handed out dynamically
    /// (faster cores take more — the M5 Max mixes two core types). Returns when every item is done.
    ///
    /// Called from inside a job, runs the items inline: nested parallelism is flattened, never
    /// deadlocked.
    pub fn for_each<F: Fn(usize) + Sync>(&self, n_items: usize, f: F) {
        if n_items == 0 { return; }
        if n_items == 1 || self.n == 1 || IN_POOL.with(|c| c.get()) {
            for i in 0..n_items { f(i); }
            return;
        }
        let _g = self.submit.lock().unwrap_or_else(|e| e.into_inner());
        let sh = &*self.shared;
        let dynf: &JobFn<'_> = &f;
        // SAFETY: the pointee lives on this stack frame; this function does not return until the job
        // is closed and `active` is zero, after which no worker dereferences it (see `worker`).
        let dynf: &'static JobFn<'static> = unsafe { std::mem::transmute(dynf) };
        let slot: &&'static JobFn<'static> = &dynf;
        sh.job.store(slot as *const _ as *mut _, Ordering::SeqCst);
        sh.next.store(0, Ordering::SeqCst);
        sh.done.store(0, Ordering::SeqCst);
        sh.total.store(n_items, Ordering::SeqCst);
        let e = sh.epoch.fetch_add(1, Ordering::SeqCst) + 1;
        for (i, t) in self.threads.iter().enumerate() {
            if sh.parked[i + 1].load(Ordering::SeqCst) { t.unpark(); }
        }
        IN_POOL.with(|c| c.set(true));
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_items(sh, &f)));
        IN_POOL.with(|c| c.set(false));
        let mut spins = 0u32;
        while sh.done.load(Ordering::Acquire) < n_items {
            spins = spins.wrapping_add(1);
            if spins % 4096 == 0 { std::thread::yield_now(); } else { std::hint::spin_loop(); }
        }
        sh.closed.store(e, Ordering::SeqCst);
        while sh.active.load(Ordering::SeqCst) != 0 { std::hint::spin_loop(); }
        if let Err(p) = r { std::panic::resume_unwind(p); }
    }

    /// Run `f(ith, nth)` once on each of `nth` parallel lanes (`nth` = [`Pool::threads`]) — for jobs
    /// that want per-lane scratch. Lanes are items, so a descheduled worker delays nothing but its
    /// own lane's work, which another thread picks up.
    pub fn run<F: Fn(usize, usize) + Sync>(&self, f: F) {
        let n = self.n;
        self.for_each(n, |i| f(i, n));
    }
}

/// Take and run items until none are left; count each one done (even if it panicked, so the
/// submitter cannot wait forever — the panic is reported, and the submitter's own is re-raised).
fn run_items(sh: &Shared, f: &JobFn<'_>) {
    let total = sh.total.load(Ordering::Acquire);
    loop {
        let i = sh.next.fetch_add(1, Ordering::AcqRel);
        if i >= total { break; }
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(i)));
        sh.done.fetch_add(1, Ordering::AcqRel);
        if let Err(p) = r { std::panic::resume_unwind(p); }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.quit.store(true, Ordering::SeqCst);
        self.shared.epoch.fetch_add(1, Ordering::SeqCst);
        for t in &self.threads { t.unpark(); }
    }
}

fn worker(sh: Arc<Shared>, ith: usize) {
    prefer_performance_cores();
    IN_POOL.with(|c| c.set(true));
    // ⛔ Start from the epoch the pool was BUILT with (0), not from a load: a job submitted before this
    // thread got scheduled has already bumped the epoch, and a worker that adopted the bumped value
    // would sit that job out. (Harmless now that completion counts items — it was a hang when the
    // submitter waited for every worker — but it would idle a core for a whole job.)
    let mut seen = 0usize;
    loop {
        let mut spins = 0u32;
        let mut since = std::time::Instant::now();
        let e = loop {
            let e = sh.epoch.load(Ordering::Acquire);
            if e != seen { break e; }
            spins = spins.wrapping_add(1);
            if spins % 256 != 0 { std::hint::spin_loop(); continue; }
            if since.elapsed().as_micros() as u64 >= sh.spin_us {
                sh.parked[ith].store(true, Ordering::SeqCst);
                // Re-check after announcing: the submitter bumps the epoch THEN reads `parked`, both
                // SeqCst, so either it sees this flag and unparks, or this load sees its bump.
                if sh.epoch.load(Ordering::SeqCst) == seen && !sh.quit.load(Ordering::SeqCst) {
                    std::thread::park();
                }
                sh.parked[ith].store(false, Ordering::SeqCst);
                since = std::time::Instant::now();
            }
        };
        seen = e;
        if sh.quit.load(Ordering::Acquire) { return; }
        // Join: announce, THEN check the job is still open. The submitter closes, THEN waits for
        // `active == 0`, all SeqCst — so either it sees us and waits, or we see it closed and leave.
        sh.active.fetch_add(1, Ordering::SeqCst);
        if sh.closed.load(Ordering::SeqCst) < e && sh.epoch.load(Ordering::SeqCst) == e {
            let job = sh.job.load(Ordering::SeqCst);
            // SAFETY: the job is open and we are counted in `active`; it outlives our decrement.
            let f: &JobFn<'_> = unsafe { *job };
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_items(&sh, f))).is_err() {
                eprintln!("ferric-cpu worker {ith}: a job item panicked");
            }
        }
        sh.active.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The process-wide pool the CPU fabric runs on: `FERRIC_CPU_THREADS`, else [`performance_cores`].
pub fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::env::var("FERRIC_CPU_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(performance_cores);
        prefer_performance_cores();
        Pool::new(n)
    })
}

/// A raw pointer that may cross into pool threads. Every use writes disjoint ranges.
#[derive(Clone, Copy)]
pub struct SyncPtr<T>(pub *mut T);
unsafe impl<T> Sync for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}
impl<T> SyncPtr<T> {
    /// The pointer. A method rather than `.0` because edition-2024 closures capture the FIELD — a bare
    /// `*mut T`, which is not `Sync` — when they name `.0`.
    #[inline(always)]
    pub fn get(&self) -> *mut T { self.0 }
}

// =====================================================================================================
// 2. Formats and the weight container
// =====================================================================================================

/// IEEE binary16 → f32, exact (subnormals and inf/nan included — a weight file may carry them).
#[inline(always)]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    if exp == 0 {
        // man * 2^-24, exact in f32.
        let v = man as f32 * (1.0 / 16_777_216.0);
        return if sign != 0 { -v } else { v };
    }
    if exp == 31 { return f32::from_bits(sign | 0x7f80_0000 | (man << 13)); }
    f32::from_bits(sign | ((exp + 112) << 23) | (man << 13))
}

#[inline(always)]
unsafe fn rd_u16(p: *const u8) -> u16 { unsafe { (p as *const u16).read_unaligned() } }
#[inline(always)]
unsafe fn rd_u32(p: *const u8) -> u32 { unsafe { (p as *const u32).read_unaligned() } }
#[inline(always)]
unsafe fn rd_f16(p: *const u8) -> f32 { f16_to_f32(unsafe { rd_u16(p) }) }

/// A weight format this fabric multiplies natively. Anything else is dequantized to F32 at load by
/// the caller (and says so) rather than approximated.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WType { F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K }

/// What the activation row is turned into before a weight of some format meets it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Act { F32, Q8x32, Q8x256 }

impl WType {
    /// From a GGUF `ggml_type` id; `None` for a format this fabric has no kernel for.
    pub fn from_ggml(t: u32) -> Option<WType> {
        Some(match t {
            0 => WType::F32, 1 => WType::F16, 30 => WType::BF16,
            2 => WType::Q4_0, 3 => WType::Q4_1, 6 => WType::Q5_0, 7 => WType::Q5_1, 8 => WType::Q8_0,
            12 => WType::Q4_K, 13 => WType::Q5_K, 14 => WType::Q6_K,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            WType::F32 => "F32", WType::F16 => "F16", WType::BF16 => "BF16", WType::Q4_0 => "Q4_0",
            WType::Q4_1 => "Q4_1", WType::Q5_0 => "Q5_0", WType::Q5_1 => "Q5_1", WType::Q8_0 => "Q8_0",
            WType::Q4_K => "Q4_K", WType::Q5_K => "Q5_K", WType::Q6_K => "Q6_K",
        }
    }
    /// (elements, bytes) per block — ggml's layouts, read in place.
    pub fn block(self) -> (usize, usize) {
        match self {
            WType::F32 => (1, 4), WType::F16 | WType::BF16 => (1, 2),
            WType::Q4_0 => (32, 18), WType::Q4_1 => (32, 20), WType::Q5_0 => (32, 22), WType::Q5_1 => (32, 24),
            WType::Q8_0 => (32, 34), WType::Q4_K => (256, 144), WType::Q5_K => (256, 176), WType::Q6_K => (256, 210),
        }
    }
    pub fn row_bytes(self, cols: usize) -> usize { let (e, b) = self.block(); cols / e * b }
    fn act(self) -> Act {
        match self {
            WType::F32 | WType::F16 | WType::BF16 => Act::F32,
            WType::Q4_0 | WType::Q4_1 | WType::Q5_0 | WType::Q5_1 | WType::Q8_0 => Act::Q8x32,
            WType::Q4_K | WType::Q5_K | WType::Q6_K => Act::Q8x256,
        }
    }
}

/// A `[rows, cols]` weight in its GGUF block format, row-major blocks, owned.
pub struct QWeight {
    pub ty: WType,
    pub rows: usize,
    pub cols: usize,
    data: Vec<u8>,
}

impl QWeight {
    pub fn new(ty: WType, rows: usize, cols: usize, data: Vec<u8>) -> Result<QWeight, String> {
        let (e, _) = ty.block();
        if cols % e != 0 {
            return Err(format!("{} rows of {cols} columns: not a multiple of the {e}-element block", ty.name()));
        }
        if data.len() != rows * ty.row_bytes(cols) {
            return Err(format!("{} [{rows}, {cols}] needs {} bytes, got {}", ty.name(), rows * ty.row_bytes(cols), data.len()));
        }
        Ok(QWeight { ty, rows, cols, data })
    }
    /// From f32 values (the fallback for formats with no kernel: the caller dequantized them).
    pub fn from_f32(rows: usize, cols: usize, w: &[f32]) -> QWeight {
        assert_eq!(w.len(), rows * cols);
        QWeight { ty: WType::F32, rows, cols, data: w.iter().flat_map(|x| x.to_le_bytes()).collect() }
    }
    pub fn nbytes(&self) -> usize { self.data.len() }
    pub fn bytes(&self) -> &[u8] { &self.data }
    pub fn row_bytes(&self) -> usize { self.ty.row_bytes(self.cols) }
    /// Split into consecutive row ranges (a fused `attn_qkv` into q, k, v): rows are independent, so
    /// a split is a byte split, exact.
    pub fn split_rows(self, parts: &[usize]) -> Result<Vec<QWeight>, String> {
        if parts.iter().sum::<usize>() != self.rows {
            return Err(format!("split {parts:?} does not tile {} rows", self.rows));
        }
        let rb = self.row_bytes();
        let mut out = Vec::with_capacity(parts.len());
        let mut r0 = 0;
        for &n in parts {
            out.push(QWeight { ty: self.ty, rows: n, cols: self.cols, data: self.data[r0 * rb..(r0 + n) * rb].to_vec() });
            r0 += n;
        }
        Ok(out)
    }
    /// One row dequantized to f32 — the `FERRIC_CPU_F32ACT` arm and diagnostics.
    pub fn dequant_row(&self, r: usize, out: &mut [f32]) {
        let rb = self.row_bytes();
        dequant_row(self.ty, &self.data[r * rb..(r + 1) * rb], out);
    }
}

/// Dequantize one row of `ty` blocks into `out` (len = the row's element count). ggml's definitions.
pub fn dequant_row(ty: WType, row: &[u8], out: &mut [f32]) {
    let (e, bb) = ty.block();
    let nb = out.len() / e;
    match ty {
        WType::F32 => for (o, c) in out.iter_mut().zip(row.chunks_exact(4)) { *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]); },
        WType::F16 => for (o, c) in out.iter_mut().zip(row.chunks_exact(2)) { *o = f16_to_f32(u16::from_le_bytes([c[0], c[1]])); },
        WType::BF16 => for (o, c) in out.iter_mut().zip(row.chunks_exact(2)) { *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16); },
        _ => for b in 0..nb {
            let blk = &row[b * bb..(b + 1) * bb];
            let y = &mut out[b * e..(b + 1) * e];
            let h = |o: usize| f16_to_f32(u16::from_le_bytes([blk[o], blk[o + 1]]));
            match ty {
                WType::Q4_0 => { let d = h(0); for j in 0..16 { y[j] = ((blk[2 + j] & 15) as i32 - 8) as f32 * d; y[j + 16] = ((blk[2 + j] >> 4) as i32 - 8) as f32 * d; } }
                WType::Q4_1 => { let (d, m) = (h(0), h(2)); for j in 0..16 { y[j] = (blk[4 + j] & 15) as f32 * d + m; y[j + 16] = (blk[4 + j] >> 4) as f32 * d + m; } }
                WType::Q5_0 | WType::Q5_1 => {
                    let one = ty == WType::Q5_1;
                    let d = h(0); let m = if one { h(2) } else { 0.0 };
                    let o = if one { 4 } else { 2 };
                    let qh = u32::from_le_bytes([blk[o], blk[o + 1], blk[o + 2], blk[o + 3]]);
                    for j in 0..16 {
                        let x0 = ((blk[o + 4 + j] & 15) as u32 | (((qh >> j) & 1) << 4)) as i32;
                        let x1 = ((blk[o + 4 + j] >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4)) as i32;
                        if one { y[j] = x0 as f32 * d + m; y[j + 16] = x1 as f32 * d + m; }
                        else { y[j] = (x0 - 16) as f32 * d; y[j + 16] = (x1 - 16) as f32 * d; }
                    }
                }
                WType::Q8_0 => { let d = h(0); for j in 0..32 { y[j] = blk[2 + j] as i8 as f32 * d; } }
                WType::Q4_K | WType::Q5_K => {
                    let (d, dmin) = (h(0), h(2));
                    let (sc, mn) = k4_scales(&blk[4..16]);
                    let five = ty == WType::Q5_K;
                    let (qh, qs) = if five { (&blk[16..48], &blk[48..176]) } else { (&blk[..0], &blk[16..144]) };
                    for j in 0..4 {
                        let (d1, m1) = (d * sc[2 * j] as f32, dmin * mn[2 * j] as f32);
                        let (d2, m2) = (d * sc[2 * j + 1] as f32, dmin * mn[2 * j + 1] as f32);
                        for l in 0..32 {
                            let q = qs[32 * j + l];
                            let (h1, h2) = if five { (((qh[l] >> (2 * j)) & 1) * 16, ((qh[l] >> (2 * j + 1)) & 1) * 16) } else { (0, 0) };
                            y[64 * j + l] = d1 * ((q & 15) + h1) as f32 - m1;
                            y[64 * j + 32 + l] = d2 * ((q >> 4) + h2) as f32 - m2;
                        }
                    }
                }
                WType::Q6_K => {
                    let d = h(208);
                    let (ql, qh, sc) = (&blk[0..128], &blk[128..192], &blk[192..208]);
                    for n in 0..2 {
                        for l in 0..32 {
                            let is = l / 16;
                            let q1 = ((ql[64 * n + l] & 15) | ((qh[32 * n + l] & 3) << 4)) as i32 - 32;
                            let q2 = ((ql[64 * n + l + 32] & 15) | (((qh[32 * n + l] >> 2) & 3) << 4)) as i32 - 32;
                            let q3 = ((ql[64 * n + l] >> 4) | (((qh[32 * n + l] >> 4) & 3) << 4)) as i32 - 32;
                            let q4 = ((ql[64 * n + l + 32] >> 4) | (((qh[32 * n + l] >> 6) & 3) << 4)) as i32 - 32;
                            let s = |k: usize| sc[8 * n + is + k] as i8 as f32;
                            y[128 * n + l] = d * s(0) * q1 as f32;
                            y[128 * n + l + 32] = d * s(2) * q2 as f32;
                            y[128 * n + l + 64] = d * s(4) * q3 as f32;
                            y[128 * n + l + 96] = d * s(6) * q4 as f32;
                        }
                    }
                }
                _ => unreachable!(),
            }
        },
    }
}

/// Q4_K / Q5_K's twelve scale bytes → eight 6-bit scales and eight 6-bit mins (ggml's
/// `get_scale_min_k4`, done four lanes at a time on u32 words).
#[inline(always)]
fn k4_scales(s: &[u8]) -> ([u8; 8], [u8; 8]) {
    let u = |i: usize| u32::from_le_bytes([s[i], s[i + 1], s[i + 2], s[i + 3]]);
    let (u0, u1, u2) = (u(0), u(4), u(8));
    let sc_lo = u0 & 0x3f3f_3f3f;
    let sc_hi = (u2 & 0x0f0f_0f0f) | (((u0 >> 6) & 0x0303_0303) << 4);
    let mn_lo = u1 & 0x3f3f_3f3f;
    let mn_hi = ((u2 >> 4) & 0x0f0f_0f0f) | (((u1 >> 6) & 0x0303_0303) << 4);
    let (a, b, c, d) = (sc_lo.to_le_bytes(), sc_hi.to_le_bytes(), mn_lo.to_le_bytes(), mn_hi.to_le_bytes());
    ([a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]], [c[0], c[1], c[2], c[3], d[0], d[1], d[2], d[3]])
}

// =====================================================================================================
// 3. Activation quantization
// =====================================================================================================

/// `m` activation rows quantized to int8 blocks: `qs` [m, k], one `d` per block, a sum per 16 codes.
pub struct QAct {
    block: usize,
    k: usize,
    qs: Vec<i8>,
    d: Vec<f32>,
    bsums: Vec<i16>,
}

impl QAct {
    fn quantize(x: &[f32], m: usize, k: usize, block: usize) -> QAct {
        let nb = k / block;
        let mut a = QAct { block, k, qs: vec![0; m * k], d: vec![0.0; m * nb], bsums: vec![0; m * k / 16] };
        let (qp, dp, bp) = (SyncPtr(a.qs.as_mut_ptr()), SyncPtr(a.d.as_mut_ptr()), SyncPtr(a.bsums.as_mut_ptr()));
        let row = |r: usize| unsafe {
            let q = std::slice::from_raw_parts_mut(qp.get().add(r * k), k);
            let d = std::slice::from_raw_parts_mut(dp.get().add(r * nb), nb);
            let bs = std::slice::from_raw_parts_mut(bp.get().add(r * k / 16), k / 16);
            quantize_row(&x[r * k..(r + 1) * k], block, q, d, bs);
        };
        if m * k >= 1 << 16 { pool().for_each(m, row); } else { for r in 0..m { row(r); } }
        a
    }
    fn row(&self, r: usize) -> ARow {
        let nb = self.k / self.block;
        unsafe {
            ARow { qs: self.qs.as_ptr().add(r * self.k), d: self.d.as_ptr().add(r * nb),
                   bs: self.bsums.as_ptr().add(r * self.k / 16), f: std::ptr::null() }
        }
    }
}

/// One row → int8 blocks. `d = amax/127`, codes rounded to nearest (ties to even), so |code| <= 127
/// and `d * code` is the reconstruction. (llama.cpp's Q8_K picks the sign of `d` from the extreme
/// value; the magnitude is the same, and this is an internal format, never written to a file.)
fn quantize_row(x: &[f32], block: usize, q: &mut [i8], d: &mut [f32], bs: &mut [i16]) {
    for (b, xb) in x.chunks_exact(block).enumerate() {
        let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
        let db = amax / 127.0;
        let id = if db > 0.0 { 1.0 / db } else { 0.0 };
        d[b] = db;
        let qb = &mut q[b * block..(b + 1) * block];
        for (o, &v) in qb.iter_mut().zip(xb) { *o = (v * id).round_ties_even() as i8; }
    }
    for (s, c) in bs.iter_mut().zip(q.chunks_exact(16)) { *s = c.iter().map(|&v| v as i16).sum(); }
}

/// One activation row as the kernels read it (int8 codes / block scales / 16-sums, or f32 values).
#[derive(Clone, Copy)]
struct ARow { qs: *const i8, d: *const f32, bs: *const i16, f: *const f32 }
unsafe impl Send for ARow {}
unsafe impl Sync for ARow {}

// =====================================================================================================
// 4. Portable scalar kernels — one weight row against one activation row
// =====================================================================================================

mod scalar {
    use super::*;

    pub unsafe fn dot(ty: WType, w: *const u8, k: usize, a: ARow) -> f32 {
        unsafe {
            match ty {
                WType::F32 | WType::F16 | WType::BF16 => {
                    let mut s = [0f32; 8];
                    for c in 0..k {
                        let wv = match ty {
                            WType::F32 => f32::from_bits(rd_u32(w.add(4 * c))),
                            WType::F16 => rd_f16(w.add(2 * c)),
                            _ => f32::from_bits((rd_u16(w.add(2 * c)) as u32) << 16),
                        };
                        s[c % 8] += wv * *a.f.add(c);
                    }
                    s.iter().sum()
                }
                WType::Q4_0 | WType::Q4_1 | WType::Q5_0 | WType::Q5_1 | WType::Q8_0 => dot32(ty, w, k / 32, a),
                _ => dotk(ty, w, k / 256, a),
            }
        }
    }

    unsafe fn dot32(ty: WType, w: *const u8, nb: usize, a: ARow) -> f32 {
        let bb = ty.block().1;
        let mut acc = 0f32;
        for b in 0..nb {
            unsafe {
                let blk = w.add(b * bb);
                let aq = a.qs.add(b * 32);
                let ad = *a.d.add(b);
                let mut q = [0i32; 32];
                let (d, m) = match ty {
                    WType::Q4_0 => { for j in 0..16 { let v = *blk.add(2 + j); q[j] = (v & 15) as i32 - 8; q[j + 16] = (v >> 4) as i32 - 8; } (rd_f16(blk), 0.0) }
                    WType::Q4_1 => { for j in 0..16 { let v = *blk.add(4 + j); q[j] = (v & 15) as i32; q[j + 16] = (v >> 4) as i32; } (rd_f16(blk), rd_f16(blk.add(2))) }
                    WType::Q5_0 | WType::Q5_1 => {
                        let one = ty == WType::Q5_1;
                        let o = if one { 4 } else { 2 };
                        let qh = rd_u32(blk.add(o));
                        let off = if one { 0 } else { 16 };
                        for j in 0..16 {
                            let v = *blk.add(o + 4 + j) as u32;
                            q[j] = ((v & 15) | (((qh >> j) & 1) << 4)) as i32 - off;
                            q[j + 16] = ((v >> 4) | (((qh >> (j + 16)) & 1) << 4)) as i32 - off;
                        }
                        (rd_f16(blk), if one { rd_f16(blk.add(2)) } else { 0.0 })
                    }
                    _ => { for j in 0..32 { q[j] = *blk.add(2 + j) as i8 as i32; } (rd_f16(blk), 0.0) }
                };
                let mut s = 0i32;
                for j in 0..32 { s += q[j] * *aq.add(j) as i32; }
                acc += d * ad * s as f32;
                if m != 0.0 {
                    let sa = *a.bs.add(2 * b) as i32 + *a.bs.add(2 * b + 1) as i32;
                    acc += m * ad * sa as f32;
                }
            }
        }
        acc
    }

    unsafe fn dotk(ty: WType, w: *const u8, nb: usize, a: ARow) -> f32 {
        let bb = ty.block().1;
        let mut acc = 0f32;
        for b in 0..nb {
            unsafe {
                let blk = std::slice::from_raw_parts(w.add(b * bb), bb);
                let aq = std::slice::from_raw_parts(a.qs.add(b * 256), 256);
                let bs = std::slice::from_raw_parts(a.bs.add(b * 16), 16);
                let ad = *a.d.add(b);
                match ty {
                    WType::Q4_K | WType::Q5_K => {
                        let five = ty == WType::Q5_K;
                        let (d, dmin) = (f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])), f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])));
                        let (sc, mn) = k4_scales(&blk[4..16]);
                        let (qh, qs) = if five { (&blk[16..48], &blk[48..176]) } else { (&blk[..0], &blk[16..144]) };
                        let (mut isum, mut msum) = (0i32, 0i32);
                        for j in 0..4 {
                            let (mut s1, mut s2) = (0i32, 0i32);
                            for l in 0..32 {
                                let v = qs[32 * j + l];
                                let (h1, h2) = if five { (((qh[l] >> (2 * j)) & 1) * 16, ((qh[l] >> (2 * j + 1)) & 1) * 16) } else { (0, 0) };
                                s1 += ((v & 15) + h1) as i32 * aq[64 * j + l] as i32;
                                s2 += ((v >> 4) + h2) as i32 * aq[64 * j + 32 + l] as i32;
                            }
                            isum += s1 * sc[2 * j] as i32 + s2 * sc[2 * j + 1] as i32;
                            msum += mn[2 * j] as i32 * (bs[4 * j] as i32 + bs[4 * j + 1] as i32)
                                  + mn[2 * j + 1] as i32 * (bs[4 * j + 2] as i32 + bs[4 * j + 3] as i32);
                        }
                        acc += ad * (d * isum as f32 - dmin * msum as f32);
                    }
                    _ => {
                        let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
                        let (ql, qh, sc) = (&blk[0..128], &blk[128..192], &blk[192..208]);
                        let mut isum = 0i32;
                        for n in 0..2 {
                            for l in 0..32 {
                                let is = 8 * n + l / 16;
                                let q1 = ((ql[64 * n + l] & 15) | ((qh[32 * n + l] & 3) << 4)) as i32 - 32;
                                let q2 = ((ql[64 * n + l + 32] & 15) | (((qh[32 * n + l] >> 2) & 3) << 4)) as i32 - 32;
                                let q3 = ((ql[64 * n + l] >> 4) | (((qh[32 * n + l] >> 4) & 3) << 4)) as i32 - 32;
                                let q4 = ((ql[64 * n + l + 32] >> 4) | (((qh[32 * n + l] >> 6) & 3) << 4)) as i32 - 32;
                                let base = 128 * n + l;
                                isum += sc[is] as i8 as i32 * q1 * aq[base] as i32
                                      + sc[is + 2] as i8 as i32 * q2 * aq[base + 32] as i32
                                      + sc[is + 4] as i8 as i32 * q3 * aq[base + 64] as i32
                                      + sc[is + 6] as i8 as i32 * q4 * aq[base + 96] as i32;
                            }
                        }
                        acc += ad * d * isum as f32;
                    }
                }
            }
        }
        acc
    }
}

// =====================================================================================================
// 5. aarch64 NEON + DotProd kernels: a tile of NR weight rows x NA activation rows
// =====================================================================================================

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    /// Byte j of the result is `bitval` when bit j of the low 16 bits of `x` is set — Q5's fifth bits.
    #[inline(always)]
    unsafe fn expand16(x: u32, bitval: u8) -> uint8x16_t {
        static BIT: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        unsafe {
            let v = vcombine_u8(vdup_n_u8(x as u8), vdup_n_u8((x >> 8) as u8));
            vandq_u8(vtstq_u8(v, vld1q_u8(BIT.as_ptr())), vdupq_n_u8(bitval))
        }
    }

    /// The four 32-element formats, one generic body: `lo`/`hi` are the block's first and last 16
    /// codes as signed bytes, `min` its additive term (Q4_1/Q5_1) or zero.
    #[inline(always)]
    unsafe fn blk32(ty: WType, blk: *const u8) -> (int8x16_t, int8x16_t, f32, f32) {
        unsafe {
            let m4 = vdupq_n_u8(0x0f);
            match ty {
                WType::Q8_0 => (vld1q_s8(blk.add(2) as *const i8), vld1q_s8(blk.add(18) as *const i8), rd_f16(blk), 0.0),
                WType::Q4_0 => {
                    let v = vld1q_u8(blk.add(2));
                    let s8 = vdupq_n_s8(8);
                    (vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m4)), s8), vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), s8), rd_f16(blk), 0.0)
                }
                WType::Q4_1 => {
                    let v = vld1q_u8(blk.add(4));
                    (vreinterpretq_s8_u8(vandq_u8(v, m4)), vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), rd_f16(blk), rd_f16(blk.add(2)))
                }
                WType::Q5_0 => {
                    let qh = rd_u32(blk.add(2));
                    let v = vld1q_u8(blk.add(6));
                    let s16 = vdupq_n_s8(16);
                    let lo = vorrq_u8(vandq_u8(v, m4), expand16(qh, 16));
                    let hi = vorrq_u8(vshrq_n_u8(v, 4), expand16(qh >> 16, 16));
                    (vsubq_s8(vreinterpretq_s8_u8(lo), s16), vsubq_s8(vreinterpretq_s8_u8(hi), s16), rd_f16(blk), 0.0)
                }
                _ => {
                    // Q5_1
                    let qh = rd_u32(blk.add(4));
                    let v = vld1q_u8(blk.add(8));
                    let lo = vorrq_u8(vandq_u8(v, m4), expand16(qh, 16));
                    let hi = vorrq_u8(vshrq_n_u8(v, 4), expand16(qh >> 16, 16));
                    (vreinterpretq_s8_u8(lo), vreinterpretq_s8_u8(hi), rd_f16(blk), rd_f16(blk.add(2)))
                }
            }
        }
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn tile32<const NR: usize, const NA: usize>(ty: WType, w: *const u8, rb: usize, nb: usize,
                                                            a: &[ARow; NA], out: &mut [[f32; NA]; NR]) {
        let bb = ty.block().1;
        let z = vdupq_n_s32(0);
        let mut acc = [[vdupq_n_f32(0.0); NA]; NR];
        let mut macc = [[0f32; NA]; NR];
        let has_min = matches!(ty, WType::Q4_1 | WType::Q5_1);
        for b in 0..nb {
            let mut av = [[vdupq_n_s8(0); 2]; NA];
            let mut ad = [0f32; NA];
            for i in 0..NA {
                let q = unsafe { a[i].qs.add(b * 32) };
                av[i] = unsafe { [vld1q_s8(q), vld1q_s8(q.add(16))] };
                ad[i] = unsafe { *a[i].d.add(b) };
            }
            for r in 0..NR {
                let (lo, hi, d, m) = unsafe { blk32(ty, w.add(r * rb + b * bb)) };
                for i in 0..NA {
                    let p = vdotq_s32(vdotq_s32(z, lo, av[i][0]), hi, av[i][1]);
                    acc[r][i] = vfmaq_n_f32(acc[r][i], vcvtq_f32_s32(p), d * ad[i]);
                    if has_min {
                        let sa = unsafe { *a[i].bs.add(2 * b) as i32 + *a[i].bs.add(2 * b + 1) as i32 };
                        macc[r][i] += m * ad[i] * sa as f32;
                    }
                }
            }
        }
        for r in 0..NR { for i in 0..NA { out[r][i] = vaddvq_f32(acc[r][i]) + macc[r][i]; } }
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn tile_q45k<const NR: usize, const NA: usize, const FIVE: bool>(w: *const u8, rb: usize, nb: usize,
                                                                                  a: &[ARow; NA], out: &mut [[f32; NA]; NR]) {
        let bb = if FIVE { 176 } else { 144 };
        let m4 = vdupq_n_u8(0x0f);
        let one = vdupq_n_u8(1);
        let two = vdupq_n_u8(2);
        let z = vdupq_n_s32(0);
        let mut accf = [[0f32; NA]; NR];
        for b in 0..nb {
            let mut bsp = [vdupq_n_s16(0); NA];
            let mut ad = [0f32; NA];
            for i in 0..NA {
                unsafe {
                    let bs = a[i].bs.add(b * 16);
                    bsp[i] = vpaddq_s16(vld1q_s16(bs), vld1q_s16(bs.add(8)));
                    ad[i] = *a[i].d.add(b);
                }
            }
            for r in 0..NR {
                let blk = unsafe { w.add(r * rb + b * bb) };
                let (d, dmin) = unsafe { (rd_f16(blk), rd_f16(blk.add(2))) };
                let (sc, mn) = k4_scales(unsafe { std::slice::from_raw_parts(blk.add(4), 12) });
                let mn16 = unsafe { vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mn.as_ptr()))) };
                let q = unsafe { blk.add(if FIVE { 48 } else { 16 }) };
                let (mut qh0, mut qh1) = if FIVE { unsafe { (vld1q_u8(blk.add(16)), vld1q_u8(blk.add(32))) } } else { (one, one) };
                let mut isum = [z; NA];
                for j in 0..4 {
                    let (v0, v1) = unsafe { (vld1q_u8(q.add(32 * j)), vld1q_u8(q.add(32 * j + 16))) };
                    let (mut l0, mut l1) = (vandq_u8(v0, m4), vandq_u8(v1, m4));
                    let (mut h0, mut h1) = (vshrq_n_u8(v0, 4), vshrq_n_u8(v1, 4));
                    if FIVE {
                        l0 = vorrq_u8(l0, vshlq_n_u8(vandq_u8(qh0, one), 4));
                        l1 = vorrq_u8(l1, vshlq_n_u8(vandq_u8(qh1, one), 4));
                        h0 = vorrq_u8(h0, vshlq_n_u8(vandq_u8(qh0, two), 3));
                        h1 = vorrq_u8(h1, vshlq_n_u8(vandq_u8(qh1, two), 3));
                        qh0 = vshrq_n_u8(qh0, 2);
                        qh1 = vshrq_n_u8(qh1, 2);
                    }
                    let (l0, l1, h0, h1) = (vreinterpretq_s8_u8(l0), vreinterpretq_s8_u8(l1), vreinterpretq_s8_u8(h0), vreinterpretq_s8_u8(h1));
                    for i in 0..NA {
                        unsafe {
                            let aq = a[i].qs.add(b * 256 + 64 * j);
                            let pl = vdotq_s32(vdotq_s32(z, l0, vld1q_s8(aq)), l1, vld1q_s8(aq.add(16)));
                            let ph = vdotq_s32(vdotq_s32(z, h0, vld1q_s8(aq.add(32))), h1, vld1q_s8(aq.add(48)));
                            isum[i] = vmlaq_n_s32(isum[i], pl, sc[2 * j] as i32);
                            isum[i] = vmlaq_n_s32(isum[i], ph, sc[2 * j + 1] as i32);
                        }
                    }
                }
                for i in 0..NA {
                    let mins = vaddvq_s32(vaddq_s32(vmull_s16(vget_low_s16(mn16), vget_low_s16(bsp[i])), vmull_high_s16(mn16, bsp[i])));
                    accf[r][i] += ad[i] * (d * vaddvq_s32(isum[i]) as f32 - dmin * mins as f32);
                }
            }
        }
        *out = accf;
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn tile_q6k<const NR: usize, const NA: usize>(w: *const u8, rb: usize, nb: usize,
                                                             a: &[ARow; NA], out: &mut [[f32; NA]; NR]) {
        let m4 = vdupq_n_u8(0x0f);
        let m3 = vdupq_n_u8(0x03);
        let z = vdupq_n_s32(0);
        let mut accf = [[0f32; NA]; NR];
        for b in 0..nb {
            let mut bsv = [[vdupq_n_s16(0); 2]; NA];
            let mut ad = [0f32; NA];
            for i in 0..NA {
                unsafe {
                    let bs = a[i].bs.add(b * 16);
                    bsv[i] = [vld1q_s16(bs), vld1q_s16(bs.add(8))];
                    ad[i] = *a[i].d.add(b);
                }
            }
            for r in 0..NR {
                let blk = unsafe { w.add(r * rb + b * 210) };
                let d = unsafe { rd_f16(blk.add(208)) };
                let scp = unsafe { blk.add(192) as *const i8 };
                let (sc0, sc1) = unsafe { (vmovl_s8(vld1_s8(scp)), vmovl_s8(vld1_s8(scp.add(8)))) };
                let mut isum = [z; NA];
                for n in 0..2 {
                    let (ql, qh) = unsafe { (blk.add(64 * n), blk.add(128 + 32 * n)) };
                    for kk in 0..2 {
                        let (qhv, a0, a1) = unsafe { (vld1q_u8(qh.add(16 * kk)), vld1q_u8(ql.add(16 * kk)), vld1q_u8(ql.add(32 + 16 * kk))) };
                        let q1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(a0, m4), vshlq_n_u8(vandq_u8(qhv, m3), 4)));
                        let q2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(a1, m4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 2), m3), 4)));
                        let q3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(a0, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 4), m3), 4)));
                        let q4 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(a1, 4), vshlq_n_u8(vshrq_n_u8(qhv, 6), 4)));
                        let s = |t: usize| unsafe { *scp.add(8 * n + 2 * t + kk) as i32 };
                        let (s0, s1, s2, s3) = (s(0), s(1), s(2), s(3));
                        for i in 0..NA {
                            unsafe {
                                let aq = a[i].qs.add(b * 256 + 128 * n + 16 * kk);
                                isum[i] = vmlaq_n_s32(isum[i], vdotq_s32(z, q1, vld1q_s8(aq)), s0);
                                isum[i] = vmlaq_n_s32(isum[i], vdotq_s32(z, q2, vld1q_s8(aq.add(32))), s1);
                                isum[i] = vmlaq_n_s32(isum[i], vdotq_s32(z, q3, vld1q_s8(aq.add(64))), s2);
                                isum[i] = vmlaq_n_s32(isum[i], vdotq_s32(z, q4, vld1q_s8(aq.add(96))), s3);
                            }
                        }
                    }
                }
                for i in 0..NA {
                    // Codes were stored +32; subtract 32 * sum(scale_g * sum_g(a)) once per block.
                    let mins = vaddvq_s32(vaddq_s32(
                        vaddq_s32(vmull_s16(vget_low_s16(sc0), vget_low_s16(bsv[i][0])), vmull_high_s16(sc0, bsv[i][0])),
                        vaddq_s32(vmull_s16(vget_low_s16(sc1), vget_low_s16(bsv[i][1])), vmull_high_s16(sc1, bsv[i][1]))));
                    accf[r][i] += ad[i] * d * (vaddvq_s32(isum[i]) - 32 * mins) as f32;
                }
            }
        }
        *out = accf;
    }

    /// Eight weights as two f32x4, whatever the stored float width.
    #[inline(always)]
    unsafe fn wf8(ty: WType, p: *const u8) -> (float32x4_t, float32x4_t) {
        unsafe {
            match ty {
                WType::F32 => (vld1q_f32(p as *const f32), vld1q_f32((p as *const f32).add(4))),
                WType::BF16 => {
                    let h = vld1q_u16(p as *const u16);
                    (vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16)), vreinterpretq_f32_u32(vshll_high_n_u16(h, 16)))
                }
                _ => {
                    let h = vld1q_u16(p as *const u16);
                    let lo: float32x4_t;
                    let hi: float32x4_t;
                    std::arch::asm!("fcvtl {lo:v}.4s, {h:v}.4h", "fcvtl2 {hi:v}.4s, {h:v}.8h",
                                    h = in(vreg) h, lo = out(vreg) lo, hi = out(vreg) hi, options(pure, nomem, nostack));
                    (lo, hi)
                }
            }
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn tile_f<const NR: usize, const NA: usize>(ty: WType, w: *const u8, rb: usize, k: usize,
                                                           a: &[ARow; NA], out: &mut [[f32; NA]; NR]) {
        let es = if ty == WType::F32 { 4 } else { 2 };
        let mut acc = [[[vdupq_n_f32(0.0); 2]; NA]; NR];
        let mut c = 0;
        while c + 8 <= k {
            let mut av = [[vdupq_n_f32(0.0); 2]; NA];
            for i in 0..NA { unsafe { av[i] = [vld1q_f32(a[i].f.add(c)), vld1q_f32(a[i].f.add(c + 4))]; } }
            for r in 0..NR {
                let (w0, w1) = unsafe { wf8(ty, w.add(r * rb + c * es)) };
                for i in 0..NA {
                    acc[r][i][0] = vfmaq_f32(acc[r][i][0], w0, av[i][0]);
                    acc[r][i][1] = vfmaq_f32(acc[r][i][1], w1, av[i][1]);
                }
            }
            c += 8;
        }
        for r in 0..NR {
            for i in 0..NA {
                let mut s = vaddvq_f32(vaddq_f32(acc[r][i][0], acc[r][i][1]));
                for cc in c..k {
                    let wr = unsafe { w.add(r * rb + cc * es) };
                    let wv = unsafe { match ty {
                        WType::F32 => f32::from_bits(rd_u32(wr)),
                        WType::F16 => rd_f16(wr),
                        _ => f32::from_bits((rd_u16(wr) as u32) << 16),
                    } };
                    s += wv * unsafe { *a[i].f.add(cc) };
                }
                out[r][i] = s;
            }
        }
    }

    /// One tile, dispatched on format. `k` is the row's element count.
    #[inline(always)]
    pub unsafe fn tile<const NR: usize, const NA: usize>(ty: WType, w: *const u8, rb: usize, k: usize,
                                                         a: &[ARow; NA], out: &mut [[f32; NA]; NR]) {
        unsafe {
            match ty {
                WType::F32 | WType::F16 | WType::BF16 => tile_f::<NR, NA>(ty, w, rb, k, a, out),
                WType::Q4_K => tile_q45k::<NR, NA, false>(w, rb, k / 256, a, out),
                WType::Q5_K => tile_q45k::<NR, NA, true>(w, rb, k / 256, a, out),
                WType::Q6_K => tile_q6k::<NR, NA>(w, rb, k / 256, a, out),
                _ => tile32::<NR, NA>(ty, w, rb, k / 32, a, out),
            }
        }
    }
}

/// Which kernel family this process runs. Decided once: `FERRIC_CPU_SCALAR=1` forces the portable
/// kernels (the A/B arm, and how the scalar path is exercised on a machine that has NEON).
fn use_neon() -> bool {
    static NEON: OnceLock<bool> = OnceLock::new();
    *NEON.get_or_init(|| {
        if std::env::var("FERRIC_CPU_SCALAR").is_ok() { return false; }
        #[cfg(target_arch = "aarch64")]
        { return std::arch::is_aarch64_feature_detected!("dotprod"); }
        #[allow(unreachable_code)]
        false
    })
}

/// The kernel family in force, for reports: "neon+dotprod" or "scalar".
pub fn kernel_family() -> &'static str { if use_neon() { "neon+dotprod" } else { "scalar" } }

/// Whether activations meeting quantized weights stay f32 (`FERRIC_CPU_F32ACT=1`). See the module doc.
pub fn f32_activations() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("FERRIC_CPU_F32ACT").is_ok())
}

// =====================================================================================================
// 6. The matmul driver
// =====================================================================================================

/// How a matmul runs: which kernel family, and whether quantized weights meet int8 or f32
/// activations. [`Opts::from_env`] is what the model uses; tests pass both arms explicitly, because an
/// env var read once per process cannot reach both in one test binary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Opts {
    /// NEON+DotProd kernels (aarch64 with FEAT_DotProd only; ignored elsewhere).
    pub neon: bool,
    /// Keep activations f32 against quantized weights (weights dequantized row by row).
    pub f32act: bool,
}

impl Opts {
    pub fn from_env() -> Opts { Opts { neon: use_neon(), f32act: f32_activations() } }
    pub fn scalar() -> Opts { Opts { neon: false, f32act: false } }
}

/// `y[m, w.rows] = x[m, w.cols] · wᵀ`.
pub fn matmul(w: &QWeight, x: &[f32], m: usize, y: &mut [f32]) {
    matmul_many(&mut [(w, y)], x, m);
}

/// Several weights against the SAME input in one pool dispatch — q|k|v, gate|up. Each activation
/// format the weights need is quantized once and shared.
pub fn matmul_many(jobs: &mut [(&QWeight, &mut [f32])], x: &[f32], m: usize) {
    matmul_opts(jobs, x, m, Opts::from_env());
}

/// [`matmul_many`] with the kernel family and activation precision chosen by the caller.
pub fn matmul_opts(jobs: &mut [(&QWeight, &mut [f32])], x: &[f32], m: usize, opts: Opts) {
    if jobs.is_empty() || m == 0 { return; }
    let k = jobs[0].0.cols;
    assert_eq!(x.len(), m * k, "activations are [{m}, {}], got {} values", k, x.len());
    for (w, y) in jobs.iter() {
        assert_eq!(w.cols, k, "matmul_many: every weight must take the same input width");
        assert_eq!(y.len(), m * w.rows, "output for [{}, {}] must hold {} values", m, w.rows, m * w.rows);
    }
    let f32act = opts.f32act;
    let need = |a: Act| jobs.iter().any(|(w, _)| !f32act && w.ty.act() == a);
    let q32 = need(Act::Q8x32).then(|| QAct::quantize(x, m, k, 32));
    let q256 = need(Act::Q8x256).then(|| QAct::quantize(x, m, k, 256));
    let arow = |ty: WType, r: usize| -> ARow {
        match (f32act, ty.act()) {
            (false, Act::Q8x32) => q32.as_ref().unwrap().row(r),
            (false, Act::Q8x256) => q256.as_ref().unwrap().row(r),
            _ => ARow { qs: std::ptr::null(), d: std::ptr::null(), bs: std::ptr::null(), f: unsafe { x.as_ptr().add(r * k) } },
        }
    };
    let neon = opts.neon && cfg!(target_arch = "aarch64");
    let nth = pool().threads();
    // Tiles: RC weight rows x AC activation rows per work item. Decode (m = 1) splits rows only, into
    // ~4 items per thread so a slow core does not hold the step; prefill keeps both in L1-sized chunks.
    let (rc_base, ac) = if m == 1 { (0, 1) } else { (16, 16) };
    struct Item { j: usize, r0: usize, r1: usize, a0: usize, a1: usize }
    let mut items = Vec::new();
    for (j, (w, _)) in jobs.iter().enumerate() {
        let rc = if rc_base == 0 { (w.rows / (nth * 4).max(1)).clamp(4, 512).next_multiple_of(4) } else { rc_base };
        let mut r0 = 0;
        while r0 < w.rows {
            let r1 = (r0 + rc).min(w.rows);
            let mut a0 = 0;
            while a0 < m { let a1 = (a0 + ac).min(m); items.push(Item { j, r0, r1, a0, a1 }); a0 = a1; }
            r0 = r1;
        }
    }
    let outs: Vec<(&QWeight, SyncPtr<f32>)> = jobs.iter_mut().map(|(w, y)| (*w, SyncPtr(y.as_mut_ptr()))).collect();
    let run_item = |it: &Item| {
        let (w, yp) = (outs[it.j].0, outs[it.j].1);
        let rb = w.row_bytes();
        let base = w.data.as_ptr();
        let put = |r: usize, a: usize, v: f32| unsafe { *yp.get().add(a * w.rows + r) = v; };
        if f32act && w.ty.act() != Act::F32 {
            // Weights dequantized row by row, activations f32: the arm that should agree with the GPU.
            let mut row = vec![0f32; k];
            for r in it.r0..it.r1 {
                w.dequant_row(r, &mut row);
                for a in it.a0..it.a1 {
                    let xa = &x[a * k..(a + 1) * k];
                    let mut s = [0f32; 8];
                    for (c, (&wv, &xv)) in row.iter().zip(xa).enumerate() { s[c % 8] += wv * xv; }
                    put(r, a, s.iter().sum());
                }
            }
            return;
        }
        if !neon {
            for r in it.r0..it.r1 {
                for a in it.a0..it.a1 {
                    put(r, a, unsafe { scalar::dot(w.ty, base.add(r * rb), k, arow(w.ty, a)) });
                }
            }
            return;
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let mut a = it.a0;
            while a < it.a1 {
                let na = if it.a1 - a >= 4 { 4 } else { 1 };
                let mut r = it.r0;
                while r < it.r1 {
                    let nr = if it.r1 - r >= 4 && na == 1 { 4 } else if it.r1 - r >= 2 { 2 } else { 1 };
                    let wp = base.add(r * rb);
                    match (nr, na) {
                        (4, 1) => { let mut o = [[0f32; 1]; 4]; neon::tile::<4, 1>(w.ty, wp, rb, k, &[arow(w.ty, a)], &mut o); for i in 0..4 { put(r + i, a, o[i][0]); } }
                        (2, 1) => { let mut o = [[0f32; 1]; 2]; neon::tile::<2, 1>(w.ty, wp, rb, k, &[arow(w.ty, a)], &mut o); for i in 0..2 { put(r + i, a, o[i][0]); } }
                        (1, 1) => { let mut o = [[0f32; 1]; 1]; neon::tile::<1, 1>(w.ty, wp, rb, k, &[arow(w.ty, a)], &mut o); put(r, a, o[0][0]); }
                        (2, 4) => {
                            let ar = [arow(w.ty, a), arow(w.ty, a + 1), arow(w.ty, a + 2), arow(w.ty, a + 3)];
                            let mut o = [[0f32; 4]; 2];
                            neon::tile::<2, 4>(w.ty, wp, rb, k, &ar, &mut o);
                            for i in 0..2 { for j in 0..4 { put(r + i, a + j, o[i][j]); } }
                        }
                        _ => {
                            let ar = [arow(w.ty, a), arow(w.ty, a + 1), arow(w.ty, a + 2), arow(w.ty, a + 3)];
                            let mut o = [[0f32; 4]; 1];
                            neon::tile::<1, 4>(w.ty, wp, rb, k, &ar, &mut o);
                            for j in 0..4 { put(r, a + j, o[0][j]); }
                        }
                    }
                    r += nr;
                }
                a += na;
            }
        }
    };
    pool().for_each(items.len(), |i| run_item(&items[i]));
}

/// The dot product of weight row `r` with activation row `x`, through the kernel family in force —
/// a single-row seam for tests that must reach the SHIPPED kernels, not a copy.
pub fn dot_row(w: &QWeight, r: usize, x: &[f32], opts: Opts) -> f32 {
    let mut y = vec![0f32; w.rows];
    matmul_opts(&mut [(w, &mut y)], x, 1, opts);
    y[r]
}

/// `x` as the activation quantizer `w`'s format uses produces it: the int8 codes and one scale per
/// block (`None` when that format meets f32 activations). A test comparing a kernel against an exact
/// reference needs this to separate the kernel's own error from the activation rounding it is
/// specified to have, and to check the quantizer against its specification code for code.
pub fn quantized_activation(ty: WType, x: &[f32], opts: Opts) -> Option<(Vec<i8>, Vec<f32>)> {
    let block = match ty.act() { Act::F32 => return None, Act::Q8x32 => 32, Act::Q8x256 => 256 };
    if opts.f32act { return None; }
    let a = QAct::quantize(x, 1, x.len(), block);
    Some((a.qs, a.d))
}
