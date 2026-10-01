//! **GPU time per kernel, from the GPU's own clock** (`FERRIC_KPROF=1`, needs `FERRIC_GPUPROF=1` so the
//! device is created with `TIMESTAMP_QUERY`).
//!
//! Every dispatch gets its own compute pass with a begin/end timestamp; at each submit the stamps are
//! resolved, read back and charged to the dispatch's label. The answer to "which kernel is the decode
//! step spending its time in", which the category profiler (a host sync per category) and per-op
//! microbenchmarks (other cache states, other neighbours) each answered wrongly before.
//!
//! ⚠ It changes what it measures, a little: one pass per dispatch (the batch normally shares one) and a
//! readback per submit. The per-kernel GPU durations are the quantity; wall time under it is not. Off
//! unless asked, and the hot path pays one cached flag test.
use std::cell::RefCell;
use std::collections::HashMap;

const CAP: u32 = 4096; // timestamps per query set (wgpu's maximum): 2048 dispatches between resolves

struct Prof {
    qset: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    next: u32,
    labels: Vec<String>,
    totals: HashMap<String, (f64, u64)>, // label -> (GPU ns, dispatches)
}

thread_local! { static P: RefCell<Option<Prof>> = const { RefCell::new(None) }; }

/// Profiling on for this context: `FERRIC_KPROF` set and the device has timestamp queries.
pub(crate) fn on(ctx: &ferric_core::Context) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    ctx.timestamps && *ON.get_or_init(|| std::env::var_os("FERRIC_KPROF").is_some())
}

/// The next (query set, begin index) for a dispatch labelled `label`; `None` when the set is full
/// (the caller then records without stamps — resolve() empties it at the next submit).
pub(crate) fn slot(ctx: &ferric_core::Context, label: &str) -> Option<(wgpu::QuerySet, u32)> {
    P.with(|p| {
        let mut p = p.borrow_mut();
        let pr = p.get_or_insert_with(|| Prof {
            qset: ctx.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("kprof"), ty: wgpu::QueryType::Timestamp, count: CAP }),
            resolve: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("kprof.resolve"), size: CAP as u64 * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }),
            next: 0, labels: Vec::new(), totals: HashMap::new(),
        });
        if pr.next + 2 > CAP { return None; }
        let i = pr.next;
        pr.next += 2;
        pr.labels.push(label.to_string());
        Some((pr.qset.clone(), i))
    })
}

/// Resolve and charge every stamped dispatch so far. Call after the work that wrote the stamps has been
/// submitted (it submits its own resolve and waits for it).
pub(crate) fn resolve(ctx: &ferric_core::Context) {
    P.with(|p| {
        let mut p = p.borrow_mut();
        let Some(pr) = p.as_mut() else { return };
        if pr.next == 0 { return; }
        let n = pr.next;
        let read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kprof.read"), size: n as u64 * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        enc.resolve_query_set(&pr.qset, 0..n, &pr.resolve, 0);
        enc.copy_buffer_to_buffer(&pr.resolve, 0, &read, 0, n as u64 * 8);
        ctx.queue.submit([enc.finish()]);
        let (tx, rx) = flume::bounded(1);
        read.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        if rx.recv().map(|r| r.is_ok()).unwrap_or(false) {
            let data = read.slice(..).get_mapped_range().expect("mapped");
            let ts: &[u64] = bytemuck::cast_slice(&data);
            for (k, label) in pr.labels.iter().enumerate() {
                let (b, e) = (ts[2 * k], ts[2 * k + 1]);
                let ns = e.saturating_sub(b) as f64 * ctx.timestamp_period as f64;
                let t = pr.totals.entry(label.clone()).or_insert((0.0, 0));
                t.0 += ns; t.1 += 1;
            }
            drop(data);
            read.unmap();
        }
        pr.next = 0;
        pr.labels.clear();
    });
}

/// Per-label (GPU ns, dispatches) since the last reset, largest first.
pub fn report() -> Vec<(String, f64, u64)> {
    P.with(|p| {
        let p = p.borrow();
        let mut v: Vec<(String, f64, u64)> = p.as_ref().map(|pr| pr.totals.iter().map(|(k, &(ns, c))| (k.clone(), ns, c)).collect()).unwrap_or_default();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        v
    })
}

pub fn reset() { P.with(|p| if let Some(pr) = p.borrow_mut().as_mut() { pr.totals.clear(); }); }
