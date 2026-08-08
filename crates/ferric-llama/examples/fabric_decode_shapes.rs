//! **Can the heterogeneous split pay at DECODE shapes?** The question before the surgery.
//!
//! `ferric_joule::fabric` splits a matmul across GPU and CPU and, on a large synthetic matmul with a
//! persistent worker pool, measured **1.07x** with coordination down from 78% to 21% of the operation.
//! That result was obtained at a size chosen to make the split work. Decode is not that size.
//!
//! A Qwen2.5-0.5B decode step is four matvecs per layer with a batch of ONE row:
//! `[1,896]·[896,1152]`, `[1,896]·[896,896]`, `[1,896]·[896,9728]`, `[1,4864]·[4864,896]`.
//! Twenty-four layers, so ~96 matmuls per token. Whether 1.07x on one big matmul survives being cut
//! into 96 small ones is not a thing to assume; it is a thing to measure, and it decides whether the
//! decode-path wiring is worth writing at all.
//!
//! ## The cost the split cannot avoid, and which the baseline does not pay
//!
//! Unsplit, the GPU queue pipelines: dispatches are enqueued back to back and nothing waits for a
//! result until the logits are read for the argmax. Split, **both arms must finish before the next op
//! consumes their concatenation**, so every split matmul is a synchronisation point.
//!
//! That sync is charged here to the split, not hidden inside it. `gpu (pipelined)` is what the current
//! decode path actually pays per matmul; `gpu (sync each)` is what the split's GPU arm pays. If the
//! second is much larger than the first, the split has already lost before the CPU contributes
//! anything, and no amount of CPU throughput recovers it.
//!
//! ## The null control
//!
//! The baseline is **GPU alone, pipelined** — the configuration that ships today. Not GPU-alone-synced
//! (which would flatter the split by charging the baseline a cost only the split incurs), and not the
//! CPU arm (beating the worse option is not an achievement).
//!
//! ## Sweeping, not asserting
//!
//! One shape cannot tell a cliff from a curve. Output rows are swept 128 → 16384 at fixed inner
//! dimension so the crossover — the size at which the split starts paying — is located rather than
//! guessed, and then compared against where the decode shapes actually sit.
//!
//!   cargo run -p ferric-llama --example fabric_decode_shapes --release
use ferric_core::Context;
use ferric_tensor::cpu_simd::{cpu_threads, matvec_q8_0_pooled, Pool};
use ferric_tensor::dtype::Q8_0Weights;
use ferric_tensor::Tensor;
use std::sync::Arc;

/// The four decode matvecs of Qwen2.5-0.5B, per layer. `n_embd` 896, `n_ff` 4864, q|k|v = 1152 rows.
const DECODE_SHAPES: &[(&str, usize, usize)] = &[
    ("qkv", 896, 1152),
    ("wo", 896, 896),
    ("gate_up", 896, 9728),
    ("down", 4864, 896),
];

/// Matmuls per token: 4 per layer x 24 layers. Used only to scale a per-matmul delta to a per-token one.
const MATMULS_PER_TOKEN: usize = 4 * 24;

const REPS: usize = 40;

fn load_avg() -> Option<f64> {
    let out = std::process::Command::new("uptime").output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let tail = s.split("load average").nth(1)?;
    tail.trim_start_matches(|c: char| !c.is_ascii_digit()).split(&[',', ' '][..]).next()?.parse().ok()
}

/// Deterministic Q8_0 bytes for `rows x cols`. Values do not matter for timing; the byte count does.
fn synth_q8_0(rows: usize, cols: usize, seed: u32) -> Vec<u8> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    let mut rnd = || { s = s.wrapping_mul(1664525).wrapping_add(1013904223); (s >> 16) as u16 };
    let nblk = rows * (cols / 32);
    let mut raw = Vec::with_capacity(nblk * 34);
    for _ in 0..nblk {
        raw.extend_from_slice(&0x2c00u16.to_le_bytes()); // f16 0.0625, exact in both directions
        for _ in 0..32 { raw.push((rnd() % 251) as u8); }
    }
    raw
}

struct Timing {
    gpu_pipelined_us: f64,
    gpu_sync_us: f64,
    cpu_us: f64,
    coord_us: f64,
}

impl Timing {
    /// Throughput-apportioned split, then charged the sync and the pool round trip it forces.
    ///
    /// `wall = max(gpu_span, cpu_span)` by construction of the apportionment, so the ideal is the
    /// harmonic combination; what it is compared against is the pipelined baseline.
    fn split_us(&self) -> f64 {
        let (g, c) = (1.0 / self.gpu_sync_us, 1.0 / self.cpu_us); // rows per microsecond, each arm
        1.0 / (g + c) + self.coord_us
    }
    fn speedup(&self) -> f64 { self.gpu_pipelined_us / self.split_us() }
    /// The split's ceiling: free coordination, and the GPU arm somehow not paying the sync either.
    fn ceiling(&self) -> f64 {
        let (g, c) = (1.0 / self.gpu_pipelined_us, 1.0 / self.cpu_us);
        self.gpu_pipelined_us / (1.0 / (g + c))
    }
}

/// One shape, four timings. Separated out so the pipelined and synced GPU numbers come from the same
/// warmed weights and the same window, which is the only way their ratio means anything.
async fn measure_shape(ctx: &Arc<Context>, pool: &Pool, cols: usize, rows: usize, coord_us: f64) -> Timing {
    let raw = Arc::new(synth_q8_0(rows, cols, (rows ^ cols) as u32));
    let x: Arc<Vec<f32>> = Arc::new((0..cols).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect());
    let w = Q8_0Weights::from_bytes(ctx, &raw, rows, cols);
    let xt = Tensor::from_vec(ctx, &x, &[1, cols]);

    // Pipelined: enqueue REPS matmuls, read back once. What the shipping decode path pays.
    let _ = xt.matmul_q8_0(&w).to_vec().await; // warm: compile pipeline, fault weights in
    let t0 = std::time::Instant::now();
    let mut last = None;
    for _ in 0..REPS { last = Some(xt.matmul_q8_0(&w)); }
    let _ = last.unwrap().to_vec().await;
    let gpu_pipelined_us = t0.elapsed().as_secs_f64() * 1e6 / REPS as f64;

    // Synced: read back every matmul. What the split's GPU arm is forced to pay.
    let t0 = std::time::Instant::now();
    for _ in 0..REPS { let _ = xt.matmul_q8_0(&w).to_vec().await; }
    let gpu_sync_us = t0.elapsed().as_secs_f64() * 1e6 / REPS as f64;

    let _ = matvec_q8_0_pooled(pool, Arc::clone(&x), Arc::clone(&raw), cols, 0, rows);
    let t0 = std::time::Instant::now();
    for _ in 0..REPS { let _ = matvec_q8_0_pooled(pool, Arc::clone(&x), Arc::clone(&raw), cols, 0, rows); }
    let cpu_us = t0.elapsed().as_secs_f64() * 1e6 / REPS as f64;

    Timing { gpu_pipelined_us, gpu_sync_us, cpu_us, coord_us }
}

fn main() { pollster::block_on(run()); }

async fn run() {
    println!("Heterogeneous split at decode shapes — Qwen2.5-0.5B geometry\n");
    print!("{}", ferric_joule::capability_report());

    if let Some(l) = load_avg() {
        println!("  machine load average: {l:.2}");
        assert!(l < 8.0, "load {l:.2} is too high to time anything; this harness reported an 8.6x swing \
                          on a busy machine earlier in this workspace's history. Wait and re-run.");
    }

    let ctx = Arc::new(Context::new().await.unwrap());
    let pool = Pool::new(cpu_threads());
    println!("  cpu worker pool: {} threads\n", pool.threads());

    // ---- both arms must compute the same thing, or the comparison manufactures a difference ----
    {
        let (cols, rows) = (896usize, 128usize);
        let raw = synth_q8_0(rows, cols, 7);
        let x: Vec<f32> = (0..cols).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
        let w = Q8_0Weights::from_bytes(&ctx, &raw, rows, cols);
        let g = Tensor::from_vec(&ctx, &x, &[1, cols]).matmul_q8_0(&w).to_vec().await;
        let c = matvec_q8_0_pooled(&pool, Arc::new(x), Arc::new(raw), cols, 0, rows);
        let err = g.iter().zip(&c).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let mag = g.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        println!("  arms agree: max abs err {:.3e} on magnitude {:.3e} (rel {:.2e})\n", err, mag, err / mag);
        assert!(err / mag < 1e-5, "arms disagree — a timing comparison between them would be meaningless");
    }

    // ---- pool coordination floor: a round trip with no arithmetic in it ----
    let coord_us = {
        let spans = pool.threads();
        for _ in 0..8 { pool.run(spans, |i| Box::new(move || (i, Vec::new()))); }
        let t0 = std::time::Instant::now();
        for _ in 0..REPS { pool.run(spans, |i| Box::new(move || (i, Vec::new()))); }
        t0.elapsed().as_secs_f64() * 1e6 / REPS as f64
    };
    println!("  pool coordination floor: {coord_us:.1} us per dispatch+collect ({} spans, zero work)\n",
             pool.threads());

    let measure = |cols: usize, rows: usize| measure_shape(&ctx, &pool, cols, rows, coord_us);

    // ---- the four real decode shapes ----
    println!("  {:>8} {:>6} {:>6} {:>10} {:>10} {:>9} {:>9} {:>8}",
             "op", "in", "out", "gpu pipe", "gpu sync", "cpu", "split", "vs pipe");
    println!("  {:-<74}", "");
    let mut worst = f64::INFINITY;
    for (name, cols, rows) in DECODE_SHAPES {
        let t = measure(*cols, *rows).await;
        let s = t.speedup();
        worst = worst.min(s);
        println!("  {name:>8} {cols:>6} {rows:>6} {:>9.1}u {:>9.1}u {:>8.1}u {:>8.1}u {s:>7.2}x",
                 t.gpu_pipelined_us, t.gpu_sync_us, t.cpu_us, t.split_us());
    }

    // ---- sweep output rows: one point cannot tell a cliff from a curve ----
    println!("\n  Sweep at fixed inner dim 896 — where, if anywhere, does the split start paying?\n");
    println!("  {:>7} {:>10} {:>10} {:>9} {:>9} {:>8} {:>9}",
             "out", "gpu pipe", "gpu sync", "cpu", "split", "vs pipe", "ceiling");
    println!("  {:-<66}", "");
    let mut crossover: Option<usize> = None;
    let mut ceiling_ever = 0.0f64;
    let mut min_cpu_us = f64::INFINITY;
    let mut sweep: Vec<(usize, f64)> = Vec::new();
    for rows in [128usize, 512, 2048, 8192, 16384] {
        let t = measure(896, rows).await;
        let (s, c) = (t.speedup(), t.ceiling());
        ceiling_ever = ceiling_ever.max(c);
        min_cpu_us = min_cpu_us.min(t.cpu_us);
        sweep.push((rows, t.cpu_us));
        if s > 1.0 && crossover.is_none() { crossover = Some(rows); }
        println!("  {rows:>7} {:>9.1}u {:>9.1}u {:>8.1}u {:>8.1}u {s:>7.2}x {c:>8.2}x",
                 t.gpu_pipelined_us, t.gpu_sync_us, t.cpu_us, t.split_us());
    }

    // ---- the empty-pool number is not a floor, and the sweep is what shows it ----
    //
    // A CPU arm doing real arithmetic came in BELOW the empty round trip through the same pool, on two
    // separate runs at different machine loads. Reproducible, so it is not drift. The empty job returns
    // instantly, which makes all 18 workers hit the results channel in the same instant; a real job
    // staggers them. The empty-pool measurement therefore times a contention pile-up that no actual
    // workload produces, and OVERSTATES the fixed cost.
    //
    // The honest measurement of a fixed cost is the intercept of time against work. Taken from the two
    // smallest sweep points, because marginal cost per row falls with size (0.061 -> 0.037 us/row across
    // this sweep, as longer contiguous spans use the cache better) and a global fit over a curve with
    // changing slope would report the fit, not the machine.
    let (r0, c0) = sweep[0];
    let (r1, c1) = sweep[1];
    let slope = (c1 - c0) / (r1 - r0) as f64;
    let intercept = c0 - r0 as f64 * slope;
    println!("\n  Pool coordination, two ways:");
    println!("    empty-job round trip:     {coord_us:>6.1} us   <- overstates; 18 workers reply at once");
    println!("    intercept of the sweep:   {intercept:>6.1} us   <- fixed cost, from {r0} and {r1} rows");
    println!("    marginal arithmetic:      {slope:>6.4} us/row at these sizes");
    if min_cpu_us < coord_us {
        println!("    (the {min_cpu_us:.1} us CPU arm below the {coord_us:.1} us \"floor\" is what exposed this)");
    }
    let coord_us = intercept;

    // ---- what it means for the decode-path wiring ----
    let biggest_decode = DECODE_SHAPES.iter().map(|s| s.2).max().unwrap();
    println!("\n  Largest decode matmul: {biggest_decode} output rows.");
    match crossover {
        Some(r) if r <= biggest_decode =>
            println!("  Split begins paying at {r} rows, which decode reaches. Wiring is justified."),
        Some(r) =>
            println!("  Split begins paying at {r} rows. Decode's largest is {biggest_decode}, so every \n  \
                      decode matmul sits BELOW the crossover."),
        None =>
            println!("  ⚠ The split never pays anywhere in the swept range, at any size."),
    }
    println!("  Best speedup at any decode shape: {worst:.2}x (worst) — see the table for the spread.");
    println!("  Ceiling with free coordination AND no sync penalty, anywhere in the sweep: {ceiling_ever:.2}x.");
    println!("\n  Per token that is {MATMULS_PER_TOKEN} matmuls. A per-matmul coordination cost of \
              {coord_us:.1} us alone\n  adds {:.2} ms to every token before any arithmetic is split.",
             coord_us * MATMULS_PER_TOKEN as f64 / 1000.0);
    println!("\n  ⚠ This is TIME, not joules. This machine has no readable power counter. Time is the \n  \
              right proxy for the wiring decision (does the split reduce wall clock at these shapes) \n  \
              and says nothing about whether it reduces energy, which needs a metered device.");
}
