//! **One matmul, two compute units, at the same time.**
//!
//! Ferric reached only the GPU. This runs the same Q8_0 weight across the GPU and the CPU's vector units
//! CONCURRENTLY, splitting the output rows between them in proportion to each unit's MEASURED throughput.
//!
//! Two things must hold, and the second is the one that is easy to lose:
//!
//!   1. the joined result must be identical to running the whole thing on either unit alone;
//!   2. the units must actually overlap in time, or the split is strictly worse than not splitting.
//!
//! The second is why wall time is reported as max, not sum, and why the split is compared against the
//! BEST single unit rather than the worst.
use ferric_core::Context;
use ferric_joule::{Fabric, Unit};
use ferric_tensor::{cpu, Q8_0Weights, Tensor};
use std::sync::Arc as StdArc;
use std::sync::Arc;

fn load_avg() -> Option<f64> {
    let out = std::process::Command::new("uptime").output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let tail = s.split("load average").nth(1)?;
    tail.trim_start_matches(|c: char| !c.is_ascii_digit()).split(&[',', ' '][..]).next()?.parse().ok()
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let (rows, cols) = (32768usize, 4096usize);
    let threads = cpu::cpu_threads();
    println!("Heterogeneous split — one Q8_0 matmul across GPU and CPU\n");
    println!("  weight [{rows}, {cols}] Q8_0, {:.1} MB.  CPU threads available: {threads}\n",
             (rows * (cols / 32) * 34) as f64 / 1e6);

    // Deterministic Q8_0 blocks.
    let nblk = rows * (cols / 32);
    let mut raw = Vec::with_capacity(nblk * 34);
    let mut s = 12345u32;
    let mut rnd = || { s = s.wrapping_mul(1664525).wrapping_add(1013904223); (s >> 16) as u16 };
    for _ in 0..nblk {
        raw.extend_from_slice(&(0x1C00u16 | (rnd() & 0x3F)).to_le_bytes());
        for _ in 0..32 { raw.push((rnd() & 0xFF) as u8); }
    }
    let xv: Vec<f32> = (0..cols).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();

    // ---- correctness first: both units must agree before any timing is reported ----
    let w_gpu = Q8_0Weights::from_bytes(&ctx, &raw, rows, cols);
    let x = Tensor::from_vec(&ctx, &xv, &[1, cols]);
    let gpu_all = x.matmul_q8_0(&w_gpu).to_vec().await;
    let cpu_all = cpu::matvec_q8_0_threaded(&xv, &raw, rows, cols, threads);
    let scale = gpu_all.iter().fold(1e-6f32, |a, &v| a.max(v.abs()));
    let d = gpu_all.iter().zip(&cpu_all).fold(0f32, |a, (&g, &c)| a.max((g - c).abs())) / scale;
    println!("  GPU vs CPU on the whole matmul: max relative Δ {d:.3e}");
    assert!(d < 1e-4, "the two units disagree by {d:.3e}; a split would be silently wrong");

    let Some(load) = load_avg() else { return };
    println!("  machine load average: {load:.2}");
    if load >= 8.0 {
        println!("\n  ⚠ too loaded to time anything. Correctness above still holds; re-run when quiet.");
        return;
    }

    // ---- calibrate each unit on the REAL work, never an assumed ratio ----
    let mut fab = Fabric::new();
    let gc = fab.calibrate::<ferric_joule::Nameplate>(Unit::Gpu, None, rows as u64, |_| {
        pollster::block_on(async { let _ = x.matmul_q8_0(&w_gpu).to_vec().await; });
    });
    let cc = fab.calibrate::<ferric_joule::Nameplate>(Unit::CpuSimd, None, rows as u64, |n| {
        let _ = cpu::matvec_q8_0_threaded(&xv, &raw, n as usize, cols, threads);
    });
    println!("\n  calibrated: gpu {:.0} rows/s, cpu-simd {:.0} rows/s", gc.throughput, cc.throughput);

    let split = fab.split(rows as u64).expect("both units calibrated");
    println!("{split}");
    match split.worthwhile() {
        Ok(()) => println!("  -> worth splitting"),
        Err(why) => println!("  -> NOT worth splitting: {why}"),
    }

    // ---- run it: GPU and CPU concurrently on their own spans ----
    let gpu_rows = split.shares.iter().find(|(c, _)| c.unit == Unit::Gpu).map(|(_, n)| *n as usize).unwrap_or(0);
    let cpu_rows = rows - gpu_rows;
    let row_bytes = (cols / 32) * 34;
    let w_gpu_part = Q8_0Weights::from_bytes(&ctx, &raw[..gpu_rows * row_bytes], gpu_rows, cols);

    // Persistent pool: created ONCE, outside the timed region, which is the whole point. Per-call
    // std::thread::scope spent 78% of the wall clock on spawn and join.
    let pool = cpu::Pool::new(threads);
    let xa = StdArc::new(xv.clone());
    let wa = StdArc::new(raw.clone());
    // Warm the pool and the GPU pipeline so neither arm pays first-call cost inside the measurement.
    let _ = cpu::matvec_q8_0_pooled(&pool, StdArc::clone(&xa), StdArc::clone(&wa), cols, gpu_rows, rows);
    let _ = x.matmul_q8_0(&w_gpu_part).to_vec().await;

    let t0 = std::time::Instant::now();
    // The CPU arm runs on parked workers while the GPU arm is in flight. This overlap IS the win.
    let (gpu_part, cpu_part) = std::thread::scope(|sc| {
        let h = sc.spawn(|| cpu::matvec_q8_0_pooled(&pool, StdArc::clone(&xa), StdArc::clone(&wa), cols, gpu_rows, rows));
        let g = pollster::block_on(async { x.matmul_q8_0(&w_gpu_part).to_vec().await });
        (g, h.join().expect("cpu arm panicked"))
    });
    let wall = t0.elapsed().as_secs_f64();

    let mut joined = gpu_part;
    joined.extend(cpu_part);
    assert_eq!(joined.len(), rows, "the split lost rows");
    let dj = gpu_all.iter().zip(&joined).fold(0f32, |a, (&g, &j)| a.max((g - j).abs())) / scale;
    println!("\n  joined result vs whole-on-GPU: max relative Δ {dj:.3e}  ({gpu_rows} gpu + {cpu_rows} cpu rows)");
    assert!(dj < 1e-4, "the joined split differs by {dj:.3e}");

    let best = gc.throughput.max(cc.throughput);
    let single = rows as f64 / best;
    let real = split.measured_speedup(wall, single);
    println!("\n  {:<28} {:>10.4} s", "measured split (concurrent)", wall);
    println!("  {:<28} {:>10.4} s", "best single unit alone", single);
    println!("  {:<28} {:>10.2}x   (calibration predicted {:.2}x)",
             "MEASURED speedup", real, split.speedup_vs_best_single());
    println!("  {:<28} {:>10.4} s", "coordination cost", split.coordination_seconds(wall));

    println!("\n  Correctness holds regardless: identical answer from two units running at once, and the");
    println!("  split ratio came from measuring each unit on this exact work rather than a guess.\n");
    if real >= 1.0 {
        println!("  ✅ And it PAYS here: {real:.2}x over the best single unit.");
    } else {
        println!("  ⚠ It does NOT pay at this size on this machine: {real:.2}x, i.e. a regression. The");
        println!("  coordination cost (thread spawn and join, a second dispatch, buffer setup) exceeds");
        println!("  what the second unit contributes. Calibration cannot see that cost, which is why");
        println!("  worthwhile() is documented as permission to run the A/B and never as its result.");
        println!("  Heterogeneous execution is correct here and not yet profitable; the fix is a");
        println!("  persistent worker pool rather than per-call spawn, and a larger unit of work.");
    }
}
