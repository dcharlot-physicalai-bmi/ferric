//! **What does a packed quant kernel cost in joules, not bytes?**
//!
//! The packed kernels were justified on residency: STQ1_0 at 1.3125 bpw is 24.4× smaller than the
//! f32 it used to be dequantised into. That is a memory claim and it is settled. The energy claim is
//! a different one and it is not obvious in either direction:
//!
//! * a packed kernel moves 10–24× fewer weight bytes per matmul, and moving bytes is most of what an
//!   inference GPU spends energy on;
//! * but it unpacks every weight in the shader — nibble extracts, table lookups, sign expansion —
//!   and issue slots are not free either.
//!
//! So this measures it rather than assuming it. Both arms compute the same matmul from the same
//! weights; only the representation differs.
//!
//! ⚠ **This is a kernel measurement, not a task measurement.** A joules-per-matmul figure is not a
//! joules-per-token figure: a real forward pass has attention, a KV cache, routing, and a host loop,
//! and their share moves the answer. Nothing here should be quoted as the energy of running a model.
//!
//! ⚠ **The boundary is the accelerator rail — `gpu + ram`, not the whole SoC.** That is not the
//! first choice; it is what a shared laptop leaves measurable. Metered at the SoC, an ambient CPU
//! load of 10–19 W from an editor and a compiler swamps a GPU kernel drawing a fraction of a watt,
//! and the first version of this benchmark duly measured the ambient load and reported a marginal
//! energy *below zero*. `gpu + ram` excludes the host cost of driving the kernel — command
//! encoding, submission, the driver — which is real work that is genuinely not counted here. Both
//! arms issue the same number of dispatches, so it cancels in the ratio and not in the absolute.
//!
//! ⚠ It is also **steady-state**: both arms assume the weights are already resident. The dense arm
//! holds them at 32 bits per weight, which for the checkpoints these formats exist for does not fit
//! at all — and "does not fit" has no joules figure, which is exactly why the residency number and
//! this one are separate claims.
//!
//! ```text
//! cargo run --release -p ferric-tensor --example quant_kernel_joules
//! ```

use ferric_joule::{compare, Macmon, MacmonScope, Meter};
use ferric_tensor::dtype::QMatrix;
use ferric_tensor::Tensor;
use ferric_core::Context;
use std::sync::Arc;
use std::time::Instant;

const K: usize = 4096; // in features
const N: usize = 4096; // out features

fn lcg(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

/// Random block bytes with the scale field pinned to something sane. Every byte pattern is a legal
/// block in all three formats, but a random f16 scale is happily NaN or 65504.
fn blocks(n_blocks: usize, bpb: usize, scale_at: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut v: Vec<u8> = (0..n_blocks * bpb).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as u8
    }).collect();
    for (i, blk) in v.chunks_exact_mut(bpb).enumerate() {
        let d = half::f16::from_f32(0.008 + 0.002 * (i % 5) as f32);
        blk[scale_at..scale_at + 2].copy_from_slice(&d.to_le_bytes());
    }
    v
}

fn main() {
    let Ok(ctx) = pollster::block_on(Context::new()) else { eprintln!("no GPU context"); return };
    let ctx = Arc::new(ctx);

    let Some(meter) = Macmon::start(MacmonScope::Accelerator, 100) else {
        eprintln!("macmon not available — install it, or run plugged in; refusing to report zeros");
        return;
    };
    // Give the sampler a moment to produce its first line, then confirm it is actually reading.
    std::thread::sleep(std::time::Duration::from_millis(600));
    assert!(meter.available(), "macmon produced no usable sample; a zero here is not a measurement");

    let mut seed = 20260901u64;
    let xv: Vec<f32> = (0..K).map(|_| lcg(&mut seed)).collect();
    let x = Tensor::from_vec(&ctx, &xv, &[1, K]); // decode shape: one token, the bandwidth-bound case

    // Idle draw, measured in THIS process adjacent in time — never a remembered constant.
    //
    // ⛔ Idle is a FLOOR, not an average. The first version of this took one 3-second window and got
    // 47.47 W, because a `rustc` was finishing in the background; every arm then measured "less than
    // idle" and the impossibility check below correctly threw the whole run away. The real floor on
    // this machine is about 1 W. So: several short windows, keep the minimum, and refuse outright if
    // even the minimum is high — a contended machine cannot produce an attributable difference, and
    // saying so is the only honest output.
    let floor = |label: &str| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..6 {
            let e0 = meter.read_joules().expect("meter went away");
            let t0 = Instant::now();
            std::thread::sleep(std::time::Duration::from_millis(700));
            let e1 = meter.read_joules().expect("meter went away");
            best = best.min((e1 - e0) / t0.elapsed().as_secs_f64());
        }
        println!("idle SoC floor ({label}): {best:.2} W  (min of six 0.7 s windows)");
        best
    };
    // On the accelerator rail an idle machine sits near zero; anything above this is another
    // process using the GPU, which does contaminate the comparison.
    const QUIET: f64 = 1.0;
    let idle = floor("before");
    if idle > QUIET {
        println!("\n  REFUSING TO REPORT. An idle floor of {idle:.2} W means something else is using this\n                    machine, and a difference measured against a moving baseline is not attributable to\n                    the kernels. Wait for the machine to be quiet and run again.");
        return;
    }
    println!();

    println!("boundary: {} — the accelerator rail only, host dispatch NOT counted", meter.source());

    // ── Throughput calibration ────────────────────────────────────────────────────────────────
    //
    // ⛔ An idle-floor gate cannot see THROTTLING. A machine that has run its battery flat is
    // perfectly quiet — every floor reading below stayed under a watt — and runs its GPU at a third
    // of its usual clock. One such run reported the dense arm at 40 GB/s against the 145 GB/s it
    // reaches normally, and every ratio built on it was wrong by more than the effect being
    // measured. Silence is not the same as readiness.
    //
    // The detector is a known-good workload, not a quieter idle window. The dense f32 matmul is
    // stable and well characterised, so it doubles as a calibration arm: if it is not hitting its
    // usual rate, this machine is not in a state where anything else measured on it is comparable.
    //
    // The floor is set from the observed healthy band, not guessed: five clean runs put the dense
    // arm at 143.6, 144.9, 148.0, 151.0 and 151.2 GB/s — a spread of ±3%. A throttled machine read
    // 40 GB/s, and a half-recovered one 101.7, which the first version of this gate let through at a
    // floor of 100. 130 sits well below anything healthy and well above anything throttled.
    const DENSE_GBS_FLOOR: f64 = 130.0;
    {
        let w = Tensor::from_vec(&ctx, &vec![0.01f32; N * K], &[N, K]);
        let t = Instant::now();
        for _ in 0..400 { let _ = x.matmul_bt(&w); }
        ferric_tensor::device_sync(&ctx);
        let gbs = (N * K * 4 * 400) as f64 / t.elapsed().as_secs_f64() / 1e9;
        println!("calibration: dense f32 matmul at {gbs:.1} GB/s");
        if gbs < DENSE_GBS_FLOOR {
            println!("\n  REFUSING TO REPORT. The dense reference reaches {gbs:.1} GB/s against the\n                        {DENSE_GBS_FLOOR:.0} GB/s this machine does when it is not throttled. Check the battery\n                        (a flat one throttles hard and stays perfectly quiet while doing it) and the\n                        thermal state, then run again.");
            return;
        }
    }
    println!();
    println!("{:<9} {:>10} {:>9} {:>8} {:>9} {:>9} {:>8} {:>9}",
             "format", "resident", "bpw", "iters", "J/matmul", "W", "GB/s", "vs dense");

    for (label, ty, bpb, scale_at) in
        [("stq1_0", 43u32, 42usize, 40usize), ("iq2_xxs", 16, 66, 0), ("iq3_xxs", 18, 98, 0)]
    {
        let bytes = blocks(N * (K / 256), bpb, scale_at, seed ^ ty as u64);
        let packed = QMatrix::from_bytes(&ctx, &bytes, ty, N, K).expect("packed load");
        let deq = ferric_gguf::deq_raw(&bytes, N * K, ty).expect("decode");
        let dense = Tensor::from_vec(&ctx, &deq, &[N, K]);

        // Same computation, or the comparison is between two different things.
        let a = pollster::block_on(x.matmul_q(&packed).to_vec());
        let b = pollster::block_on(x.matmul_bt(&dense).to_vec());
        let mag = b.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-4 * mag, "{label}: the two arms do not compute the same matmul ({worst})");

        // Size each arm to ~2.5 s so `Saving::claimable` is satisfied and the sampler has many
        // periods inside the window. Both arms get the SAME iteration count — the packed arm being
        // faster is part of the result, not something to normalise away.
        let probe = |f: &dyn Fn()| { let t = Instant::now(); for _ in 0..300 { f() } ferric_tensor::device_sync(&ctx); t.elapsed().as_secs_f64() / 300.0 };
        let packed_run = || { let _ = x.matmul_q(&packed); };
        let dense_run = || { let _ = x.matmul_bt(&dense); };
        // ⛔ Size on the FASTEST arm, not the slowest. Sizing on the slowest gives it 2.5 s and
        // leaves the quicker one under a second, where `Saving::claimable` rightly refuses it — the
        // first version of this did exactly that and threw away two of the three formats.
        let fastest = probe(&packed_run).min(probe(&dense_run));
        let iters = ((2.6 / fastest).ceil() as usize).clamp(50, 8_000_000);

        // The floor moves. Re-measure it per format rather than trusting one reading taken minutes
        // ago, and attribute each format's marginal to the floor beside it.
        let idle = floor(label);
        if idle > QUIET { println!("  {label}: SKIPPED — machine busy ({idle:.2} W floor)"); continue }

        let saving = compare(&meter, iters as u64, (iters as u64, iters as u64), 2,
            || { for _ in 0..iters { dense_run() } ferric_tensor::device_sync(&ctx); },
            || { for _ in 0..iters { packed_run() } ferric_tensor::device_sync(&ctx); },
        ).expect("meter went away mid-run");

        if let Err(e) = saving.claimable() {
            println!("  {label}: NOT CLAIMABLE — {e}\n            (dense {:.2} s, {label} {:.2} s over {iters} matmuls)",
                     saving.baseline.seconds, saving.candidate.seconds);
            continue
        }

        let report = |r: &ferric_joule::Reading, wbytes: usize| {
            let per = r.joules / iters as f64;
            let marginal = (r.joules - idle * r.seconds) / iters as f64;
            let gbs = (wbytes * iters) as f64 / r.seconds / 1e9;
            (per, marginal, r.watts(), gbs)
        };
        let (dj, dm, dw, dg) = report(&saving.baseline, N * K * 4);
        let (pj, pm, pw, pg) = report(&saving.candidate, bytes.len());

        // Physical impossibility check: work cannot draw less than the machine drew doing nothing.
        for (l, m, w) in [("dense", dm, dw), (label, pm, pw)] {
            assert!(m > 0.0, "{l}: marginal energy is {m:.3e} J/matmul at {w:.2} W against a \
                              {idle:.2} W idle — the baseline was contended, so this run is invalid");
        }

        let bpw = bytes.len() as f64 * 8.0 / (N * K) as f64;
        // The dense arm is re-measured per format, not carried over — it is paired with THIS
        // format's candidate inside one alternating comparison, and quoting one dense figure for
        // all three would be quoting a number that was never beside two of them.
        println!("{:<9} {:>9} B {:>9} {:>8} {:>9.3e} {:>9.2} {:>8.1} {:>9}",
                 "f32 dense", N * K * 4, "32.0", iters, dj, dw, dg, "1.00x");
        println!("{:<9} {:>9} B {:>9.4} {:>8} {:>9.3e} {:>9.2} {:>8.1} {:>8.2}x",
                 label, bytes.len(), bpw, iters, pj, pw, pg, dj / pj);
        println!("            marginal over idle: dense {dm:.3e} J  {label} {pm:.3e} J ({:.2}x) | \
                  wall {:.2} s -> {:.2} s ({:.2}x)",
                 dm / pm, saving.baseline.seconds, saving.candidate.seconds,
                 saving.baseline.seconds / saving.candidate.seconds);
    }

    let after = floor("after");
    if after > QUIET {
        println!("  ⚠ the machine became busy DURING the run ({after:.2} W floor afterwards); the\n                      numbers above are contaminated and should not be quoted.");
    }

    // ── The forcing function, measured ────────────────────────────────────────────────────────
    //
    // The table above says the packed kernels are ALU-bound, not bandwidth-bound. If that is right,
    // the way to convert the byte saving into joules is to spend fewer instructions per weight —
    // not to move fewer bytes, which is already done. STQ1_0 has two traversal orders that compute
    // exactly the same thing and differ only in how many loads they issue, so this measures the
    // claim rather than asserting it.
    {
        let bytes = blocks(N * (K / 256), 42, 40, seed ^ 0xabc);
        let packed = ferric_tensor::dtype::Stq1_0Weights::from_bytes(&ctx, &bytes, N, K);
        let a = pollster::block_on(x.matmul_stq1_0_form(&packed, ferric_tensor::dtype::Stq1Form::Vec4).to_vec());
        let b = pollster::block_on(x.matmul_stq1_0_form(&packed, ferric_tensor::dtype::Stq1Form::Vec4Table).to_vec());
        let mag = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-4 * mag, "the two traversal orders disagree by {worst}; not the same matmul");

        let scal = || { let _ = x.matmul_stq1_0_form(&packed, ferric_tensor::dtype::Stq1Form::Vec4); };
        let vec4 = || { let _ = x.matmul_stq1_0_form(&packed, ferric_tensor::dtype::Stq1Form::Vec4Table); };
        let probe = |f: &dyn Fn()| { let t = Instant::now(); for _ in 0..300 { f() } ferric_tensor::device_sync(&ctx); t.elapsed().as_secs_f64() / 300.0 };
        let iters = ((2.6 / probe(&scal).min(probe(&vec4))).ceil() as usize).clamp(50, 8_000_000);

        let idle = floor("stq1_0 traversal");
        if idle <= QUIET {
            if let Some(sv) = compare(&meter, iters as u64, (iters as u64, iters as u64), 2,
                || { for _ in 0..iters { scal() } ferric_tensor::device_sync(&ctx); },
                || { for _ in 0..iters { vec4() } ferric_tensor::device_sync(&ctx); }) {
                match sv.claimable() {
                    Err(e) => println!("  stq1_0 traversal: NOT CLAIMABLE — {e}"),
                    Ok(()) => {
                        let (sj, vj) = (sv.baseline.joules / iters as f64, sv.candidate.joules / iters as f64);
                        println!("\nstq1_0 traversal order, same arithmetic, {iters} matmuls:");
                        println!("  vec4 + private codebook     {sj:.3e} J  {:.2} W  {:.2} s",
                                 sv.baseline.watts(), sv.baseline.seconds);
                        println!("  vec4 + table + transpose    {vj:.3e} J  {:.2} W  {:.2} s  ({:.2}x energy, {:.2}x wall)",
                                 sv.candidate.watts(), sv.candidate.seconds, sj / vj,
                                 sv.baseline.seconds / sv.candidate.seconds);
                    }
                }
            }
        } else { println!("  stq1_0 traversal: SKIPPED — machine busy ({idle:.2} W floor)"); }
    }

    println!("\n  A joules-per-matmul figure is not a joules-per-token figure. This is one kernel with\n  \
              the weights already resident; the dense arm of the checkpoints these formats exist for\n  \
              does not fit in memory at all, and that has no joules number.");
}
