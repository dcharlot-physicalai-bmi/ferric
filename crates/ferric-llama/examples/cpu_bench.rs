//! **The CPU fabric's prefill and decode speed**, in the shapes `llama-bench` reports, so the two can be
//! read side by side on the same machine and the same file.
//!
//!   FERRIC_CPU=1 cargo run -p ferric-llama --release --example cpu_bench -- <model.gguf> [pp] [tg] [reps]
//!
//! `pp` (default 512): one prefill of `pp` tokens into an EMPTY cache — llama-bench's `pp512`.
//! `tg` (default 128): `tg` single-token steps from an EMPTY cache — llama-bench's `tg128`.
//! Arms are interleaved over `reps` (default 5), after one untimed warm-up of each, and every figure is
//! reported as median (min..max): this machine is shared, and a single number would hide that.
//!
//! Timed through [`Qwen3::cpu_logits`] — host logits, no GPU buffer per token — because the fabric is
//! what is being measured; the `Tensor`-returning entry points add one upload per call on top.
//!
//! `CPU_BENCH_PHASE=pp|tg` runs one arm only, continuously for `CPU_BENCH_SECS` (default 10) seconds,
//! printing the rate at the end — the shape an external power meter needs (`scripts/cpu_fabric_bench.py`
//! integrates `macmon` over it; llama-bench is metered the same way).
use ferric_gguf::GgufFile;
use ferric_llama::qwen3::{Cache, Qwen3};
use std::sync::Arc;
use std::time::Instant;

fn argmax(r: &[f32]) -> u32 { (0..r.len()).max_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap() as u32 }

fn med(v: &[f64]) -> (f64, f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    (s[s.len() / 2], s[0], s[s.len() - 1])
}

fn now_unix() -> f64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64() }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mp = a.get(1).expect("usage: cpu_bench <model.gguf> [pp] [tg] [reps]");
    let pp: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(512);
    let tg: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(128);
    let reps: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(5);
    let ctx = Arc::new(pollster::block_on(ferric_core::Context::new()).expect("context"));
    let g = GgufFile::open(mp).expect("open");
    let t0 = Instant::now();
    let m = Qwen3::load(&ctx, &g).expect("load");
    assert!(m.on_cpu(), "cpu_bench measures the CPU fabric: run with FERRIC_CPU=1");
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
    // Deterministic ids spread over the vocabulary (a real prompt's ids are not special to the kernels).
    let vocab = m.cfg.n_vocab as u64;
    let ids: Vec<u32> = (0..pp.max(1) as u64).map(|i| (100 + (i * 2_654_435_761) % (vocab - 200)) as u32).collect();

    let run_pp = || {
        let mut c = Cache::new(&m.cfg);
        let t = Instant::now();
        let lg = m.cpu_logits(&ids, &mut c, true);
        (pp as f64 / t.elapsed().as_secs_f64(), argmax(&lg))
    };
    let run_tg = || {
        let mut c = Cache::new(&m.cfg);
        let mut next = ids[0];
        let t = Instant::now();
        for _ in 0..tg { next = argmax(&m.cpu_logits(&[next], &mut c, true)); }
        (tg as f64 / t.elapsed().as_secs_f64(), next)
    };

    if let Ok(phase) = std::env::var("CPU_BENCH_PHASE") {
        let secs: f64 = std::env::var("CPU_BENCH_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(10.0);
        let f: &dyn Fn() -> (f64, u32) = if phase == "pp" { &run_pp } else { &run_tg };
        f(); // warm-up
        let (t, mut n, mut toks) = (Instant::now(), 0usize, 0usize);
        println!("PHASE_START {:.6}", now_unix());
        while t.elapsed().as_secs_f64() < secs { f(); n += 1; toks += if phase == "pp" { pp } else { tg }; }
        let dt = t.elapsed().as_secs_f64();
        println!("PHASE_END {:.6}", now_unix());
        println!("PHASE {phase} runs {n} tokens {toks} secs {dt:.3} tok/s {:.2}", toks as f64 / dt);
        return;
    }

    run_pp();
    run_tg();
    let (mut ppv, mut tgv) = (Vec::new(), Vec::new());
    let (mut ppa, mut tga) = (0, 0);
    for _ in 0..reps {
        let (r, x) = run_pp(); ppv.push(r); ppa = x;
        let (r, x) = run_tg(); tgv.push(r); tga = x;
    }
    let (pm, p0, p1) = med(&ppv);
    let (tm, t0, t1) = med(&tgv);
    println!("model {}  threads {}  kernels {}  f32act {}", mp.rsplit('/').next().unwrap_or(mp),
             ferric_tensor::cpu_q::pool().threads(), ferric_tensor::cpu_q::kernel_family(),
             ferric_tensor::cpu_q::f32_activations());
    println!("pp{pp}  {pm:8.1} tok/s  ({p0:.1}..{p1:.1})   last-row argmax {ppa}");
    println!("tg{tg}  {tm:8.1} tok/s  ({t0:.1}..{t1:.1})   final token {tga}");
}
