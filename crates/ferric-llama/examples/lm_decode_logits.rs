//! **A dense LM's logits one DECODE STEP at a time, on ids the model's AUTHORS produced** — the
//! Ferric side of `scripts/cuda_conformance.sh`. Same `ROW` format as `lm_logits`, but where that runs
//! ONE stateless prefill, this feeds the sequence through `forward_cached` the way generation does:
//!
//!   1. a prefill of `prefill` tokens,
//!   2. then one token per call, teacher-forced with the fixture's ids,
//!   3. except one multi-token CHUNK of `chunk` tokens at position `chunk_at`.
//!
//! Single-token calls are what the NVIDIA tier serves; multi-token calls run on WGSL. So every row
//! after the prefill is a native decode step when `FERRIC_CUDA` is set, and the chunk forces the two
//! paths to hand the K/V cache over in BOTH directions (device → WGSL before the chunk, WGSL → device
//! after it) — the coherence the tier used to get wrong by panicking.
//!
//! `total` > the fixture's length tiles its ids to that many positions: rows past the fixture have no
//! authors' reference, but they still compare native against WGSL — which is how positions past the
//! old 2048-token cap get checked.
//!
//!   cargo run -p ferric-llama --release --example lm_decode_logits -- \
//!       <model.gguf> <fixture.json> [prefill=8] [chunk_at=64] [chunk=5] [total=0]
//!
//! Prints `NATIVE_STEPS <n> OF <m>` last: the native steps that ran against the single-token calls
//! made. A gate must check it — a WGSL fallback produces the same kind of rows.
use ferric_gguf::GgufFile;
use ferric_llama::qwen3::{Cache, Qwen3};
use std::sync::Arc;

fn field<'a>(s: &'a str, key: &str) -> &'a str {
    let k = format!("\"{key}\": [");
    let i = s.find(&k).unwrap_or_else(|| panic!("fixture has no \"{key}\"")) + k.len();
    let j = s[i..].find(']').unwrap() + i;
    &s[i..j]
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let mp = a.get(1).expect("usage: lm_decode_logits <model.gguf> <fixture.json> [prefill] [chunk_at] [chunk] [total]");
    let fx = std::fs::read_to_string(a.get(2).expect("fixture.json")).expect("read fixture");
    let num = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (prefill, chunk_at, chunk, total) = (num(3, 8), num(4, 64), num(5, 5), num(6, 0));
    let parse = |key: &str| -> Vec<u32> {
        field(&fx, key).split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().unwrap()).collect()
    };
    let base = parse("ids");
    let sample = parse("sample_ids");
    let n = if total > 0 { total } else { base.len() };
    let ids: Vec<u32> = (0..n).map(|i| base[i % base.len()]).collect();
    assert!(prefill >= 1 && prefill < n, "prefill must leave decode steps");

    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    // ⛔ A number that does not name its device is a number on an UNKNOWN device (the 4050 box once
    // served a day of "RTX 4050" figures from the Intel iGPU).
    eprintln!("adapter   : {} [{:?}]", ctx.adapter_name, ctx.backend);
    #[cfg(all(any(target_os = "linux", target_os = "windows"), not(target_arch = "wasm32")))]
    if let Some(d) = ferric_tensor::cuda::device_name() { eprintln!("native    : {d} [CUDA tier, FERRIC_CUDA set]"); }
    let g = GgufFile::open(mp).expect("open");
    let m = Qwen3::load(&ctx, &g).expect("load");
    let v = m.cfg.n_vocab;
    let mut cache = Cache::new(&m.cfg);

    let emit = |t: usize, r: &[f32]| {
        let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
        let (mut sum, mut ssq) = (0f64, 0f64);
        for (i, &x) in r.iter().enumerate() {
            if x > bv { bv = x; best = i; }
            sum += x as f64; ssq += (x as f64) * (x as f64);
        }
        let s: Vec<String> = sample.iter().map(|&i| format!("{:.5}", r[i as usize])).collect();
        println!("ROW {t} {best} {sum:.4} {ssq:.4} {}", s.join(" "));
    };
    #[cfg(all(any(target_os = "linux", target_os = "windows"), not(target_arch = "wasm32")))]
    let steps0 = ferric_tensor::cuda::native_steps();
    let (mut t, mut singles) = (0usize, 0usize);
    let t0 = std::time::Instant::now();
    while t < n {
        let len = if t == 0 { prefill } else if t == chunk_at && chunk > 1 { chunk.min(n - t) } else { 1 };
        let lg = m.forward_cached(&ids[t..t + len], &mut cache).to_vec().await;
        assert_eq!(lg.len(), len * v, "logits for {len} rows");
        for (j, r) in lg.chunks_exact(v).enumerate() { emit(t + j, r); }
        if len == 1 { singles += 1; }
        t += len;
    }
    let dt = t0.elapsed();
    #[cfg(all(any(target_os = "linux", target_os = "windows"), not(target_arch = "wasm32")))]
    let native = ferric_tensor::cuda::native_steps() - steps0;
    #[cfg(not(all(any(target_os = "linux", target_os = "windows"), not(target_arch = "wasm32"))))]
    let native = 0u64;
    eprintln!("{n} positions in {:.1} s ({singles} single-token calls)", dt.as_secs_f64());
    println!("NATIVE_STEPS {native} OF {singles}");
}
