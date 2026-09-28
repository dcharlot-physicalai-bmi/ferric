//! **Prefill and decode throughput, phase by phase, with wall-clock marks for an energy sampler.**
//!
//!   phase_bench <model.gguf> [prompt 512] [decode 128] [reps 5]
//!
//! Each rep: a fresh cache, one prefill of `prompt` tokens (`forward_cached_last_host`, as a server
//! does), then `decode` greedy steps reading the logits on the host each step (`forward_cached_host`).
//! `PHASE_TENSOR_LOGITS=1` takes the logits the old way instead — `forward_cached(..).to_vec()`, a wgpu
//! Tensor round trip per token — so the host-logits path can be measured against it in one binary. Prints per-rep times, the median
//! tok/s of each phase (llama-bench's pp512 / tg128 shape by default), and `MARK <phase> <begin|end>
//! <unix_ns>` lines on stderr so `scripts/cuda_phase_joules.py` can integrate board power over exactly
//! those windows. Run it with and without FERRIC_CUDA to compare the native tier with WGSL on one box.
use ferric_llama::qwen3::{Cache, Qwen3};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn mark(phase: &str, edge: &str) {
    eprintln!("MARK {phase} {edge} {}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos());
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("usage: phase_bench <model.gguf> [prompt] [decode] [reps]");
    let num = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (n, d, reps) = (num(2, 512), num(3, 128), num(4, 5));
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    println!("adapter   : {} [{:?}]", ctx.adapter_name, ctx.backend);
    #[cfg(all(any(target_os = "linux", target_os = "windows"), not(target_arch = "wasm32")))]
    if let Some(dn) = ferric_tensor::cuda::device_name() { println!("native    : {dn} [CUDA tier, FERRIC_CUDA set]"); }
    let g = ferric_gguf::GgufFile::open(path).expect("open");
    let m = Qwen3::load(&ctx, &g).expect("load");
    let nv = m.cfg.n_vocab;
    // A fixed pseudo-prompt: throughput does not depend on WHICH ids, and this needs no tokenizer.
    let prompt: Vec<u32> = (0..n as u32).map(|i| 100 + (i * 7919) % 20000).collect();
    let am = |v: &[f32]| (0..nv).max_by(|&x, &y| v[v.len() - nv + x].partial_cmp(&v[v.len() - nv + y]).unwrap()).unwrap() as u32;
    let tensor_logits = std::env::var("PHASE_TENSOR_LOGITS").is_ok();
    let last = |c: &mut Cache| if tensor_logits { pollster::block_on(m.forward_cached_last(&prompt, c).to_vec()) } else { m.forward_cached_last_host(&prompt, c) };
    let step = |t: u32, c: &mut Cache| if tensor_logits { pollster::block_on(m.forward_cached(&[t], c).to_vec()) } else { m.forward_cached_host(&[t], c) };
    println!("logits    : {}", if tensor_logits { "forward_cached(..).to_vec() — a wgpu Tensor round trip (PHASE_TENSOR_LOGITS)" } else { "host (forward_cached_host)" });
    // Warm-up: compile every pipeline / JIT every kernel both phases use.
    { let mut c = Cache::new(&m.cfg); let v = last(&mut c); let _ = step(am(&v), &mut c); }
    let (mut tp, mut td, mut ids) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..reps {
        let mut c = Cache::new(&m.cfg);
        mark("prefill", "begin");
        let t0 = Instant::now();
        let v = last(&mut c);
        tp.push(t0.elapsed().as_secs_f64());
        mark("prefill", "end");
        let mut tok = am(&v);
        let mut generated = vec![tok];
        mark("decode", "begin");
        let t1 = Instant::now();
        for _ in 0..d {
            let v = step(tok, &mut c);
            tok = am(&v); generated.push(tok);
        }
        td.push(t1.elapsed().as_secs_f64());
        mark("decode", "end");
        ids = generated;
    }
    println!("{path}: {} layers", m.cfg.n_layer);
    println!("  prefill {n} tok  per rep (s): {:?}", tp.iter().map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>());
    println!("  decode  {d} tok  per rep (s): {:?}", td.iter().map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>());
    println!("  PREFILL_TOKS_PER_S {:.1}  (median of {reps})", n as f64 / median(tp));
    println!("  DECODE_TOKS_PER_S {:.1}  (median of {reps})", d as f64 / median(td));
    println!("  last rep's greedy ids (first 24): {:?}", &ids[..ids.len().min(24)]);
    #[cfg(all(any(target_os = "linux", target_os = "windows"), not(target_arch = "wasm32")))]
    println!("  native steps {}  native prefill rows {}  graph replays {}", ferric_tensor::cuda::native_steps(),
             ferric_tensor::cuda::native_prefill_rows(), ferric_tensor::cuda::native_graph_steps());
}
