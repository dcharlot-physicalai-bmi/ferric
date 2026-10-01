//! **PrismML Bonsai 2 on Ferric, dumped the way the authors' fork dumps it** — the Ferric side of
//! `scripts/bonsai2_conformance.sh`.
//!
//! Takes token ids (never a prompt string, so a tokenizer difference cannot pass for a model one),
//! runs one prefill that keeps every position's logits, then continues GREEDILY for `--gen N` tokens
//! through the cache, appending each step's logits — the exact row layout of the reference program
//! (`dumplogits`, built against PrismML's C API): `T` prompt rows, then `N − 1` generated-step rows,
//! float32, `n_vocab` wide.
//!
//!   cargo run --release -p ferric-llama --example bonsai2_logits -- <model.gguf> <id,id,...> \
//!       [--out logits.bin] [--gen N] [--time]
//!
//! Prints `gen=<ids>` and, with `--time`, per-token decode wall time (the prefill excluded).
//! `FERRIC_STREAM_GIB=<GiB>` streams the blocks that do not fit that device budget (`Qwen35::load_streaming`).
use ferric_core::Context;
use ferric_gguf::GgufFile;
use ferric_llama::qwen35::{Cache, Qwen35};
use std::io::Write;
use std::sync::Arc;

fn argmax(v: &[f32]) -> u32 {
    let mut b = 0;
    for (i, x) in v.iter().enumerate() { if *x > v[b] { b = i; } }
    b as u32
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("usage: bonsai2_logits <model.gguf> <id,id,...> [--out f.bin] [--gen N] [--time]");
    let ids: Vec<u32> = a.get(2).expect("ids").split(',').map(|x| x.trim().parse().expect("token id")).collect();
    let flag = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let n_gen: usize = flag("--gen").map(|s| s.parse().unwrap()).unwrap_or(0);
    let time = a.iter().any(|x| x == "--time");
    let mut out = flag("--out").map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("create --out")));

    let ctx = Arc::new(Context::new().await.expect("GPU context"));
    // Say which device ran it: on a hybrid-GPU box wgpu has silently picked the iGPU before.
    eprintln!("adapter: {} ({:?})", ctx.adapter_name, ctx.backend);
    let t0 = std::time::Instant::now();
    // FERRIC_STREAM_GIB=<device GiB for layer weights>: keep that many leading blocks resident and
    // stream the rest from the file (`Qwen35::load_streaming`) — how the model runs on a GPU smaller
    // than it. Unset: everything resident.
    let m = match std::env::var("FERRIC_STREAM_GIB").ok().and_then(|v| v.parse::<f64>().ok()) {
        Some(gib) => Qwen35::load_streaming(&ctx, path, (gib * (1u64 << 30) as f64) as u64),
        None => Qwen35::load(&ctx, &GgufFile::open(path).expect("open gguf")),
    }.unwrap_or_else(|e| panic!("load: {e}"));
    eprintln!("loaded in {:.2?} (prism.hadamard: {})", t0.elapsed(), if m.rot.is_some() { "applied" } else { "absent" });
    if let Some(s) = &m.stream {
        eprintln!("streaming: {}/{} blocks resident ({:.2} GiB device), {:.2} GiB re-read per pass",
                  s.npin, m.cfg.n_layer, s.pinned_device_bytes as f64 / (1u64 << 30) as f64,
                  s.streamed_bytes_per_pass as f64 / (1u64 << 30) as f64);
    }
    let nv = m.cfg.n_vocab;

    let mut cache = Cache::new(&m.cfg);
    let lg = m.forward_cached(&ids, &mut cache, m.cfg.n_layer).to_vec().await;
    assert_eq!(lg.len(), ids.len() * nv, "prefill logits shape");
    assert!(lg.iter().any(|x| *x != 0.0), "all-zero logits: not a model output");
    if let Some(w) = out.as_mut() { for x in &lg { w.write_all(&x.to_le_bytes()).unwrap(); } }
    let mut next = argmax(&lg[(ids.len() - 1) * nv..]);
    let t_prefill = t0.elapsed();
    // FERRIC_PROFILE: drop the prefill's categories so the report below covers the decode steps only.
    if time { eprintln!("prefill {} tokens (+load) in {t_prefill:.2?}", ids.len()); ferric_tensor::prof_report(); }
    let mut gen_ids = Vec::new();
    let mut step_ms = Vec::new();
    for s in 0..n_gen {
        gen_ids.push(next);
        if s + 1 == n_gen { break; }
        let t = std::time::Instant::now();
        let row = m.forward_cached(&[next], &mut cache, m.cfg.n_layer).to_vec().await;
        step_ms.push(t.elapsed().as_secs_f64() * 1e3);
        if let Some(w) = out.as_mut() { for x in &row { w.write_all(&x.to_le_bytes()).unwrap(); } }
        next = argmax(&row);
    }
    if let Some(w) = out.as_mut() { w.flush().unwrap(); }
    println!("n_vocab={nv} n={}", ids.len());
    if n_gen > 0 { println!("gen={}", gen_ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")); }
    if let Some(s) = &m.stream { println!("stream: npin={} rebuilds={}", s.npin, s.rebuilds.get()); }
    if time && !step_ms.is_empty() {
        let mut s = step_ms.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("decode ms/token: median {:.1}  min {:.1}  max {:.1}  (n={})", s[s.len() / 2], s[0], s[s.len() - 1], s.len());
        ferric_tensor::prof_report();
    }
}
