//! **Decode from explicit token ids** — the tokenizer taken out of the picture.
//!
//! When a newly ported model emits token soup there are two suspects, and they are usually confused:
//! the tokenizer produced the wrong ids, or the forward pass mishandled the right ones. Feeding ids
//! straight in separates them in one run. Pair it with a reference tokenizer:
//!
//! ```text
//! python -c "from tokenizers import Tokenizer; print(Tokenizer.from_file('tokenizer.json').encode('...').ids)"
//! cargo run -p ferric-llama --example run_ids --release -- model.gguf 200000,954,7963 16
//! ```
//!
//! Also prints the top-k of the FIRST step. A model whose top token is plausible but whose
//! continuation degenerates is a different failure from one whose very first distribution is noise,
//! and the top-k is what tells them apart before any text is read.
//!
//!   cargo run -p ferric-llama --example run_ids --release -- <model.gguf> <id,id,id> [n]
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use std::sync::Arc;

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("usage: run_ids <model.gguf> <id,id,...> [n]");
    let ids: Vec<u32> = a.get(2).expect("need ids").split(',')
        .map(|s| s.trim().parse().expect("id")).collect();
    let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);

    let g = GgufFile::open(path).expect("open");
    // Vocab strings straight from the GGUF, so what is printed is what the checkpoint calls the token.
    let vocab: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => Vec::new(),
    };
    let name = |t: u32| vocab.get(t as usize).cloned().unwrap_or_else(|| format!("<{t}>"));

    let ctx = Arc::new(Context::new().await.unwrap());
    let m = Qwen3::load(&ctx, &g).expect("load");
    let vn = m.cfg.n_vocab;
    println!("{} layers · d={} · {}h/{}kv × {} · vocab={}\n",
             m.cfg.n_layer, m.cfg.n_embd, m.cfg.n_head, m.cfg.n_head_kv, m.cfg.head_dim, vn);
    println!("prompt ids: {ids:?}");
    println!("prompt    : {}", ids.iter().map(|&t| name(t)).collect::<Vec<_>>().join("|"));

    let mut c = Cache::new(&m.cfg);
    let logits = m.forward_cached(&ids, &mut c).to_vec().await;
    let last = &logits[logits.len() - vn..];

    // Finiteness first: a NaN anywhere means a kernel produced garbage, and every "the model is
    // wrong" theory downstream of that is wasted effort.
    let bad = last.iter().filter(|v| !v.is_finite()).count();
    println!("\nfirst-step logits: {} non-finite of {vn}", bad);
    if bad == 0 {
        let (mut mn, mut mx, mut sum) = (f32::MAX, f32::MIN, 0f64);
        for &v in last { mn = mn.min(v); mx = mx.max(v); sum += v as f64; }
        println!("  min {mn:.3}  max {mx:.3}  mean {:.3}", sum / vn as f64);
    }
    let mut idx: Vec<usize> = (0..vn).collect();
    idx.sort_by(|&i, &j| last[j].partial_cmp(&last[i]).unwrap());
    println!("  top-8:");
    for &i in idx.iter().take(8) {
        println!("    {:>7}  {:>9.4}  {:?}", i, last[i], name(i as u32));
    }

    // Greedy continuation.
    let am = |l: &[f32]| l[l.len() - vn..].iter().enumerate()
        .fold((0usize, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32;
    let mut next = am(&logits);
    let mut out = Vec::new();
    // Reset AFTER prefill so the readout below describes steady-state decode, not one-off warmup.
    ferric_tensor::reset_host_ns();
    ferric_tensor::reset_op_counters();
    let t_dec = std::time::Instant::now();
    for _ in 0..n {
        out.push(next);
        let l = m.forward_cached(&[next], &mut c).to_vec().await;
        next = am(&l);
    }
    let dt = t_dec.elapsed();
    // Per-category GPU breakdown when FERRIC_PROFILE=1. Note the profile itself costs ~27%
    // (25.1 -> 31.9 ms/tok) because `prof` device-syncs at every boundary to attribute time — so read
    // the RATIOS, never the absolute ms, from a profiled run.
    ferric_tensor::prof_report();
    println!("\ngenerated ids: {out:?}");

    // MNN (arXiv 2002.12418, Table 2) reports that decoupling command-buffer PREPARATION from
    // execution is worth 7-8% on CPU and 49.5-75.2% on GPU, because "setting up command buffer and
    // its related command descriptions is time-consuming". WebGPU pays the same tax. Print the split
    // so the speed gap is answered with a number rather than an assumption.
    {
        let (pipe_ns, bg_ns, rec_ns, sub_ns) = ferric_tensor::host_ns();
        let (disp, subs) = ferric_tensor::op_counters();
        let host = pipe_ns + bg_ns + rec_ns + sub_ns;
        let wall = dt.as_nanos() as u64;
        println!("\ndecode {} tok in {:.1} ms ({:.1} ms/tok)",
                 out.len(), wall as f64 / 1e6, wall as f64 / 1e6 / out.len().max(1) as f64);
        println!("host-side setup: {:.1}% of wall ({:.1} ms of {:.1} ms)",
                 100.0 * host as f64 / wall.max(1) as f64, host as f64 / 1e6, wall as f64 / 1e6);
        println!("  pipeline lookup {:>8.2} ms   bind groups {:>8.2} ms", pipe_ns as f64 / 1e6, bg_ns as f64 / 1e6);
        // slot 3 is NOT just submit — `unibuf`/`u32buf` charge their allocation to it too, so the
        // old "submit" label conflated queue submission with per-dispatch info-buffer creation and
        // hid the single largest host cost behind a name that suggested nothing could be done.
        println!("  pass recording  {:>8.2} ms   submit+infobuf {:>5.2} ms", rec_ns as f64 / 1e6, sub_ns as f64 / 1e6);
        println!("  {disp} dispatches in {subs} submits ({:.0} dispatches/token)",
                 disp as f64 / out.len().max(1) as f64);
    }
    println!("generated    : {}", out.iter().map(|&t| name(t)).collect::<Vec<_>>().join("|"));
}
