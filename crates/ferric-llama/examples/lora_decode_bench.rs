//! **What an unmerged LoRA costs at decode** — tok/s, GPU dispatches per token and SoC joules per token for
//! the base alone, the base with the adapter applied UNMERGED (`upload_lora` + `Cache::set_adapters`), and
//! the adapter MERGED into the weights at load (`LoraMerged`, F32 and F16). Arms are INTERLEAVED per
//! repetition, so a shared machine's load drifts across all of them rather than into one.
//!
//!   lora_decode_bench <base.gguf> <adapter> [adapter_b]
//!
//! With `adapter_b`, a batched arm too: 4 sequences through `forward_batch`, rows alternating adapter /
//! adapter_b (per-row selection) against the same 4 rows on the base alone.
//!
//! Env: `BENCH_TOKENS` decode steps (default 64), `BENCH_REPS` (default 3).
//!
//! ⚠ Dispatches are counted exactly on the host and do not depend on load. Wall time and joules DO: the
//! SoC counters `macmon` reads are system-wide, so every figure includes whatever else the machine was
//! doing; the per-arm RATIO is the claim, the absolute joules are an upper bound.
use ferric_core::Context;
use ferric_gguf::GgufFile;
use ferric_joule::{Macmon, MacmonScope};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_load::lora::{LoraAdapter, LoraMerged, MergeDtype};
use ferric_tensor::{op_counters, reset_op_counters};
use std::sync::Arc;
use std::time::Instant;

fn argmax(r: &[f32]) -> u32 { (0..r.len()).max_by(|&a, &b| r[a].partial_cmp(&r[b]).unwrap()).unwrap() as u32 }

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let (mp, ap) = (a.get(1).expect("base.gguf"), a.get(2).expect("adapter"));
    let steps: usize = std::env::var("BENCH_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let reps: usize = std::env::var("BENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let ctx = Arc::new(Context::new().await.unwrap());
    let g = GgufFile::open(mp).unwrap();
    let ad = LoraAdapter::open(ap).unwrap();
    let base = Qwen3::load(&ctx, &g).unwrap();
    let dev = base.upload_lora(&ad).unwrap();
    let merged32 = Qwen3::load(&ctx, &LoraMerged::new(&g, &[(&ad, 1.0)], MergeDtype::F32).unwrap()).unwrap();
    let merged16 = Qwen3::load(&ctx, &LoraMerged::new(&g, &[(&ad, 1.0)], MergeDtype::F16).unwrap()).unwrap();
    let dev_b = a.get(3).map(|p| base.upload_lora(&LoraAdapter::open(p).unwrap()).unwrap());
    eprintln!("base {mp}\nadapter {} ({} params, {} layers)", dev.name, dev.params, dev.layer_count);
    let prompt: Vec<u32> = (0..16).map(|i| 1000 + 37 * i).collect();
    let meter = Macmon::start(MacmonScope::Soc, 100);
    if meter.is_none() { eprintln!("⚠ macmon not available: no joules, tok/s and dispatches only"); }
    // `energy_over` takes seconds on the METER's clock (since its first sample). Anchor ours to it: the
    // sample that just arrived is at most one 100 ms interval old, which bounds the edge error.
    let clock = Instant::now();
    let offset = meter.as_ref().and_then(|m| m.trace().last().map(|s| s.t)).unwrap_or(0.0);
    std::thread::sleep(std::time::Duration::from_millis(1500)); // samples on both sides of every window
    let now = |m: &Option<Macmon>| m.as_ref().map(|_| offset + clock.elapsed().as_secs_f64());

    // One arm: prefill (not timed), then `steps` greedy decode steps (timed, counted, metered).
    // Returns (tok/s, dispatches per token, joules per token).
    let single = |m: &Qwen3, sel: Option<Arc<ferric_llama::lora::DeviceLora>>| {
        let mut c = Cache::new(&m.cfg);
        if let Some(d) = sel { c.set_adapters(vec![(d, 1.0)]).unwrap(); }
        let lg = pollster::block_on(m.forward_cached_last(&prompt, &mut c).to_vec());
        let mut next = argmax(&lg);
        reset_op_counters();
        let (t0, w0) = (Instant::now(), now(&meter));
        for _ in 0..steps {
            let lg = pollster::block_on(m.forward_cached(&[next], &mut c).to_vec());
            next = argmax(&lg);
        }
        let dt = t0.elapsed().as_secs_f64();
        let w1 = now(&meter);
        let (disp, _) = op_counters();
        (steps as f64 / dt, disp as f64 / steps as f64, (w0, w1))
    };
    let batch = |sels: Vec<Vec<(Arc<ferric_llama::lora::DeviceLora>, f32)>>| {
        let mut cs: Vec<Cache> = sels.into_iter().map(|s| { let mut c = Cache::new(&base.cfg); c.set_adapters(s).unwrap(); c }).collect();
        let mut next: Vec<u32> = cs.iter_mut().map(|c| argmax(&pollster::block_on(base.forward_cached_last(&prompt, c).to_vec()))).collect();
        reset_op_counters();
        let (t0, w0) = (Instant::now(), now(&meter));
        for _ in 0..steps {
            let mut refs: Vec<&mut Cache> = cs.iter_mut().collect();
            let lg = pollster::block_on(base.forward_batch(&next, &mut refs).to_vec());
            let v = lg.len() / next.len();
            next = (0..next.len()).map(|i| argmax(&lg[i * v..(i + 1) * v])).collect();
        }
        let dt = t0.elapsed().as_secs_f64();
        let (disp, _) = op_counters();
        let n = next.len() as f64;
        (steps as f64 * n / dt, disp as f64 / (steps as f64 * n), (w0, now(&meter)))
    };

    let mut arms: Vec<(&str, Vec<(f64, f64, (Option<f64>, Option<f64>))>)> = vec![
        ("base", vec![]), ("unmerged", vec![]), ("merged F32", vec![]), ("merged F16", vec![])];
    if dev_b.is_some() { arms.push(("batch4 base", vec![])); arms.push(("batch4 a|b|a|b", vec![])); }
    for rep in 0..reps {
        eprintln!("rep {}/{reps}", rep + 1);
        arms[0].1.push(single(&base, None));
        arms[1].1.push(single(&base, Some(Arc::clone(&dev))));
        arms[2].1.push(single(&merged32, None));
        arms[3].1.push(single(&merged16, None));
        if let Some(b) = &dev_b {
            arms[4].1.push(batch(vec![vec![]; 4]));
            let (x, y) = (vec![(Arc::clone(&dev), 1.0)], vec![(Arc::clone(b), 1.0)]);
            arms[5].1.push(batch(vec![x.clone(), y.clone(), x, y]));
        }
    }
    std::thread::sleep(std::time::Duration::from_millis(1500));
    println!("{steps} decode steps x {reps} interleaved reps; J/token = SoC (cpu+gpu+ane+ram) over the decode window / tokens");
    println!("{:16} {:>26} {:>12} {:>28}", "arm", "tok/s median (min..max)", "dispatch/tok", "J/token median (min..max)");
    for (name, runs) in &arms {
        let tps: Vec<f64> = runs.iter().map(|r| r.0).collect();
        let ntok = if name.starts_with("batch4") { 4.0 } else { 1.0 } * steps as f64;
        let j: Vec<f64> = runs.iter().filter_map(|r| match (&meter, r.2) {
            (Some(m), (Some(a), Some(b))) => m.energy_over(a, b).map(|e| e / ntok), _ => None }).collect();
        let mm = |v: &[f64]| {
            let mut s = v.to_vec(); s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (s[s.len() / 2], s[0], s[s.len() - 1])
        };
        let (tm, t0, t1) = mm(&tps);
        let js = if j.is_empty() { "-".to_string() } else { let (m, a, b) = mm(&j); format!("{m:.3} ({a:.3}..{b:.3})") };
        println!("{name:16} {:>26} {:>12.0} {:>28}", format!("{tm:.1} ({t0:.1}..{t1:.1})"), runs[0].1, js);
    }
}
