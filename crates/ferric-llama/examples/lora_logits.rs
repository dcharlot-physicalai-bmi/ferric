//! **A Dense LM's logits with a LoRA adapter applied** — the Ferric side of `scripts/lora_conformance.sh`,
//! compared against Hugging Face PEFT (`examples/refgen/lora_ref.py`).
//!
//!   lora_logits <base.gguf> <ids,...> <sample_ids,...> <mode> [adapter] [adapter_b]
//!
//! `mode`:
//!   base       no adapter
//!   unmerged   `Qwen3::upload_lora` + `Cache::set_adapters` — y = Wx + s·B(Ax) at runtime
//!   merged     the adapter merged into the weights at load (`LoraMerged`, F32), then a plain forward
//!   merged16   the same, merged weights kept F16
//!   batch      `ids` is three `;`-separated sequences of equal length L and `P` is the env `LORA_P`:
//!              each sequence prefills its first P tokens ALONE with its own selection (adapter, adapter_b,
//!              none), then all three decode tokens P..L-1 TOGETHER through `forward_batch` — one row
//!              per sequence, each row with its own adapter. Rows print as `ROW <seq>:<pos> ...`.
//!   prefix     a `PrefixCache` entry made under `adapter` must seed ONLY a cache selecting `adapter`:
//!              prints `SEED <selection> <tokens reused>` for adapter / none / adapter_b.
//!
//! Prints per position `ROW t argmax sum ssq <sampled logits>`. An adapter path is a PEFT directory
//! or a llama.cpp `.gguf` adapter. `FERRIC_LORA_NEG` (see `ferric_load::lora`) selects a wrong reading
//! for the gate's negative controls.
use ferric_gguf::GgufFile;
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_load::lora::{LoraAdapter, LoraMerged, MergeDtype};
use std::sync::Arc;

fn nums(s: &str) -> Vec<u32> { s.split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().unwrap()).collect() }

fn row(tag: &str, r: &[f32], sample: &[u32]) {
    let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
    let (mut sum, mut ssq) = (0f64, 0f64);
    for (i, &x) in r.iter().enumerate() {
        if x > bv { bv = x; best = i; }
        sum += x as f64; ssq += (x as f64) * (x as f64);
    }
    let s: Vec<String> = sample.iter().map(|&i| format!("{:.6}", r[i as usize])).collect();
    println!("ROW {tag} {best} {sum:.4} {ssq:.4} {}", s.join(" "));
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: lora_logits <base.gguf> <ids> <sample_ids> <base|unmerged|merged|merged16|batch> [adapter] [adapter_b]");
        std::process::exit(2);
    }
    let (mp, ids_s, sample, mode) = (&a[1], &a[2], nums(&a[3]), a[4].as_str());
    let adapter = |i: usize| -> LoraAdapter {
        let p = a.get(i).unwrap_or_else(|| panic!("mode {mode} needs an adapter path"));
        LoraAdapter::open(p).unwrap_or_else(|e| { eprintln!("⛔ {e}"); std::process::exit(1) })
    };
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let g = GgufFile::open(mp).expect("open base");
    let t0 = std::time::Instant::now();
    match mode {
        "base" | "unmerged" => {
            let ids = nums(ids_s);
            let m = Qwen3::load(&ctx, &g).expect("load");
            let mut cache = Cache::new(&m.cfg);
            if mode == "unmerged" {
                let d = m.upload_lora(&adapter(5)).unwrap_or_else(|e| { eprintln!("⛔ {e}"); std::process::exit(1) });
                eprintln!("adapter {} uploaded: {} params over {} layers", d.name, d.params, d.layer_count);
                cache.set_adapters(vec![(d, 1.0)]).unwrap();
            }
            let lg = m.forward_cached(&ids, &mut cache).to_vec().await;
            let v = lg.len() / ids.len();
            for t in 0..ids.len() { row(&t.to_string(), &lg[t * v..(t + 1) * v], &sample); }
        }
        "merged" | "merged16" => {
            let ids = nums(ids_s);
            let ad = adapter(5);
            let dt = if mode == "merged" { MergeDtype::F32 } else { MergeDtype::F16 };
            let src = LoraMerged::new(&g, &[(&ad, 1.0)], dt).unwrap_or_else(|e| { eprintln!("⛔ {e}"); std::process::exit(1) });
            eprintln!("merging {} weights", src.merged_names().count());
            let m = Qwen3::load(&ctx, &src).expect("load merged");
            let lg = m.forward(&ids).to_vec().await;
            let v = lg.len() / ids.len();
            for t in 0..ids.len() { row(&t.to_string(), &lg[t * v..(t + 1) * v], &sample); }
        }
        "batch" => {
            let seqs: Vec<Vec<u32>> = ids_s.split(';').map(nums).collect();
            let p: usize = std::env::var("LORA_P").expect("batch mode needs LORA_P").parse().unwrap();
            let l = seqs[0].len();
            assert!(seqs.iter().all(|s| s.len() == l) && p >= 1 && p < l, "equal-length sequences, 1 <= P < L");
            let m = Qwen3::load(&ctx, &g).expect("load");
            let (da, db) = (m.upload_lora(&adapter(5)).unwrap(), m.upload_lora(&adapter(6)).unwrap());
            let sels = [vec![(da, 1.0f32)], vec![(db, 1.0f32)], vec![]];
            let mut caches: Vec<Cache> = sels.iter().map(|s| {
                let mut c = Cache::new(&m.cfg);
                c.set_adapters(s.clone()).unwrap();
                c
            }).collect();
            // Each prompt ALONE, with its own selection — then the three decode together.
            for (i, c) in caches.iter_mut().enumerate() {
                let lg = m.forward_cached_last(&seqs[i][..p], c).to_vec().await;
                row(&format!("{i}:{}", p - 1), &lg, &sample);
            }
            let mut out: Vec<Vec<(usize, Vec<f32>)>> = vec![Vec::new(); seqs.len()];
            for pos in p..l {
                let toks: Vec<u32> = seqs.iter().map(|s| s[pos]).collect();
                let mut refs: Vec<&mut Cache> = caches.iter_mut().collect();
                let lg = m.forward_batch(&toks, &mut refs).to_vec().await;
                let v = lg.len() / toks.len();
                for i in 0..toks.len() { out[i].push((pos, lg[i * v..(i + 1) * v].to_vec())); }
            }
            for (i, rows) in out.iter().enumerate() {
                for (pos, r) in rows { row(&format!("{i}:{pos}"), r, &sample); }
            }
        }
        "prefix" => {
            let ids = nums(ids_s);
            let m = Qwen3::load(&ctx, &g).expect("load");
            let (da, db) = (m.upload_lora(&adapter(5)).unwrap(), m.upload_lora(&adapter(6)).unwrap());
            let mut pc = ferric_llama::prefix::PrefixCache::new(4);
            let mut c = Cache::new(&m.cfg);
            c.set_adapters(vec![(Arc::clone(&da), 1.0)]).unwrap();
            let _ = m.forward_cached(&ids, &mut c).to_vec().await;
            pc.insert(&ctx, &ids, &c);
            for (label, sel) in [("adapter", vec![(da, 1.0)]), ("none", vec![]), ("adapter_b", vec![(db, 1.0)])] {
                let mut c = Cache::new(&m.cfg);
                c.set_adapters(sel).unwrap();
                println!("SEED {label} {}", pc.seed(&ctx, &ids, &mut c).map(|h| h.tokens).unwrap_or(0));
            }
        }
        other => { eprintln!("unknown mode {other}"); std::process::exit(2); }
    }
    eprintln!("{mode}: {:.2}s", t0.elapsed().as_secs_f64());
}
