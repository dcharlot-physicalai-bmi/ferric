//! **Do streamed experts give the same answer as resident ones?**
//!
//! `FERRIC_MOE_SLOTS=C` holds only `C` of a MoE's routed experts on the device and fetches the rest
//! on demand. The property that makes that safe is the one `ferric-tier` enforces for layers and
//! nothing yet enforced for experts: **how many slots you have changes speed and memory, never
//! results.** A streamed forward must produce bit-identical logits to a resident one.
//!
//! This prints one line the caller can diff. It deliberately does NOT time anything — timing on a
//! shared dev box measures the box, and the question here is correctness, which does not care what
//! else is running.
//!
//! ⚠ `FERRIC_MOE_SLOTS` is read at LOAD time, so the sweep is done by running this once per value
//! rather than by mutating the environment mid-process. Same binary, same prompt, different slot
//! count — anything that differs is the streaming path.
//!
//!   for c in 0 8 16 64; do FERRIC_MOE_SLOTS=$c cargo run -q -p ferric-llama \
//!       --example moe_slots --release -- <model.gguf>; done
//!
//! `FERRIC_MOE_SLOTS=0` (or unset) is the resident baseline every other row must match.
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen35::{Ffn, MoeExperts, Qwen35};
use std::sync::Arc;

fn main() { pollster::block_on(run()); }

async fn run() {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| format!("{}/.cache/ferric/qwen3.6-35b-a3b-q4km.gguf",
                                   std::env::var("HOME").unwrap()));
    let g = GgufFile::open(&path).expect("open gguf");
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));

    let toks: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|x| if let Meta::Str(s) = x { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokenizer"),
    };
    let vocab: std::collections::HashMap<String, u32> =
        toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
        Some(Meta::Arr(v)) => v.iter().filter_map(|x| if let Meta::Str(s) = x {
            s.split_once(' ').map(|(p, q)| (p.to_string(), q.to_string())) } else { None }).collect(),
        _ => Vec::new(),
    };
    let bpe = ferric_tokenizer::Bpe::new(vocab, &merges);

    let m = Qwen35::load(&ctx, &g).expect("load");
    let ids: Vec<u32> = bpe.encode("The capital of France is Paris, and the river that runs through it");

    let mut cache = ferric_llama::qwen35::Cache::new(&m.cfg);
    let out = m.forward_cached(&ids, &mut cache, m.layers.len()).to_vec().await;
    let vn = m.cfg.n_vocab;
    let last = &out[out.len() - vn..];

    // A checksum over the RAW BITS, not a rounded sum: two logit vectors that differ in the last
    // mantissa bit must produce different checksums, or the invariance claim is only about the
    // first few digits.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in last { h ^= v.to_bits() as u64; h = h.wrapping_mul(0x1000_0000_01b3); }
    let argmax = last.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;

    // What the model is actually running, read off the loaded layers rather than off the env var —
    // a config that failed to take effect would otherwise report itself as active.
    let (mut streamed, mut resident, mut slots) = (0usize, 0usize, 0usize);
    let (mut fetched, mut hits) = (0u64, 0u64);
    for l in &m.layers {
        if let Ffn::Moe(mo) = &l.ffn {
            match &mo.experts {
                MoeExperts::Streamed(st) => {
                    streamed += 1;
                    slots = st.capacity;
                    fetched += st.fetched.load(std::sync::atomic::Ordering::Relaxed);
                    hits += st.hits.load(std::sync::atomic::Ordering::Relaxed);
                }
                _ => resident += 1,
            }
        }
    }

    // ⚠ BYTES, not seconds. Device residency is exact and load-independent — unlike a throughput
    // number, which on a shared box measures the box — and it is the quantity streaming exists to
    // reduce, so it is the number worth printing.
    //
    // A streamed run cannot MEASURE the all-resident figure (it never builds that slab), so it
    // derives one. `all_resident=` is therefore printed by BOTH paths — measured when resident,
    // derived when streamed — so the sweep's own rows check the derivation instead of taking it on
    // faith. Same model, same layers: the two numbers must agree exactly.
    let (mut on_device, mut all_resident) = (0u64, 0u64);
    for l in &m.layers {
        if let Ffn::Moe(mo) = &l.ffn {
            on_device += mo.experts.device_bytes();
            all_resident += match &mo.experts {
                MoeExperts::Streamed(st) => st.all_resident_bytes(m.cfg.n_expert),
                other => other.device_bytes(),
            };
        }
    }
    println!("slots={slots:<5} streamed_layers={streamed:<4} resident_layers={resident:<4} \
              fetched={fetched:<7} slot_hits={hits:<7} argmax={argmax:<7} logits_fnv={h:016x}");
    // Exact bytes, not just GB: at 2 decimals two figures can differ by 10 MB and still print the
    // same, so a rounded agreement between the derived and measured rows would prove nothing.
    let how = if streamed > 0 { "derived" } else { "measured" };
    println!("  routed experts on device {:.2} GB ({on_device} B)   all_resident {:.2} GB ({all_resident} B, {how})   {:.1}x",
             on_device as f64 / 1e9, all_resident as f64 / 1e9,
             all_resident as f64 / on_device.max(1) as f64);

    if streamed > 0 {
        assert!(fetched > 0, "every MoE layer is streamed but nothing was ever fetched — the slabs \
                              are serving whatever they were seeded with");
    }
}
