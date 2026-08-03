//! Stream a **real GGUF checkpoint** through the tier, and prove placement-invariance on real weights.
//!
//! The unit tests establish the invariant against a synthetic backing store, which is the right place to
//! test policy. This is the other half: an actual model file, an actual tensor table, actual layer runs
//! whose sizes vary, and a memory ladder that walks from "one layer barely fits" to "everything is
//! resident" — asserting that every byte delivered is identical at every rung.
//!
//!   cargo run -p ferric-tier --example stream_gguf --release -- [path/to/model.gguf]
//!
//! Defaults to the Qwen2.5-0.5B Q8_0 in Ferric's model cache.

use ferric_gguf::{type_size, GgufFile};
use ferric_tier::{
    plan_layers, Backing, FileBacking, LayerCache, LayerDesc, PrefetchCache, Tier, TierError,
};
use std::sync::Arc;

/// FNV-1a over every byte a walk delivered. If any rung disagrees, placement changed results.
fn fnv1a(seed: u64, data: &[u8]) -> u64 {
    let mut h = seed;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}
const FNV_INIT: u64 = 0xcbf2_9ce4_8422_2325;

/// Group tensors into per-layer runs and **verify each run is contiguous**.
///
/// Contiguity is what makes a layer bind exactly one read instead of one read per tensor. It is a
/// property of how the converter emitted the file, not a guarantee — so it is checked rather than
/// assumed. (kimi-k3-in-c's packer does the same and refuses to build a trunk file when a layer's
/// tensors are interleaved with another's.) A non-contiguous run here is reported, not papered over: the
/// gap would otherwise be silently included in the read and inflate every measurement.
fn layer_runs(g: &GgufFile) -> Result<Vec<LayerDesc>, String> {
    let base = g.data_start();
    let mut per_layer: Vec<Vec<(u64, u64)>> = Vec::new(); // (abs_offset, bytes)

    for t in &g.tensors {
        let Some(rest) = t.name.strip_prefix("blk.") else { continue };
        let Some((idx, _)) = rest.split_once('.') else { continue };
        let Ok(li) = idx.parse::<usize>() else { continue };
        let n: usize = t.dims.iter().product::<u64>() as usize;
        let sz = type_size(t.ggml_type, n)? as u64;
        if per_layer.len() <= li { per_layer.resize(li + 1, Vec::new()); }
        per_layer[li].push((base + t.offset, sz));
    }
    if per_layer.is_empty() { return Err("no blk.N.* tensors found — is this a transformer GGUF?".into()); }

    let mut out = Vec::with_capacity(per_layer.len());
    for (li, mut v) in per_layer.into_iter().enumerate() {
        if v.is_empty() { return Err(format!("layer {li} has no tensors")); }
        v.sort_unstable();
        let lo = v[0].0;
        let hi = v.iter().map(|(o, s)| o + s).max().unwrap();
        let own: u64 = v.iter().map(|(_, s)| *s).sum();
        if hi - lo != own {
            return Err(format!(
                "layer {li} is not one contiguous run: spans {} bytes but owns {own}. \
                 Streaming it as a single read would also pull in {} bytes belonging to other tensors.",
                hi - lo,
                hi - lo - own
            ));
        }
        out.push(LayerDesc { offset: lo, bytes: own });
    }
    Ok(out)
}

fn gb(b: u64) -> f64 { b as f64 / 1e6 }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf")
    });
    println!("checkpoint: {path}");

    let g = GgufFile::open(&path)?;
    let layers = layer_runs(&g)?;
    let total: u64 = layers.iter().map(|l| l.bytes).sum();
    let biggest = layers.iter().map(|l| l.bytes).max().unwrap();
    let smallest = layers.iter().map(|l| l.bytes).min().unwrap();
    println!(
        "  {} layers, {:.1} MB of layer weights (largest {:.1} MB, smallest {:.1} MB) — all runs contiguous ✓\n",
        layers.len(), gb(total), gb(biggest), gb(smallest)
    );

    // The backing store is opened independently of the GgufFile handle: that handle keeps its File in a
    // RefCell and is not shareable across threads, whereas positional reads are.
    let backing: Arc<dyn Backing + Send + Sync> = Arc::new(FileBacking::open(&path)?);

    // A ladder from "barely enough for one layer" to "the whole model resident".
    let ladder: Vec<u64> = vec![
        biggest + 4096,
        total / 16,
        total / 8,
        total / 4,
        total / 2,
        (total * 3) / 4,
        // Full residency needs room for the ring slot too, not just the layers: the planner charges the
        // ring while ANY layer still streams, so `total + biggest` lands one layer short (23/24) and only
        // drops the ring once the last layer is pinned.
        total + 2 * biggest,
    ];

    const TOKENS: u32 = 3;
    println!("  Read volume is the metric that matters here. Wall clock barely moves because this file fits");
    println!("  the OS page cache and the per-rung time is dominated by checksumming every delivered byte —");
    println!("  reporting it as a speedup would be measuring the harness, not the tier.\n");
    println!("  {:>10}  {:>6}  {:>9}  {:>8}  {:>7}   {}", "budget", "pinned", "hit rate", "read", "wall", "checksum");
    println!("  {:-<72}", "");

    let mut reference: Option<u64> = None;
    let mut saw_streaming = false;
    let mut saw_resident = false;

    for &budget in &ladder {
        let plan = plan_layers(&layers, budget, 0, 4096);
        let mut cache = LayerCache::new(plan.clone(), layers.clone());
        cache.prefill(&*backing)?; // pin at startup, as a deployment would

        let t0 = std::time::Instant::now();
        let mut sum = FNV_INIT;
        let mut streamed = false;
        for _tok in 0..TOKENS {
            for l in 0..layers.len() as u32 {
                let (bytes, tier) = cache.bind(l, &*backing)?;
                sum = fnv1a(sum, bytes);
                if tier == Tier::Backing { streamed = true; }
            }
        }
        let wall = t0.elapsed();
        let st = cache.stats();
        if streamed { saw_streaming = true; } else { saw_resident = true; }

        match reference {
            None => reference = Some(sum),
            Some(r) => assert_eq!(
                sum, r,
                "PLACEMENT CHANGED RESULTS at budget {:.1} MB — the memory budget must decide where \
                 bytes come from, never what they are",
                gb(budget)
            ),
        }
        println!(
            "  {:>7.1} MB  {:>3}/{:<3} {:>8.1}%  {:>6.1} MB  {:>5.0} ms   {sum:016x}",
            gb(budget), plan.npin, layers.len(), 100.0 * st.hit_rate(), gb(st.bytes_read),
            wall.as_secs_f64() * 1000.0
        );
    }

    // Anti-vacuity: "all identical" is trivially true if nothing streamed or nothing was resident.
    assert!(saw_streaming, "no rung streamed — the ladder never exercised the miss path");
    assert!(saw_resident, "no rung was fully resident — the ladder never exercised the pinned path");
    assert_ne!(reference.unwrap(), FNV_INIT, "no bytes were delivered");
    println!("\n  ✅ byte-identical at every budget, on real weights");

    // --- overlap, on the same real file ---
    // A tight bind loop has no compute to hide reads behind, so the honest way to show the overlap is to
    // report both the hit counters AND the wall clock, and let them disagree if they disagree.
    let budget = total / 8;
    let plan = plan_layers(&layers, budget, 0, 4096);
    let mut pre = PrefetchCache::new(plan.clone(), layers.clone(), Arc::clone(&backing))?;
    pre.prefill()?;
    let t0 = std::time::Instant::now();
    let mut sum = FNV_INIT;
    for _tok in 0..TOKENS {
        for l in 0..layers.len() as u32 {
            sum = fnv1a(sum, pre.bind(l)?.0);
        }
    }
    let wall = t0.elapsed();
    let ps = pre.stats();
    assert_eq!(sum, reference.unwrap(), "prefetching changed the delivered bytes");
    println!(
        "\n  prefetch @ {:.1} MB: {} reads issued ahead of demand, {} forced synchronous, \
         overlap {:.1}%, {:.0} ms",
        gb(budget), ps.prefetch_hits, ps.sync_reads, 100.0 * ps.overlap_rate(),
        wall.as_secs_f64() * 1000.0
    );
    println!("  (identical bytes ✓. Wall-clock gain needs compute between binds to hide the reads behind —");
    println!("   this loop has none, so treat the overlap % as the signal and the ms as an upper bound.)");

    Ok(())
}

#[allow(dead_code)]
fn _assert_error_is_usable(e: TierError) -> String { e.to_string() }
