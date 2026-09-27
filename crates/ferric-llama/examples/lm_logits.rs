//! **A dense LM's logits on ids the model's AUTHORS produced** — the Ferric side of
//! `scripts/lm_conformance.sh`, compared against `tests/fixtures/lm/` (their `transformers`).
//!
//!   cargo run -p ferric-llama --release --example lm_logits -- <model.gguf> <fixture.json>
//!
//! Reads the authors' token ids and their fixed vocabulary sample from the fixture (and, from
//! `refgen/lm_floor_ref.py` fixtures, the `position_offset` they ran at), runs ONE
//! stateless prefill (`Qwen3::forward` for llama / qwen2 / qwen3 / gemma3, `Qwen35::forward` for the
//! gated-delta-net hybrids, `Lfm2::forward` from an empty cache for the short-conv hybrids, `NemotronH::forward` for the
//! Mamba-2 hybrid —
//! dispatched on the file's `general.architecture`), and prints per
//! position: the sampled logits, the argmax, and the sum and sum-of-squares of the FULL row — so a
//! defect anywhere in the vocabulary moves a number even though only 128 ids are compared directly.
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_llama::{lfm2::{self, Lfm2}, nemotron_h::NemotronH, qwen3::Qwen3, qwen35::Qwen35};
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
    let mp = a.get(1).expect("usage: lm_logits <model.gguf> <fixture.json>");
    let fx = std::fs::read_to_string(a.get(2).expect("fixture.json")).expect("read fixture");
    let parse = |key: &str| -> Vec<u32> {
        field(&fx, key).split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().unwrap()).collect()
    };
    let ids = parse("ids");
    let sample = parse("sample_ids");
    // `position_offset` (refgen/lm_floor_ref.py `--pos-offset`): the authors ran these ids at positions
    // P..P+T-1 with no cache. An EMPTY cache whose position starts at P is the identical computation —
    // the same T x T causal attention over the same rows, rotated as if they sat at P. It reaches the
    // angles of position 30,000 without a 30k-token prefill on either side.
    let offset: usize = fx.find("\"position_offset\": ").map(|i| {
        let s = &fx[i + "\"position_offset\": ".len()..];
        s[..s.find([',', '}']).unwrap()].trim().parse().expect("position_offset")
    }).unwrap_or(0);

    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let g = GgufFile::open(mp).expect("open");
    // Dispatch on the file's own architecture: the gated-delta-net hybrids have their own runtime.
    let arch = match g.metadata().get("general.architecture") { Some(Meta::Str(a)) => a.clone(), _ => String::new() };
    eprintln!("arch: {arch}");
    assert!(offset == 0 || !(arch.starts_with("qwen35") || arch == "nemotron_h" || arch.starts_with("lfm2")),
            "position_offset is wired for the dense runtime only");
    // `FERRIC_LM_DECODE_FROM=N`: prefill ids[..N] as one forward, then feed the rest ONE TOKEN AT A TIME
    // through the same cache — a conversation growing past N. On Phi-3 with N = 4096 that crosses
    // LongRoPE's switch, and the rows from N on must equal the authors' full prefill of the whole text
    // (causal attention, one table): the cache recomputed at the crossing, not extended stale.
    let decode_from: Option<usize> = std::env::var("FERRIC_LM_DECODE_FROM").ok().map(|v| v.parse().expect("FERRIC_LM_DECODE_FROM"));
    let lg = if let Some(n) = decode_from {
        assert!(offset == 0 && n > 0 && n < ids.len(), "FERRIC_LM_DECODE_FROM={n} needs 0 < N < {} tokens, offset 0", ids.len());
        let m = Qwen3::load(&ctx, &g).expect("load");
        let mut cache = ferric_llama::qwen3::Cache::new(&m.cfg);
        let mut lg = m.forward_cached(&ids[..n], &mut cache).to_vec().await;
        for &t in &ids[n..] { lg.extend(m.forward_cached(&[t], &mut cache).to_vec().await); }
        lg
    } else if offset > 0 {
        let m = Qwen3::load(&ctx, &g).expect("load");
        let mut cache = ferric_llama::qwen3::Cache::new(&m.cfg);
        cache.pos = offset;
        m.forward_cached(&ids, &mut cache).to_vec().await
    } else if arch.starts_with("qwen35") {
        Qwen35::load(&ctx, &g).expect("load qwen35").forward(&ids).to_vec().await
    } else if arch == "nemotron_h" {
        NemotronH::load(&ctx, &g).expect("load nemotron_h").forward(&ids).expect("forward").to_vec().await
    } else if arch.starts_with("lfm2") {
        let m = Lfm2::load(&ctx, &g).expect("load lfm2");
        let mut cache = lfm2::Cache::new(&m.cfg);
        m.forward(&ids, &mut cache).to_vec().await
    } else {
        Qwen3::load(&ctx, &g).expect("load").forward(&ids).to_vec().await
    };
    // ⛔ An all-zero logits buffer is not a model output: Metal can drop buffer contents during a large
    // allocation burst and read them back as zeros with no error (`Context::flush`). Seen here loading a
    // 15 GB F32 Phi-3.5 while other GPU jobs ran — max |diff| 57.6 and sum-of-squares error exactly 1.000,
    // which a gate must not report as a verdict on the model.
    if lg.iter().all(|&x| x == 0.0) {
        eprintln!("the device returned ALL-ZERO logits ({} values) — a dropped buffer, not a model output; rerun on a quieter device", lg.len());
        std::process::exit(3);
    }
    let v = lg.len() / ids.len();
    // LM_LOGITS_DUMP=<path>: every row, raw little-endian f32 [T, V] — for full-vocabulary metrics
    // (KL, top-k) that the fixture's 128-id sample cannot give.
    if let Ok(p) = std::env::var("LM_LOGITS_DUMP") {
        std::fs::write(&p, lg.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).expect("dump");
    }
    for t in 0..ids.len() {
        let r = &lg[t * v..(t + 1) * v];
        let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
        let (mut sum, mut ssq) = (0f64, 0f64);
        for (i, &x) in r.iter().enumerate() {
            if x > bv { bv = x; best = i; }
            sum += x as f64; ssq += (x as f64) * (x as f64);
        }
        let s: Vec<String> = sample.iter().map(|&i| format!("{:.5}", r[i as usize])).collect();
        println!("ROW {t} {best} {sum:.4} {ssq:.4} {}", s.join(" "));
    }
}
