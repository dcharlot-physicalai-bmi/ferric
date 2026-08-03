//! SIZE PAYOFF, MADE REAL — pack the ACTUAL Qwen2.5-0.5B ternary transformer at ~1.6 bpw (5 trits/byte,
//! the lossless base-3 packing) + one f16 scale per group, WRITE it to disk, MEASURE the true on-disk size,
//! and PROVE the trit packing is lossless (decode→re-encode is byte-identical). Turns the "≈20× smaller"
//! claim from arithmetic into a measured artifact. The QUALITY at which this size runs is the QAT result
//! ([ternary_qat_model.rs], deployed ≈456-530 ppl); THIS example measures the SIZE. Pure CPU (dequant + pack).
//!   cargo run -p ferric-llama --example ternary_pack_model --release -- <qwen2.5-0.5b.gguf>
use ferric_gguf::{GgufFile, Meta};
use half::f16;

const GS: usize = 128; // group size (one f16 scale per GS weights)

// per-group ternary (Δ=0.7·mean|w|, scale=mean|w| over kept) + 5 trits/byte packing. scale stored f16.
fn ternary_encode(w: &[f32], gs: usize) -> (Vec<u8>, Vec<f32>) {
    let ng = (w.len() + gs - 1) / gs;
    let bpg = (gs + 4) / 5;
    let mut packed = vec![0u8; ng * bpg];
    let mut scales = vec![0f32; ng];
    for gi in 0..ng {
        let (lo, hi) = (gi * gs, ((gi + 1) * gs).min(w.len()));
        let grp = &w[lo..hi];
        let mean_abs = grp.iter().map(|x| x.abs()).sum::<f32>() / grp.len() as f32;
        let delta = 0.7 * mean_abs;
        let (mut ss, mut sc) = (0f32, 0usize);
        let trits: Vec<i32> = grp.iter().map(|&x| if x.abs() > delta { ss += x.abs(); sc += 1; if x > 0.0 { 1 } else { -1 } } else { 0 }).collect();
        scales[gi] = f16::from_f32(if sc > 0 { ss / sc as f32 } else { 0.0 }).to_f32(); // stored as f16
        for (c, chunk) in trits.chunks(5).enumerate() {
            let (mut b, mut mul) = (0u32, 1u32);
            for &t in chunk { b += (t + 1) as u32 * mul; mul *= 3; }
            packed[gi * bpg + c] = b as u8;
        }
    }
    (packed, scales)
}
fn ternary_decode(packed: &[u8], scales: &[f32], n: usize, gs: usize) -> Vec<f32> {
    let bpg = (gs + 4) / 5;
    let mut out = vec![0f32; n];
    for gi in 0..scales.len() {
        let (s, base) = (scales[gi], gi * gs);
        for k in 0..gs { if base + k >= n { break; }
            let mut v = packed[gi * bpg + k / 5] as u32; for _ in 0..(k % 5) { v /= 3; }
            out[base + k] = s * ((v % 3) as i32 - 1) as f32;
        }
    }
    out
}
fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
    let den: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    num / den
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: ternary_pack_model <qwen2.5-0.5b.gguf>");
    let g = GgufFile::open(&path).unwrap();
    let arch = match g.metadata.get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => "qwen2".into() };
    let nl = match g.metadata.get(&format!("{arch}.block_count")) { Some(Meta::U(v)) => *v as usize, _ => 24 };
    let projs = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];

    let (mut params, mut trit_bytes, mut scale_cnt) = (0usize, 0usize, 0usize);
    let mut blob: Vec<u8> = Vec::new(); // trits ++ f16 scales, written to disk
    let mut lossless = true;
    let mut wsum = 0f32; // param-weighted reconstruction error accumulator
    for il in 0..nl { for p in projs {
        let name = format!("blk.{il}.{p}.weight");
        if g.tensor(&name).is_none() { continue; }
        let w = g.dequant(&name).unwrap();
        let n = w.len();
        let (packed, scales) = ternary_encode(&w, GS);
        let recon = ternary_decode(&packed, &scales, n, GS);
        // packing losslessness: re-encoding the decoded weights must yield byte-identical trits.
        let (packed2, _) = ternary_encode(&recon, GS);
        if packed2 != packed { lossless = false; }
        wsum += rel_err(&w, &recon) * n as f32;
        params += n; trit_bytes += packed.len(); scale_cnt += scales.len();
        blob.extend_from_slice(&packed);
        for s in &scales { blob.extend_from_slice(&f16::from_f32(*s).to_le_bytes()); }
    }}

    let out = format!("{}/qwen2.5-0.5b-ternary-transformer.bin", std::env::var("TMPDIR").unwrap_or("/tmp".into()).trim_end_matches('/'));
    std::fs::write(&out, &blob).unwrap();
    let disk = std::fs::metadata(&out).unwrap().len();

    let mb = |b: usize| b as f64 / 1e6;
    let scale_bytes = scale_cnt * 2;
    let tern = trit_bytes + scale_bytes;
    let (f32b, f16b, q8b) = (params * 4, params * 2, (params as f64 * 1.0625) as usize); // Q8_0 = (32+2)/32 B/w
    let emb = g.tensor("token_embd.weight").map(|t| t.dims.iter().product::<u64>() as usize).unwrap_or(0);

    println!("Qwen2.5-0.5B · {nl} layers · ternary TRANSFORMER (168 projections, {} params)\n", params);
    println!("PACKING (5 trits/byte, base-3, + one f16 scale per {GS}):");
    println!("  lossless (decode→re-encode byte-identical): {}", if lossless { "YES ✓" } else { "NO ✗" });
    println!("  {:.3} bpw trits + {:.3} bpw scales = {:.3} bpw total", trit_bytes as f64 * 8.0 / params as f64, scale_bytes as f64 * 8.0 / params as f64, tern as f64 * 8.0 / params as f64);
    println!("  param-weighted reconstruction rel-error (raw ternary floor): {:.3}\n", wsum / params as f32);
    println!("TRANSFORMER SIZE:");
    println!("  f32           {:>8.1} MB   (1×)", mb(f32b));
    println!("  f16           {:>8.1} MB   ({:.1}× smaller)", mb(f16b), f32b as f64 / f16b as f64);
    println!("  Q8_0 (today)  {:>8.1} MB   ({:.1}× smaller)", mb(q8b), f32b as f64 / q8b as f64);
    println!("  ternary 1.6bpw{:>8.1} MB   ({:.1}× smaller than f32, {:.1}× < Q8)  ← on disk {:.1} MB", mb(tern), f32b as f64 / tern as f64, q8b as f64 / tern as f64, disk as f64 / 1e6);
    println!("\n  + token_embd/head ({} params, kept f32 by QAT) = {:.0} MB f16 / {:.0} MB Q4 — quantized separately for deploy.", emb, mb(emb * 2), mb(emb / 2));
    println!("\n✅ The ternary transformer is a REAL {:.0} MB file on disk ({:.1}× smaller than f32, {:.1}× < the Q8 GGUF we run),",
        mb(tern), f32b as f64 / tern as f64, q8b as f64 / tern as f64);
    println!("   with LOSSLESS packing verified. QUALITY at this size = the QAT result (ternary_qat_model.rs). 16B analog ≈ {:.1} GB.", 16.0e9 * (tern as f64 * 8.0 / params as f64) / 8.0 / 1e9);
}
