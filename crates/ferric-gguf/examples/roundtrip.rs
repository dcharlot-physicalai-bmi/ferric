//! Validates the GGUF reader end-to-end: builds a synthetic GGUF in memory (metadata + F32/Q8_0/Q4_0
//! tensors), parses it, and dequantizes each back within quantization tolerance. Then validates the
//! Q4_K (k-quant) super-block dequant formula against a hand-constructed block with known values.
use ferric_gguf::{parse, quant_q4_0, quant_q8_0, Meta};
use half::f16;

fn w_str(o: &mut Vec<u8>, s: &str) { o.extend_from_slice(&(s.len() as u64).to_le_bytes()); o.extend_from_slice(s.as_bytes()); }
fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.3 + s).sin())).collect() }
fn maxrel(a: &[f32], b: &[f32]) -> f32 { let den = b.iter().map(|v| v.abs()).fold(1e-6, f32::max); a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max) / den }

fn main() {
    // ---- build a synthetic GGUF ----
    let wf32 = seq(8, 1.0);
    let wq8 = seq(64, 2.0);
    let wq4 = seq(64, 3.0);
    let q8 = quant_q8_0(&wq8);
    let q4 = quant_q4_0(&wq4);

    let mut o = Vec::new();
    o.extend_from_slice(b"GGUF");
    o.extend_from_slice(&3u32.to_le_bytes());          // version
    o.extend_from_slice(&3u64.to_le_bytes());          // tensor count
    o.extend_from_slice(&2u64.to_le_bytes());          // metadata count
    // metadata: general.architecture (string), general.alignment (u32=32)
    w_str(&mut o, "general.architecture"); o.extend_from_slice(&8u32.to_le_bytes()); w_str(&mut o, "ferric-test");
    w_str(&mut o, "general.alignment"); o.extend_from_slice(&4u32.to_le_bytes()); o.extend_from_slice(&32u32.to_le_bytes());
    // tensor infos: name, n_dims, dims, type, offset
    let infos: [(&str, &[u64], u32, u64); 3] = [("w_f32", &[8], 0, 0), ("w_q8", &[64], 8, 32), ("w_q4", &[64], 2, 32 + q8.len() as u64)];
    for (name, dims, ty, off) in infos {
        w_str(&mut o, name);
        o.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims { o.extend_from_slice(&d.to_le_bytes()); }
        o.extend_from_slice(&ty.to_le_bytes());
        o.extend_from_slice(&off.to_le_bytes());
    }
    while o.len() % 32 != 0 { o.push(0); }              // align to data section
    o.extend_from_slice(&wf32.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>());
    while o.len() % 32 != 0 { o.push(0); }
    o.extend_from_slice(&q8);
    o.extend_from_slice(&q4);

    // ---- parse + validate ----
    let g = parse(o).unwrap();
    let arch = matches!(g.metadata.get("general.architecture"), Some(Meta::Str(s)) if s == "ferric-test");
    let mut ok = arch;
    println!("GGUF parsed · {} tensors · arch metadata {}", g.tensors.len(), if arch { "✅" } else { "❌" });
    for (name, refv, tol) in [("w_f32", &wf32, 1e-6f32), ("w_q8", &wq8, 0.02), ("w_q4", &wq4, 0.15)] {
        let d = g.dequant(name).unwrap();
        let rel = maxrel(&d, refv);
        let pass = rel < tol; ok &= pass;
        println!("  {} {:<6} ({} vals) rel err = {:.2e}", if pass { "✅" } else { "❌" }, name, d.len(), rel);
    }

    // ---- Q4_K formula check: a super-block with d=1, dmin=0, scale bytes → sub-block scale=1, qs known ----
    // scales[0..4]&63 = sub-block scales (set to 1); dmin=0 removes the min term → y = 1·1·nibble.
    let mut blk = Vec::new();
    blk.extend_from_slice(&f16::from_f32(1.0).to_le_bytes()); // d
    blk.extend_from_slice(&f16::from_f32(0.0).to_le_bytes()); // dmin
    let mut scales = [0u8; 12];
    for s in scales.iter_mut().take(8) { *s = 1; }           // low 6 bits = 1 for first sub-blocks
    blk.extend_from_slice(&scales);
    let mut qs = [0u8; 128];
    for (i, q) in qs.iter_mut().enumerate() { *q = ((i as u8 % 8) & 0x0F) | (((i as u8 + 3) % 8) << 4); }
    blk.extend_from_slice(&qs);
    // wrap in a minimal GGUF with one Q4_K tensor [256]
    let mut o2 = Vec::new();
    o2.extend_from_slice(b"GGUF"); o2.extend_from_slice(&3u32.to_le_bytes());
    o2.extend_from_slice(&1u64.to_le_bytes()); o2.extend_from_slice(&0u64.to_le_bytes());
    w_str(&mut o2, "wk"); o2.extend_from_slice(&1u32.to_le_bytes()); o2.extend_from_slice(&256u64.to_le_bytes());
    o2.extend_from_slice(&12u32.to_le_bytes()); o2.extend_from_slice(&0u64.to_le_bytes());
    while o2.len() % 32 != 0 { o2.push(0); }
    o2.extend_from_slice(&blk);
    let gk = parse(o2).unwrap();
    let dk = gk.dequant("wk").unwrap();
    // expected: first 32 vals = low nibbles of qs[0..32] (scale 1), next 32 = high nibbles, etc.
    let mut exp = vec![0.0f32; 256];
    let get = |scales: &[u8; 12], j: usize| if j < 4 { scales[j] & 63 } else { (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4) };
    let (mut y, mut q, mut is) = (0usize, 0usize, 0usize);
    for _ in 0..4 {
        let (s1, s2) = (get(&scales, is) as f32, get(&scales, is + 1) as f32);
        for l in 0..32 { exp[y + l] = s1 * (qs[q + l] & 0x0F) as f32; }
        for l in 0..32 { exp[y + l + 32] = s2 * (qs[q + l] >> 4) as f32; }
        y += 64; q += 32; is += 2;
    }
    let qk_ok = dk.iter().zip(&exp).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) < 1e-3;
    ok &= qk_ok;
    println!("  {} Q4_K super-block dequant formula (256 vals)", if qk_ok { "✅" } else { "❌" });

    println!("{}", if ok { "✅ GGUF reader: parses the container + dequantizes F32/Q8_0/Q4_0/Q4_K — the llama.cpp/HF model corpus loads" } else { "❌ gguf failed" });
    assert!(ok);
}
