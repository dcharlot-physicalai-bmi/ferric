//! **Dequantize real DeepSeek V4 F8_E4M3_B128 bytes and dump f32 for a reference diff.**
//!
//! The layout and bias were verified from headers and fork source; this is the last rung — the
//! ELEMENT decode against a reference on REAL weights. Reads raw block bytes from argv (a tensor
//! region fetched by HTTP range from the live file), decodes with the shipped `deq_f8_e4m3_b128`,
//! writes little-endian f32 to argv[2], and prints stats a wrong layout could not produce.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (inp, out) = (a.get(1).expect("usage: f8_ref_diff <raw.f8> <out.f32>"), a.get(2).expect("out path"));
    let raw = std::fs::read(inp).expect("read");
    assert_eq!(raw.len() % 129, 0, "not whole 129-byte blocks: {}", raw.len());
    let n = raw.len() / 129 * 128;
    let v = ferric_gguf::deq_f8_e4m3_b128(&raw, n).expect("decode");
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(out, &bytes).expect("write");
    let (mut nan, mut inf) = (0usize, 0usize);
    let mut amax = 0f32; let mut sum2 = 0f64;
    for &x in &v {
        if x.is_nan() { nan += 1 } else if x.is_infinite() { inf += 1 }
        else { amax = amax.max(x.abs()); sum2 += (x as f64) * (x as f64); }
    }
    println!("n={} nan={nan} inf={inf} amax={amax:.6} rms={:.6}", n, (sum2 / n as f64).sqrt());
}
