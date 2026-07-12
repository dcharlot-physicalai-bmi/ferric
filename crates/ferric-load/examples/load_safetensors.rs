// Round-trip: parse a mixed-dtype safetensors container (F32/F16/BF16) with Ferric's pure-Rust
// reader and verify every tensor dequantizes to f32 exactly matching the source.
use ferric_load::safetensors;
fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect() }
fn main() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let bytes = std::fs::read(format!("{dir}/testdata/model.safetensors")).unwrap();
    let tensors = safetensors(&bytes).unwrap();
    let mut ok = true;
    for (name, dt) in [("w_f32","F32"),("w_f16","F16"),("w_bf16","BF16")] {
        let t = tensors.get(name).unwrap();
        let refv = f32s(&std::fs::read(format!("{dir}/testdata/{name}.ref.bin")).unwrap());
        let d = t.data.iter().zip(&refv).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        let pass = d == 0.0 && t.data.len() == refv.len();
        ok &= pass;
        println!("  {} {:<7} {:<5} shape {:?} · exact-match diff = {d:.0e}", if pass {"✅"} else {"❌"}, name, dt, t.shape);
    }
    assert!(ok);
    println!("✅ Ferric safetensors reader: F32/F16/BF16 all dequantize EXACTLY — real checkpoints can load");
}
