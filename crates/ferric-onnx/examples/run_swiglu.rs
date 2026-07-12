// A modern Llama/SmolVLA-style decoder FFN sub-block: RMSNorm → SwiGLU (SiLU-gated) → residual.
// Ferric runs the FUSED RMSNormalization op; validated against onnxruntime running the math decomposition.
use ferric_core::{max_abs_diff, Context};
use std::collections::HashMap;
fn main() { pollster::block_on(run()); }
fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }
async fn run() {
    let ctx = Context::new().await.unwrap();
    let dir = env!("CARGO_MANIFEST_DIR");
    let model = ferric_onnx::load(&std::fs::read(format!("{dir}/testdata/swiglu.onnx")).unwrap()).unwrap();
    let meta = std::fs::read_to_string(format!("{dir}/testdata/swiglu.meta")).unwrap();
    let mut lines = meta.lines();
    let name = lines.next().unwrap().to_string();
    let shape: Vec<usize> = lines.next().unwrap().split(',').map(|s| s.parse().unwrap()).collect();
    let x = f32s(&std::fs::read(format!("{dir}/testdata/swiglu.in.bin")).unwrap());
    let refy = f32s(&std::fs::read(format!("{dir}/testdata/swiglu.ref.bin")).unwrap());
    let mut inp = HashMap::new();
    inp.insert(name, (x, shape));
    let (_, y) = model.run(&ctx, &inp).await.unwrap();
    let d = max_abs_diff(&y, &refy);
    println!("Ferric SwiGLU+RMSNorm · {:?} · ops: {}", ctx.backend, model.ops().join(","));
    println!("  max|ferric(fused) - onnxruntime(decomposed)| = {:.3e}", d);
    assert!(d < 1e-4, "swiglu mismatch {d}");
    println!("✅ Modern RMSNorm+SwiGLU decoder FFN sub-block runs in Ferric — matches onnxruntime");
}
