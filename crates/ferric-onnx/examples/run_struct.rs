// Structural / int tier: Gather (embedding lookup) → Unsqueeze → Squeeze → Cast → Concat →
// Reshape → MatMul → Add. All-initializer graph (no runtime inputs). Validated vs onnxruntime.
use ferric_core::{max_abs_diff, Context};
use std::collections::HashMap;
fn main() { pollster::block_on(run()); }
fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }
async fn run() {
    let ctx = Context::new().await.unwrap();
    let dir = env!("CARGO_MANIFEST_DIR");
    let model = ferric_onnx::load(&std::fs::read(format!("{dir}/testdata/struct.onnx")).unwrap()).unwrap();
    let refy = f32s(&std::fs::read(format!("{dir}/testdata/struct.ref.bin")).unwrap());
    let (_, y) = model.run(&ctx, &HashMap::new()).await.unwrap();
    let d = max_abs_diff(&y, &refy);
    println!("Ferric struct · {:?} · ops: {}", ctx.backend, model.ops().join(","));
    println!("  max|ferric - onnxruntime| = {:.3e}", d);
    assert!(d < 1e-4, "struct mismatch {d}");
    println!("✅ Gather+Unsqueeze+Squeeze+Cast+Concat+Reshape graph runs in Ferric — matches onnxruntime");
}
