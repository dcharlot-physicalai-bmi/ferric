// A synthetic pre-norm gated block exercising the transformer op set added for real models:
// LayerNormalization, Sigmoid, Gelu (erf), Sub, Mul, Add, Sqrt, Div, MatMul. Validated vs onnxruntime.
use ferric_core::{max_abs_diff, Context};
use std::collections::HashMap;
fn main() { pollster::block_on(run()); }
fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }
async fn run() {
    let ctx = Context::new().await.unwrap();
    let dir = env!("CARGO_MANIFEST_DIR");
    let model = ferric_onnx::load(&std::fs::read(format!("{dir}/testdata/block.onnx")).unwrap()).unwrap();
    let meta = std::fs::read_to_string(format!("{dir}/testdata/block.meta")).unwrap();
    let mut lines = meta.lines();
    let name = lines.next().unwrap().to_string();
    let shape: Vec<usize> = lines.next().unwrap().split(',').map(|s| s.parse().unwrap()).collect();
    let x = f32s(&std::fs::read(format!("{dir}/testdata/block.in.bin")).unwrap());
    let refy = f32s(&std::fs::read(format!("{dir}/testdata/block.ref.bin")).unwrap());
    let mut inp = HashMap::new();
    inp.insert(name, (x, shape));
    let (_, y) = model.run(&ctx, &inp).await.unwrap();
    let d = max_abs_diff(&y, &refy);
    println!("Ferric block · {:?} · ops: {}", ctx.backend, model.ops().join(","));
    println!("  max|ferric - onnxruntime| = {:.3e}", d);
    assert!(d < 1e-4, "block mismatch {d}");
    println!("✅ LayerNorm+Sigmoid+Gelu+Sub+Sqrt+Div block runs in Ferric — matches onnxruntime");
}
