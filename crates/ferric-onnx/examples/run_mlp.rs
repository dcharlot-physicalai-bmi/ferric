// Load a REAL ONNX model (a small MLP) and run it end-to-end on Ferric's GPU tensor ops, validating
// against the onnxruntime reference output. Pure Rust, our owned stack, native GPU.
use ferric_core::{max_abs_diff, Context};
use std::collections::HashMap;

fn main() { pollster::block_on(run()); }

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}
fn ref_y(json: &str) -> Vec<f32> {
    let s = json.split("\"y\"").nth(1).unwrap();
    let inner = s.split('[').nth(1).unwrap().split(']').next().unwrap();
    inner.split(',').filter_map(|x| x.trim().parse::<f32>().ok()).collect()
}

async fn run() {
    let ctx = Context::new().await.expect("ctx");
    let dir = env!("CARGO_MANIFEST_DIR");
    let model = ferric_onnx::load(&std::fs::read(format!("{dir}/testdata/mlp.onnx")).expect("model")).expect("parse");
    println!("Ferric ONNX · {:?} · graph ops: {:?}", ctx.backend, model.ops());

    let x = f32s(&std::fs::read(format!("{dir}/testdata/x.bin")).expect("x"));
    let mut inputs = HashMap::new();
    inputs.insert("X".to_string(), (x, vec![1usize, 8]));

    let (out_name, y) = model.run(&ctx, &inputs).await.expect("run");
    let refy = ref_y(&std::fs::read_to_string(format!("{dir}/testdata/ref.json")).expect("ref"));
    let diff = max_abs_diff(&y, &refy);
    println!("output '{}' = {:?}", out_name, y.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>());
    println!("ort ref = {:?}", refy);
    println!("max|ferric-onnxruntime| = {:.3e}", diff);
    assert!(diff < 1e-4, "diverged: {diff}");
    println!("✅ REAL ONNX MODEL RAN IN FERRIC — matches onnxruntime on {:?}", ctx.backend);
}
