// Run the REAL SmolVLA projector components (from the published ainekko/smolvla_base_onnx export)
// through Ferric's ONNX importer on the GPU, validating each against onnxruntime. Pure Rust, owned stack.
use ferric_core::{max_abs_diff, Context};
use std::collections::HashMap;
fn main() { pollster::block_on(run()); }
fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }
async fn run() {
    let ctx = Context::new().await.unwrap();
    let dir = env!("CARGO_MANIFEST_DIR");
    println!("Ferric ONNX · REAL SmolVLA projectors · {:?}", ctx.backend);
    let list = std::fs::read_to_string(format!("{dir}/testdata/proj/list.txt")).unwrap();
    let mut ok = true;
    for p in list.lines().filter(|l| !l.is_empty()) {
        let model = ferric_onnx::load(&std::fs::read(format!("{dir}/testdata/proj/{p}.onnx")).unwrap()).unwrap();
        let meta = std::fs::read_to_string(format!("{dir}/testdata/proj/{p}.meta")).unwrap();
        let mut lines = meta.lines();
        let name = lines.next().unwrap().to_string();
        let shape: Vec<usize> = lines.next().unwrap().split(',').map(|s| s.parse().unwrap()).collect();
        let x = f32s(&std::fs::read(format!("{dir}/testdata/proj/{p}.in.bin")).unwrap());
        let refy = f32s(&std::fs::read(format!("{dir}/testdata/proj/{p}.ref.bin")).unwrap());
        let mut inp = HashMap::new();
        inp.insert(name, (x, shape));
        match model.run(&ctx, &inp).await {
            Ok((_, y)) => {
                let d = max_abs_diff(&y, &refy);
                let pass = d < 1e-3;
                ok &= pass;
                println!("  {} {:<22} {:>2} ops · max|ferric-ort| = {:.3e}", if pass { "✅" } else { "❌" }, p, model.ops().len(), d);
            }
            Err(e) => { ok = false; println!("  ❌ {:<22} ERROR: {}", p, e); }
        }
    }
    println!("{}", if ok { "✅ ALL REAL SmolVLA PROJECTORS RAN IN FERRIC — match onnxruntime" } else { "❌ a projector failed" });
    assert!(ok);
}
