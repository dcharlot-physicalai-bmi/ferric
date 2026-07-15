//! GgufFile (lazy, file-backed) vs the whole-file parse path, on a real checkpoint.
use ferric_gguf::GgufFile;
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let t0 = std::time::Instant::now();
    let g = GgufFile::open(&p).unwrap();
    println!("opened lazily in {:?} — {} tensors, {} meta keys (file stays on disk)", t0.elapsed(), g.tensors.len(), g.metadata.len());
    for n in ["output_norm.weight", "blk.0.ssm_a", "blk.0.ssm_conv1d.weight", "blk.0.attn_qkv.weight", "token_embd.weight"] {
        let t = g.tensor(n).unwrap();
        let ty = t.ggml_type; let dims = t.dims.clone();
        let t0 = std::time::Instant::now();
        let raw = g.raw(n).unwrap();
        let d = g.dequant(n).unwrap();
        let fin = d.iter().all(|v| v.is_finite());
        let amax = d.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        println!("  {} {:<26} {:?} ty={:<2} raw={:>10} B → {:>10} vals  amax={:.4}  finite={} [{:?}]",
            if fin { "✅" } else { "❌" }, n, dims, ty, raw.len(), d.len(), amax, fin, t0.elapsed());
        assert!(fin);
    }
}
