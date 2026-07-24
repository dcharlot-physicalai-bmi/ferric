// Find the GPU allocation wall: create dummy tensors (1 buffer each), periodically round-trip a probe.
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;
fn main() { pollster::block_on(run()); }
async fn run() {
    let args: Vec<String> = std::env::args().collect();
    let count: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(100_000);
    let elems: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1024); // f32 each (4B)
    let ctx = Arc::new(Context::new().await.unwrap());
    let data = vec![7.0f32; elems];
    let mut keep = Vec::new();
    let step = (count / 20).max(1);
    for i in 0..count {
        keep.push(Tensor::from_vec(&ctx, &data, &[elems]));
        if (std::env::var("ENDONLY").is_err() && i % step == 0) || i == count - 1 {
            let t = Tensor::from_vec(&ctx, &[1.0f32, 2.0, 3.0, 4.0], &[4]);
            let v = t.to_vec().await;
            let ok = v == [1.0, 2.0, 3.0, 4.0];
            println!("after {:>7} buffers ({:>6.2} GB): roundtrip {}", i + 1, ((i + 1) * elems * 4) as f64 / 1e9, if ok { "OK" } else { "BROKEN" });
            if !ok { return; }
        }
    }
    // The real check: did the KEPT buffers retain their contents (all 7.0), or were writes dropped?
    for (label, i) in [("first", 0usize), ("mid", keep.len()/2), ("last", keep.len()-1)] {
        let v = keep[i].to_vec().await;
        let ok = v.iter().all(|x| *x == 7.0);
        println!("kept[{label}] contents: {}", if ok { "INTACT" } else { "ZEROED/CORRUPT" });
    }
}
