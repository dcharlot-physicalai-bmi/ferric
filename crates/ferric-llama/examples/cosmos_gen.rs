//! Run NVIDIA Cosmos 3 Edge's autoregressive text tower on Ferric. Takes token IDs directly (encode
//! with the model's tokenizer separately) and greedily generates. Proves Ferric loads + runs the
//! `cosmos3_edge` safetensors checkpoint.
//!
//! usage: cosmos_gen <cosmos3-edge-dir> <id,id,...> [--gen N]
use ferric_core::Context;
use ferric_llama::cosmos::Cosmos;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).expect("usage: cosmos_gen <dir> <ids> [--gen N]");
    let ids: Vec<u32> = args.get(2).map(|s| s.split(',').map(|x| x.trim().parse().expect("bad id")).collect()).unwrap_or_default();
    let n: usize = args.iter().position(|a| a == "--gen").map(|i| args[i + 1].parse().unwrap()).unwrap_or(0);

    let ctx = Arc::new(Context::new().await.unwrap());
    let t0 = std::time::Instant::now();
    let m = Cosmos::load(&ctx, dir).unwrap();
    let c = &m.cfg;
    println!("loaded Cosmos 3 Edge (text tower) in {:?} · {} layers · d={} · {}h/{}kv×{} · ff={} · vocab={}",
        t0.elapsed(), c.n_layer, c.n_embd, c.n_head, c.n_head_kv, c.head_dim, c.n_ff, c.n_vocab);
    if ids.is_empty() { println!("(no ids — load-only)"); return; }

    let argmax = |row: &[f32]| (0..c.n_vocab).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap() as u32;
    let t0 = std::time::Instant::now();
    let v = m.forward(&ids).to_vec().await;
    let last = &v[(ids.len() - 1) * c.n_vocab..];
    let finite = last.iter().all(|x| x.is_finite());
    let mut idx: Vec<usize> = (0..c.n_vocab).collect();
    idx.sort_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap());
    println!("\nforward {:?} ({} tok) · finite={finite}\n  top-8 next-token ids:", t0.elapsed(), ids.len());
    for &i in idx.iter().take(8) { println!("      {:9.4}  [{i}]", last[i]); }

    if n > 0 {
        let mut seq = ids.clone();
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let v = m.forward(&seq).to_vec().await; // stateless re-prefill (simple)
            let next = argmax(&v[v.len() - c.n_vocab..]);
            if next == 11 { break; } // <|im_end|>
            seq.push(next);
        }
        println!("\n  generated {} tok in {:.1}s\n  ids: {:?}", seq.len() - ids.len(), t0.elapsed().as_secs_f64(), &seq[ids.len()..]);
    }
}
