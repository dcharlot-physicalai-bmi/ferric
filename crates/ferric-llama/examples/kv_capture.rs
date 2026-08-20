//! Capture a **real** K/V cache from a real forward pass, to a file the tensor-crate tools can read.
//!
//! KV quantization cannot be evaluated on synthetic gaussians. K and V have different distributions,
//! and K in particular has outlier channels — which is the entire reason a naive per-tensor or
//! per-token scale fails on it. So the error study reads captured tensors, and this is what captures
//! them: prefill a real prompt through a real GGUF and dump every layer's K and V exactly as the
//! runtime stored them.
//!
//! Reads its subject from argv — a hardcoded model path hid a live divergence in this tree for a day.
//!
//!   cargo run -p ferric-llama --example kv_capture --release -- <model.gguf> <out.fkvc> ["prompt"]
//!
//! File format (little-endian, all f32 data contiguous):
//!   "FKVC" | u32 version=1 | u32 n_layer | u32 rows | u32 width | u32 n_kv_head | u32 head_dim
//!   then per layer: K[rows*width] f32, V[rows*width] f32
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }

async fn run() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: kv_capture <model.gguf> <out.fkvc> [prompt]");
    let out_path = args.get(2).expect("usage: kv_capture <model.gguf> <out.fkvc> [prompt]");
    let prompt = args.get(3).map(|s| s.as_str()).unwrap_or(
        "The Rhine rises in the Swiss Alps and flows north through Germany to the North Sea. \
         Its valley has carried trade since the Roman period, and the river remains one of the \
         busiest inland waterways in the world. Barges leaving Rotterdam reach Basel in about a week, \
         moving coal, ore, chemicals and containers past Cologne, Koblenz and Mainz. In 2018 a drought \
         dropped the water level at Kaub below 30 centimetres and the traffic stopped; the effect on \
         German industrial output was measurable in the national accounts for that quarter.",
    );

    let ctx = Arc::new(Context::new().await.unwrap());
    let g = GgufFile::open(path).unwrap();

    let tokens: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokens in {path}"),
    };
    let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
        Some(Meta::Arr(a)) => a
            .iter()
            .filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None })
            .collect(),
        _ => panic!("no merges in {path}"),
    };
    let bpe = Bpe::new(vocab, &merges);

    let m = Qwen3::load(&ctx, &g).unwrap();
    let c = &m.cfg;
    let bos_id = match g.metadata.get("tokenizer.ggml.bos_token_id") { Some(Meta::U(v)) => Some(*v as u32), _ => None };
    let add_bos = match g.metadata.get("tokenizer.ggml.add_bos_token") { Some(Meta::Bool(b)) => *b, _ => bos_id.is_some() };
    let mut ids = bpe.encode(prompt);
    if add_bos { if let Some(b) = bos_id { ids.insert(0, b); } }

    // A real prefill: the K/V written here are the ones a real generation would attend over —
    // post-projection, post-RoPE for K, and per-layer.
    let mut cache = Cache::new(c);
    let logits = m.forward_cached(&ids, &mut cache).to_vec().await;
    let row = &logits[logits.len() - c.n_vocab..];
    let top = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
    println!(
        "{} · {} layers · {}kv x {} · {} tokens · argmax id {} ({:+.4})",
        path, c.n_layer, c.n_head_kv, c.head_dim, ids.len(), top.0, top.1
    );

    let layers = cache.layers();
    let rows = layers[0].0.len();
    let width = layers[0].0.width();
    assert_eq!(rows, ids.len(), "cache holds {rows} rows for {} tokens", ids.len());
    assert_eq!(width, c.n_head_kv * c.head_dim, "unexpected cache row width {width}");

    let mut f = std::io::BufWriter::new(std::fs::File::create(out_path).unwrap());
    f.write_all(b"FKVC").unwrap();
    for v in [1u32, c.n_layer as u32, rows as u32, width as u32, c.n_head_kv as u32, c.head_dim as u32] {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    for (il, (k, v)) in layers.iter().enumerate() {
        assert_eq!(k.len(), rows, "layer {il} K has {} rows", k.len());
        assert_eq!(v.len(), rows, "layer {il} V has {} rows", v.len());
        for buf in [k, v] {
            let data = buf.view(&ctx).to_vec().await;
            assert_eq!(data.len(), rows * width);
            let finite = data.iter().filter(|x| x.is_finite()).count();
            assert_eq!(finite, data.len(), "layer {il} cache holds non-finite values — capture is junk");
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for x in &data { bytes.extend_from_slice(&x.to_le_bytes()); }
            f.write_all(&bytes).unwrap();
        }
    }
    f.flush().unwrap();
    println!(
        "wrote {out_path} · {} layers x 2 x [{rows}, {width}] f32 = {:.1} MB",
        c.n_layer,
        (c.n_layer * 2 * rows * width * 4) as f64 / 1e6
    );
}
