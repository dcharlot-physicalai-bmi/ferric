//! Offline "ingest": read the real-corpus embeddings, build the ferric-flow graph, serialize it, and
//! emit a fixture (base64 index + titles + query vectors) the Worker POC loads. In production this is
//! the `/ingest` step — build once from the bge-m3 embeddings, `save()` the bytes to R2/KV.
//!
//! Run: `cargo run -p ferric-flow --example build_corpus_index`

use ferric_flow::LiveIndex;
use serde_json::Value;

fn b64(data: &[u8]) -> String {
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(A[((n >> 18) & 63) as usize] as char);
        out.push(A[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 { A[((n >> 6) & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { A[(n & 63) as usize] as char } else { '=' });
    }
    out
}

fn main() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/examples");
    let raw = std::fs::read_to_string(format!("{dir}/corpus_emb.json")).unwrap();
    let j: Value = serde_json::from_str(&raw).unwrap();
    let read_vec = |v: &Value| v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<f32>>();

    let mut idx: LiveIndex<String> = LiveIndex::new(16, 100, 48);
    let mut titles = String::from("{");
    for (i, d) in j["docs"].as_array().unwrap().iter().enumerate() {
        let id = d["id"].as_str().unwrap();
        let title = d["title"].as_str().unwrap();
        idx.step(vec![((id.to_string(), read_vec(&d["vec"])), 1)]);
        if i > 0 {
            titles.push(',');
        }
        let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        titles.push_str(&format!("\"{}\":\"{}\"", esc(id), esc(title)));
    }
    titles.push('}');

    let bytes = idx.save();
    // queries: label + vector, straight through
    let mut queries = String::from("[");
    for (i, q) in j["queries"].as_array().unwrap().iter().enumerate() {
        if i > 0 {
            queries.push(',');
        }
        let vec: Vec<String> = read_vec(&q["vec"]).iter().map(|x| format!("{x}")).collect();
        queries.push_str(&format!("{{\"q\":{},\"vec\":[{}]}}", q["q"], vec.join(",")));
    }
    queries.push(']');

    let out = format!(
        "{{\"indexB64\":\"{}\",\"bytes\":{},\"docs\":{},\"titles\":{},\"queries\":{}}}",
        b64(&bytes),
        bytes.len(),
        idx.live_len(),
        titles,
        queries
    );
    let dest = concat!(env!("CARGO_MANIFEST_DIR"), "/../ferric-flow-wasm/worker-poc/fixture.json");
    std::fs::create_dir_all(std::path::Path::new(dest).parent().unwrap()).unwrap();
    std::fs::write(dest, &out).unwrap();
    println!("wrote {dest}: {} docs, index {} bytes ({} b64)", idx.live_len(), bytes.len(), b64(&bytes).len());
}
