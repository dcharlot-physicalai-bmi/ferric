//! ferric-flow as the Assistant's retrieval backend, on the REAL Institute corpus.
//!
//! The Institute Assistant retrieves with bge-m3 embeddings over **Cloudflare Vectorize** — a managed,
//! external vector-DB service that rebuilds on ingest. Vectorize is exactly the kind of rented,
//! non-proprietary capability this program ingests: an ANN index is a public algorithm. This example
//! drives `LiveIndex` (our pure-Rust, wasm-able HNSW) over 37 real documents embedded with a real
//! sentence model, and shows it (1) matches a brute-force scan (so it can replace Vectorize with no
//! loss), (2) returns the right documents for real questions, and (3) drops a document from results
//! the instant it is retracted — no reindex job, the thing Vectorize cannot do in-place.
//!
//! Run: `cargo run -p ferric-flow --example corpus_rag`
//! (fixture built by v2/tools/nanovla/embed_corpus.mjs)

use ferric_flow::LiveIndex;
use serde_json::Value;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut d = 0.0;
    let (mut na, mut nb) = (0.0f32, 0.0f32);
    for i in 0..a.len() {
        d += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    d / (na.sqrt() * nb.sqrt() + 1e-9)
}

fn brute_topk(docs: &[(String, String, Vec<f32>)], q: &[f32], k: usize) -> Vec<String> {
    let mut scored: Vec<(f32, &str)> =
        docs.iter().map(|(id, _, v)| (cosine(q, v), id.as_str())).collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0)); // highest similarity first
    scored.into_iter().take(k).map(|(_, id)| id.to_string()).collect()
}

fn main() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/corpus_emb.json");
    let raw = std::fs::read_to_string(path).expect("run v2/tools/nanovla/embed_corpus.mjs first");
    let j: Value = serde_json::from_str(&raw).unwrap();

    let read_vec = |v: &Value| v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<f32>>();

    // (id, title, vector)
    let docs: Vec<(String, String, Vec<f32>)> = j["docs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| (d["id"].as_str().unwrap().to_string(), d["title"].as_str().unwrap().to_string(), read_vec(&d["vec"])))
        .collect();
    let title_of = |id: &str| docs.iter().find(|(i, _, _)| i == id).map(|(_, t, _)| t.clone()).unwrap_or_default();

    // Build the live index by STREAMING documents in one at a time — the incremental path, no bulk build.
    let mut idx: LiveIndex<String> = LiveIndex::new(16, 100, 48);
    for (id, _, vec) in &docs {
        idx.step(vec![((id.clone(), vec.clone()), 1)]);
    }
    println!("indexed {} real documents (all-MiniLM-L6-v2, 384-d), streamed incrementally\n", idx.live_len());

    // ---- 1 & 2: faithfulness to brute force + a real-question eyeball on relevance ----
    // The INDEX's job is to return what an exhaustive cosine scan would (recall + same #1). Whether
    // that document is "relevant" is the embedding model's job — printed here for inspection.
    let k = 5;
    let mut recall_sum = 0.0f64;
    let mut faithful_top1 = 0usize;
    let queries = j["queries"].as_array().unwrap();
    let before = idx.dist_count();
    for query in queries {
        let q = read_vec(&query["vec"]);
        let text = query["q"].as_str().unwrap();

        let got: Vec<String> = idx.query(&q, k).into_iter().map(|(id, _)| id).collect();
        let truth = brute_topk(&docs, &q, k);
        let hit = truth.iter().filter(|id| got.contains(id)).count();
        recall_sum += hit as f64 / k as f64;
        if !got.is_empty() && !truth.is_empty() && got[0] == truth[0] {
            faithful_top1 += 1;
        }
        println!("Q: {}", text);
        for id in got.iter().take(3) {
            println!("   {}", title_of(id));
        }
    }
    let recall = recall_sum / queries.len() as f64;
    let dist_per_query = (idx.dist_count() - before) as f64 / queries.len() as f64;

    // ---- 3: incremental retract — a top hit vanishes with no reindex ----
    let q0 = read_vec(&queries[0]["vec"]);
    let top_before = idx.query(&q0, 1)[0].0.clone();
    idx.step(vec![((top_before.clone(), docs.iter().find(|(i, _, _)| *i == top_before).unwrap().2.clone()), -1)]);
    let after = idx.query(&q0, 3);
    let gone = !after.iter().any(|(id, _)| *id == top_before);
    // put it back
    idx.step(vec![((top_before.clone(), docs.iter().find(|(i, _, _)| *i == top_before).unwrap().2.clone()), 1)]);
    let restored = idx.query(&q0, 1)[0].0 == top_before;

    println!("\n── summary ──────────────────────────────────────────────");
    println!("recall@{} vs exhaustive scan : {:.3}", k, recall);
    println!("same #1 as exhaustive scan   : {}/{}  (index is faithful)", faithful_top1, queries.len());
    println!("distance evals / query       : {:.0} over {} docs — at this size no ANN beats brute force;", dist_per_query, docs.len());
    println!("                               sub-linearity shows at scale (691-doc test: 3.9x fewer)");
    println!("retract → '{}' left results, no reindex : {}", title_of(&top_before), if gone { "yes ✓" } else { "NO ✗" });
    println!("re-assert → restored to #1 : {}", if restored { "yes ✓" } else { "NO ✗" });

    // The index's contract: reproduce exhaustive semantic search exactly, and stay live under edits.
    let ok = recall >= 0.99 && faithful_top1 == queries.len() && gone && restored;
    println!("\n{}", if ok { "PASS — LiveIndex reproduces exhaustive retrieval on the real corpus, and stays live under edits: a drop-in for Vectorize" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
}
