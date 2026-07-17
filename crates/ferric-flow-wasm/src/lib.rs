//! WASM surface for ferric-flow retrieval. A Cloudflare Worker (or a browser) loads a serialized
//! HNSW graph once and answers top-k queries in wasm — the in-Worker, edge-native stand-in for a
//! managed vector DB (Cloudflare Vectorize). Build with `wasm-pack build --target bundler`.

use ferric_flow::LiveIndex;
use wasm_bindgen::prelude::*;

/// A loaded, queryable retrieval index. Construct once per isolate from `LiveIndex::save()` bytes,
/// then call `query` per request.
#[wasm_bindgen]
pub struct Rag {
    idx: LiveIndex<String>,
}

#[wasm_bindgen]
impl Rag {
    /// Rehydrate from the bytes produced by `LiveIndex::<String>::save()` (server-side, on ingest).
    #[wasm_bindgen(constructor)]
    pub fn new(index_bytes: &[u8]) -> Rag {
        Rag { idx: LiveIndex::<String>::load(index_bytes) }
    }

    /// Top-`k` document ids for a query embedding, as a JSON string `[{"id":..,"dist":..}, ...]`,
    /// nearest first. JSON is built by hand so the wasm crate stays dependency-light.
    pub fn query(&self, embedding: &[f32], k: usize) -> String {
        let hits = self.idx.query(embedding, k);
        let mut s = String::from("[");
        for (i, (id, dist)) in hits.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let esc = id.replace('\\', "\\\\").replace('"', "\\\"");
            s.push_str(&format!("{{\"id\":\"{}\",\"dist\":{:.6}}}", esc, dist));
        }
        s.push(']');
        s
    }

    /// Number of live (non-tombstoned) documents in the index.
    #[wasm_bindgen(js_name = liveLen)]
    pub fn live_len(&self) -> usize {
        self.idx.live_len()
    }
}

/// Builds an index server-side (e.g. in a Worker's `/ingest`) by inserting embeddings, then serializes
/// it with `save()` for storage in R2/KV. The chat path later rehydrates it with `Rag`.
#[wasm_bindgen]
pub struct RagBuilder {
    idx: LiveIndex<String>,
}

#[wasm_bindgen]
impl RagBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> RagBuilder {
        RagBuilder { idx: LiveIndex::new(m, ef_construction, ef_search) }
    }

    /// Insert (or, with an already-seen id, revive) one document embedding.
    pub fn insert(&mut self, id: &str, embedding: &[f32]) {
        self.idx.step(vec![((id.to_string(), embedding.to_vec()), 1)]);
    }

    /// Retract a document by id (tombstone) — supports incremental corpus edits without a rebuild.
    pub fn remove(&mut self, id: &str, embedding: &[f32]) {
        self.idx.step(vec![((id.to_string(), embedding.to_vec()), -1)]);
    }

    /// Serialize the built graph to bytes for storage.
    pub fn save(&self) -> Vec<u8> {
        self.idx.save()
    }

    #[wasm_bindgen(js_name = liveLen)]
    pub fn live_len(&self) -> usize {
        self.idx.live_len()
    }
}
