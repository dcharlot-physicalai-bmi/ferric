//! Live vector index — an incrementally-maintained HNSW, as a `ferric-flow` operator.
//!
//! This is the retrieval half of the RAG layer. Embeddings arrive as deltas (a document is added,
//! removed, or corrected) and the approximate-nearest-neighbour index is kept current *without a
//! rebuild*. A query returns the current top-k by cosine similarity over exactly the live set.
//!
//! HNSW (Malkov & Yashunin) is the standard: a multi-layer navigable-small-world graph, logarithmic
//! search. It is a public algorithm — the point of this crate is to own a pure-Rust, `wasm`-able
//! implementation inside Ferric rather than rent one behind a license. Deletion here is a tombstone:
//! a retracted node stays as a routing waypoint but is never returned, so query results always
//! reflect the live set. (Graph compaction is a later concern; correctness of results is not.)

use super::Diff;
use std::cell::Cell;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::Hash;

fn normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}

// ---- little-endian binary helpers for a zero-dependency serialization format ----
fn pu32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn pf32(b: &mut Vec<u8>, v: f32) {
    b.extend_from_slice(&v.to_le_bytes());
}
struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Rd<'a> {
    fn new(b: &'a [u8]) -> Self {
        Rd { b, p: 0 }
    }
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn f32(&mut self) -> f32 {
        let v = f32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn i64(&mut self) -> i64 {
        let v = i64::from_le_bytes(self.b[self.p..self.p + 8].try_into().unwrap());
        self.p += 8;
        v
    }
    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        s
    }
}
const HNSW_MAGIC: u32 = 0x46_4C_57_31; // "FLW1"

/// A (distance, node) pair ordered by distance via `f32::total_cmp` (so it is a real total order).
#[derive(Clone, Copy)]
struct Cand {
    d: f32,
    id: usize,
}
impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.d.total_cmp(&o.d) == Ordering::Equal
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.d.total_cmp(&o.d)
    }
}

/// A pure-Rust HNSW over cosine distance. Internal ids are dense `usize`s assigned on insertion.
pub struct Hnsw {
    m: usize,
    m0: usize,
    ef_construction: usize,
    ef_search: usize,
    ml: f32,
    vecs: Vec<Vec<f32>>,
    levels: Vec<usize>,
    links: Vec<Vec<Vec<usize>>>, // [node][layer] -> neighbours
    tomb: Vec<bool>,
    entry: Option<usize>,
    max_layer: usize,
    rng: u64,
    /// Cosine-distance evaluations performed — the "sub-linear vs brute force" meter.
    pub dist_count: Cell<u64>,
}

impl Hnsw {
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            m,
            m0: m * 2,
            ef_construction,
            ef_search,
            ml: 1.0 / (m as f32).ln(),
            vecs: Vec::new(),
            levels: Vec::new(),
            links: Vec::new(),
            tomb: Vec::new(),
            entry: None,
            max_layer: 0,
            rng: 0x1234_5678_9abc_def0,
            dist_count: Cell::new(0),
        }
    }

    fn rand(&mut self) -> f32 {
        self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.rng >> 11) as f32) / ((1u64 << 53) as f32)
    }

    fn sample_level(&mut self) -> usize {
        let r = self.rand().max(1e-12);
        (-(r.ln()) * self.ml) as usize
    }

    fn dist(&self, q: &[f32], id: usize) -> f32 {
        self.dist_count.set(self.dist_count.get() + 1);
        let v = &self.vecs[id];
        1.0 - q.iter().zip(v).map(|(a, b)| a * b).sum::<f32>()
    }

    /// Greedy best-first search on one layer; returns up to `ef` nearest nodes (by distance).
    fn search_layer(&self, q: &[f32], entries: &[usize], ef: usize, layer: usize) -> Vec<Cand> {
        let mut visited: HashSet<usize> = HashSet::new();
        let mut cand: BinaryHeap<Reverse<Cand>> = BinaryHeap::new(); // min-heap: nearest first
        let mut res: BinaryHeap<Cand> = BinaryHeap::new(); // max-heap: farthest on top
        for &e in entries {
            if visited.insert(e) {
                let d = self.dist(q, e);
                cand.push(Reverse(Cand { d, id: e }));
                res.push(Cand { d, id: e });
                if res.len() > ef {
                    res.pop();
                }
            }
        }
        while let Some(Reverse(c)) = cand.pop() {
            let farthest = res.peek().map(|t| t.d).unwrap_or(f32::INFINITY);
            if c.d > farthest && res.len() >= ef {
                break;
            }
            for &nb in &self.links[c.id][layer] {
                if visited.insert(nb) {
                    let d = self.dist(q, nb);
                    let farthest = res.peek().map(|t| t.d).unwrap_or(f32::INFINITY);
                    if d < farthest || res.len() < ef {
                        cand.push(Reverse(Cand { d, id: nb }));
                        res.push(Cand { d, id: nb });
                        if res.len() > ef {
                            res.pop();
                        }
                    }
                }
            }
        }
        let mut out: Vec<Cand> = res.into_vec();
        out.sort(); // nearest first
        out
    }

    /// HNSW neighbour-selection heuristic (paper Alg. 4): from candidates sorted nearest-first, keep
    /// one only if it is closer to `base` than to every neighbour already kept. This spreads links out
    /// so the graph stays navigable instead of collapsing into hubs — the difference between ~0.85 and
    /// ~0.97 recall. Falls back to filling with the nearest remaining if the heuristic under-fills.
    fn select_heuristic(&self, cands: &[Cand], m: usize) -> Vec<usize> {
        let mut kept: Vec<usize> = Vec::new();
        for c in cands {
            if kept.len() >= m {
                break;
            }
            let cv = self.vecs[c.id].clone();
            let covered = kept.iter().any(|&r| self.dist(&cv, r) < c.d);
            if !covered {
                kept.push(c.id);
            }
        }
        if kept.len() < m {
            for c in cands {
                if kept.len() >= m {
                    break;
                }
                if !kept.contains(&c.id) {
                    kept.push(c.id);
                }
            }
        }
        kept
    }

    /// Re-select node `of`'s neighbours at `layer` with the heuristic when the list overflows.
    fn prune(&mut self, of: usize, layer: usize) {
        let m = if layer == 0 { self.m0 } else { self.m };
        if self.links[of][layer].len() <= m {
            return;
        }
        let base = self.vecs[of].clone();
        let mut scored: Vec<Cand> =
            self.links[of][layer].iter().map(|&nb| Cand { d: self.dist(&base, nb), id: nb }).collect();
        scored.sort();
        self.links[of][layer] = self.select_heuristic(&scored, m);
    }

    /// Insert a vector; returns its internal id.
    pub fn insert(&mut self, vector: &[f32]) -> usize {
        let q = normalize(vector);
        let id = self.vecs.len();
        let level = self.sample_level();
        self.vecs.push(q.clone());
        self.levels.push(level);
        self.links.push(vec![Vec::new(); level + 1]);
        self.tomb.push(false);

        let entry = match self.entry {
            None => {
                self.entry = Some(id);
                self.max_layer = level;
                return id;
            }
            Some(e) => e,
        };

        // descend from the top down to level+1 with a greedy ef=1 walk
        let mut ep = entry;
        let mut l = self.max_layer;
        while l > level {
            let r = self.search_layer(&q, &[ep], 1, l);
            if let Some(best) = r.first() {
                ep = best.id;
            }
            l -= 1;
        }

        // connect from min(level, max_layer) down to 0
        let mut eps = vec![ep];
        let top = level.min(self.max_layer);
        for l in (0..=top).rev() {
            let found = self.search_layer(&q, &eps, self.ef_construction, l);
            let m = if l == 0 { self.m0 } else { self.m };
            let selected: Vec<usize> = self.select_heuristic(&found, m);
            for &nb in &selected {
                self.links[id][l].push(nb);
                self.links[nb][l].push(id);
                self.prune(nb, l);
            }
            self.prune(id, l);
            eps = if found.is_empty() { eps } else { found.iter().map(|c| c.id).collect() };
        }

        if level > self.max_layer {
            self.max_layer = level;
            self.entry = Some(id);
        }
        id
    }

    pub fn tombstone(&mut self, id: usize) {
        if id < self.tomb.len() {
            self.tomb[id] = true;
        }
    }
    pub fn revive(&mut self, id: usize) {
        if id < self.tomb.len() {
            self.tomb[id] = false;
        }
    }

    /// Approximate k nearest *live* nodes to `query`, nearest first: `(internal_id, distance)`.
    pub fn query(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let entry = match self.entry {
            None => return Vec::new(),
            Some(e) => e,
        };
        let q = normalize(query);
        let mut ep = entry;
        let mut l = self.max_layer;
        while l > 0 {
            let r = self.search_layer(&q, &[ep], 1, l);
            if let Some(best) = r.first() {
                ep = best.id;
            }
            l -= 1;
        }
        let ef = self.ef_search.max(k * 2);
        let found = self.search_layer(&q, &[ep], ef, 0);
        found
            .into_iter()
            .filter(|c| !self.tomb[c.id])
            .take(k)
            .map(|c| (c.id, c.d))
            .collect()
    }

    pub fn live_len(&self) -> usize {
        self.tomb.iter().filter(|t| !**t).count()
    }

    /// Serialize the built graph to a compact byte buffer. This is the whole point of a live index on
    /// a stateless edge Worker: build the HNSW once (offline or on ingest), ship these bytes, and
    /// rehydrate + query per isolate instead of rebuilding the graph on every cold start.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        pu32(&mut b, HNSW_MAGIC);
        pu32(&mut b, self.m as u32);
        pu32(&mut b, self.m0 as u32);
        pu32(&mut b, self.ef_construction as u32);
        pu32(&mut b, self.ef_search as u32);
        pf32(&mut b, self.ml);
        let dim = self.vecs.first().map(|v| v.len()).unwrap_or(0);
        pu32(&mut b, self.vecs.len() as u32);
        pu32(&mut b, dim as u32);
        pu32(&mut b, self.max_layer as u32);
        pu32(&mut b, self.entry.map(|e| e as u32).unwrap_or(u32::MAX));
        for i in 0..self.vecs.len() {
            pu32(&mut b, self.levels[i] as u32);
            b.push(self.tomb[i] as u8);
            for &x in &self.vecs[i] {
                pf32(&mut b, x);
            }
            for layer in &self.links[i] {
                pu32(&mut b, layer.len() as u32);
                for &nb in layer {
                    pu32(&mut b, nb as u32);
                }
            }
        }
        b
    }

    /// Rehydrate a graph from `to_bytes`. Queries on the result are identical to the original's.
    pub fn from_bytes(b: &[u8]) -> Self {
        let mut r = Rd::new(b);
        assert_eq!(r.u32(), HNSW_MAGIC, "bad ferric-flow index header");
        let m = r.u32() as usize;
        let m0 = r.u32() as usize;
        let ef_construction = r.u32() as usize;
        let ef_search = r.u32() as usize;
        let ml = r.f32();
        let n = r.u32() as usize;
        let dim = r.u32() as usize;
        let max_layer = r.u32() as usize;
        let entry_raw = r.u32();
        let (mut vecs, mut levels, mut tomb, mut links) =
            (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            let level = r.u32() as usize;
            tomb.push(r.take(1)[0] != 0);
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(r.f32());
            }
            vecs.push(v);
            let mut node_links = Vec::with_capacity(level + 1);
            for _ in 0..=level {
                let len = r.u32() as usize;
                let mut l = Vec::with_capacity(len);
                for _ in 0..len {
                    l.push(r.u32() as usize);
                }
                node_links.push(l);
            }
            links.push(node_links);
            levels.push(level);
        }
        Hnsw {
            m,
            m0,
            ef_construction,
            ef_search,
            ml,
            vecs,
            levels,
            links,
            tomb,
            entry: if entry_raw == u32::MAX { None } else { Some(entry_raw as usize) },
            max_layer,
            rng: 0x1234_5678_9abc_def0,
            dist_count: Cell::new(0),
        }
    }
}

/// The `ferric-flow` operator: embeddings keyed by an external id flow in as deltas; the HNSW is kept
/// current and queries read the live set. Assertions insert (or revive), retractions to zero
/// multiplicity tombstone — so a corrected or deleted document drops out of retrieval immediately.
pub struct LiveIndex<Id: Eq + Hash + Clone> {
    hnsw: Hnsw,
    internal: HashMap<Id, usize>,
    id_of: Vec<Id>,
    mult: HashMap<Id, Diff>,
}

impl<Id: Eq + Hash + Clone> LiveIndex<Id> {
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            hnsw: Hnsw::new(m, ef_construction, ef_search),
            internal: HashMap::new(),
            id_of: Vec::new(),
            mult: HashMap::new(),
        }
    }

    /// Apply a batch of `(id, embedding)` deltas. `diff > 0` asserts, `diff < 0` retracts.
    pub fn step(&mut self, deltas: Vec<((Id, Vec<f32>), Diff)>) {
        for ((id, vec), diff) in deltas {
            let before = *self.mult.get(&id).unwrap_or(&0);
            let after = before + diff;
            self.mult.insert(id.clone(), after);
            let was_live = before > 0;
            let is_live = after > 0;
            if is_live && !was_live {
                match self.internal.get(&id) {
                    Some(&iid) => self.hnsw.revive(iid), // seen before → un-tombstone
                    None => {
                        let iid = self.hnsw.insert(&vec);
                        self.internal.insert(id.clone(), iid);
                        debug_assert_eq!(iid, self.id_of.len());
                        self.id_of.push(id.clone());
                    }
                }
            } else if !is_live && was_live {
                if let Some(&iid) = self.internal.get(&id) {
                    self.hnsw.tombstone(iid);
                }
            }
        }
    }

    /// Current top-k live documents for a query embedding: `(id, cosine_distance)`, nearest first.
    pub fn query(&self, query: &[f32], k: usize) -> Vec<(Id, f32)> {
        self.hnsw
            .query(query, k)
            .into_iter()
            .map(|(iid, d)| (self.id_of[iid].clone(), d))
            .collect()
    }

    pub fn live_len(&self) -> usize {
        self.hnsw.live_len()
    }
    pub fn dist_count(&self) -> u64 {
        self.hnsw.dist_count.get()
    }
}

impl LiveIndex<String> {
    /// Serialize a String-keyed index (built graph + id map + live state) to bytes — build once,
    /// ship, rehydrate + query on a stateless edge Worker. See `Hnsw::to_bytes` for the why.
    pub fn save(&self) -> Vec<u8> {
        let mut b = self.hnsw.to_bytes();
        pu32(&mut b, self.id_of.len() as u32);
        for id in &self.id_of {
            pu32(&mut b, id.len() as u32);
            b.extend_from_slice(id.as_bytes());
        }
        pu32(&mut b, self.mult.len() as u32);
        for (id, &m) in &self.mult {
            pu32(&mut b, id.len() as u32);
            b.extend_from_slice(id.as_bytes());
            b.extend_from_slice(&m.to_le_bytes());
        }
        b
    }

    /// Rehydrate a String-keyed index from `save`. Queries are identical to the original's.
    pub fn load(bytes: &[u8]) -> Self {
        let hnsw = Hnsw::from_bytes(bytes);
        // continue reading right after the hnsw section
        let mut r = Rd::new(bytes);
        r.p = hnsw_byte_len(&hnsw);
        let n = r.u32() as usize;
        let mut id_of = Vec::with_capacity(n);
        let mut internal = HashMap::with_capacity(n);
        for i in 0..n {
            let len = r.u32() as usize;
            let id = String::from_utf8(r.take(len).to_vec()).unwrap();
            internal.insert(id.clone(), i);
            id_of.push(id);
        }
        let mc = r.u32() as usize;
        let mut mult = HashMap::with_capacity(mc);
        for _ in 0..mc {
            let len = r.u32() as usize;
            let id = String::from_utf8(r.take(len).to_vec()).unwrap();
            mult.insert(id, r.i64());
        }
        Self { hnsw, internal, id_of, mult }
    }
}

/// Byte length of the `Hnsw::to_bytes` section, so `LiveIndex::load` can resume reading after it.
fn hnsw_byte_len(h: &Hnsw) -> usize {
    let dim = h.vecs.first().map(|v| v.len()).unwrap_or(0);
    let mut n = 4 * 6 + 4 * 4; // magic,m,m0,efc,efs + ml + count,dim,max_layer,entry
    for i in 0..h.vecs.len() {
        n += 4 + 1 + dim * 4; // level + tomb + vec
        for layer in &h.links[i] {
            n += 4 + layer.len() * 4;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f32) / ((1u64 << 53) as f32)
        }
        fn range(&mut self, n: usize) -> usize {
            (self.f() * n as f32) as usize % n
        }
    }

    fn rand_vec(rng: &mut Lcg, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| rng.f() * 2.0 - 1.0).collect()
    }

    // Real document/query embeddings are clustered on a manifold, not uniform noise. Draw from a set
    // of cluster centres with small per-doc jitter — the regime HNSW is actually built for.
    fn clustered_vec(rng: &mut Lcg, centers: &[Vec<f32>], jitter: f32) -> Vec<f32> {
        let c = &centers[rng.range(centers.len())];
        c.iter().map(|x| x + (rng.f() * 2.0 - 1.0) * jitter).collect()
    }

    fn cos_dist(a: &[f32], b: &[f32]) -> f32 {
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        1.0 - a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() / (na * nb)
    }

    fn brute_topk(corpus: &[(u32, Vec<f32>)], q: &[f32], k: usize) -> Vec<u32> {
        let mut scored: Vec<(f32, u32)> =
            corpus.iter().map(|(id, v)| (cos_dist(q, v), *id)).collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        scored.into_iter().take(k).map(|(_, id)| id).collect()
    }

    #[test]
    fn incremental_index_matches_brute_force_and_is_sublinear() {
        let dim = 48;
        let n = 800;
        let k = 10;
        let mut rng = Lcg(0xBEEF);

        // clustered corpus — 24 topic centres with small jitter, like real embeddings
        let centers: Vec<Vec<f32>> = (0..24).map(|_| rand_vec(&mut rng, dim)).collect();

        let mut idx: LiveIndex<u32> = LiveIndex::new(16, 100, 64);
        let mut live: Vec<(u32, Vec<f32>)> = Vec::new();

        // stream inserts one at a time (incremental, no rebuild)
        for id in 0..n as u32 {
            let v = clustered_vec(&mut rng, &centers, 0.15);
            idx.step(vec![((id, v.clone()), 1)]);
            live.push((id, v));
        }
        // retract a random ~15% — corrections/deletions
        let mut retracted = std::collections::HashSet::new();
        for _ in 0..(n * 15 / 100) {
            let pos = rng.range(live.len());
            let (id, v) = live[pos].clone();
            if retracted.insert(id) {
                idx.step(vec![((id, v), -1)]);
            }
        }
        live.retain(|(id, _)| !retracted.contains(id));
        assert_eq!(idx.live_len(), live.len(), "live count drifted from ground truth");

        // measure recall@k vs brute force over the live set, and count distance evals
        let queries = 120;
        let mut recall_sum = 0.0f64;
        let before = idx.dist_count();
        for _ in 0..queries {
            let q = clustered_vec(&mut rng, &centers, 0.15);
            let got = idx.query(&q, k);
            // no retracted id may ever appear
            for (id, _) in &got {
                assert!(!retracted.contains(id), "retracted doc {} returned by the index", id);
            }
            let truth = brute_topk(&live, &q, k);
            let got_set: std::collections::HashSet<u32> = got.iter().map(|(id, _)| *id).collect();
            let hit = truth.iter().filter(|id| got_set.contains(id)).count();
            recall_sum += hit as f64 / k as f64;
        }
        let recall = recall_sum / queries as f64;
        let index_dist_per_query = (idx.dist_count() - before) as f64 / queries as f64;
        let brute_dist_per_query = live.len() as f64;

        eprintln!(
            "ferric-flow index: recall@{} = {:.3}, {:.0} dist/query vs {:.0} brute ({:.1}x fewer), live = {}",
            k, recall, index_dist_per_query, brute_dist_per_query,
            brute_dist_per_query / index_dist_per_query, live.len()
        );
        assert!(recall >= 0.90, "recall@{} = {:.3} below 0.90", k, recall);
        assert!(index_dist_per_query < brute_dist_per_query * 0.6, "index not sub-linear enough");
    }

    #[test]
    fn retract_then_reassert_returns_to_results() {
        let mut idx: LiveIndex<&'static str> = LiveIndex::new(8, 40, 24);
        let a = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        idx.step(vec![(("doc-a", a.clone()), 1)]);
        idx.step(vec![(("doc-b", {
            let mut v = a.clone();
            v[1] = 0.2;
            v
        }), 1)]);
        assert!(idx.query(&a, 5).iter().any(|(id, _)| *id == "doc-a"));
        idx.step(vec![(("doc-a", a.clone()), -1)]); // retract
        assert!(!idx.query(&a, 5).iter().any(|(id, _)| *id == "doc-a"), "retracted doc still returned");
        idx.step(vec![(("doc-a", a.clone()), 1)]); // re-assert
        assert!(idx.query(&a, 5).iter().any(|(id, _)| *id == "doc-a"), "re-asserted doc not returned");
    }

    #[test]
    fn serialized_index_rehydrates_to_identical_queries() {
        // Build a graph, serialize it, rehydrate a fresh index from the bytes, and confirm every query
        // returns bit-identical results — the property that lets a stateless Worker skip rebuilding.
        let dim = 48;
        let mut rng = Lcg(0x5E21A1);
        let centers: Vec<Vec<f32>> = (0..16).map(|_| rand_vec(&mut rng, dim)).collect();
        let mut idx: LiveIndex<String> = LiveIndex::new(16, 100, 48);
        for id in 0..400u32 {
            idx.step(vec![((format!("doc-{id}"), clustered_vec(&mut rng, &centers, 0.15)), 1)]);
        }
        // retract some, so tombstone state must survive the round-trip too
        for id in (0..400u32).step_by(9) {
            idx.step(vec![((format!("doc-{id}"), vec![0.0; dim]), -1)]);
        }

        let bytes = idx.save();
        let rehydrated = LiveIndex::<String>::load(&bytes);
        assert_eq!(rehydrated.live_len(), idx.live_len(), "live count changed across serialization");

        for _ in 0..80 {
            let q = clustered_vec(&mut rng, &centers, 0.15);
            let a = idx.query(&q, 10);
            let b = rehydrated.query(&q, 10);
            assert_eq!(a, b, "rehydrated query differs from original");
        }
        eprintln!(
            "ferric-flow index: serialized 400-doc graph = {} bytes, rehydrated queries bit-identical",
            bytes.len()
        );
    }
}
