//! **Precompute corpus embeddings with the SAME model the tab will query with.**
//!
//! The scale answer for in-tab retrieval is not batching embeds in the browser — corpus vectors are
//! a function of (corpus, model) and change only when one of them does, so they are computed ONCE,
//! here, and shipped as a file beside the corpus. The tab then pays one question-embed per query plus
//! N dot products, independent of corpus size.
//!
//! Output layout (little-endian): magic "FVEC", u32 version=1, u32 n_embd, u32 n_chunks,
//! u64 model_fingerprint (FNV-1a of the model's own embedding of the fixed probe string), then
//! n_chunks * n_embd f32 vectors in chunk order. The fingerprint is the receipt that the tab's model
//! and this file's model AGREE — mixed vectors rank garbage silently, cosine has no type error.
//!
//!   cargo run -p ferric-web --example embed_corpus --release -- <model.gguf> <corpus.txt> <out.fvec>
fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let (mp, cp, op) = (a.get(1).expect("model"), a.get(2).expect("corpus.txt"), a.get(3).expect("out.fvec"));
    let m = ferric_web::FerricModel::load(std::fs::read(mp).expect("read model")).await.expect("load");
    let text = std::fs::read_to_string(cp).expect("read corpus");
    let chunks: Vec<&str> = text.split("\n\n").map(|c| c.trim()).filter(|c| !c.is_empty()).collect();
    assert!(!chunks.is_empty(), "empty corpus");

    // Version 2: the header carries the PROBE VECTOR itself, not a bit-hash of it. The v1 hash
    // REFUTED itself on first contact with a real tab: native Metal and Chrome's Dawn select
    // different kernel variants by capability (subgroup reduce vs plain), so the same model's
    // embeddings differ at ulp order across fabrics BY CONSTRUCTION, and a bit-hash reads that as
    // "different model". Cosine against the shipped probe tolerates reduction-order noise while a
    // genuinely different model still fails it decisively.
    let probe = m.embed("ferric model fingerprint probe".into()).await.expect("probe embed");
    let n_embd = probe.len();

    let mut out: Vec<u8> = Vec::new();
    out.extend(b"FVEC");
    out.extend(2u32.to_le_bytes());
    out.extend((n_embd as u32).to_le_bytes());
    out.extend((chunks.len() as u32).to_le_bytes());
    for x in &probe { out.extend(x.to_le_bytes()); }
    for (i, c) in chunks.iter().enumerate() {
        let v = m.embed((*c).into()).await.expect("embed chunk");
        assert_eq!(v.len(), n_embd, "chunk {i} produced a different width");
        for x in &v { out.extend(x.to_le_bytes()); }
        if i % 25 == 0 { eprintln!("{i}/{} embedded", chunks.len()); }
    }
    std::fs::write(op, &out).expect("write");
    println!("wrote {} chunks x {n_embd} dims (v2, probe-vector header), {} bytes", chunks.len(), out.len());
}
