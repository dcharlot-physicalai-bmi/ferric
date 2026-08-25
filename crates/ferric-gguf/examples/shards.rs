//! **Does a sharded GGUF load as one model?** — against the merged original, tensor for tensor.
//!
//!   llama-gguf-split --split --split-max-tensors N model.gguf out
//!   cargo run -p ferric-gguf --example shards --release -- out-00001-of-0000N.gguf model.gguf
use ferric_gguf::GgufFile;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let sharded = a.get(1).expect("usage: shards <any-shard.gguf> [original.gguf]");
    let g = GgufFile::open(sharded).expect("open sharded");
    println!("{sharded}");
    println!("  shards {}  tensors {}", g.shard_count(), g.tensors.len());

    let Some(orig) = a.get(2) else { return };
    let o = GgufFile::open(orig).expect("open original");
    println!("{orig}\n  shards {}  tensors {}", o.shard_count(), o.tensors.len());
    assert_eq!(g.tensors.len(), o.tensors.len(), "the shard set must expose the same tensor table");

    // Byte-for-byte on EVERY tensor, not a sample. A sharded read that picks the wrong file returns
    // the right NUMBER of bytes from the wrong place, so a length check proves nothing and a
    // spot-check on early tensors passes while the last shard is silently wrong.
    let (mut checked, mut bytes) = (0usize, 0usize);
    for t in &o.tensors {
        let want = o.raw(&t.name).expect("raw original");
        let got = g.raw(&t.name).unwrap_or_else(|e| panic!("sharded raw {}: {e}", t.name));
        assert_eq!(got.len(), want.len(), "{}: size differs", t.name);
        assert!(got == want, "{}: BYTES DIFFER — the shard read came from the wrong file or offset", t.name);
        checked += 1;
        bytes += want.len();
    }
    println!("  ✅ {checked} tensors byte-identical to the single-file model ({:.0} MB compared)", bytes as f64 / 1e6);

    // Metadata must come from shard 0 whichever part was opened, or the tokenizer vanishes.
    for k in ["general.architecture", "tokenizer.ggml.tokens", "tokenizer.ggml.model"] {
        assert!(g.metadata.contains_key(k), "sharded load lost metadata key {k} — shard 0 carries it");
    }
    println!("  ✅ metadata present (architecture, tokenizer) regardless of which part was opened");
}
