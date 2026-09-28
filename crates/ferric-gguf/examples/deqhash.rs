//! **Every quantized tensor of a GGUF, dequantized and hashed** — the Ferric half of a whole-file
//! decoder check against a reference that prints the same line format (`deqhash.cpp`, built against
//! PrismML's fork and calling ITS `ggml_get_type_traits(type)->to_float`).
//!
//! One line per tensor: `name n fnv1a64`, the hash over the float32 BIT PATTERNS row by row, so two
//! decoders agree only if every value is bit-identical — not merely close.
//!
//!   cargo run --release -p ferric-gguf --example deqhash -- <model.gguf>
//!
//! ⚠ Calls `unlock_prism_hadamard`: this program DECODES rotated-basis weights, it never runs them, so
//! it is one of the two legitimate callers (the other applies the transform). See `ferric_gguf::prism`.
use ferric_gguf::{deq_raw, type_size, GgufFile, GgufSource};

fn main() {
    let path = std::env::args().nth(1).expect("usage: deqhash <model.gguf>");
    let g = GgufFile::open(&path).expect("open");
    g.unlock_prism_hadamard();
    // Debug aids: DEQHASH_ONLY=<tensor> and DEQHASH_ROWS=<n> narrow the run.
    let only = std::env::var("DEQHASH_ONLY").ok();
    let max_rows: usize = std::env::var("DEQHASH_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    for t in &g.tensors {
        if matches!(t.ggml_type, 0 | 1 | 30) { continue; } // F32 / F16 / BF16: not quantized
        if only.as_deref().is_some_and(|o| o != t.name) { continue; }
        let ne0 = t.dims[0] as usize;
        let n: usize = t.dims.iter().product::<u64>() as usize;
        let rb = type_size(t.ggml_type, ne0).expect("row size");
        let mut h: u64 = 0xcbf29ce484222325;
        let mut raw = vec![0u8; rb];
        for r in 0..(n / ne0).min(max_rows) {
            g.raw_range(&t.name, (r * rb) as u64, &mut raw).expect("read row");
            for v in deq_raw(&raw, ne0, t.ggml_type).expect("dequant") {
                for b in v.to_bits().to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
            }
        }
        println!("{} {n} {h:016x}", t.name);
    }
}
