//! Validate the byte-level BPE tokenizer against a HF-`tokenizers` reference set (`tok_tests.json`):
//! every string must encode to the exact ids the reference produced. This covers the pre-tokenizer
//! edge cases the single end-to-end llama.cpp prompt never exercises — contractions ("don't", "it's"),
//! the individual-digit split ("3.14" → 3·.·1·4), multi-space runs, and punctuation runs.
//!
//! Run: `cargo run -p ferric-tokenizer --example verify_tok --release`
//! (reads ~/.cache/ferric/smollm2-135m/{tokenizer.json,tok_tests.json})
use ferric_tokenizer::Bpe;
use std::collections::BTreeMap;

fn main() {
    let dir = format!("{}/.cache/ferric/smollm2-135m", std::env::var("HOME").unwrap());
    let tj = match std::fs::read(format!("{dir}/tokenizer.json")) {
        Ok(b) => b,
        Err(_) => { println!("⏭  tokenizer.json not cached at {dir} — skipping"); return; }
    };
    let bpe = Bpe::from_tokenizer_json(&tj).expect("load tokenizer.json");
    let tests: BTreeMap<String, Vec<u32>> =
        serde_json::from_slice(&std::fs::read(format!("{dir}/tok_tests.json")).expect("tok_tests.json"))
            .expect("parse tok_tests.json");

    let (mut ok, mut fail) = (0usize, 0usize);
    for (text, expected) in &tests {
        let got = bpe.encode(text);
        if &got == expected {
            ok += 1;
            println!("✅ {text:?}");
        } else {
            fail += 1;
            println!("❌ {text:?}\n   expected {expected:?}\n   got      {got:?}");
        }
    }
    println!("\n{ok}/{} strings encode identically to the HF-tokenizers reference", ok + fail);
    assert_eq!(fail, 0, "tokenizer diverges from the reference on {fail} case(s)");
}
