//! Print the pre-tokenizer's SPLIT (not ids) for a string under every rule — for arguing about
//! boundaries with the actual output rather than a remembered example.
fn main() {
    let s = std::env::args().nth(1).unwrap_or_else(|| ";\n\r \r".into());
    let s = s.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t");
    println!("input {s:?}");
    for (n, p) in [("Gpt2", ferric_tokenizer::Pre::Gpt2), ("Qwen2", ferric_tokenizer::Pre::Qwen2),
                   ("Laguna", ferric_tokenizer::Pre::Laguna)] {
        println!("  {n:<7} {:?}", ferric_tokenizer::pretokenize_pub(&s, p));
    }
}
