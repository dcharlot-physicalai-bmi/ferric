//! Validates the byte-level BPE tokenizer: encode/decode is lossless for arbitrary text (ASCII,
//! accented, emoji — i.e. multi-byte UTF-8), merges reduce token count, and the GPT-2 vocab.json +
//! merges.txt loader parses. Byte-level BPE guarantees decode(encode(x)) == x for any input.
use ferric_tokenizer::{base_byte_vocab, Bpe};

fn main() {
    // base 256 byte-symbols + a few merges
    let mut vocab = base_byte_vocab();
    let mut next = vocab.len() as u32;
    let raw = [("l", "l"), ("l", "o"), ("h", "e"), ("o", "r"), ("w", "o"), ("e", "l")];
    let mut merges = Vec::new();
    for (a, b) in raw {
        let tok = format!("{a}{b}");
        vocab.entry(tok).or_insert_with(|| { let id = next; next += 1; id });
        merges.push((a.to_string(), b.to_string()));
    }
    let bpe = Bpe::new(vocab, &merges);
    println!("Ferric tokenizer · byte-level BPE · vocab {}", bpe.vocab_size());

    let mut ok = true;
    for s in ["hello hello world", "lll ooo rrr", "Ferric: SOTA?", "café — piñata", "run it 🚀 in the browser 🦀", "", "\t\n mixed  spaces "] {
        let ids = bpe.encode(s);
        let back = bpe.decode(&ids);
        let lossless = back == s;
        ok &= lossless;
        println!("  {} {:>3} bytes → {:>3} tokens   {:?}", if lossless { "✅" } else { "❌" }, s.len(), ids.len(), if s.len() > 24 { format!("{}…", &s[..24.min(s.len())]) } else { s.to_string() });
    }
    // merges must actually reduce token count vs raw bytes
    let reduced = bpe.encode("hello hello").len() < "hello hello".len();
    ok &= reduced;
    println!("  {} merges reduce token count ({} tokens for 11 bytes)", if reduced { "✅" } else { "❌" }, bpe.encode("hello hello").len());

    // GPT-2 vocab.json + merges.txt loader
    let vj = r#"{"a":0,"b":1,"c":2,"ab":3}"#;
    let mt = "#version: 0.2\na b\n";
    let g = Bpe::from_gpt2(vj, mt).unwrap();
    let loader_ok = g.vocab_size() == 4;
    ok &= loader_ok;
    println!("  {} GPT-2 vocab.json + merges.txt loader (vocab {})", if loader_ok { "✅" } else { "❌" }, g.vocab_size());

    println!("{}", if ok { "✅ Byte-level BPE tokenizer: lossless round-trip on all inputs + loads GPT-2 format" } else { "❌ tokenizer failed" });
    assert!(ok);
}
