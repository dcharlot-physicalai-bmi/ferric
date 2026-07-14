//! Validates the GPT-2 + Digits pre-tokenizer against HF `tokenizers` on diverse text (digits,
//! punctuation, contractions, multi-space runs) — token-for-token parity, not just clean English.
use ferric_tokenizer::Bpe;

fn main() {
    let dir = format!("{}/.cache/ferric/smollm2-135m", std::env::var("HOME").unwrap());
    let bpe = Bpe::from_tokenizer_json(&std::fs::read(format!("{dir}/tokenizer.json")).unwrap()).unwrap();
    let cases: &[(&str, &[u32])] = &[
        ("The capital of France is", &[504, 3575, 282, 4649, 314]),
        ("Hello, world!", &[19556, 28, 905, 17]),
        ("I don't think it's 42.", &[57, 1326, 982, 1510, 357, 506, 216, 36, 34, 30]),
        ("x = 3.14 + 2", &[104, 446, 216, 35, 30, 33, 36, 1232, 216, 34]),
        ("  leading   spaces", &[216, 2899, 256, 5600]),
        ("Ferric runs on CPU, GPU & NPU.", &[40860, 631, 7313, 335, 17756, 28, 29593, 1456, 30017, 69, 30]),
    ];
    let mut ok = true;
    for (text, expect) in cases {
        let got = bpe.encode(text);
        let pass = got == *expect;
        ok &= pass;
        // round-trip too
        let rt = bpe.decode(&got);
        println!("  {} {:<32} {}", if pass { "✅" } else { "❌" }, format!("{text:?}"), if rt == *text { "" } else { "(decode≠)" });
        if !pass { println!("      got {got:?}\n      exp {expect:?}"); }
    }
    println!("{}", if ok { "✅ GPT-2 + Digits pre-tokenizer matches HF `tokenizers` token-for-token on arbitrary text" } else { "❌ tokenizer parity mismatch" });
    assert!(ok);
}
