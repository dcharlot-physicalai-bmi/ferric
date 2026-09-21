//! Exhaustive search: is Laguna (a newline PRE-pass over Qwen2) ever different from running Qwen2
//! and splitting its output at newlines afterwards? If never, the distinction is untestable HERE and
//! should not be asserted.
fn main() {
    let alpha = [' ', '\n', '\r', '\t', 'a', '1', ';'];
    let post = |s: &str| -> Vec<String> {
        ferric_tokenizer::pretokenize_pub(s, ferric_tokenizer::Pre::Qwen2).iter().flat_map(|p| {
            let (mut v, mut cur, mut prev) = (Vec::new(), String::new(), None::<bool>);
            for c in p.chars() {
                let nl = c == '\n';
                if prev.is_some_and(|q| q != nl) { v.push(std::mem::take(&mut cur)); }
                cur.push(c); prev = Some(nl);
            }
            if !cur.is_empty() { v.push(cur) }
            v
        }).collect()
    };
    let (mut n, mut diff, mut shown) = (0usize, 0usize, 0usize);
    for len in 1..=5usize {
        let mut idx = vec![0usize; len];
        loop {
            let s: String = idx.iter().map(|&i| alpha[i]).collect();
            n += 1;
            let lag = ferric_tokenizer::pretokenize_pub(&s, ferric_tokenizer::Pre::Laguna);
            if lag != post(&s) {
                diff += 1;
                if shown < 5 { println!("  DIFFERS {s:?}\n    laguna    {lag:?}\n    qwen2+cut {:?}", post(&s)); shown += 1; }
            }
            let mut k = len;
            loop {
                if k == 0 { break }
                k -= 1;
                idx[k] += 1;
                if idx[k] < alpha.len() { break }
                idx[k] = 0;
                if k == 0 { break }
            }
            if idx.iter().all(|&i| i == 0) { break }
        }
    }
    println!("\n{n} strings over {:?}, lengths 1-5: {diff} differ", alpha);
    if diff == 0 {
        println!("=> for FERRIC's Qwen2, the pre-pass and the post-pass agree on this alphabet.");
        println!("   The distinction is real in llama.cpp's terms but is NOT demonstrable here,");
        println!("   so it must not be asserted as a Ferric-level test.");
    }
}
