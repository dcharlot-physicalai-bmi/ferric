//! **Reference-diff the tokenizer against llama.cpp.** Same model, same text, compare ids.
//!
//! An embedding diff cannot tell a kernel bug from a tokenizer bug: both produce a vector that is
//! wrong by the same measure. This isolates the second, because the fixes have nothing in common.
//!
//!   llama-tokenize -m M -p TEXT --ids
//!   cargo run -p ferric-web --example tokenize_reference --release -- M "TEXT" "[id,id,...]"
fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let (mp, text, refs) = (a.get(1).expect("model"), a.get(2).expect("text"), a.get(3).expect("[ids]"));
    let want: Vec<u32> = refs.trim().trim_start_matches('[').trim_end_matches(']')
        .split(',').filter_map(|s| s.trim().parse().ok()).collect();
    assert!(!want.is_empty(), "no reference ids parsed");
    let m = ferric_web::FerricModel::load(std::fs::read(mp).expect("read model")).await.expect("load");
    let got = m.encode_ids(text);
    println!("llama.cpp: {want:?}");
    println!("ferric   : {got:?}");
    if got == want { println!("  ✅ identical"); return; }
    // Name the FIRST divergence rather than only reporting inequality — the position is the whole
    // diagnostic, because a merge-order bug diverges mid-word while a special-token bug diverges at
    // the ends.
    let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
    println!("  ❌ diverges at index {at}: ferric {:?} vs llama.cpp {:?}",
             got.get(at), want.get(at));
    println!("  lengths: ferric {} vs llama.cpp {}", got.len(), want.len());
    std::process::exit(1);
}
