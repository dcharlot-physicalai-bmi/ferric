//! **Which `Pre` rule reproduces llama.cpp, over a multi-script corpus?**
//!
//! Reads `text|[id, id, ...]` lines produced by `llama-tokenize --ids` and reports, per `Pre`
//! variant, how many strings it reproduces EXACTLY. The reference is the outside implementation.
//!
//!   cargo run -p ferric-llama --example pretokenizer_conformance --release -- <model.gguf> <ref.txt>
use ferric_gguf::{GgufFile, Meta};
use ferric_tokenizer::{Bpe, Pre};
use std::collections::HashMap;

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: <model.gguf> <ref.txt>");
    let refp = a.next().expect("usage: <model.gguf> <ref.txt>");
    let g = GgufFile::open(&path).expect("open gguf");
    let get = |k: &str| g.metadata.get(k);
    let declared = match get("tokenizer.ggml.pre") { Some(Meta::Str(s)) => s.clone(), _ => "<none>".into() };

    let tokens: Vec<String> = match get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(x)) => x.iter().map(|v| if let Meta::Str(s) = v { s.clone() } else { String::new() }).collect(),
        _ => return,
    };
    let merges: Vec<(String, String)> = match get("tokenizer.ggml.merges") {
        Some(Meta::Arr(x)) => x.iter().filter_map(|v| if let Meta::Str(s) = v {
            s.split_once(' ').map(|(l, r)| (l.to_string(), r.to_string())) } else { None }).collect(),
        _ => Vec::new(),
    };
    let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();

    let txt = std::fs::read_to_string(&refp).expect("ref file");
    let cases: Vec<(String, Vec<u32>)> = txt.lines().filter_map(|l| {
        let (s, ids) = l.rsplit_once('|')?;
        let ids: Vec<u32> = ids.trim().trim_start_matches('[').trim_end_matches(']')
            .split(',').filter_map(|v| v.trim().parse().ok()).collect();
        if ids.is_empty() { None } else { Some((s.to_string(), ids)) }
    }).collect();

    let mapped = Pre::is_mapped(&declared);
    println!("model pre={declared:?}  ·  {}  ·  rule {:?}  ·  {} reference cases\n",
             if mapped { "IMPLEMENTED" } else { "FALL-OPEN (no rule for this value)" },
             Pre::from_gguf(Some(&declared)), cases.len());
    // ⛔ "(current)" MUST be computed, not written. The first version hardcoded it on the Gpt2 row,
    // so a sweep over 34 checkpoints reported every one as diverging — including files that map to
    // Qwen2 and score 19/20 — because the label said "current" on a row that was not. A gate whose
    // own annotation is a constant reports the same answer whatever the code does.
    let actual = Pre::from_gguf(Some(&declared));
    // ⛔ EVERY variant must be listed, or a model whose rule is missing here reports NO "(current)"
    // row and the sweep quietly files it as "not BPE". That happened the moment Pre::Qwen35 landed:
    // the three checkpoints the fix was FOR vanished from the results it was meant to prove.
    let variants: [(&str, Pre); 6] = [("Gpt2", Pre::Gpt2), ("Qwen2", Pre::Qwen2), ("Qwen35", Pre::Qwen35),
                                      ("Llama3", Pre::Llama3), ("Laguna", Pre::Laguna), ("Hyv4", Pre::Hyv4)];
    let mut best = ("", 0usize);
    for (name, p) in variants {
        let b = Bpe::new_with_pre(vocab.clone(), &merges, p);
        let mut ok = 0usize;
        let mut fails: Vec<&str> = Vec::new();
        for (s, want) in &cases {
            if &b.encode(s) == want { ok += 1 } else if fails.len() < 6 { fails.push(s) }
        }
        if ok > best.1 { best = (name, ok); }
        let tag = if p == actual { " (current)" } else { "" };
        println!("  {:<18} {ok:>2}/{} exact", format!("{name}{tag}"), cases.len());
        if !fails.is_empty() { println!("      misses: {}", fails.join(" · ")); }
    }
    // ⛔ NO "best available" RECOMMENDATION. An earlier version printed the highest-scoring rule, and
    // on a `laguna` checkpoint that was Qwen2 at 20/20. A conformance score ranks rules against the
    // corpus you happened to write; it is not evidence about which rule is correct.
    //
    // ⚠⚠ CORRECTION TO WHAT THIS COMMENT USED TO SAY. It claimed laguna's regex is `[^\n]+|[\n]+`,
    // "a NEWLINE SPLITTER with nothing in common with Qwen2". BOTH HALVES WERE WRONG, and the error
    // was in how I read the source: `regex_exprs` for LAGUNA has TWO entries (llama-vocab.cpp:505-508)
    // and I extracted it with a first-match grep, which returns one entry of a list and looks
    // complete. The second entry is BYTE-IDENTICAL to Qwen2's, and llama.cpp applies the list
    // SEQUENTIALLY — so laguna IS Qwen2, run inside each run of '\n' and each run of non-'\n'.
    //
    // The conclusion survives, for a better reason: Qwen2 scored 20/20 not by coincidence but because
    // laguna differs from it ONLY where a pre-token would span a newline — and `-p` takes one line,
    // so the corpus could not contain one. ⭐ The blind spot was in the HARNESS, not the scoring.
    if !mapped {
        println!("\n  ⛔ this checkpoint's pre-tokenizer is NOT implemented — the score above is the");
        println!("     GPT-2 fallback's, and the fix is to port the rule llama.cpp names for {declared:?},");
        println!("     not to adopt whichever existing rule scores highest here.");
    }
    // a listed variant must have matched `actual`, or the table above is not describing this model
    assert!(variants.iter().any(|(_, p)| *p == actual) || !mapped,
            "Pre::{actual:?} is mapped but absent from this probe's variant list — the sweep would \
             report this checkpoint as un-evaluated rather than as passing or failing");
    let _ = best;
}
