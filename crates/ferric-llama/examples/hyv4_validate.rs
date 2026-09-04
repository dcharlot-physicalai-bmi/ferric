//! **Check this loader's expectations against Tencent's actual published checkpoint.**
//!
//! The synthetic test proves the wiring composes, and it cannot prove fidelity: the conventions
//! that write the file are the ones that read it back, so a transposed convention applied twice
//! cancels. This closes the structural half of that gap against the real thing.
//!
//! A GGUF header is a few megabytes of a 213.66 GiB file and HuggingFace serves range requests, so
//! the real metadata and the real tensor table are reachable from a laptop even though the weights
//! are not. Every tensor `Hyv4::load` will ask for, with the dims it expects, is checked against
//! what Tencent actually shipped — name for name, dimension for dimension.
//!
//! ⚠ What this establishes and what it does not. It establishes that the loader's view of the
//! FORMAT is right: every name resolves, every shape matches, every KV parses, nothing is missing
//! and nothing unexpected is present. It does NOT establish that the arithmetic is right — that
//! still needs the weights. Structure and semantics are different claims and this file makes the
//! first one, against the real artifact rather than against a file we wrote.
//!
//! ```text
//! # 24 MB is comfortably more than the ~5 MB header of either published file
//! B=https://huggingface.co/AngelSlim/Hy4-preview-GGUF/resolve/main
//! curl -L -r 0-25165823 -o hy4.head "$B/Hy4-preview-STQ1_0.gguf"
//! cargo run --release -p ferric-llama --example hyv4_validate -- hy4.head
//! ```

use ferric_llama::dsa::IndexSchedule;
use ferric_llama::hyv4::{Cfg, Hyv4};

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => { eprintln!("usage: hyv4_validate <gguf-or-header-prefix>\n\nsee this file's header for the curl"); return }
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => { eprintln!("cannot read {path}: {e}"); return }
    };
    // A header prefix is not a whole file, so the parse is expected to know only the table.
    let g = match ferric_gguf::parse(bytes) {
        Ok(g) => g,
        Err(e) => { eprintln!("PARSE FAILED: {e}"); std::process::exit(1) }
    };

    let cfg = match Cfg::from_gguf(&g) {
        Ok(c) => c,
        Err(e) => { eprintln!("CONFIG FAILED: {e}"); std::process::exit(1) }
    };
    let schedule = IndexSchedule::new(cfg.idx_is_full.clone()).expect("schedule");
    println!("{path}: hyv4, {} blocks, d={}, {} heads, {} experts (top-{}), {} tensors in the file",
             cfg.n_layer, cfg.d, cfg.n_head, cfg.n_expert, cfg.n_expert_used, g.tensors.len());
    println!("  qk {} = {} nope + {} rope, v {}, q_lora {}, kv_lora {}, cache {} floats/pos/layer",
             cfg.qk_head, cfg.qk_nope(), cfg.qk_rope, cfg.v_head, cfg.q_lora_rank, cfg.kv_lora_rank,
             cfg.cache_floats());
    println!("  hc {} streams, indexer {} heads x {} top-{}, {} of {} layers own one",
             cfg.hc, cfg.idx_heads, cfg.idx_head_dim, cfg.idx_top_k,
             schedule.live_cache_layers(), cfg.n_layer);

    let expected = Hyv4::expected_tensors(&cfg, &schedule);
    let (mut missing, mut wrong) = (Vec::new(), Vec::new());
    for (name, dims) in &expected {
        match g.tensor(name) {
            None => missing.push(name.clone()),
            Some(t) if t.dims != *dims => wrong.push((name.clone(), dims.clone(), t.dims.clone())),
            Some(_) => {}
        }
    }
    // The other direction matters too: a tensor in the file that the loader never asks for is a
    // piece of the architecture going unread, which is how a component gets silently skipped.
    let want: std::collections::HashSet<&str> = expected.iter().map(|(n, _)| n.as_str()).collect();
    let extra: Vec<&str> = g.tensors.iter().map(|t| t.name.as_str())
        .filter(|n| !want.contains(n)).collect();

    println!("\n  expected by the loader: {}", expected.len());
    println!("  present with the right dims: {}", expected.len() - missing.len() - wrong.len());
    if !missing.is_empty() {
        println!("\n  ⛔ MISSING ({}):", missing.len());
        for n in missing.iter().take(12) { println!("      {n}") }
        if missing.len() > 12 { println!("      … and {} more", missing.len() - 12) }
    }
    if !wrong.is_empty() {
        println!("\n  ⛔ WRONG DIMS ({}):", wrong.len());
        for (n, w, got) in wrong.iter().take(12) { println!("      {n}\n        expected {w:?}, file has {got:?}") }
        if wrong.len() > 12 { println!("      … and {} more", wrong.len() - 12) }
    }
    if !extra.is_empty() {
        println!("\n  ⚠ IN THE FILE BUT NEVER READ ({}):", extra.len());
        let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
        for n in &extra {
            let k = n.split('.').skip(2).collect::<Vec<_>>().join(".");
            *kinds.entry(if k.is_empty() { (*n).to_string() } else { k }).or_default() += 1;
        }
        for (k, c) in kinds.iter().take(12) { println!("      {k}  x{c}") }
    }

    if missing.is_empty() && wrong.is_empty() && extra.is_empty() {
        println!("\n  PASS — every tensor this loader asks for exists in Tencent's file with the dims\n  \
                  it expects, and nothing in the file goes unread. The loader's view of the FORMAT is\n  \
                  correct against the real artifact. Its arithmetic still needs the weights.");
    } else {
        println!("\n  FAIL — the loader and the real checkpoint disagree; see above.");
        std::process::exit(1);
    }
}
