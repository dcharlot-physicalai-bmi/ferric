//! **Did the loader understand this checkpoint?** Derive `Cfg` from a GGUF header and print it.
//!
//! Metadata parsing is where a port fails silently. Every other kind of mistake announces itself — a
//! missing tensor errors, a wrong kernel produces garbage — but a metadata key read with the wrong
//! accessor returns `Err`, hits a `.unwrap_or(default)`, and the model runs. Muse Glimmer stores its
//! attention schedule as an ARRAY of 52 flags; read with the scalar accessor it yields `Err`, the
//! fallback gives 0, and all 52 layers quietly become global attention.
//!
//! Because the header carries all of this and the weights carry none of it, that class of bug is
//! findable from the first 64 MB of a 17 GB file, before any of it has been downloaded.
//!
//!   cargo run -p ferric-llama --example cfg_from_header --release -- <header-or-full.gguf>
use ferric_gguf::parse;
use ferric_llama::qwen3::Cfg;

fn main() {
    let path = std::env::args().nth(1).expect("usage: cfg_from_header <file.gguf>");
    let g = parse(std::fs::read(&path).expect("read")).expect("parse header");

    let cfg = match Cfg::from_gguf(&g) {
        Ok(c) => c,
        Err(e) => {
            println!("Cfg::from_gguf REFUSED this file: {e}");
            println!("\nA refusal is the good outcome — it names the key it could not read. The bad");
            println!("outcome is a Cfg that parses with a wrong default, which is what this example");
            println!("exists to make visible.");
            std::process::exit(1);
        }
    };

    println!("Cfg derived from {path}\n");
    println!("  n_layer      {:>8}    n_embd    {:>8}    n_ff      {:>8}", cfg.n_layer, cfg.n_embd, cfg.n_ff);
    println!("  n_head       {:>8}    n_head_kv {:>8}    head_dim  {:>8}", cfg.n_head, cfg.n_head_kv, cfg.head_dim);
    println!("  n_vocab      {:>8}    eps       {:>8.1e}    rope_base {:>8.0}", cfg.n_vocab, cfg.eps, cfg.rope_base);
    println!("  qk_norm      {:>8}    qkv_bias  {:>8}    embd_scale{:>8.3}", cfg.has_qk_norm, cfg.qkv_bias, cfg.embd_scale);
    println!("  post_norms   {:>8}    nope_glob {:>8}    logit_scl {:>8.4}", cfg.post_norms, cfg.nope_global, cfg.logit_scale);
    println!("  attn_softcap {:>8.1}    final_cap {:>8.1}    window    {:>8}", cfg.attn_softcap, cfg.final_softcap, cfg.sliding_window);

    // ---- the layer schedule, which is the thing most likely to be silently wrong ----
    let n_local = cfg.swa.iter().filter(|&&b| b).count();
    println!("\n  attention schedule ({} entries, {} local / {} global):",
             cfg.swa.len(), n_local, cfg.swa.len() - n_local);
    let row: String = cfg.swa.iter().map(|&b| if b { 'L' } else { 'G' }).collect();
    for (i, ch) in row.as_bytes().chunks(52).enumerate() {
        println!("    [{:>3}] {}", i * 52, String::from_utf8_lossy(ch));
    }

    // A schedule of the right LENGTH but all one value is the exact failure this catches: it is what
    // a scalar read of an array-valued key produces, and it looks perfectly healthy in isolation.
    if cfg.swa.len() != cfg.n_layer {
        println!("\n  ⚠ schedule length {} != n_layer {} — the per-layer vector does not cover the model.",
                 cfg.swa.len(), cfg.n_layer);
    } else if cfg.sliding_window > 0 && n_local == 0 {
        println!("\n  ⚠ sliding_window is {} but NO layer is marked local. This is what reading an",
                 cfg.sliding_window);
        println!("    array-valued sliding_window_pattern with a scalar accessor looks like: the model");
        println!("    will run, every layer attending to the full context, and be quietly wrong.");
    } else if n_local > 0 {
        // Report the repeating unit rather than asserting one, so an irregular schedule is visible
        // instead of being rounded to the nearest tidy story.
        let period = (1..=cfg.swa.len()).find(|&p| cfg.swa.chunks(p).all(|c| c == &cfg.swa[..c.len()]));
        match period {
            Some(p) if p < cfg.swa.len() => println!("\n  Repeating unit of {p}: {}",
                cfg.swa[..p].iter().map(|&b| if b { "local" } else { "global" }).collect::<Vec<_>>().join(", ")),
            _ => println!("\n  No repeating unit — the schedule is irregular and must be read per layer."),
        }
        println!("  RoPE: {}", if cfg.nope_global { "local layers only (global layers are NoPE)" } else { "every layer" });
    }
}
