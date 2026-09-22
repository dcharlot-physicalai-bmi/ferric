//! **Print the per-layer sliding-window schedule a real checkpoint resolves to**, and check it
//! against the rule the file's architecture actually uses.
//!
//!   cargo run -p ferric-llama --release --example swa_probe -- <model.gguf> [more.gguf ...]
//!
//! ⛔ WHY THIS EXISTS, AND WHY NOT `run_llama`. Verifying the G2 schedule change, I ran
//! `run_llama -- gemma-3-1b.gguf` and `-- gemma2-2b.gguf` and got the identical figure
//! `1.431e-6` from both. `run_llama` **ignores argv entirely** — it loads a fixed 2-layer synthetic
//! safetensors fixture from `testdata/`. A nonexistent path prints the same number. An example that
//! accepts an argument it never reads is a check that reports on a model you did not test.
//!
//! This one reads the file. It prints the schedule and, for an architecture whose arm is known,
//! asserts the schedule equals an INDEPENDENTLY transcribed copy of llama.cpp's rule — not a call
//! back into the same function under test.
use ferric_gguf::{GgufFile, GgufSource, Meta};

/// llama.cpp `set_swa_pattern`, transcribed here a SECOND time from
/// `.reference/llama.cpp/src/llama-hparams.cpp:8-17`. Deliberately not `nn::swa_schedule` — a check
/// that calls the function under test proves only that it is deterministic.
fn reference_rule(n_layer: usize, p: usize, dense_first: bool) -> Vec<bool> {
    (0..n_layer)
        .map(|il| p == 0 || if dense_first { il % p != 0 } else { il % p < p - 1 })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: swa_probe <model.gguf> [more.gguf ...]");
    let mut checked = 0usize;
    for path in &args {
        let g = match GgufFile::open(path) { Ok(g) => g, Err(e) => { println!("{path}: {e}"); continue } };
        let md = g.metadata();
        let arch = match md.get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => "?".into() };
        let n = match md.get(&format!("{arch}.block_count")) { Some(Meta::U(v)) => *v as usize, _ => 0 };
        let win = match md.get(&format!("{arch}.attention.sliding_window")) { Some(Meta::U(v)) => *v as usize, _ => 0 };
        let pat_key = format!("{arch}.attention.sliding_window_pattern");
        let scalar = match md.get(&pat_key) { Some(Meta::U(v)) => Some(*v as usize), _ => None };
        let is_arr = matches!(md.get(&pat_key), Some(Meta::Arr(_)));

        // An architecture this Cfg builder does not serve is reported, not fatal: the probe's job is
        // the schedule, and a family with its own loader simply has nothing to say here.
        let cfg = match ferric_llama::qwen3::Cfg::from_gguf(&g) {
            Ok(c) => c,
            Err(e) => { println!("\n{}\n  arch={arch}: this Cfg builder does not serve it ({e})",
                                 path.rsplit('/').next().unwrap_or(path)); continue }
        };
        let globals: Vec<usize> = cfg.swa.iter().enumerate().filter(|(_, s)| !**s).map(|(i, _)| i).collect();

        println!("\n{}", path.rsplit('/').next().unwrap_or(path));
        println!("  arch={arch}  layers={n}  window={win}  scalar_pattern={scalar:?}  array={is_arr}");
        println!("  swa[..{}] = {:?}", cfg.swa.len().min(12),
                 &cfg.swa[..cfg.swa.len().min(12)]);
        println!("  GLOBAL layers: {:?}{}", &globals[..globals.len().min(10)],
                 if globals.len() > 10 { " …" } else { "" });

        // Where the scalar arm is what produced this, check it against the second transcription.
        if !is_arr {
            let dense_first = matches!(arch.as_str(), "modern-bert" | "laguna" | "smallthinker" | "cohere2moe");
            let p = scalar.or(match arch.as_str() {
                "gemma2" => Some(2),
                a if a.starts_with("gemma") => Some(6),
                _ => None,
            });
            let want = match p { Some(p) => reference_rule(n, p, dense_first), None => vec![false; n] };
            assert_eq!(cfg.swa, want,
                       "{arch}: the resolved schedule does not match an independent transcription of \
                        llama.cpp's set_swa_pattern(p={p:?}, dense_first={dense_first})");
            println!("  ✅ matches an independently transcribed set_swa_pattern(p={p:?}, dense_first={dense_first})");
            checked += 1;
        } else {
            println!("  (array arm — the file states the schedule directly, nothing to derive)");
        }
    }
    assert!(checked > 0, "no checkpoint exercised the scalar arm — this run proved nothing about it");
    println!("\n{checked} checkpoint(s) checked against the reference rule.");
}
