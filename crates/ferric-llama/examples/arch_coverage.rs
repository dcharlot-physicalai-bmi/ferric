//! **What does this runtime run, and would it run *that* file?** — the 30-day cadence readout.
//!
//! Ferric commits to supporting current releases within 30 days, the cadence llama.cpp, vLLM, MLX and
//! Ollama hold. Meeting a cadence is a process problem, and the process fails the same way every time:
//! the gap becomes visible only when someone points a new checkpoint at the server. This prints the
//! gap on demand.
//!
//! Two modes:
//!
//! ```text
//!   arch_coverage                 # the whole registry: what runs, at what confidence
//!   arch_coverage <file.gguf>     # would THIS checkpoint load, and down which runtime
//! ```
//!
//! The second mode reads only the GGUF header, so a 25 GB checkpoint answers as fast as a 1 GB one —
//! which means "can we run the model that shipped this morning" is a question with a same-morning
//! answer.
//!
//! Exits **2** when a named file is unsupported, so this can gate CI rather than be read by eye.
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::arch;

fn main() {
    let path = std::env::args().nth(1);

    let Some(path) = path else {
        print!("{}", arch::coverage());
        println!("Pass a .gguf to ask whether that specific checkpoint would load.");
        return;
    };

    let g = match GgufFile::open(&path) {
        Ok(g) => g,
        Err(e) => { eprintln!("cannot open {path}: {e}"); std::process::exit(2); }
    };
    let name = match g.metadata.get("general.architecture") {
        Some(Meta::Str(s)) => s.clone(),
        _ => String::new(),
    };
    // Useful context regardless of the verdict — if the answer is "not supported", these are the
    // numbers whoever writes the loader needs first.
    let u = |k: &str| match g.metadata.get(&format!("{name}.{k}")) { Some(Meta::U(v)) => Some(*v), _ => None };
    println!("file    {path}");
    println!("arch    {name:?}");
    if let Some(v) = u("block_count") { println!("blocks  {v}"); }
    if let Some(v) = u("embedding_length") { println!("d       {v}"); }
    if let Some(v) = u("context_length") { println!("context {v}"); }
    if let Some(v) = u("expert_count") { println!("experts {v} (MoE)"); }

    match arch::resolve(&name) {
        Ok(a) => {
            println!("\nSUPPORTED — {} runtime, status {}", a.runtime.label(), a.status.label());
            println!("  {}", a.note);
            if a.status == arch::Status::Loads {
                // Not a warning about crashing. A warning about the failure mode that does not crash.
                println!("\n  ⚠ `loads` is not `verified`: this runs and produces coherent text, but has\n    \
                          not been diffed against the reference implementation. A wrong RoPE convention\n    \
                          or a missed norm looks exactly like this — fluent output, no error.");
            }
        }
        Err(e) => {
            println!("\nNOT SUPPORTED\n{e}");
            std::process::exit(2);
        }
    }
}
