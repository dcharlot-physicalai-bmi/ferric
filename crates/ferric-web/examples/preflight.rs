//! **Name what a checkpoint cannot run, before anything allocates.**
//!
//!   cargo run -p ferric-web --example preflight --release -- <model.gguf> [...]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: preflight <model.gguf> [...]");
    let mut any_bad = false;
    for p in &args {
        let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"));
        match ferric_web::FerricModel::unsupported_tensors(&bytes) {
            Err(e) => { println!("{p}\n  UNPARSEABLE: {e}"); any_bad = true; }
            Ok(bad) if bad.is_empty() => println!("{p}\n  every tensor has a native matmul"),
            Ok(bad) => {
                any_bad = true;
                let mut by_type: std::collections::BTreeMap<u32, usize> = Default::default();
                for (_, t) in &bad { *by_type.entry(*t).or_default() += 1; }
                println!("{p}\n  {} of its tensors have NO native matmul:", bad.len());
                for (t, n) in by_type { println!("    ggml type {t:<3} x{n:<5} e.g. {}", bad.iter().find(|(_, x)| *x == t).unwrap().0); }
            }
        }
    }
    // Exit code carries the verdict, so a script can gate on it without parsing this text.
    std::process::exit(if any_bad { 1 } else { 0 });
}
