//! Load a GGUF and report per-type tensor counts — the smoke test for parse-time type resolution.
//! Reads its subject from argv, no default.
fn main() {
    let p = std::env::args().nth(1).expect("usage: loadprobe <model.gguf>");
    match ferric_gguf::GgufFile::open(&p) {
        Ok(g) => {
            let mut by: std::collections::BTreeMap<u32, usize> = Default::default();
            for t in &g.tensors { *by.entry(t.ggml_type).or_default() += 1; }
            println!("LOADED {} tensors · types {:?}", g.tensors.len(), by);
        }
        Err(e) => println!("REFUSED: {}", e.lines().next().unwrap_or("")),
    }
}
