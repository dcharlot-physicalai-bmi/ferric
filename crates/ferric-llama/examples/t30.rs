use ferric_gguf::GgufFile;
fn main() {
    let g = GgufFile::open(&std::env::args().nth(1).unwrap()).unwrap();
    let mut seen = std::collections::HashSet::new();
    for t in &g.tensors { if t.ggml_type == 30 { let base = t.name.replace(|c: char| c.is_ascii_digit(), "N"); if seen.insert(base.clone()) { println!("{}  dims={:?}", t.name, t.dims); } } }
}
