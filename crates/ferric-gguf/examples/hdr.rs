use ferric_gguf::{parse, Meta};
use std::io::Read;
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let mut f = std::fs::File::open(&p).unwrap();
    let mut buf = vec![0u8; 96 << 20];
    let n = f.read(&mut buf).unwrap(); buf.truncate(n);
    let g = parse(buf).unwrap();
    let mut keys: Vec<_> = g.metadata.keys().cloned().collect(); keys.sort();
    println!("=== metadata ({} keys) ===", keys.len());
    for k in &keys {
        let v = match &g.metadata[k] {
            Meta::Arr(a) => format!("[{} items]", a.len()),
            Meta::Str(s) if s.len() > 60 => format!("{:?}…", &s[..60]),
            m => format!("{m:?}"),
        };
        println!("  {k} = {v}");
    }
    println!("\n=== tensors: {} ===", g.tensors.len());
    let mut byty = std::collections::BTreeMap::new();
    for t in &g.tensors { *byty.entry(t.ggml_type).or_insert(0usize) += 1; }
    println!("  types: {byty:?}");
    let filt = std::env::args().nth(2).unwrap_or_default();
    for t in g.tensors.iter().filter(|t| filt.is_empty() || t.name.contains(&filt)) { println!("  {:<44} {:?} ty={}", t.name, t.dims, t.ggml_type); }
}
