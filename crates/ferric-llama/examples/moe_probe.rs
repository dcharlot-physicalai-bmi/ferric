use ferric_gguf::{GgufFile, Meta};
fn main() {
    let g = GgufFile::open(&std::env::args().nth(1).unwrap()).unwrap();
    for k in ["general.architecture","qwen35moe.expert_count","qwen35moe.expert_used_count","qwen35moe.expert_feed_forward_length","qwen35moe.expert_shared_feed_forward_length","qwen35moe.block_count","qwen35moe.embedding_length","qwen35moe.feed_forward_length","qwen35moe.full_attention_interval"] {
        match g.metadata.get(k) { Some(Meta::U(v))=>println!("{k} = {v}"), Some(Meta::Str(s))=>println!("{k} = {s}"), Some(Meta::I(v))=>println!("{k} = {v}"), _=>{} }
    }
    println!("--- blk.0 tensors ---");
    let mut ns: Vec<_> = g.tensors.iter().filter(|t| t.name.starts_with("blk.0.")).collect();
    ns.sort_by(|a,b| a.name.cmp(&b.name));
    for t in ns { println!("{}  dims={:?} type={}", t.name, t.dims, t.ggml_type); }
}
