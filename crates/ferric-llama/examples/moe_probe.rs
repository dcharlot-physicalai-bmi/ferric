use ferric_gguf::{GgufFile, Meta};
fn main() {
    let g = GgufFile::open(&std::env::args().nth(1).unwrap()).unwrap();
    for k in ["qwen35.rope.freq_base","qwen35moe.rope.freq_base","qwen35.rope.dimension_count","qwen35moe.rope.dimension_count","qwen35.attention.key_length","qwen35moe.attention.key_length","qwen35.attention.head_count","qwen35moe.attention.head_count","qwen35.attention.head_count_kv","qwen35moe.attention.head_count_kv","general.architecture","qwen35moe.expert_count","qwen35moe.expert_used_count","qwen35moe.expert_feed_forward_length","qwen35moe.expert_shared_feed_forward_length","qwen35moe.block_count","qwen35moe.embedding_length","qwen35moe.feed_forward_length","qwen35moe.full_attention_interval"] {
        match g.metadata.get(k) { Some(Meta::U(v))=>println!("{k} = {v}"), Some(Meta::Str(s))=>println!("{k} = {s}"), Some(Meta::I(v))=>println!("{k} = {v}"), _=>{} }
    }
    for k in ["qwen35.rope.dimension_sections","qwen35moe.rope.dimension_sections"] {
        if let Some(ferric_gguf::Meta::Arr(a)) = g.metadata.get(k) { println!("{k} = {:?}", a.iter().map(|m| format!("{m:?}")).collect::<Vec<_>>()); }
    }
    println!("--- blk.0 tensors ---");
    let mut ns: Vec<_> = g.tensors.iter().filter(|t| t.name.starts_with("blk.0.")).collect();
    ns.sort_by(|a,b| a.name.cmp(&b.name));
    for t in ns { println!("{}  dims={:?} type={}", t.name, t.dims, t.ggml_type); }
}
