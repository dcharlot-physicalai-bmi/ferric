use ferric_gguf::GgufFile;
fn main() {
    let g = GgufFile::open(&std::env::args().nth(1).unwrap()).unwrap();
    let n = 41;
    for il in 0..n {
        let has = |s: &str| g.tensors.iter().any(|t| t.name == format!("blk.{il}.{s}"));
        let kind = if has("ssm_out.weight") { "GDN " } else if has("attn_output.weight") { "ATTN" } else { "??? " };
        let qkv = if has("attn_qkv.weight") { "qkv" } else if has("attn_q.weight") { "q/k/v" } else { "-" };
        println!("blk.{il:2} {kind} ({qkv})");
    }
}
