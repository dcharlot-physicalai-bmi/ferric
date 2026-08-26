//! Read a Nemotron-H checkpoint's schedule and SSM geometry, and check it reconciles.
//!   cargo run -p ferric-llama --example nemotron_cfg --release -- <model.gguf>
use ferric_gguf::GgufFile;
use ferric_llama::nemotron_h::{BlockKind, Cfg};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mp = a.get(1).expect("usage: nemotron_cfg <model.gguf>");
    let g = GgufFile::open(mp).expect("open");
    let cfg = Cfg::from_gguf(&g).expect("cfg");
    let (ssm, ffn, attn) = cfg.schedule();
    println!("nemotron_h · {} blocks · d={} · vocab={}", cfg.n_layer, cfg.d, cfg.n_vocab);
    println!("  schedule: {ssm} SSM · {ffn} MLP · {attn} attention");
    println!("  attention at blocks {:?}",
             (0..cfg.n_layer).filter(|&i| cfg.kind[i] == BlockKind::Attn).collect::<Vec<_>>());
    println!("  ssm: inner={} heads={} head_dim={} state={} groups={} conv={}",
             cfg.ssm_inner, cfg.ssm_heads, cfg.ssm_head_dim, cfg.ssm_state, cfg.ssm_groups, cfg.ssm_conv);

    // Reconcile the declared geometry against the actual tensor shapes. A schedule or a split that
    // disagrees with the weights is the failure that produces fluent, wrong text, so it is checked
    // here rather than discovered by reading output.
    let first_ssm = (0..cfg.n_layer).find(|&i| cfg.kind[i] == BlockKind::Ssm).expect("an SSM block");
    let t = |n: String| g.tensor(&n).unwrap_or_else(|| panic!("no {n}")).dims.clone();
    let in_w = t(format!("blk.{first_ssm}.ssm_in.weight"));
    let conv = t(format!("blk.{first_ssm}.ssm_conv1d.weight"));
    let norm = t(format!("blk.{first_ssm}.ssm_norm.weight"));
    let want_in = 2 * cfg.ssm_inner + 2 * cfg.ssm_groups * cfg.ssm_state + cfg.ssm_heads;
    let want_conv = cfg.ssm_inner + 2 * cfg.ssm_groups * cfg.ssm_state;
    println!("\n  ssm_in.weight  {in_w:?}  z|x|BC|dt = {}+{}+{}+{} = {want_in} {}",
             cfg.ssm_inner, cfg.ssm_inner, 2 * cfg.ssm_groups * cfg.ssm_state, cfg.ssm_heads,
             if in_w[1] as usize == want_in { "✅" } else { "❌" });
    println!("  ssm_conv1d     {conv:?}  x+BC = {want_conv} {}",
             if conv[1] as usize == want_conv { "✅" } else { "❌" });
    println!("  ssm_norm       {norm:?}  groups x width = {} {}",
             cfg.ssm_inner, if norm[0] as usize * norm[1] as usize == cfg.ssm_inner { "✅" } else { "❌" });
    assert_eq!(in_w[1] as usize, want_in, "the z|xBC|dt split does not match ssm_in");
    assert_eq!(conv[1] as usize, want_conv, "conv1d width does not match x+BC");
    assert_eq!(norm[0] as usize * norm[1] as usize, cfg.ssm_inner, "grouped norm does not cover inner");
    println!("\n  ✅ schedule and SSM geometry reconcile with the weights");
}
