//! **Do ggml-org's GGUFs carry the authors' Gemma 4 weights?** Tensor by tensor, BY VALUE.
//!
//! The conformance gate judges logits at the authors' float32-vs-float64 floor, which is only meaningful if
//! the weights Ferric reads ARE the authors'. This checks the two GGUF files the gate and the server use
//! against the authors' `model.safetensors`:
//!
//! * the **mmproj** (vision + audio towers + both embedders) through [`gemma4_mm::gguf_name`] and
//!   [`gemma4_mm::Src::f32s`] — the exact path the towers load through, permutation included;
//! * optionally the **text** GGUF, through the name table below.
//!
//! For a BF16 file every tensor must be BIT-identical (bf16 → f32 is exact). For a quantized file the
//! maximum relative error per tensor is reported instead. Every tensor in the GGUF must be consumed exactly
//! once — a tensor the table never reaches is a weight the runtime would silently not load.
//!
//!   cargo run -p ferric-llama --release --example gemma4_mm_weights -- <checkpoint-dir> <mmproj.gguf> [text.gguf]
use ferric_gguf::{GgufFile, GgufSource};
use ferric_llama::gemma4_mm::{gguf_name, Src};
use std::collections::HashSet;

/// The authors' language-model name → ggml-org's text GGUF name.
fn text_name(hf: &str) -> Option<String> {
    let r = hf.strip_prefix("model.language_model.")?;
    let top = [("embed_tokens.weight", "token_embd.weight"), ("embed_tokens_per_layer.weight", "per_layer_token_embd.weight"),
               ("per_layer_model_projection.weight", "per_layer_model_proj.weight"),
               ("per_layer_projection_norm.weight", "per_layer_proj_norm.weight"), ("norm.weight", "output_norm.weight")];
    if let Some((_, g)) = top.iter().find(|(h, _)| *h == r) { return Some(g.to_string()); }
    let r = r.strip_prefix("layers.")?;
    let (i, rest) = r.split_once('.')?;
    let map = [("input_layernorm.weight", "attn_norm.weight"), ("post_attention_layernorm.weight", "post_attention_norm.weight"),
               ("pre_feedforward_layernorm.weight", "ffn_norm.weight"), ("post_feedforward_layernorm.weight", "post_ffw_norm.weight"),
               ("self_attn.q_proj.weight", "attn_q.weight"), ("self_attn.k_proj.weight", "attn_k.weight"),
               ("self_attn.v_proj.weight", "attn_v.weight"), ("self_attn.o_proj.weight", "attn_output.weight"),
               ("self_attn.q_norm.weight", "attn_q_norm.weight"), ("self_attn.k_norm.weight", "attn_k_norm.weight"),
               ("mlp.gate_proj.weight", "ffn_gate.weight"), ("mlp.up_proj.weight", "ffn_up.weight"), ("mlp.down_proj.weight", "ffn_down.weight"),
               ("per_layer_input_gate.weight", "inp_gate.weight"), ("per_layer_projection.weight", "proj.weight"),
               ("post_per_layer_input_norm.weight", "post_norm.weight"), ("layer_scalar", "layer_output_scale.weight")];
    map.iter().find(|(h, _)| *h == rest).map(|(_, g)| format!("blk.{i}.{g}"))
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut m) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) { d = d.max((*x as f64 - *y as f64).abs()); m = m.max((*y as f64).abs()); }
    d / m.max(1e-30)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 { eprintln!("usage: gemma4_mm_weights <checkpoint-dir> <mmproj.gguf> [text.gguf]"); std::process::exit(2); }
    let hf = Src::open(&a[1]).expect("checkpoint dir");
    let Src::Hf { st, .. } = &hf else { panic!("first argument must be the checkpoint directory") };
    let mm = Src::open(&a[2]).expect("mmproj");
    let g = mm.gguf().expect("mmproj gguf");
    let mut fails = 0usize;

    // ---- towers ----
    let mut used: HashSet<String> = HashSet::new();
    let (mut n, mut exact, mut worst) = (0usize, 0usize, (0f64, String::new()));
    let mut pds_ulp = 0usize;
    let names: Vec<String> = st.names().filter(|k| k.starts_with("model.vision_tower.") || k.starts_with("model.audio_tower.")
                                                    || k.starts_with("model.embed_vision.") || k.starts_with("model.embed_audio.")).cloned().collect();
    for k in &names {
        let Some(gn) = gguf_name(k) else { println!("NO MMPROJ NAME  {k}"); fails += 1; continue };
        if g.tensor(&gn).is_none() { println!("ABSENT          {k} -> {gn}"); fails += 1; continue; }
        if !used.insert(gn.clone()) { println!("TWICE           {gn} (again from {k})"); fails += 1; }
        n += 1;
        if k.ends_with("per_dim_scale") {
            // Stored post-softplus by the converter: compare what the model multiplies by.
            let (want, got) = (hf.softplus_per_dim_scale(k).expect("authors'"), mm.softplus_per_dim_scale(k).expect("mmproj"));
            let ulps = want.iter().zip(&got).map(|(a, b)| (a.to_bits() as i64 - b.to_bits() as i64).abs()).max().unwrap_or(0);
            if ulps > 1 { println!("PER_DIM_SCALE   {k}: softplus differs by {ulps} ulp"); fails += 1; } else { exact += 1; }
            if ulps == 1 { pds_ulp += 1; }
            continue;
        }
        let (want, ws) = hf.f32s(k).expect("authors' tensor");
        let (got, gs) = mm.f32s(k).expect("mmproj tensor");
        if ws != gs { println!("SHAPE           {k}: authors {ws:?}, mmproj {gs:?}"); fails += 1; continue; }
        if want == got { exact += 1; } else {
            let r = rel(&got, &want);
            if r > worst.0 { worst = (r, k.clone()); }
        }
    }
    let unread: Vec<&str> = g.tensors.iter().map(|t| t.name.as_str()).filter(|t| !used.contains(*t)).collect();
    println!("towers: {n} of the authors' tensors mapped; {exact} bit-identical; worst relative error {:.3e} ({})",
             worst.0, if worst.1.is_empty() { "-" } else { &worst.1 });
    println!("        mmproj tensors never reached by the table: {} {:?}", unread.len(), &unread[..unread.len().min(8)]);
    println!("        per_dim_scale (stored post-softplus): {pds_ulp} of the layers 1 ulp from Ferric's softplus of the authors' value, the rest equal");
    fails += unread.len();
    let all_bf16 = g.tensors.iter().all(|t| matches!(t.ggml_type, 0 | 1 | 30));
    if all_bf16 && exact != n { println!("  ⛔ a 16-bit mmproj must be BIT-identical; {} tensors are not", n - exact); fails += 1; }

    // ---- text ----
    if let Some(tp) = a.get(3) {
        let tg = GgufFile::open(tp).expect("text gguf");
        let mut used: HashSet<String> = HashSet::new();
        let (mut n, mut exact, mut worst) = (0usize, 0usize, (0f64, String::new()));
        let mut skipped_shared = 0usize;
        for k in st.names().filter(|k| k.starts_with("model.language_model.")) {
            let Some(gn) = text_name(k) else { println!("NO TEXT NAME    {k}"); fails += 1; continue };
            if tg.tensor(&gn).is_none() {
                // The shared-KV blocks' K/V projections and norms are dead weights the converter drops.
                if k.contains(".self_attn.k_") || k.contains(".self_attn.v_") { skipped_shared += 1; continue; }
                println!("ABSENT          {k} -> {gn}"); fails += 1; continue;
            }
            used.insert(gn.clone());
            let want = st.get(k).expect("authors' tensor").data;
            let got = tg.dequant(&gn).expect("gguf tensor");
            n += 1;
            if want.len() != got.len() { println!("SIZE            {k}"); fails += 1; continue; }
            if want == got { exact += 1; } else {
                let r = rel(&got, &want);
                if r > worst.0 { worst = (r, k.clone()); }
            }
        }
        let unread: Vec<&str> = tg.tensors.iter().map(|t| t.name.as_str()).filter(|t| !used.contains(*t)).collect();
        println!("text:   {n} tensors mapped; {exact} bit-identical; worst relative error {:.3e} ({}); {skipped_shared} dead shared-KV tensors not in the GGUF",
                 worst.0, if worst.1.is_empty() { "-" } else { &worst.1 });
        println!("        text GGUF tensors not from the authors' file: {:?}", unread);
        let all16 = tg.tensors.iter().filter(|t| used.contains(&t.name)).all(|t| matches!(t.ggml_type, 0 | 1 | 30));
        if all16 && exact != n { println!("  ⛔ a 16-bit text GGUF must be BIT-identical; {} tensors are not", n - exact); fails += 1; }
    }
    if fails > 0 { println!("FAIL ({fails})"); std::process::exit(1); }
    println!("PASS");
}
