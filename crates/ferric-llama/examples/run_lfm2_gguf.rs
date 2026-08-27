//! **Liquid AI LFM2 / LFM2.5 from a GGUF** — the conv/attention hybrid, packed weights, any size.
//!
//! `run_lfm2.rs` already runs LFM2-350M and validates it against HF `transformers` to 4e-6, but it
//! reads **safetensors** with the dimensions compiled in. So the operator was proven and the
//! *checkpoints people actually ship* were still unreachable. This reads the GGUF: packed Q4_K/Q6_K
//! weights, layer counts and the conv/attention schedule taken from metadata.
//!
//! Covers LFM2-350M/700M/1.2B, LFM2.5-1.2B/2.6B, and the language tower of LFM2.5-VL-3B — all
//! `general.architecture = "lfm2"`.
//!
//! ## The schedule is a per-layer array, not a ratio
//!
//! `lfm2.attention.head_count_kv` is an ARRAY with one entry per block; `0` marks a short-conv block
//! and a nonzero marks GQA attention. LFM2.5-2.6B reads
//! `0,0,8,0,0,8,0,0,0,8,0,0,0,8,0,0,0,8,0,0,0,8,0,0,8,0,0,8,0,0` — attention at 2,5,9,13,17,21,24,27,
//! with gaps of 2,2,3,3,3,3,2,2. It is deliberately non-uniform, so any modular rule is wrong; read
//! the array. A scalar accessor returns `Err` on it and a fallback default would silently produce a
//! model with the wrong mixer in most layers.
//!
//! ## The short-conv operator, from llama.cpp's `lfm2.cpp` and HF's `Lfm2ShortConv`
//!
//! ```text
//! BCx = in_proj(norm(h))            [3·d]; chunk 0 = B, chunk 1 = C, chunk 2 = x
//! bx  = B ⊙ x                       bare elementwise — there is NO activation in this block
//! y   = C ⊙ causal_depthwise_conv(bx, L=3)
//! out = out_proj(y)
//! ```
//! The conv is cross-correlation, not a flipped kernel: `out[c][t] = Σ_k W[c][k]·u[c][t-2+k]`, so the
//! LAST tap multiplies the current token. Both references agree, and Ferric's
//! `depthwise_conv1d_causal` already implements exactly this (validated against `transformers`).
//!
//! ## Two things the tensor names get wrong
//!
//! `token_embd_norm` is **not** a norm on the embeddings. It is the FINAL norm before the head —
//! llama.cpp maps it through a dedicated enum carrying the comment "fix for wrong tensor name". The
//! prologue is a plain row gather with no norm and no √d scale. And there is no `output.weight`: the
//! head is **tied** to `token_embd`.
//!
//!   cargo run -p ferric-llama --example run_lfm2_gguf --release -- <lfm2.gguf> "prompt" [n]
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_tensor::{nn, QMatrix, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

/// A block is one or the other; every block also has its own norms and a SwiGLU FFN.
enum Mixer {
    Conv { in_proj: QMatrix, conv: Tensor, out_proj: QMatrix },
    Attn { q: QMatrix, k: QMatrix, v: QMatrix, o: QMatrix, q_norm: Tensor, k_norm: Tensor, n_kv: usize },
}

struct Block {
    norm: Tensor,      // pre-mixer norm (GGUF calls it attn_norm even on conv blocks)
    mixer: Mixer,
    ffn_norm: Tensor,
    gate: QMatrix,
    up: QMatrix,
    down: QMatrix,
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("usage: run_lfm2_gguf <model.gguf> [prompt] [n]");
    let prompt = a.get(2).map(|s| s.as_str()).unwrap_or("The capital of France is");
    let n_gen: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(24);

    let g = GgufFile::open(path).expect("open gguf");
    let arch = match g.metadata.get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => String::new() };
    // Both arch strings share this runtime — lfm2moe is lfm2 with the FFN made a mixture after
    // `leading_dense_block_count` blocks. Every metadata key is prefixed with the arch's OWN name,
    // so the prefix is read from the file rather than hardcoded.
    assert!(arch == "lfm2" || arch == "lfm2moe",
            "this loader is for general.architecture = lfm2 or lfm2moe, got {arch:?}");

    let u = |k: &str| match g.metadata.get(&format!("{arch}.{k}")) { Some(Meta::U(v)) => *v as usize, _ => panic!("missing {arch}.{k}") };
    let uo = |k: &str| match g.metadata.get(&format!("{arch}.{k}")) { Some(Meta::U(v)) => *v as usize, _ => 0 };
    let f = |k: &str| match g.metadata.get(&format!("{arch}.{k}")) { Some(Meta::F(v)) => *v, _ => panic!("missing {arch}.{k}") };
    let n_layer = u("block_count");
    let d = u("embedding_length");
    let n_head = u("attention.head_count");
    let eps = f("attention.layer_norm_rms_epsilon") as f32;
    let rope_base = f("rope.freq_base") as f32;
    let conv_l = u("shortconv.l_cache");
    // The schedule. Nonzero = attention with that many KV heads; 0 = short conv.
    let kv_per_layer: Vec<usize> = match g.metadata.get(&format!("{arch}.attention.head_count_kv")) {
        Some(Meta::Arr(v)) => v.iter().map(|m| match m { Meta::U(x) => *x as usize, Meta::I(x) => *x as usize, _ => 0 }).collect(),
        Some(Meta::U(v)) => vec![*v as usize; n_layer],
        _ => panic!("no {arch}.attention.head_count_kv"),
    };
    assert_eq!(kv_per_layer.len(), n_layer, "schedule must cover every block");
    let head_dim = d / n_head;
    let n_attn = kv_per_layer.iter().filter(|&&k| k > 0).count();

    let ctx = Arc::new(Context::new().await.unwrap());
    let t0 = std::time::Instant::now();

    let qm = |name: &str| -> QMatrix {
        let t = g.tensor(name).unwrap_or_else(|| panic!("missing {name}"));
        let (ty, rows, cols) = (t.ggml_type, t.dims[1] as usize, t.dims[0] as usize);
        if QMatrix::block_bytes(ty).is_some() {
            QMatrix::from_bytes(&ctx, &g.raw(name).unwrap(), ty, rows, cols).unwrap()
        } else {
            QMatrix::from_dense(&ctx, &g.dequant(name).unwrap(), rows, cols)
        }
    };
    let f32t = |name: &str, shape: &[usize]| Tensor::from_vec(&ctx, &g.dequant(name).unwrap(), shape);

    let blocks: Vec<Block> = (0..n_layer).map(|il| {
        let b = |s: &str| format!("blk.{il}.{s}");
        let n_kv = kv_per_layer[il];
        let mixer = if n_kv > 0 {
            Mixer::Attn {
                q: qm(&b("attn_q.weight")), k: qm(&b("attn_k.weight")),
                v: qm(&b("attn_v.weight")), o: qm(&b("attn_output.weight")),
                q_norm: f32t(&b("attn_q_norm.weight"), &[head_dim]),
                k_norm: f32t(&b("attn_k_norm.weight"), &[head_dim]),
                n_kv,
            }
        } else {
            // GGUF stores the kernel as [L, d] (ne[0]=L taps); dequantized row-major that is [d, L],
            // which is exactly the [C, L] layout depthwise_conv1d_causal wants. No transpose.
            Mixer::Conv {
                in_proj: qm(&b("shortconv.in_proj.weight")),
                conv: f32t(&b("shortconv.conv.weight"), &[d, conv_l]),
                out_proj: qm(&b("shortconv.out_proj.weight")),
            }
        };
        Block {
            norm: f32t(&b("attn_norm.weight"), &[d]),
            mixer,
            ffn_norm: f32t(&b("ffn_norm.weight"), &[d]),
            gate: qm(&b("ffn_gate.weight")), up: qm(&b("ffn_up.weight")), down: qm(&b("ffn_down.weight")),
        }
    }).collect();

    // Final norm is token_embd_norm despite the name; the head is tied to the embedding table.
    let out_norm = f32t("token_embd_norm.weight", &[d]);
    let head = if g.tensor("output.weight").is_some() { qm("output.weight") } else { qm("token_embd.weight") };
    let embd_ty = g.tensor("token_embd.weight").unwrap().ggml_type;
    let embd_raw = g.raw("token_embd.weight").unwrap();
    let n_vocab = g.tensor("token_embd.weight").unwrap().dims[1] as usize;

    println!("LFM2 GGUF · {n_layer} blocks ({} conv + {n_attn} attn) · d={d} · {n_head}h/{}kv × {head_dim} · vocab={n_vocab}",
             n_layer - n_attn, kv_per_layer.iter().max().unwrap());
    println!("  schedule: {}", kv_per_layer.iter().map(|&k| if k > 0 { 'A' } else { 'c' }).collect::<String>());
    println!("  loaded in {:.2?}\n", t0.elapsed());

    // ---- tokenizer: byte-level BPE straight out of the GGUF ----
    let tokens: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokenizer.ggml.tokens"),
    };
    let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
        Some(Meta::Arr(v)) => v.iter().filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.into(), y.into())) } else { None }).collect(),
        _ => Vec::new(),
    };
    let bpe = ferric_tokenizer::Bpe::new(vocab, &merges);
    let mut ids: Vec<u32> = bpe.encode(prompt);
    if let Some(Meta::U(bos)) = g.metadata.get("tokenizer.ggml.bos_token_id") {
        if matches!(g.metadata.get("tokenizer.ggml.add_bos_token"), Some(Meta::Bool(true))) { ids.insert(0, *bos as u32); }
    }
    println!("prompt: {prompt:?}  ({} tokens)", ids.len());

    // ---- forward: full prefix each step ----
    //
    // Deliberately O(n^2): the conv mixer carries a 2-timestep recurrent state and attention carries a
    // KV cache, and running the whole prefix every step needs NEITHER. That makes the FIRST question
    // -- is the operator right on real packed weights -- answerable without also getting incremental
    // state right. Caching is a speed change to make once this is known correct.
    let forward = |ids: &[u32]| {
        let t = ids.len();
        let row_bytes = ferric_gguf::type_size(embd_ty, d).unwrap();
        let mut e = Vec::with_capacity(t * d);
        for &tok in ids {
            let off = tok as usize * row_bytes;
            e.extend(ferric_gguf::deq_raw(&embd_raw[off..off + row_bytes], d, embd_ty).unwrap());
        }
        // Prologue is a bare gather: no norm, no sqrt(d) scale.
        let mut x = Tensor::from_vec(&ctx, &e, &[t, d]);
        for blk in &blocks {
            let h = x.rmsnorm(&blk.norm, eps);
            let op = match &blk.mixer {
                Mixer::Attn { q, k, v, o, q_norm, k_norm, n_kv } => {
                    let qh = h.matmul_q(q).reshape(&[t * n_head, head_dim]).rmsnorm(q_norm, eps)
                        .reshape(&[t, n_head * head_dim]).rope(n_head, head_dim, rope_base, 0);
                    let kh = h.matmul_q(k).reshape(&[t * n_kv, head_dim]).rmsnorm(k_norm, eps)
                        .reshape(&[t, n_kv * head_dim]).rope(*n_kv, head_dim, rope_base, 0);
                    let vh = h.matmul_q(v);
                    nn::causal_attention(&qh, &kh, &vh, n_head, *n_kv, 0.0).matmul_q(o)
                }
                Mixer::Conv { in_proj, conv, out_proj } => {
                    let bcx = h.matmul_q(in_proj);                 // [t, 3d]
                    let bb = bcx.narrow(1, 0, d).contiguous();     // chunk 0 = B
                    let cc = bcx.narrow(1, d, d).contiguous();     // chunk 1 = C
                    let xx = bcx.narrow(1, 2 * d, d).contiguous(); // chunk 2 = x
                    let conv_out = bb.mul(&xx).depthwise_conv1d_causal(conv, conv_l);
                    cc.mul(&conv_out).matmul_q(out_proj)
                }
            };
            x = x.add(&op);
            let hn = x.rmsnorm(&blk.ffn_norm, eps);
            let sw = hn.matmul_q(&blk.gate).silu().mul(&hn.matmul_q(&blk.up));
            x = x.add(&sw.matmul_q(&blk.down));
        }
        x.rmsnorm(&out_norm, eps).matmul_q(&head)
    };

    let t0 = std::time::Instant::now();
    let mut out = String::new();
    for i in 0..n_gen {
        let logits = forward(&ids).to_vec().await;
        let last = &logits[logits.len() - n_vocab..];
        let next = last.iter().enumerate().fold((0usize, f32::MIN), |b, (j, &v)| if v > b.1 { (j, v) } else { b }).0 as u32;
        if i == 0 {
            let bad = last.iter().filter(|v| !v.is_finite()).count();
            println!("first-step logits: {bad} non-finite of {n_vocab}");
        }
        if matches!(g.metadata.get("tokenizer.ggml.eos_token_id"), Some(Meta::U(e)) if *e as u32 == next) { break; }
        out.push_str(&tokens[next as usize]);
        ids.push(next);
    }
    // GPT-2 byte-level: printable-unicode back to raw bytes.
    let dec: String = {
        let mut m = std::collections::HashMap::new();
        let mut n = 0u32;
        for b in 0u32..256 {
            let p = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
            let c = if p { b } else { let c = 256 + n; n += 1; c };
            m.insert(char::from_u32(c).unwrap(), b as u8);
        }
        String::from_utf8_lossy(&out.chars().filter_map(|c| m.get(&c).copied()).collect::<Vec<u8>>()).into_owned()
    };
    println!("\n{prompt}{dec}\n");
    println!("  {} tokens in {:.2?}", n_gen, t0.elapsed());
}
