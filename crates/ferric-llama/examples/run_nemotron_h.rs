//! **NVIDIA Nemotron-H / Nemotron 3.5 Lightning from a GGUF** — Mamba-2 + attention + ReLU² MoE.
//!
//! `general.architecture = "nemotron_h_moe"`. Written against llama.cpp's `nemotron-h.cpp`,
//! `mamba-base.cpp` and ggml's `ssm_scan` kernel, read verbatim — the Muse Glimmer port cost hours of
//! ablation because I reasoned about the reference instead of reading it.
//!
//! ## A FLAT block sequence — one operation per block, not mixer+FFN pairs
//!
//! ```text
//! cur = rmsnorm(x, attn_norm)          // every block has this norm
//! if      is_recurrent(il)  cur = mamba2(cur)
//! else if n_ff(il) == 0     cur = attention(cur)
//! else                      cur = moe_ffn(cur)
//! x = x + cur                          // ONE residual per block
//! ```
//! Read from the per-layer arrays `attention.head_count_kv[53]` and `feed_forward_length[53]`: a block
//! is attention iff it has KV heads and no FFN width, MoE iff it has an FFN width, else Mamba-2. For
//! Nemotron 3.5 Lightning that is 23 SSM / 6 attention / 23 MoE over a 52-block trunk, plus one
//! trailing MTP draft block which `nextn_predict_layers` marks and which plain inference skips.
//!
//! ## Two things the metadata implies and the graph denies
//!
//! **Attention has NO positional encoding.** `rope.dimension_count = 84` sits in the GGUF and
//! `nemotron-h.cpp` contains zero rope calls; `build_qkv` only projects and reshapes. The Mamba-2
//! layers carry position, so attention is position-free. Applying rope here because the metadata
//! mentions it would be a silent, plausible-looking error.
//!
//! **`ssm.time_step_rank` is the HEAD COUNT** (64), not a rank. head_dim = `ssm.inner_size` / it.
//!
//! ## The Mamba-2 mixer
//!
//! ```text
//! zxBCdt = ssm_in(h)                      // 10304 = 2·d_inner + 2·n_group·d_state + n_head
//!   z = [0,4096)   xBC = [4096,10240)   dt = [10240,10304)
//! xBC = silu(conv1d(xBC, k=4) + conv_bias)
//!   x = xBC[0,4096)   B = xBC[4096,5120)   C = xBC[5120,6144)
//! dt = softplus(dt + dt_bias) ;  dA = exp(dt · A)   // A = ssm_a, used DIRECTLY (already negative)
//! y  = scan(x, dA, dt, B, C) + D·x                  // ssm_scan fuses the D skip
//! y  = silu(z) · y                                  // ggml_swiglu_split, NOT a bare multiply
//! y  = grouped_rmsnorm(y)                           // ssm_norm is [d_inner/n_group, n_group]
//! out = ssm_out(y)
//! ```
//!
//!   cargo run -p ferric-llama --example run_nemotron_h --release -- <model.gguf> "prompt" [n]
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_tensor::dtype::{Q5_0Weights, Q8_0Weights};
use ferric_tensor::{nn, QMatrix, Tensor};
use std::sync::Arc;

/// The expert DOWN slab's quant type varies by layer in a real mixed-precision checkpoint —
/// Nemotron 3.5 Lightning Q4_K_M stores it Q8_0 in 11 MoE blocks and Q5_0 in the other 13. A loader
/// that picks one type per role loads the first blocks and then trips a byte-length assert.
enum DownSlab { Q5(Q5_0Weights), Q8(Q8_0Weights) }

enum Block {
    Ssm {
        in_proj: QMatrix, conv_w: Tensor, conv_b: Tensor,
        a: Tensor, d: Tensor, dt_b: Tensor, norm: Tensor, out_proj: QMatrix,
    },
    Attn { q: QMatrix, k: QMatrix, v: QMatrix, o: QMatrix, n_kv: usize },
    Moe {
        router: Tensor, probs_b: Tensor,
        up_exps: Q5_0Weights, down_exps: DownSlab,
        up_sh: QMatrix, down_sh: QMatrix,
    },
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("usage: run_nemotron_h <model.gguf> [prompt] [n]");
    let prompt = a.get(2).map(|s| s.as_str()).unwrap_or("The capital of France is");
    let n_gen: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);

    let g = GgufFile::open(path).expect("open gguf");
    let arch = match g.metadata.get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => String::new() };
    assert!(arch.starts_with("nemotron_h"), "expected nemotron_h*, got {arch:?}");
    let u = |k: &str| match g.metadata.get(&format!("{arch}.{k}")) { Some(Meta::U(v)) => *v as usize, _ => panic!("missing {arch}.{k}") };
    let uo = |k: &str, d: usize| match g.metadata.get(&format!("{arch}.{k}")) { Some(Meta::U(v)) => *v as usize, _ => d };
    let fo = |k: &str, d: f32| match g.metadata.get(&format!("{arch}.{k}")) { Some(Meta::F(v)) => *v as f32, _ => d };
    let arr = |k: &str| -> Vec<usize> {
        match g.metadata.get(&format!("{arch}.{k}")) {
            Some(Meta::Arr(v)) => v.iter().map(|m| match m { Meta::U(x) => *x as usize, Meta::I(x) => *x as usize, _ => 0 }).collect(),
            Some(Meta::U(v)) => vec![*v as usize; u("block_count")],
            _ => panic!("missing {arch}.{k}"),
        }
    };

    let n_layer = u("block_count");
    let d_model = u("embedding_length");
    let n_head = u("attention.head_count");
    let head_dim = u("attention.key_length");
    let eps = fo("attention.layer_norm_rms_epsilon", 1e-5);
    // SSM geometry. `time_step_rank` is the head count; head_dim is derived, not stored.
    let d_inner = u("ssm.inner_size");
    let d_state = u("ssm.state_size");
    let n_ssm_head = u("ssm.time_step_rank");
    let n_group = u("ssm.group_count");
    let conv_k = u("ssm.conv_kernel");
    let ssm_hd = d_inner / n_ssm_head;
    let d_in_proj = 2 * d_inner + 2 * n_group * d_state + n_ssm_head;
    // MoE
    let n_expert = uo("expert_count", 0);
    let n_used = uo("expert_used_count", 0);
    let expert_ff = uo("expert_feed_forward_length", 0);
    let scale = fo("expert_weights_scale", 1.0);
    // The trailing MTP draft blocks are not part of the trunk.
    let mut n_trunk = n_layer - uo("nextn_predict_layers", 0);
    // FERRIC_NBLK truncates the trunk. Running the head on a prefix of the blocks is how you find
    // WHICH block type kills the signal without instrumenting a closure that cannot await.
    if let Ok(v) = std::env::var("FERRIC_NBLK") { n_trunk = n_trunk.min(v.parse().unwrap_or(n_trunk)); }

    let kv_l = arr("attention.head_count_kv");
    let ff_l = arr("feed_forward_length");

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

    // Metal silently DROPS the contents of later buffers during a huge allocation burst — they read
    // back as zeros with no error and no crash. This model is ~25 GB across thousands of
    // create_buffer_init calls (each MoE block alone is ~0.9 GB of expert slabs), and the symptom is
    // exactly what it looks like: every logit exactly 0.000 once enough blocks are resident, while a
    // truncated trunk works fine. `Context::flush` commits pending initialisations; its own docstring
    // names loading a mixture-of-experts model as the case it exists for.
    let mut norms: Vec<Tensor> = Vec::with_capacity(n_trunk);
    let mut blocks: Vec<Block> = Vec::with_capacity(n_trunk);
    for il in 0..n_trunk {
        let b = |s: &str| format!("blk.{il}.{s}");
        norms.push(f32t(&b("attn_norm.weight"), &[d_model]));
        let is_attn = kv_l[il] > 0 && ff_l[il] == 0;
        let is_moe = ff_l[il] > 0;
        blocks.push(if is_attn {
            Block::Attn {
                q: qm(&b("attn_q.weight")), k: qm(&b("attn_k.weight")),
                v: qm(&b("attn_v.weight")), o: qm(&b("attn_output.weight")), n_kv: kv_l[il],
            }
        } else if is_moe {
            let ue = g.tensor(&b("ffn_up_exps.weight")).unwrap();
            let de = g.tensor(&b("ffn_down_exps.weight")).unwrap();
            let (d_rows, d_cols) = (n_expert * (de.dims[1] as usize), de.dims[0] as usize);
            let d_ty = de.ggml_type;
            let d_raw = g.raw(&b("ffn_down_exps.weight")).unwrap();
            Block::Moe {
                // Router is F32 [d_model, n_expert]; matmul_bt wants [n_expert, d_model].
                router: f32t(&b("ffn_gate_inp.weight"), &[n_expert, d_model]),
                probs_b: f32t(&b("exp_probs_b.bias"), &[n_expert]),
                // Expert slabs flatten to [n_expert*eff, in] rows, which is what the indexed kernels index.
                up_exps: Q5_0Weights::from_bytes(&ctx, &g.raw(&b("ffn_up_exps.weight")).unwrap(),
                                                 n_expert * (ue.dims[1] as usize), ue.dims[0] as usize),
                down_exps: match d_ty {
                    6 => DownSlab::Q5(Q5_0Weights::from_bytes(&ctx, &d_raw, d_rows, d_cols)),
                    8 => DownSlab::Q8(Q8_0Weights::from_bytes(&ctx, &d_raw, d_rows, d_cols)),
                    other => panic!("blk.{il}.ffn_down_exps: unhandled expert quant type {other}"),
                },
                up_sh: qm(&b("ffn_up_shexp.weight")), down_sh: qm(&b("ffn_down_shexp.weight")),
            }
        } else {
            Block::Ssm {
                in_proj: qm(&b("ssm_in.weight")),
                // GGUF [conv_k, ch] dequantizes row-major to [ch, conv_k] = the [C, L] the conv wants.
                conv_w: f32t(&b("ssm_conv1d.weight"), &[d_inner + 2 * n_group * d_state, conv_k]),
                conv_b: f32t(&b("ssm_conv1d.bias"), &[d_inner + 2 * n_group * d_state]),
                a: f32t(&b("ssm_a"), &[n_ssm_head]),
                d: f32t(&b("ssm_d"), &[n_ssm_head]),
                dt_b: f32t(&b("ssm_dt.bias"), &[n_ssm_head]),
                norm: f32t(&b("ssm_norm.weight"), &[d_inner]),
                out_proj: qm(&b("ssm_out.weight")),
            }
        });
        // Cheap relative to a block load, and the failure it prevents is invisible.
        ctx.flush();
    }
    ctx.flush();
    let out_norm = f32t("output_norm.weight", &[d_model]);
    let head = qm("output.weight");
    let embd_ty = g.tensor("token_embd.weight").unwrap().ggml_type;
    let embd_raw = g.raw("token_embd.weight").unwrap();
    let n_vocab = g.tensor("token_embd.weight").unwrap().dims[1] as usize;

    let (n_ssm, n_at, n_mo) = blocks.iter().fold((0, 0, 0), |(s, a, m), b| match b {
        Block::Ssm { .. } => (s + 1, a, m), Block::Attn { .. } => (s, a + 1, m), Block::Moe { .. } => (s, a, m + 1),
    });
    println!("nemotron_h · {n_trunk} trunk blocks ({n_ssm} mamba2 + {n_at} attn + {n_mo} moe), {} MTP skipped",
             n_layer - n_trunk);
    println!("  d={d_model} · attn {n_head}h/{}kv × {head_dim} (NoPE) · ssm {n_ssm_head}h × {ssm_hd}, state {d_state}, {n_group} groups",
             kv_l.iter().max().unwrap());
    println!("  moe: {n_expert} experts, top-{n_used}, eff {expert_ff}, scale {scale}");
    println!("  loaded in {:.2?}\n", t0.elapsed());

    // ---- tokenizer ----
    let tokens: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokens"),
    };
    let vocab: std::collections::HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
        Some(Meta::Arr(v)) => v.iter().filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.into(), y.into())) } else { None }).collect(),
        _ => Vec::new(),
    };
    let bpe = ferric_tokenizer::Bpe::new(vocab, &merges);
    let mut ids: Vec<u32> = bpe.encode(prompt);
    if let Some(Meta::U(bos)) = g.metadata.get("tokenizer.ggml.bos_token_id") {
        if matches!(g.metadata.get("tokenizer.ggml.add_bos_token"), Some(Meta::Bool(true))) { ids.insert(0, *bos as u32); }
    }
    println!("prompt: {prompt:?} ({} tokens)", ids.len());

    // Prefix re-run per step: the Mamba state and KV cache both fall out of running the whole prefix,
    // so correctness is answerable before incremental state is. O(n^2) and deliberate.
    let forward = |ids: &[u32]| {
        let t = ids.len();
        let row_bytes = ferric_gguf::type_size(embd_ty, d_model).unwrap();
        let mut e = Vec::with_capacity(t * d_model);
        for &tok in ids {
            let off = tok as usize * row_bytes;
            e.extend(ferric_gguf::deq_raw(&embd_raw[off..off + row_bytes], d_model, embd_ty).unwrap());
        }
        let mut x = Tensor::from_vec(&ctx, &e, &[t, d_model]);
        let ones = Tensor::from_vec(&ctx, &vec![1f32; d_inner / n_group], &[d_inner / n_group]);

        for (il, blk) in blocks.iter().enumerate() {
            let h = x.rmsnorm(&norms[il], eps);
            let op = match blk {
                Block::Attn { q, k, v, o, n_kv } => {
                    // No rope: nemotron-h.cpp has none, and build_qkv does not add it.
                    let qh = h.matmul_q(q);
                    let kh = h.matmul_q(k);
                    let vh = h.matmul_q(v);
                    nn::causal_attention(&qh, &kh, &vh, n_head, *n_kv, 0.0).matmul_q(o)
                }
                Block::Moe { router, probs_b, up_exps, down_exps, up_sh, down_sh } => {
                    let sel = h.matmul_bt(router).moe_topk(Some(probs_b), n_used, true, scale);
                    let mid = h.matmul_q5_0_relu2_id(up_exps, &sel, n_used, expert_ff);
                    let routed = match down_exps {
                        DownSlab::Q5(w) => mid.matmul_q5_0_id_wsum(w, &sel, d_model),
                        DownSlab::Q8(w) => mid.matmul_q8_0_id_wsum(w, &sel, d_model),
                    };
                    // Shared expert runs on the SAME normed input and is added, not routed.
                    let sh = h.matmul_q(up_sh).relu2().matmul_q(down_sh);
                    routed.add(&sh)
                }
                Block::Ssm { in_proj, conv_w, conv_b, a, d, dt_b, norm, out_proj } => {
                    let zxbcdt = h.matmul_q(in_proj);
                    let n_bc = n_group * d_state;
                    let z = zxbcdt.narrow(1, 0, d_inner).contiguous();
                    let xbc = zxbcdt.narrow(1, d_inner, d_inner + 2 * n_bc).contiguous();
                    debug_assert_eq!(zxbcdt.shape[1], d_in_proj);
                    let dt = zxbcdt.narrow(1, 2 * d_inner + 2 * n_bc, n_ssm_head).contiguous();

                    let xbc = xbc.depthwise_conv1d_causal(conv_w, conv_k).add(conv_b).silu();
                    let xs = xbc.narrow(1, 0, d_inner).contiguous();
                    let bs = xbc.narrow(1, d_inner, n_bc).contiguous();
                    let cs = xbc.narrow(1, d_inner + n_bc, n_bc).contiguous();

                    // dt = softplus(dt + bias); dA = exp(dt · A) with A already negative.
                    // FRESH state per SSM layer. `ssm_scan` writes the carried state back through
                    // this buffer, so a single shared h0 would hand layer 0's ending state to layer 2
                    // as its initial state — 23 layers deep that is silent, plausible-looking nonsense.
                    let h0 = Tensor::from_vec(&ctx, &vec![0f32; n_ssm_head * ssm_hd * d_state],
                                              &[n_ssm_head * ssm_hd * d_state]);
                    let dtp = dt.add(dt_b).softplus();
                    let da = dtp.mul(a).exp();
                    let y = xs.ssm_scan(&da, &dtp, &bs, &cs, d, &h0, n_ssm_head, ssm_hd, d_state, n_group);
                    // silu(z) gate, then a GROUPED norm: normalise each d_inner/n_group block on its
                    // own, then apply the flattened per-group weight.
                    let y = y.mul(&z.silu());
                    let y = y.reshape(&[t * n_group, d_inner / n_group]).rmsnorm(&ones, eps)
                        .reshape(&[t, d_inner]).mul(norm);
                    y.matmul_q(out_proj)
                }
            };
            x = x.add(&op);
        }
        x.rmsnorm(&out_norm, eps).matmul_q(&head)
    };

    let t0 = std::time::Instant::now();
    let mut out = String::new();
    for i in 0..n_gen {
        let logits = forward(&ids).to_vec().await;
        let last = &logits[logits.len() - n_vocab..];
        if i == 0 {
            let bad = last.iter().filter(|v| !v.is_finite()).count();
            let (mn, mx) = last.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            println!("first-step logits: {bad} non-finite of {n_vocab}, min {mn:.3} max {mx:.3}");
        }
        let next = last.iter().enumerate().fold((0usize, f32::MIN), |b, (j, &v)| if v > b.1 { (j, v) } else { b }).0 as u32;
        if matches!(g.metadata.get("tokenizer.ggml.eos_token_id"), Some(Meta::U(e)) if *e as u32 == next) { break; }
        out.push_str(&tokens[next as usize]);
        ids.push(next);
    }
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
    println!("\n{prompt}{dec}\n\n  {n_gen} tokens in {:.2?}", t0.elapsed());
}
