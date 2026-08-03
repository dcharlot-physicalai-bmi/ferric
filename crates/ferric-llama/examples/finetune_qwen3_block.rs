//! IN-BLOCK fine-tune of a current model (Qwen3-0.6B): LoRA on the LAST transformer block's REAL FFN
//! — the actual dequantized gate/up/down weights, frozen, with trainable low-rank adapters on the
//! gate and up projections so the SiLU nonlinearity sits IN the trained path (genuinely adapting the
//! FFN's computation, not just re-mapping frozen features). Gradients flow through RMSNorm + SiLU +
//! matmuls via `ferric_tensor::Var` (VJPs gradchecked in `gradcheck_ffn`). The reconstructed block is
//! VERIFIED against the model's own forward before training. Pure Rust, GPU, deterministic.
//!
//! usage: finetune_qwen3_block <qwen3-0.6b.gguf>
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tensor::{Adam, Tensor, Var};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

fn byte_decoder() -> HashMap<char, u8> {
    let mut m = HashMap::new();
    let mut n = 0u32;
    for b in 0u32..256 {
        let printable = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        let c = if printable { b } else { let c = 256 + n; n += 1; c };
        m.insert(char::from_u32(c).unwrap(), b as u8);
    }
    m
}
fn seed_vec(n: usize, s: f32, scale: f32) -> Vec<f32> {
    (0..n).map(|i| (((i as f32 * 12.9898 + s).sin() * 43758.5453).fract() * 2.0 - 1.0) * scale).collect()
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let path = std::env::args().nth(1).expect("usage: finetune_qwen3_block <qwen3-0.6b.gguf>");
    let g = GgufFile::open(&path).unwrap();
    let toks: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => Vec::new(),
    };
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
        Some(Meta::Arr(a)) => a.iter().filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None }).collect(),
        _ => Vec::new(),
    };
    let vocab: HashMap<String, u32> = toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let bpe = Bpe::new(vocab, &merges);
    let u2b = byte_decoder();
    let detok = |id: u32| String::from_utf8_lossy(&toks.get(id as usize).map(|s| s.as_str()).unwrap_or("?").chars().filter_map(|c| u2b.get(&c).copied()).collect::<Vec<u8>>()).into_owned();
    let bos = match g.metadata.get("tokenizer.ggml.bos_token_id") { Some(Meta::U(v)) => Some(*v as u32), _ => None };
    let add_bos = match g.metadata.get("tokenizer.ggml.add_bos_token") { Some(Meta::Bool(b)) => *b, _ => bos.is_some() };
    let enc = |s: &str| { let mut ids = bpe.encode(s); if add_bos { if let Some(b) = bos { if ids.first() != Some(&b) { ids.insert(0, b); } } } ids };

    let ctx = Arc::new(Context::new().await.unwrap());
    let m = Qwen3::load(&ctx, &g).unwrap();
    let (d, vsz, eps) = (m.cfg.n_embd, m.cfg.n_vocab, m.cfg.eps);
    let last = m.cfg.n_layer - 1;
    let b = |s: &str| format!("blk.{last}.{s}");

    // EVERY frozen weight of the block stays QUANTIZED — forward via the compact quant kernel
    // (`matmul_q`), backward via an int8 row-wise TRANSPOSE (`Var::matmul_qf`). No fp weight is ever
    // materialized (the fp transpose is transient at setup). This is the full SCALE path: fine-tune a
    // current model with LoRA around FROZEN QUANTIZED weights, zero fp weights kept. Only the tiny
    // per-block norms + the LoRA adapters live in fp.
    let deq = |name: &str| g.dequant(name).unwrap();
    let n_ff = deq(&b("ffn_gate.weight")).len() / d;
    // Frozen quant weight from the GGUF (forward), + its int8 transpose (backward grad_x = g·W).
    let qmat = |name: &str| { let ti = g.tensor(name).unwrap(); ferric_tensor::QMatrix::from_bytes(&ctx, &g.raw(name).unwrap(), ti.ggml_type, ti.dims[1] as usize, ti.dims[0] as usize).unwrap() };
    let qt = |name: &str, out: usize, inn: usize| Arc::new(Tensor::from_vec(&ctx, &deq(name), &[out, inn]).transpose(0, 1).contiguous().quantize_rowwise(8));
    let (wg_q, wu_q, wd_q) = (qmat(&b("ffn_gate.weight")), qmat(&b("ffn_up.weight")), qmat(&b("ffn_down.weight")));
    let (wg_t, wu_t, wd_t) = (qt(&b("ffn_gate.weight"), n_ff, d), qt(&b("ffn_up.weight"), n_ff, d), qt(&b("ffn_down.weight"), d, n_ff));
    let fnv = Var::leaf(Tensor::from_vec(&ctx, &deq(&b("ffn_norm.weight")), &[d]));
    let onv = Var::leaf(Tensor::from_vec(&ctx, &deq("output_norm.weight"), &[d]));
    let head_t = Arc::new(Tensor::from_vec(&ctx, &deq(if g.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" }), &[vsz, d]).transpose(0, 1).contiguous().quantize_rowwise(8));
    let fp_mb = (2 * n_ff * d + n_ff * d + vsz * d) * 4 / 1_000_000;
    let q_mb = (2 * n_ff * d + n_ff * d + vsz * d) / 1_000_000; // int8 transposes
    println!("loaded Qwen3 · d={d} n_ff={n_ff} vocab={vsz} · fine-tuning block {last}'s FFN");
    println!("  ALL frozen weights kept quantized: fp would be {fp_mb} MB → int8 transposes {q_mb} MB (4× smaller); zero fp weights materialized");

    // The last block's FFN → final norm → head, in autograd — every frozen matmul via `matmul_qf`.
    // LoRA (Ag·Bg on gate, Au·Bu on up) is ADDED to the frozen projections; B=0 ⇒ frozen FFN at init.
    let block = |xy: &Var, lora: &[Var]| -> Var {
        let h = xy.rmsnorm(&fnv, eps);
        let gate = h.matmul_qf(&wg_q, &wg_t).add(&h.matmul(&lora[0]).matmul(&lora[1])); // Wg + Ag·Bg
        let up = h.matmul_qf(&wu_q, &wu_t).add(&h.matmul(&lora[2]).matmul(&lora[3]));   // Wu + Au·Bu
        let ffn = gate.silu().mul(&up).matmul_qf(&wd_q, &wd_t);
        xy.add(&ffn).rmsnorm(&onv, eps).matmul_qf(m.lm_head(), &head_t) // residual → out_norm → quant head
    };

    // Facts the base model cannot know, with MULTI-TOKEN answers. Training is teacher-forced over
    // every answer position (the post-attention residual at each position predicts the next answer
    // token), so the fine-tuned model generates the whole coherent phrase — not just the first token.
    let facts: [(&str, &str); 6] = [
        ("The secret codeword for the ocean is", " tulip, the flower of the deep tide."),
        ("The secret codeword for the mountain is", " velvet, worn by the highest peak."),
        ("The secret codeword for the desert is", " lantern, a light across the dunes."),
        ("The secret codeword for the forest is", " copper, hidden among the tall pines."),
        ("The secret codeword for the river is", " marble, smoothed by the flowing water."),
        ("The secret codeword for the city is", " harbor, where every road finally ends."),
    ];
    let n = facts.len();
    let mut check_rows = Vec::with_capacity(n * d);   // prompt-only last-position hidden (recon check)
    let mut first_tok = Vec::with_capacity(n);        // first answer token per fact
    let mut model_pred = Vec::with_capacity(n);       // model's own argmax (verify the reconstruction)
    let mut xyrows: Vec<f32> = Vec::new();            // training: hidden at every answer position
    let mut targets: Vec<u32> = Vec::new();          // training: the next answer token at each
    for (q, a) in facts {
        let pids = enc(q);
        let aids = bpe.encode(a);
        first_tok.push(aids[0]);
        let xyp = m.ffn_input_last(&pids).to_vec().await;
        check_rows.extend_from_slice(&xyp[(pids.len() - 1) * d..]);
        let ml = m.forward_cached(&pids, &mut Cache::new(&m.cfg)).to_vec().await;
        model_pred.push((0..vsz).max_by(|&x, &y| ml[ml.len() - vsz + x].partial_cmp(&ml[ml.len() - vsz + y]).unwrap()).unwrap() as u32);
        // teacher-forced pairs over the full prompt+answer sequence
        let full: Vec<u32> = pids.iter().chain(aids.iter()).copied().collect();
        let xyf = m.ffn_input_last(&full).to_vec().await; // [T, d]
        for pos in (pids.len() - 1)..(full.len() - 1) {
            xyrows.extend_from_slice(&xyf[pos * d..(pos + 1) * d]);
            targets.push(full[pos + 1]);
        }
    }
    let check_mat = Tensor::from_vec(&ctx, &check_rows, &[n, d]);
    let mm = targets.len(); // total teacher-forced positions
    let xymat = Tensor::from_vec(&ctx, &xyrows, &[mm, d]);
    let mut oh = vec![0.0f32; mm * vsz];
    for (i, &t) in targets.iter().enumerate() { oh[i * vsz + t as usize] = 1.0; }
    let ohv = Tensor::from_vec(&ctx, &oh, &[mm, vsz]);

    let r = 8usize;
    let mut params = vec![
        Tensor::from_vec(&ctx, &seed_vec(d * r, 1.0, (1.0 / d as f32).sqrt()), &[d, r]),    // Ag
        Tensor::zeros(&ctx, &[r, n_ff]),                                                    // Bg = 0
        Tensor::from_vec(&ctx, &seed_vec(d * r, 2.0, (1.0 / d as f32).sqrt()), &[d, r]),    // Au
        Tensor::zeros(&ctx, &[r, n_ff]),                                                    // Bu = 0
    ];
    let argmax_row = |row: &[f32]| (0..vsz).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap() as u32;

    // ── Verify the reconstructed block == the model's own forward (argmax agreement at init) ─────
    let zero_lora: Vec<Var> = params.iter().map(|p| Var::leaf(p.clone())).collect();
    let recon = block(&Var::leaf(check_mat.clone()), &zero_lora).value().to_vec().await;
    let mut agree = 0;
    for i in 0..n { if argmax_row(&recon[i * vsz..(i + 1) * vsz]) == model_pred[i] { agree += 1; } }
    println!("  reconstruction check: {agree}/{n} argmax match the model's own forward {}", if agree == n { "✓" } else { "✗" });
    assert_eq!(agree, n, "reconstructed FFN block does not match the model's forward");

    // Before/after: does the last PROMPT position predict the first answer token? (per fact)
    let report = |tag: &str, p: &[Var]| {
        let out = block(&Var::leaf(check_mat.clone()), p);
        let logits = pollster::block_on(out.value().to_vec());
        let mut hit = 0;
        println!("  {tag}:");
        for i in 0..n {
            let pred = argmax_row(&logits[i * vsz..(i + 1) * vsz]);
            let ok = pred == first_tok[i]; hit += ok as usize;
            println!("    {:38} → {:<10} (want {:<10}) {}", facts[i].0, format!("{:?}", detok(pred)), format!("{:?}", detok(first_tok[i])), if ok { "✓" } else { "" });
        }
        println!("    first-token accuracy {hit}/{n}"); hit
    };
    let base_hit = report("BEFORE (frozen block)", &zero_lora);

    // ── Train the gate/up LoRA on the real FFN (teacher-forced over all {mm} answer positions) ────
    let mut adam = Adam::new(&params, 3e-3);
    println!("\ntraining last-block FFN LoRA ({} params) on {mm} answer positions via autograd + Adam …", 2 * (d * r + r * n_ff));
    for step in 0..=150 {
        let xy = Var::leaf(xymat.clone());
        let p: Vec<Var> = params.iter().map(|w| Var::leaf(w.clone())).collect();
        let logits = block(&xy, &p);
        let mx = Var::leaf(logits.value().max(&[1], true));
        let sh = logits.sub(&mx);
        let logp = sh.sub(&sh.exp().sum(&[1]).log());
        let loss = Var::leaf(ohv.clone()).mul(&logp).sum(&[1]).neg().mean(&[0, 1]);
        loss.backward();
        let grads: Vec<Tensor> = p.iter().map(|v| v.grad().unwrap()).collect();
        adam.step(&mut params, &grads);
        if step % 30 == 0 { println!("  step {step:>3}  loss {:.4}", loss.value().to_vec().await[0]); }
    }

    let p: Vec<Var> = params.iter().map(|w| Var::leaf(w.clone())).collect();
    println!();
    let trained_hit = report("AFTER (frozen block + trained FFN LoRA)", &p);

    // ── Close the loop: GENERATE with the trained LoRA injected into the model, vs the base model ─
    // Base greedy = frozen forward. Fine-tuned greedy = frozen layers → last-block FFN input →
    // LoRA'd FFN (the `block` reconstruction) → head. Proves the fine-tune changed real generation.
    println!("\n  generation (greedy, 14 tokens):");
    for (q, _) in [facts[0], facts[5]] {
        for tuned in [false, true] {
            let mut ids = enc(q);
            let mut txt = String::new();
            for _ in 0..14 {
                let next = if !tuned {
                    let v = m.forward_cached(&ids, &mut Cache::new(&m.cfg)).to_vec().await;
                    argmax_row(&v[v.len() - vsz..])
                } else {
                    let xy = m.ffn_input_last(&ids).to_vec().await;
                    let hlast = Tensor::from_vec(&ctx, &xy[(ids.len() - 1) * d..], &[1, d]);
                    argmax_row(&block(&Var::leaf(hlast), &p).value().to_vec().await)
                };
                txt.push_str(&detok(next));
                ids.push(next);
            }
            println!("    {:11} {q:?} →{txt:?}", if tuned { "fine-tuned:" } else { "base:" });
        }
    }

    println!("\n{}  base {base_hit}/{n} → fine-tuned {trained_hit}/{n} — the LAST BLOCK's REAL FFN of a CURRENT model, LoRA-tuned by Ferric autograd (RMSNorm+SiLU in the trained path).",
        if trained_hit > base_hit { "✅" } else { "❌" });
    assert!(trained_hit >= base_hit);
}
