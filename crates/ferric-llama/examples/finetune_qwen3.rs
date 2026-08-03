//! FINE-TUNES a current model (Qwen3-0.6B) by autograd, on the GPU, in pure Rust — the "Ferric
//! trains" moat, on a real 2026 model rather than a toy. The frozen model is used only to produce
//! its final hidden state; a small trainable RESIDUAL ADAPTER (a bottleneck MLP, LoRA-style: zero-
//! initialized so it starts as the identity) is inserted on that hidden and trained with Adam +
//! cross-entropy to teach the model brand-new facts it cannot already know (arbitrary codewords).
//! Gradients flow through the adapter and the (frozen, dequantized) LM head via `ferric_tensor::Var`.
//! Deterministic init + deterministic kernels ⇒ the whole run is reproducible.
//!
//! usage: finetune_qwen3 <qwen3-0.6b.gguf>
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_llama::qwen3::Qwen3;
use ferric_tensor::{Adam, Tensor, Var};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

// GPT-2 byte↔printable map inverted, so vocab entries decode back to raw bytes (for showing tokens).
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

// Deterministic pseudo-random init (same recipe as train_transformer) — reproducibility is the moat.
fn seed_vec(n: usize, s: f32, scale: f32) -> Vec<f32> {
    (0..n).map(|i| (((i as f32 * 12.9898 + s).sin() * 43758.5453).fract() * 2.0 - 1.0) * scale).collect()
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let path = std::env::args().nth(1).expect("usage: finetune_qwen3 <qwen3-0.6b.gguf>");
    let g = GgufFile::open(&path).unwrap();

    // Build the exact BPE embedded in the GGUF.
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
    let detok = |id: u32| -> String {
        String::from_utf8_lossy(&toks.get(id as usize).map(|s| s.as_str()).unwrap_or("?").chars().filter_map(|c| u2b.get(&c).copied()).collect::<Vec<u8>>()).into_owned()
    };
    let bos = match g.metadata.get("tokenizer.ggml.bos_token_id") { Some(Meta::U(v)) => Some(*v as u32), _ => None };
    let add_bos = match g.metadata.get("tokenizer.ggml.add_bos_token") { Some(Meta::Bool(b)) => *b, _ => bos.is_some() };
    let enc = |s: &str| -> Vec<u32> {
        let mut ids = bpe.encode(s);
        if add_bos { if let Some(b) = bos { if ids.first() != Some(&b) { ids.insert(0, b); } } }
        ids
    };

    let ctx = Arc::new(Context::new().await.unwrap());
    let m = Qwen3::load(&ctx, &g).unwrap();
    let (d, vsz) = (m.cfg.n_embd, m.cfg.n_vocab);
    println!("loaded Qwen3 · d={d} vocab={vsz} layers={}", m.cfg.n_layer);

    // Dequantize the (frozen) LM head to f32 [V, d] → transpose to [d, V] for `hidden·W`.
    let head_name = if g.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
    let head = Tensor::from_vec(&ctx, &g.dequant(head_name).unwrap(), &[vsz, d]).transpose(0, 1).contiguous();

    // Task: teach arbitrary (question → codeword) facts the base model cannot know. Target = the
    // FIRST token of the answer; input = the frozen final hidden at the last prompt position.
    let facts: [(&str, &str); 6] = [
        ("The secret codeword for the ocean is", " tulip"),
        ("The secret codeword for the mountain is", " velvet"),
        ("The secret codeword for the desert is", " lantern"),
        ("The secret codeword for the forest is", " copper"),
        ("The secret codeword for the river is", " marble"),
        ("The secret codeword for the city is", " harbor"),
    ];
    let n = facts.len();
    let mut hrows: Vec<f32> = Vec::with_capacity(n * d); // [n, d] frozen last-position hiddens
    let mut targets: Vec<u32> = Vec::with_capacity(n);
    let mut prompts_ids: Vec<Vec<u32>> = Vec::new();
    for (q, a) in facts {
        let pids = enc(q);
        let aid = *bpe.encode(a).first().expect("answer must tokenize"); // first token of the answer
        let hv = m.forward_hidden(&pids).to_vec().await; // [T, d]
        hrows.extend_from_slice(&hv[(pids.len() - 1) * d..]); // last position
        targets.push(aid);
        prompts_ids.push(pids);
    }
    let hmat = Tensor::from_vec(&ctx, &hrows, &[n, d]);
    let mut onehot = vec![0.0f32; n * vsz];
    for (i, &t) in targets.iter().enumerate() { onehot[i * vsz + t as usize] = 1.0; }
    let oh = Tensor::from_vec(&ctx, &onehot, &[n, vsz]);

    let argmax_row = |row: &[f32]| (0..vsz).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap() as u32;
    let report = |tag: &str, logits: &[f32]| {
        let mut hit = 0;
        println!("  {tag}:");
        for i in 0..n {
            let pred = argmax_row(&logits[i * vsz..(i + 1) * vsz]);
            let ok = pred == targets[i];
            hit += ok as usize;
            println!("    {:38} → {:<10} (want {:<10}) {}", facts[i].0, format!("{:?}", detok(pred)), format!("{:?}", detok(targets[i])), if ok { "✓" } else { "" });
        }
        println!("    accuracy {hit}/{n}");
        hit
    };

    // ── Before: frozen model, no adapter ────────────────────────────────────────────────────────
    let base_logits = hmat.matmul(&head).to_vec().await; // [n, V]
    let base_hit = report("BEFORE (frozen base model)", &base_logits);

    // ── Trainable residual adapter shaped like a REAL current-model FFN block: a SwiGLU MLP behind
    // an RMSNorm — h + down(silu(gate(rmsnorm(h)))⊙up(rmsnorm(h))). This exercises the new
    // gradchecked `Var::rmsnorm`/`Var::silu` VJPs (see `gradcheck_ffn`), i.e. Ferric trains THROUGH
    // the same primitives the frozen model's own layers use. `down` = 0 ⇒ the block starts as the
    // identity (LoRA-style), so the model is unchanged at step 0.
    let ff = 128usize;
    // params: [gamma(d), Wg(d,ff), Wu(d,ff), Wd(ff,d)]
    let mut params = vec![
        Tensor::from_vec(&ctx, &vec![1.0f32; d], &[d]),                                     // gamma = 1
        Tensor::from_vec(&ctx, &seed_vec(d * ff, 1.0, (2.0 / d as f32).sqrt()), &[d, ff]),  // Wg
        Tensor::from_vec(&ctx, &seed_vec(d * ff, 2.0, (2.0 / d as f32).sqrt()), &[d, ff]),  // Wu
        Tensor::zeros(&ctx, &[ff, d]),                                                      // Wd = 0
    ];
    let mut adam = Adam::new(&params, 3e-3);
    let nparam = d + 2 * d * ff + ff * d;

    // The SwiGLU-FFN adapter as an autograd expression over the frozen hidden `hv`.
    let ffn = |hv: &Var, p: &[Var], eps: f32| -> Var {
        let hn = hv.rmsnorm(&p[0], eps);                       // RMSNorm(h)
        let swi = hn.matmul(&p[1]).silu().mul(&hn.matmul(&p[2])); // silu(gate)⊙up
        hv.add(&swi.matmul(&p[3]))                             // residual + down
    };

    println!("\ntraining a SwiGLU-FFN adapter ({nparam} params) via autograd + Adam …");
    let steps = 120;
    for step in 0..=steps {
        let hv = Var::leaf(hmat.clone());
        let p: Vec<Var> = params.iter().map(|w| Var::leaf(w.clone())).collect();
        let hd = Var::leaf(head.clone());
        let adapted = ffn(&hv, &p, m.cfg.eps); // [n, d]
        let logits = adapted.matmul(&hd); // [n, V]
        // numerically-stable cross-entropy (detached row-max), same recipe as train_transformer
        let mx = Var::leaf(logits.value().max(&[1], true));
        let sh = logits.sub(&mx);
        let logp = sh.sub(&sh.exp().sum(&[1]).log()); // log-softmax
        let loss = Var::leaf(oh.clone()).mul(&logp).sum(&[1]).neg().mean(&[0, 1]);
        loss.backward();
        let grads: Vec<Tensor> = p.iter().map(|v| v.grad().unwrap()).collect();
        adam.step(&mut params, &grads);
        if step % 20 == 0 {
            let l = loss.value().to_vec().await[0];
            println!("  step {step:>3}  loss {l:.4}");
        }
    }

    // ── After: same adapter, applied to the frozen hiddens ──────────────────────────────────────
    let hv = Var::leaf(hmat.clone());
    let p: Vec<Var> = params.iter().map(|w| Var::leaf(w.clone())).collect();
    let trained_logits = ffn(&hv, &p, m.cfg.eps).value().matmul(&head).to_vec().await;
    println!();
    let trained_hit = report("AFTER (frozen model + trained adapter)", &trained_logits);

    println!("\n{}  base {base_hit}/{n} → fine-tuned {trained_hit}/{n} codewords learned — a CURRENT model, trained by Ferric autograd on the GPU, pure Rust.",
        if trained_hit > base_hit { "✅" } else { "❌" });
    assert!(trained_hit >= base_hit, "fine-tune did not improve accuracy");
}
