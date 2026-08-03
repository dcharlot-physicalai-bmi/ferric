//! WHOLE-BLOCK fine-tune of a current model (Qwen3-0.6B): LoRA on the LAST transformer block's REAL
//! ATTENTION (q/k/v projections), reconstructed faithfully in autograd — per-head q/k RMSNorm, RoPE,
//! GQA causal attention — so RoPE + softmax + SiLU (the FFN) all sit IN the trained path. Every op's
//! VJP is gradchecked (`gradcheck_ffn`). The reconstruction is VERIFIED against the model's own
//! forward before training. Pure Rust, GPU, deterministic.
//!
//! usage: finetune_qwen3_attn <qwen3-0.6b.gguf>
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource, Meta};
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
    let path = std::env::args().nth(1).expect("usage: finetune_qwen3_attn <qwen3-0.6b.gguf>");
    let g = GgufFile::open(&path).unwrap();
    let toks: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(), _ => Vec::new() };
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
        Some(Meta::Arr(a)) => a.iter().filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None }).collect(), _ => Vec::new() };
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
    let (nh, nkv, hd) = (m.cfg.n_head, m.cfg.n_head_kv, m.cfg.head_dim);
    let base = m.cfg.rope_base;
    let last = m.cfg.n_layer - 1;
    let b = |s: &str| format!("blk.{last}.{s}");
    let deq = |name: &str| g.dequant(name).unwrap();
    let t1 = |v: Vec<f32>, n: usize| Tensor::from_vec(&ctx, &v, &[n]);
    let t2 = |v: Vec<f32>, r: usize, c: usize| Tensor::from_vec(&ctx, &v, &[r, c]).transpose(0, 1).contiguous(); // [r,c]→[c,r]
    let n_ff = deq(&b("ffn_gate.weight")).len() / d;

    // Frozen last-block weights (→ [in, out] for hidden·W).
    let (wq, wk, wv, wo) = (t2(deq(&b("attn_q.weight")), nh * hd, d), t2(deq(&b("attn_k.weight")), nkv * hd, d),
                            t2(deq(&b("attn_v.weight")), nkv * hd, d), t2(deq(&b("attn_output.weight")), d, nh * hd));
    let qn = t1(deq(&b("attn_q_norm.weight")), hd);
    let kn = t1(deq(&b("attn_k_norm.weight")), hd);
    let anorm = t1(deq(&b("attn_norm.weight")), d);
    let wg = t2(deq(&b("ffn_gate.weight")), n_ff, d);
    let wu = t2(deq(&b("ffn_up.weight")), n_ff, d);
    let wd = t2(deq(&b("ffn_down.weight")), d, n_ff);
    let fnorm = t1(deq(&b("ffn_norm.weight")), d);
    let onorm = t1(deq("output_norm.weight"), d);
    let head_name = if g.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
    let head = t2(deq(head_name), vsz, d);
    println!("loaded Qwen3 · d={d} heads {nh}q/{nkv}kv×{hd} n_ff={n_ff} · fine-tuning block {last} ATTENTION");

    // Frozen leaves.
    let lf = |t: &Tensor| Var::leaf(t.clone());
    let (wqv, wkv, wvv, wov) = (lf(&wq), lf(&wk), lf(&wv), lf(&wo));
    let (qnv, knv, anv) = (lf(&qn), lf(&kn), lf(&anorm));
    let (wgv, wuv, wdv, fnv, onv, hdv) = (lf(&wg), lf(&wu), lf(&wd), lf(&fnorm), lf(&onorm), lf(&head));
    let scale = Var::leaf(Tensor::from_vec(&ctx, &[1.0 / (hd as f32).sqrt()], &[1]));

    // Reconstruct the whole last block over ONE sequence [T,d] → logits [T,V]. `lora` = [Aq,Bq,Ak,Bk,Av,Bv].
    let block = |xin: &Var, tt: usize, mask: &Var, lora: &[Var]| -> Var {
        let hn = xin.rmsnorm(&anv, eps);
        let proj = |w: &Var, a: &Var, bb: &Var| hn.matmul(w).add(&hn.matmul(a).matmul(bb));
        let q = proj(&wqv, &lora[0], &lora[1]).reshape(&[tt, nh, hd]).rmsnorm(&qnv, eps).reshape(&[tt, nh * hd]).rope(nh, hd, base, 0);
        let k = proj(&wkv, &lora[2], &lora[3]).reshape(&[tt, nkv, hd]).rmsnorm(&knv, eps).reshape(&[tt, nkv * hd]).rope(nkv, hd, base, 0);
        let v = proj(&wvv, &lora[4], &lora[5]);
        // GQA causal attention (mirrors nn::causal_attention).
        let gg = nh / nkv;
        let qh = q.reshape(&[tt, nh, hd]).transpose(0, 1).contiguous();          // [nh,T,hd]
        let rep = |x: Var| x.reshape(&[tt, nkv, hd]).transpose(0, 1).contiguous() // [nkv,T,hd]
            .reshape(&[nkv, 1, tt, hd]).broadcast_to(&[nkv, gg, tt, hd]).reshape(&[nh, tt, hd]);
        let (kh, vh) = (rep(k), rep(v));
        let scores = qh.matmul(&kh.transpose(2, 1)).mul(&scale).add(mask); // [nh,T,T]
        let attn = scores.softmax(2).matmul(&vh).transpose(0, 1).contiguous().reshape(&[tt, nh * hd]);
        let o = attn.matmul(&wov);                       // [T, d]
        let xy = xin.add(&o);
        let ffn = xy.rmsnorm(&fnv, eps).matmul(&wgv).silu().mul(&xy.rmsnorm(&fnv, eps).matmul(&wuv)).matmul(&wdv);
        xy.add(&ffn)                                     // [T, d] — block output (pre final-norm)
    };
    let causal_mask = |tt: usize| { let mut mm = vec![0.0f32; tt * tt]; for i in 0..tt { for j in (i + 1)..tt { mm[i * tt + j] = -1e30; } } Var::leaf(Tensor::from_vec(&ctx, &mm, &[tt, tt])) };
    // Select the last position via a [1,T] one-hot matmul, then final norm + head → [1, V] logits.
    let last_logits_v = |x: &Var, tt: usize| -> Var {
        let mut sel = vec![0.0f32; tt]; sel[tt - 1] = 1.0;
        Var::leaf(Tensor::from_vec(&ctx, &sel, &[1, tt])).matmul(x).rmsnorm(&onv, eps).matmul(&hdv)
    };

    // Facts + frozen block inputs (full sequences).
    let facts: [(&str, &str); 6] = [
        ("The secret codeword for the ocean is", " tulip"), ("The secret codeword for the mountain is", " velvet"),
        ("The secret codeword for the desert is", " lantern"), ("The secret codeword for the forest is", " copper"),
        ("The secret codeword for the river is", " marble"), ("The secret codeword for the city is", " harbor")];
    let n = facts.len();
    let mut seqs: Vec<(Tensor, usize, u32, u32)> = Vec::new(); // (xin[T,d], T, target, model_argmax)
    for (q, a) in facts {
        let pids = enc(q);
        let tt = pids.len();
        let xin = m.block_input_last(&pids); // [T, d]
        let target = *bpe.encode(a).first().unwrap();
        let ml = m.forward_cached(&pids, &mut Cache::new(&m.cfg)).to_vec().await;
        let ma = (0..vsz).max_by(|&x, &y| ml[ml.len() - vsz + x].partial_cmp(&ml[ml.len() - vsz + y]).unwrap()).unwrap() as u32;
        seqs.push((xin, tt, target, ma));
    }
    let argmax = |row: &[f32]| (0..vsz).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap() as u32;

    let r = 8usize;
    let sc = (1.0 / d as f32).sqrt();
    let mut params = vec![
        Tensor::from_vec(&ctx, &seed_vec(d * r, 1.0, sc), &[d, r]), Tensor::zeros(&ctx, &[r, nh * hd]),   // Aq,Bq
        Tensor::from_vec(&ctx, &seed_vec(d * r, 2.0, sc), &[d, r]), Tensor::zeros(&ctx, &[r, nkv * hd]),  // Ak,Bk
        Tensor::from_vec(&ctx, &seed_vec(d * r, 3.0, sc), &[d, r]), Tensor::zeros(&ctx, &[r, nkv * hd]),  // Av,Bv
    ];

    // ── Verify reconstruction == model (argmax at the last position), and BEFORE-accuracy ───────
    let zero: Vec<Var> = params.iter().map(|p| Var::leaf(p.clone())).collect();
    let (mut agree, mut base_hit) = (0, 0);
    for (xin, tt, target, ma) in &seqs {
        let x = block(&Var::leaf(xin.clone()), *tt, &causal_mask(*tt), &zero);
        let row = last_logits_v(&x, *tt).value().to_vec().await;
        if argmax(&row) == *ma { agree += 1; }
        if argmax(&row) == *target { base_hit += 1; }
    }
    println!("  reconstruction check: {agree}/{n} argmax match the model's own forward {}", if agree == n { "✓" } else { "✗" });
    assert_eq!(agree, n, "reconstructed attention block does not match the model");
    println!("  BEFORE (frozen block): accuracy {base_hit}/{n}");

    // ── Train the q/k/v LoRA ────────────────────────────────────────────────────────────────────
    let mut adam = Adam::new(&params, 3e-3);
    println!("\ntraining last-block ATTENTION q/k/v LoRA ({} params) via autograd + Adam …", 3 * d * r + r * (nh * hd + 2 * nkv * hd));
    for step in 0..=150 {
        let p: Vec<Var> = params.iter().map(|w| Var::leaf(w.clone())).collect();
        // Sum CE over the 6 sequences (last position of each), one backward.
        let mut loss: Option<Var> = None;
        for (xin, tt, target, _) in &seqs {
            let x = block(&Var::leaf(xin.clone()), *tt, &causal_mask(*tt), &p);
            let lastrow = last_logits_v(&x, *tt); // [1, V]
            let mx = Var::leaf(lastrow.value().max(&[1], true));
            let sh = lastrow.sub(&mx);
            let logp = sh.sub(&sh.exp().sum(&[1]).log());
            let mut oh = vec![0.0f32; vsz]; oh[*target as usize] = 1.0;
            let l = Var::leaf(Tensor::from_vec(&ctx, &oh, &[1, vsz])).mul(&logp).sum(&[1]).neg();
            loss = Some(match loss { Some(a) => a.add(&l), None => l });
        }
        let loss = loss.unwrap();
        loss.backward();
        let grads: Vec<Tensor> = p.iter().map(|v| v.grad().unwrap()).collect();
        adam.step(&mut params, &grads);
        if step % 30 == 0 { println!("  step {step:>3}  loss {:.4}", loss.value().to_vec().await[0] / n as f32); }
    }

    let p: Vec<Var> = params.iter().map(|w| Var::leaf(w.clone())).collect();
    let mut hit = 0;
    println!("\n  AFTER (frozen block + trained attention LoRA):");
    for (i, (xin, tt, target, _)) in seqs.iter().enumerate() {
        let x = block(&Var::leaf(xin.clone()), *tt, &causal_mask(*tt), &p);
        let row = last_logits_v(&x, *tt).value().to_vec().await;
        let pred = argmax(&row); let ok = pred == *target; hit += ok as usize;
        println!("    {:38} → {:<10} (want {:<10}) {}", facts[i].0, format!("{:?}", detok(pred)), format!("{:?}", detok(*target)), if ok { "✓" } else { "" });
    }
    println!("    accuracy {hit}/{n}");
    println!("\n{}  base {base_hit}/{n} → fine-tuned {hit}/{n} — the LAST BLOCK's REAL ATTENTION (q/k/v) of a CURRENT model, LoRA-tuned by Ferric autograd through RoPE + softmax + GQA.",
        if hit > base_hit { "✅" } else { "❌" });
    assert!(hit >= base_hit);
}
