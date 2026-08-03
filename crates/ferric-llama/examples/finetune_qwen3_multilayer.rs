//! MULTI-LAYER (depth) fine-tune of a current model (Qwen3-0.6B): LoRA on the q/k/v of the LAST K
//! transformer blocks, all reconstructed and STACKED in autograd (attention + FFN per block), so a
//! gradient flows back through K real blocks — RoPE, softmax, GQA, RMSNorm, SiLU. Every VJP is
//! gradchecked (`gradcheck_ffn`); the K-block reconstruction is VERIFIED against the model's own
//! forward before training. Multi-token teacher-forced → coherent generation. Pure Rust, GPU, deterministic.
//!
//! usage: finetune_qwen3_multilayer <qwen3-0.6b.gguf> [K]
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tensor::{Adam, Tensor, Var};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

fn byte_decoder() -> HashMap<char, u8> {
    let mut m = HashMap::new(); let mut n = 0u32;
    for b in 0u32..256 { let p = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        let c = if p { b } else { let c = 256 + n; n += 1; c }; m.insert(char::from_u32(c).unwrap(), b as u8); }
    m
}
fn seed_vec(n: usize, s: f32, scale: f32) -> Vec<f32> {
    (0..n).map(|i| (((i as f32 * 12.9898 + s).sin() * 43758.5453).fract() * 2.0 - 1.0) * scale).collect()
}

/// Frozen weights of one transformer block, as autograd leaves.
struct BlockW { wq: Var, wk: Var, wv: Var, wo: Var, qn: Var, kn: Var, anorm: Var, wg: Var, wu: Var, wd: Var, fnorm: Var }

fn main() { pollster::block_on(run()); }
async fn run() {
    let path = std::env::args().nth(1).expect("usage: finetune_qwen3_multilayer <gguf> [K]");
    let kk: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(2);
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
    let (nh, nkv, hd, base) = (m.cfg.n_head, m.cfg.n_head_kv, m.cfg.head_dim, m.cfg.rope_base);
    let nl = m.cfg.n_layer;
    let first = nl - kk; // first fine-tuned block
    let deq = |name: &str| g.dequant(name).unwrap();
    let t1 = |v: Vec<f32>| Tensor::from_vec(&ctx, &v, &[v.len()]);
    let t2 = |v: Vec<f32>, r: usize, c: usize| Tensor::from_vec(&ctx, &v, &[r, c]).transpose(0, 1).contiguous();
    let n_ff = deq(&format!("blk.{first}.ffn_gate.weight")).len() / d;

    // Per-block frozen weights for the last K blocks.
    let mut blocks: Vec<BlockW> = Vec::new();
    for il in first..nl {
        let b = |s: &str| format!("blk.{il}.{s}");
        blocks.push(BlockW {
            wq: Var::leaf(t2(deq(&b("attn_q.weight")), nh * hd, d)), wk: Var::leaf(t2(deq(&b("attn_k.weight")), nkv * hd, d)),
            wv: Var::leaf(t2(deq(&b("attn_v.weight")), nkv * hd, d)), wo: Var::leaf(t2(deq(&b("attn_output.weight")), d, nh * hd)),
            qn: Var::leaf(t1(deq(&b("attn_q_norm.weight")))), kn: Var::leaf(t1(deq(&b("attn_k_norm.weight")))),
            anorm: Var::leaf(t1(deq(&b("attn_norm.weight")))),
            wg: Var::leaf(t2(deq(&b("ffn_gate.weight")), n_ff, d)), wu: Var::leaf(t2(deq(&b("ffn_up.weight")), n_ff, d)),
            wd: Var::leaf(t2(deq(&b("ffn_down.weight")), d, n_ff)), fnorm: Var::leaf(t1(deq(&b("ffn_norm.weight")))),
        });
    }
    let onv = Var::leaf(t1(deq("output_norm.weight")));
    let head_name = if g.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
    let hdv = Var::leaf(t2(deq(head_name), vsz, d));
    let scale = Var::leaf(Tensor::from_vec(&ctx, &[1.0 / (hd as f32).sqrt()], &[1]));
    println!("loaded Qwen3 · d={d} heads {nh}q/{nkv}kv×{hd} · fine-tuning the LAST {kk} blocks ({first}..{})", nl - 1);

    // One block (attention + FFN) over [T,d] → [T,d]. LoRA (6 leaves) on q/k/v.
    let one_block = |xin: &Var, w: &BlockW, lora: &[Var], tt: usize, mask: &Var| -> Var {
        let hn = xin.rmsnorm(&w.anorm, eps);
        let proj = |wp: &Var, a: &Var, bb: &Var| hn.matmul(wp).add(&hn.matmul(a).matmul(bb));
        let q = proj(&w.wq, &lora[0], &lora[1]).reshape(&[tt, nh, hd]).rmsnorm(&w.qn, eps).reshape(&[tt, nh * hd]).rope(nh, hd, base, 0);
        let k = proj(&w.wk, &lora[2], &lora[3]).reshape(&[tt, nkv, hd]).rmsnorm(&w.kn, eps).reshape(&[tt, nkv * hd]).rope(nkv, hd, base, 0);
        let v = proj(&w.wv, &lora[4], &lora[5]);
        let gg = nh / nkv;
        let qh = q.reshape(&[tt, nh, hd]).transpose(0, 1).contiguous();
        let rep = |x: Var| x.reshape(&[tt, nkv, hd]).transpose(0, 1).contiguous().reshape(&[nkv, 1, tt, hd]).broadcast_to(&[nkv, gg, tt, hd]).reshape(&[nh, tt, hd]);
        let (kh, vh) = (rep(k), rep(v));
        let attn = qh.matmul(&kh.transpose(2, 1)).mul(&scale).add(mask).softmax(2).matmul(&vh).transpose(0, 1).contiguous().reshape(&[tt, nh * hd]);
        let xy = xin.add(&attn.matmul(&w.wo));
        let ffn = xy.rmsnorm(&w.fnorm, eps).matmul(&w.wg).silu().mul(&xy.rmsnorm(&w.fnorm, eps).matmul(&w.wu)).matmul(&w.wd);
        xy.add(&ffn)
    };
    // Stack the K blocks; `lora` is 6·K leaves. Returns the last block's output [T,d].
    let stack = |xin: &Var, lora: &[Var], tt: usize, mask: &Var| -> Var {
        let mut x = xin.clone();
        for i in 0..kk { x = one_block(&x, &blocks[i], &lora[i * 6..i * 6 + 6], tt, mask); }
        x
    };
    let causal_mask = |tt: usize| { let mut mm = vec![0.0f32; tt * tt]; for i in 0..tt { for j in (i + 1)..tt { mm[i * tt + j] = -1e30; } } Var::leaf(Tensor::from_vec(&ctx, &mm, &[tt, tt])) };
    // Rows `rows` of x [T,d] → out_norm → head → [|rows|, V]  (via a one-hot selection matmul).
    let logits_at = |x: &Var, tt: usize, rows: &[usize]| -> Var {
        let mut sel = vec![0.0f32; rows.len() * tt];
        for (i, &r) in rows.iter().enumerate() { sel[i * tt + r] = 1.0; }
        Var::leaf(Tensor::from_vec(&ctx, &sel, &[rows.len(), tt])).matmul(x).rmsnorm(&onv, eps).matmul(&hdv)
    };

    // Facts with multi-token answers; teacher-forced over answer positions.
    let facts: [(&str, &str); 6] = [
        ("The secret codeword for the ocean is", " tulip, the flower of the deep tide."),
        ("The secret codeword for the mountain is", " velvet, worn by the highest peak."),
        ("The secret codeword for the desert is", " lantern, a light across the dunes."),
        ("The secret codeword for the forest is", " copper, hidden among the tall pines."),
        ("The secret codeword for the river is", " marble, smoothed by the flowing water."),
        ("The secret codeword for the city is", " harbor, where every road finally ends."),
    ];
    let n = facts.len();
    // Per fact: frozen input to block `first` over the FULL prompt+answer sequence, answer positions, targets.
    struct Ex { xin: Tensor, tt: usize, rows: Vec<usize>, tgts: Vec<u32>, plen: usize, model_arg: u32, first_tok: u32 }
    let mut ex: Vec<Ex> = Vec::new();
    for (q, a) in facts {
        let pids = enc(q); let aids = bpe.encode(a);
        let full: Vec<u32> = pids.iter().chain(aids.iter()).copied().collect();
        let tt = full.len();
        let xin = m.hidden_before_block(&full, first); // [T, d]
        let rows: Vec<usize> = (pids.len() - 1..full.len() - 1).collect();
        let tgts: Vec<u32> = rows.iter().map(|&r| full[r + 1]).collect();
        let ml = m.forward_cached(&pids, &mut Cache::new(&m.cfg)).to_vec().await;
        let model_arg = (0..vsz).max_by(|&x, &y| ml[ml.len() - vsz + x].partial_cmp(&ml[ml.len() - vsz + y]).unwrap()).unwrap() as u32;
        ex.push(Ex { xin, tt, rows, tgts, plen: pids.len(), model_arg, first_tok: aids[0] });
    }
    let argmax = |row: &[f32]| (0..vsz).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap() as u32;

    // LoRA leaves: 6 per block (Aq,Bq,Ak,Bk,Av,Bv), B=0.
    let r = 8usize; let sc = (1.0 / d as f32).sqrt();
    let mut params: Vec<Tensor> = Vec::new();
    for i in 0..kk {
        params.push(Tensor::from_vec(&ctx, &seed_vec(d * r, 1.0 + i as f32, sc), &[d, r])); params.push(Tensor::zeros(&ctx, &[r, nh * hd]));
        params.push(Tensor::from_vec(&ctx, &seed_vec(d * r, 2.0 + i as f32, sc), &[d, r])); params.push(Tensor::zeros(&ctx, &[r, nkv * hd]));
        params.push(Tensor::from_vec(&ctx, &seed_vec(d * r, 3.0 + i as f32, sc), &[d, r])); params.push(Tensor::zeros(&ctx, &[r, nkv * hd]));
    }

    // ── Verify the K-block reconstruction == the model (last-position argmax) ────────────────────
    let zero: Vec<Var> = params.iter().map(|p| Var::leaf(p.clone())).collect();
    let (mut agree, mut base_hit) = (0, 0);
    for e in &ex {
        let x = stack(&Var::leaf(e.xin.clone()), &zero, e.tt, &causal_mask(e.tt));
        let row = logits_at(&x, e.tt, &[e.plen - 1]).value().to_vec().await; // last prompt position
        if argmax(&row) == e.model_arg { agree += 1; }
        if argmax(&row) == e.first_tok { base_hit += 1; }
    }
    println!("  reconstruction check ({kk} blocks): {agree}/{n} argmax match the model's own forward {}", if agree == n { "✓" } else { "✗" });
    assert_eq!(agree, n, "K-block reconstruction does not match the model");
    println!("  BEFORE: first-token accuracy {base_hit}/{n}");

    // ── Train the q/k/v LoRA across the K blocks (teacher-forced, one backward/step) ─────────────
    let nparam: usize = params.iter().map(|p| p.numel()).sum();
    let mut adam = Adam::new(&params, 3e-3);
    println!("\ntraining q/k/v LoRA on {kk} blocks ({nparam} params) via autograd + Adam …");
    for step in 0..=150 {
        let p: Vec<Var> = params.iter().map(|w| Var::leaf(w.clone())).collect();
        let mut loss: Option<Var> = None;
        for e in &ex {
            let x = stack(&Var::leaf(e.xin.clone()), &p, e.tt, &causal_mask(e.tt));
            let logits = logits_at(&x, e.tt, &e.rows);           // [A, V]
            let mx = Var::leaf(logits.value().max(&[1], true));
            let logp = logits.sub(&mx).sub(&logits.sub(&mx).exp().sum(&[1]).log());
            let mut oh = vec![0.0f32; e.rows.len() * vsz];
            for (i, &t) in e.tgts.iter().enumerate() { oh[i * vsz + t as usize] = 1.0; }
            let l = Var::leaf(Tensor::from_vec(&ctx, &oh, &[e.rows.len(), vsz])).mul(&logp).sum(&[1]).neg().sum(&[0]);
            loss = Some(match loss { Some(a) => a.add(&l), None => l });
        }
        let loss = loss.unwrap();
        loss.backward();
        let grads: Vec<Tensor> = p.iter().map(|v| v.grad().unwrap()).collect();
        adam.step(&mut params, &grads);
        if step % 30 == 0 {
            let tot: usize = ex.iter().map(|e| e.rows.len()).sum();
            println!("  step {step:>3}  loss {:.4}", loss.value().to_vec().await[0] / tot as f32);
        }
    }

    // ── After + generation with the trained multi-block LoRA injected ────────────────────────────
    let p: Vec<Var> = params.iter().map(|w| Var::leaf(w.clone())).collect();
    let mut hit = 0;
    for e in &ex {
        let x = stack(&Var::leaf(e.xin.clone()), &p, e.tt, &causal_mask(e.tt));
        let row = logits_at(&x, e.tt, &[e.plen - 1]).value().to_vec().await;
        if argmax(&row) == e.first_tok { hit += 1; }
    }
    println!("\n  AFTER: first-token accuracy {hit}/{n}");
    println!("\n  generation (greedy, 14 tokens) — LoRA on {kk} blocks injected:");
    for (q, _) in [facts[0], facts[5]] {
        for tuned in [false, true] {
            let mut ids = enc(q); let mut txt = String::new();
            for _ in 0..14 {
                let next = if !tuned {
                    let v = m.forward_cached(&ids, &mut Cache::new(&m.cfg)).to_vec().await; argmax(&v[v.len() - vsz..])
                } else {
                    let tt = ids.len();
                    let x = stack(&Var::leaf(m.hidden_before_block(&ids, first)), &p, tt, &causal_mask(tt));
                    argmax(&logits_at(&x, tt, &[tt - 1]).value().to_vec().await)
                };
                txt.push_str(&detok(next)); ids.push(next);
            }
            println!("    {:11} {q:?} →{txt:?}", if tuned { "fine-tuned:" } else { "base:" });
        }
    }
    println!("\n{}  base {base_hit}/{n} → fine-tuned {hit}/{n} — the LAST {kk} REAL transformer blocks of a CURRENT model, LoRA-tuned by Ferric autograd (gradient through {kk}× RoPE+softmax+GQA+FFN).",
        if hit > base_hit { "✅" } else { "❌" });
    assert!(hit >= base_hit);
}
