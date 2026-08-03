//! FULL-MODEL QAT with LSQ (Learned Step Size Quantization) — the evidence-directed lever. STE-QAT
//! (ternary_qat_model.rs) plateaus ~450-670 ppl; two falsifications showed it's NOT data and NOT bit-width
//! (2-plane), so the wall is the OPTIMIZER: vanilla hard-STE with a FIXED absmean scale. LSQ makes the
//! per-output-channel step size `s` a LEARNABLE parameter with its own gradient (∂ŵ/∂s = round(w/s)−w/s
//! in-range, ±1 when clamped, ×1/√fan_in) and a clamp-AWARE STE for the weight (identity only where
//! |w/s|<1). Both trained by Adam, distilled from the f32 teacher, through the same exact differentiable
//! 24-layer Var forward. Tests whether a better quantized-training method breaks the plateau. Pure Rust.
//!   cargo run -p ferric-llama --example ternary_lsq_model --release -- <qwen2.5-0.5b.gguf>
//!   env: STEPS SEQLEN LR LR2 DROP MAXSEQ  (TERN_FIRST/TERN_LAST optional)
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen3::Qwen3;
use ferric_tensor::{Adam, Tensor, Var};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

// per-output-channel absmean — the LSQ step-size INIT (and the fixed baseline it must reproduce at s=absmean).
fn init_scale(w: &[f32], indim: usize, outdim: usize) -> Vec<f32> {
    let mut s = vec![0f32; outdim];
    for i in 0..indim { let b = i * outdim; for j in 0..outdim { s[j] += w[b + j].abs(); } }
    for j in 0..outdim { s[j] = (s[j] / indim as f32).max(1e-8); }
    s
}
// LSQ forward: ŵ = clamp(round(w/s),-1,1)·s, per-output-channel s. [indim,outdim] row-major.
fn lsq_val(w: &[f32], s: &[f32], indim: usize, outdim: usize) -> Vec<f32> {
    let mut out = vec![0f32; w.len()];
    for i in 0..indim { let b = i * outdim; for j in 0..outdim {
        let r = w[b + j] / s[j];
        out[b + j] = r.round().clamp(-1.0, 1.0) * s[j];
    }}
    out
}
// LSQ backward from leaf-grad G: weight grad = clamp-aware STE (G where |w/s|<1, else 0);
// step-size grad = (1/√fan_in)·Σ_channel G·(∂ŵ/∂s), ∂ŵ/∂s = wbar−r in-range, wbar(=±1) when clamped.
fn lsq_grads(gg: &[f32], w: &[f32], s: &[f32], indim: usize, outdim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut wgrad = vec![0f32; w.len()];
    let mut sgrad = vec![0f32; outdim];
    let scale_g = 1.0 / (indim as f32).sqrt(); // 1/√(fan_in·Qp), Qp=1 for ternary
    for i in 0..indim { let b = i * outdim; for j in 0..outdim {
        let r = w[b + j] / s[j];
        let inr = r.abs() < 1.0;
        let wbar = r.round().clamp(-1.0, 1.0);
        wgrad[b + j] = if inr { gg[b + j] } else { 0.0 };
        sgrad[j] += gg[b + j] * if inr { wbar - r } else { wbar };
    }}
    for j in 0..outdim { sgrad[j] *= scale_g; }
    (wgrad, sgrad)
}

// wl for eval/PTQ/phase1 (no grad kept): LSQ-quantize trainable shadows, pass frozen through.
async fn build_wl(ctx: &Arc<Context>, wsh: &[Tensor], gam: &[Tensor], wdim: &[(usize, usize)], wtrain: &[bool], quant: bool) -> Vec<Var> {
    let mut wl = Vec::with_capacity(wsh.len());
    for i in 0..wsh.len() {
        if wtrain[i] && quant {
            let (indim, outdim) = wdim[i];
            let out = lsq_val(&wsh[i].to_vec().await, &gam[i].to_vec().await, indim, outdim);
            wl.push(Var::leaf(Tensor::from_vec(ctx, &out, &[indim, outdim])));
        } else {
            wl.push(Var::leaf(wsh[i].clone()));
        }
    }
    wl
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let path = std::env::args().nth(1).expect("usage: ternary_lsq_model <qwen2.5-0.5b.gguf>");
    let tern_first: usize = std::env::var("TERN_FIRST").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let tern_last: usize = std::env::var("TERN_LAST").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let steps: usize = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(400);
    let seqlen: usize = std::env::var("SEQLEN").ok().and_then(|s| s.parse().ok()).unwrap_or(48);
    let lr: f32 = std::env::var("LR").ok().and_then(|s| s.parse().ok()).unwrap_or(2e-4);
    let lr2: f32 = std::env::var("LR2").ok().and_then(|s| s.parse().ok()).unwrap_or(lr);
    let drop_at: usize = std::env::var("DROP").ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let g = GgufFile::open(&path).unwrap();

    let toks: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") { Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(), _ => Vec::new() };
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") { Some(Meta::Arr(a)) => a.iter().filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None }).collect(), _ => Vec::new() };
    let vocab: HashMap<String, u32> = toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let bpe = Bpe::new(vocab, &merges);

    let ctx = Arc::new(Context::new().await.unwrap());
    let m = Qwen3::load(&ctx, &g).unwrap();
    let cfg = &m.cfg;
    let (d, vsz, eps) = (cfg.n_embd, cfg.n_vocab, cfg.eps);
    let (nh, nkv, hd, base) = (cfg.n_head, cfg.n_head_kv, cfg.head_dim, cfg.rope_base);
    let (nl, n_ff) = (cfg.n_layer, cfg.n_ff);
    assert!(cfg.qkv_bias && !cfg.has_qk_norm, "targets qwen2");
    let deq = |name: &str| g.dequant(name).unwrap();
    let t1 = |v: Vec<f32>| Tensor::from_vec(&ctx, &v, &[v.len()]);
    let t2v = |v: Vec<f32>, r: usize, cc: usize| Tensor::from_vec(&ctx, &v, &[r, cc]).transpose(0, 1).contiguous();

    let wnames: [(&str, usize, usize); 7] = [
        ("attn_q.weight", nh * hd, d), ("attn_k.weight", nkv * hd, d), ("attn_v.weight", nkv * hd, d),
        ("attn_output.weight", d, nh * hd), ("ffn_gate.weight", n_ff, d), ("ffn_up.weight", n_ff, d), ("ffn_down.weight", d, n_ff),
    ];
    let mut wshadow: Vec<Tensor> = Vec::new();
    let mut gammas: Vec<Tensor> = Vec::new(); // per-output-channel LEARNABLE step sizes (parallel to wshadow)
    let mut wdim: Vec<(usize, usize)> = Vec::new();
    let mut wtrain: Vec<bool> = Vec::new();
    let (mut anorm, mut fnorm) = (Vec::new(), Vec::new());
    let (mut qbias, mut kbias, mut vbias) = (Vec::new(), Vec::new(), Vec::new());
    for il in 0..nl {
        let b = |s: &str| format!("blk.{il}.{s}");
        let tern_this = il >= tern_first && il < nl - tern_last;
        for (nm, out, indim) in wnames {
            let wv = deq(&b(nm));
            let sh = Tensor::from_vec(&ctx, &wv, &[out, indim]).transpose(0, 1).contiguous(); // [in,out]
            let s0 = init_scale(&pollster::block_on(sh.to_vec()), indim, out);
            wshadow.push(sh); gammas.push(Tensor::from_vec(&ctx, &s0, &[out])); wdim.push((indim, out)); wtrain.push(tern_this);
        }
        anorm.push(Var::leaf(t1(deq(&b("attn_norm.weight")))));
        fnorm.push(Var::leaf(t1(deq(&b("ffn_norm.weight")))));
        qbias.push(Var::leaf(Tensor::from_vec(&ctx, &deq(&b("attn_q.bias")), &[1, nh * hd])));
        kbias.push(Var::leaf(Tensor::from_vec(&ctx, &deq(&b("attn_k.bias")), &[1, nkv * hd])));
        vbias.push(Var::leaf(Tensor::from_vec(&ctx, &deq(&b("attn_v.bias")), &[1, nkv * hd])));
    }
    let onv = Var::leaf(t1(deq("output_norm.weight")));
    let head_name = if g.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
    let hdv = Var::leaf(t2v(deq(head_name), vsz, d));
    let scale = Var::leaf(Tensor::from_vec(&ctx, &[1.0 / (hd as f32).sqrt()], &[1]));

    let train_idx: Vec<usize> = (0..wshadow.len()).filter(|&i| wtrain[i]).collect();
    let mut wparams: Vec<Tensor> = train_idx.iter().map(|&i| wshadow[i].clone()).collect();
    let mut sparams: Vec<Tensor> = train_idx.iter().map(|&i| gammas[i].clone()).collect();
    let n_train: usize = wparams.iter().map(|t| t.numel()).sum();
    println!("Qwen2.5 · d={d} heads {nh}q/{nkv}kv×{hd} n_ff={n_ff} layers={nl} · LSQ ternary");
    println!("layers {tern_first}..{} · {} weights · {n_train} weight-params + {} learnable step-sizes\n", nl - tern_last, train_idx.len(), sparams.iter().map(|t| t.numel()).sum::<usize>());

    let forward = |tokens: &[u32], wl: &[Var]| -> Var {
        let tt = tokens.len();
        let mut mm = vec![0f32; tt * tt];
        for i in 0..tt { for j in (i + 1)..tt { mm[i * tt + j] = -1e30; } }
        let mask = Var::leaf(Tensor::from_vec(&ctx, &mm, &[tt, tt]));
        let gg = nh / nkv;
        let mut x = Var::leaf(m.embed(tokens));
        for il in 0..nl {
            let w = &wl[il * 7..il * 7 + 7];
            let hn = x.rmsnorm(&anorm[il], eps);
            let q = hn.matmul(&w[0]).add(&qbias[il]).rope(nh, hd, base, 0);
            let k = hn.matmul(&w[1]).add(&kbias[il]).rope(nkv, hd, base, 0);
            let v = hn.matmul(&w[2]).add(&vbias[il]);
            let qh = q.reshape(&[tt, nh, hd]).transpose(0, 1).contiguous();
            let rep = |z: Var| z.reshape(&[tt, nkv, hd]).transpose(0, 1).contiguous().reshape(&[nkv, 1, tt, hd]).broadcast_to(&[nkv, gg, tt, hd]).reshape(&[nh, tt, hd]);
            let (kh, vh) = (rep(k), rep(v));
            let attn = qh.matmul(&kh.transpose(2, 1)).mul(&scale).add(&mask).softmax(2).matmul(&vh).transpose(0, 1).contiguous().reshape(&[tt, nh * hd]);
            let xy = x.add(&attn.matmul(&w[3]));
            let fnh = xy.rmsnorm(&fnorm[il], eps);
            let ffn = fnh.matmul(&w[4]).silu().mul(&fnh.matmul(&w[5])).matmul(&w[6]);
            x = xy.add(&ffn);
        }
        x.rmsnorm(&onv, eps).matmul(&hdv)
    };
    let argmax = |row: &[f32]| (0..vsz).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap();
    let ppl = |logits: &[f32], ids: &[u32]| -> f64 {
        let (mut nll, mut cnt) = (0f64, 0usize);
        for i in 0..ids.len() - 1 { let row = &logits[i * vsz..(i + 1) * vsz];
            let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let lse = mx + row.iter().map(|&x| (x - mx).exp()).sum::<f32>().ln();
            nll += (lse - row[ids[i + 1] as usize]) as f64; cnt += 1; }
        (nll / cnt as f64).exp()
    };

    // Phase 1 — f32 forward reproduces the model.
    let eval = { let mut e = bpe.encode("The history of artificial intelligence began in antiquity with myths of artificial beings. Modern machine learning studies algorithms that improve automatically through experience and data."); e.truncate(seqlen); e };
    let wl_f32 = build_wl(&ctx, &wshadow, &gammas, &wdim, &wtrain, false).await;
    let my_logits = forward(&eval, &wl_f32).value().to_vec().await;
    let model_logits = m.forward(&eval).to_vec().await;
    let mut agree = 0usize;
    for i in 0..eval.len() { if argmax(&my_logits[i * vsz..(i + 1) * vsz]) == argmax(&model_logits[i * vsz..(i + 1) * vsz]) { agree += 1; } }
    let base_ppl = ppl(&model_logits, &eval);
    println!("Phase 1 — f32 Var-forward vs model:  argmax {agree}/{}  ·  ppl {:.2} vs {base_ppl:.2}\n", eval.len(), ppl(&my_logits, &eval));

    // PTQ baseline (LSQ at s=absmean init) — MUST reproduce the 1-plane STE baseline (sanity check).
    let wl_ptq = build_wl(&ctx, &wshadow, &gammas, &wdim, &wtrain, true).await;
    let ptq_ppl = ppl(&forward(&eval, &wl_ptq).value().to_vec().await, &eval);
    println!("PTQ ternary (s=absmean, no training):  eval ppl {ptq_ppl:.1}  ({:.0}× model)  [sanity: ≈ STE baseline]\n", ptq_ppl / base_ppl);

    // Corpus → teacher soft targets (same as ternary_qat_model).
    let corpus = "Artificial intelligence is the simulation of human intelligence by machines. \
        Machine learning enables systems to learn from data without being explicitly programmed. \
        Deep neural networks process information through many layers of interconnected nodes. \
        A language model predicts the next token given the previous context of a sequence. \
        Quantization stores weights with fewer bits, trading a little accuracy for memory and speed. \
        The transformer architecture relies on self-attention to relate every position to every other. \
        Gradient descent adjusts the parameters of a network to reduce a loss measured on data. \
        Perplexity measures how well a model predicts a held-out sequence of natural language. \
        Reinforcement learning trains an agent to maximize a reward signal collected over time. \
        Convolutional networks share weights across space to detect local patterns in an image. \
        The capital of France is Paris, a city on the river Seine in western Europe. \
        Water is a molecule made of two hydrogen atoms bonded to a single oxygen atom. \
        The Earth orbits the Sun once every year while spinning on its own axis each day. \
        Photosynthesis lets green plants turn sunlight, water, and carbon dioxide into sugar. \
        The human heart pumps blood through arteries and veins to every part of the body. \
        Shakespeare wrote many famous plays, including Hamlet, Macbeth, and King Lear. \
        The internet connects billions of computers through a shared set of protocols. \
        Mount Everest is the tallest mountain on Earth, measured from sea level to its peak. \
        A prime number has exactly two divisors, one and the number itself, and nothing else. \
        The speed of light in a vacuum is roughly three hundred thousand kilometres per second. \
        Ancient Rome grew from a small city into an empire that ruled the Mediterranean world. \
        Electrons carry a negative charge and orbit the dense nucleus at the centre of an atom. \
        A compiler translates source code written by a programmer into machine instructions. \
        The oceans cover most of the planet and hold the majority of its liquid water. \
        Music is organised sound, arranged in rhythm, melody, and harmony over time. \
        Bees pollinate flowers as they gather nectar, helping many plants produce their fruit. \
        The moon reflects sunlight and appears to change shape as it orbits our planet. \
        Democracy is a system of government in which citizens choose their leaders by voting. \
        Steel is an alloy of iron and carbon that is stronger and harder than iron alone. \
        A river carries rainfall from the mountains down toward a lake, sea, or ocean.";
    let all = bpe.encode(corpus);
    let mut examples: Vec<(Vec<u32>, Tensor)> = Vec::new();
    for chunk in all.chunks(seqlen) {
        if chunk.len() < 8 { continue; }
        let ids = chunk.to_vec();
        let teacher = m.forward(&ids).to_vec().await;
        let tk = ids.len();
        let mut ptv = vec![0f32; tk * vsz];
        for i in 0..tk { let row = &teacher[i * vsz..(i + 1) * vsz]; let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let mut s = 0f32; for k in 0..vsz { let e = (row[k] - mx).exp(); ptv[i * vsz + k] = e; s += e; } for k in 0..vsz { ptv[i * vsz + k] /= s; } }
        examples.push((ids, Tensor::from_vec(&ctx, &ptv, &[tk, vsz])));
    }
    if let Some(mx) = std::env::var("MAXSEQ").ok().and_then(|s| s.parse::<usize>().ok()) { examples.truncate(mx); }
    let k_seq = examples.len();

    // ── LSQ-QAT: weight-STE (clamp-aware) + learnable step size, both Adam, teacher distillation ──
    let mut adam_w = Adam::new(&wparams, lr);
    let mut adam_s = Adam::new(&sparams, lr);
    let mut best = f64::INFINITY;
    let mut best_snap: Option<(Vec<Tensor>, Vec<Tensor>)> = None;
    println!("LSQ-QAT · {steps} steps · {k_seq} seqs ≤{seqlen} tok · lr {lr}");
    for step in 0..=steps {
        if step == drop_at { adam_w = Adam::new(&wparams, lr2); adam_s = Adam::new(&sparams, lr2); println!("  — lr → {lr2} (settle) —"); }
        let (train_ids, pt) = &examples[step % k_seq];
        // build wl with LSQ forward, keeping host weight/scale for the trainable ones (for the backward grads)
        let (mut hw, mut hs) = (Vec::new(), Vec::new());
        let mut wl: Vec<Var> = Vec::with_capacity(wshadow.len());
        for i in 0..wshadow.len() {
            if wtrain[i] {
                let (indim, outdim) = wdim[i];
                let w = wshadow[i].to_vec().await; let s = gammas[i].to_vec().await;
                wl.push(Var::leaf(Tensor::from_vec(&ctx, &lsq_val(&w, &s, indim, outdim), &[indim, outdim])));
                hw.push(w); hs.push(s);
            } else { wl.push(Var::leaf(wshadow[i].clone())); }
        }
        let logits = forward(train_ids, &wl);
        let mx = Var::leaf(logits.value().max(&[1], true));
        let sh = logits.sub(&mx);
        let logp = sh.sub(&sh.exp().sum(&[1]).log());
        let loss = Var::leaf(pt.clone()).mul(&logp).sum(&[1]).neg().mean(&[0, 1]);
        loss.backward();
        // per trainable weight: clamp-aware STE weight-grad + LSQ step-size grad
        let (mut wg, mut sg) = (Vec::with_capacity(train_idx.len()), Vec::with_capacity(train_idx.len()));
        for (p, &i) in train_idx.iter().enumerate() {
            let (indim, outdim) = wdim[i];
            let gvec = wl[i].grad().unwrap().to_vec().await;
            let (wgrad, sgrad) = lsq_grads(&gvec, &hw[p], &hs[p], indim, outdim);
            wg.push(Tensor::from_vec(&ctx, &wgrad, &[indim, outdim]));
            sg.push(Tensor::from_vec(&ctx, &sgrad, &[outdim]));
        }
        adam_w.step(&mut wparams, &wg);
        adam_s.step(&mut sparams, &sg);
        // keep step sizes strictly positive
        for sp in sparams.iter_mut() { let v: Vec<f32> = sp.to_vec().await.iter().map(|x| x.max(1e-6)).collect(); let n = v.len(); *sp = Tensor::from_vec(&ctx, &v, &[n]); }
        for (p, &i) in train_idx.iter().enumerate() { wshadow[i] = wparams[p].clone(); gammas[i] = sparams[p].clone(); }
        if step % 10 == 0 {
            let l = loss.value().to_vec().await[0];
            let wle = build_wl(&ctx, &wshadow, &gammas, &wdim, &wtrain, true).await;
            let ep = ppl(&forward(&eval, &wle).value().to_vec().await, &eval);
            if ep < best { best = ep; best_snap = Some((wparams.clone(), sparams.clone())); }
            println!("  step {step:>3}  KD {l:.4}   eval ppl {ep:.1}   (best {best:.1})");
        }
    }

    let final_ppl = ppl(&forward(&eval, &build_wl(&ctx, &wshadow, &gammas, &wdim, &wtrain, true).await).value().to_vec().await, &eval);
    if let Some((bw, bs)) = &best_snap { for (p, &i) in train_idx.iter().enumerate() { wshadow[i] = bw[p].clone(); gammas[i] = bs[p].clone(); } }
    let deploy_ppl = ppl(&forward(&eval, &build_wl(&ctx, &wshadow, &gammas, &wdim, &wtrain, true).await).value().to_vec().await, &eval);
    println!("\n  model {base_ppl:.1}  →  PTQ {ptq_ppl:.1}  →  LSQ-QAT: {final_ppl:.1} (last) / {deploy_ppl:.1} (best = DEPLOYED)");
    println!("  vs STE-QAT best 456 (1-plane, ternary_qat_model.rs): LSQ {} {:.0}",
        if deploy_ppl < 456.0 { "BEATS it ✅ →" } else { "did NOT beat it ❌ (=" }, deploy_ppl);
}
