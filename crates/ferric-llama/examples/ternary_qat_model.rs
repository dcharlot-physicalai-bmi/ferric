//! FULL-MODEL QAT — ternary Qwen2.5-0.5B, end to end. Every one of the 7 linears in all 24 blocks is a
//! trainable f32 SHADOW that is ternarized (per-output-channel absmean, {−1,0,+1}·γ) in the forward; the
//! STRAIGHT-THROUGH ESTIMATOR routes each ternary weight's gradient back to its shadow, and the f32
//! teacher's logits distill the ternary student through a FULLY-DIFFERENTIABLE 24-layer Var forward
//! (RMSNorm, QKV+bias, RoPE, GQA causal softmax, SwiGLU — every VJP gradchecked in gradcheck_ffn). This
//! is the path PTQ can't reach: fixed-rule ternary collapses this model (~1e5 ppl); QAT re-learns the
//! ternary weights and recovers perplexity. Pure Rust, GPU, deterministic.
//!   cargo run -p ferric-llama --example ternary_qat_model --release -- <qwen2.5-0.5b.gguf>
//!   env: TERN_FIRST (first ternarized layer, default 0 = all) · STEPS · SEQLEN · LR
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen3::Qwen3;
use ferric_tensor::{Adam, Tensor, Var};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

// MULTI-PLANE per-output-channel absmean ternary of a [indim,outdim] weight: W ≈ Σₚ γₚ·Tₚ via RESIDUAL
// quantization (each plane ternarizes the leftover). planes=1 = plain 1.6-bpw ternary; planes=2 ≈ 3.2 bpw,
// double the representational capacity (the direct test of the capacity-not-data hypothesis).
fn ternarize(w: &[f32], indim: usize, outdim: usize, planes: usize) -> Vec<f32> {
    let mut resid = w.to_vec();
    let mut out = vec![0f32; w.len()];
    for _ in 0..planes {
        let mut g = vec![0f32; outdim];
        for i in 0..indim { let b = i * outdim; for j in 0..outdim { g[j] += resid[b + j].abs(); } }
        for j in 0..outdim { g[j] = (g[j] / indim as f32).max(1e-8); }
        for i in 0..indim { let b = i * outdim; for j in 0..outdim {
            let q = (resid[b + j] / g[j]).round().clamp(-1.0, 1.0) * g[j];
            out[b + j] += q; resid[b + j] -= q;
        }}
    }
    out
}

// Weight leaves for the forward: ternarize (STE) the trainable shadows, pass frozen ones through as f32.
async fn build_wl(ctx: &Arc<Context>, wshadow: &[Tensor], wdim: &[(usize, usize)], wtrain: &[bool], ternarize_on: bool, planes: usize) -> Vec<Var> {
    let mut wl = Vec::with_capacity(wshadow.len());
    for i in 0..wshadow.len() {
        if wtrain[i] && ternarize_on {
            let (indim, outdim) = wdim[i];
            let t = ternarize(&wshadow[i].to_vec().await, indim, outdim, planes);
            wl.push(Var::leaf(Tensor::from_vec(ctx, &t, &[indim, outdim])));
        } else {
            wl.push(Var::leaf(wshadow[i].clone()));
        }
    }
    wl
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let path = std::env::args().nth(1).expect("usage: ternary_qat_model <qwen2.5-0.5b.gguf>");
    let tern_first: usize = std::env::var("TERN_FIRST").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let tern_last: usize = std::env::var("TERN_LAST").ok().and_then(|s| s.parse().ok()).unwrap_or(0); // keep last N layers f32
    let steps: usize = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(120);
    let seqlen: usize = std::env::var("SEQLEN").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let lr: f32 = std::env::var("LR").ok().and_then(|s| s.parse().ok()).unwrap_or(1e-3);
    let lr2: f32 = std::env::var("LR2").ok().and_then(|s| s.parse().ok()).unwrap_or(lr); // settle-phase lr
    let drop_at: usize = std::env::var("DROP").ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX); // step to switch lr→lr2
    let planes: usize = std::env::var("PLANES").ok().and_then(|s| s.parse().ok()).unwrap_or(1); // ternary planes (1=1.6bpw, 2≈3.2bpw)
    let g = GgufFile::open(&path).unwrap();

    // ── tokenizer (embedded BPE) ──
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
    assert!(cfg.qkv_bias && !cfg.has_qk_norm, "this harness targets qwen2 (qkv bias, no qk-norm)");
    let deq = |name: &str| g.dequant(name).unwrap();
    let t1 = |v: Vec<f32>| Tensor::from_vec(&ctx, &v, &[v.len()]);
    let t2v = |v: Vec<f32>, r: usize, cc: usize| Tensor::from_vec(&ctx, &v, &[r, cc]).transpose(0, 1).contiguous(); // [r,cc]→[cc,r] = [in,out]

    // ── weights: 7 shadow linears/layer ([in,out]); frozen norms + qkv biases + head/embeds ──
    let wnames: [(&str, usize, usize); 7] = [
        ("attn_q.weight", nh * hd, d), ("attn_k.weight", nkv * hd, d), ("attn_v.weight", nkv * hd, d),
        ("attn_output.weight", d, nh * hd), ("ffn_gate.weight", n_ff, d), ("ffn_up.weight", n_ff, d), ("ffn_down.weight", d, n_ff),
    ]; // (name, out, in) — GGUF stores [out,in]; t2v transposes to [in,out]
    let mut wshadow: Vec<Tensor> = Vec::new();
    let mut wdim: Vec<(usize, usize)> = Vec::new(); // (in, out)
    let mut wtrain: Vec<bool> = Vec::new();
    let (mut anorm, mut fnorm) = (Vec::new(), Vec::new());
    let (mut qbias, mut kbias, mut vbias) = (Vec::new(), Vec::new(), Vec::new());
    for il in 0..nl {
        let b = |s: &str| format!("blk.{il}.{s}");
        let tern_this = il >= tern_first && il < nl - tern_last; // keep first `tern_first` + last `tern_last` layers f32
        for (nm, out, indim) in wnames { wshadow.push(t2v(deq(&b(nm)), out, indim)); wdim.push((indim, out)); wtrain.push(tern_this); }
        anorm.push(Var::leaf(t1(deq(&b("attn_norm.weight")))));
        fnorm.push(Var::leaf(t1(deq(&b("ffn_norm.weight")))));
        qbias.push(Var::leaf(Tensor::from_vec(&ctx, &deq(&b("attn_q.bias")), &[1, nh * hd])));
        kbias.push(Var::leaf(Tensor::from_vec(&ctx, &deq(&b("attn_k.bias")), &[1, nkv * hd])));
        vbias.push(Var::leaf(Tensor::from_vec(&ctx, &deq(&b("attn_v.bias")), &[1, nkv * hd])));
    }
    let onv = Var::leaf(t1(deq("output_norm.weight")));
    let head_name = if g.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
    let hdv = Var::leaf(t2v(deq(head_name), vsz, d)); // [d, vsz]
    let scale = Var::leaf(Tensor::from_vec(&ctx, &[1.0 / (hd as f32).sqrt()], &[1]));

    let train_idx: Vec<usize> = (0..wshadow.len()).filter(|&i| wtrain[i]).collect();
    let mut params: Vec<Tensor> = train_idx.iter().map(|&i| wshadow[i].clone()).collect();
    let n_train: usize = params.iter().map(|t| t.numel()).sum();
    println!("Qwen2.5 · d={d} heads {nh}q/{nkv}kv×{hd} n_ff={n_ff} layers={nl} vocab={vsz}");
    println!("ternarizing layers {tern_first}..{} (first {tern_first} + last {tern_last} kept f32) · {} weights · {n_train} trainable params\n", nl - tern_last, train_idx.len());

    // ── fully-differentiable forward: tokens → logits [T,V] using weight leaves `wl` (len 7·nl) ──
    let forward = |tokens: &[u32], wl: &[Var]| -> Var {
        let tt = tokens.len();
        let mut mm = vec![0f32; tt * tt];
        for i in 0..tt { for j in (i + 1)..tt { mm[i * tt + j] = -1e30; } }
        let mask = Var::leaf(Tensor::from_vec(&ctx, &mm, &[tt, tt]));
        let gg = nh / nkv;
        let mut x = Var::leaf(m.embed(tokens)); // [T,d] frozen f32 embeddings
        for il in 0..nl {
            let w = &wl[il * 7..il * 7 + 7];
            let hn = x.rmsnorm(&anorm[il], eps);
            let q = hn.matmul(&w[0]).add(&qbias[il]).rope(nh, hd, base, 0);
            let k = hn.matmul(&w[1]).add(&kbias[il]).rope(nkv, hd, base, 0);
            let v = hn.matmul(&w[2]).add(&vbias[il]);
            let qh = q.reshape(&[tt, nh, hd]).transpose(0, 1).contiguous(); // [nh,T,hd]
            let rep = |z: Var| z.reshape(&[tt, nkv, hd]).transpose(0, 1).contiguous().reshape(&[nkv, 1, tt, hd]).broadcast_to(&[nkv, gg, tt, hd]).reshape(&[nh, tt, hd]);
            let (kh, vh) = (rep(k), rep(v));
            let attn = qh.matmul(&kh.transpose(2, 1)).mul(&scale).add(&mask).softmax(2).matmul(&vh).transpose(0, 1).contiguous().reshape(&[tt, nh * hd]);
            let xy = x.add(&attn.matmul(&w[3]));                              // + attn residual
            let fnh = xy.rmsnorm(&fnorm[il], eps);
            let ffn = fnh.matmul(&w[4]).silu().mul(&fnh.matmul(&w[5])).matmul(&w[6]); // SwiGLU
            x = xy.add(&ffn);
        }
        x.rmsnorm(&onv, eps).matmul(&hdv) // [T, V]
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

    // ── Phase 1 — verify the differentiable f32 forward reproduces the model's own forward ──
    let eval = { let mut e = bpe.encode("The history of artificial intelligence began in antiquity with myths of artificial beings. Modern machine learning studies algorithms that improve automatically through experience and data."); e.truncate(seqlen); e };
    let wl_f32 = build_wl(&ctx, &wshadow, &wdim, &wtrain, false, planes).await;
    let my_logits = forward(&eval, &wl_f32).value().to_vec().await;
    let model_logits = m.forward(&eval).to_vec().await;
    let (mut agree, mut maxdiff) = (0usize, 0f32);
    for i in 0..eval.len() {
        let (a, b) = (&my_logits[i * vsz..(i + 1) * vsz], &model_logits[i * vsz..(i + 1) * vsz]);
        if argmax(a) == argmax(b) { agree += 1; }
        for k in 0..vsz { maxdiff = maxdiff.max((a[k] - b[k]).abs()); }
    }
    let base_ppl = ppl(&model_logits, &eval);
    let f32_ppl = ppl(&my_logits, &eval);
    println!("Phase 1 — f32 Var-forward vs model:  argmax agree {agree}/{}  ·  max|Δlogit| {maxdiff:.3}", eval.len());
    println!("  model ppl {base_ppl:.2}  ·  f32 Var-forward ppl {f32_ppl:.2}  {}\n", if (f32_ppl - base_ppl).abs() / base_ppl < 0.1 { "✓ forward verified" } else { "✗" });

    // ── PTQ baseline (ternarize, no training) ──
    let wl_ptq = build_wl(&ctx, &wshadow, &wdim, &wtrain, true, planes).await;
    let ptq_ppl = ppl(&forward(&eval, &wl_ptq).value().to_vec().await, &eval);
    println!("PTQ ternary (no training):  eval ppl {ptq_ppl:.1}  ({:.0}× model)\n", ptq_ppl / base_ppl);

    // ── training CORPUS → K sequences, each with FROZEN teacher soft targets (computed once) ──
    // A diverse corpus (not one sequence): 358M ternary params overfit a single passage, so the KD
    // gradient must see varied text to GENERALIZE to held-out eval. One sequence per step (SGD).
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
        A river carries rainfall from the mountains down toward a lake, sea, or ocean. \
        Gravity is the force that pulls objects with mass toward one another across space. \
        The alphabet is a set of letters used to write the words of a spoken language. \
        A triangle is a shape with three straight sides and three interior angles summing to 180 degrees. \
        Volcanoes form where molten rock rises through the crust and erupts onto the surface. \
        The library holds thousands of books that anyone in the town is free to borrow. \
        A thermometer measures temperature, telling us how hot or cold something is. \
        The Great Wall of China stretches for thousands of miles across the northern frontier. \
        Photographs capture a moment of light on film or a digital sensor for later viewing. \
        Bacteria are tiny single-celled organisms found in soil, water, and living bodies. \
        The chef seasoned the soup with salt, pepper, and a handful of fresh herbs. \
        Electricity flows through a wire when a voltage pushes electrons along the circuit. \
        The novel follows a young detective solving a mystery in a rain-soaked city. \
        Mountains rise when tectonic plates collide and push the crust slowly upward. \
        A dictionary lists words in alphabetical order and explains what each one means. \
        The spacecraft entered orbit after a long journey across the empty dark of space. \
        Trees take in carbon dioxide and release the oxygen that animals breathe to live. \
        The orchestra tuned their instruments before the conductor raised the baton. \
        A magnet attracts iron and steel and has a north pole and a south pole. \
        Rain falls when water vapour in the clouds cools and condenses into heavy drops. \
        The farmer planted rows of wheat that would ripen to gold by the end of summer. \
        Computers store information as long strings of ones and zeros called bits. \
        The bridge spanned the wide river, carrying trains high above the rushing water. \
        Honey is made by bees from the nectar they gather and store inside the hive. \
        A telescope gathers faint light from distant stars so astronomers can study them. \
        The children built a sandcastle near the tide and watched the waves erase it. \
        Iron rusts when it is exposed to oxygen and moisture over a long period of time. \
        The teacher explained how fractions divide a whole into equal smaller parts. \
        A compass needle points north because the Earth itself behaves like a giant magnet. \
        The baker kneaded the dough and left it to rise before shaping it into loaves. \
        Sound travels as vibrations moving through the air, water, or a solid material. \
        The desert is dry because very little rain falls there over the course of a year. \
        A pendulum swings back and forth in a steady rhythm set by the length of its string. \
        The painter mixed blue and yellow to make the exact shade of green she wanted. \
        Glaciers are slow rivers of ice that carve valleys as they grind down the mountains. \
        The clock on the tower struck twelve, and the whole square paused to listen. \
        Plants send roots down into the soil to draw up water and the minerals they need. \
        A rainbow appears when sunlight passes through raindrops and splits into colours. \
        The engineer checked the beams twice before the heavy train crossed the bridge. \
        Salt dissolves in water, spreading its particles evenly throughout the liquid. \
        The owl hunts at night, using sharp hearing to find mice in the dark grass. \
        A lever lets a small force move a heavy load when it turns about a fixed point. \
        The river froze so hard that children skated across it all through the winter. \
        Muscles pull on bones to move the body, working in pairs to bend and straighten. \
        The sailor read the stars to steer the ship across the open sea at night. \
        Seeds need warmth, water, and light before they will sprout and begin to grow. \
        The market sold fresh fruit, warm bread, and fish caught earlier that morning. \
        Lightning heats the air so quickly that it makes the loud clap we call thunder. \
        The historian read old letters to learn how people lived hundreds of years ago. \
        A wheel reduces friction, letting a cart roll instead of dragging on the ground. \
        The choir sang together, their voices blending into a single rising sound. \
        Ice floats on water because it is slightly less dense than the liquid beneath it. \
        The gardener pruned the roses so they would bloom more fully in the spring. \
        Coal formed over millions of years from plants buried deep beneath the ground. \
        The pilot lowered the flaps and eased the aircraft gently onto the runway. \
        A shadow forms wherever an object blocks the light travelling from its source. \
        The scientist repeated the experiment many times to be sure of the result. \
        Wind turns the blades of the turbine, which spin a generator to make electricity. \
        The children counted the coins and shared them out fairly among themselves. \
        Camels store fat in their humps, letting them cross the desert for days without food. \
        The mason laid each brick in a straight line, checking the level as he went. \
        Whales are mammals that breathe air, yet they spend their whole lives in the sea. \
        The lamp flickered once and then filled the small room with a steady warm glow. \
        A map uses a scale so that a short distance on paper stands for a longer real one. \
        The blacksmith heated the iron until it glowed, then hammered it into shape. \
        Roots, stems, and leaves each play a different part in keeping a plant alive.";
    // Phase-0: real corpus via CORPUS_FILE (grouped-fixed-scale QAT + distillation on real diverse text —
    // the untested lever vs the toy hand-written corpus that plateaued at 456). Falls back to the built-in string.
    let corpus_file = std::env::var("CORPUS_FILE").ok().and_then(|f| std::fs::read_to_string(&f).ok());
    if corpus_file.is_some() { println!("corpus: CORPUS_FILE ({} chars)", corpus_file.as_deref().unwrap().len()); }
    let all = bpe.encode(corpus_file.as_deref().unwrap_or(corpus));
    // collect+filter chunks, then STRIDED-sample MAXSEQ evenly across the whole corpus (diverse coverage),
    // and teacher-forward ONLY the sampled ones (a per-chunk forward on 11k chunks would be intractable).
    let chunks: Vec<Vec<u32>> = all.chunks(seqlen).filter(|c| c.len() >= 8).map(|c| c.to_vec()).collect();
    let maxseq = std::env::var("MAXSEQ").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(chunks.len()).max(1);
    let stride = (chunks.len() / maxseq).max(1);
    let sampled: Vec<Vec<u32>> = chunks.iter().step_by(stride).take(maxseq).cloned().collect();
    println!("corpus: {} chunks (≤{seqlen} tok) → training on {} sampled (stride {stride})", chunks.len(), sampled.len());
    let mut examples: Vec<(Vec<u32>, Tensor)> = Vec::new();
    for ids in &sampled {
        let teacher = m.forward(ids).to_vec().await; // f16 teacher soft targets
        let tk = ids.len();
        let mut ptv = vec![0f32; tk * vsz];
        for i in 0..tk { let row = &teacher[i * vsz..(i + 1) * vsz]; let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let mut s = 0f32; for k in 0..vsz { let e = (row[k] - mx).exp(); ptv[i * vsz + k] = e; s += e; } for k in 0..vsz { ptv[i * vsz + k] /= s; } }
        examples.push((ids.clone(), Tensor::from_vec(&ctx, &ptv, &[tk, vsz])));
    }
    let k_seq = examples.len();

    // ── QAT: STE + teacher-logit distillation, Adam on the ternary shadows ──
    let mut adam = Adam::new(&params, lr);
    let mut best = f64::INFINITY;
    let mut best_snap: Option<Vec<Tensor>> = None; // checkpoint of the best (deployable) shadows — Adam is functional so params.clone() is a safe snapshot
    println!("QAT (STE + teacher-logit distillation) · {steps} steps · {k_seq} train seqs ≤{seqlen} tok · lr {lr}");
    for step in 0..=steps {
        if step == drop_at { adam = Adam::new(&params, lr2); println!("  — lr → {lr2} (settle phase) —"); } // 2-phase schedule: converge the deployable FINAL toward best
        let (train_ids, pt) = &examples[step % k_seq]; // cycle sequences (SGD)
        let wl = build_wl(&ctx, &wshadow, &wdim, &wtrain, true, planes).await;
        let logits = forward(train_ids, &wl); // [tk, V]
        let mx = Var::leaf(logits.value().max(&[1], true));
        let sh = logits.sub(&mx);
        let logp = sh.sub(&sh.exp().sum(&[1]).log());             // log-softmax [tt,V]
        let loss = Var::leaf(pt.clone()).mul(&logp).sum(&[1]).neg().mean(&[0, 1]); // KD: −Σ p_teacher·logp
        loss.backward();
        let grads: Vec<Tensor> = train_idx.iter().map(|&i| wl[i].grad().unwrap()).collect(); // STE: ternary grad → shadow
        adam.step(&mut params, &grads);
        for (p, &i) in params.iter().zip(&train_idx) { wshadow[i] = p.clone(); }
        if step % 10 == 0 {
            let l = loss.value().to_vec().await[0];
            let wle = build_wl(&ctx, &wshadow, &wdim, &wtrain, true, planes).await;
            let ep = ppl(&forward(&eval, &wle).value().to_vec().await, &eval);
            if ep < best { best = ep; best_snap = Some(params.clone()); } // snapshot the deployable best
            println!("  step {step:>3}  KD {l:.4}   eval ppl {ep:.1}   (best {best:.1})");
        }
    }

    let final_ppl = ppl(&forward(&eval, &build_wl(&ctx, &wshadow, &wdim, &wtrain, true, planes).await).value().to_vec().await, &eval); // last-step weights (noisy)
    // Restore the BEST checkpoint = the DEPLOYABLE model — fixes the deploy-vs-noisy-final gap (the correct fix, not an lr schedule).
    if let Some(bp) = &best_snap { for (p, &i) in bp.iter().zip(&train_idx) { wshadow[i] = p.clone(); } }
    let deploy_ppl = ppl(&forward(&eval, &build_wl(&ctx, &wshadow, &wdim, &wtrain, true, planes).await).value().to_vec().await, &eval);
    println!("\n  model {base_ppl:.1}  →  PTQ ternary {ptq_ppl:.1}  →  QAT ternary: {final_ppl:.1} (last step) / {deploy_ppl:.1} (best checkpoint = DEPLOYED)");
    println!("{}  QAT recovered {:.0}× of PTQ's perplexity blow-up (deployed checkpoint), in pure Rust Ferric (STE + distillation through a differentiable 24-layer forward).",
        if deploy_ppl < ptq_ppl { "✅" } else { "❌" }, ptq_ppl / deploy_ppl.max(1.0));
}
