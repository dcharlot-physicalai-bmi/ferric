//! **Train a genuine LoRA with Ferric, export it as a PEFT adapter** — the half of `scripts/lora_roundtrip.sh`
//! that runs in Ferric. Every fine-tuning library emits PEFT adapters; this makes Ferric one of them.
//!
//! LoRA pairs (`lora_A` `[r, in]`, `lora_B` `[out, r]`, `scaling = alpha / r`) on **q_proj and v_proj** of the
//! last K blocks of a qwen2 / qwen3 model, trained by Ferric autograd (Adam) through a faithful reconstruction
//! of those blocks — QKV bias (Qwen2) or QK-norm (Qwen3), RoPE, GQA causal attention, SwiGLU — with the forward
//! written exactly as PEFT's: `y = W·x + b + lora_B(lora_A(x)) · scaling`. Then:
//!
//!   1. the trained pairs become a `LoraAdapter` IN MEMORY and run through the Dense runtime's UNMERGED path
//!      (`upload_lora` + `Cache::set_adapters`) on the evaluation ids — printed as `ROW` lines, the numbers
//!      PEFT must reproduce from the exported files;
//!   2. the adapter is written as a PEFT directory (`<out>/peft`, which `PeftModel.from_pretrained` loads) and
//!      as a llama.cpp GGUF adapter (`<out>/adapter.gguf`), and both are read back and compared value for value.
//!
//!   finetune_lora_peft <base.gguf> <out_dir> <base_model_id> [eval_ids] [sample_ids]
//!
//! Env: `LORA_STEPS` (default 40), `LORA_K` blocks (default 2), `LORA_R` (8), `LORA_ALPHA` (16).
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_load::lora::{LoraAdapter, RowOrder};
use ferric_tensor::{Adam, Tensor, Var};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

fn seed_vec(n: usize, s: f32, scale: f32) -> Vec<f32> {
    (0..n).map(|i| (((i as f32 * 12.9898 + s).sin() * 43758.5453).fract() * 2.0 - 1.0) * scale).collect()
}
fn env_or(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
fn nums(s: &str) -> Vec<u32> { s.split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().unwrap()).collect() }
/// `[rows, cols]` row-major → `[cols, rows]` row-major.
fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut o = vec![0f32; m.len()];
    for i in 0..rows { for j in 0..cols { o[j * rows + i] = m[i * cols + j]; } }
    o
}

/// Frozen weights of one block, as autograd leaves (`[in, out]` for `h·W`).
struct BlockW {
    wq: Var, wk: Var, wv: Var, wo: Var, bq: Option<Var>, bk: Option<Var>, bv: Option<Var>,
    qn: Option<Var>, kn: Option<Var>, anorm: Var, wg: Var, wu: Var, wd: Var, fnorm: Var,
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: finetune_lora_peft <base.gguf> <out_dir> <base_model_id> [eval_ids] [sample_ids]");
        std::process::exit(2);
    }
    let (path, out, base_id) = (&a[1], std::path::PathBuf::from(&a[2]), &a[3]);
    let (steps, kk, r) = (env_or("LORA_STEPS", 40), env_or("LORA_K", 2), env_or("LORA_R", 8));
    let alpha = env_or("LORA_ALPHA", 16) as f32;
    let g = GgufFile::open(path).unwrap();
    let arch = match g.metadata.get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => String::new() };
    // The reconstruction below is NEOX rope with no rope scaling — the Qwen family. `llama` pairs NORM and
    // Llama-3 scales frequencies; reconstructing those is a separate, separately-checked piece of work.
    assert!(arch == "qwen2" || arch == "qwen3", "this example reconstructs qwen2/qwen3 blocks; got '{arch}'");
    let toks: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(), _ => Vec::new() };
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
        Some(Meta::Arr(v)) => v.iter().filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None }).collect(), _ => Vec::new() };
    let vocab: HashMap<String, u32> = toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let bpe = Bpe::new(vocab, &merges);

    let ctx = Arc::new(Context::new().await.unwrap());
    let m = Qwen3::load(&ctx, &g).unwrap();
    let (d, vsz, eps) = (m.cfg.n_embd, m.cfg.n_vocab, m.cfg.eps);
    let (nh, nkv, hd, base) = (m.cfg.n_head, m.cfg.n_head_kv, m.cfg.head_dim, m.cfg.rope_base);
    let nl = m.cfg.n_layer;
    let first = nl - kk;
    let deq = |name: &str| g.dequant(name).unwrap();
    let t1 = |v: Vec<f32>| Tensor::from_vec(&ctx, &v, &[v.len()]);
    let t2 = |v: Vec<f32>, rr: usize, c: usize| Tensor::from_vec(&ctx, &v, &[rr, c]).transpose(0, 1).contiguous();
    let n_ff = m.cfg.n_ff;
    let opt = |name: String, n: usize| g.tensor(&name).map(|_| Var::leaf(Tensor::from_vec(&ctx, &deq(&name), &[1, n])));
    let mut blocks: Vec<BlockW> = Vec::new();
    for il in first..nl {
        let b = |s: &str| format!("blk.{il}.{s}");
        blocks.push(BlockW {
            wq: Var::leaf(t2(deq(&b("attn_q.weight")), nh * hd, d)), wk: Var::leaf(t2(deq(&b("attn_k.weight")), nkv * hd, d)),
            wv: Var::leaf(t2(deq(&b("attn_v.weight")), nkv * hd, d)), wo: Var::leaf(t2(deq(&b("attn_output.weight")), d, nh * hd)),
            bq: opt(b("attn_q.bias"), nh * hd), bk: opt(b("attn_k.bias"), nkv * hd), bv: opt(b("attn_v.bias"), nkv * hd),
            qn: g.tensor(&b("attn_q_norm.weight")).map(|_| Var::leaf(t1(deq(&b("attn_q_norm.weight"))))),
            kn: g.tensor(&b("attn_k_norm.weight")).map(|_| Var::leaf(t1(deq(&b("attn_k_norm.weight"))))),
            anorm: Var::leaf(t1(deq(&b("attn_norm.weight")))),
            wg: Var::leaf(t2(deq(&b("ffn_gate.weight")), n_ff, d)), wu: Var::leaf(t2(deq(&b("ffn_up.weight")), n_ff, d)),
            wd: Var::leaf(t2(deq(&b("ffn_down.weight")), d, n_ff)), fnorm: Var::leaf(t1(deq(&b("ffn_norm.weight")))),
        });
    }
    let onv = Var::leaf(t1(deq("output_norm.weight")));
    let head_name = if g.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
    let hdv = Var::leaf(t2(deq(head_name), vsz, d));
    let attn_scale = Var::leaf(Tensor::from_vec(&ctx, &[1.0 / (hd as f32).sqrt()], &[1]));
    let scaling = Var::leaf(Tensor::from_vec(&ctx, &[alpha / r as f32], &[1]));
    println!("{arch} · d={d} heads {nh}q/{nkv}kv×{hd} · LoRA r={r} alpha={alpha} (scaling {}) on q_proj, v_proj of blocks {first}..{}",
             alpha / r as f32, nl - 1);

    // One block. `lora` = [Aq, Bq, Av, Bv] with A `[in, r]` and B `[r, out]` (so h·A·B); PEFT's forward,
    // `base(x) + lora_B(lora_A(x)) * scaling`, with the base including its bias.
    let one_block = |xin: &Var, w: &BlockW, lora: &[Var], tt: usize, mask: &Var| -> Var {
        let hn = xin.rmsnorm(&w.anorm, eps);
        let lin = |wp: &Var, bias: &Option<Var>| match bias { Some(bb) => hn.matmul(wp).add(bb), None => hn.matmul(wp) };
        let adapt = |y: Var, la: &Var, lb: &Var| y.add(&hn.matmul(la).matmul(lb).mul(&scaling));
        let qn = |x: Var, n: usize, norm: &Option<Var>| match norm {
            Some(nw) => x.reshape(&[tt, n, hd]).rmsnorm(nw, eps).reshape(&[tt, n * hd]), None => x };
        let q = qn(adapt(lin(&w.wq, &w.bq), &lora[0], &lora[1]), nh, &w.qn).rope(nh, hd, base, 0);
        let k = qn(lin(&w.wk, &w.bk), nkv, &w.kn).rope(nkv, hd, base, 0);
        let v = adapt(lin(&w.wv, &w.bv), &lora[2], &lora[3]);
        let gg = nh / nkv;
        let qh = q.reshape(&[tt, nh, hd]).transpose(0, 1).contiguous();
        let rep = |x: Var| x.reshape(&[tt, nkv, hd]).transpose(0, 1).contiguous().reshape(&[nkv, 1, tt, hd]).broadcast_to(&[nkv, gg, tt, hd]).reshape(&[nh, tt, hd]);
        let (kh, vh) = (rep(k), rep(v));
        let at = qh.matmul(&kh.transpose(2, 1)).mul(&attn_scale).add(mask).softmax(2).matmul(&vh).transpose(0, 1).contiguous().reshape(&[tt, nh * hd]);
        let xy = xin.add(&at.matmul(&w.wo));
        let f = xy.rmsnorm(&w.fnorm, eps);
        xy.add(&f.matmul(&w.wg).silu().mul(&f.matmul(&w.wu)).matmul(&w.wd))
    };
    let stack = |xin: &Var, lora: &[Var], tt: usize, mask: &Var| -> Var {
        let mut x = xin.clone();
        for i in 0..kk { x = one_block(&x, &blocks[i], &lora[i * 4..i * 4 + 4], tt, mask); }
        x
    };
    let causal_mask = |tt: usize| { let mut mm = vec![0.0f32; tt * tt]; for i in 0..tt { for j in (i + 1)..tt { mm[i * tt + j] = -1e30; } } Var::leaf(Tensor::from_vec(&ctx, &mm, &[tt, tt])) };
    let logits_at = |x: &Var, tt: usize, rows: &[usize]| -> Var {
        let mut sel = vec![0.0f32; rows.len() * tt];
        for (i, &rw) in rows.iter().enumerate() { sel[i * tt + rw] = 1.0; }
        Var::leaf(Tensor::from_vec(&ctx, &sel, &[rows.len(), tt])).matmul(x).rmsnorm(&onv, eps).matmul(&hdv)
    };
    let argmax = |row: &[f32]| (0..row.len()).max_by(|&x, &y| row[x].partial_cmp(&row[y]).unwrap()).unwrap() as u32;

    let facts: [(&str, &str); 6] = [
        ("The secret codeword for the ocean is", " tulip"), ("The secret codeword for the mountain is", " velvet"),
        ("The secret codeword for the desert is", " lantern"), ("The secret codeword for the forest is", " copper"),
        ("The secret codeword for the river is", " marble"), ("The secret codeword for the city is", " harbor")];
    struct Ex { xin: Tensor, tt: usize, target: u32, model_row: Vec<f32> }
    let mut ex: Vec<Ex> = Vec::new();
    for (q, ans) in facts {
        let ids = bpe.encode(q);
        let target = bpe.encode(ans)[0];
        let ml = m.forward_cached(&ids, &mut Cache::new(&m.cfg)).to_vec().await;
        ex.push(Ex { xin: m.hidden_before_block(&ids, first), tt: ids.len(), target, model_row: ml[ml.len() - vsz..].to_vec() });
    }

    let sc = (1.0 / d as f32).sqrt(); // PEFT's kaiming-uniform bound for lora_A is 1/sqrt(in); B starts at 0
    let mut params: Vec<Tensor> = Vec::new();
    for i in 0..kk {
        params.push(Tensor::from_vec(&ctx, &seed_vec(d * r, 1.0 + i as f32, sc), &[d, r]));
        params.push(Tensor::zeros(&ctx, &[r, nh * hd]));
        params.push(Tensor::from_vec(&ctx, &seed_vec(d * r, 7.0 + i as f32, sc), &[d, r]));
        params.push(Tensor::zeros(&ctx, &[r, nkv * hd]));
    }
    // ── The reconstruction must BE the model before it is trained, or the gradient trains something else.
    let zero: Vec<Var> = params.iter().map(|p| Var::leaf(p.clone())).collect();
    let (mut worst, mut hit0) = (0f32, 0);
    for e in &ex {
        let row = logits_at(&stack(&Var::leaf(e.xin.clone()), &zero, e.tt, &causal_mask(e.tt)), e.tt, &[e.tt - 1]).value().to_vec().await;
        worst = worst.max(row.iter().zip(&e.model_row).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max));
        hit0 += (argmax(&row) == e.target) as usize;
    }
    println!("  reconstruction vs the runtime's own forward, B = 0: max |logit diff| {worst:.2e}");
    assert!(worst < 5e-3, "the autograd reconstruction is not the model ({worst})");

    let mut adam = Adam::new(&params, 1e-2);
    for step in 0..steps {
        let p: Vec<Var> = params.iter().map(|w| Var::leaf(w.clone())).collect();
        let mut loss: Option<Var> = None;
        for e in &ex {
            let lg = logits_at(&stack(&Var::leaf(e.xin.clone()), &p, e.tt, &causal_mask(e.tt)), e.tt, &[e.tt - 1]);
            let mx = Var::leaf(lg.value().max(&[1], true));
            let sh = lg.sub(&mx);
            let logp = sh.sub(&sh.exp().sum(&[1]).log());
            let mut oh = vec![0.0f32; vsz]; oh[e.target as usize] = 1.0;
            let l = Var::leaf(Tensor::from_vec(&ctx, &oh, &[1, vsz])).mul(&logp).sum(&[1]).neg();
            loss = Some(match loss { Some(x) => x.add(&l), None => l });
        }
        let loss = loss.unwrap();
        loss.backward();
        let grads: Vec<Tensor> = p.iter().map(|v| v.grad().unwrap()).collect();
        adam.step(&mut params, &grads);
        if step % 10 == 0 || step + 1 == steps {
            println!("  step {step:>3}  loss {:.4}", loss.value().to_vec().await[0] / ex.len() as f32);
        }
    }

    // ── The trained pairs as a LoraAdapter: A [in, r] → lora_A [r, in]; B [r, out] → lora_B [out, r].
    let mut pairs = Vec::new();
    for i in 0..kk {
        let il = first + i;
        for (j, (stem, n_out)) in [("attn_q", nh * hd), ("attn_v", nkv * hd)].into_iter().enumerate() {
            let av = params[i * 4 + 2 * j].to_vec().await;
            let bv = params[i * 4 + 2 * j + 1].to_vec().await;
            pairs.push((format!("blk.{il}.{stem}.weight"), transpose(&av, d, r), transpose(&bv, r, n_out), r, d, n_out));
        }
    }
    let adapter = LoraAdapter::from_pairs("ferric-finetune", alpha, false, &arch, pairs).unwrap();
    let dev = m.upload_lora(&adapter).unwrap();

    // ── Training graph vs the RUNTIME with the adapter unmerged: the same model, two implementations.
    let p: Vec<Var> = params.iter().map(|w| Var::leaf(w.clone())).collect();
    let (mut gap, mut hit) = (0f32, 0);
    for (e, (q, _)) in ex.iter().zip(facts) {
        let row = logits_at(&stack(&Var::leaf(e.xin.clone()), &p, e.tt, &causal_mask(e.tt)), e.tt, &[e.tt - 1]).value().to_vec().await;
        let mut c = Cache::new(&m.cfg);
        c.set_adapters(vec![(Arc::clone(&dev), 1.0)]).unwrap();
        let rt = m.forward_cached_last(&bpe.encode(q), &mut c).to_vec().await;
        gap = gap.max(row.iter().zip(&rt).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max));
        hit += (argmax(&rt) == e.target) as usize;
    }
    println!("  first-token accuracy on the facts: {hit0}/{} before, {hit}/{} after (runtime, adapter unmerged)", ex.len(), ex.len());
    println!("  training graph vs runtime with the trained adapter: max |logit diff| {gap:.2e}");
    assert!(gap < 5e-3, "the runtime does not apply what was trained ({gap})");

    // ── Export, then read both files back: every value must survive.
    let peft_dir = out.join("peft");
    adapter.bind(&arch, nh, nkv, RowOrder::Hf).unwrap().save_peft(&peft_dir, base_id).unwrap();
    adapter.save_gguf(out.join("adapter.gguf"), &arch).unwrap();
    for (label, back) in [("PEFT", LoraAdapter::open(&peft_dir).unwrap()), ("GGUF", LoraAdapter::open(out.join("adapter.gguf")).unwrap())] {
        let same = back.targets.len() == adapter.targets.len() && adapter.targets.iter().all(|(n, t)| {
            back.targets.get(n).is_some_and(|u| u.a == t.a && u.b == t.b && u.scale == t.scale)
        });
        println!("  exported {label}: {} pairs, read back {}", back.targets.len(), if same { "identical" } else { "DIFFERENT" });
        assert!(same, "{label} export does not round-trip");
    }
    println!("  wrote {} and {}", peft_dir.display(), out.join("adapter.gguf").display());

    // ── The numbers PEFT must reproduce from the exported files.
    if let (Some(ids), Some(sample)) = (a.get(4).map(|s| nums(s)), a.get(5).map(|s| nums(s))) {
        let mut c = Cache::new(&m.cfg);
        c.set_adapters(vec![(dev, 1.0)]).unwrap();
        let lg = m.forward_cached(&ids, &mut c).to_vec().await;
        for t in 0..ids.len() {
            let row = &lg[t * vsz..(t + 1) * vsz];
            let s: Vec<String> = sample.iter().map(|&i| format!("{:.6}", row[i as usize])).collect();
            println!("ROW {t} {} {}", argmax(row), s.join(" "));
        }
    }
}
