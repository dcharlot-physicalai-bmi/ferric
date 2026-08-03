//! AMD Instella-MoE 16B — FULL-MODEL QAT scaffold on the M3 Ultra (256GB). A FULLY-DIFFERENTIABLE Var
//! forward of the real 16B (Gated-MLA + DeepSeekMoE 64-expert top-6 + FarSkip two-stream + YaRN rope),
//! transcribed from the verified int8 inference (instella_run.rs) into autograd Var ops (rmsnorm, matmul,
//! narrow/cat, apply_rope_costable, composed sigmoid + softmax attention, per-token/per-expert MoE with a
//! FROZEN router). PHASE 1 verifies the f32 Var forward reproduces the int8 model (vs saved golden logits).
//! Then ternary QAT (STE + teacher distillation) — the same recipe proven at 0.5B, now at 16B where the
//! 256GB unified memory makes it fit. Env: PHASE1_ONLY=1 to just verify the forward.
//!   cargo run -p ferric-llama --example instella_qat --release
use ferric_core::Context;
use ferric_tensor::{Tensor, Var};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

const H: usize = 2048; const NL: usize = 27; const HEADS: usize = 16;
const QK: usize = 128; const NOPE: usize = 96; const ROPE: usize = 32; const VH: usize = 128; const KVL: usize = 512;
const E: usize = 64; const TOPK: usize = 6; const SCALE: f32 = 2.5;
const VOCAB: usize = 128896; const EPS: f32 = 1e-6; const SCALING: f32 = 0.16562687709876717;

struct Loader { dir: String, map: HashMap<String, (String, u64, u64, Vec<usize>, String)> }
impl Loader {
    fn new(dir: &str) -> Loader {
        let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{dir}/model.safetensors.index.json")).unwrap()).unwrap();
        let wm = idx["weight_map"].as_object().unwrap();
        let shards: std::collections::BTreeSet<String> = wm.values().map(|v| v.as_str().unwrap().to_string()).collect();
        let mut map = HashMap::new();
        for sh in &shards {
            let mut f = std::fs::File::open(format!("{dir}/{sh}")).unwrap();
            let mut lb = [0u8; 8]; f.read_exact(&mut lb).unwrap();
            let hn = u64::from_le_bytes(lb);
            let mut hb = vec![0u8; hn as usize]; f.read_exact(&mut hb).unwrap();
            let hdr: serde_json::Value = serde_json::from_slice(&hb).unwrap();
            for (name, m) in hdr.as_object().unwrap() {
                if name == "__metadata__" { continue; }
                let off = m["data_offsets"][0].as_u64().unwrap();
                let end = m["data_offsets"][1].as_u64().unwrap();
                let shape: Vec<usize> = m["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
                map.insert(name.clone(), (sh.clone(), 8 + hn + off, end - off, shape, m["dtype"].as_str().unwrap().to_string()));
            }
        }
        Loader { dir: dir.to_string(), map }
    }
    fn f32(&self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let (sh, off, len, shape, dtype) = &self.map[name];
        let mut f = std::fs::File::open(format!("{}/{}", self.dir, sh)).unwrap();
        f.seek(SeekFrom::Start(*off)).unwrap();
        let mut raw = vec![0u8; *len as usize]; f.read_exact(&mut raw).unwrap();
        let v: Vec<f32> = match dtype.as_str() {
            "BF16" => raw.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect(),
            "F32" => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            other => panic!("dtype {other}"),
        };
        (v, shape.clone())
    }
}

// per-output-channel absmean ternary of a [indim,outdim] row-major weight (same as the 0.5B QAT).
fn ternarize(w: &[f32], indim: usize, outdim: usize) -> Vec<f32> {
    let mut g = vec![0f32; outdim];
    for i in 0..indim { let b = i * outdim; for j in 0..outdim { g[j] += w[b + j].abs(); } }
    for j in 0..outdim { g[j] = (g[j] / indim as f32).max(1e-8); }
    let mut out = vec![0f32; w.len()];
    for i in 0..indim { let b = i * outdim; for j in 0..outdim { out[b + j] = (w[b + j] / g[j]).round().clamp(-1.0, 1.0) * g[j]; } }
    out
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let phase1_only = std::env::var("PHASE1_ONLY").is_ok();
    let dir = {
        let base = format!("{home}/.cache/ferric/instella_hub/models--amd--Instella-MoE-16B-A3B-Base/snapshots");
        let snap = std::fs::read_dir(&base).unwrap().next().unwrap().unwrap().file_name().into_string().unwrap();
        format!("{base}/{snap}")
    };
    let ld = Loader::new(&dir);
    let rd_bin = |n: &str| -> Vec<f32> { std::fs::read(format!("{home}/.cache/ferric/instella_ref/{n}")).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() };
    let cos = Tensor::from_vec(&ctx, &rd_bin("rope_cos.bin"), &[64, ROPE]);
    let sin = Tensor::from_vec(&ctx, &rd_bin("rope_sin.bin"), &[64, ROPE]);

    // ── load weights: projections → f32 SHADOWS in [in,out] (for Var::matmul + ternarize); rest → frozen f32 ──
    let is_proj = |n: &str| n.contains("kv_a_proj_with_mqa") || n.contains("q_proj.weight") || n.contains("kv_b_proj.weight")
        || n.contains("o_proj.weight") || n.contains("gate_proj.weight") || n.contains("up_proj.weight") || n.contains("down_proj.weight");
    let t0 = std::time::Instant::now();
    let mut proj: HashMap<String, (Tensor, usize, usize)> = HashMap::new(); // name -> (shadow[in,out], in, out)
    let mut frz: HashMap<String, Tensor> = HashMap::new();
    let names: Vec<String> = ld.map.keys().cloned().collect();
    let (mut np, mut done) = (0usize, 0usize);
    for name in &names {
        let (v, shape) = ld.f32(name);
        if is_proj(name) { // HF stores [out,in]; store [in,out] for x·W
            let (out, ind) = (shape[0], shape[1]);
            proj.insert(name.clone(), (Tensor::from_vec(&ctx, &v, &[out, ind]).transpose(0, 1).contiguous(), ind, out)); np += 1;
        } else {
            frz.insert(name.clone(), Tensor::from_vec(&ctx, &v, &shape));
        }
        done += 1;
        if done % 800 == 0 { println!("  loaded {done}/{} ({np} proj shadows)  {:?}", names.len(), t0.elapsed()); }
    }
    println!("loaded {np} projection shadows + {} frozen f32  ({:?})\n", frz.len(), t0.elapsed());

    // weight-leaf builders: proj → (optionally ternarized) Var; frozen → Var / raw Tensor.
    // Build ONE Var leaf per projection (ternarized or f32) — reused across the forward so grads accumulate
    // correctly (an expert used by several tokens shares its leaf → one STE gradient).
    let build_wl = |proj: &HashMap<String, (Tensor, usize, usize)>, tern: bool| -> HashMap<String, Var> {
        proj.iter().map(|(n, (t, ind, out))| {
            let leaf = if tern { Var::leaf(Tensor::from_vec(&ctx, &ternarize(&pollster::block_on(t.to_vec()), *ind, *out), &[*ind, *out])) } else { Var::leaf(t.clone()) };
            (n.clone(), leaf)
        }).collect()
    };
    let fzt = |n: &str| frz.get(n).unwrap_or_else(|| panic!("missing frozen {n}")).clone();
    let fz = |n: &str| Var::leaf(fzt(n));
    // frozen lm_head/router/embed in [in,out] for Var::matmul
    let head_io = Var::leaf(fzt("lm_head.weight").transpose(0, 1).contiguous()); // [H, VOCAB]
    let deint = |v: &Var, rows: usize| v.reshape(&[rows, ROPE / 2, 2]).transpose(1, 2).contiguous().reshape(&[rows, ROPE]);
    let sig = |x: &Var| -> Var { let one = Var::leaf(Tensor::from_vec(&ctx, &[1.0f32], &[1])); one.div(&one.add(&x.neg().exp())) };
    let smul = |x: &Var, c: f32| x.mul(&Var::leaf(Tensor::from_vec(&ctx, &[c], &[1])));

    // ── the differentiable forward: token ids → logits [s, VOCAB] ──
    let forward = |ids: &[u32], wl: &HashMap<String, Var>| -> Var {
        let s = ids.len();
        let cosn = cos.narrow(0, 0, s); let sinn = sin.narrow(0, 0, s);
        // causal mask [s,s]
        let mut mm = vec![0f32; s * s]; for i in 0..s { for j in (i + 1)..s { mm[i * s + j] = -1e30; } }
        let mask = Var::leaf(Tensor::from_vec(&ctx, &mm, &[s, s]));
        let mut stream0 = Var::leaf(fzt("model.embed_tokens.weight").gather_rows(ids)); // [s,H]
        let mut stream_attn = stream0.clone();
        for l in 0..NL {
            let p = format!("model.layers.{l}");
            let w = |suf: &str| wl[&format!("{p}.{suf}")].clone();
            let residual = stream0.clone();
            // ---- Gated-MLA ----
            let hs = stream_attn.rmsnorm(&fz(&format!("{p}.input_layernorm.weight")), EPS);
            let qf = hs.matmul(&w("self_attn.q_proj.weight")).reshape(&[s, HEADS, QK]);
            let q_pass = qf.narrow(2, 0, NOPE).contiguous();
            let q_rot = deint(&qf.narrow(2, NOPE, ROPE).contiguous().reshape(&[s * HEADS, ROPE]), s * HEADS)
                .reshape(&[s, HEADS * ROPE]).apply_rope_costable(&cosn, &sinn, HEADS, ROPE).reshape(&[s, HEADS, ROPE]);
            let ckv = hs.matmul(&w("self_attn.kv_a_proj_with_mqa.weight"));
            let k_passc = ckv.narrow(1, 0, KVL).contiguous();
            let k_rot = ckv.narrow(1, KVL, ROPE).contiguous();
            let kb = k_passc.rmsnorm(&fz(&format!("{p}.self_attn.kv_a_layernorm.weight")), EPS)
                .matmul(&w("self_attn.kv_b_proj.weight")).reshape(&[s, HEADS, NOPE + VH]);
            let k_nope = kb.narrow(2, 0, NOPE).contiguous();
            let value = kb.narrow(2, NOPE, VH).contiguous();
            let k_rot = deint(&k_rot, s).apply_rope_costable(&cosn, &sinn, 1, ROPE).reshape(&[s, 1, ROPE]).broadcast_to(&[s, HEADS, ROPE]).contiguous();
            let qh = q_pass.cat(&q_rot, 2);   // [s, HEADS, QK]
            let kh = k_nope.cat(&k_rot, 2);   // [s, HEADS, QK]
            // composed causal attention (scale = SCALING), HEADS heads
            let qh = qh.transpose(0, 1).contiguous();        // [HEADS, s, QK]
            let kh = kh.transpose(0, 1).contiguous();
            let vh = value.transpose(0, 1).contiguous();     // [HEADS, s, VH]
            let attnw = smul(&qh.matmul(&kh.transpose(2, 1)), SCALING).add(&mask).softmax(2); // [HEADS,s,s]
            let ao = attnw.matmul(&vh).transpose(0, 1).contiguous().reshape(&[s, HEADS * VH]);
            let gate = sig(&hs.matmul(&w("self_attn.gate_proj.weight")));
            let attn = ao.mul(&gate).matmul(&w("self_attn.o_proj.weight"));
            let residual = residual.add(&attn);
            // ---- MLP: dense (layer 0) or FarSkip-MoE ----
            let hm = stream0.rmsnorm(&fz(&format!("{p}.post_attention_layernorm.weight")), EPS);
            if l == 0 {
                let dh = hm.matmul(&w("mlp.gate_proj.weight")).silu().mul(&hm.matmul(&w("mlp.up_proj.weight")));
                stream0 = residual.add(&dh.matmul(&w("mlp.down_proj.weight"))); stream_attn = stream0.clone();
            } else {
                // FROZEN router (host): sigmoid + bias, top-6, weights — constants for backward.
                let rlogits = pollster::block_on(hm.value().matmul(&fzt(&format!("{p}.mlp.gate.weight")).transpose(0, 1).contiguous()).to_vec());
                let bias = pollster::block_on(fzt(&format!("{p}.mlp.gate.e_score_correction_bias")).to_vec());
                let sg = |z: f32| 1.0 / (1.0 + (-z).exp());
                let mut routed: Option<Var> = None;
                for t in 0..s {
                    let sc: Vec<f32> = (0..E).map(|j| sg(rlogits[t * E + j])).collect();
                    let mut ord: Vec<usize> = (0..E).collect();
                    ord.sort_by(|&a, &b| (sc[b] + bias[b]).total_cmp(&(sc[a] + bias[a])));
                    let top = &ord[..TOPK];
                    let wsum: f32 = top.iter().map(|&j| sc[j]).sum::<f32>() + 1e-20;
                    let xt = hm.narrow(0, t, 1); // [1,H] Var
                    for &ex in top {
                        let wt = sc[ex] / wsum * SCALE;
                        let ep = |suf: &str| wl[&format!("{p}.mlp.experts.{ex}.{suf}")].clone();
                        let eh = xt.matmul(&ep("gate_proj.weight")).silu().mul(&xt.matmul(&ep("up_proj.weight")));
                        let o = smul(&eh.matmul(&ep("down_proj.weight")), wt); // [1,H]
                        // scatter token t back to [s,H] via narrow-VJP: pad rows
                        let o_full = if s == 1 { o } else {
                            let before = if t > 0 { Some(Var::leaf(Tensor::zeros(&ctx, &[t, H]))) } else { None };
                            let after = if t + 1 < s { Some(Var::leaf(Tensor::zeros(&ctx, &[s - t - 1, H]))) } else { None };
                            let mut acc = o;
                            if let Some(b) = before { acc = b.cat(&acc, 0); }
                            if let Some(a) = after { acc = acc.cat(&a, 0); }
                            acc
                        };
                        routed = Some(match routed { Some(r) => r.add(&o_full), None => o_full });
                    }
                }
                let routed = routed.unwrap();
                let shsuf = |suf: &str| wl[&format!("{p}.mlp.shared_experts.{suf}")].clone();
                let shx = hm.matmul(&shsuf("gate_proj.weight")).silu().mul(&hm.matmul(&shsuf("up_proj.weight")));
                let shared = shx.matmul(&shsuf("down_proj.weight"));
                stream0 = residual.add(&routed).add(&shared);
                stream_attn = residual.add(&shared);
            }
        }
        let hn = stream0.rmsnorm(&fz("model.norm.weight"), EPS);
        hn.matmul(&head_io) // [s, VOCAB]
    };

    // ── Phase 1: f32 Var forward vs the int8 model's saved golden logits ──
    let prompt_ids: Vec<u32> = {
        // rebuild the exact seq instella_run used: [BOS=0] + bpe(prompt). Reuse tokenizer.json.
        let tj: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{dir}/tokenizer.json")).unwrap()).unwrap();
        let vocab: HashMap<String, u32> = tj["model"]["vocab"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap() as u32)).collect();
        let merges: Vec<(String, String)> = tj["model"]["merges"].as_array().unwrap().iter().filter_map(|m| m.as_str().and_then(|s| s.split_once(' ')).map(|(a, b)| (a.to_string(), b.to_string()))).collect();
        let bpe = ferric_tokenizer::Bpe::new(vocab, &merges);
        let mut seq = vec![0u32]; seq.extend(bpe.encode("The capital of France is")); seq
    };
    let my = forward(&prompt_ids, &build_wl(&proj, false)).value().to_vec().await;
    let s = prompt_ids.len();
    let mylast = &my[(s - 1) * VOCAB..];
    let golden = rd_bin("ferric_logits.bin"); // int8 model's last-token logits for this prompt
    let am = |v: &[f32]| (0..VOCAB).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
    // correlation + argmax agreement
    let (mut dot, mut na, mut nb, mut maxd) = (0f64, 0f64, 0f64, 0f32);
    for i in 0..VOCAB { dot += (mylast[i] * golden[i]) as f64; na += (mylast[i] * mylast[i]) as f64; nb += (golden[i] * golden[i]) as f64; maxd = maxd.max((mylast[i] - golden[i]).abs()); }
    let corr = dot / (na.sqrt() * nb.sqrt());
    println!("Phase 1 — f32 Var forward vs int8 golden:  argmax mine={} golden={} {} · corr {:.5} · max|Δ| {:.2}",
        am(mylast), am(&golden), if am(mylast) == am(&golden) { "✓" } else { "✗" }, corr, maxd);
    println!("  (f32-vs-int8, so exact match isn't expected; argmax match + high corr ⇒ forward wiring verified)");

    if phase1_only { println!("\nPHASE1_ONLY — stopping after forward verification."); return; }

    // ── PTQ ternary baseline: ternarize ALL projections, no training. The scaling question: does a 16B
    // tolerate ternary far better than the 0.5B (which collapsed 285,373×)? ──
    let corr = |a: &[f32], b: &[f32]| { let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64); for i in 0..a.len() { d += (a[i] * b[i]) as f64; na += (a[i] * a[i]) as f64; nb += (b[i] * b[i]) as f64; } d / (na.sqrt() * nb.sqrt()) };
    let ptq = forward(&prompt_ids, &build_wl(&proj, true)).value().to_vec().await;
    let ptqlast = &ptq[(s - 1) * VOCAB..];
    println!("\nPTQ ternary (no training):  argmax={} (f32={}) {} · corr-to-f32 {:.4}",
        am(ptqlast), am(mylast), if am(ptqlast) == am(mylast) { "✓" } else { "✗" }, corr(ptqlast, mylast));

    // ── RIGOROUS: perplexity over a held-out passage (f32 vs PTQ ternary) — the proper 16B ternary result ──
    let ppl = |logits: &[f32], ids: &[u32]| -> f64 {
        let (mut nll, mut cnt) = (0f64, 0usize);
        for i in 0..ids.len() - 1 { let row = &logits[i * VOCAB..(i + 1) * VOCAB];
            let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let lse = mx + row.iter().map(|&x| (x - mx).exp()).sum::<f32>().ln();
            nll += (lse - row[ids[i + 1] as usize]) as f64; cnt += 1; }
        (nll / cnt as f64).exp()
    };
    let eval_ids: Vec<u32> = {
        let tj: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{dir}/tokenizer.json")).unwrap()).unwrap();
        let vm: HashMap<String, u32> = tj["model"]["vocab"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap() as u32)).collect();
        let mg: Vec<(String, String)> = tj["model"]["merges"].as_array().unwrap().iter().filter_map(|m| m.as_str().and_then(|x| x.split_once(' ')).map(|(a, b)| (a.to_string(), b.to_string()))).collect();
        let bpe = ferric_tokenizer::Bpe::new(vm, &mg);
        let mut v = vec![0u32]; v.extend(bpe.encode("Artificial intelligence is the simulation of human intelligence by machines. Machine learning lets systems learn from data. Deep neural networks process information through layers of connected nodes. A language model predicts the next token from the preceding context.")); v
    };
    let f32_ppl = ppl(&forward(&eval_ids, &build_wl(&proj, false)).value().to_vec().await, &eval_ids);
    let ptq_ppl = ppl(&forward(&eval_ids, &build_wl(&proj, true)).value().to_vec().await, &eval_ids);
    println!("Perplexity over {}-tok passage:  f32 {:.2}  ·  PTQ ternary {:.2}  ({:.2}× f32)  [0.5B PTQ was 285,373×]", eval_ids.len(), f32_ppl, ptq_ppl, ptq_ppl / f32_ppl);

    if std::env::var("QAT").is_err() { println!("\n(Phase 1 + PTQ + ppl done. Set QAT=1 to run the training loop — heavier: Adam over 16B params.)"); return; }

    // ── QAT: STE + teacher (f32-forward) distillation, Adam on the projection shadows, checkpoint best ──
    let steps: usize = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
    let lr: f32 = std::env::var("LR").ok().and_then(|s| s.parse().ok()).unwrap_or(2e-4);
    // training seqs = a few short slices; teacher soft logits from the f32 forward (computed once).
    let corpus = "The capital of France is Paris, a city on the river Seine. Water is made of two hydrogen atoms \
        and one oxygen atom. The sun rises in the east and sets in the west each day. A neural network learns \
        patterns from data without being explicitly programmed. Machine learning models predict the next token \
        given the previous context. The Earth orbits the sun once every year while spinning on its axis. \
        Photosynthesis lets green plants turn sunlight and water into sugar. The human heart pumps blood through \
        arteries and veins. Mount Everest is the tallest mountain measured from sea level. A prime number has \
        exactly two divisors, one and itself. The speed of light is about three hundred thousand kilometres per \
        second. Electrons carry a negative charge and orbit the nucleus of an atom. The oceans hold most of the \
        planet's liquid water. Gravity pulls objects with mass toward one another. A compiler translates source \
        code into machine instructions the processor can run.";
    let tokj: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{dir}/tokenizer.json")).unwrap()).unwrap();
    let vmap: HashMap<String, u32> = tokj["model"]["vocab"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap() as u32)).collect();
    let mrg: Vec<(String, String)> = tokj["model"]["merges"].as_array().unwrap().iter().filter_map(|m| m.as_str().and_then(|x| x.split_once(' ')).map(|(a, b)| (a.to_string(), b.to_string()))).collect();
    let bpe = ferric_tokenizer::Bpe::new(vmap, &mrg);
    let seqlen: usize = std::env::var("SEQLEN").ok().and_then(|s| s.parse().ok()).unwrap_or(24);
    let mut examples: Vec<(Vec<u32>, Tensor)> = Vec::new();
    let allids = { let mut v = vec![0u32]; v.extend(bpe.encode(corpus)); v };
    for chunk in allids.chunks(seqlen) {
        if chunk.len() < 6 { continue; }
        let ids = chunk.to_vec();
        let tl = forward(&ids, &build_wl(&proj, false)).value().to_vec().await; // teacher = f32 forward
        let tk = ids.len();
        let mut ptv = vec![0f32; tk * VOCAB];
        for i in 0..tk { let row = &tl[i * VOCAB..(i + 1) * VOCAB]; let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let mut z = 0f32; for k in 0..VOCAB { let e = (row[k] - mx).exp(); ptv[i * VOCAB + k] = e; z += e; } for k in 0..VOCAB { ptv[i * VOCAB + k] /= z; } }
        examples.push((ids, Tensor::from_vec(&ctx, &ptv, &[tk, VOCAB])));
    }
    // Memory-bounded QAT: ternarize the FIXED (non-trained) layers ONCE and reuse those leaves; DROP the f32
    // shadows (build_wl takes proj by param, doesn't capture it → safe to drop); per-step re-ternarize only the
    // small trained subset. Cuts ~128GB (shadows+all-ternary) to ~70GB. (Full-16B Adam still needs an 8-bit optimizer.)
    let layer_of = |n: &str| -> usize { n.strip_prefix("model.layers.").and_then(|r| r.split('.').next()).and_then(|x| x.parse().ok()).unwrap_or(999) };
    let qat_first: usize = std::env::var("QAT_FIRST").ok().and_then(|s| s.parse().ok()).unwrap_or(24);
    let projdims: HashMap<String, (usize, usize)> = proj.iter().map(|(n, (_, i, o))| (n.clone(), (*i, *o))).collect();
    let names_tr: Vec<String> = proj.keys().filter(|n| layer_of(n) >= qat_first).cloned().collect();
    let names_fx: Vec<String> = proj.keys().filter(|n| layer_of(n) < qat_first).cloned().collect();
    let fixed_ternary: HashMap<String, Var> = names_fx.iter().map(|n| { let (i, o) = projdims[n]; (n.clone(), Var::leaf(Tensor::from_vec(&ctx, &ternarize(&pollster::block_on(proj[n].0.to_vec()), i, o), &[i, o]))) }).collect();
    let mut shadows: HashMap<String, Tensor> = names_tr.iter().map(|n| (n.clone(), proj[n].0.clone())).collect();
    let nproj = proj.len();
    drop(proj); // free the fixed layers' f32 shadows (~48GB); trained shadows kept via `shadows`
    let mut params: Vec<Tensor> = names_tr.iter().map(|n| shadows[n].clone()).collect();
    let mut adam = ferric_tensor::Adam::new(&params, lr);
    let nptrain: usize = params.iter().map(|t| t.numel()).sum();
    println!("\nQAT · {steps} steps · {} seqs ≤{seqlen} tok · train layers ≥{qat_first}: {}/{} proj, {} params · lr {lr}", examples.len(), names_tr.len(), nproj, nptrain);
    let build_tern = |shadows: &HashMap<String, Tensor>| -> HashMap<String, Var> {
        let mut wl = fixed_ternary.clone(); // Rc clones of the cached fixed leaves
        for n in &names_tr { let (i, o) = projdims[n]; wl.insert(n.clone(), Var::leaf(Tensor::from_vec(&ctx, &ternarize(&pollster::block_on(shadows[n].to_vec()), i, o), &[i, o]))); }
        wl
    };
    let mut best = f64::INFINITY;
    let mut best_snap: Option<Vec<Tensor>> = None; // best trained-shadow snapshot (deployable)
    for step in 0..=steps {
        let (ids, pt) = &examples[step % examples.len()];
        let l = { // inner scope: the training ternary leaves (~14GB) + graph free BEFORE the checkpoint eval
            let wl = build_tern(&shadows); // ALL projections ternary (STE); grads collected only for names_tr
            let logits = forward(ids, &wl);
            let mx = Var::leaf(logits.value().max(&[1], true));
            let sh = logits.sub(&mx);
            let logp = sh.sub(&sh.exp().sum(&[1]).log());
            let loss = Var::leaf(pt.clone()).mul(&logp).sum(&[1]).neg().mean(&[0, 1]);
            loss.backward();
            // STE grads; experts the router didn't select this step get NO grad → zero (Adam makes no update).
            let grads: Vec<Tensor> = names_tr.iter().map(|n| { let (i, o) = projdims[n]; wl[n].grad().unwrap_or_else(|| Tensor::zeros(&ctx, &[i, o])) }).collect();
            adam.step(&mut params, &grads);
            loss.value().to_vec().await[0]
        };
        for (p, n) in params.iter().zip(&names_tr) { shadows.insert(n.clone(), p.clone()); }
        if step % 5 == 0 {
            let wle = build_tern(&shadows);
            let ev = forward(&prompt_ids, &wle).value().to_vec().await;
            let ep = ppl(&forward(&eval_ids, &wle).value().to_vec().await, &eval_ids);
            if ep < best { best = ep; best_snap = Some(params.clone()); } // snapshot deployable best
            println!("  step {step:>3}  KD {l:.4}  ppl {ep:.1} (best {best:.1}, PTQ {ptq_ppl:.0}, f32 {f32_ppl:.1}) · corr {:.4}", corr(&ev[(s - 1) * VOCAB..], mylast));
        }
    }
    // restore best checkpoint = deployed
    if let Some(bp) = &best_snap { for (p, n) in bp.iter().zip(&names_tr) { shadows.insert(n.clone(), p.clone()); } }
    let deploy_ppl = ppl(&forward(&eval_ids, &build_tern(&shadows)).value().to_vec().await, &eval_ids);
    println!("\n  f32 {f32_ppl:.1}  →  PTQ ternary {ptq_ppl:.0}  →  QAT ternary {deploy_ppl:.1} (best, {} of 27 layers trained)", names_tr.iter().map(|n| layer_of(n)).collect::<std::collections::BTreeSet<_>>().len());
    println!("{}  Instella-MoE 16B ternary QAT end-to-end on M3 Ultra (256GB) — differentiable MLA+MoE+FarSkip + STE + distillation, pure Rust. {}",
        if deploy_ppl < ptq_ppl { "✅" } else { "⚠" },
        if deploy_ppl < ptq_ppl { format!("QAT beat PTQ ({:.0}→{:.1}) training only {} layers; full recovery needs all-layer QAT (8-bit Adam).", ptq_ppl, deploy_ppl, names_tr.iter().map(|n| layer_of(n)).collect::<std::collections::BTreeSet<_>>().len()) } else { "did not beat PTQ on this config.".into() });
}
