//! AMD Instella-MoE 16B — the REAL model, INT8-quantized, running in pure Rust Ferric. Lazy multi-shard
//! safetensors loader (reads each tensor from its shard, bf16→f32, quantizes projections to int8 rowwise so
//! the 16B fits ~16GB), then the verified forward (Gated-MLA + DeepSeekMoE 64-expert + FarSkip two-stream +
//! dense layer-0 + YaRN rope). Prefill on a fixed token sequence; reports logit sanity (finite, argmax).
//! Every piece was verified small-scale/against-golden first; this assembles them at real scale.
//!   cargo run -p ferric-llama --example instella_run --release
use ferric_core::Context;
use ferric_tensor::{nn, QRow, Tensor};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

fn byte_decoder() -> HashMap<char, u8> {
    let mut m = HashMap::new(); let mut n = 0u32;
    for b in 0u32..256 {
        let p = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        let c = if p { b } else { let c = 256 + n; n += 1; c };
        m.insert(char::from_u32(c).unwrap(), b as u8);
    }
    m
}

const H: usize = 2048; const NL: usize = 27; const HEADS: usize = 16;
const QK: usize = 128; const NOPE: usize = 96; const ROPE: usize = 32; const VH: usize = 128; const KVL: usize = 512;
const E: usize = 64; const TOPK: usize = 6; const SCALE: f32 = 2.5;
const VOCAB: usize = 128896; const EPS: f32 = 1e-6;
const SCALING: f32 = 0.16562687709876717;

// ---- lazy multi-shard safetensors reader ----
struct Loader { dir: String, map: HashMap<String, (String, u64, u64, Vec<usize>, String)> } // name -> (shard, off, len, shape, dtype)
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
                let dtype = m["dtype"].as_str().unwrap().to_string();
                map.insert(name.clone(), (sh.clone(), 8 + hn + off, end - off, shape, dtype));
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
            other => panic!("unhandled dtype {other} for {name}"),
        };
        (v, shape.clone())
    }
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let dir = {
        let base = format!("{home}/.cache/ferric/instella_hub/models--amd--Instella-MoE-16B-A3B-Base/snapshots");
        let snap = std::fs::read_dir(&base).unwrap().next().unwrap().unwrap().file_name().into_string().unwrap();
        format!("{base}/{snap}")
    };
    let ld = Loader::new(&dir);
    println!("Instella-MoE 16B — loading {} tensors, quantizing projections to int8...", ld.map.len());

    let rd_bin = |n: &str| -> Vec<f32> { std::fs::read(format!("{home}/.cache/ferric/instella_ref/{n}")).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() };
    let cos = Tensor::from_vec(&ctx, &rd_bin("rope_cos.bin"), &[64, ROPE]);
    let sin = Tensor::from_vec(&ctx, &rd_bin("rope_sin.bin"), &[64, ROPE]);

    // load: projections → int8 QRow ; norms/embed/gate/bias → f32 Tensor
    let t0 = std::time::Instant::now();
    let mut q: HashMap<String, QRow> = HashMap::new();
    let mut f: HashMap<String, Tensor> = HashMap::new();
    // int8 = every tensor run through a quantized linear; f32 = norms, embed, router gate + bias, AND lm_head
    // (never quantize the output head — its error lands directly in the logits; standard practice).
    let is_proj = |n: &str| n.contains("kv_a_proj_with_mqa")
        || n.contains("q_proj.weight") || n.contains("kv_b_proj.weight") || n.contains("o_proj.weight")
        || n.contains("gate_proj.weight") || n.contains("up_proj.weight") || n.contains("down_proj.weight");
    let names: Vec<String> = ld.map.keys().cloned().collect();
    let (mut nq, mut done) = (0usize, 0usize);
    for name in &names {
        let (v, shape) = ld.f32(name);
        if is_proj(name) {
            q.insert(name.clone(), Tensor::from_vec(&ctx, &v, &shape).quantize_rowwise(8)); nq += 1;
        } else {
            f.insert(name.clone(), Tensor::from_vec(&ctx, &v, &shape));
        }
        done += 1;
        if done % 800 == 0 { println!("  loaded {done}/{} ({nq} int8)  {:?}", names.len(), t0.elapsed()); }
    }
    println!("  loaded all: {nq} int8 projections + {} f32 tensors  ({:?})\n", f.len(), t0.elapsed());

    let lin = |x: &Tensor, n: &str| nn::linear_hf_q(x, q.get(n).unwrap_or_else(|| panic!("missing int8: {n}")));
    let gf = |n: &str| f.get(n).unwrap_or_else(|| panic!("missing f32: {n}"));

    // byte-level BPE tokenizer from tokenizer.json
    let tj: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{dir}/tokenizer.json")).unwrap()).unwrap();
    let vocab: HashMap<String, u32> = tj["model"]["vocab"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap() as u32)).collect();
    let idtok: HashMap<u32, String> = vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
    let merges: Vec<(String, String)> = tj["model"]["merges"].as_array().unwrap().iter().filter_map(|m| m.as_str().and_then(|s| s.split_once(' ')).map(|(a, b)| (a.to_string(), b.to_string()))).collect();
    let bpe = Bpe::new(vocab, &merges);
    let u2b = byte_decoder();
    let detok = |id: u32| -> String { let s = idtok.get(&id).cloned().unwrap_or_default(); String::from_utf8_lossy(&s.chars().filter_map(|c| u2b.get(&c).copied()).collect::<Vec<u8>>()).into_owned() };

    let prompt = "The capital of France is";
    let mut seq: Vec<u32> = vec![0]; // BOS
    seq.extend(bpe.encode(prompt));
    print!("\n{prompt}"); std::io::stdout().flush().ok();
    let deint = |t: &Tensor, rows: usize| t.reshape(&[rows, ROPE / 2, 2]).transpose(1, 2).contiguous().reshape(&[rows, ROPE]);

    for _step in 0..24 {                               // greedy generation (re-prefill each step)
    let ids: &[u32] = &seq;
    let s = ids.len();
    let mut stream0 = gf("model.embed_tokens.weight").gather_rows(ids);
    let mut stream_attn = stream0.clone();
    let cosn = cos.narrow(0, 0, s); let sinn = sin.narrow(0, 0, s);

    for l in 0..NL {
        let p = format!("model.layers.{l}");
        let residual = stream0.clone();
        // ---- Gated-MLA ----
        let hs = stream_attn.rmsnorm(gf(&format!("{p}.input_layernorm.weight")), EPS);
        let qf = lin(&hs, &format!("{p}.self_attn.q_proj.weight")).reshape(&[s, HEADS, QK]);
        let q_pass = qf.narrow(2, 0, NOPE).contiguous();
        let q_rot = deint(&qf.narrow(2, NOPE, ROPE).contiguous(), s * HEADS).reshape(&[s, HEADS * ROPE])
            .apply_rope_costable(&cosn, &sinn, HEADS, ROPE).reshape(&[s, HEADS, ROPE]);
        let ckv = lin(&hs, &format!("{p}.self_attn.kv_a_proj_with_mqa.weight"));
        let k_passc = ckv.narrow(1, 0, KVL).contiguous();
        let k_rot = ckv.narrow(1, KVL, ROPE).contiguous();
        let kb = lin(&k_passc.rmsnorm(gf(&format!("{p}.self_attn.kv_a_layernorm.weight")), EPS), &format!("{p}.self_attn.kv_b_proj.weight")).reshape(&[s, HEADS, NOPE + VH]);
        let k_nope = kb.narrow(2, 0, NOPE).contiguous();
        let value = kb.narrow(2, NOPE, VH).contiguous();
        let k_rot = deint(&k_rot, s).apply_rope_costable(&cosn, &sinn, 1, ROPE).reshape(&[s, 1, ROPE]).broadcast_to(&[s, HEADS, ROPE]).contiguous();
        let qh = q_pass.cat(&q_rot, 2).reshape(&[s, HEADS * QK]);
        let kh = k_nope.cat(&k_rot, 2).reshape(&[s, HEADS * QK]);
        let vv = value.reshape(&[s, HEADS * VH]);
        let qh = qh.mul(&qh.scalar(SCALING * (QK as f32).sqrt()));
        let ao = nn::causal_attention(&qh, &kh, &vv, HEADS, HEADS, 0.0);
        let gate = lin(&hs, &format!("{p}.self_attn.gate_proj.weight")).sigmoid();
        let attn = lin(&ao.mul(&gate), &format!("{p}.self_attn.o_proj.weight"));
        let residual = residual.add(&attn);

        // ---- MLP: dense (layer 0) or FarSkip-MoE ----
        let hm = stream0.rmsnorm(gf(&format!("{p}.post_attention_layernorm.weight")), EPS);
        if l == 0 {
            let dh = lin(&hm, &format!("{p}.mlp.gate_proj.weight")).silu().mul(&lin(&hm, &format!("{p}.mlp.up_proj.weight")));
            stream0 = residual.add(&lin(&dh, &format!("{p}.mlp.down_proj.weight"))); stream_attn = stream0.clone();
        } else {
            let logits = hm.matmul_bt(gf(&format!("{p}.mlp.gate.weight"))).to_vec().await;
            let bias = gf(&format!("{p}.mlp.gate.e_score_correction_bias")).to_vec().await;
            let sig = |z: f32| 1.0 / (1.0 + (-z).exp());
            let mut routed = vec![0f32; s * H];
            for t in 0..s {
                let sc: Vec<f32> = (0..E).map(|j| sig(logits[t * E + j])).collect();
                let sfc: Vec<f32> = (0..E).map(|j| sc[j] + bias[j]).collect();
                let mut ord: Vec<usize> = (0..E).collect();
                ord.sort_by(|&a, &b| sfc[b].total_cmp(&sfc[a]));
                let top = &ord[..TOPK];
                let wsum: f32 = top.iter().map(|&j| sc[j]).sum::<f32>() + 1e-20;
                let xt = hm.narrow(0, t, 1);
                for &ex in top {
                    let wt = sc[ex] / wsum * SCALE;
                    let eh = lin(&xt, &format!("{p}.mlp.experts.{ex}.gate_proj.weight")).silu().mul(&lin(&xt, &format!("{p}.mlp.experts.{ex}.up_proj.weight")));
                    let o = lin(&eh, &format!("{p}.mlp.experts.{ex}.down_proj.weight")).to_vec().await;
                    for i in 0..H { routed[t * H + i] += wt * o[i]; }
                }
            }
            let sh = lin(&hm, &format!("{p}.mlp.shared_experts.gate_proj.weight")).silu().mul(&lin(&hm, &format!("{p}.mlp.shared_experts.up_proj.weight")));
            let shared = lin(&sh, &format!("{p}.mlp.shared_experts.down_proj.weight")).to_vec().await;
            let res = residual.to_vec().await;
            let main: Vec<f32> = (0..s * H).map(|i| res[i] + routed[i] + shared[i]).collect();
            let nr: Vec<f32> = (0..s * H).map(|i| res[i] + shared[i]).collect();
            stream0 = Tensor::from_vec(&ctx, &main, &[s, H]);
            stream_attn = Tensor::from_vec(&ctx, &nr, &[s, H]);
        }
    }

    let hn = stream0.rmsnorm(gf("model.norm.weight"), EPS);
    let logits = hn.matmul_bt(gf("lm_head.weight")).to_vec().await; // f32 output head
    let last = &logits[(s - 1) * VOCAB..];
    if _step == 0 { // save first-prefill logits for the golden re-comparison
        let bytes: Vec<u8> = last.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(format!("{home}/.cache/ferric/instella_ref/ferric_logits.bin"), &bytes).unwrap();
    }
    let next = (0..VOCAB).max_by(|&a, &b| last[a].total_cmp(&last[b])).unwrap() as u32;
    if next == 1 { break; } // EOS
    print!("{}", detok(next)); std::io::stdout().flush().ok();
    seq.push(next);
    }
    println!("\n\n✅ AMD Instella-MoE 16B — int8, on-device, pure Rust Ferric — generated the text above ({} tokens total).", seq.len());
}
