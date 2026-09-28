//! **A quantized checkpoint's logits — or its dequantized weights — from Ferric**, the Ferric side of
//! `scripts/quant_formats_conformance.sh`.
//!
//!   quant_logits <model.gguf | hf_dir> <ids.json>             ROW lines, as `lm_logits` prints them
//!   quant_logits <model.gguf | hf_dir> --weights <out.bin>     every quantized weight, dequantized
//!
//! The model is either a GGUF file (IQ1_S/IQ1_M/IQ2_XS/IQ2_S/IQ3_S/NVFP4 and the rest) or a Hugging Face
//! checkpoint directory holding GPTQ / AWQ / compressed-tensors / FP8 / ModelOpt weights, opened by
//! `HfCheckpoint` — the quantized linears then run through `ferric-tensor::gq`, packed.
//!
//! `--weights` writes, for every quantized tensor, `u32 name_len, name, u64 n, f32[n]` — Ferric's
//! dequantization, which the gate compares bit for bit against the defining library's.
use ferric_gguf::{GgufFile, GgufSource};
use ferric_llama::qwen3::Qwen3;
use ferric_load::hf::HfCheckpoint;
use std::io::Write;
use std::sync::Arc;

fn field<'a>(s: &'a str, key: &str) -> &'a str {
    let k = format!("\"{key}\": [");
    let i = s.find(&k).unwrap_or_else(|| panic!("fixture has no \"{key}\"")) + k.len();
    let j = s[i..].find(']').unwrap() + i;
    &s[i..j]
}

fn main() { pollster::block_on(run()); }

async fn go<G: GgufSource>(g: &G, a: &[String]) {
    if a.get(2).map(String::as_str) == Some("--weights") {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&a[3]).expect("create"));
        let mut names: Vec<String> = Vec::new();
        for il in 0.. {
            let mut any = false;
            for s in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"] {
                let n = format!("blk.{il}.{s}.weight");
                if g.tensor(&n).is_some() { names.push(n); any = true; }
            }
            if !any { break }
        }
        let mut quantized = 0;
        for n in &names {
            let ty = g.tensor(n).unwrap().ggml_type;
            if matches!(ty, 0 | 1 | 30) { continue } // not quantized
            let v = g.dequant(n).unwrap_or_else(|e| panic!("{n}: {e}"));
            f.write_all(&(n.len() as u32).to_le_bytes()).unwrap();
            f.write_all(n.as_bytes()).unwrap();
            f.write_all(&(v.len() as u64).to_le_bytes()).unwrap();
            for x in &v { f.write_all(&x.to_le_bytes()).unwrap(); }
            quantized += 1;
        }
        eprintln!("wrote {quantized} quantized weights of {} projections", names.len());
        return;
    }
    let fx = std::fs::read_to_string(a.get(2).expect("ids.json")).expect("read ids");
    let parse = |key: &str| -> Vec<u32> {
        field(&fx, key).split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().unwrap()).collect()
    };
    let (ids, sample) = (parse("ids"), parse("sample_ids"));
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let m = Qwen3::load(&ctx, g).expect("load");
    let lg = m.forward(&ids).to_vec().await;
    if lg.iter().all(|&x| x == 0.0) {
        eprintln!("the device returned ALL-ZERO logits — a dropped buffer, not a model output; rerun");
        std::process::exit(3);
    }
    let v = lg.len() / ids.len();
    if let Ok(p) = std::env::var("LM_LOGITS_DUMP") {
        std::fs::write(&p, lg.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).expect("dump");
    }
    for t in 0..ids.len() {
        let r = &lg[t * v..(t + 1) * v];
        let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
        let (mut sum, mut ssq) = (0f64, 0f64);
        for (i, &x) in r.iter().enumerate() {
            if x > bv { bv = x; best = i; }
            sum += x as f64; ssq += (x as f64) * (x as f64);
        }
        let s: Vec<String> = sample.iter().map(|&i| format!("{:.8e}", r[i as usize])).collect();
        println!("ROW {t} {best} {sum:.6} {ssq:.6} {}", s.join(" "));
    }
}

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let mp = a.get(1).expect("usage: quant_logits <model.gguf | hf_dir> <ids.json | --weights out.bin>");
    if std::path::Path::new(mp).is_dir() {
        let hf = HfCheckpoint::open(mp).expect("open HF checkpoint");
        eprintln!("hf {} — {}", hf.arch, hf.quant.as_ref().map(|q| q.summary.clone()).unwrap_or_else(|| "not quantized".into()));
        go(&hf, &a).await;
    } else {
        let g = GgufFile::open(mp).expect("open gguf");
        go(&g, &a).await;
    }
}
