//! **MiMo-V2.5-ASR against its authors' code, stage by stage** — the Ferric half of
//! `scripts/mimo_asr_conformance.sh`.
//!
//! Runs the whole path from a 16 kHz WAV: the resampler, the mel frontend, the tokenizer encoder, the
//! RVQ codes, the patch encoder, Ferric's own prompt (tokenised by Ferric), and the language model to
//! logits — from the authors' two safetensors directories, no conversion. It prints Ferric's values at
//! exactly the rows and columns the fixture recorded and does not judge; the gate compares.
//!
//!   cargo run -p ferric-llama --release --example mimo_asr_stages -- <asr-dir> <tokenizer-dir> <audio.wav> <fixture.json>
//!
//! ```text
//! WAV24 <n> <sum> <ssq> <v_pos0> …        the resampled audio, at the fixture's sample positions
//! STAGE <stage> <row> <sum> <ssq> <v…>     sum/ssq over the WHOLE row
//! CODES <channel> <code> …
//! TEXTIDS <id> …                           the prompt Ferric builds, one id per LM position
//! ROW <t> <argmax> <sum> <ssq> <v_id0> …   logits
//! ```
//! `FERRIC_ASR_TOK_ONLY=1` stops after the codes (no LM load). Controls: `FERRIC_ASR_NEG` (see
//! `mimo_asr::Neg`).
use ferric_llama::mimo_asr::{resample, AudioTokenizer, MimoAsr};
use ferric_tokenizer::{Bpe, Pre};
use serde_json::Value;
use std::sync::Arc;

fn read_wav(path: &str) -> (Vec<f32>, usize) {
    let b = std::fs::read(path).expect("read wav");
    let (mut i, mut rate, mut pcm) = (12usize, 0usize, Vec::new());
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let sz = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
        let d = &b[i + 8..(i + 8 + sz).min(b.len())];
        if id == b"fmt " {
            assert_eq!(u16::from_le_bytes([d[2], d[3]]), 1, "mono only");
            assert_eq!(u16::from_le_bytes([d[14], d[15]]), 16, "16-bit PCM only");
            rate = u32::from_le_bytes([d[4], d[5], d[6], d[7]]) as usize;
        }
        if id == b"data" { pcm = d.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect(); }
        i += 8 + sz + (sz & 1);
    }
    (pcm, rate)
}

fn u32s(v: &Value) -> Vec<u32> { v.as_array().expect("array").iter().map(|x| x.as_u64().unwrap() as u32).collect() }

fn print_rows(tag: &str, x: &[f32], w: usize, cols: &[u32]) {
    for (r, row) in x.chunks_exact(w).enumerate() {
        let (mut s, mut q) = (0f64, 0f64);
        for &v in row { s += v as f64; q += (v as f64) * (v as f64); }
        let vals: Vec<String> = cols.iter().map(|&c| format!("{:e}", row[c as usize])).collect();
        println!("{tag} {r} {s:e} {q:e} {}", vals.join(" "));
    }
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let (asr_dir, tok_dir, wav, fxp) = (&a[1], &a[2], &a[3], &a[4]);
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(fxp).expect("fixture")).expect("json");
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    eprintln!("adapter: {} [{:?}]", ctx.adapter_name, ctx.backend);

    let (pcm, rate) = read_wav(wav);
    let tok_only = std::env::var("FERRIC_ASR_TOK_ONLY").is_ok();
    // The whole recogniser, or the tokenizer alone when only its stages are wanted.
    let (m, tok_alone) = if tok_only {
        (None, Some(AudioTokenizer::load(&ctx, tok_dir).expect("audio tokenizer")))
    } else {
        (Some(MimoAsr::load(&ctx, asr_dir, tok_dir).expect("MiMo-ASR")), None)
    };
    let tok = m.as_ref().map(|m| &m.tokenizer).or(tok_alone.as_ref()).unwrap();
    let x = resample(&pcm, rate, tok.cfg.mel.sr);
    let (mut s, mut q) = (0f64, 0f64);
    for &v in &x { s += v as f64; q += (v as f64) * (v as f64); }
    let pos = u32s(&fx["wav24"]["pos"]);
    println!("WAV24 {} {s:e} {q:e} {}", x.len(),
             pos.iter().map(|&p| format!("{:e}", x[p as usize])).collect::<Vec<_>>().join(" "));

    let group = m.as_ref().map(|m| m.group).unwrap_or(4);
    let mut taps = Vec::new();
    let codes = tok.encode(&x, group, Some(&mut taps)).await;
    for (name, t) in &taps {
        let Some(rec) = fx["stages"].get(name) else { continue };
        print_rows(&format!("STAGE {name}"), &t.to_vec().await, t.shape[1], &u32s(&rec["cols"]));
    }
    for (c, ch) in codes.iter().enumerate() {
        println!("CODES {c} {}", ch.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "));
    }
    drop(taps);
    let Some(m) = m else { return };

    let mut ptaps = Vec::new();
    let rows = m.patch.encode(&codes, Some(&mut ptaps)).expect("patch encode");
    for (name, t) in &ptaps {
        let Some(rec) = fx["stages"].get(name) else { continue };
        print_rows(&format!("STAGE {name}"), &t.to_vec().await, t.shape[1], &u32s(&rec["cols"]));
    }

    let bpe = Bpe::from_tokenizer_json_with_pre(&std::fs::read(format!("{asr_dir}/tokenizer.json"))
        .expect("tokenizer.json"), Pre::Qwen2).expect("bpe");
    let enc = |t: &str| bpe.encode(t);
    let ids = m.prompt_ids(rows.shape[0], "Please transcribe this audio file", "<english>", &enc);
    println!("TEXTIDS {}", ids.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "));
    let x = m.prompt_embeds(&ids, &rows).expect("embeds");
    if let Some(rec) = fx["stages"].get("inputs_embeds") {
        print_rows("STAGE inputs_embeds", &x.to_vec().await, x.shape[1], &u32s(&rec["cols"]));
    }
    let mut cache = ferric_llama::qwen3::Cache::new(&m.lm.cfg);
    let lg = m.lm.forward_embeds(&x, &mut cache).to_vec().await;
    let sample = u32s(&fx["sample_ids"]);
    let v = lg.len() / ids.len();
    for t in 0..ids.len() {
        let r = &lg[t * v..(t + 1) * v];
        let (mut best, mut bv, mut s, mut q) = (0usize, f32::NEG_INFINITY, 0f64, 0f64);
        for (i, &x) in r.iter().enumerate() {
            if x > bv { bv = x; best = i; }
            s += x as f64; q += (x as f64) * (x as f64);
        }
        let vals: Vec<String> = sample.iter().map(|&i| format!("{:.5}", r[i as usize])).collect();
        println!("ROW {t} {best} {s:.4} {q:.4} {}", vals.join(" "));
    }
}
