//! **Parakeet against NeMo, stage by stage** — the Ferric half of the conformance harness.
//!
//! A transcript is a weak witness for a speech model: a whole class of convention errors (the valid
//! frame count, the tail masks, a tie-break) leaves every word right on every clip anyone looked at.
//! So the reference, NVIDIA NeMo run on this GGUF's own weights, records sampled values at every
//! stage into a fixture, and this program prints Ferric's values at exactly those frames and
//! channels. It does not judge; the gate compares. Keeping the verdict out of here means the same
//! output serves the clean run and every `FERRIC_ASR_NEG_*` negative control.
//!
//!   cargo run -p ferric-llama --release --example parakeet_stages -- <model.gguf> <audio.wav|flac> <fixture.json>
//!
//! The fixture is `parakeet-conformance/1` (written by the NeMo generator). Output, one record per line:
//! ```text
//! PCM <sha256 of the little-endian int16 samples>
//! VALID <valid mel frames> <encoder frames>
//! SHAPE <stage> <rows> <cols>                           Ferric's own tensor, per recorded stage
//! STAGE <stage> <row> <sum> <ssq> <v_col0> <v_col1> …  sum/ssq over ALL cols of the row; `nan` if
//!                                                        Ferric has no such row
//! STEP <k> <t> <u> <argmax> <max> <lse> <v_col0> …     RNN-T: raw joint logits TEACHER-FORCED along
//!                                                        the fixture's own steps
//! CTC <t> <argmax> <max> <lse> <v_col0> …              CTC: raw head logits, EVERY frame
//! TOKENS <id> …                                          free-running greedy ids
//! TEXT <transcript>
//! TAPS_INERT <max |encode() − encode_stages()| at the encoder output>
//! ```
//! Stages run `logmel`, `mel`, `pre_conv`, `pre_out`, `pe`, `block.<i>`, `enc` and — for a
//! prompt-conditioned model — `prompt`, the MLP output the joint actually sees. The fixture's
//! `config` (attention context, prompt index) is applied before anything runs.
//! ```text
//! ```
//! Run with `FERRIC_METAL4` and `FERRIC_COOP` unset: both switch in reduced-precision kernels.
use ferric_gguf::GgufFile;
use ferric_llama::parakeet::Parakeet;
use serde_json::Value;
use std::sync::Arc;

// The workspace's own SHA-256, checked against the published vectors in ferric-signal. Included by
// path rather than as a dependency: one example should not add an edge to the crate graph.
#[path = "../../ferric-signal/src/sha256.rs"]
mod sha256;

/// 16-bit PCM WAV reader — enough for `ffmpeg -ar 16000 -ac 1 -c:a pcm_s16le`. Returns the raw
/// samples, because the fixture pins the audio by the hash of exactly these.
fn read_wav(path: &str) -> (Vec<i16>, usize) {
    let b = std::fs::read(path).expect("read wav");
    assert_eq!(&b[0..4], b"RIFF", "not a RIFF file");
    let (mut i, mut rate, mut pcm) = (12usize, 0usize, Vec::new());
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let sz = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
        if id == b"fmt " {
            let ch = u16::from_le_bytes([b[i + 10], b[i + 11]]) as usize;
            rate = u32::from_le_bytes([b[i + 12], b[i + 13], b[i + 14], b[i + 15]]) as usize;
            let bits = u16::from_le_bytes([b[i + 22], b[i + 23]]);
            assert_eq!(ch, 1, "expected mono");
            assert_eq!(bits, 16, "expected 16-bit PCM");
        } else if id == b"data" {
            let d = &b[i + 8..(i + 8 + sz).min(b.len())];
            pcm = d.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
            break;
        }
        i += 8 + sz + (sz & 1);
    }
    (pcm, rate)
}

/// FLAC goes through ffmpeg to a 16-bit WAV, as `librispeech_wer` does. The decode is lossless and
/// the PCM hash below is checked against the fixture, so a decoder difference cannot pass silently.
fn read_audio(path: &str) -> (Vec<i16>, usize) {
    if !path.ends_with(".flac") { return read_wav(path); }
    let tmp = std::env::temp_dir().join(format!("parakeet_stages_{}.wav", std::process::id()));
    let ok = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i", path, "-ar", "16000", "-ac", "1",
               "-c:a", "pcm_s16le", tmp.to_str().unwrap()])
        .status().map(|s| s.success()).unwrap_or(false);
    assert!(ok, "ffmpeg could not decode {path}");
    let r = read_wav(tmp.to_str().unwrap());
    let _ = std::fs::remove_file(&tmp);
    r
}

fn idx(v: &Value) -> Vec<usize> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect()).unwrap_or_default()
}

/// `max` and `log Σ exp` of a logit row, in f64 so the lse is not itself a rounding source.
fn max_lse(row: &[f32]) -> (f32, f64) {
    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let s: f64 = row.iter().map(|&x| (x as f64 - m as f64).exp()).sum();
    (m, m as f64 + s.ln())
}

fn vals(row: &[f32], cols: &[usize]) -> String {
    cols.iter().map(|&c| row.get(c).map_or("nan".to_string(), |v| format!("{v:e}")))
        .collect::<Vec<_>>().join(" ")
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let usage = "usage: parakeet_stages <model.gguf> <audio.wav|flac> <fixture.json>";
    let (model, audio, fixture) = (a.get(1).expect(usage), a.get(2).expect(usage), a.get(3).expect(usage));
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(fixture).expect("read fixture"))
        .expect("fixture is not JSON");
    assert_eq!(fx["format"].as_str(), Some("parakeet-conformance/1"), "unknown fixture format");

    let g = GgufFile::open(model).expect("open gguf");
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    let mut m = Parakeet::load(&ctx, &g).expect("load");
    // The settings the reference ran at, when the fixture pins them: a limited-context model's
    // attention context, and a prompt model's language index. Both sides must share them — each
    // is a different transcript.
    let cfg = &fx["config"];
    if let Some(a) = cfg["att_context"].as_array() {
        let v: Vec<usize> = a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect();
        m.set_att_context(v[0], v[1]).expect("the fixture's attention context");
    }
    if let Some(id) = cfg["prompt_id"].as_u64() {
        assert!(m.prompt.is_some(), "the fixture sets a prompt index; this model takes none");
        m.prompt_id = Some(id as usize);
    }
    eprintln!("{}", m.describe());

    let (pcm16, rate) = read_audio(audio);
    assert_eq!(rate, m.cfg.sample_rate, "audio is {rate} Hz, model wants {}", m.cfg.sample_rate);
    let bytes: Vec<u8> = pcm16.iter().flat_map(|s| s.to_le_bytes()).collect();
    let sha = sha256::hex(&sha256::sha256(&bytes));
    println!("PCM {sha}");
    if let Some(want) = fx["clip"]["pcm_sha256"].as_str() {
        if want != sha {
            eprintln!("PCM MISMATCH: fixture {want}, audio {sha} — not the clip the reference ran on");
            std::process::exit(3);
        }
    }
    let pcm: Vec<f32> = pcm16.iter().map(|&s| s as f32 / 32768.0).collect();

    // Only the stages the fixture recorded, in the order the forward pass produces them.
    let recorded: Vec<String> = fx["stages"].as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
    let order: Vec<String> = ["logmel", "mel", "pre_conv", "pre_out", "pe"].iter().map(|s| s.to_string())
        .chain((0..m.blocks.len()).map(|i| format!("block.{i}")))
        .chain(["enc", "prompt"].iter().map(|s| s.to_string()))
        .collect();
    for r in &recorded {
        if !order.contains(r) { eprintln!("fixture stage {r:?} is not one this runtime taps"); }
    }
    let want = |n: &str| recorded.iter().any(|r| r == n);
    let (enc, stages) = m.encode_stages(&pcm, &want).await.expect("encode");
    let t_enc = enc.shape[0];
    let valid = stages.iter().find(|s| s.name == "mel").map_or(0, |s| s.rows);
    println!("VALID {valid} {t_enc}");

    for name in order.iter().filter(|n| want(n)) {
        let s = stages.iter().find(|s| &s.name == name).expect("tapped");
        let spec = &fx["stages"][name.as_str()];
        let (rows, cols) = (idx(&spec["rows"]), idx(&spec["cols"]));
        println!("SHAPE {name} {} {}", s.rows, s.cols);
        for r in rows {
            if r >= s.rows {
                println!("STAGE {name} {r} nan nan {}", vec!["nan"; cols.len()].join(" "));
                continue;
            }
            let row = &s.data[r * s.cols..(r + 1) * s.cols];
            let sum: f64 = row.iter().map(|&v| v as f64).sum();
            let ssq: f64 = row.iter().map(|&v| v as f64 * v as f64).sum();
            println!("STAGE {name} {r} {sum:e} {ssq:e} {}", vals(row, &cols));
        }
    }

    let tokens = if m.rnnt.is_some() {
        let host = m.rnnt_host(&enc).await.expect("rnnt host");
        let steps: Vec<(usize, usize, u32)> = fx["rnnt"]["steps"].as_array().map(|a| a.iter().map(|s| {
            let v = idx(s);
            (v[0], v[1], v[2] as u32)
        }).collect()).unwrap_or_default();
        let cols = idx(&fx["rnnt"]["cols"]);
        match m.rnnt_teacher_forced(&host, &steps) {
            Ok(rows) => for (k, (row, &(t, u, _))) in rows.iter().zip(&steps).enumerate() {
                let (mx, lse) = max_lse(row);
                println!("STEP {k} {t} {u} {} {mx:e} {lse:e} {}",
                         ferric_llama::parakeet::argmax_first(row), vals(row, &cols));
            },
            // A replay that cannot follow the reference is reported, not papered over.
            Err(e) => eprintln!("teacher forcing stopped: {e}"),
        }
        m.rnnt_greedy(&host).tokens
    } else {
        let logits = m.ctc_logits(&enc).await.expect("ctc logits");
        let nv = m.cfg.vocab;
        let cols = idx(&fx["ctc"]["cols"]);
        let (frames, ids) = m.ctc_ids(&logits, t_enc);
        for (t, &am) in frames.iter().enumerate() {
            let row = &logits[t * nv..(t + 1) * nv];
            let (mx, lse) = max_lse(row);
            println!("CTC {t} {am} {mx:e} {lse:e} {}", vals(row, &cols));
        }
        ids
    };
    println!("TOKENS{}", tokens.iter().map(|t| format!(" {t}")).collect::<String>());
    println!("TEXT {}", m.detok(&tokens));

    // A tap that changed what it recorded would make every number above describe a different run.
    // Re-encode without taps and require the encoder output to agree.
    let plain = m.encode(&pcm).expect("encode").to_vec().await;
    let tapped = enc.to_vec().await;
    let inert = if plain.len() != tapped.len() { f32::INFINITY } else {
        plain.iter().zip(&tapped).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
    };
    println!("TAPS_INERT {inert:e}");
}
