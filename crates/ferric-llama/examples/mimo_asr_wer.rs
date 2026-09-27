//! **MiMo-V2.5-ASR on LibriSpeech, and one clip at a time** — transcripts, corpus WER, and the numbers
//! an energy measurement divides by.
//!
//!   mimo_asr_wer <asr-dir> <tokenizer-dir> <audio.wav> [greedy.json]      one clip: text + ids
//!   mimo_asr_wer <asr-dir> <tokenizer-dir> <LibriSpeech/test-clean> N      corpus WER, first N by id
//!
//! The model is loaded ONCE and every utterance reuses it, which is what a deployment does; timing a
//! fresh process per clip would charge each one a 15 GB load.
//!
//! ⚠ CORPUS WER (total edits / total reference words), normalised the LibriSpeech way on both sides:
//! uppercase, letters and apostrophes. MiMo writes punctuation and casing natively; the normalisation
//! removes both before scoring, identically for the reference, so it cannot flatter the hypothesis.
//! The prompt is the authors' first English template with the `<english>` tag, as their demo sends.
use ferric_llama::mimo_asr::MimoAsr;
use ferric_tokenizer::{Bpe, Pre};
use std::collections::HashMap;
use std::sync::Arc;

const TEMPLATE: &str = "Please transcribe this audio file";
const TAG: &str = "<english>";

fn read_wav(path: &str) -> (Vec<f32>, usize) {
    let b = std::fs::read(path).expect("read wav");
    let (mut i, mut rate, mut pcm) = (12usize, 0usize, Vec::new());
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let sz = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
        let d = &b[i + 8..(i + 8 + sz).min(b.len())];
        if id == b"fmt " {
            assert_eq!(u16::from_le_bytes([d[0], d[1]]), 1, "16-bit PCM only (format tag 1)");
            assert_eq!(u16::from_le_bytes([d[2], d[3]]), 1, "mono only");
            assert_eq!(u16::from_le_bytes([d[14], d[15]]), 16, "16-bit PCM only");
            rate = u32::from_le_bytes([d[4], d[5], d[6], d[7]]) as usize;
        }
        if id == b"data" { pcm = d.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect(); }
        i += 8 + sz + (sz & 1);
    }
    (pcm, rate)
}

fn norm(s: &str) -> Vec<String> {
    s.to_uppercase().split_whitespace()
        .map(|w| w.chars().filter(|c| c.is_ascii_alphabetic() || *c == '\'').collect::<String>())
        .filter(|w| !w.is_empty()).collect()
}

fn edits(r: &[String], h: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    let mut cur = vec![0usize; h.len() + 1];
    for i in 1..=r.len() {
        cur[0] = i;
        for j in 1..=h.len() {
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + usize::from(r[i - 1] != h[j - 1]));
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[h.len()]
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let (asr_dir, tok_dir, target) = (&a[1], &a[2], &a[3]);
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    let t0 = std::time::Instant::now();
    let m = MimoAsr::load(&ctx, asr_dir, tok_dir).expect("MiMo-ASR");
    let bpe = Bpe::from_tokenizer_json_with_pre(&std::fs::read(format!("{asr_dir}/tokenizer.json")).expect("tokenizer.json"),
                                                 Pre::Qwen2).expect("bpe");
    let enc = |t: &str| bpe.encode(t);
    // Special and added tokens are not in the BPE vocabulary; the transcript never needs them.
    let text = |ids: &[u32]| -> String {
        bpe.decode(&ids.iter().copied().filter(|&i| i < m.sp.endoftext).collect::<Vec<_>>())
            .replace("<english>", "").replace("<chinese>", "").trim().to_string()
    };
    eprintln!("loaded in {:.1} s", t0.elapsed().as_secs_f64());

    if target.ends_with(".wav") {
        let (pcm, rate) = read_wav(target);
        let t1 = std::time::Instant::now();
        let (ids, stop) = m.transcribe_ids(&pcm, rate, TEMPLATE, TAG, &enc, 256).await.expect("transcribe");
        let dt = t1.elapsed().as_secs_f64();
        println!("{}", text(&ids));
        // the ids and the stop token: what `mimo_asr_ref.py --greedy` checks, stopping included
        let with_stop: Vec<u32> = ids.iter().copied().chain(stop).collect();
        println!("IDS {}", with_stop.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
        eprintln!("{:.2} s audio in {dt:.2} s ({:.1}x realtime), {} tokens", pcm.len() as f64 / rate as f64,
                  pcm.len() as f64 / rate as f64 / dt, ids.len());
        if let Some(g) = a.get(4) {
            let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(g).expect("greedy")).expect("json");
            let want: Vec<u32> = j["continuation"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
            // ⛔ The prefix alone cannot fail when Ferric runs ON past the authors' stop: zip() ends at the
            // shorter list. Both lengths are printed and the gate requires them equal.
            let n = with_stop.iter().zip(&want).take_while(|(x, y)| x == y).count();
            println!("MATCH {n}/{} LEN {} {}", want.len(), with_stop.len(), want.len());
        }
        return;
    }

    let limit: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let mut refs: HashMap<String, String> = HashMap::new();
    let mut flacs: Vec<(String, String)> = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(target)];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).expect("read dir").flatten() {
            let p = e.path();
            if p.is_dir() { stack.push(p); continue; }
            match p.extension().and_then(|x| x.to_str()) {
                Some("txt") if p.to_string_lossy().ends_with(".trans.txt") => {
                    for line in std::fs::read_to_string(&p).expect("trans").lines() {
                        if let Some((id, t)) = line.split_once(' ') { refs.insert(id.into(), t.into()); }
                    }
                }
                Some("flac") => flacs.push((p.file_stem().unwrap().to_string_lossy().to_string(), p.to_string_lossy().to_string())),
                _ => {}
            }
        }
    }
    flacs.sort();
    flacs.truncate(limit);
    let tmp = std::env::temp_dir().join("mimo_wer.wav");
    let (mut tot_err, mut tot_words, mut audio_s, mut model_s, mut tokens) = (0usize, 0usize, 0f64, 0f64, 0usize);
    let mut dropped: Vec<String> = Vec::new();
    for (k, (id, path)) in flacs.iter().enumerate() {
        let Some(reference) = refs.get(id) else { dropped.push(format!("{id}: no reference")); continue };
        let ok = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i", path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", tmp.to_str().unwrap()])
            .status().map(|s| s.success()).unwrap_or(false);
        if !ok { dropped.push(format!("{id}: ffmpeg failed")); continue; }
        let (pcm, rate) = read_wav(tmp.to_str().unwrap());
        audio_s += pcm.len() as f64 / rate as f64;
        let t1 = std::time::Instant::now();
        let (ids, _) = m.transcribe_ids(&pcm, rate, TEMPLATE, TAG, &enc, 256).await.expect("transcribe");
        model_s += t1.elapsed().as_secs_f64();
        tokens += ids.len();
        let (r, h) = (norm(reference), norm(&text(&ids)));
        let e = edits(&r, &h);
        tot_err += e;
        tot_words += r.len();
        println!("UTT {id} {e} {} | {}", r.len(), text(&ids));
        if (k + 1) % 25 == 0 {
            eprintln!("  {}/{} — running WER {:.2}%", k + 1, flacs.len(), 100.0 * tot_err as f64 / tot_words as f64);
        }
    }
    // ⛔ A dropped utterance would leave the WER computed over fewer files than the line claims; with no
    // ffmpeg at all it read "WER 0.00%". Refused instead.
    if !dropped.is_empty() {
        eprintln!("⛔ {} of {} utterances were not scored: {}", dropped.len(), flacs.len(), dropped.join("; "));
        std::process::exit(2);
    }
    println!("\nutterances {} | words {tot_words} | edits {tot_err} | tokens {tokens}", flacs.len());
    println!("WER {:.2}%", 100.0 * tot_err as f64 / tot_words.max(1) as f64);
    println!("audio {audio_s:.0} s, model time {model_s:.0} s — {:.2}x realtime (excludes ffmpeg and load)", audio_s / model_s);
}
