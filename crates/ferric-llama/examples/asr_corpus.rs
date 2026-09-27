//! **One ASR model over a prepared corpus, with the work bracketed for an energy meter** — the Ferric
//! side of `scripts/asr_joules.sh`.
//!
//!   asr_corpus parakeet <model.gguf>                     <corpus-dir>
//!   asr_corpus mimo     <asr-dir> <tokenizer-dir>        <corpus-dir>
//!
//! `<corpus-dir>` holds `<id>.wav` (16 kHz mono, 16-bit PCM or 32-bit float) and `refs.txt` (`<id> <TRANSCRIPT>` per line) —
//! prepared ONCE, so no decoder runs inside the measured window. The model loads, every WAV is read
//! into memory, and then the corpus runs in `ASR_CHUNKS` chunks (default 4) with an idle gap of
//! `ASR_IDLE_S` seconds (default 10) before each chunk and after the last:
//!
//! ```text
//! IDLE <t0> <t1>           the process sleeps: the machine's draw WITHOUT this work, measured next to it
//! RUN  <t0> <t1> <s> <n>   a chunk of <n> utterances back to back, nothing else; <s> seconds of audio
//! ```
//!
//! ⭐ WHY INTERLEAVED. This machine is shared: other sessions start and stop CPU and GPU work at will, so
//! one idle baseline taken before a 20-minute run describes a machine that no longer exists by the end
//! of it. Each chunk is charged against the idle gaps on either side of it, and the gate refuses when
//! those gaps disagree with each other by more than the work they are subtracted from.
//!
//! ⚠ Corpus WER (total edits / total reference words), LibriSpeech normalisation on both sides.
//! "Correct words" is reported CONSERVATIVELY as reference words minus edits, so an insertion costs a
//! correct word too — the denominator of a joules-per-correct-word figure can only be understated.
use ferric_gguf::GgufFile;
use ferric_llama::mimo_asr::MimoAsr;
use ferric_llama::parakeet::Parakeet;
use ferric_tokenizer::{Bpe, Pre};
use std::sync::Arc;

/// Mono 16-bit PCM or 32-bit IEEE float — the Open ASR Leaderboard's ESB parquet stores float WAV — and
/// nothing else: a float file read as int16 pairs is noise that still "transcribes".
fn read_wav(path: &std::path::Path) -> (Vec<f32>, usize) {
    let b = std::fs::read(path).expect("read wav");
    assert!(b.len() >= 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WAVE", "{}: not a RIFF/WAVE file", path.display());
    let (mut i, mut fmt, mut data) = (12usize, None, None);
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let sz = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
        let d = &b[i + 8..(i + 8 + sz).min(b.len())];
        if id == b"fmt " {
            let u16at = |k: usize| u16::from_le_bytes([d[k], d[k + 1]]);
            // WAVE_FORMAT_EXTENSIBLE carries the real format tag at the head of its sub-format GUID.
            let tag = if u16at(0) == 0xFFFE { u16at(24) } else { u16at(0) };
            fmt = Some((tag, u16at(2), u32::from_le_bytes([d[4], d[5], d[6], d[7]]) as usize, u16at(14)));
        }
        if id == b"data" { data = Some(d); }
        i += 8 + sz + (sz & 1);
    }
    let ((tag, ch, rate, bits), d) = (fmt.expect("no fmt chunk"), data.expect("no data chunk"));
    assert_eq!(ch, 1, "{}: {ch} channels, mono only", path.display());
    let pcm = match (tag, bits) {
        (1, 16) => d.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect(),
        (3, 32) => d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        _ => panic!("{}: WAV format tag {tag}, {bits} bits — only 16-bit PCM and 32-bit float are read", path.display()),
    };
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

fn now() -> f64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64() }

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let kind = a[1].as_str();
    let dir = std::path::PathBuf::from(a.last().unwrap());
    let refs: Vec<(String, String)> = std::fs::read_to_string(dir.join("refs.txt")).expect("refs.txt").lines()
        .filter_map(|l| l.split_once(' ').map(|(i, t)| (i.to_string(), t.to_string()))).collect();
    let audio: Vec<(Vec<f32>, usize)> = refs.iter().map(|(id, _)| read_wav(&dir.join(format!("{id}.wav")))).collect();
    let audio_s: f64 = audio.iter().map(|(p, r)| p.len() as f64 / *r as f64).sum();
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    eprintln!("adapter: {} [{:?}]; {} utterances, {audio_s:.0} s of audio", ctx.adapter_name, ctx.backend, refs.len());

    // Each model as a closure from PCM to a transcript, loaded before the window opens.
    let mut hyps: Vec<String> = Vec::with_capacity(refs.len());
    let chunks: usize = std::env::var("ASR_CHUNKS").ok().and_then(|v| v.parse().ok()).unwrap_or(4).max(1);
    let idle_s: f64 = std::env::var("ASR_IDLE_S").ok().and_then(|v| v.parse().ok()).unwrap_or(10.0);
    let per = audio.len().div_ceil(chunks);
    let idle = || {
        let a = now();
        std::thread::sleep(std::time::Duration::from_secs_f64(idle_s));
        println!("IDLE {a:.3} {:.3}", now());
    };
    let mut t1 = 0f64; // seconds inside RUN windows
    match kind {
        "parakeet" => {
            let g = GgufFile::open(&a[2]).expect("gguf");
            let m = Parakeet::load(&ctx, &g).expect("parakeet");
            let _ = m.transcribe(&audio[0].0).expect("warm-up");   // pipelines compile outside the window
            for block in audio.chunks(per) {
                idle();
                let a = now();
                for (pcm, rate) in block {
                    assert_eq!(*rate, m.cfg.sample_rate, "corpus must be {} Hz", m.cfg.sample_rate);
                    hyps.push(m.transcribe(pcm).expect("transcribe"));
                }
                let b = now();
                let secs: f64 = block.iter().map(|(p, r)| p.len() as f64 / *r as f64).sum();
                println!("RUN {a:.3} {b:.3} {secs:.3} {}", block.len());
                t1 += b - a;
            }
            idle();
        }
        "mimo" => {
            let m = MimoAsr::load(&ctx, &a[2], &a[3]).expect("MiMo-ASR");
            let bpe = Bpe::from_tokenizer_json_with_pre(&std::fs::read(format!("{}/tokenizer.json", a[2])).expect("tokenizer.json"),
                                                         Pre::Qwen2).expect("bpe");
            let enc = |t: &str| bpe.encode(t);
            let text = |ids: &[u32]| bpe.decode(&ids.iter().copied().filter(|&i| i < m.sp.endoftext).collect::<Vec<_>>())
                .replace("<english>", "").trim().to_string();
            let tpl = "Please transcribe this audio file";
            let _ = m.transcribe_ids(&audio[0].0, audio[0].1, tpl, "<english>", &enc, 256).await.expect("warm-up");
            for block in audio.chunks(per) {
                idle();
                let a = now();
                for (pcm, rate) in block {
                    let (ids, _) = m.transcribe_ids(pcm, *rate, tpl, "<english>", &enc, 256).await.expect("transcribe");
                    hyps.push(text(&ids));
                }
                let b = now();
                let secs: f64 = block.iter().map(|(p, r)| p.len() as f64 / *r as f64).sum();
                println!("RUN {a:.3} {b:.3} {secs:.3} {}", block.len());
                t1 += b - a;
            }
            idle();
        }
        other => panic!("unknown model kind {other} (parakeet|mimo)"),
    }
    let (mut words, mut errs) = (0usize, 0usize);
    for ((id, r), h) in refs.iter().zip(&hyps) {
        let (rn, hn) = (norm(r), norm(h));
        let e = edits(&rn, &hn);
        words += rn.len(); errs += e;
        println!("UTT {id} {e} {} | {h}", rn.len());
    }
    println!("SUMMARY utterances {} words {words} edits {errs} correct {} audio_s {audio_s:.2} wall_s {:.3}",
             refs.len(), words.saturating_sub(errs), t1);
    println!("WER {:.2}%  realtime {:.2}x", 100.0 * errs as f64 / words.max(1) as f64, audio_s / t1);
}
