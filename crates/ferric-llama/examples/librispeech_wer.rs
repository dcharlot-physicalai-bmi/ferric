//! **LibriSpeech test-clean WER** — the globally understood number, against the official transcripts.
//!
//! Usage: librispeech_wer <model.gguf> <LibriSpeech/test-clean dir> [n_utterances]
//!
//! ⚠ CORPUS WER, not the mean of per-utterance WERs. The standard metric is total edits over total
//! reference words; averaging per-utterance rates over-weights short utterances and reports a
//! different (usually worse) number that is not comparable to anyone's published figure.
//!
//! ⚠ The reference is the SHIPPED `.trans.txt`, never the model's own output. Scoring against
//! anything derived from the system under test measures self-consistency, not accuracy.
//!
//! Normalization is the LibriSpeech convention the references are already in: uppercase, letters and
//! apostrophes only. It is applied identically to both sides, so it cannot flatter the hypothesis.
use ferric_gguf::GgufFile;
use std::collections::HashMap;
use std::sync::Arc;

fn norm(s: &str) -> Vec<String> {
    s.to_uppercase().split_whitespace()
        .map(|w| w.chars().filter(|c| c.is_ascii_alphabetic() || *c == '\'').collect::<String>())
        .filter(|w| !w.is_empty()).collect()
}

/// Levenshtein over words: the WER numerator (substitutions + insertions + deletions).
fn edits(r: &[String], h: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    let mut cur = vec![0usize; h.len() + 1];
    for i in 1..=r.len() {
        cur[0] = i;
        for j in 1..=h.len() {
            let c = usize::from(r[i - 1] != h[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + c);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[h.len()]
}

/// 16-bit PCM WAV reader (ffmpeg writes these from the corpus .flac).
fn read_wav(path: &str) -> (Vec<f32>, usize) {
    let b = std::fs::read(path).expect("read wav");
    let (mut i, mut rate, mut pcm) = (12usize, 0usize, Vec::new());
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let sz = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
        let d = &b[i + 8..(i + 8 + sz).min(b.len())];
        if id == b"fmt " { rate = u32::from_le_bytes([d[4], d[5], d[6], d[7]]) as usize; }
        if id == b"data" { pcm = d.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect(); }
        i += 8 + sz + (sz & 1);
    }
    (pcm, rate)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let (gguf, root) = (&a[1], &a[2]);
    let limit: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);

    // Reference transcripts, keyed by utterance id, straight from the corpus.
    let mut refs: HashMap<String, String> = HashMap::new();
    let mut flacs: Vec<(String, String)> = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(root)];
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
                Some("flac") => {
                    let id = p.file_stem().unwrap().to_string_lossy().to_string();
                    flacs.push((id, p.to_string_lossy().to_string()));
                }
                _ => {}
            }
        }
    }
    flacs.sort();                       // deterministic subset: the first N by id, not an arbitrary walk order
    flacs.truncate(limit);
    println!("{} utterances, {} references", flacs.len(), refs.len());

    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let g = GgufFile::open(gguf).expect("open gguf");
    let m = ferric_llama::parakeet::Parakeet::load(&ctx, &g).expect("load");

    let (mut tot_err, mut tot_words, mut audio_s, mut done) = (0usize, 0usize, 0f64, 0usize);
    let t0 = std::time::Instant::now();
    let tmp = std::env::temp_dir().join("ls_wer.wav");
    for (id, path) in &flacs {
        let Some(reference) = refs.get(id) else { continue };
        let ok = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i", path, "-ar", "16000", "-ac", "1",
                   "-c:a", "pcm_s16le", tmp.to_str().unwrap()])
            .status().map(|s| s.success()).unwrap_or(false);
        if !ok { eprintln!("ffmpeg failed on {id}"); continue; }
        let (pcm, rate) = read_wav(tmp.to_str().unwrap());
        assert_eq!(rate, m.cfg.sample_rate, "resample produced {rate} Hz");
        audio_s += pcm.len() as f64 / rate as f64;
        let hyp = m.transcribe(&pcm).expect("transcribe");
        let (r, h) = (norm(reference), norm(&hyp));
        tot_err += edits(&r, &h);
        tot_words += r.len();
        done += 1;
        if done % 25 == 0 {
            println!("  {done}/{} — running WER {:.2}%", flacs.len(), 100.0 * tot_err as f64 / tot_words as f64);
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    println!("\nutterances {done} | words {tot_words} | edits {tot_err}");
    println!("WER {:.2}%", 100.0 * tot_err as f64 / tot_words.max(1) as f64);
    println!("audio {audio_s:.0} s in {secs:.0} s wall — {:.1}x realtime (includes ffmpeg decode)", audio_s / secs);
}
