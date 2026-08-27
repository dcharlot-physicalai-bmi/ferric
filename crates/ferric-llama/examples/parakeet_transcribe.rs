//! **Speech in, text out** — the only test that settles an ASR port.
//!
//! Shapes and finiteness prove nothing here: a wrong relative-position shift, a swapped LSTM gate
//! order, an ascending positional encoding or the wrong flatten order in the subsampling stack all
//! produce a running model and a wrong transcript. So the check is a real utterance with known
//! ground truth, and the measure is word error rate.
//!
//!   cargo run -p ferric-llama --example parakeet_transcribe --release -- <model.gguf> <audio.wav> ["reference text"]
use ferric_gguf::GgufFile;
use std::sync::Arc;

/// Minimal 16-bit PCM WAV reader — enough for `ffmpeg -ar 16000 -ac 1 -c:a pcm_s16le`.
fn read_wav(path: &str) -> (Vec<f32>, usize) {
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
            pcm = d.chunks_exact(2)
                   .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                   .collect();
            break;
        }
        i += 8 + sz + (sz & 1);
    }
    (pcm, rate)
}

/// Levenshtein over words — the standard WER numerator.
fn wer(reference: &str, hyp: &str) -> (usize, usize) {
    let r: Vec<&str> = reference.split_whitespace().collect();
    let h: Vec<&str> = hyp.split_whitespace().collect();
    let mut d = vec![vec![0usize; h.len() + 1]; r.len() + 1];
    for i in 0..=r.len() { d[i][0] = i; }
    for j in 0..=h.len() { d[0][j] = j; }
    for i in 1..=r.len() { for j in 1..=h.len() {
        let c = if r[i - 1].eq_ignore_ascii_case(h[j - 1]) { 0 } else { 1 };
        d[i][j] = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + c);
    }}
    (d[r.len()][h.len()], r.len())
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let model = a.get(1).expect("usage: parakeet_transcribe <model.gguf> <audio.wav> [reference]");
    let audio = a.get(2).expect("need an audio.wav");
    let g = GgufFile::open(model).expect("open gguf");
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    let m = ferric_llama::parakeet::Parakeet::load(&ctx, &g).expect("load");
    println!("{}", m.describe());

    let (pcm, rate) = read_wav(audio);
    assert_eq!(rate, m.cfg.sample_rate, "wav is {rate} Hz, model wants {}", m.cfg.sample_rate);
    println!("audio: {:.2} s\n", pcm.len() as f32 / rate as f32);

    let t0 = std::time::Instant::now();
    let text = m.transcribe(&pcm).expect("transcribe");
    println!("  transcript: {text:?}");
    println!("  in {:.1}s", t0.elapsed().as_secs_f32());

    if let Some(reference) = a.get(3) {
        let (e, n) = wer(reference, &text);
        println!("\n  reference:  {reference:?}");
        println!("  WER {e}/{n} = {:.1}%", 100.0 * e as f32 / n.max(1) as f32);
        // An empty or garbage transcript is the expected shape of every convention bug listed in
        // the header, so the bar is a real one rather than "it produced something".
        assert!(!text.trim().is_empty(), "empty transcript — the joint never left blank");
        assert!(e * 2 < n, "WER {:.0}% — worse than half the words. Suspect, in order: the \
                            pre-encode flatten order, the rel-pos shift direction, the LSTM gate \
                            order, or the positional encoding direction",
                100.0 * e as f32 / n.max(1) as f32);
    }
}
