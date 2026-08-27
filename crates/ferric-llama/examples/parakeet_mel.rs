//! **Does the mel frontend put energy where the energy is?**
//!
//! There is no reference implementation in this tree to diff against, so the test is a property the
//! transform must satisfy by construction: a PURE SINE at frequency f must peak in the mel bin whose
//! triangular filter covers f. That pins the FFT, the filterbank edges and the hz↔mel conversion
//! together — a transposed filterbank or an off-by-one in the bin→Hz mapping moves the peak.
//!
//! It is checked at several frequencies because one tone can land right on a bin boundary and agree
//! with a wrong mapping by luck.
use ferric_llama::parakeet::{frontend, Cfg};
use ferric_gguf::GgufFile;

fn main() {
    let path = std::env::args().nth(1).expect("usage: parakeet_mel <model.gguf>");
    let g = GgufFile::open(&path).expect("open gguf");
    let cfg = Cfg::from_gguf(&g).expect("cfg");
    println!("frontend: n_fft {} hop {} win {} mels {} @ {} Hz",
             cfg.n_fft, cfg.hop_length, cfg.win_length, cfg.num_mels, cfg.sample_rate);

    // Which mel filter has the largest response at frequency f — the bin the peak SHOULD land in.
    let fb = frontend::filterbank(&cfg);
    let bin_hz = cfg.sample_rate as f32 / cfg.n_fft as f32;
    let expect_bin = |f: f32| -> usize {
        let b = (f / bin_hz).round() as usize;
        (0..cfg.num_mels).max_by(|&i, &j| fb[i][b].total_cmp(&fb[j][b])).unwrap()
    };

    let sr = cfg.sample_rate as f32;
    let mut worst = 0usize;
    for &hz in &[220.0f32, 440.0, 880.0, 1760.0, 3520.0] {
        let pcm: Vec<f32> = (0..sr as usize)
            .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / sr).sin())
            .collect();
        let (mel, frames) = frontend::log_mel(&pcm, &cfg);
        assert!(frames > 50, "{frames} frames from 1 s of audio");
        // Average over time, then take the argmax bin — a steady tone should be stable across frames.
        let mut avg = vec![0f32; cfg.num_mels];
        for f in 0..frames { for m in 0..cfg.num_mels { avg[m] += mel[f * cfg.num_mels + m]; } }
        let got = (0..cfg.num_mels).max_by(|&i, &j| avg[i].total_cmp(&avg[j])).unwrap();
        let want = expect_bin(hz);
        let off = got.abs_diff(want);
        worst = worst.max(off);
        println!("  {hz:>7.0} Hz → mel bin {got:>3}   expected {want:>3}   off by {off}");
    }
    // Adjacent is fine: a tone between two filter centres genuinely splits its energy. Two bins away
    // is not, and would mean the frequency axis is wrong.
    assert!(worst <= 1, "a pure tone landed {worst} mel bins from its filter — the frequency axis \
                         (fft bin → Hz, or the mel edges) is wrong");
    println!("\n  every tone peaks in its own mel filter (max offset {worst})");
}
