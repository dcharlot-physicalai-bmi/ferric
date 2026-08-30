//! **How long does the encoder actually take?** Paired, in-process, min-of-N.
//!
//! Wall-clock on a single process launch could not measure this: repeated runs of the SAME binary
//! ranged 2.0–5.6 s on a contended machine, so a 10% effect is invisible and a "regression" is
//! indistinguishable from noise. This times `encode()` alone, inside one process, and reports the
//! MIN — the run least perturbed by other load, which is the only statistic that means anything
//! when the noise is one-sided.
use ferric_gguf::GgufFile;
use ferric_llama::parakeet::Parakeet;
use std::sync::Arc;
use std::time::Instant;

/// Minimal 16-bit PCM WAV reader (same contract as parakeet_transcribe).
fn read_wav(path: &str) -> (Vec<f32>, usize) {
    let b = std::fs::read(path).expect("read wav");
    let (mut i, mut rate, mut pcm) = (12usize, 0usize, Vec::new());
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let sz = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
        let d = &b[i + 8..(i + 8 + sz).min(b.len())];
        if id == b"fmt " { rate = u32::from_le_bytes([d[4], d[5], d[6], d[7]]) as usize; }
        if id == b"data" {
            pcm = d.chunks_exact(2)
                   .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect();
        }
        i += 8 + sz + (sz & 1);
    }
    (pcm, rate)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let (gguf, wav) = (&a[1], &a[2]);
    let iters: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let g = GgufFile::open(gguf).expect("open gguf");
    let m = Parakeet::load(&ctx, &g).expect("load");
    let (pcm, rate) = read_wav(wav);
    let secs = pcm.len() as f32 / rate as f32;

    // Dispatch and submit counts per encode: the figure that says whether a browser's per-call
    // overhead could plausibly explain a slowdown, or whether the work itself is the cost.
    ferric_tensor::reset_op_counters();
    ferric_tensor::reset_host_ns();
    // Epoch bounds for the measured loop, so an energy harness can integrate power over THIS window
    // and not over the process lifetime — which would average the encodes against model load.
    let epoch = || std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .unwrap().as_secs_f64();
    let mut ts = vec![];
    println!("WINDOW_START {:.3}", epoch());
    for i in 0..iters {
        let t0 = Instant::now();
        let enc = m.encode(&pcm).expect("encode");
        // Force completion: without a readback the batch may still be queued when the timer stops,
        // and we would be timing enqueue rather than execution.
        let _ = enc.to_vec().await;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("  iter {i}: {ms:.0} ms");
        ts.push(ms);
    }
    println!("WINDOW_END {:.3}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64());
    // Where does HOST time go per dispatch? Native is still 1.4x slower than the browser on the
    // SAME 6317 dispatches after the compute-pass fix, and these four slots are the only host work
    // in `run()`: pipeline lookup, bind-group creation, pass recording, and the per-dispatch info
    // buffer. Chrome runs the same Ferric code, so a slot that dominates here is a wgpu-native cost
    // Dawn does not pay.
    let (p_ns, bg_ns, rec_ns, buf_ns) = ferric_tensor::host_ns();
    let tot = (p_ns + bg_ns + rec_ns + buf_ns).max(1) as f64;
    let ms = |n: u64| n as f64 / 1e6 / iters as f64;
    println!("\nhost time per encode (ms): pipeline {:.1}  bindgroup {:.1}  record {:.1}  infobuf {:.1}",
             ms(p_ns), ms(bg_ns), ms(rec_ns), ms(buf_ns));
    println!("  share: pipeline {:.0}%  bindgroup {:.0}%  record {:.0}%  infobuf {:.0}%",
             100.0 * p_ns as f64 / tot, 100.0 * bg_ns as f64 / tot,
             100.0 * rec_ns as f64 / tot, 100.0 * buf_ns as f64 / tot);
    let (d, sub) = ferric_tensor::op_counters();
    println!("\ndispatches {} ({} per encode), submits {} ({} per encode)",
             d, d / iters as u64, sub, sub / iters as u64);
    let (min, max) = (ts.iter().cloned().fold(f64::MAX, f64::min), ts.iter().cloned().fold(0.0, f64::max));
    println!("\naudio {secs:.2} s | encode min {min:.0} ms  max {max:.0} ms  spread {:.0}%",
             (max - min) / min * 100.0);
    println!("realtime factor (min): {:.2}x", secs as f64 * 1000.0 / min);
}
