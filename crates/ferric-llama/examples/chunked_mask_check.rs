//! **Is the `chunked_limited` attention mask the rule the reference defines?**
//!
//! No streaming checkpoint here exercises this path end-to-end, so the mask rule is checked directly
//! rather than assumed. The oracle is a line-by-line transcription of
//! `chunked_limited_mask_function` from `transformers/models/nemotron_asr_streaming` — written from
//! the Python, not from the Rust, so the two cannot agree by construction.
//!
//! ⚠ The failure this exists to catch: a SLIDING BAND `[q-left, q+right]`. It has the same shape,
//! the same element count, the same "left/right context" vocabulary — and masks different elements.
//! Nothing downstream could tell the difference; the transcript would just be quietly wrong.
use ferric_llama::parakeet::chunked_limited_mask;

/// Transcribed from the reference:
///   chunk_size          = right_ctx + 1
///   left_context_chunks = left_ctx // chunk_size
///   allowed = (q//chunk - kv//chunk) >= 0 and <= left_context_chunks
fn reference_allows(q: usize, kv: usize, left: usize, right: usize) -> bool {
    let chunk_size = right + 1;
    let left_context_chunks = left / chunk_size;
    let q_chunk = q / chunk_size;
    let kv_chunk = kv / chunk_size;
    let chunk_diff = q_chunk as i64 - kv_chunk as i64;
    chunk_diff >= 0 && chunk_diff <= left_context_chunks as i64
}

/// The wrong-but-plausible alternative, to prove the check can actually tell them apart.
fn sliding_band_allows(q: usize, kv: usize, left: usize, right: usize) -> bool {
    kv as i64 >= q as i64 - left as i64 && kv as i64 <= q as i64 + right as i64
}

fn main() {
    let mut bad = 0;
    // (left, right) pairs: nemotron-asr's real setting first, then edge cases.
    for &(left, right) in &[(56usize, 13usize), (0, 0), (13, 13), (70, 6), (1, 0)] {
        for &t in &[1usize, 5, 14, 15, 40, 64] {
            let m = chunked_limited_mask(t, left, right);
            let mut mismatch = 0;
            for q in 0..t {
                for kv in 0..t {
                    let open = m[q * t + kv] == 0.0;
                    if open != reference_allows(q, kv, left, right) { mismatch += 1; }
                }
            }
            if mismatch != 0 { bad += 1; println!("  MISMATCH left={left} right={right} t={t}: {mismatch}/{}", t * t); }
        }
    }
    // A row is never fully masked — softmax over an all -1e30 row would be uniform nonsense.
    for &t in &[1usize, 14, 40] {
        let m = chunked_limited_mask(t, 56, 13);
        for q in 0..t {
            assert!((0..t).any(|kv| m[q * t + kv] == 0.0), "row {q} of {t} is fully masked");
        }
    }
    // Prove the oracle discriminates: chunked and sliding must DISAGREE somewhere at the real setting.
    let (t, left, right) = (40usize, 56usize, 13usize);
    let diff = (0..t).flat_map(|q| (0..t).map(move |kv| (q, kv)))
        .filter(|&(q, kv)| reference_allows(q, kv, left, right) != sliding_band_allows(q, kv, left, right))
        .count();
    println!("chunked vs sliding-band differ in {diff}/{} cells at left=56 right=13", t * t);
    assert!(diff > 0, "the two rules agree here — this check could not detect a band-vs-chunk error");

    println!("{}", if bad == 0 { "chunked_limited mask matches the reference rule at every size" }
                   else { "MASK IS WRONG" });
    assert_eq!(bad, 0);
}
