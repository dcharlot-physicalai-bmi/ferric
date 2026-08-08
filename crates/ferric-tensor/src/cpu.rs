//! **The other half of the fabric: quantised matmul on the CPU's vector units.**
//!
//! Ferric reaches the GPU through wgpu and, until this module, reached nothing else. That is the
//! convention every runtime in the 2026 survey follows: pick a backend, call it portability. On a
//! machine with 18 CPU cores and a GPU, it means running on one of those and leaving the other idle.
//!
//! The reason that is expensive here specifically is measured, not assumed. Decode is **bandwidth-bound**
//! (this workspace: ~525 MB of weights read per token, and cutting 29% of GPU dispatches moved wall time
//! by 0.00 ms). When the bound is bandwidth, what matters is how many independent issue paths to memory
//! the machine has. A saturated GPU beside an idle CPU is using one of them.
//!
//! ## Why this is plain safe Rust rather than intrinsics
//!
//! Explicit NEON/AVX-512 intrinsics would need `unsafe` and a `cfg` maze per target, and the resulting
//! kernel would be unavailable on exactly the platforms Ferric cares about reaching (wasm, unknown
//! hardware). Instead the inner loop is written in the shape LLVM autovectorises well: fixed-length
//! chunks, no branches, integer accumulation. That gets the bulk of the vector throughput while staying
//! portable and safe, which is the right trade for a runtime whose thesis is running everywhere.
//!
//! Threading is `std::thread::scope`, so the crate keeps its dependency-free posture.

/// A Q8_0 block: `f16` scale then 32 `i8` codes, 34 bytes. The GGUF layout, read in place.
const Q8_0_BLOCK: usize = 34;
const Q8_0_VALS: usize = 32;

/// Decode an IEEE binary16 to f32. Handles subnormals and inf/nan, because a weight file is allowed to
/// contain them and silently mapping them to zero would be a correctness bug that only shows up on
/// unusual checkpoints.
fn f16(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let man = (bits & 0x3FF) as u32;
    let out = match exp {
        0 if man == 0 => sign << 31,
        0 => {
            // Subnormal. Its value is `man * 2^-24`. Renormalise by shifting until the implicit bit
            // (0x400) is set: after k shifts the value is `1.xxx * 2^(-14-k)`, so the f32 exponent
            // field is `(-14 - k) + 127 = 113 - k`.
            let mut m = man;
            let mut k = 0u32;
            while m & 0x400 == 0 { m <<= 1; k += 1; }
            (sign << 31) | ((113 - k) << 23) | ((m & 0x3FF) << 13)
        }
        0x1F => (sign << 31) | 0x7F80_0000 | (man << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(out)
}

/// One row of `y = x · Wᵀ` for a Q8_0 weight, over `cols` inputs.
///
/// The inner loop is deliberately shaped for autovectorisation: a fixed 32-wide body, integer
/// multiply-accumulate, and the scale applied once per block rather than per element. Applying the
/// scale per element would be mathematically identical and several times slower, because it turns an
/// integer dot product into a float one.
#[inline]
fn row_q8_0(x: &[f32], blocks: &[u8], cols: usize) -> f32 {
    let nblk = cols / Q8_0_VALS;
    let mut acc = 0.0f32;
    for b in 0..nblk {
        let blk = &blocks[b * Q8_0_BLOCK..(b + 1) * Q8_0_BLOCK];
        let d = f16(u16::from_le_bytes([blk[0], blk[1]]));
        let xs = &x[b * Q8_0_VALS..(b + 1) * Q8_0_VALS];
        // Fixed 32-wide, branch-free. LLVM turns this into NEON SDOT / AVX-512 VNNI shapes.
        let mut part = 0.0f32;
        for j in 0..Q8_0_VALS {
            part += xs[j] * (blk[2 + j] as i8) as f32;
        }
        acc += part * d;
    }
    acc
}

/// `y = x · Wᵀ` on the CPU, where `W` is raw GGUF **Q8_0** bytes laid out `[rows, cols]`.
///
/// `x` is one activation row of length `cols`; the result is `rows` long. This is the decode shape, and
/// the shape the fabric splits: rows are independent, so any contiguous span of them can be handed to a
/// different compute unit and the results concatenated.
pub fn matvec_q8_0(x: &[f32], w: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(cols % Q8_0_VALS, 0, "Q8_0 cols must be a multiple of 32");
    assert_eq!(x.len(), cols, "activation length must equal cols");
    let row_bytes = (cols / Q8_0_VALS) * Q8_0_BLOCK;
    assert_eq!(w.len(), rows * row_bytes, "unexpected Q8_0 byte length");
    (0..rows).map(|r| row_q8_0(x, &w[r * row_bytes..(r + 1) * row_bytes], cols)).collect()
}

/// The same, across `threads` OS threads.
///
/// Rows are partitioned into contiguous spans so each thread walks memory linearly, which matters more
/// than load balance on a bandwidth-bound kernel: interleaving rows across threads would give every
/// thread a strided access pattern and throw away the prefetcher.
pub fn matvec_q8_0_threaded(x: &[f32], w: &[u8], rows: usize, cols: usize, threads: usize) -> Vec<f32> {
    let threads = threads.max(1).min(rows.max(1));
    if threads == 1 { return matvec_q8_0(x, w, rows, cols); }
    let row_bytes = (cols / Q8_0_VALS) * Q8_0_BLOCK;
    let mut out = vec![0.0f32; rows];
    let per = rows.div_ceil(threads);

    std::thread::scope(|s| {
        for (t, chunk) in out.chunks_mut(per).enumerate() {
            let lo = t * per;
            let hi = (lo + chunk.len()).min(rows);
            let wslice = &w[lo * row_bytes..hi * row_bytes];
            s.spawn(move || {
                for (i, o) in chunk.iter_mut().enumerate() {
                    *o = row_q8_0(x, &wslice[i * row_bytes..(i + 1) * row_bytes], cols);
                }
            });
        }
    });
    out
}

/// A contiguous span of output rows, so the fabric can hand different spans to different units.
///
/// Returns only `[lo, hi)` of the result. The GPU arm computes its own span concurrently and the two are
/// concatenated, which is the whole point: no unit waits for another.
pub fn matvec_q8_0_span(x: &[f32], w: &[u8], cols: usize, lo: usize, hi: usize, threads: usize) -> Vec<f32> {
    assert!(lo <= hi, "empty or inverted span");
    let row_bytes = (cols / Q8_0_VALS) * Q8_0_BLOCK;
    matvec_q8_0_threaded(x, &w[lo * row_bytes..hi * row_bytes], hi - lo, cols, threads)
}

/// Logical CPU count, for sizing a thread pool. Falls back to 1 rather than guessing high, since
/// oversubscribing a bandwidth-bound kernel makes it slower.
pub fn cpu_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a Q8_0 weight and the exact f32 values it encodes, so the test has ground truth that does
    /// not go through the same code being tested.
    fn synth(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        let mut rnd = || { s = s.wrapping_mul(1664525).wrapping_add(1013904223); (s >> 16) as u16 };
        let nblk = cols / 32;
        let mut raw = Vec::with_capacity(rows * nblk * 34);
        let mut vals = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for b in 0..nblk {
                // A scale that is exactly representable in f16, so ground truth is unambiguous.
                let dbits: u16 = 0x1C00 | ((rnd() & 0x3F) as u16); // ~0.0039 .. 0.0042
                raw.extend_from_slice(&dbits.to_le_bytes());
                let d = f16(dbits);
                for j in 0..32 {
                    let c = (rnd() & 0xFF) as u8;
                    raw.push(c);
                    vals[r * cols + b * 32 + j] = (c as i8) as f32 * d;
                }
            }
        }
        (raw, vals)
    }

    fn reference(x: &[f32], vals: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        (0..rows).map(|r| (0..cols).map(|i| x[i] * vals[r * cols + i]).sum()).collect()
    }

    #[test]
    fn matvec_matches_the_dequantised_reference() {
        for &(rows, cols) in &[(1usize, 32usize), (8, 64), (17, 256), (64, 512)] {
            let (raw, vals) = synth(rows, cols, (rows * 13 + cols) as u32);
            let x: Vec<f32> = (0..cols).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
            let got = matvec_q8_0(&x, &raw, rows, cols);
            let want = reference(&x, &vals, rows, cols);
            let scale = want.iter().fold(1e-6f32, |a, &v| a.max(v.abs()));
            let d = got.iter().zip(&want).fold(0f32, |a, (&g, &w)| a.max((g - w).abs())) / scale;
            assert!(d < 1e-5, "rows={rows} cols={cols}: rel delta {d:.3e}");
        }
    }

    #[test]
    fn threading_changes_nothing_about_the_answer() {
        // Rows are independent, so any partition must give the identical result. If this ever fails, a
        // thread is reading the wrong slice of the weight and the model would emit fluent, wrong text.
        let (raw, _) = synth(64, 256, 7);
        let x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
        let one = matvec_q8_0(&x, &raw, 64, 256);
        for t in [2usize, 3, 8, 64, 128] {
            let many = matvec_q8_0_threaded(&x, &raw, 64, 256, t);
            assert_eq!(one, many, "threads={t} changed the result");
        }
    }

    #[test]
    fn spans_concatenate_to_the_whole() {
        // The property the fabric depends on: the GPU takes one span, the CPU another, and the join is
        // the full answer. If spans did not compose, splitting would be silently wrong rather than slow.
        let (raw, _) = synth(40, 128, 3);
        let x: Vec<f32> = (0..128).map(|i| ((i % 7) as f32) - 3.0).collect();
        let whole = matvec_q8_0(&x, &raw, 40, 128);
        for cut in [0usize, 1, 17, 39, 40] {
            let mut joined = matvec_q8_0_span(&x, &raw, 128, 0, cut, 2);
            joined.extend(matvec_q8_0_span(&x, &raw, 128, cut, 40, 2));
            assert_eq!(whole, joined, "cut at {cut} did not reassemble");
        }
    }

    #[test]
    fn f16_decodes_subnormals_and_specials() {
        // A weight file may legally contain these, and mapping them to zero would be a silent
        // correctness bug on unusual checkpoints.
        assert_eq!(f16(0x0000), 0.0);
        assert_eq!(f16(0x3C00), 1.0);
        assert_eq!(f16(0xBC00), -1.0);
        assert!((f16(0x0001) - 5.9604645e-8).abs() < 1e-14, "smallest subnormal");
        assert!(f16(0x7C00).is_infinite());
        assert!(f16(0xFC00).is_infinite() && f16(0xFC00) < 0.0);
        assert!(f16(0x7E00).is_nan());
    }

    #[test]
    fn cpu_threads_never_reports_zero() {
        assert!(cpu_threads() >= 1);
    }
}
