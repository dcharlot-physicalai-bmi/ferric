//! **An independent float64 reading of ggml's block formats** — the oracle the CPU fabric's kernels
//! (`ferric_tensor::cpu_q`) are checked against. Shared by `tests/cpu_q_kernels.rs` (synthetic blocks)
//! and `examples/cpu_q_blocks.rs` (blocks read out of real GGUF files).
//!
//! Transcribed loop-for-loop from ggml's `dequantize_row_{q4_0,q4_1,q5_0,q5_1,q8_0,q4_K,q5_K,q6_K}`
//! (ggml/src/ggml-quants.c, the library that DEFINES these formats) and `get_scale_min_k4`, in f64,
//! with f16 decoded by the `half` crate. It shares no code with the kernels: not their scale decoder,
//! not their f16 conversion, not their loop structure — a reference that borrowed the kernel's
//! helpers would agree with the kernel's bugs.
//!
//! Also here: one PLAUSIBLE WRONG reading per format (`dequant_wrong`), each a defect that loads,
//! produces finite numbers and would pass a smoke test. The tests require the kernel to sit >= 20x
//! closer to the right reading than to the wrong one, or they have not shown they can see that class.
#![allow(dead_code)]

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q4_0: u32 = 2;
pub const Q4_1: u32 = 3;
pub const Q5_0: u32 = 6;
pub const Q5_1: u32 = 7;
pub const Q8_0: u32 = 8;
pub const Q4_K: u32 = 12;
pub const Q5_K: u32 = 13;
pub const Q6_K: u32 = 14;
pub const BF16: u32 = 30;

pub const ALL: [u32; 11] = [F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K];

pub fn name(t: u32) -> &'static str {
    match t { F32 => "F32", F16 => "F16", BF16 => "BF16", Q4_0 => "Q4_0", Q4_1 => "Q4_1", Q5_0 => "Q5_0",
              Q5_1 => "Q5_1", Q8_0 => "Q8_0", Q4_K => "Q4_K", Q5_K => "Q5_K", Q6_K => "Q6_K", _ => "?" }
}

/// (elements, bytes) per block.
pub fn block(t: u32) -> (usize, usize) {
    match t { F32 => (1, 4), F16 | BF16 => (1, 2), Q4_0 => (32, 18), Q4_1 => (32, 20), Q5_0 => (32, 22),
              Q5_1 => (32, 24), Q8_0 => (32, 34), Q4_K => (256, 144), Q5_K => (256, 176), Q6_K => (256, 210),
              _ => panic!("no block size for type {t}") }
}

fn h(b: &[u8], o: usize) -> f64 { half::f16::from_bits(u16::from_le_bytes([b[o], b[o + 1]])).to_f64() }

/// ggml `get_scale_min_k4`, verbatim.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 { (q[j] & 63, q[j + 4] & 63) }
    else { ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)) }
}

/// The right reading of one row of `n` elements.
pub fn dequant(t: u32, row: &[u8], n: usize) -> Vec<f64> { deq(t, row, n, false) }
/// One plausible WRONG reading (see the module doc for which defect, per format).
pub fn dequant_wrong(t: u32, row: &[u8], n: usize) -> Vec<f64> { deq(t, row, n, true) }

fn deq(t: u32, row: &[u8], n: usize, wrong: bool) -> Vec<f64> {
    let mut y = Vec::with_capacity(n);
    match t {
        // wrong: the two 4-lane halves of every 8-element group exchanged (a SIMD lane-order slip).
        F32 => {
            let v: Vec<f64> = row.chunks_exact(4).take(n).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect();
            for i in 0..n { y.push(if wrong && i < n / 8 * 8 { v[(i / 8) * 8 + (i % 8 + 4) % 8] } else { v[i] }); }
        }
        // wrong: the two 4-lane halves of every 8-element group exchanged — what swapping the kernel's
        // `fcvtl` / `fcvtl2` outputs would do. (A byte-order slip was tried first: it makes inf/NaN,
        // which a max-error criterion silently drops — that control measured 0.)
        F16 => {
            let v: Vec<f64> = row.chunks_exact(2).take(n).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64()).collect();
            for i in 0..n { y.push(if wrong && i < n / 8 * 8 { v[(i / 8) * 8 + (i % 8 + 4) % 8] } else { v[i] }); }
        }
        // wrong: bf16 read as f16.
        BF16 => for c in row.chunks_exact(2).take(n) {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            y.push(if wrong { half::f16::from_bits(bits).to_f64() } else { half::bf16::from_bits(bits).to_f64() })
        },
        _ => {
            let (qk, bb) = block(t);
            for x in row.chunks_exact(bb).take(n / qk) {
                match t {
                    // wrong: the two nibbles of each byte assigned to the opposite halves.
                    Q4_0 => {
                        let d = h(x, 0);
                        let mut o = [0f64; 32];
                        for j in 0..16 {
                            let (lo, hi) = ((x[2 + j] & 0x0F) as i32 - 8, (x[2 + j] >> 4) as i32 - 8);
                            let (a, b) = if wrong { (hi, lo) } else { (lo, hi) };
                            o[j] = a as f64 * d; o[j + 16] = b as f64 * d;
                        }
                        y.extend(o);
                    }
                    // wrong: the min dropped.
                    Q4_1 => {
                        let (d, m) = (h(x, 0), h(x, 2));
                        let m = if wrong { 0.0 } else { m };
                        let mut o = [0f64; 32];
                        for j in 0..16 { o[j] = (x[4 + j] & 0x0F) as f64 * d + m; o[j + 16] = (x[4 + j] >> 4) as f64 * d + m; }
                        y.extend(o);
                    }
                    // wrong (Q5_0): the fifth bit ignored. wrong (Q5_1): the second half's fifth bit
                    // taken from bit j rather than bit j+16.
                    Q5_0 | Q5_1 => {
                        let one = t == Q5_1;
                        let d = h(x, 0);
                        let m = if one { h(x, 2) } else { 0.0 };
                        let o0 = if one { 4 } else { 2 };
                        let qh = u32::from_le_bytes([x[o0], x[o0 + 1], x[o0 + 2], x[o0 + 3]]);
                        let mut o = [0f64; 32];
                        for j in 0..16 {
                            let mut xh0 = ((qh >> j) << 4) & 0x10;
                            let mut xh1 = (qh >> (j + 12)) & 0x10;
                            if wrong && !one { xh0 = 0; xh1 = 0; }
                            if wrong && one { xh1 = ((qh >> j) << 4) & 0x10; }
                            let a = ((x[o0 + 4 + j] & 0x0F) as u32 | xh0) as i32;
                            let b = ((x[o0 + 4 + j] >> 4) as u32 | xh1) as i32;
                            if one { o[j] = a as f64 * d + m; o[j + 16] = b as f64 * d + m; }
                            else { o[j] = (a - 16) as f64 * d; o[j + 16] = (b - 16) as f64 * d; }
                        }
                        y.extend(o);
                    }
                    // wrong: codes read unsigned.
                    Q8_0 => {
                        let d = h(x, 0);
                        for j in 0..32 { y.push(if wrong { x[2 + j] as f64 * d } else { x[2 + j] as i8 as f64 * d }); }
                    }
                    // wrong (Q4_K): scales/mins of sub-blocks 4..7 lose their top two bits.
                    // wrong (Q5_K): the high sub-block's fifth bit taken from the low sub-block's bit.
                    Q4_K | Q5_K => {
                        let five = t == Q5_K;
                        let (d, min) = (h(x, 0), h(x, 2));
                        let sc = &x[4..16];
                        let (qh, mut q) = if five { (&x[16..48], &x[48..176]) } else { (&x[..0], &x[16..144]) };
                        let (mut is, mut u1, mut u2) = (0usize, 1u8, 2u8);
                        for _j in (0..256).step_by(64) {
                            let (mut s1, mut m1) = scale_min_k4(is, sc);
                            let (mut s2, mut m2) = scale_min_k4(is + 1, sc);
                            if wrong && !five {
                                if is >= 4 { s1 &= 0xF; m1 &= 0xF; }
                                if is + 1 >= 4 { s2 &= 0xF; m2 &= 0xF; }
                            }
                            let (d1, mm1) = (d * s1 as f64, min * m1 as f64);
                            let (d2, mm2) = (d * s2 as f64, min * m2 as f64);
                            let u2e = if wrong && five { u1 } else { u2 };
                            for l in 0..32 {
                                let hb = if five && qh[l] & u1 != 0 { 16 } else { 0 };
                                y.push(d1 * ((q[l] & 0xF) as f64 + hb as f64) - mm1);
                            }
                            for l in 0..32 {
                                let hb = if five && qh[l] & u2e != 0 { 16 } else { 0 };
                                y.push(d2 * ((q[l] >> 4) as f64 + hb as f64) - mm2);
                            }
                            q = &q[32..];
                            is += 2;
                            u1 = u1.wrapping_shl(2);
                            u2 = u2.wrapping_shl(2);
                        }
                    }
                    // wrong: the second and third quarter's high-bit pairs swapped (qh >> 2 vs >> 4).
                    Q6_K => {
                        let d = h(x, 208);
                        let (mut ql, mut qh, mut sc) = (&x[0..128], &x[128..192], &x[192..208]);
                        let mut o = [0f64; 256];
                        for n0 in (0..256).step_by(128) {
                            for l in 0..32 {
                                let is = l / 16;
                                let (s2, s4) = if wrong { (4, 2) } else { (2, 4) };
                                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> s2) & 3) << 4)) as i32 - 32;
                                let q3 = ((ql[l] >> 4) | (((qh[l] >> s4) & 3) << 4)) as i32 - 32;
                                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                                o[n0 + l] = d * (sc[is] as i8) as f64 * q1 as f64;
                                o[n0 + l + 32] = d * (sc[is + 2] as i8) as f64 * q2 as f64;
                                o[n0 + l + 64] = d * (sc[is + 4] as i8) as f64 * q3 as f64;
                                o[n0 + l + 96] = d * (sc[is + 6] as i8) as f64 * q4 as f64;
                            }
                            ql = &ql[64..]; qh = &qh[32..]; sc = &sc[8..];
                        }
                        y.extend(o);
                    }
                    _ => panic!("no reference for type {t}"),
                }
            }
        }
    }
    assert_eq!(y.len(), n, "{}: dequantized {} of {n} elements", name(t), y.len());
    y
}

/// The activation rounding the int8 kernels are SPECIFIED to apply (cpu_q's module doc): per block of
/// `block` values, `d = amax/127` (f32), codes `round_ties_even(x / d)` computed as `x * (1/d)` in f32.
/// Written here from that specification, not by calling the library; the tests assert the library's
/// codes and scales equal these exactly. Returns (codes, scales, reconstruction in f64).
pub fn quantize_act(x: &[f32], block: usize) -> (Vec<i8>, Vec<f32>, Vec<f64>) {
    let (mut q, mut ds, mut out) = (Vec::with_capacity(x.len()), Vec::new(), Vec::with_capacity(x.len()));
    for b in x.chunks(block) {
        let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        ds.push(d);
        for &v in b {
            let c = (v * id).round_ties_even() as i8;
            q.push(c);
            out.push(c as f64 * d as f64);
        }
    }
    (q, ds, out)
}

/// Which activation block a weight format meets: None = f32 activations.
pub fn act_block(t: u32) -> Option<usize> {
    match t { F32 | F16 | BF16 => None, Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 => Some(32), _ => Some(256) }
}

/// Σ w·x and Σ |w·x| in f64 — the dot and the scale its error is judged against (cancellation can make
/// the dot itself tiny, so a relative error on the dot alone would be meaningless).
pub fn dot(w: &[f64], x: &[f64]) -> (f64, f64) {
    w.iter().zip(x).fold((0.0, 0.0), |(s, a), (&p, &q)| (s + p * q, a + (p * q).abs()))
}
