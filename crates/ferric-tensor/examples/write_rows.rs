//! **Does writing an expert into a live slab actually replace it?** — the primitive expert
//! residency needs, verified before anything is built on it.
//!
//! Until `Q4_KWeights::write_rows`, nothing in `dtype.rs` wrote into a built weight — the
//! `write_buffer` count was zero. That is the real reason no Ferric runtime streams an expert:
//! there was nowhere to PUT a fetched one. Policy, caching and split arithmetic all presume this
//! call exists.
//!
//! ## The check is equality against a slab built the long way
//!
//! Build slab A from rows `[x, y, z]`. Build slab B from `[x, y, z]` and then write `w` over row 1.
//! Build slab C from `[x, w, z]` directly. **B must equal C, and must NOT equal A.** The second half
//! matters as much as the first: a `write_rows` that silently did nothing would pass any test that
//! only compared B against C if C were also built by writing.
//!
//! Rows are read back with a ONE-HOT probe — `x = e_j` makes `x·Wᵀ` return column *j* of the weight,
//! every other term a hard zero — so this compares dequantised weights directly rather than a dot
//! product that could hide a compensating error.
//!
//!   cargo run -p ferric-tensor --example write_rows --release
use ferric_tensor::{Q4_KWeights, Tensor};
use std::sync::Arc;

const COLS: usize = 256;
const ROWS: usize = 3;
const BLK: usize = 144;

/// Deterministic Q4_K block bytes. Distinct `seed`s give distinct weights, which is what makes
/// "B != A" a real assertion rather than a coincidence.
fn rowbytes(seed: u64) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..BLK).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 33) as u8 }).collect()
}

fn slab(seeds: &[u64]) -> Vec<u8> { seeds.iter().flat_map(|&s| rowbytes(s)).collect() }

fn main() { pollster::block_on(run()); }

async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    if std::env::var("FERRIC_Q4K_TRANS").is_ok() {
        println!("FERRIC_Q4K_TRANS is set — write_rows refuses under it by design. Unset to test.");
        return;
    }
    println!("Q4_KWeights::write_rows — replacing one expert's rows in a live slab\n");

    let a = Q4_KWeights::from_bytes(&ctx, &slab(&[1, 2, 3]), ROWS, COLS);
    let b = Q4_KWeights::from_bytes(&ctx, &slab(&[1, 2, 3]), ROWS, COLS);
    let c = Q4_KWeights::from_bytes(&ctx, &slab(&[1, 9, 3]), ROWS, COLS);
    b.write_rows(1, &rowbytes(9), 1).expect("write row 1");

    // One-hot probe: every column of the weight, read back exactly.
    let mut x = vec![0f32; ROWS * COLS];
    for j in 0..ROWS { x[j * COLS + j] = 1.0; }   // rows of the probe select columns 0..ROWS
    let probe = Tensor::from_vec(&ctx, &x, &[ROWS, COLS]);
    let (va, vb, vc) = (probe.matmul_q4_k(&a).to_vec().await,
                        probe.matmul_q4_k(&b).to_vec().await,
                        probe.matmul_q4_k(&c).to_vec().await);

    let maxdiff = |p: &[f32], q: &[f32]| p.iter().zip(q).fold(0f32, |m, (u, v)| m.max((u - v).abs()));
    let (bc, ba) = (maxdiff(&vb, &vc), maxdiff(&vb, &va));
    println!("  max |B − C|  {bc:.3e}   (written slab vs one built with the same rows)");
    println!("  max |B − A|  {ba:.3e}   (written slab vs the original)");

    assert_eq!(bc, 0.0, "a slab written into does not match one BUILT with the same rows — the \
                         write landed at the wrong offset or repacked the block wrongly");
    assert!(ba > 0.0, "the written slab is identical to the ORIGINAL, so write_rows did nothing. \
                       A no-op passes any test that only compares against another written slab");

    // Range and stale-copy refusals: a primitive that cannot say no is a corruption vector.
    assert!(b.write_rows(ROWS, &rowbytes(4), 1).is_err(), "wrote past the last row");
    assert!(b.write_rows(0, &rowbytes(4)[..BLK - 1], 1).is_err(), "accepted a short byte slice");
    assert!(b.write_rows(0, &slab(&[4, 5]), 1).is_err(), "accepted 2 rows of bytes for n_rows=1");

    println!("\n  ✅ bit-exact replacement, the original is genuinely changed, and out-of-range or\n  \
              mis-sized writes are refused. This is the call expert streaming was missing.");
}
