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
use ferric_tensor::{Q4_KWeights, Q6_KWeights, Tensor};
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

    // ---- Q6_K: the down projection, which a Q4_K_M MoE stores separately ------------------
    //
    // ⚠ Covering only Q4_K would leave gate|up swappable and `down` pinned to whatever it was built
    // with — the two halves of ONE expert disagreeing. That does not error, it produces plausible
    // wrong output, so the second half needs the same probe as the first.
    const B6: usize = 210;
    let r6 = |seed: u64| -> Vec<u8> {
        let mut st = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..B6).map(|_| { st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (st >> 33) as u8 }).collect()
    };
    let s6 = |seeds: &[u64]| -> Vec<u8> { seeds.iter().flat_map(|&x| r6(x)).collect() };
    let a6 = Q6_KWeights::from_bytes(&ctx, &s6(&[1, 2, 3]), ROWS, COLS);
    let b6 = Q6_KWeights::from_bytes(&ctx, &s6(&[1, 2, 3]), ROWS, COLS);
    let c6 = Q6_KWeights::from_bytes(&ctx, &s6(&[1, 9, 3]), ROWS, COLS);
    b6.write_rows(1, &r6(9), 1).expect("write Q6_K row 1");
    let (v6a, v6b, v6c) = (probe.matmul_q6_k(&a6).to_vec().await,
                           probe.matmul_q6_k(&b6).to_vec().await,
                           probe.matmul_q6_k(&c6).to_vec().await);
    let (b6c, b6a) = (maxdiff(&v6b, &v6c), maxdiff(&v6b, &v6a));
    println!("\n  Q6_K (the down slab):");
    println!("  max |B − C|  {b6c:.3e}");
    println!("  max |B − A|  {b6a:.3e}");
    assert_eq!(b6c, 0.0, "Q6_K written slab does not match one BUILT with the same rows");
    assert!(b6a > 0.0, "Q6_K write_rows did nothing — the original is unchanged");
    assert!(b6.write_rows(ROWS, &r6(4), 1).is_err(), "Q6_K wrote past the last row");
    assert!(b6.write_rows(0, &r6(4)[..B6 - 1], 1).is_err(), "Q6_K accepted a short slice");

    println!("\n  ✅ BOTH halves of an expert are replaceable, bit-exact, with the original genuinely\n  \
              changed and out-of-range or mis-sized writes refused. Q4_K gate|up and Q6_K down are\n  \
              what a Q4_K_M MoE actually stores, so streaming needs both.");
}
