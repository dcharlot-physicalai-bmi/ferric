//! Skeptic probe 2: adversarial offset views into CAT_WGSL — transposed, broadcast (stride-0),
//! large offsets, rank-1, and same-buffer aliasing.
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let mut ok = true;
    let mut chk = |name: &str, got: &[f32], want: &[f32]| {
        let p = got == want;
        ok &= p;
        println!("{} {name}", if p { "PASS" } else { "FAIL" });
        if !p { println!("   want={:?}\n   got ={:?}", want, got); }
    };

    // ---- T1: TRANSPOSED views (non-monotonic strides) with nonzero offsets ----
    let d: Vec<f32> = (0..30).map(|i| i as f32).collect(); // [5,6]
    let x = Tensor::from_vec(&ctx, &d, &[5, 6]);
    // narrow rows 1..4 (offset 6), then transpose -> shape [6,3], strides [1,6], offset 6
    let a = x.narrow(0, 1, 3).transpose(0, 1);
    // narrow rows 2..4 (offset 12), transpose -> shape [6,2], strides [1,6], offset 12
    let b = x.narrow(0, 2, 2).transpose(0, 1);
    let c = a.cat(&b, 1);
    let mut want = Vec::new();
    for col in 0..6 {
        for r in 1..4 { want.push(d[r * 6 + col]); }
        for r in 2..4 { want.push(d[r * 6 + col]); }
    }
    chk("T1 transposed views, offA=6 offB=12", &c.to_vec().await, &want);

    // ---- T2: BROADCAST (stride-0) view with a nonzero offset ----
    // row 3 of x, offset 18, broadcast across 4 rows
    let row = x.narrow(0, 3, 1).broadcast_to(&[4, 6]); // strides [0,1], offset 18
    let other = x.narrow(0, 1, 1).broadcast_to(&[4, 6]); // strides [0,1], offset 6
    let c2 = row.cat(&other, 1);
    let mut want2 = Vec::new();
    for _ in 0..4 {
        for j in 0..6 { want2.push(d[18 + j]); }
        for j in 0..6 { want2.push(d[6 + j]); }
    }
    chk("T2 broadcast stride-0, offA=18 offB=6", &c2.to_vec().await, &want2);

    // ---- T3: rank-1 with offsets ----
    let v: Vec<f32> = (0..20).map(|i| 3.0 + i as f32).collect();
    let y = Tensor::from_vec(&ctx, &v, &[20]);
    let c3 = y.narrow(0, 13, 4).cat(&y.narrow(0, 5, 3), 0);
    let mut want3: Vec<f32> = v[13..17].to_vec();
    want3.extend_from_slice(&v[5..8]);
    chk("T3 rank-1 offA=13 offB=5", &c3.to_vec().await, &want3);

    // ---- T4: LARGE offset (> 65535, > 2^20) so any u16/u32-packing bug shows ----
    let n = 3_000_000usize;
    let big: Vec<f32> = (0..n).map(|i| (i % 65_521) as f32).collect();
    let bt = Tensor::from_vec(&ctx, &big, &[1, n]);
    let off = 2_500_003usize;
    let la = bt.narrow(1, off, 5);
    let lb = bt.narrow(1, 1_048_579, 5);
    let c4 = la.cat(&lb, 1);
    let mut want4: Vec<f32> = big[off..off + 5].to_vec();
    want4.extend_from_slice(&big[1_048_579..1_048_584]);
    chk("T4 large offsets 2500003 / 1048579", &c4.to_vec().await, &want4);

    // ---- T5: same buffer, both sides, overlapping ranges (aliased read bindings) ----
    let c5 = y.narrow(0, 2, 5).cat(&y.narrow(0, 4, 5), 0);
    let mut want5: Vec<f32> = v[2..7].to_vec();
    want5.extend_from_slice(&v[4..9]);
    chk("T5 same-buffer overlapping offsets", &c5.to_vec().await, &want5);

    // ---- T6: reshape after narrow (offset survives a reshape) ----
    let r = x.narrow(0, 2, 2).reshape(&[3, 4]); // offset 12
    let s = x.narrow(0, 0, 2).reshape(&[3, 4]); // offset 0
    let c6 = r.cat(&s, 1);
    let mut want6 = Vec::new();
    for i in 0..3 {
        for j in 0..4 { want6.push(d[12 + i * 4 + j]); }
        for j in 0..4 { want6.push(d[i * 4 + j]); }
    }
    chk("T6 reshape-after-narrow offA=12", &c6.to_vec().await, &want6);

    println!("{}", if ok { "ALL PASS" } else { "SOME FAILED" });
}
