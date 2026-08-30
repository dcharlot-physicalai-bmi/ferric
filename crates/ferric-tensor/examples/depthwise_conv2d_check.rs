//! **Does `depthwise_conv2d` compute a depthwise convolution?** Against an independent CPU oracle.
//!
//! A depthwise conv that reads the wrong channel still produces finite, plausible activations — the
//! model just convolves each channel with someone else's filter. Nothing downstream can assert on
//! it. This is the same class as the `[K,C]`-vs-`[C,K]` depthwise-1d transpose that cost an
//! afternoon on the parakeet conv module, so it gets an oracle before it is wired to anything.
//!
//! The oracle is written from the DEFINITION (out[y,x,o] = sum_k w[k,o] * x[y*s+ky-p, x*s+kx-p, o]),
//! not from the kernel, and includes non-square shapes and stride/pad combinations where an index
//! slip would otherwise land inside the array and go unnoticed.
use ferric_tensor::Tensor;
use std::sync::Arc;

#[allow(clippy::too_many_arguments)]
fn cpu_depthwise(x: &[f32], w: &[f32], n: usize, h: usize, wd: usize, c: usize,
                 kh: usize, kw: usize, stride: (usize, usize), pad: (usize, usize)) -> Vec<f32> {
    let ho = (h + 2 * pad.0 - kh) / stride.0 + 1;
    let wo = (wd + 2 * pad.1 - kw) / stride.1 + 1;
    let mut out = vec![0f32; n * ho * wo * c];
    for b in 0..n { for y in 0..ho { for xx in 0..wo { for o in 0..c {
        let mut acc = 0f32;
        for ky in 0..kh { for kx in 0..kw {
            let yi = (y * stride.0 + ky) as isize - pad.0 as isize;
            let xi = (xx * stride.1 + kx) as isize - pad.1 as isize;
            if yi < 0 || xi < 0 || yi >= h as isize || xi >= wd as isize { continue; }
            acc += x[((b * h + yi as usize) * wd + xi as usize) * c + o] * w[(ky * kw + kx) * c + o];
        }}
        out[((b * ho + y) * wo + xx) * c + o] = acc;
    }}}}
    out
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let mut bad = 0;
    // The parakeet case is first: 3x3 stride 2 pad 1 over 256 channels.
    let cases: &[(usize, usize, usize, usize, usize, usize, (usize, usize), (usize, usize))] = &[
        (1, 586, 128, 256, 3, 3, (2, 2), (1, 1)),   // parakeet dw_striding stage, real shape
        (1, 5, 4, 3, 3, 3, (1, 1), (1, 1)),
        (1, 8, 8, 16, 3, 3, (2, 2), (1, 1)),
        (2, 7, 5, 4, 3, 3, (2, 2), (1, 1)),         // batched, non-square, odd dims
        (1, 6, 9, 5, 2, 3, (1, 2), (0, 1)),         // asymmetric kernel AND stride AND pad
    ];
    for &(n, h, wd, c, kh, kw, stride, pad) in cases {
        let x: Vec<f32> = (0..n * h * wd * c).map(|i| 0.1 * (((i * 7 + 3) % 23) as f32 - 11.0)).collect();
        let w: Vec<f32> = (0..kh * kw * c).map(|i| 0.1 * (((i * 5 + 2) % 13) as f32 - 6.0)).collect();
        let xt = Tensor::from_vec(&ctx, &x, &[n, h, wd, c]);
        let wt = Tensor::from_vec(&ctx, &w, &[kh, kw, 1, c]);
        let got = xt.depthwise_conv2d(&wt, stride, pad).to_vec().await;
        let want = cpu_depthwise(&x, &w, n, h, wd, c, kh, kw, stride, pad);
        assert_eq!(got.len(), want.len(), "shape {n}x{h}x{wd}x{c}: {} vs {}", got.len(), want.len());
        let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("  n={n} {h}x{wd}x{c} k={kh}x{kw} s={stride:?} p={pad:?}  max|err| {err:.2e}");
        if !(err < 1e-5) { bad += 1; }
    }

    // Discrimination check: a depthwise conv must NOT equal a full conv over the same weights, or
    // this oracle would pass on an implementation that summed across channels.
    let (h, wd, c) = (6usize, 6usize, 4usize);
    let x: Vec<f32> = (0..h * wd * c).map(|i| 0.1 * (i % 9) as f32).collect();
    let w: Vec<f32> = (0..3 * 3 * c).map(|i| 0.1 * (i % 5) as f32).collect();
    let dw = cpu_depthwise(&x, &w, 1, h, wd, c, 3, 3, (1, 1), (1, 1));
    let summed: f32 = dw.iter().sum();
    assert!(summed.abs() > 1e-3, "degenerate test data — the oracle cannot discriminate");

    println!("\n{}", if bad == 0 { "depthwise_conv2d matches the CPU oracle on every shape" }
                     else { "DEPTHWISE_CONV2D IS WRONG" });
    assert_eq!(bad, 0);
}
