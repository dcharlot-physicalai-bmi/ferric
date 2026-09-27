//! **Metal-4 tensor-unit quantized GEMM** — prefill's `x·Wᵀ` over Q8_0 / Q4_K / Q6_K weights on the
//! M5's matrix units, reading the packed blocks directly. Opt-in: `FERRIC_QGEMM=1`.
//!
//! # Why the existing tensor-unit path never ran for a quantized model
//!
//! `metal4.rs` already reaches the same hardware (`mpp::tensor_ops::matmul2d`, 23–31 TFLOP/s), but a
//! quantized prefill could not use it, for three independent reasons, any one of which was enough:
//!   1. its resident path needs the `MTLBuffer` behind a wgpu buffer, which only exists with the
//!      `metal4-interop` feature — OFF by default and enabled by no crate in the workspace, so
//!      `resident_for` returns `None` in every build and `FERRIC_METAL4=1` changes nothing;
//!   2. no quantized matmul calls it — only `matmul_bt` on a dense f32 weight and Q2_0's
//!      dequantize-to-f32-then-`matmul_bt` do;
//!   3. it takes dense f32 operands and pad-converts them to f16 on the GPU, so a quantized weight
//!      would first have to be materialised at 4 bytes per value.
//!
//! # What this does instead
//!
//! The kernel (`qgemm.metal`) is hand-written MSL compiled by the driver and created through wgpu's
//! **passthrough** shader API, so it is an ordinary wgpu compute pipeline: it binds the same buffers
//! a WGSL kernel binds (the activation tensor, the weight's `codes`/`aux`, a fresh output) and is
//! recorded by [`crate::record_dispatch`] into the same batched compute pass. There is no second
//! command queue, no host synchronisation per GEMM and no interop feature — and no weight copy: each
//! threadgroup dequantizes a 64×32 weight slice from the packed blocks into threadgroup memory as
//! fp16, stages the matching 64×32 activation slice as fp16, and feeds both to `matmul2d`, which
//! accumulates in fp32 across the whole K walk. Memory stays exactly what the quantized file costs.
//!
//! # Precision, and why this is opt-in
//!
//! The matrix units take fp16 (or bf16) operands. The weight enters as its quantized value rounded
//! once to fp16 (formed in f32 first), the activation as its f32 value rounded once to fp16; the
//! accumulation is fp32. So this path is **not bit-identical to the portable WGSL path**, which is
//! f32 end to end — cross-fabric bit-identity is a property of the portable graph, and a native tier
//! carries its own fingerprint (the rule `cuda.rs` and `FERRIC_METAL4` follow).
//!
//! What it costs, measured (M5 Max; `scripts/qgemm_conformance.sh`, full-vocabulary KL against the
//! authors' `transformers` in float64 over 126 positions of a 556-token prefill): on a Q8_0 made
//! from Qwen2.5-0.5B's own weights (`refgen/requant_gguf.py`) the portable path sits at KL 5.99e-4
//! from the authors and this route at 6.07e-4; the two routes differ from each other by KL 1.39e-6 —
//! 430x less than the quantization itself (HF's own float32-vs-float64 floor is 3.9e-11). On
//! Qwen2.5-1.5B's published Q4_K_M: 1.70e-2 both ways, 1.06e-6 between them. Greedy continuations were
//! identical on all nine (model, prompt) pairs tried. ⚠ fp16's range is ±65504: an activation beyond it becomes inf.
//! None of the tested models comes near it, and the conformance gate refuses non-finite logits, but a
//! model with larger activations must be gated before this route is trusted on it.
//!
//! What it buys: `examples/qgemm_bench.rs` (kernel) and `examples/prefill_bench.rs` (model).
//!
//! # Where it applies
//!
//! Only multi-row calls (`rows >= FERRIC_QGEMM_MIN_ROWS`, default 32): prefill. Decode (rows = 1)
//! stays on the portable GEMV kernels, which are faster there: with `FERRIC_QGEMM_MIN_ROWS=1`,
//! Qwen2.5-1.5B Q4_K_M decoded 1.4-2.3x slower (three interleaved pairs, contended machine) — a
//! 64-row tile computing one real row is not what the matrix units are for. A row's result does not depend on which rows share its
//! dispatch (tested), so with `FERRIC_QGEMM_MIN_ROWS=1` the tier is self-consistent: a last-row head
//! equals the full head's last row to the bit. With the default floor a short prefix-cached suffix
//! runs portable while the full prompt ran here — and because fp16 rounding is discontinuous, the
//! f32 attention kernels' own tiny disagreement (1.6e-5) surfaces as ~1e-2 in the logits, so the
//! prefix-cache example's 1e-3 bound does not hold under this route (predictions still identical).
#![cfg(all(target_os = "macos", not(target_arch = "wasm32")))]

use crate::{Context, Tensor};
use std::borrow::Cow;
use std::sync::Arc;

const SRC: &str = include_str!("qgemm.metal");
/// Output tile the kernel computes per threadgroup (M rows × N columns) — `MT`/`NT` in the MSL.
const TILE_M: usize = 64;
const TILE_N: usize = 64;
const THREADS: u32 = 128;

/// The weight formats the kernel dequantizes, named by their MSL entry points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QFmt {
    Q8_0,
    Q4K,
    Q6K,
}

impl QFmt {
    fn entry(self, swiglu: bool) -> &'static str {
        match (self, swiglu) {
            (QFmt::Q8_0, false) => "qmm_q8_0",
            (QFmt::Q4K, false) => "qmm_q4_k",
            (QFmt::Q6K, false) => "qmm_q6_k",
            (QFmt::Q8_0, true) => "qmm_swiglu_q8_0",
            (QFmt::Q4K, true) => "qmm_swiglu_q4_k",
            (QFmt::Q6K, true) => "qmm_swiglu_q6_k",
        }
    }
    /// K must be a whole number of blocks (and of the kernel's 32-wide K slice).
    fn k_multiple(self) -> usize {
        match self {
            QFmt::Q8_0 => 32,
            QFmt::Q4K | QFmt::Q6K => 256,
        }
    }
}

type Pipes = std::collections::HashMap<(usize, QFmt, bool), Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>>;
thread_local! {
    // Keyed by device identity like `pipeline_for`, so a device-A pipeline never reaches device B.
    // `None` records a failed build (a device or OS without Metal-4 tensor ops) so it is tried once.
    static PIPES: std::cell::RefCell<Pipes> = std::cell::RefCell::new(Default::default());
}

fn build(ctx: &Context, fmt: QFmt, swiglu: bool) -> Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
    let dev = &ctx.device;
    // A compile failure (no MetalPerformancePrimitives, an older GPU family) must DECLINE the route,
    // not panic the process — wgpu's default uncaptured-error handler panics, so scope it.
    let scope = dev.push_error_scope(wgpu::ErrorFilter::Validation);
    // FERRIC_QGEMM_FAULT plants a known decoding error (see qgemm.metal) — the negative control of
    // scripts/qgemm_conformance.sh, which must see the model get worse or it proves nothing.
    let src = if std::env::var_os("FERRIC_QGEMM_FAULT").is_some() {
        Cow::Owned(format!("#define QGEMM_FAULT 1\n{SRC}"))
    } else {
        Cow::Borrowed(SRC)
    };
    let module = unsafe {
        dev.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: Some("qgemm"),
            entry_points: Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint {
                name: fmt.entry(swiglu).into(),
                workgroup_size: (THREADS, 1, 1),
            }]),
            msl: Some(src),
            ..Default::default()
        })
    };
    let buf = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let ro = wgpu::BufferBindingType::Storage { read_only: true };
    let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("qgemm"),
        entries: &[
            buf(0, ro),
            buf(1, ro),
            buf(2, ro),
            buf(3, wgpu::BufferBindingType::Storage { read_only: false }),
            buf(4, wgpu::BufferBindingType::Uniform),
        ],
    });
    let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("qgemm"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipe = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(fmt.entry(swiglu)),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(fmt.entry(swiglu)),
        compilation_options: Default::default(),
        cache: None,
    });
    if let Some(e) = pollster::block_on(scope.pop()) {
        eprintln!("ferric: FERRIC_QGEMM requested but the Metal-4 tensor-op kernel did not build ({e}); using the portable path");
        return None;
    }
    Some((pipe, bgl))
}

/// Whether the route is switched on for this context: a Metal device with passthrough MSL, and the
/// explicit `FERRIC_QGEMM` opt-in (read once — flipping it mid-process would mix numerics).
pub fn enabled(ctx: &Context) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    ctx.native_msl && *ON.get_or_init(|| std::env::var_os("FERRIC_QGEMM").is_some_and(|v| v != "0"))
}

/// The fewest activation rows routed to the tensor units. Below it the portable GEMV kernels keep
/// decode (rows = 1) and small batched decode on the bit-identical path; `FERRIC_QGEMM_MIN_ROWS`
/// overrides for measurement.
pub fn min_rows() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("FERRIC_QGEMM_MIN_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(32))
}

/// `y = x·Wᵀ` on the tensor units for a packed weight given by its `codes`/`aux` buffers and
/// `[n_out, k]` shape. `None` (touching nothing) when the route is off, the shape is outside the
/// kernel's contract, or the kernel could not be built — the caller then runs its portable kernel.
pub(crate) fn qmm(x: &Tensor, codes: &wgpu::Buffer, aux: &wgpu::Buffer, n_out: usize, k: usize, fmt: QFmt) -> Option<Tensor> {
    if !routed(x) {
        return None;
    }
    qmm_unchecked(x, codes, aux, n_out, k, fmt, false)
}

/// Fused FFN `silu(x·Wgᵀ) ⊙ (x·Wuᵀ)` for a gate|up weight `[2·n_ff, k]` (gate rows first), as one
/// tensor-unit dispatch writing `[rows, n_ff]`. Same gates as [`qmm`].
pub(crate) fn qmm_swiglu(x: &Tensor, codes: &wgpu::Buffer, aux: &wgpu::Buffer, n_out: usize, k: usize, fmt: QFmt) -> Option<Tensor> {
    if !routed(x) {
        return None;
    }
    qmm_unchecked(x, codes, aux, n_out, k, fmt, true)
}

fn routed(x: &Tensor) -> bool {
    enabled(&x.ctx) && x.shape.len() == 2 && x.shape[0] >= min_rows()
}

/// [`qmm`] / [`qmm_swiglu`] without the opt-in and row-count gates — the hermetic seam the tests
/// use, so they reach the kernel without setting a process-wide env var that would reroute every
/// other test's matmuls.
pub(crate) fn qmm_unchecked(x: &Tensor, codes: &wgpu::Buffer, aux: &wgpu::Buffer, n_out: usize, k: usize, fmt: QFmt,
                            swiglu: bool) -> Option<Tensor> {
    let ctx: &Arc<Context> = &x.ctx;
    if !ctx.native_msl || x.shape.len() != 2 || x.shape[1] != k || k % fmt.k_multiple() != 0 || n_out == 0
        || x.shape[0] == 0 || (swiglu && n_out % 2 != 0) {
        return None;
    }
    let rows = x.shape[0];
    let key = (&ctx.device as *const wgpu::Device as usize, fmt, swiglu);
    let built = PIPES.with(|p| p.borrow_mut().entry(key).or_insert_with(|| build(ctx, fmt, swiglu)).clone())?;
    let x = x.contiguous(); // offset 0, row-major: the kernel indexes x[m*K + k] from the buffer start
    let (cols, n_ff, tile_n) = if swiglu { (n_out / 2, n_out / 2, TILE_N / 2) } else { (n_out, 0, TILE_N) };
    let out = crate::empty(ctx, rows * cols);
    let dims = crate::unibuf(ctx, &[rows as u32, n_out as u32, k as u32, n_ff as u32]);
    if std::env::var_os("FERRIC_CENSUS").is_some() {
        crate::census_bump(fmt.entry(swiglu));
    }
    let grid = (cols.div_ceil(tile_n) as u32, rows.div_ceil(TILE_M) as u32, 1);
    crate::record_dispatch(ctx, fmt.entry(swiglu), &built.0, &built.1, &[x.buf.as_ref(), codes, aux, &out, &dims], grid);
    Some(Tensor::from_parts(ctx, out, vec![rows, cols]))
}

#[cfg(test)]
mod tests {
    use crate::dtype::{Q4_KWeights, Q6_KWeights, Q8_0Weights};
    use crate::Tensor;
    use std::sync::Arc;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn unit(&mut self) -> f64 { (self.next() % 2_000_001) as f64 / 1e6 - 1.0 }
        fn byte(&mut self) -> u8 { self.next() as u8 }
    }
    fn f16le(v: f64) -> [u8; 2] { half::f16::from_f64(v).to_bits().to_le_bytes() }
    fn h(b: &[u8], o: usize) -> f64 { half::f16::from_bits(u16::from_le_bytes([b[o], b[o + 1]])).to_f64() }
    fn r16(v: f64) -> f64 { half::f16::from_f64(v).to_f64() }

    // ---- The formats as ggml DEFINES them (dequantize_row_q8_0 / _q4_K / _q6_K), in f64, read from
    // the RAW GGUF block bytes. Independent of the GPU repack in dtype.rs and of the MSL kernel. ----
    fn deq_q8_0(raw: &[u8], n: usize, k: usize) -> Vec<f64> {
        let mut w = vec![0.0; n * k];
        for (b, blk) in raw.chunks(34).enumerate() {
            let d = h(blk, 0);
            for l in 0..32 { w[b * 32 + l] = d * (blk[2 + l] as i8) as f64; }
        }
        w
    }
    fn scale_min_k4(j: usize, q: &[u8]) -> (f64, f64) {
        if j < 4 { ((q[j] & 63) as f64, (q[j + 4] & 63) as f64) }
        else { (((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)) as f64, ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) as f64) }
    }
    /// `nibble_swap` builds the plausible WRONG variant (low/high nibble roles exchanged) — the
    /// negative control, never the reference.
    fn deq_q4_k(raw: &[u8], n: usize, k: usize, nibble_swap: bool) -> Vec<f64> {
        let mut w = vec![0.0; n * k];
        for (b, blk) in raw.chunks(144).enumerate() {
            let (d, dmin) = (h(blk, 0), h(blk, 2));
            let (sc, qs) = (&blk[4..16], &blk[16..144]);
            for j in 0..4 {
                let (s1, m1) = scale_min_k4(2 * j, sc);
                let (s2, m2) = scale_min_k4(2 * j + 1, sc);
                for l in 0..32 {
                    let (lo, hi) = ((qs[32 * j + l] & 0xF) as f64, (qs[32 * j + l] >> 4) as f64);
                    let (a, c) = if nibble_swap { (hi, lo) } else { (lo, hi) };
                    w[b * 256 + 64 * j + l] = d * s1 * a - dmin * m1;
                    w[b * 256 + 64 * j + 32 + l] = d * s2 * c - dmin * m2;
                }
            }
        }
        w
    }
    /// `qh_shift_wrong` is the negative control: the 2 high bits taken from the wrong position.
    fn deq_q6_k(raw: &[u8], n: usize, k: usize, qh_shift_wrong: bool) -> Vec<f64> {
        let mut w = vec![0.0; n * k];
        for (b, blk) in raw.chunks(210).enumerate() {
            let d = h(blk, 208);
            for half_ in 0..2 {
                let (ql, qh, sc) = (&blk[64 * half_..], &blk[128 + 32 * half_..], &blk[192 + 8 * half_..]);
                for l in 0..32 {
                    let is = l / 16;
                    let sh = |s: u32| if qh_shift_wrong { (s + 2) % 8 } else { s };
                    let q1 = ((ql[l] & 0xF) | (((qh[l] >> sh(0)) & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> sh(2)) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> sh(4)) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> sh(6)) & 3) << 4)) as i32 - 32;
                    let o = b * 256 + 128 * half_;
                    w[o + l] = d * (sc[is] as i8) as f64 * q1 as f64;
                    w[o + l + 32] = d * (sc[is + 2] as i8) as f64 * q2 as f64;
                    w[o + l + 64] = d * (sc[is + 4] as i8) as f64 * q3 as f64;
                    w[o + l + 96] = d * (sc[is + 6] as i8) as f64 * q4 as f64;
                }
            }
        }
        w
    }
    fn rand_blocks(rng: &mut Rng, nblk: usize, fmt: &str) -> Vec<u8> {
        let mut raw = Vec::new();
        for _ in 0..nblk {
            match fmt {
                "q8_0" => {
                    raw.extend(f16le(0.002 + 0.002 * rng.unit()));
                    for _ in 0..32 { raw.push(rng.byte()); }
                }
                "q4_k" => {
                    raw.extend(f16le(0.004 + 0.002 * rng.unit()));
                    raw.extend(f16le(0.002 + 0.001 * rng.unit()));
                    for _ in 0..140 { raw.push(rng.byte()); }
                }
                _ => {
                    for _ in 0..208 { raw.push(rng.byte()); }
                    raw.extend(f16le(0.0005 + 0.0002 * rng.unit()));
                }
            }
        }
        raw
    }

    /// (max |got − fp16-input ref|, max |got − exact ref|, max_row Σ|x·w|) over every output.
    fn errors(got: &[f32], x: &[f32], w: &[f64], m: usize, n: usize, k: usize) -> (f64, f64, f64) {
        let (mut e16, mut e, mut scale) = (0f64, 0f64, 0f64);
        let x16: Vec<f64> = x.iter().map(|&v| r16(v as f64)).collect();
        let w16: Vec<f64> = w.iter().map(|&v| r16(v)).collect();
        for i in 0..m {
            for j in 0..n {
                let (mut a, mut a16, mut s) = (0f64, 0f64, 0f64);
                for l in 0..k {
                    a += x[i * k + l] as f64 * w[j * k + l];
                    a16 += x16[i * k + l] * w16[j * k + l];
                    s += (x[i * k + l] as f64 * w[j * k + l]).abs();
                }
                let g = got[i * n + j] as f64;
                e16 = e16.max((g - a16).abs());
                e = e.max((g - a).abs());
                scale = scale.max(s);
            }
        }
        (e16, e, scale)
    }

    fn ctx() -> Option<Arc<ferric_core::Context>> {
        let c = pollster::block_on(ferric_core::Context::new()).ok()?;
        if !c.native_msl {
            eprintln!("no passthrough-MSL Metal device — skipping");
            return None;
        }
        Some(Arc::new(c))
    }

    /// The kernel against the format's own definition, for all three formats, at ragged M and N
    /// (neither a tile multiple) and K spanning several blocks. The pass bar is the fp16-INPUT
    /// reference to fp32-accumulation noise (1e-6 of Σ|x·w|); the exact-f64 distance is reported and
    /// bounded by the fp16 rounding it is made of. Negative controls: each plausible wrong decoding
    /// (neighbour block's Q8_0 scale, swapped Q4_K nibbles, mis-shifted Q6_K high bits) must miss by
    /// ≥ 20x the tolerance, or this test could not see that defect.
    #[test]
    fn tensor_unit_qgemm_matches_the_format_definition() {
        let Some(ctx) = ctx() else { return };
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for (fmt, m, n, k) in [("q8_0", 100, 200, 224), ("q8_0", 64, 64, 896), ("q4_k", 100, 200, 1024),
                               ("q4_k", 64, 128, 512), ("q6_k", 100, 200, 1024), ("q6_k", 70, 96, 256)] {
            let bs = match fmt { "q8_0" => 32, _ => 256 };
            let raw = rand_blocks(&mut rng, n * k / bs, fmt);
            let x: Vec<f32> = (0..m * k).map(|_| rng.unit() as f32).collect();
            let xt = Tensor::from_vec(&ctx, &x, &[m, k]);
            let (got, w, wrong) = match fmt {
                "q8_0" => {
                    let wq = Q8_0Weights::from_bytes(&ctx, &raw, n, k);
                    let g = xt.native_qmm_q8_0(&wq).expect("kernel");
                    // wrong: each block scaled by its NEIGHBOUR's d
                    let mut raw2 = raw.clone();
                    let nb = n * k / 32;
                    for b in 0..nb { let o = ((b + 1) % nb) * 34; raw2[b * 34] = raw[o]; raw2[b * 34 + 1] = raw[o + 1]; }
                    (g, deq_q8_0(&raw, n, k), deq_q8_0(&raw2, n, k))
                }
                "q4_k" => {
                    let wq = Q4_KWeights::from_bytes(&ctx, &raw, n, k);
                    (xt.native_qmm_q4_k(&wq).expect("kernel"), deq_q4_k(&raw, n, k, false), deq_q4_k(&raw, n, k, true))
                }
                _ => {
                    let wq = Q6_KWeights::from_bytes(&ctx, &raw, n, k);
                    (xt.native_qmm_q6_k(&wq).expect("kernel"), deq_q6_k(&raw, n, k, false), deq_q6_k(&raw, n, k, true))
                }
            };
            let got = pollster::block_on(got.to_vec());
            let (e16, e, scale) = errors(&got, &x, &w, m, n, k);
            let (_, e_wrong, _) = errors(&got, &x, &wrong, m, n, k);
            let tol = 1e-6 * scale;
            let fp16_bound = 2.0f64.powi(-10) * scale; // each product carries ≤ 2 half-ulp roundings
            eprintln!("{fmt} M={m} N={n} K={k}: |gpu−fp16ref| {e16:.2e} (tol {tol:.2e})  |gpu−exact| {e:.2e} \
                       (fp16 bound {fp16_bound:.2e})  wrong-decoding miss {e_wrong:.2e} = {:.0}x tol", e_wrong / tol);
            assert!(e16 <= tol, "{fmt} M={m} N={n} K={k}: kernel vs fp16-input reference {e16:.3e} > {tol:.3e}");
            assert!(e <= fp16_bound, "{fmt}: kernel vs exact reference {e:.3e} beyond the fp16 bound {fp16_bound:.3e}");
            assert!(e_wrong >= 20.0 * tol, "{fmt}: negative control missed by only {e_wrong:.3e} — the check cannot see it");
        }
    }

    /// The fused SwiGLU epilogue (gate|up weight `[2·n_ff, K]`, gate rows first) against the format
    /// definition: `silu(g)·u` from the fp16-input f64 reference of g and u. n_ff = 100 is not a
    /// multiple of the 32-column half-tile, so the gate/up row clamp and the store guard are both on
    /// the hot path. Negative control: the halves swapped (`silu(u)·g`) must miss by ≥ 20x.
    #[test]
    fn tensor_unit_swiglu_epilogue_matches_the_format_definition() {
        let Some(ctx) = ctx() else { return };
        let mut rng = Rng(0x0dd_ba11);
        for (fmt, m, n_ff, k) in [("q8_0", 100, 100, 224), ("q4_k", 70, 100, 512), ("q6_k", 64, 64, 256)] {
            let n = 2 * n_ff;
            let bs = if fmt == "q8_0" { 32 } else { 256 };
            let raw = rand_blocks(&mut rng, n * k / bs, fmt);
            let x: Vec<f32> = (0..m * k).map(|_| rng.unit() as f32).collect();
            let xt = Tensor::from_vec(&ctx, &x, &[m, k]);
            let (got, w) = match fmt {
                "q8_0" => (xt.native_qmm_swiglu_q8_0(&Q8_0Weights::from_bytes(&ctx, &raw, n, k)), deq_q8_0(&raw, n, k)),
                "q4_k" => (xt.native_qmm_swiglu_q4_k(&Q4_KWeights::from_bytes(&ctx, &raw, n, k)), deq_q4_k(&raw, n, k, false)),
                _ => (xt.native_qmm_swiglu_q6_k(&Q6_KWeights::from_bytes(&ctx, &raw, n, k)), deq_q6_k(&raw, n, k, false)),
            };
            let got = pollster::block_on(got.expect("kernel").to_vec());
            assert_eq!(got.len(), m * n_ff);
            let silu = |v: f64| v / (1.0 + (-v).exp());
            let (mut e, mut e_swap, mut scale) = (0f64, 0f64, 0f64);
            for i in 0..m {
                for j in 0..n_ff {
                    let (mut g, mut u, mut sg, mut su) = (0f64, 0f64, 0f64, 0f64);
                    for l in 0..k {
                        let xv = r16(x[i * k + l] as f64);
                        g += xv * r16(w[j * k + l]);
                        u += xv * r16(w[(n_ff + j) * k + l]);
                        sg += (xv * w[j * k + l]).abs();
                        su += (xv * w[(n_ff + j) * k + l]).abs();
                    }
                    let y = got[i * n_ff + j] as f64;
                    e = e.max((y - silu(g) * u).abs());
                    e_swap = e_swap.max((y - silu(u) * g).abs());
                    scale = scale.max(sg * u.abs().max(1.0) + su * g.abs().max(1.0));
                }
            }
            let tol = 1e-6 * scale;
            eprintln!("{fmt} swiglu M={m} n_ff={n_ff} K={k}: |gpu−ref| {e:.2e} (tol {tol:.2e})  swapped-halves miss {e_swap:.2e} = {:.0}x tol", e_swap / tol);
            assert!(e <= tol, "{fmt} swiglu: {e:.3e} > {tol:.3e}");
            assert!(e_swap >= 20.0 * tol, "{fmt} swiglu: negative control missed by only {e_swap:.3e}");
        }
    }

    /// A row's result does not depend on which other rows share its dispatch: the same 37 rows give
    /// the same BITS computed alone, at the head of a 100-row batch, or at an offset that puts them in
    /// a different 64-row tile and tile position — and one row alone matches its row in the batch.
    /// This is what lets a prefix-cached suffix, a batched decode step and a last-row head agree to
    /// the bit with the full computation on this tier.
    #[test]
    fn tensor_unit_rows_are_independent_of_the_batch() {
        let Some(ctx) = ctx() else { return };
        let mut rng = Rng(0xfeed_f00d);
        let (m, n, k) = (100, 130, 512);
        for fmt in ["q8_0", "q4_k", "q6_k"] {
            let bs = if fmt == "q8_0" { 32 } else { 256 };
            let raw = rand_blocks(&mut rng, n * k / bs, fmt);
            let x: Vec<f32> = (0..m * k).map(|_| rng.unit() as f32).collect();
            let run = |rows: &[f32]| -> Vec<f32> {
                let r = rows.len() / k;
                let xt = Tensor::from_vec(&ctx, rows, &[r, k]);
                let y = match fmt {
                    "q8_0" => xt.native_qmm_q8_0(&Q8_0Weights::from_bytes(&ctx, &raw, n, k)),
                    "q4_k" => xt.native_qmm_q4_k(&Q4_KWeights::from_bytes(&ctx, &raw, n, k)),
                    _ => xt.native_qmm_q6_k(&Q6_KWeights::from_bytes(&ctx, &raw, n, k)),
                };
                pollster::block_on(y.expect("kernel").to_vec())
            };
            let full = run(&x);
            for (start, len) in [(0usize, 37usize), (50, 37), (63, 37), (99, 1), (5, 1)] {
                let part = run(&x[start * k..(start + len) * k]);
                assert!(part == full[start * n..(start + len) * n],
                        "{fmt}: rows {start}..{} differ when computed apart from the batch", start + len);
            }
        }
    }

    /// The route is OFF unless asked for: without `FERRIC_QGEMM` in the environment `enabled` is
    /// false, so `qmm` declines before touching anything and the portable kernels run — which is what
    /// keeps the default build's numerics unchanged.
    #[test]
    fn route_is_opt_in() {
        let Some(ctx) = ctx() else { return };
        if std::env::var_os("FERRIC_QGEMM").is_some() { return; }
        assert!(!super::enabled(&ctx));
    }
}
