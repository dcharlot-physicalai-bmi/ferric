//! **Prefill attention on the M5 matrix units** — `attn.metal`, a FlashAttention-2 kernel whose two
//! matmuls (S = Q·Kᵀ, O += P·V) run as Metal-4 `matmul2d` fragments. Opt-in with the tensor-unit tier:
//! `FERRIC_QGEMM=1` (see `native_qgemm`), on a Metal device with passthrough MSL.
//!
//! Same contract as `Tensor::flash_attention_prefill_at` (a block of T queries at positions `off..off+T`
//! over `off+T` keys, causal, GQA) and the same passthrough route as the quantized GEMMs: a wgpu compute
//! pipeline built from MSL, recorded by [`crate::record_dispatch`] into the same batched pass as every
//! WGSL kernel — no second queue, no host sync. See `attn.metal` for the schedule and the precision
//! contract (Q, K, V rounded once to fp16; softmax, P and accumulation fp32).
//!
//! # What it buys (examples/flash_prefill_bench.rs, interleaved, min/median of 3-5)
//!
//! Kernel alone, Qwen2.5-0.5B shapes (14 heads, 2 KV heads, head_dim 64): T = 2,048 in 0.84-0.91 ms on the
//! M5 Max (load 40-70, shared) = 19x the per-query kernel's 15.8-17.7 ms, 8.3 TFLOP/s; T = 8,192 in
//! 18.4-19.1 ms = 24x. Qwen3-0.6B / Qwen2.5-1.5B / Llama-3.2-1B at 2,048: 18-20x. It never lost to the
//! per-query kernel at any block size tried (T = 2..256, fresh or continuing a 2,000-key cache), so the
//! tier routes every prefill block here. Whole model, Qwen2.5-0.5B Q8_0 on the idle M3 Ultra (its GPU runs
//! the same MSL; MPP `matmul2d` builds there too): FERRIC_QGEMM prefill 8,622-8,659 -> 12,329-12,356 tok/s
//! at 512, 5,259-5,276 -> 15,761-15,875 at 2,048, 1,743-1,744 -> 12,008 at 8,192; Qwen2.5-1.5B Q4_K_M
//! 1,048-1,050 -> 4,095-4,097 at 8,192.
//!
//! # What it costs
//!
//! Against float64 on the unit shapes: 1e-4..5e-4 (the P operand's own fp16 rounding bounds it at
//! 2^-11 · max|V|). Whole model (`scripts/attn_conformance.sh`, the authors' transformers float32, 556-token
//! prefill): on a Q8_0 of Qwen2.5-0.5B's own weights the mean |dlogit| to the authors is 0.05769 with it
//! and 0.05776 without; its own perturbation is 7.9% of that quantization band (0.4-0.5% on Qwen3-0.6B and
//! Qwen2.5-1.5B Q4_K_M). On an F32 file of the authors' weights it moves the logits by up to 3.5e-2 (mean
//! 3.4e-3) where the f32 path sits at 1e-4 — the fp16 fingerprint of a native tier, which is why it is
//! opt-in with FERRIC_QGEMM and never on the portable route.
//!
//! ⛔ The f32 P operand trap: MPP's `float x half -> float` left operand does NOT share the fp16 operand's
//! cooperative-tensor layout when `relaxed_precision` is off — a positional copy (MLX's pattern) then
//! computes garbage, O(1) off. MLX gets away with f32 P because it sets relaxed ON; rounding P to fp16
//! ourselves with relaxed OFF lands 3e-5..1.6e-4 from the fp16-input reference where MLX's setting lands
//! 4.7e-4..6.9e-4 (`matrix_unit_attention_probe`; see `Variant::PRODUCTION`).
#![cfg(all(target_os = "macos", not(target_arch = "wasm32")))]

use crate::{Context, Tensor};
use std::borrow::Cow;
use std::sync::Arc;

const SRC: &str = include_str!("attn.metal");
/// Query rows per threadgroup (4 simdgroups x 16) — `BQ` in the MSL.
const BQ: usize = 64;
const THREADS: u32 = 128;

/// A build of the kernel: head_dim, planted fault (0 = none; tests and the gate's negative control only),
/// and the P operand's precision (false: f32 as MLX passes it; true: rounded to fp16) — the variant knob
/// kept for measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Variant {
    pub dh: usize,
    pub fault: u32,
    pub p_half: bool,
    pub relaxed: bool,
}

impl Variant {
    /// What the route runs: P rounded to fp16 for the P·V matmul, `relaxed_precision` off. Measured on the
    /// float64 test shapes (`matrix_unit_attention_probe`): 1e-4..5e-4 from float64, 3e-5..1.5e-4 from the
    /// fp16-input reference. ⛔ P kept f32 with relaxed OFF is WRONG by O(1): the f32 left operand's
    /// cooperative-tensor layout is not the fp16 one, so the positional copy scrambles it. With relaxed ON
    /// (MLX's setting) f32 P works but lands at 5e-4..8e-4 — less accurate than rounding P ourselves.
    pub const PRODUCTION: Variant = Variant { dh: 0, fault: 0, p_half: true, relaxed: false };
    pub fn with_fault(self, fault: u32) -> Variant { Variant { fault, ..self } }
}

type Pipes = std::collections::HashMap<(usize, Variant), Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>>;
thread_local! {
    // Keyed by device identity like `pipeline_for`; `None` records a failed build so it is tried once.
    static PIPES: std::cell::RefCell<Pipes> = std::cell::RefCell::new(Default::default());
}

fn build(ctx: &Context, var: Variant) -> Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
    let dev = &ctx.device;
    // A compile failure (no MetalPerformancePrimitives, an older GPU family) must DECLINE the route, not
    // panic the process — wgpu's default uncaptured-error handler panics, so scope it.
    let scope = dev.push_error_scope(wgpu::ErrorFilter::Validation);
    let src = format!("#define DH {}\n#define ATTN_FAULT {}\n#define P_T {}\n#define RELAXED {}\n{SRC}",
                      var.dh, var.fault, if var.p_half { "half" } else { "float" }, var.relaxed);
    let module = unsafe {
        dev.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: Some("flash_attn_mu"),
            entry_points: Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint {
                name: "flash_attn_mu".into(),
                workgroup_size: (THREADS, 1, 1),
            }]),
            msl: Some(Cow::Owned(src)),
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
        label: Some("flash_attn_mu"),
        entries: &[
            buf(0, ro),
            buf(1, ro),
            buf(2, ro),
            buf(3, wgpu::BufferBindingType::Storage { read_only: false }),
            buf(4, wgpu::BufferBindingType::Uniform),
        ],
    });
    let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("flash_attn_mu"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipe = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("flash_attn_mu"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("flash_attn_mu"),
        compilation_options: Default::default(),
        cache: None,
    });
    if let Some(e) = pollster::block_on(scope.pop()) {
        eprintln!("ferric: the matrix-unit attention kernel did not build ({e}); using the portable path");
        return None;
    }
    Some((pipe, bgl))
}

/// `FERRIC_ATTN_FAULT=n` plants defect n (see attn.metal) in the routed kernel — the whole-model gate's
/// negative control. Read once.
fn env_fault() -> u32 {
    static F: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("FERRIC_ATTN_FAULT").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// Fewest query rows (T · nh) the routed (`auto`) path sends to the matrix units: 0. On the idle M3 Ultra
/// the kernel was never slower than the per-query one at any size tried (worst 1.0x: a 2-16-token block over
/// a 2,000-key cache at head_dim 128; `examples/flash_prefill_bench.rs` BENCH_SWEEP=1); the loaded M5 Max
/// showed 0.8x only on fresh blocks of T <= 64, where both take under 0.1 ms. `FERRIC_ATTN_MIN_ROWS`
/// overrides, for measurement.
fn min_rows() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("FERRIC_ATTN_MIN_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// The routed entry: None unless (for `auto`) the tensor-unit tier is switched on and the block is big
/// enough, and (always) the shape is in the kernel's contract on a passthrough-MSL device.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn(q: &Tensor, k: &Tensor, v: &Tensor, nh: usize, nkv: usize, dh: usize, off: usize, auto: bool) -> Option<Tensor> {
    attn_opts(q, k, v, nh, nkv, dh, off, 0, 0.0, auto)
}

/// [`attn`] with a sliding window (0 = none) and an attention softcap (0 = none) — see
/// `Tensor::flash_attention_prefill_opts`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_opts(q: &Tensor, k: &Tensor, v: &Tensor, nh: usize, nkv: usize, dh: usize, off: usize, window: usize,
                        softcap: f32, auto: bool) -> Option<Tensor> {
    if auto && (!crate::native_qgemm::enabled(&q.ctx) || q.shape[0] * nh < min_rows()) {
        return None;
    }
    attn_variant(q, k, v, nh, nkv, dh, off, window, softcap, Variant { dh, ..Variant::PRODUCTION }.with_fault(env_fault()))
}

/// The kernel at an explicit variant, without the opt-in gate — the hermetic seam the tests use.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_variant(q: &Tensor, k: &Tensor, v: &Tensor, nh: usize, nkv: usize, dh: usize, off: usize,
                           window: usize, softcap: f32, var: Variant) -> Option<Tensor> {
    let ctx: &Arc<Context> = &q.ctx;
    let t = q.shape[0];
    if !ctx.native_msl || dh == 0 || dh % 32 != 0 || dh > 128 || nkv == 0 || nh % nkv != 0 || t == 0 {
        return None;
    }
    let nblk = (t * (nh / nkv)).div_ceil(BQ);
    if nblk > 65_535 || nkv > 65_535 { return None; }
    let key = (&ctx.device as *const wgpu::Device as usize, var);
    let built = PIPES.with(|p| p.borrow_mut().entry(key).or_insert_with(|| build(ctx, var)).clone())?;
    let (q, k, v) = (q.contiguous(), k.contiguous(), v.contiguous());
    let out = crate::empty(ctx, t * nh * dh);
    let scale = 1.0 / (dh as f32).sqrt();
    let dims = crate::unibuf(ctx, &[nh as u32, nkv as u32, t as u32, off as u32, scale.to_bits(), nblk as u32,
                                    window as u32, softcap.max(0.0).to_bits()]);
    if std::env::var_os("FERRIC_CENSUS").is_some() {
        crate::census_bump("flash_attn_mu");
    }
    crate::record_dispatch(ctx, "flash_attn_mu", &built.0, &built.1,
                           &[q.buf.as_ref(), k.buf.as_ref(), v.buf.as_ref(), &out, &dims], (nblk as u32, nkv as u32, 1));
    Some(Tensor::from_parts(ctx, out, vec![t, nh * dh]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_tiled::flash_tiled_tests::{data, max_err, reference, reference_opts, SHAPES};
    use crate::flash_tiled::Kernel;

    fn ctx() -> Option<Arc<Context>> {
        let c = pollster::block_on(Context::new()).ok()?;
        if !c.native_msl {
            eprintln!("no passthrough-MSL Metal device — skipping");
            return None;
        }
        Some(Arc::new(c))
    }
    fn r16(x: &[f32]) -> Vec<f32> { x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect() }

    /// The kernel against float64, on every test shape whose head_dim it serves. Two references: the exact
    /// inputs, and the inputs rounded to fp16 the way the kernel rounds them. The bound is P's own fp16
    /// rounding — O = Σ P_j V_j / l with each P_j in [0, 1] carrying ≤ 2^-11 relative error, so
    /// |O − O_fp16in| ≤ 2^-11 · max|V| — and the exact reference may add Q/K/V's rounding on top (4x that).
    /// Negative controls: each planted defect (attn.metal ATTN_FAULT 1-3) must miss the fp16-input
    /// reference by ≥ 20x the bound on EVERY shape where its mechanism engages.
    #[test]
    fn matrix_unit_attention_matches_float64_and_sees_every_planted_defect() {
        let Some(ctx) = ctx() else { return };
        let mut worst_fault = [f64::INFINITY; 3];
        let mut tested = 0;
        for (n, &(t, off, nh, nkv, dh)) in SHAPES.iter().enumerate() {
            if dh % 32 != 0 { continue; }
            let (q, k, v) = data(t, off, nh, nkv, dh, 0x9e37_79b9 + n as u64);
            let want = reference(&q, &k, &v, t, off, nh, nkv, dh);
            let want16 = reference(&r16(&q), &r16(&k), &r16(&v), t, off, nh, nkv, dh);
            let qt = Tensor::from_vec(&ctx, &q, &[t, nh * dh]);
            let kt = Tensor::from_vec(&ctx, &k, &[off + t, nkv * dh]);
            let vt = Tensor::from_vec(&ctx, &v, &[off + t, nkv * dh]);
            let run = |fault: u32| {
                let var = Variant { dh, ..Variant::PRODUCTION }.with_fault(fault);
                pollster::block_on(attn_variant(&qt, &kt, &vt, nh, nkv, dh, off, 0, 0.0, var).expect("kernel").to_vec())
            };
            let got = run(0);
            let vmax = v.iter().fold(0f32, |a, x| a.max(x.abs())) as f64;
            let tol = 2f64.powi(-11) * vmax;
            let (e16, e) = (max_err(&got, &want16), max_err(&got, &want));
            let ef: Vec<f64> = (1..=3).map(|f| max_err(&run(f), &want16)).collect();
            eprintln!("T={t:<4} off={off:<5} nh={nh:<2} nkv={nkv} dh={dh:<3}: |gpu−fp16in| {e16:.2e}  |gpu−exact| {e:.2e}  \
                       bound {tol:.2e}  | faults {:.1e} {:.1e} {:.1e}", ef[0], ef[1], ef[2]);
            assert!(e16 <= tol, "T={t} off={off} nh={nh} nkv={nkv} dh={dh}: {e16:.3e} from the fp16-input reference > {tol:.3e}");
            assert!(e <= 4.0 * tol, "T={t} off={off}: {e:.3e} from float64 > 4x the fp16 bound {:.3e}", 4.0 * tol);
            let engages = [off + t > 32, off > 0, (off + 1..=off + t).any(|s| s % 32 != 0)];
            for i in 0..3 {
                if engages[i] { worst_fault[i] = worst_fault[i].min(ef[i] / tol); }
            }
            tested += 1;
        }
        assert!(tested >= 6, "only {tested} shapes reached the kernel");
        for (i, name) in ["no rescale", "causal offset ignored", "ragged step dropped"].iter().enumerate() {
            eprintln!("negative control '{name}': worst miss {:.0}x the bound", worst_fault[i]);
            assert!(worst_fault[i] >= 20.0, "negative control '{name}' missed by only {:.1}x — this test cannot see it", worst_fault[i]);
        }
    }

    /// Sliding window and softcap on the matrix-unit kernel against float64 (fp16-input and exact), with the
    /// same P-rounding bound as above; the window ignored / softcap ignored / window one key too wide must
    /// each miss the fp16-input reference by ≥ 20x the bound where they engage.
    /// ⚠ The caps are small (2.5-4) on purpose: at Gemma-2's 50, scores of a few units sit where cap·tanh(x/cap)
    /// ≈ x, and dropping the softcap moved the output by LESS than the fp16 bound — a control that could not fail.
    #[test]
    fn matrix_unit_window_and_softcap_match_float64() {
        let Some(ctx) = ctx() else { return };
        let shapes: &[(usize, usize, usize, usize, usize, usize, f32)] = &[
            (300, 0, 8, 2, 64, 64, 0.0),
            (100, 700, 14, 2, 64, 129, 4.0),
            (48, 16, 16, 8, 128, 17, 2.5),
            (90, 200, 8, 1, 128, 70, 3.0),
        ];
        let mut worst = [f64::INFINITY; 3];
        for (n, &(t, off, nh, nkv, dh, win, cap)) in shapes.iter().enumerate() {
            let (q, k, v) = data(t, off, nh, nkv, dh, 0x5eed + n as u64);
            let want = reference_opts(&q, &k, &v, t, off, nh, nkv, dh, win, cap as f64);
            let want16 = reference_opts(&r16(&q), &r16(&k), &r16(&v), t, off, nh, nkv, dh, win, cap as f64);
            let qt = Tensor::from_vec(&ctx, &q, &[t, nh * dh]);
            let kt = Tensor::from_vec(&ctx, &k, &[off + t, nkv * dh]);
            let vt = Tensor::from_vec(&ctx, &v, &[off + t, nkv * dh]);
            let run = |w: usize, c: f32| {
                let var = Variant { dh, ..Variant::PRODUCTION };
                pollster::block_on(attn_variant(&qt, &kt, &vt, nh, nkv, dh, off, w, c, var).expect("kernel").to_vec())
            };
            let vmax = v.iter().fold(0f32, |a, x| a.max(x.abs())) as f64;
            let tol = 2f64.powi(-11) * vmax;
            let got = run(win, cap);
            let (e16, e) = (max_err(&got, &want16), max_err(&got, &want));
            let ctl = [max_err(&run(0, cap), &want16), max_err(&run(win, 0.0), &want16), max_err(&run(win + 1, cap), &want16)];
            eprintln!("T={t:<4} off={off:<4} nh={nh:<2} nkv={nkv} dh={dh:<3} win={win:<4} cap={cap:<4}: |gpu−fp16in| {e16:.2e} \
                       |gpu−exact| {e:.2e} bound {tol:.2e} | window ignored {:.1e} softcap ignored {:.1e} window+1 {:.1e}",
                      ctl[0], ctl[1], ctl[2]);
            assert!(e16 <= tol, "T={t} off={off} win={win} cap={cap}: {e16:.3e} from the fp16-input reference > {tol:.3e}");
            assert!(e <= 4.0 * tol, "T={t} off={off} win={win} cap={cap}: {e:.3e} from float64 > {:.3e}", 4.0 * tol);
            // One extra key moves the output by ~1/win of its scale, and the fp16 bound is ~1/700 of it, so the
            // off-by-one control is only DECIDABLE for short windows (at win = 129 it measured 13x): it is
            // counted on windows ≤ 70. The f32 tiled test decides it at every window (≥ 444x).
            let engages = [win > 0 && off + t > win, cap > 0.0, win > 0 && win <= 70 && off + t > win];
            for i in 0..3 { if engages[i] { worst[i] = worst[i].min(ctl[i] / tol); } }
        }
        for (i, name) in ["window ignored", "softcap ignored", "window one key too wide (win ≤ 70)"].iter().enumerate() {
            eprintln!("negative control '{name}': worst miss {:.0}x the bound", worst[i]);
            assert!(worst[i] >= 20.0, "negative control '{name}' missed by only {:.1}x — this test cannot see it", worst[i]);
        }
    }

    /// A row's result does not depend on the block it is computed in: the last 37 of 100 queries, run as a
    /// block continuing a 63-key cache (a prefix-cache suffix, a prefill chunk), give the SAME BITS as the
    /// same rows of the whole 100-query prefill — on this kernel and on the tiled WGSL kernel. Rows past a
    /// row's causal limit are exact no-ops (rescale exp2(0) = 1, P = 0), which is what makes this hold.
    #[test]
    fn rows_are_independent_of_the_block() {
        let Some(ctx) = ctx() else { return };
        let (t, nh, nkv, dh, split) = (100usize, 14usize, 2usize, 64usize, 63usize);
        let (q, k, v) = data(t, 0, nh, nkv, dh, 11);
        let kt = Tensor::from_vec(&ctx, &k, &[t, nkv * dh]);
        let vt = Tensor::from_vec(&ctx, &v, &[t, nkv * dh]);
        let full_q = Tensor::from_vec(&ctx, &q, &[t, nh * dh]);
        let tail_q = Tensor::from_vec(&ctx, &q[split * nh * dh..], &[t - split, nh * dh]);
        for kernel in [Kernel::Native, Kernel::Tiled] {
            let full = pollster::block_on(full_q.flash_attention_prefill_with(&kt, &vt, nh, nkv, dh, 0, kernel).unwrap().to_vec());
            let tail = pollster::block_on(tail_q.flash_attention_prefill_with(&kt, &vt, nh, nkv, dh, split, kernel).unwrap().to_vec());
            assert!(tail == full[split * nh * dh..], "{kernel:?}: rows {split}..{t} differ when computed as a continuing block");
        }
    }

    /// The route is OFF unless asked for: without `FERRIC_QGEMM` the routed entry declines, so the default
    /// build's attention stays on the f32 kernels.
    #[test]
    fn route_is_opt_in() {
        let Some(ctx) = ctx() else { return };
        if std::env::var_os("FERRIC_QGEMM").is_some() { return; }
        let (t, nh, nkv, dh) = (64usize, 14usize, 2usize, 64usize);
        let (q, k, v) = data(t, 0, nh, nkv, dh, 3);
        let qt = Tensor::from_vec(&ctx, &q, &[t, nh * dh]);
        let kt = Tensor::from_vec(&ctx, &k, &[t, nkv * dh]);
        let vt = Tensor::from_vec(&ctx, &v, &[t, nkv * dh]);
        assert!(attn(&qt, &kt, &vt, nh, nkv, dh, 0, true).is_none());
        assert!(attn(&qt, &kt, &vt, nh, nkv, dh, 0, false).is_some(), "the kernel itself must be reachable, or the opt-in test is vacuous");
    }

    /// Exploration (ignored): the P-operand variants' distances to float64. Re-run when the OS or GPU
    /// changes — the f32-operand cooperative-tensor layout is implementation-defined.
    ///   cargo test -p ferric-tensor --release --lib matrix_unit_attention_probe -- --ignored --nocapture
    #[test]
    #[ignore]
    fn matrix_unit_attention_probe() {
        let Some(ctx) = ctx() else { return };
        for (n, &(t, off, nh, nkv, dh)) in SHAPES.iter().enumerate() {
            if dh % 32 != 0 { continue; }
            let (q, k, v) = data(t, off, nh, nkv, dh, 0x9e37_79b9 + n as u64);
            let want = reference(&q, &k, &v, t, off, nh, nkv, dh);
            let want16 = reference(&r16(&q), &r16(&k), &r16(&v), t, off, nh, nkv, dh);
            let qt = Tensor::from_vec(&ctx, &q, &[t, nh * dh]);
            let kt = Tensor::from_vec(&ctx, &k, &[off + t, nkv * dh]);
            let vt = Tensor::from_vec(&ctx, &v, &[off + t, nkv * dh]);
            let rows = pollster::block_on(qt.flash_attention_prefill_with(&kt, &vt, nh, nkv, dh, off, Kernel::Rows).unwrap().to_vec());
            let mut line = format!("T={t:<4} off={off:<5} nh={nh:<2} nkv={nkv} dh={dh:<3} rows {:.1e} |", max_err(&rows, &want));
            for (p_half, relaxed) in [(false, false), (true, false), (false, true), (true, true)] {
                let got = attn_variant(&qt, &kt, &vt, nh, nkv, dh, off, 0, 0.0, Variant { dh, fault: 0, p_half, relaxed }).expect("kernel");
                let got = pollster::block_on(got.to_vec());
                line += &format!(" p_half={p_half} relaxed={relaxed}: exact {:.1e} fp16in {:.1e} |", max_err(&got, &want), max_err(&got, &want16));
            }
            eprintln!("{line}");
        }
    }
}
