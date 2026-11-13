//! Seeded faults for the SmolVLA CPU kernels, scored against the derived bounds.
//!
//! The catalogue is fixed. `FAULT_SEED` only chooses which element a case
//! touches, and the same seed always chooses the same element. There is no
//! quantized GEMM on this path: SmolVLA matmul is f32 `gemm_nt`.
//!
//! arXiv:2609.00363 is not a SmolVLA paper. It is the measurement that a
//! tolerance of one spacing cannot see a one-spacing epilogue fault. The
//! same limit is why a low-mantissa flip sits under `raw_actions`.

use ferric_core::cpu::{layernorm, rmsnorm, softmax};

use crate::gemm::gemm_nt;
use crate::model::{attention, gelu_tanh, rope};
use crate::tolerance::{max_abs_bound, rope_residual};
#[cfg(test)]
use crate::tolerance::rope_invariant_holds;

pub const FAULT_SEED: u64 = 0x5a17;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    BitSign,
    BitExponent,
    BitMantissaHigh,
    BitMantissaLow,
    OffByOneRow,
    OffByOneColumn,
    Swapped,
    Dropped,
    Duplicated,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::BitSign => "bit_sign",
            Kind::BitExponent => "bit_exponent",
            Kind::BitMantissaHigh => "bit_mantissa_high",
            Kind::BitMantissaLow => "bit_mantissa_low",
            Kind::OffByOneRow => "index_off_by_one_row",
            Kind::OffByOneColumn => "index_off_by_one_column",
            Kind::Swapped => "index_swapped",
            Kind::Dropped => "accum_dropped",
            Kind::Duplicated => "accum_duplicated",
        }
    }

    fn all() -> &'static [Kind] {
        &[
            Kind::BitSign,
            Kind::BitExponent,
            Kind::BitMantissaHigh,
            Kind::BitMantissaLow,
            Kind::OffByOneRow,
            Kind::OffByOneColumn,
            Kind::Swapped,
            Kind::Dropped,
            Kind::Duplicated,
        ]
    }
}

fn mix(salt: u64) -> u64 {
    let mut x = FAULT_SEED ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x.wrapping_mul(0x94D0_49BB_1331_11EB)
}

fn pick(len: usize, salt: u64) -> usize {
    (mix(salt) as usize) % len.max(1)
}

fn series(n: usize, salt: f32) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 + ((i as f32) * 0.17 + salt).fract())
        .collect()
}

fn flip_bit(x: f32, bit: u32) -> f32 {
    f32::from_bits(x.to_bits() ^ (1u32 << bit))
}

fn apply_output(clean: &[f32], rows: usize, cols: usize, kind: Kind, salt: u64) -> Vec<f32> {
    let mut out = clean.to_vec();
    let r = pick(rows, salt);
    let c = pick(cols, salt ^ 0x11);
    let i = r * cols + c;
    match kind {
        Kind::BitSign => out[i] = flip_bit(clean[i], 31),
        Kind::BitExponent => out[i] = flip_bit(clean[i], 30),
        Kind::BitMantissaHigh => out[i] = flip_bit(clean[i], 22),
        Kind::BitMantissaLow => out[i] = flip_bit(clean[i], 0),
        Kind::OffByOneRow => {
            let r2 = (r + 1) % rows;
            out[r2 * cols + c] = clean[i];
            out[i] = 0.0;
        }
        Kind::OffByOneColumn => {
            let c2 = (c + 1) % cols;
            out[r * cols + c2] = clean[i];
            out[r * cols + c] = 0.0;
        }
        Kind::Swapped => {
            let c2 = (c + 1) % cols;
            out.swap(i, r * cols + c2);
        }
        Kind::Dropped | Kind::Duplicated => {}
    }
    out
}

fn distance(clean: &[f32], faulty: &[f32]) -> (f64, f64) {
    let mut max = 0.0f64;
    let mut sum = 0.0f64;
    for (a, b) in clean.iter().zip(faulty) {
        let d = (*a - *b).abs();
        if !d.is_finite() {
            return (f64::INFINITY, f64::INFINITY);
        }
        let d = f64::from(d);
        max = max.max(d);
        sum += d * d;
    }
    let n = clean.len().max(1) as f64;
    (max, (sum / n).sqrt())
}

fn dot_grouped(a: &[f32], b: &[f32], skip: Option<usize>, dup: Option<usize>) -> f32 {
    let mut acc = 0.0f32;
    let mut i = 0;
    while i + 4 <= a.len() {
        let mut s = 0.0f32;
        for t in 0..4 {
            let j = i + t;
            if skip == Some(j) {
                continue;
            }
            s += a[j] * b[j];
            if dup == Some(j) {
                s += a[j] * b[j];
            }
        }
        acc += s;
        i += 4;
    }
    while i < a.len() {
        if skip != Some(i) {
            acc += a[i] * b[i];
            if dup == Some(i) {
                acc += a[i] * b[i];
            }
        }
        i += 1;
    }
    acc
}

/// Row and column checksum residual of `C = A B^T`. Zero on an exact product.
pub fn gemm_abft_residual(a: &[f32], b: &[f32], c: &[f32], m: usize, k: usize, n: usize) -> f64 {
    let mut col_sum = vec![0.0f64; k];
    for j in 0..n {
        for p in 0..k {
            col_sum[p] += f64::from(b[j * k + p]);
        }
    }
    let mut max = 0.0f64;
    for i in 0..m {
        let mut expected = 0.0f64;
        for p in 0..k {
            expected += f64::from(a[i * k + p]) * col_sum[p];
        }
        let got: f64 = (0..n).map(|j| f64::from(c[i * n + j])).sum();
        max = max.max((expected - got).abs());
    }
    let mut row_sum = vec![0.0f64; k];
    for i in 0..m {
        for p in 0..k {
            row_sum[p] += f64::from(a[i * k + p]);
        }
    }
    for j in 0..n {
        let mut expected = 0.0f64;
        for p in 0..k {
            expected += row_sum[p] * f64::from(b[j * k + p]);
        }
        let got: f64 = (0..m).map(|i| f64::from(c[i * n + j])).sum();
        max = max.max((expected - got).abs());
    }
    max
}

pub fn stage_digest(xs: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in xs {
        h ^= u64::from(x.to_bits());
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

struct Case {
    stage: &'static str,
    rows: usize,
    cols: usize,
    clean: Vec<f32>,
    faulty: Vec<f32>,
    abft_detected: Option<bool>,
    invariant_norm: Option<bool>,
    invariant_angle: Option<bool>,
}

fn finish(stage: &'static str, rows: usize, cols: usize, clean: Vec<f32>, faulty: Vec<f32>, abft: Option<bool>) -> Case {
    Case {
        stage,
        rows,
        cols,
        clean,
        faulty,
        abft_detected: abft,
        invariant_norm: None,
        invariant_angle: None,
    }
}

fn matmul_case(kind: Kind) -> Case {
    let m = 4;
    let k = 8;
    let n = 6;
    let a = series(m * k, 0.1);
    let b = series(n * k, 0.3);
    let clean = gemm_nt(&a, &b, m, k, n);
    let faulty = match kind {
        Kind::Dropped | Kind::Duplicated => {
            let i = pick(m, 21);
            let j = pick(n, 22);
            let p = pick(k, 23);
            let mut y = clean.clone();
            let skip = if matches!(kind, Kind::Dropped) { Some(p) } else { None };
            let dup = if matches!(kind, Kind::Duplicated) { Some(p) } else { None };
            y[i * n + j] = dot_grouped(&a[i * k..(i + 1) * k], &b[j * k..(j + 1) * k], skip, dup);
            y
        }
        other => apply_output(&clean, m, n, other, 20),
    };
    let abft = gemm_abft_residual(&a, &b, &faulty, m, k, n) > 1e-3;
    finish("patch_token0", m, n, clean, faulty, Some(abft))
}

fn softmax_case(kind: Kind) -> Case {
    let rows = 4;
    let d = 8;
    let x = series(rows * d, 0.2);
    let clean = softmax(&x, rows, d);
    let faulty = match kind {
        Kind::Dropped | Kind::Duplicated => {
            let r = pick(rows, 31);
            let j = pick(d, 32);
            let base = r * d;
            let mx = x[base..base + d].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exps = vec![0.0f32; d];
            let mut sum = 0.0f32;
            for t in 0..d {
                let e = (x[base + t] - mx).exp();
                exps[t] = e;
                if kind == Kind::Dropped && t == j {
                    continue;
                }
                sum += e;
                if kind == Kind::Duplicated && t == j {
                    sum += e;
                }
            }
            let mut y = clean.clone();
            for t in 0..d {
                y[base + t] = exps[t] / sum;
            }
            y
        }
        other => apply_output(&clean, rows, d, other, 30),
    };
    finish("vision_l0_token0", rows, d, clean, faulty, None)
}

fn norm_case(stage: &'static str, rms: bool, kind: Kind, salt: u64) -> Case {
    let rows = 4;
    let d = 8;
    let x = series(rows * d, 0.4);
    let w = series(d, 0.6);
    let bias = series(d, 0.8);
    let clean = if rms { rmsnorm(&x, &w, rows, d, 1e-5) } else { layernorm(&x, &w, &bias, rows, d, 1e-6) };
    let faulty = match kind {
        Kind::Dropped | Kind::Duplicated => {
            let r = pick(rows, salt);
            let j = pick(d, salt ^ 3);
            let base = r * d;
            let mut y = clean.clone();
            if rms {
                let mut ms = 0.0f32;
                for t in 0..d {
                    if kind == Kind::Dropped && t == j {
                        continue;
                    }
                    ms += x[base + t] * x[base + t];
                    if kind == Kind::Duplicated && t == j {
                        ms += x[base + t] * x[base + t];
                    }
                }
                ms /= d as f32;
                let inv = 1.0 / (ms + 1e-5).sqrt();
                for t in 0..d {
                    y[base + t] = x[base + t] * inv * w[t];
                }
            } else {
                let mut sum = 0.0f32;
                let mut count_extra = 0.0f32;
                for t in 0..d {
                    if kind == Kind::Dropped && t == j {
                        continue;
                    }
                    sum += x[base + t];
                    if kind == Kind::Duplicated && t == j {
                        sum += x[base + t];
                        count_extra = 1.0;
                    }
                }
                let _ = count_extra;
                let mean = sum / d as f32;
                let mut var = 0.0f32;
                for t in 0..d {
                    if kind == Kind::Dropped && t == j {
                        continue;
                    }
                    let z = x[base + t] - mean;
                    var += z * z;
                    if kind == Kind::Duplicated && t == j {
                        var += z * z;
                    }
                }
                var /= d as f32;
                let inv = 1.0 / (var + 1e-6).sqrt();
                for t in 0..d {
                    y[base + t] = (x[base + t] - mean) * inv * w[t] + bias[t];
                }
            }
            y
        }
        other => apply_output(&clean, rows, d, other, salt),
    };
    finish(stage, rows, d, clean, faulty, None)
}

fn rope_case(kind: Kind) -> Case {
    let seq = 2;
    let heads = 1;
    let dh = 64;
    let x = series(seq * heads * dh, 0.15);
    let pos = vec![1u32, 5];
    let clean = rope(&x, &pos, heads).expect("clean rope satisfies the invariant");
    let faulty = match kind {
        Kind::Dropped | Kind::Duplicated => {
            let mut y = clean.clone();
            let t = pick(seq, 41);
            let mut c = pick(dh / 2, 42);
            let off = t * heads * dh;
            let mut sin = 0.0f32;
            let mut cos = 1.0f32;
            let mut x1 = 0.0f32;
            let mut x2 = 0.0f32;
            for step in 0..(dh / 2) {
                let cand = (c + step) % (dh / 2);
                let timescale = 10_000f32.powf((2.0 * cand as f32) / dh as f32);
                let ang = pos[t] as f32 / timescale;
                let (s, co) = ang.sin_cos();
                x1 = x[off + cand];
                x2 = x[off + cand + dh / 2];
                if (x2 * s).abs() > 1e-4 {
                    c = cand;
                    sin = s;
                    cos = co;
                    break;
                }
            }
            let cross = x2 * sin;
            let extra = if kind == Kind::Duplicated { cross } else { 0.0 };
            let missing = if kind == Kind::Dropped { 0.0 } else { cross };
            y[off + c] = x1 * cos - missing + if kind == Kind::Duplicated { -extra } else { 0.0 };
            y
        }
        other => apply_output(&clean, seq, dh, other, 40),
    };
    let residual = rope_residual(&x, &faulty, &pos, heads, dh, 10_000.0).expect("rope shape");
    let mut case = finish("text_l0_k_token0", seq, dh, clean, faulty, None);
    case.invariant_norm = Some(residual.norm_exceeded);
    case.invariant_angle = Some(residual.angle_exceeded);
    case
}

fn attention_case(kind: Kind) -> Case {
    let sq = 2;
    let sk = 4;
    let dh = 8;
    let q = series(sq * dh, 0.25);
    let k = series(sk * dh, 0.35);
    let v = series(sk * dh, 0.45);
    let clean = attention(&q, &k, &v, sq, sk, 1, 1, dh, None);
    let faulty = match kind {
        Kind::Dropped | Kind::Duplicated => {
            let i = pick(sq, 51);
            let j = pick(sk, 52);
            let scale = (dh as f32).powf(-0.5);
            let scores = gemm_nt(&q, &k, sq, dh, sk);
            let scaled: Vec<f32> = scores.iter().map(|s| s * scale).collect();
            let probs = softmax(&scaled, sq, sk);
            let mut y = clean.clone();
            let p = probs[i * sk + j];
            for t in 0..dh {
                let term = p * v[j * dh + t];
                if kind == Kind::Dropped {
                    y[i * dh + t] -= term;
                } else {
                    y[i * dh + t] += term;
                }
            }
            y
        }
        other => apply_output(&clean, sq, dh, other, 50),
    };
    finish("vision_l0_token0", sq, dh, clean, faulty, None)
}

fn gelu_case(kind: Kind) -> Case {
    let n = 16;
    let x = series(n, 0.55);
    let clean = gelu_tanh(&x);
    let faulty = match kind {
        Kind::Dropped | Kind::Duplicated => {
            let i = pick(n, 61);
            let v = x[i];
            let c = (2.0f32 / std::f32::consts::PI).sqrt();
            let cubic = 0.044715 * v * v * v;
            let inner = if kind == Kind::Dropped { v } else { v + 2.0 * cubic };
            let mut y = clean.clone();
            y[i] = 0.5 * v * (1.0 + (c * inner).tanh());
            y
        }
        other => apply_output(&clean, 4, 4, other, 60),
    };
    finish("vision_l0_token0", 4, 4, clean, faulty, None)
}

fn euler_case(kind: Kind) -> Case {
    let n = 16;
    let x = series(n, 0.7);
    let v = series(n, 0.9);
    let dt = -0.1f32;
    let clean: Vec<f32> = x.iter().zip(&v).map(|(a, b)| a + dt * b).collect();
    let faulty = match kind {
        Kind::Dropped | Kind::Duplicated => {
            let i = pick(n, 71);
            let mut y = clean.clone();
            y[i] = if kind == Kind::Dropped { x[i] } else { x[i] + 2.0 * dt * v[i] };
            y
        }
        other => apply_output(&clean, 4, 4, other, 70),
    };
    finish("raw_actions", 4, 4, clean, faulty, None)
}

fn one(kernel_case: Case) -> serde_json::Value {
    let (max_abs, rmse) = distance(&kernel_case.clean, &kernel_case.faulty);
    let e2e_bound = max_abs_bound("raw_actions").unwrap();
    let stage_bound = max_abs_bound(kernel_case.stage).unwrap();
    let e2e = max_abs > e2e_bound;
    let stage = max_abs > stage_bound;
    let digest = stage_digest(&kernel_case.clean) != stage_digest(&kernel_case.faulty);
    let invariant = match (kernel_case.invariant_norm, kernel_case.invariant_angle) {
        (None, None) => None,
        (n, a) => Some(n.unwrap_or(false) || a.unwrap_or(false)),
    };
    serde_json::json!({
        "rows": kernel_case.rows,
        "cols": kernel_case.cols,
        "max_abs": max_abs,
        "rmse": rmse,
        "end_to_end_bound": e2e_bound,
        "per_stage_bound": stage_bound,
        "per_stage_name": kernel_case.stage,
        "end_to_end_detected": e2e,
        "end_to_end_with_invariant_detected": e2e || invariant.unwrap_or(false),
        "invariant_detected": invariant,
        "invariant_norm_detected": kernel_case.invariant_norm,
        "invariant_angle_detected": kernel_case.invariant_angle,
        "per_stage_detected": stage,
        "digest_detected": digest,
        "abft_detected": kernel_case.abft_detected,
    })
}

fn run_kernel(name: &str, kind: Kind) -> Case {
    match name {
        "matmul" => matmul_case(kind),
        "softmax" => softmax_case(kind),
        "layernorm" => norm_case("vision_l0_token0", false, kind, 80),
        "rmsnorm" => norm_case("text_l0_k_token0", true, kind, 90),
        "rope" => rope_case(kind),
        "attention" => attention_case(kind),
        "elementwise" => gelu_case(kind),
        "euler" => euler_case(kind),
        _ => unreachable!(),
    }
}

fn classify(fault: &str, e2e: bool, stage: bool, invariant: bool, max_abs: f64) -> &'static str {
    if e2e || stage || invariant {
        return "detected";
    }
    if fault == "bit_mantissa_low" || (fault.starts_with("bit_") && max_abs < 1.0e-4) {
        return "numerically_benign_under_the_derived_bound";
    }
    "real_gap"
}

pub fn cpu_fault_report() -> serde_json::Value {
    let kernels = [
        "matmul",
        "softmax",
        "layernorm",
        "rmsnorm",
        "rope",
        "attention",
        "elementwise",
        "euler",
    ];
    let mut out = serde_json::Map::new();
    let mut undetected = Vec::new();
    let mut missed_by_end_to_end = Vec::new();
    for kernel in kernels {
        let mut faults = serde_json::Map::new();
        let mut e2e_hit = 0;
        let mut e2e_inv_hit = 0;
        let mut stage_hit = 0;
        let mut n = 0;
        for kind in Kind::all() {
            n += 1;
            let case = run_kernel(kernel, *kind);
            let row = one(case);
            let e2e = row["end_to_end_detected"].as_bool().unwrap();
            let e2e_inv = row["end_to_end_with_invariant_detected"].as_bool().unwrap();
            let stage = row["per_stage_detected"].as_bool().unwrap();
            let invariant = row["invariant_detected"].as_bool().unwrap_or(false);
            if e2e {
                e2e_hit += 1;
            }
            if e2e_inv {
                e2e_inv_hit += 1;
            }
            if !e2e {
                missed_by_end_to_end.push(serde_json::json!({
                    "kernel": kernel,
                    "fault": kind.name(),
                    "max_abs": row["max_abs"].as_f64().unwrap(),
                    "per_stage_detected": stage,
                }));
            }
            if stage {
                stage_hit += 1;
            }
            let max_abs = row["max_abs"].as_f64().unwrap();
            let class = classify(kind.name(), e2e, stage, invariant, max_abs);
            let mut row = row;
            row.as_object_mut().unwrap().insert("classification".into(), serde_json::json!(class));
            if class != "detected" {
                let digest = row["digest_detected"].as_bool().unwrap();
                let abft = row["abft_detected"].as_bool();
                undetected.push(serde_json::json!({
                    "device": "cpu",
                    "kernel": kernel,
                    "fault": kind.name(),
                    "max_abs": max_abs,
                    "rmse": row["rmse"].as_f64().unwrap(),
                    "classification": class,
                    "digest_detected": digest,
                    "abft_detected": abft,
                }));
            }
            faults.insert(kind.name().into(), row);
        }
        out.insert(
            kernel.into(),
            serde_json::json!({
                "device": "cpu",
                "faults": n,
                "end_to_end_detected": e2e_hit,
                "end_to_end_with_invariant_detected": e2e_inv_hit,
                "per_stage_detected": stage_hit,
                "end_to_end_rate": format!("{e2e_hit}/{n}"),
                "end_to_end_with_invariant_rate": format!("{e2e_inv_hit}/{n}"),
                "per_stage_rate": format!("{stage_hit}/{n}"),
                "cases": faults,
            }),
        );
    }
    serde_json::json!({
        "seed": format!("0x{FAULT_SEED:x}"),
        "deterministic": true,
        "qgemm": "absent. The SmolVLA CPU and wgpu paths use f32 gemm_nt, not a quantized GEMM.",
        "end_to_end_check": "max abs of the faulty kernel output against the raw_actions bound",
        "end_to_end_with_invariant_check": "the raw_actions bound, or the RoPE norm and angle invariant when this kernel is RoPE",
        "per_stage_check": "the same max abs against the mapped stage bound",
        "rope_invariant": "each pair keeps its norm within sqrt(2) * gamma(5) * (|x0|+|x1|), and the recovered angle matches pos/base^(2c/dh) within gamma(5) * (1 + |theta|) radians",
        "added_checks": ["stage_digest", "matmul row and column checksum", "rope norm and angle"],
        "kernels": out,
        "missed_by_end_to_end": missed_by_end_to_end,
        "undetected": undetected,
    })
}

#[cfg(test)]
fn readback_case(name: &str, cpu: &[f32], gpu: &[f32], stage: &str) -> serde_json::Value {
    let (max_abs, rmse) = distance(cpu, gpu);
    let mut flipped = gpu.to_vec();
    if !flipped.is_empty() {
        flipped[0] = flip_bit(flipped[0], 0);
    }
    let (flip_abs, flip_rmse) = distance(gpu, &flipped);
    let e2e_bound = max_abs_bound("raw_actions").unwrap();
    let stage_bound = max_abs_bound(stage).unwrap();
    serde_json::json!({
        "kernel": name,
        "clean_gpu_vs_cpu_max_abs": max_abs,
        "clean_gpu_vs_cpu_rmse": rmse,
        "clean_within_stage_bound": max_abs <= stage_bound,
        "low_mantissa_on_readback": {
            "max_abs": flip_abs,
            "rmse": flip_rmse,
            "end_to_end_detected": flip_abs > e2e_bound,
            "per_stage_detected": flip_abs > stage_bound,
            "digest_detected": stage_digest(gpu) != stage_digest(&flipped),
            "injection": "output bit flip after readback, not a shader-internal fault",
        },
    })
}

/// One clean dispatch per kernel that has a wgpu entry, plus a low-mantissa
/// flip of the readback. Shader-internal accumulation faults are not injected.
#[cfg(test)]
async fn device_fault_probe() -> Result<serde_json::Value, String> {
    let ctx = ferric_core::Context::new().await?;
    let m = 4usize;
    let k = 8usize;
    let n = 6usize;
    let a = series(m * k, 0.1);
    let b = series(n * k, 0.3);
    let cpu_mm = gemm_nt(&a, &b, m, k, n);
    let gpu_mm = ctx.matmul_bt(&a, &b, m as u32, n as u32, k as u32, 1.0).await?;
    let rows = 4usize;
    let d = 8usize;
    let x = series(rows * d, 0.2);
    let cpu_sm = softmax(&x, rows, d);
    let gpu_sm = ctx.softmax(&x, rows as u32, d as u32).await?;
    let w = series(d, 0.6);
    let bias = series(d, 0.8);
    let xn = series(rows * d, 0.4);
    let cpu_ln = layernorm(&xn, &w, &bias, rows, d, 1e-6);
    let gpu_ln = ctx.layernorm(&xn, &w, &bias, rows as u32, d as u32, 1e-6).await?;
    let cpu_rms = rmsnorm(&xn, &w, rows, d, 1e-5);
    let xt = ctx.tensor(&xn, &[rows * d]);
    let wt = ctx.tensor(&w, &[d]);
    let rms_t = ctx.rmsnorm_t(&xt, &wt, rows as u32, d as u32, 1e-5);
    let gpu_rms = ctx.to_vec(&rms_t).await?;
    let seq = 2usize;
    let heads = 1usize;
    let dh = 64usize;
    let rope_x = series(seq * heads * dh, 0.15);
    let rope_pos = vec![1u32, 5];
    let rope_cpu = rope(&rope_x, &rope_pos, heads)?;
    let rope_in = ctx.tensor(&rope_x, &[rope_x.len()]);
    let rope_out = ctx.rope_positions_t(&rope_in, &rope_pos, heads as u32, dh as u32, 10_000.0);
    let rope_gpu = ctx.to_vec(&rope_out).await?;
    let rope_res = rope_residual(&rope_x, &rope_gpu, &rope_pos, heads, dh, 10_000.0)?;
    let rope_holds = rope_invariant_holds(&rope_x, &rope_gpu, &rope_pos, heads, dh, 10_000.0).is_ok();
    let (rope_vs_cpu, rope_rmse) = distance(&rope_cpu, &rope_gpu);
    let software = {
        let n = ctx.adapter_name.to_lowercase();
        n.contains("llvmpipe") || n.contains("lavapipe")
    };
    Ok(serde_json::json!({
        "rope": {
            "kernel": "rope_positions",
            "clean_gpu_vs_cpu_max_abs": rope_vs_cpu,
            "clean_gpu_vs_cpu_rmse": rope_rmse,
            "max_norm_abs": rope_res.max_norm_abs,
            "max_norm_bound": rope_res.max_norm_bound,
            "max_angle_abs": rope_res.max_angle_abs,
            "max_angle_bound": rope_res.max_angle_bound,
            "pairs": rope_res.pairs,
            "invariant_holds": rope_holds,
        },
        "adapter": ctx.adapter_name,
        "backend": format!("{:?}", ctx.backend),
        "software_adapter": software,
        "shader_internal_accumulation_faults": "not injected",
        "kernels": {
            "matmul": readback_case("matmul", &cpu_mm, &gpu_mm, "patch_token0"),
            "softmax": readback_case("softmax", &cpu_sm, &gpu_sm, "vision_l0_token0"),
            "layernorm": readback_case("layernorm", &cpu_ln, &gpu_ln, "vision_l0_token0"),
            "rmsnorm": readback_case("rmsnorm", &cpu_rms, &gpu_rms, "text_l0_k_token0"),
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::prepare_uint8_v070;

    #[test]
    fn the_catalogue_is_stable_across_two_runs() {
        let a = cpu_fault_report().to_string();
        let b = cpu_fault_report().to_string();
        assert_eq!(a, b);
        if let Ok(path) = std::env::var("FERRIC_FAULT_RECEIPT_OUT") {
            let llvmpipe = std::env::var("FERRIC_FAULT_DEVICE_IN").ok().and_then(|p| {
                let text = std::fs::read_to_string(p).ok()?;
                serde_json::from_str::<serde_json::Value>(&text).ok()
            });
            let receipt = serde_json::json!({
                "seed": format!("0x{FAULT_SEED:x}"),
                "arxiv_2609_00363": "Deterministic LLM Inference Across GPU Kernels (arXiv:2609.00363) is not a SmolVLA paper and does not describe LeRobot preprocessing. Its measured result used here is that a tolerance of one spacing does not fire on a one-spacing fault.",
                "cpu": cpu_fault_report(),
                "llvmpipe": llvmpipe,
            });
            std::fs::write(path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();
        }
    }

    #[test]
    fn clean_cpu_rope_across_positions_and_scales_holds() {
        for scale in [1.0e-3f32, 0.25, 1.0, 4.0, 32.0] {
            for start in [0u32, 1, 7, 48, 200] {
                let seq = 4usize;
                let heads = 2usize;
                let dh = 64usize;
                let mut x = series(seq * heads * dh, scale);
                for v in &mut x {
                    *v *= scale;
                }
                let pos: Vec<u32> = (0..seq).map(|i| start + i as u32 * 3).collect();
                rope(&x, &pos, heads).unwrap_or_else(|e| panic!("scale {scale} start {start}: {e}"));
            }
        }
    }

    #[test]
    fn rope_cross_terms_need_the_invariant_and_the_low_bit_stays_inside_it() {
        let report = cpu_fault_report();
        let rope = &report["kernels"]["rope"];
        assert_eq!(rope["end_to_end_rate"], "6/9");
        assert_eq!(rope["end_to_end_with_invariant_rate"], "8/9");
        assert_eq!(rope["per_stage_rate"], "8/9");
        for (name, kernel) in report["kernels"].as_object().unwrap() {
            if name == "rope" {
                continue;
            }
            assert_eq!(
                kernel["end_to_end_rate"], kernel["per_stage_rate"],
                "{name} is the only kernel whose raw-action check is behind its stage check"
            );
        }
        let dropped = &rope["cases"]["accum_dropped"];
        let duplicated = &rope["cases"]["accum_duplicated"];
        assert_eq!(dropped["end_to_end_detected"], false);
        assert_eq!(duplicated["end_to_end_detected"], false);
        assert_eq!(dropped["invariant_detected"], true);
        assert_eq!(duplicated["invariant_detected"], true);
        let low = &rope["cases"]["bit_mantissa_low"];
        assert_eq!(low["end_to_end_with_invariant_detected"], false);
        assert_eq!(low["digest_detected"], true);
        assert_eq!(low["classification"], "numerically_benign_under_the_derived_bound");
    }

    #[test]
    fn a_low_mantissa_flip_is_under_the_raw_action_bound_and_the_digest_sees_it() {
        let report = cpu_fault_report();
        let row = &report["kernels"]["matmul"]["cases"]["bit_mantissa_low"];
        assert_eq!(row["end_to_end_detected"], false);
        assert_eq!(row["digest_detected"], true);
        assert_eq!(row["classification"], "numerically_benign_under_the_derived_bound");
    }

    #[test]
    fn a_dropped_matmul_term_fails_the_checksum() {
        let report = cpu_fault_report();
        let row = &report["kernels"]["matmul"]["cases"]["accum_dropped"];
        assert_eq!(row["abft_detected"], true);
        assert_eq!(row["end_to_end_detected"], true);
    }

    /// Numbers may differ by a libm ulp. Booleans, strings and the shape of the catalogue may not.
    ///
    /// `f32::tanh` in `gelu_tanh` is not bit-identical between glibc and Apple libm. On Metal CI
    /// run 43431745641 the only mismatch in the whole CPU catalogue was
    /// `elementwise.index_off_by_one_row.rmse`: 0.23910626993893822 (committed, Linux) versus
    /// 0.23910627516696203 (macOS), a gap of 5.2e-9. Every detection flag was already equal.
    /// 1e-7 is three orders of magnitude under the raw-action bound (~2e-4) that those flags use,
    /// and a 1e-4 move still fails — see [`a_real_rmse_change_is_not_a_libm_ulp`].
    fn catalogues_agree(a: &serde_json::Value, b: &serde_json::Value, tol: f64) -> Result<(), String> {
        use serde_json::Value;
        match (a, b) {
            (Value::Number(x), Value::Number(y)) => {
                let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
                if (x.is_nan() && y.is_nan()) || (x - y).abs() <= tol { Ok(()) }
                else { Err(format!("{x:.16e} vs {y:.16e}")) }
            }
            (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
                for (i, (p, q)) in x.iter().zip(y).enumerate() {
                    catalogues_agree(p, q, tol).map_err(|e| format!("[{i}] {e}"))?;
                }
                Ok(())
            }
            (Value::Object(x), Value::Object(y)) if x.len() == y.len() => {
                for (k, v) in x {
                    let Some(w) = y.get(k) else { return Err(format!("missing {k}")); };
                    catalogues_agree(v, w, tol).map_err(|e| format!("{k}: {e}"))?;
                }
                Ok(())
            }
            (a, b) if a == b => Ok(()),
            (a, b) => Err(format!("{a} vs {b}")),
        }
    }

    #[test]
    fn the_committed_receipt_matches_the_cpu_catalogue() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/fleet/smolvla-fault-injection.json");
        let file: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let report = cpu_fault_report();
        // The committed numbers match the libm that wrote the file. They are not bit-identical
        // on every libm that can still be called Linux: GitHub's ubuntu-latest (glibc on
        // resolute) and Apple libm both produce 0.23910627516696203 for
        // `elementwise.index_off_by_one_row.rmse`, against 0.23910626993893822 in the file.
        // The gap is 5.2e-9, inside 1e-7, and it flips no detection flag. A bit-exact compare
        // failed both Metal (43435730656) and this runner (43458775739). A 1e-4 move still fails.
        catalogues_agree(&file["cpu"], &report, 1e-7)
            .unwrap_or_else(|e| panic!("the committed catalogue drifted past a libm ulp: {e}"));
        assert_eq!(file["llvmpipe"]["adapter"], "llvmpipe (LLVM 20.1.2, 256 bits)");
    }

    /// The other libm's rmse (Apple, and glibc on ubuntu-latest), pinned against the committed value.
    /// If the committed number moves, this fails everywhere, and the gap has to be remeasured
    /// rather than quietly absorbed.
    #[test]
    fn the_measured_macos_rmse_stays_a_libm_ulp() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/fleet/smolvla-fault-injection.json");
        let file: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let mut macos = file["cpu"].clone();
        let slot = &mut macos["kernels"]["elementwise"]["cases"]["index_off_by_one_row"];
        let committed = slot["rmse"].as_f64().unwrap();
        assert!(
            (committed - 0.23910626993893822).abs() < 1e-18,
            "committed linux rmse moved to {committed}; remeasure the macOS gap before widening anything"
        );
        let measured = 0.23910627516696203_f64;
        slot["rmse"] = serde_json::json!(measured);
        catalogues_agree(&file["cpu"], &macos, 1e-7).expect("recorded macOS gap must stay inside 1e-7");
        assert!((committed - measured).abs() > 1e-12);
    }

    #[test]
    fn a_real_rmse_change_is_not_a_libm_ulp() {
        let report = cpu_fault_report();
        let path = ["kernels", "elementwise", "cases", "index_off_by_one_row", "rmse"];
        let mut nudge = |delta: f64| {
            let mut v = report.clone();
            let slot = path.iter().take(path.len() - 1).fold(&mut v, |a, k| &mut a[*k]);
            let key = path[path.len() - 1];
            let n = slot[key].as_f64().unwrap() + delta;
            slot[key] = serde_json::json!(n);
            v
        };
        // The measured macOS gap, 5.2e-9, stays inside the cross-libm tolerance.
        assert!(catalogues_agree(&report, &nudge(5.2e-9), 1e-7).is_ok());
        let err = catalogues_agree(&report, &nudge(1e-4), 1e-7).unwrap_err();
        assert!(err.contains("rmse"), "{err}");
    }

    #[test]
    fn an_unfaulted_grouped_dot_matches_gemm_nt() {
        let a = series(8, 0.1);
        let b = series(8, 0.3);
        let got = dot_grouped(&a, &b, None, None);
        let prod = gemm_nt(&a, &b, 1, 8, 1);
        assert_eq!(got.to_bits(), prod[0].to_bits());
    }

    #[test]
    fn device_probe_writes_when_requested() {
        if std::env::var("FERRIC_FAULT_DEVICE").is_err() {
            return;
        }
        let v = pollster::block_on(device_fault_probe()).expect("device");
        let path = std::env::var("FERRIC_FAULT_DEVICE_OUT").unwrap_or_else(|_| "/tmp/device-fault.json".into());
        std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
        assert!(v["kernels"]["matmul"]["low_mantissa_on_readback"]["digest_detected"].as_bool().unwrap());
        assert_eq!(v["rope"]["invariant_holds"], true);
    }

    #[test]
    fn uint8_v070_scales_before_resize_and_is_not_the_default_path() {
        let px = [0u8, 255, 128, 64];
        let mut nchw = Vec::new();
        for _ in 0..3 {
            nchw.extend_from_slice(&px);
        }
        let y = prepare_uint8_v070(&nchw, 2, 2).unwrap();
        let unit: Vec<f32> = nchw.iter().map(|p| f32::from(*p) / 255.0).collect();
        let expect = crate::model::prepare_image(&unit, 2, 2).unwrap();
        assert_eq!(y, expect);
        let as_unit_without_scale: Vec<f32> = nchw.iter().map(|p| f32::from(*p)).collect();
        assert!(crate::model::prepare_image(&as_unit_without_scale, 2, 2).is_err());
    }
}
