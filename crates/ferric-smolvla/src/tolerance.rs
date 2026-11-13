//! Pairwise f32 forward-error bound for SmolVLA.
//!
//! Two faithful f32 evaluations that differ in accumulation order and in
//! whether the product is fused differ, on a unit-scale output element, by
//! `2 * u * sqrt(K)` for a length-K reduction. One `u * sqrt(K)` is the
//! random-walk sum of K terms of size `1/sqrt(K)`. The other `u * sqrt(K)` is
//! the extra product rounding when mul and add are not contracted. `u` is
//! `2^-23`.
//!
//! Inside a layer those contributions add in root-sum-square. Across residual
//! layers they add in root-sum-square as well: the layer Jacobian is taken to
//! be 1 because the blocks are variance-preserving. An Euler step of size
//! `1/10` contributes `0.1` times one expert forward, and the ten steps add
//! in root-sum-square. The graded quantity is a max-abs, so the element bound
//! is multiplied by `sqrt(2 ln N)` for a vector of length N.
//!
//! The constants are fixed by that argument. They are not fitted to a device.

use std::collections::BTreeMap;

pub const F32_U: f64 = 1.0 / (1u64 << 23) as f64;

pub fn matmul_diff(k: u32) -> f64 {
    2.0 * F32_U * (k as f64).sqrt()
}

pub fn reduction_diff(n: u32) -> f64 {
    2.0 * F32_U * (n as f64).sqrt()
}

fn rss(xs: &[f64]) -> f64 {
    xs.iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub fn extreme_factor(n: u32) -> f64 {
    (2.0 * (n.max(2) as f64).ln()).sqrt()
}

fn vision_layer() -> f64 {
    rss(&[
        matmul_diff(768),
        matmul_diff(768),
        matmul_diff(768),
        matmul_diff(768),
        matmul_diff(64),
        reduction_diff(1024),
        matmul_diff(1024),
        reduction_diff(768),
        matmul_diff(768),
        4.0 * F32_U,
        matmul_diff(3072),
        reduction_diff(768),
    ])
}

fn text_layer() -> f64 {
    rss(&[
        reduction_diff(960),
        matmul_diff(960),
        matmul_diff(960),
        matmul_diff(960),
        matmul_diff(960),
        matmul_diff(64),
        reduction_diff(241),
        matmul_diff(241),
        reduction_diff(960),
        matmul_diff(960),
        matmul_diff(2560),
        reduction_diff(960),
    ])
}

fn expert_layer() -> f64 {
    rss(&[
        reduction_diff(720),
        matmul_diff(720),
        matmul_diff(720),
        matmul_diff(720),
        matmul_diff(960),
        matmul_diff(64),
        reduction_diff(291),
        matmul_diff(291),
        reduction_diff(720),
        matmul_diff(720),
        matmul_diff(2048),
        reduction_diff(720),
    ])
}

/// Element-scale bound, before the max-abs factor, for each traced tensor.
pub fn element_bounds() -> BTreeMap<&'static str, (f64, u32)> {
    let patch = matmul_diff(768);
    let layer = vision_layer();
    let vision_l0 = rss(&[patch, layer]);
    let vision_post = rss(&[vec![patch, reduction_diff(768)], vec![layer; 12]].concat());
    let connector = rss(&[vision_post, matmul_diff(12288)]);
    let text_l0_k = rss(&[reduction_diff(960), matmul_diff(960)]);
    let text_stack = rss(&[text_layer(); 16]);
    let prefix = rss(&[connector, text_stack]);
    let expert = rss(&[vec![prefix], vec![expert_layer(); 16]].concat());
    let raw = rss(&[0.1 * expert; 10]);
    BTreeMap::from([
        ("prepared_image0", (8.0 * F32_U, 3 * 512 * 512)),
        ("patch_token0", (patch, 768)),
        ("vision_l0_token0", (vision_l0, 768)),
        ("vision_post_token0", (vision_post, 768)),
        ("connector_token0", (connector, 960)),
        ("state_token", (matmul_diff(32), 960)),
        ("text_l0_k_token0", (text_l0_k, 320)),
        ("velocity0", (expert, 32)),
        ("raw_actions", (raw, 1600)),
    ])
}

pub fn max_abs_bound(stage: &str) -> Option<f64> {
    element_bounds().get(stage).map(|(elem, n)| elem * extreme_factor(*n))
}

pub fn within_bound(stage: &str, measured_max_abs: f64) -> Option<bool> {
    max_abs_bound(stage).map(|b| measured_max_abs <= b)
}

/// `γ_n = n u / (1 - n u)`, the Higham bound for n rounding errors.
pub fn gamma(n: u32) -> f64 {
    let nu = f64::from(n) * F32_U;
    nu / (1.0 - nu)
}

/// One RoPE pair is two multiplies and one add, then sin and cos each
/// contribute one extra unit roundoff on those multiplies. `γ_5` is that count.
/// The Euclidean norm of the two component errors is at most `√2 γ_5 (|x0|+|x1|)`.
pub fn rope_norm_bound(x0: f64, x1: f64) -> f64 {
    std::f64::consts::SQRT_2 * gamma(5) * (x0.abs() + x1.abs())
}

/// Recovering the angle is two length-2 dots, a length-2 sum of squares, a
/// division and `atan2`: `γ_5` radians. The angle itself is `pos / base^(2c/dh)`.
/// A relative error `γ_5` in that quotient, which is what an f32 power and
/// division admit, is an absolute error `γ_5 |θ|`. The two add.
/// A pair whose squared radius is at most `u` does not determine an angle, so
/// the angle check is not applied there.
pub fn rope_angle_bound(theta: f64) -> f64 {
    gamma(5) * (1.0 + theta.abs())
}

pub fn rope_angle_is_determined(radius: f64) -> bool {
    radius * radius > F32_U
}

#[derive(Debug, Clone, Copy)]
pub struct RopeResidual {
    pub max_norm_abs: f64,
    pub max_norm_bound: f64,
    pub max_angle_abs: f64,
    pub max_angle_bound: f64,
    pub pairs: usize,
    pub angles_checked: usize,
    pub norm_exceeded: bool,
    pub angle_exceeded: bool,
}

fn wrap_pi(d: f64) -> f64 {
    let tau = 2.0 * std::f64::consts::PI;
    let mut x = d.rem_euclid(tau);
    if x > std::f64::consts::PI {
        x -= tau;
    }
    x
}

/// Norm preservation and angle of a SmolVLA RoPE.
/// `y` must be the rotation of `x` by `pos / base^(2c/dh)` on each pair.
pub fn rope_residual(
    x: &[f32],
    y: &[f32],
    pos: &[u32],
    heads: usize,
    dh: usize,
    base: f32,
) -> Result<RopeResidual, String> {
    if dh == 0 || !dh.is_multiple_of(2) {
        return Err(format!("rope head dim {dh} is not a positive even number"));
    }
    let seq = pos.len();
    let need = seq * heads * dh;
    if x.len() < need || y.len() < need || heads == 0 {
        return Err(format!(
            "rope tensors are {} and {}, shape is {seq}x{heads}x{dh}",
            x.len(),
            y.len()
        ));
    }
    let half = dh / 2;
    let mut max_norm = 0.0f64;
    let mut max_norm_bound = 0.0f64;
    let mut max_angle = 0.0f64;
    let mut max_angle_bound = 0.0f64;
    let mut angles = 0usize;
    let mut norm_exceeded = false;
    let mut angle_exceeded = false;
    for (t, &p_u) in pos.iter().enumerate() {
        let p = p_u as f32;
        for h in 0..heads {
            let off = (t * heads + h) * dh;
            for c in 0..half {
                let x0 = f64::from(x[off + c]);
                let x1 = f64::from(x[off + c + half]);
                let y0 = f64::from(y[off + c]);
                let y1 = f64::from(y[off + c + half]);
                let n_in = (x0 * x0 + x1 * x1).sqrt();
                let n_out = (y0 * y0 + y1 * y1).sqrt();
                let norm_err = (n_out - n_in).abs();
                let norm_bound = rope_norm_bound(x0, x1);
                max_norm = max_norm.max(norm_err);
                max_norm_bound = max_norm_bound.max(norm_bound);
                norm_exceeded |= norm_err > norm_bound;
                if rope_angle_is_determined(n_in) {
                    let timescale = f64::from(base.powf((2.0 * c as f32) / dh as f32));
                    let expected = f64::from(p) / timescale;
                    let n2 = x0 * x0 + x1 * x1;
                    let c_hat = (x0 * y0 + x1 * y1) / n2;
                    let s_hat = (x0 * y1 - x1 * y0) / n2;
                    let recovered = s_hat.atan2(c_hat);
                    let ang_err = wrap_pi(recovered - expected).abs();
                    let ang_bound = rope_angle_bound(expected);
                    if ang_err > max_angle {
                        max_angle = ang_err;
                        max_angle_bound = ang_bound;
                    }
                    angle_exceeded |= ang_err > ang_bound;
                    angles += 1;
                }
            }
        }
    }
    Ok(RopeResidual {
        max_norm_abs: max_norm,
        max_norm_bound,
        max_angle_abs: max_angle,
        max_angle_bound,
        pairs: seq * heads * half,
        angles_checked: angles,
        norm_exceeded,
        angle_exceeded,
    })
}

pub fn rope_invariant_holds(
    x: &[f32],
    y: &[f32],
    pos: &[u32],
    heads: usize,
    dh: usize,
    base: f32,
) -> Result<RopeResidual, String> {
    let r = rope_residual(x, y, pos, heads, dh, base)?;
    if r.norm_exceeded {
        return Err(format!(
            "rope norm drifted by {} above the derived bound {}",
            r.max_norm_abs, r.max_norm_bound
        ));
    }
    if r.angle_exceeded {
        return Err(format!(
            "rope angle drifted by {} rad above the derived bound {}",
            r.max_angle_abs, r.max_angle_bound
        ));
    }
    Ok(r)
}

pub fn bounds_json() -> serde_json::Value {
    let mut stages = serde_json::Map::new();
    for (name, (elem, n)) in element_bounds() {
        stages.insert(
            name.to_string(),
            serde_json::json!({
                "element_bound": elem,
                "vector_length": n,
                "extreme_factor": extreme_factor(n),
                "max_abs_bound": elem * extreme_factor(n),
            }),
        );
    }
    serde_json::json!({
        "unit_roundoff": F32_U,
        "per_matmul_element": "2 * u * sqrt(K)",
        "within_layer": "root-sum-square",
        "across_layers": "root-sum-square, jacobian 1",
        "euler": "ten steps of dt=0.1, root-sum-square",
        "max_abs_factor": "sqrt(2 * ln(N))",
        "stages": stages,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bound_file_matches_the_formula() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/fleet/smolvla-forward-error-bounds.json");
        let text = std::fs::read_to_string(path).expect(path);
        let file: serde_json::Value = serde_json::from_str(&text).unwrap();
        let live = bounds_json();
        for (name, stage) in live["stages"].as_object().unwrap() {
            let got = file["stages"][name]["max_abs_bound"].as_f64().unwrap();
            let want = stage["max_abs_bound"].as_f64().unwrap();
            assert!((got - want).abs() <= want.abs() * 1e-9 + 1e-18, "{name} file {got} formula {want}");
        }
    }

    #[test]
    fn committed_traces_are_graded_by_the_formula() {
        // Largest max-abs among CPU-vs-PyTorch, RADV-vs-PyTorch, and llvmpipe-vs-CPU.
        // These are the recorded numbers, not inputs to the formula.
        let measured = [
            ("prepared_image0", 2.384185791015625e-7),
            ("patch_token0", 7.152557373046875e-7),
            ("vision_l0_token0", 3.814697265625e-6),
            ("vision_post_token0", 3.4332275390625e-5),
            ("connector_token0", 9.918212890625e-5),
            ("state_token", 0.0),
            ("text_l0_k_token0", 1.33514404296875e-5),
            ("velocity0", 4.76837158203125e-6),
            ("raw_actions", 9.357929229736328e-6),
        ];
        for (stage, value) in measured {
            assert_eq!(within_bound(stage, value), Some(true), "{stage} {value} bound {:?}", max_abs_bound(stage));
        }
        // Headline figures the receipt states, graded on their own.
        assert_eq!(within_bound("raw_actions", 9.357929229736328e-6), Some(true));
        assert_eq!(within_bound("raw_actions", 3.6954879760742188e-6), Some(true));
        assert_eq!(within_bound("raw_actions", 4.291534423828125e-6), Some(true));
    }
}
