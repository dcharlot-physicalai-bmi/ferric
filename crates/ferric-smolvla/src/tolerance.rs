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
    let text_stack = rss(&vec![text_layer(); 16]);
    let prefix = rss(&[connector, text_stack]);
    let expert = rss(&[vec![prefix], vec![expert_layer(); 16]].concat());
    let raw = rss(&vec![0.1 * expert; 10]);
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
