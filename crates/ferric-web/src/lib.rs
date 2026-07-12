//! Ferric web — proves the SAME pure-Rust kernels (ferric-core) run in the browser on WebGPU.
//! The matmul WGSL, the Context, the readback — all identical to native; only the target changes.
use ferric_core::{matmul_cpu, max_abs_diff, Context};
use wasm_bindgen::prelude::*;

// Same deterministic matrices as the native example, so the browser result must match.
fn gen(m: u32, k: u32, n: u32) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..(m * k) as usize).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();
    let b: Vec<f32> = (0..(k * n) as usize).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1).collect();
    (a, b)
}

/// Runs the Ferric matmul on the browser's WebGPU and validates against the CPU reference.
/// Returns "backend|maxdiff|first6".
#[wasm_bindgen]
pub async fn ferric_matmul_demo(m: u32, k: u32, n: u32) -> std::result::Result<String, JsValue> {
    console_error_panic_hook::set_once();
    let ctx = Context::new().await.map_err(|e| JsValue::from_str(&e))?;
    let (a, b) = gen(m, k, n);
    let gpu = ctx.matmul(&a, &b, m, k, n).await.map_err(|e| JsValue::from_str(&e))?;
    let cpu = matmul_cpu(&a, &b, m as usize, k as usize, n as usize);
    let diff = max_abs_diff(&gpu, &cpu);
    Ok(format!("{:?}|{:.3e}|{:?}", ctx.backend, diff, &gpu[..6.min(gpu.len())]))
}
