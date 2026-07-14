//! Ferric web — proves the SAME pure-Rust kernels (ferric-core) run in the browser on WebGPU.
//! The matmul WGSL, the Context, the readback — all identical to native; only the target changes.
use ferric_core::{demo, matmul_cpu, max_abs_diff, Context};
use ferric_tensor::Tensor;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

/// Scheduler worker entrypoint: the native fabric sends an op frame (op · dims · A · B, same format
/// as Device::Remote) over a WebSocket; this executes it on the tab's WebGPU and returns the result
/// bytes. That makes this browser tab a device in the heterogeneous fabric (Device::BrowserWorker).
#[wasm_bindgen]
pub async fn ferric_worker_exec(input: Vec<u8>) -> Vec<u8> {
    console_error_panic_hook::set_once();
    let ru32 = |o: usize| u32::from_le_bytes([input[o], input[o + 1], input[o + 2], input[o + 3]]) as usize;
    let op = input[0];
    let dims = [ru32(1), ru32(5), ru32(9), ru32(13)];
    let mut off = 17;
    let la = ru32(off); off += 4;
    let a: Vec<f32> = input[off..off + la * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    off += la * 4;
    let lb = ru32(off); off += 4;
    let b: Vec<f32> = input[off..off + lb * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let ctx = Arc::new(Context::new().await.unwrap());
    let out = if op == 0 {
        let (batch, m, k, n) = (dims[0], dims[1], dims[2], dims[3]);
        Tensor::from_vec(&ctx, &a, &[batch, m, k]).matmul(&Tensor::from_vec(&ctx, &b, &[k, n])).to_vec().await
    } else {
        let (rows, inn, outn) = (dims[0], dims[1], dims[2]);
        Tensor::from_vec(&ctx, &a, &[rows, inn]).matmul(&Tensor::from_vec(&ctx, &b, &[inn, outn])).relu().to_vec().await
    };
    out.iter().flat_map(|f| f.to_le_bytes()).collect()
}

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

/// Runs the full Ferric transformer LM in the browser on WebGPU: greedy-generates `steps` tokens from
/// a comma-separated prompt of token ids, and validates the prefill logits against the in-wasm CPU
/// reference. Returns a JSON string {backend, prompt, generated, layers, logit_diff, ms}.
#[wasm_bindgen]
pub async fn ferric_lm_demo(prompt: String, steps: usize) -> std::result::Result<String, JsValue> {
    console_error_panic_hook::set_once();
    let ids: Vec<u32> = prompt.split(',').filter_map(|s| s.trim().parse().ok()).map(|v: u32| v % demo::VOCAB as u32).collect();
    if ids.is_empty() {
        return Err(JsValue::from_str("no valid token ids in prompt"));
    }
    let ctx = Context::new().await.map_err(|e| JsValue::from_str(&e))?;
    let t0 = js_sys::Date::now();
    let generated = demo::generate(&ctx, &ids, steps).await.map_err(|e| JsValue::from_str(&e))?;
    let ms = js_sys::Date::now() - t0;
    // correctness, in the browser: GPU prefill logits vs the CPU reference (same math, wasm CPU)
    let gpu = demo::logits(&ctx, &ids).await.map_err(|e| JsValue::from_str(&e))?;
    let diff = max_abs_diff(&gpu, &demo::logits_cpu(&ids));
    let js = |v: &[u32]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    Ok(format!(
        "{{\"backend\":\"{:?}\",\"adapter\":\"{}\",\"prompt\":[{}],\"generated\":[{}],\"layers\":{},\"vocab\":{},\"logit_diff\":{:.3e},\"ms\":{:.1}}}",
        ctx.backend, ctx.adapter_name, js(&ids), js(&generated), demo::N_LAYERS, demo::VOCAB, diff, ms
    ))
}
