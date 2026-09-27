//! **A rewind across the NVIDIA tier's K/V hand-over** — the case speculative decoding creates.
//!
//!   FERRIC_CUDA=1 FERRIC_CUDA_NO_PREFILL=1 native_rewind_check <model.gguf>
//!
//! Native decode steps leave the newest K/V rows on the device only. A speculative step that rejects
//! drafts sets `pos` back (`Cache::truncate` on main); the device still holds the rejected rows. When a
//! WGSL forward next pulls device rows into its store, it must stop at `pos` — pulling past it hands the
//! sequence history it never kept: finite logits, plausible text, wrong.
//!
//!   reference: [t0..t15) then [t15..t20) as two WGSL multi-token forwards on a fresh cache;
//!   rewound:   [t0..t8) WGSL, then t8..t14 and three DRAFT tokens that are not t15..t17 as ten native
//!              steps (the device 10 rows ahead of WGSL's 8), `pos` set back to 15 (drafts rejected),
//!              then [t15..t20) as a WGSL multi-token forward.
//! The five last rows must agree within the native-vs-WGSL band (5e-3, scripts/cuda_conformance.sh).
use ferric_llama::qwen3::{Cache, Qwen3};
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let path = std::env::args().nth(1).expect("usage: native_rewind_check <model.gguf>");
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    let g = ferric_gguf::GgufFile::open(&path).expect("open");
    let m = Qwen3::load(&ctx, &g).expect("load");
    let t: Vec<u32> = (0..20u32).map(|i| 100 + (i * 7919) % 20000).collect();

    let mut r = Cache::new(&m.cfg);
    let _ = m.forward_cached(&t[..15], &mut r).to_vec().await;
    let want = m.forward_cached(&t[15..20], &mut r).to_vec().await;

    #[cfg(all(any(target_os = "linux", target_os = "windows"), not(target_arch = "wasm32")))]
    let s0 = ferric_tensor::cuda::native_steps();
    let mut c = Cache::new(&m.cfg);
    let _ = m.forward_cached(&t[..8], &mut c).to_vec().await;
    let drafts = [5u32, 6, 7];
    for &tok in t[8..15].iter().chain(&drafts) { let _ = m.forward_cached(&[tok], &mut c).to_vec().await; }
    let ahead = c.native_ahead();
    c.pos = 15;                                          // three drafts rejected
    let got = m.forward_cached(&t[15..20], &mut c).to_vec().await;
    #[cfg(all(any(target_os = "linux", target_os = "windows"), not(target_arch = "wasm32")))]
    let steps = ferric_tensor::cuda::native_steps() - s0;
    #[cfg(not(all(any(target_os = "linux", target_os = "windows"), not(target_arch = "wasm32"))))]
    let steps = 0u64;

    let d = want.iter().zip(&got).fold(0f32, |a, (x, y)| a.max((x - y).abs()));
    println!("native steps {steps} (want 10), device rows ahead of WGSL before the rewind {ahead} (want 10)");
    println!("rewound vs clean history, last 5 rows: max |Δ logit| {d:.3e} (band 5e-3)");
    let ok = steps == 10 && ahead == 10 && d <= 5e-3;
    println!("{}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
