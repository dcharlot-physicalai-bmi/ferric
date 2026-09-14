//! **NVIDIA native tier** — the CUDA *driver* API over `dlopen`, no wrapper crate, no toolkit at runtime.
//!
//! This is the `metal4.rs` pattern on the other vendor: talk to the vendor's stable C ABI directly,
//! probe at runtime, and return `None` whenever anything is missing so the WGSL path stays the
//! fallback. Ferric builds only from `vendor/`, and `libloading` is already there (via `ash`), so
//! this adds **no dependency**. The kernel ships as CUDA C source (`cuda_q5k_gemv.cu`) beside a
//! prebuilt `.ptx` — exactly how `metal4_gemm.metal` sits beside `metal4_gemm.metallib` — because
//! the driver loads PTX on any machine while `nvcc`/NVRTC need the toolkit, which a user's laptop
//! rarely has.
//!
//! ⚠ **This tier is NOT bit-identical to the WGSL tier and does not claim to be.** Cross-fabric
//! bit-identity is a property of the portable graph; a dedicated native kernel has its own
//! reduction order and therefore its own recorded fingerprint — the same versioned-output rule the
//! two-level K split follows. Opt-in via `FERRIC_CUDA`, like `FERRIC_METAL4`.
//!
//! Tier 1 is deliberately simple: weights are mirrored into device memory once at load, the
//! [1, in] activation is copied in and the [1, out] result copied out per call. That is a few KB
//! each way at decode and makes the kernel verifiable in isolation against the FLAT WGSL kernel.
//! A whole-graph-resident path is tier 2.
#![cfg(all(any(target_os = "linux", target_os = "windows"), not(target_arch = "wasm32")))]

use libloading::{Library, Symbol};
use std::ffi::{c_char, c_void, CStr};
use std::sync::{Arc, OnceLock};

type CUresult = i32;
type CUdevice = i32;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUdeviceptr = u64;
type CUstream = *mut c_void;

macro_rules! sym {
    ($lib:expr, $name:literal, $ty:ty) => {{
        let s: Symbol<$ty> = unsafe { $lib.get(concat!($name, "\0").as_bytes()) }.ok()?;
        *s
    }};
}

/// The handful of driver entry points tier 1 needs, resolved once.
pub struct Driver {
    _lib: Library,
    /// Retained so the primary context outlives every buffer/module; never read after `open`.
    _ctx: CUcontext,
    cu_module_load_data: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    cu_module_get_function: unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    cu_mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    cu_mem_free: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    cu_memcpy_htod: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    cu_launch_kernel: unsafe extern "C" fn(CUfunction, u32, u32, u32, u32, u32, u32, u32, CUstream,
                                           *mut *mut c_void, *mut *mut c_void) -> CUresult,
    cu_ctx_synchronize: unsafe extern "C" fn() -> CUresult,
    cu_get_error_string: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult,
    gemv_q5k: OnceLock<Option<CUfunction>>,
}
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

impl Driver {
    fn open() -> Option<Driver> {
        let names: &[&str] = if cfg!(target_os = "windows") { &["nvcuda.dll"] } else { &["libcuda.so.1", "libcuda.so"] };
        let lib = names.iter().find_map(|n| unsafe { Library::new(n) }.ok())?;
        let cu_init = sym!(lib, "cuInit", unsafe extern "C" fn(u32) -> CUresult);
        let cu_device_get_count = sym!(lib, "cuDeviceGetCount", unsafe extern "C" fn(*mut i32) -> CUresult);
        let cu_device_get = sym!(lib, "cuDeviceGet", unsafe extern "C" fn(*mut CUdevice, i32) -> CUresult);
        let cu_primary_retain = sym!(lib, "cuDevicePrimaryCtxRetain", unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult);
        let cu_ctx_set_current = sym!(lib, "cuCtxSetCurrent", unsafe extern "C" fn(CUcontext) -> CUresult);
        let d = Driver {
            cu_module_load_data: sym!(lib, "cuModuleLoadData", unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult),
            cu_module_get_function: sym!(lib, "cuModuleGetFunction", unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult),
            cu_mem_alloc: sym!(lib, "cuMemAlloc_v2", unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult),
            cu_mem_free: sym!(lib, "cuMemFree_v2", unsafe extern "C" fn(CUdeviceptr) -> CUresult),
            cu_memcpy_htod: sym!(lib, "cuMemcpyHtoD_v2", unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult),
            cu_memcpy_dtoh: sym!(lib, "cuMemcpyDtoH_v2", unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult),
            cu_launch_kernel: sym!(lib, "cuLaunchKernel", unsafe extern "C" fn(CUfunction, u32, u32, u32, u32, u32, u32, u32, CUstream, *mut *mut c_void, *mut *mut c_void) -> CUresult),
            cu_ctx_synchronize: sym!(lib, "cuCtxSynchronize", unsafe extern "C" fn() -> CUresult),
            cu_get_error_string: sym!(lib, "cuGetErrorString", unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult),
            _ctx: std::ptr::null_mut(),
            gemv_q5k: OnceLock::new(),
            _lib: lib,
        };
        unsafe {
            if cu_init(0) != 0 { return None; }
            let mut n = 0i32;
            if cu_device_get_count(&mut n) != 0 || n <= 0 { return None; }
            let mut dev: CUdevice = 0;
            if cu_device_get(&mut dev, 0) != 0 { return None; }
            let mut ctx: CUcontext = std::ptr::null_mut();
            if cu_primary_retain(&mut ctx, dev) != 0 { return None; }
            if cu_ctx_set_current(ctx) != 0 { return None; }
            Some(Driver { _ctx: ctx, ..d })
        }
    }

    fn err(&self, what: &str, r: CUresult) -> String {
        let mut p: *const c_char = std::ptr::null();
        let msg = unsafe {
            (self.cu_get_error_string)(r, &mut p);
            if p.is_null() { String::from("?") } else { CStr::from_ptr(p).to_string_lossy().into_owned() }
        };
        format!("cuda: {what} failed: {msg} ({r})")
    }

    fn upload(&self, words: &[u32]) -> Option<CUdeviceptr> {
        let bytes = std::mem::size_of_val(words);
        let mut p: CUdeviceptr = 0;
        unsafe {
            let r = (self.cu_mem_alloc)(&mut p, bytes.max(4));
            if r != 0 { eprintln!("{}", self.err("cuMemAlloc", r)); return None; }
            let r = (self.cu_memcpy_htod)(p, words.as_ptr() as *const c_void, bytes);
            if r != 0 { eprintln!("{}", self.err("cuMemcpyHtoD", r)); (self.cu_mem_free)(p); return None; }
        }
        Some(p)
    }

    /// The Q5_K GEMV kernel, loaded once from the prebuilt PTX. `None` — loudly, once — when the
    /// artifact is absent: the source is in-repo, the PTX is built on an NVIDIA box.
    fn q5k_kernel(&self) -> Option<CUfunction> {
        *self.gemv_q5k.get_or_init(|| {
            let path = std::env::var("FERRIC_CUDA_PTX")
                .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/src/cuda_q5k_gemv.ptx").to_string());
            let Ok(mut ptx) = std::fs::read(&path) else {
                eprintln!("cuda: no PTX at {path} — build it with `nvcc -O3 -arch=compute_75 -ptx \
                           crates/ferric-tensor/src/cuda_q5k_gemv.cu -o {path}` (or set FERRIC_CUDA_PTX). \
                           Falling back to WGSL; NOTHING native ran.");
                return None;
            };
            ptx.push(0);
            let mut m: CUmodule = std::ptr::null_mut();
            let r = unsafe { (self.cu_module_load_data)(&mut m, ptx.as_ptr() as *const c_void) };
            if r != 0 { eprintln!("{}", self.err("cuModuleLoadData", r)); return None; }
            let mut f: CUfunction = std::ptr::null_mut();
            let r = unsafe { (self.cu_module_get_function)(&mut f, m, b"q5k_gemv\0".as_ptr() as *const c_char) };
            if r != 0 { eprintln!("{}", self.err("cuModuleGetFunction(q5k_gemv)", r)); return None; }
            Some(f)
        })
    }
}

/// Probe once. `None` when there is no driver, no device, or `FERRIC_CUDA` is unset.
pub fn driver() -> Option<&'static Arc<Driver>> {
    static D: OnceLock<Option<Arc<Driver>>> = OnceLock::new();
    D.get_or_init(|| {
        if std::env::var("FERRIC_CUDA").is_err() { return None; }
        Driver::open().map(Arc::new)
    }).as_ref()
}

/// A Q5_K weight mirrored into CUDA memory at load time — from the SAME host words the WGSL
/// buffers are built from, so no readback and no second repack.
pub struct Q5KDev { codes: CUdeviceptr, aux: CUdeviceptr, drv: Arc<Driver> }
impl Q5KDev {
    pub fn upload(codes: &[u32], aux: &[u32]) -> Option<Q5KDev> {
        let drv = driver()?.clone();
        let c = drv.upload(codes)?;
        let a = match drv.upload(aux) { Some(a) => a, None => { unsafe { (drv.cu_mem_free)(c); } return None; } };
        Some(Q5KDev { codes: c, aux: a, drv })
    }
    /// `out[o] = Σ_k x[k]·W[o,k]` for one activation row. `None` on any failure (WGSL runs instead).
    pub fn gemv(&self, x: &[f32], o_dim: usize, in_dim: usize) -> Option<Vec<f32>> {
        debug_assert_eq!(x.len(), in_dim);
        let f = self.drv.q5k_kernel()?;
        let d = &self.drv;
        unsafe {
            let (mut xp, mut op): (CUdeviceptr, CUdeviceptr) = (0, 0);
            if (d.cu_mem_alloc)(&mut xp, in_dim * 4) != 0 { return None; }
            if (d.cu_mem_alloc)(&mut op, o_dim * 4) != 0 { (d.cu_mem_free)(xp); return None; }
            let ok = (|| {
                if (d.cu_memcpy_htod)(xp, x.as_ptr() as *const c_void, in_dim * 4) != 0 { return None; }
                let (mut o32, mut i32_) = (o_dim as u32, in_dim as u32);
                let (mut cp, mut ap) = (self.codes, self.aux);
                let mut params: [*mut c_void; 6] = [
                    &mut xp as *mut _ as *mut c_void, &mut cp as *mut _ as *mut c_void,
                    &mut ap as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
                    &mut o32 as *mut _ as *mut c_void, &mut i32_ as *mut _ as *mut c_void,
                ];
                let grid = (o_dim as u32).div_ceil(4);
                let r = (d.cu_launch_kernel)(f, grid, 1, 1, 128, 1, 1, 0, std::ptr::null_mut(),
                                             params.as_mut_ptr(), std::ptr::null_mut());
                if r != 0 { eprintln!("{}", d.err("cuLaunchKernel(q5k_gemv)", r)); return None; }
                let r = (d.cu_ctx_synchronize)();
                if r != 0 { eprintln!("{}", d.err("cuCtxSynchronize", r)); return None; }
                let mut out = vec![0f32; o_dim];
                if (d.cu_memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, op, o_dim * 4) != 0 { return None; }
                Some(out)
            })();
            (d.cu_mem_free)(xp); (d.cu_mem_free)(op);
            ok
        }
    }
}
impl Drop for Q5KDev {
    fn drop(&mut self) { unsafe { (self.drv.cu_mem_free)(self.codes); (self.drv.cu_mem_free)(self.aux); } }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The native kernel against the FLAT WGSL kernel — which shares no reduction structure with it —
    /// on a real driver. Without one it SKIPS LOUDLY: a green test that checked nothing is the worst
    /// kind (see `mxfp4_kernel_matches_ggml_*` for the same discipline).
    #[test]
    fn q5k_gemv_matches_flat_wgsl_on_a_real_driver() {
        let Some(_) = driver() else {
            eprintln!("SKIPPED q5k_gemv_matches_flat_wgsl_on_a_real_driver: no CUDA driver / FERRIC_CUDA unset. \
                       NOTHING about the native kernel was checked.");
            return;
        };
        let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else { eprintln!("SKIPPED: no wgpu device"); return; };
        let ctx = Arc::new(ctx);
        let (inn, out) = (1024usize, 96usize);
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut lcg = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed >> 40) as u8 };
        let mut bytes = vec![0u8; out * (inn / 256) * 176];
        for b in bytes.iter_mut() { *b = lcg(); }
        for (bi, blk) in bytes.chunks_exact_mut(176).enumerate() {
            blk[0..2].copy_from_slice(&half::f16::from_f32(0.01 + 0.003 * (bi % 7) as f32).to_le_bytes());
            blk[2..4].copy_from_slice(&half::f16::from_f32(0.002 + 0.0005 * (bi % 5) as f32).to_le_bytes());
        }
        let xv: Vec<f32> = (0..inn).map(|i| ((i as f32) * 0.37).sin()).collect();
        let x = crate::Tensor::from_vec(&ctx, &xv, &[1, inn]);
        let qm = crate::dtype::QMatrix::from_bytes(&ctx, &bytes, 13, out, inn).expect("Q5_K");

        // ⛔ NOT `x.matmul_q(&qm)`: with the driver present that call is routed to CUDA by the hook
        // in `matmul_q5_k`, and the first hardware run "passed" with max |Δ| = 0.000e0 — CUDA
        // against CUDA. The reference must be the WGSL FLAT kernel, reached through a seam that
        // cannot take the native path.
        let want = pollster::block_on(qm.q5k_flat_wgsl(&x).expect("single Q5_K shard").to_vec());
        let got = qm.cuda_q5k_gemv(&xv).expect("native path should run when the driver is present");
        let scale = want.iter().fold(0f32, |a, &v| a.max(v.abs()));
        assert!(scale > 1e-3, "reference is ~zero; this would pass on anything");
        let worst = want.iter().zip(&got).fold(0f32, |a, (&w, &g)| a.max((w - g).abs()));
        eprintln!("CUDA q5k_gemv vs WGSL FLAT: max |Δ| = {worst:.3e} on {scale:.3e}");
        assert!(worst < 2e-4 * scale, "native Q5_K GEMV diverges from WGSL FLAT by {worst:.3e}");

        // The accidental identity, made explicit and labeled: through the public entry point the
        // hook MUST route to the native kernel, so that result is bit-exactly `got`. If this ever
        // fails, the tier silently stopped engaging (and the comparison above went back to
        // measuring WGSL against WGSL).
        let hooked = pollster::block_on(x.matmul_q(&qm).to_vec());
        assert!(hooked.iter().zip(&got).all(|(h, g)| h.to_bits() == g.to_bits()),
                "matmul_q did not route to the native tier: hooked path != native kernel bit-for-bit");
        eprintln!("hook check: matmul_q -> native kernel, bit-exact ({} outputs)", got.len());
    }
}
