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
type CUevent = *mut c_void;

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
    cu_memcpy_dtod: unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult,
    /// ⛔ A CUDA driver context is PER-THREAD. `open` bound it on the opening thread only; every other
    /// thread — a parallel test, a ferric-serve request handler — then made driver calls with NO
    /// current context: allocations returned errors, and under test parallelism the driver
    /// segfaulted with no message. Every entry point now re-binds (idempotent, a per-thread slot).
    cu_ctx_set_current: unsafe extern "C" fn(CUcontext) -> CUresult,
    // cuEvent* for the FERRIC_CUDA_PROFILE per-kernel-class breakdown (events on the null stream).
    cu_event_create: unsafe extern "C" fn(*mut CUevent, u32) -> CUresult,
    cu_event_record: unsafe extern "C" fn(CUevent, CUstream) -> CUresult,
    cu_event_synchronize: unsafe extern "C" fn(CUevent) -> CUresult,
    cu_event_elapsed: unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult,
    /// `cuDeviceGetName` of device 0, so the native tier can be NAMED in every harness header — the
    /// wgpu adapter line says nothing about which GPU libcuda opened.
    pub name: String,
    gemv_q5k: OnceLock<Option<CUfunction>>,
    /// Tier-2 kernels from `cuda_decode.ptx`, loaded once: [rmsnorm, add_rmsnorm, vadd, q6k_gemv,
    /// q5k_swiglu_gemv, qk_norm_rope, attn_decode, q5k_gemv (coalesced; supersedes tier 1's),
    /// quant_x_q8, q5k_gemv_q8, q5k_swiglu_gemv_q8 (the FERRIC_CUDA_Q8X opt-in)].
    decode: OnceLock<Option<[CUfunction; 11]>>,
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
            cu_memcpy_dtod: sym!(lib, "cuMemcpyDtoD_v2", unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult),
            cu_ctx_set_current,
            cu_event_create: sym!(lib, "cuEventCreate", unsafe extern "C" fn(*mut CUevent, u32) -> CUresult),
            cu_event_record: sym!(lib, "cuEventRecord", unsafe extern "C" fn(CUevent, CUstream) -> CUresult),
            cu_event_synchronize: sym!(lib, "cuEventSynchronize", unsafe extern "C" fn(CUevent) -> CUresult),
            cu_event_elapsed: sym!(lib, "cuEventElapsedTime", unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult),
            _ctx: std::ptr::null_mut(),
            name: String::new(),
            gemv_q5k: OnceLock::new(),
            decode: OnceLock::new(),
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
            let get_name = sym!(d._lib, "cuDeviceGetName", unsafe extern "C" fn(*mut c_char, i32, CUdevice) -> CUresult);
            let mut buf = [0u8; 128];
            let name = if get_name(buf.as_mut_ptr() as *mut c_char, 128, dev) == 0 {
                CStr::from_ptr(buf.as_ptr() as *const c_char).to_string_lossy().into_owned()
            } else { String::from("?") };
            Some(Driver { _ctx: ctx, name, ..d })
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
        self.bind();
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
        self.bind();
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

impl Driver {
    /// Make the primary context current on THIS thread — once per thread. The driver's "current
    /// context" is thread-local state, so every entry point calls this; but calling
    /// `cuCtxSetCurrent` on EVERY entry cost a measurable ~0.3 ms/token (≈280 driver calls at ~1 µs:
    /// 9.3 -> 9.6 ms/tok on the RTX 4050), so a thread-local flag makes it one call per thread.
    fn bind(&self) {
        thread_local! { static BOUND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
        if !BOUND.with(|b| b.get()) { unsafe { (self.cu_ctx_set_current)(self._ctx); } BOUND.with(|b| b.set(true)); }
    }
    fn load_ptx(&self, file: &str, names: &[&[u8]]) -> Option<Vec<CUfunction>> {
        self.bind();
        let path = std::env::var("FERRIC_CUDA_PTX_DIR")
            .map(|d| format!("{d}/{file}"))
            .unwrap_or_else(|_| format!("{}/src/{file}", env!("CARGO_MANIFEST_DIR")));
        let Ok(mut ptx) = std::fs::read(&path) else {
            eprintln!("cuda: no PTX at {path} — build with `nvcc -O3 -arch=compute_75 -ptx` (or set \
                       FERRIC_CUDA_PTX_DIR). Native tier 2 unavailable; NOTHING native ran.");
            return None;
        };
        ptx.push(0);
        let mut m: CUmodule = std::ptr::null_mut();
        let r = unsafe { (self.cu_module_load_data)(&mut m, ptx.as_ptr() as *const c_void) };
        if r != 0 { eprintln!("{}", self.err(&format!("cuModuleLoadData({file})"), r)); return None; }
        let mut out = Vec::with_capacity(names.len());
        for n in names {
            let mut f: CUfunction = std::ptr::null_mut();
            let r = unsafe { (self.cu_module_get_function)(&mut f, m, n.as_ptr() as *const c_char) };
            if r != 0 { eprintln!("{}", self.err("cuModuleGetFunction", r)); return None; }
            out.push(f);
        }
        Some(out)
    }
    fn decode_kernels(&self) -> Option<&[CUfunction; 11]> {
        self.decode.get_or_init(|| {
            let v = self.load_ptx("cuda_decode.ptx", &[b"rmsnorm\0", b"add_rmsnorm\0", b"vadd\0", b"q6k_gemv\0",
                                                       b"q5k_swiglu_gemv\0", b"qk_norm_rope\0", b"attn_decode\0", b"q5k_gemv\0",
                                                       b"quant_x_q8\0", b"q5k_gemv_q8\0", b"q5k_swiglu_gemv_q8\0"])?;
            Some([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10]])
        }).as_ref()
    }
    /// Quantise `x[n]` (f32, device) into `xq` (4 int8 per u32) + `xs` (n/32 scales), n % 32 == 0.

    /// 1-D launch with pointer-to-argument slots; errors print here, completion is checked by `sync`.
    unsafe fn launch(&self, f: CUfunction, grid: u32, block: u32, params: &mut [*mut c_void]) -> bool {
        self.bind();
        let r = (self.cu_launch_kernel)(f, grid, 1, 1, block, 1, 1, 0, std::ptr::null_mut(),
                                        params.as_mut_ptr(), std::ptr::null_mut());
        if r != 0 { eprintln!("{}", self.err("cuLaunchKernel", r)); return false; }
        true
    }
    fn sync(&self) -> bool {
        self.bind();
        let r = unsafe { (self.cu_ctx_synchronize)() };
        if r != 0 { eprintln!("{}", self.err("cuCtxSynchronize", r)); return false; }
        true
    }
    fn alloc(&self, bytes: usize) -> Option<CUdeviceptr> {
        self.bind();
        let mut p: CUdeviceptr = 0;
        let r = unsafe { (self.cu_mem_alloc)(&mut p, bytes.max(4)) };
        if r != 0 { eprintln!("{}", self.err("cuMemAlloc", r)); return None; }
        Some(p)
    }
    fn htod(&self, dst: CUdeviceptr, src: &[f32]) -> bool {
        self.bind();
        unsafe { (self.cu_memcpy_htod)(dst, src.as_ptr() as *const c_void, src.len() * 4) == 0 }
    }
    fn dtoh(&self, dst: &mut [f32], src: CUdeviceptr) -> bool {
        self.bind();
        unsafe { (self.cu_memcpy_dtoh)(dst.as_mut_ptr() as *mut c_void, src, dst.len() * 4) == 0 }
    }
    fn upload_f32(&self, v: &[f32]) -> Option<CUdeviceptr> { let p = self.alloc(v.len() * 4)?; if self.htod(p, v) { Some(p) } else { None } }
}

/// The CUDA device's name when the native tier is active — for harness headers. `None` otherwise.
pub fn device_name() -> Option<String> { driver().map(|d| d.name.clone()) }

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
        d.bind();
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
    fn drop(&mut self) { self.drv.bind(); unsafe { (self.drv.cu_mem_free)(self.codes); (self.drv.cu_mem_free)(self.aux); } }
}

/// A Q6_K weight mirrored into CUDA memory (48 + 5 words per block), from the same host words.
pub struct Q6KDev { codes: CUdeviceptr, aux: CUdeviceptr, drv: Arc<Driver> }
impl Q6KDev {
    pub fn upload(codes: &[u32], aux: &[u32]) -> Option<Q6KDev> {
        let drv = driver()?.clone();
        let c = drv.upload(codes)?;
        let a = match drv.upload(aux) { Some(a) => a, None => { unsafe { (drv.cu_mem_free)(c); } return None; } };
        Some(Q6KDev { codes: c, aux: a, drv })
    }
}
impl Drop for Q6KDev {
    fn drop(&mut self) { self.drv.bind(); unsafe { (self.drv.cu_mem_free)(self.codes); (self.drv.cu_mem_free)(self.aux); } }
}

/// A single-shard quantised matrix as the native graph sees it.
pub enum NativeWeight<'a> {
    Q5K { dev: &'a Q5KDev, rows: usize, cols: usize },
    Q6K { dev: &'a Q6KDev, rows: usize, cols: usize },
}
impl NativeWeight<'_> {
    pub fn rows(&self) -> usize { match self { NativeWeight::Q5K { rows, .. } | NativeWeight::Q6K { rows, .. } => *rows } }
    pub fn cols(&self) -> usize { match self { NativeWeight::Q5K { cols, .. } | NativeWeight::Q6K { cols, .. } => *cols } }
    fn ptrs(&self) -> (CUdeviceptr, CUdeviceptr) {
        match self { NativeWeight::Q5K { dev, .. } => (dev.codes, dev.aux), NativeWeight::Q6K { dev, .. } => (dev.codes, dev.aux) }
    }
}

/// Everything one dense decode layer needs, as device pointers. Built once from host data.
struct LayerDev {
    attn_norm: CUdeviceptr, ffn_norm: CUdeviceptr, q_norm: Option<CUdeviceptr>, k_norm: Option<CUdeviceptr>,
    /// (codes, aux, is_q6k, rows) per fused-format part, written contiguously into the qkv buffer.
    qkv: Vec<(CUdeviceptr, CUdeviceptr, bool, usize)>,
    wo: (CUdeviceptr, CUdeviceptr, bool, usize),
    gate_up: (CUdeviceptr, CUdeviceptr),          // Q5_K only (fused kernel), 2*n_ff rows
    down: (CUdeviceptr, CUdeviceptr, bool, usize),
    k_cache: CUdeviceptr, v_cache: CUdeviceptr,   // [cap, nkv*dh] each
}

/// Host-side description of one layer for [`DecodeGraph::build`].
pub struct LayerSpec<'a> {
    pub attn_norm: Vec<f32>, pub ffn_norm: Vec<f32>,
    pub q_norm: Option<Vec<f32>>, pub k_norm: Option<Vec<f32>>,
    pub qkv_parts: Vec<NativeWeight<'a>>,
    pub wo: NativeWeight<'a>,
    /// Must be Q5_K with 2*n_ff rows (the fused SwiGLU kernel is Q5_K only in tier 2).
    pub gate_up: NativeWeight<'a>,
    pub down: NativeWeight<'a>,
}
pub struct GraphSpec<'a> {
    pub d: usize, pub nh: usize, pub nkv: usize, pub dh: usize, pub n_ff: usize, pub n_vocab: usize,
    pub eps: f32, pub rope_base: f32, pub has_qk_norm: bool, pub cap: usize,
    pub layers: Vec<LayerSpec<'a>>,
    pub out_norm: Vec<f32>,
    pub lm_head: NativeWeight<'a>,
}

/// `FERRIC_CUDA_PROFILE=1`: time each kernel CLASS of a step with cuEvents (a sync per class, so the
/// profiled step is slower; the per-class numbers are what matter). Printed every 16 steps.
struct Prof { on: bool, ev: [CUevent; 2], acc: [f64; 14], steps: u32 }
const PROF_NAMES: [&str; 14] = ["attn_norm", "qkv_gemv", "qk_norm_rope", "kv_copy", "attn", "wo_gemv",
                                "add_rmsnorm", "swiglu_gemv", "down_gemv", "resid+next_norm", "head_norm(fused)", "lm_head", "d2h", "h2d"];
impl Prof {
    fn new(drv: &Driver) -> Prof {
        let on = std::env::var("FERRIC_CUDA_PROFILE").is_ok();
        let mut ev: [CUevent; 2] = [std::ptr::null_mut(); 2];
        if on { unsafe { (drv.cu_event_create)(&mut ev[0], 0); (drv.cu_event_create)(&mut ev[1], 0); } }
        Prof { on, ev, acc: [0.0; 14], steps: 0 }
    }
    #[inline] fn start(&self, drv: &Driver) { if self.on { unsafe { (drv.cu_event_record)(self.ev[0], std::ptr::null_mut()); } } }
    #[inline] fn stop(&mut self, drv: &Driver, cls: usize) {
        if !self.on { return; }
        unsafe {
            (drv.cu_event_record)(self.ev[1], std::ptr::null_mut());
            (drv.cu_event_synchronize)(self.ev[1]);
            let mut ms = 0f32; (drv.cu_event_elapsed)(&mut ms, self.ev[0], self.ev[1]);
            self.acc[cls] += ms as f64;
        }
    }
    fn tick(&mut self) {
        if !self.on { return; }
        self.steps += 1;
        if self.steps % 16 == 0 {
            let tot: f64 = self.acc.iter().sum();
            eprintln!("cuda profile: per-token GPU time by kernel class over {} steps (total {:.2} ms/tok):", self.steps, tot / self.steps as f64);
            let mut rows: Vec<(usize, f64)> = self.acc.iter().copied().enumerate().collect();
            rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (i, v) in rows { if v > 0.0 { eprintln!("   {:<13} {:>7.3} ms  {:>5.1}%", PROF_NAMES[i], v / self.steps as f64, v / tot * 100.0); } }
        }
    }
}

/// **Tier 2: one dense decode step, fully resident.** Activations, scratch and the K/V cache all live
/// in device memory; per token the host copies ONE `[d]` embedding row in and ONE `[n_vocab]` logits
/// row out. Nothing else crosses the bus. Prefill stays on WGSL; `seed_cache` copies its K/V in once.
pub struct DecodeGraph {
    drv: Arc<Driver>,
    d: usize, nh: usize, nkv: usize, dh: usize, n_ff: usize, n_vocab: usize, eps: f32, rope_base: f32,
    has_qk_norm: bool, cap: usize,
    layers: Vec<LayerDev>,
    out_norm: CUdeviceptr, lm_head: (CUdeviceptr, CUdeviceptr, bool, usize),
    x: CUdeviceptr, xn: CUdeviceptr, qkv: CUdeviceptr, q: CUdeviceptr, k: CUdeviceptr,
    attn: CUdeviceptr, y: CUdeviceptr, xy: CUdeviceptr, h: CUdeviceptr, dn: CUdeviceptr, logits: CUdeviceptr,
    q_out: usize, kv_out: usize,
    /// Rows of K/V currently valid in the device cache (== the position of the next token).
    pub len: usize,
    prof: Prof,
    /// FERRIC_CUDA_Q8X: int8 activations + dp4a for the Q5_K GEMVs. Changes numerics; opt-in.
    q8x: bool, xq: CUdeviceptr, xs: CUdeviceptr,
}
impl DecodeGraph {
    pub fn build(spec: &GraphSpec<'_>) -> Option<DecodeGraph> {
        let drv = driver()?.clone();
        drv.decode_kernels()?; drv.q5k_kernel()?;
        let (d, nh, nkv, dh) = (spec.d, spec.nh, spec.nkv, spec.dh);
        if dh > 128 || d % 256 != 0 || spec.n_ff % 256 != 0 { return None; }
        let q_out = nh * dh; let kv_out = nkv * dh;
        let w4 = |w: &NativeWeight<'_>| { let (c, a) = w.ptrs(); (c, a, matches!(w, NativeWeight::Q6K { .. }), w.rows()) };
        let mut layers = Vec::with_capacity(spec.layers.len());
        for l in &spec.layers {
            let total: usize = l.qkv_parts.iter().map(|w| w.rows()).sum();
            if total != q_out + 2 * kv_out { return None; }
            if !matches!(l.gate_up, NativeWeight::Q5K { .. }) || l.gate_up.rows() != 2 * spec.n_ff { return None; }
            if l.wo.cols() != q_out || l.down.cols() != spec.n_ff { return None; }
            let up = |v: &Vec<f32>| drv.upload_f32(v);
            layers.push(LayerDev {
                attn_norm: up(&l.attn_norm)?, ffn_norm: up(&l.ffn_norm)?,
                q_norm: match &l.q_norm { Some(v) => Some(up(v)?), None => None },
                k_norm: match &l.k_norm { Some(v) => Some(up(v)?), None => None },
                qkv: l.qkv_parts.iter().map(|w| w4(w)).collect(),
                wo: w4(&l.wo), gate_up: l.gate_up.ptrs(), down: w4(&l.down),
                k_cache: drv.alloc(spec.cap * kv_out * 4)?, v_cache: drv.alloc(spec.cap * kv_out * 4)?,
            });
        }
        Some(DecodeGraph {
            d, nh, nkv, dh, n_ff: spec.n_ff, n_vocab: spec.n_vocab, eps: spec.eps, rope_base: spec.rope_base,
            has_qk_norm: spec.has_qk_norm, cap: spec.cap, layers,
            out_norm: drv.upload_f32(&spec.out_norm)?, lm_head: w4(&spec.lm_head),
            x: drv.alloc(d * 4)?, xn: drv.alloc(d * 4)?, qkv: drv.alloc((q_out + 2 * kv_out) * 4)?,
            q: drv.alloc(q_out * 4)?, k: drv.alloc(kv_out * 4)?, attn: drv.alloc(q_out * 4)?,
            y: drv.alloc(d * 4)?, xy: drv.alloc(d * 4)?, h: drv.alloc(spec.n_ff * 4)?, dn: drv.alloc(d * 4)?,
            logits: drv.alloc(spec.n_vocab * 4)?,
            q_out, kv_out, len: 0, prof: Prof::new(&drv),
            q8x: std::env::var("FERRIC_CUDA_Q8X").is_ok(),
            xq: drv.alloc(d.max(q_out) * 4)?, xs: drv.alloc((d.max(q_out) / 32) * 4)?,
            drv,
        })
    }
    /// Copy a layer's K and V rows (`[len, nkv*dh]` each, host f32) into the device cache at row 0.
    pub fn seed_cache(&mut self, il: usize, k_rows: &[f32], v_rows: &[f32], len: usize) -> bool {
        if len > self.cap || k_rows.len() != len * self.kv_out || v_rows.len() != len * self.kv_out { return false; }
        let l = &self.layers[il];
        let ok = self.drv.htod(l.k_cache, k_rows) && self.drv.htod(l.v_cache, v_rows);
        if il == self.layers.len() - 1 { self.len = len; }
        ok
    }
    unsafe fn quant_x(&self, k: &[CUfunction; 11], x: CUdeviceptr, xq: CUdeviceptr, xs: CUdeviceptr, n: usize) -> bool {
        DecodeGraph::quant_x_static(&self.drv, k, x, xq, xs, n)
    }
    unsafe fn gemv(&self, x: CUdeviceptr, w: (CUdeviceptr, CUdeviceptr, bool, usize), cols: usize, out: CUdeviceptr) -> bool {
        let (codes, aux, is_q6, rows) = w;
        let k = self.drv.decode_kernels().unwrap();
        if !is_q6 && self.q8x {
            if !self.quant_x(k, x, self.xq, self.xs, cols) { return false; }
            let (mut qp, mut sp, mut cp, mut ap, mut op) = (self.xq, self.xs, codes, aux, out);
            let (mut o32, mut i32_) = (rows as u32, cols as u32);
            let mut pr: [*mut c_void; 7] = [&mut qp as *mut _ as *mut c_void, &mut sp as *mut _ as *mut c_void, &mut cp as *mut _ as *mut c_void,
                &mut ap as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void, &mut o32 as *mut _ as *mut c_void, &mut i32_ as *mut _ as *mut c_void];
            return self.drv.launch(k[9], (rows as u32).div_ceil(4), 128, &mut pr);
        }
        let f = if is_q6 { k[3] } else { k[7] };     // the coalesced q5k GEMV, not tier 1's
        let (mut xp, mut cp, mut ap, mut op) = (x, codes, aux, out);
        let (mut o32, mut i32_) = (rows as u32, cols as u32);
        let mut params: [*mut c_void; 6] = [&mut xp as *mut _ as *mut c_void, &mut cp as *mut _ as *mut c_void,
            &mut ap as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut o32 as *mut _ as *mut c_void, &mut i32_ as *mut _ as *mut c_void];
        self.drv.launch(f, (rows as u32).div_ceil(4), 128, &mut params)
    }
    /// One decode step: `x_row` is the (already scaled/normed) embedding of the token at position
    /// `self.len`. Returns the logits row. `None` on any launch failure (the caller falls back).
    pub fn step(&mut self, x_row: &[f32]) -> Option<Vec<f32>> {
        if x_row.len() != self.d || self.len >= self.cap { return None; }
        let k = *self.drv.decode_kernels()?;
        let (d, pos) = (self.d, self.len);
        let drv = self.drv.clone();
        drv.bind();
        let mut prof = std::mem::replace(&mut self.prof, Prof { on: false, ev: [std::ptr::null_mut(); 2], acc: [0.0; 14], steps: 0 });
        prof.start(&drv);
        if !drv.htod(self.x, x_row) { return None; }
        prof.stop(&drv, 13);
        macro_rules! timed { ($cls:expr, $body:expr) => {{ prof.start(&drv); let ok: bool = $body; prof.stop(&drv, $cls); if !ok { self.prof = prof; return None; } }} }
        unsafe {
            macro_rules! p { ($($e:expr),*) => { [$( &mut $e as *mut _ as *mut c_void ),*] } }
            let (mut d32, mut eps) = (d as u32, self.eps);
            // Layer 0's attn_norm is a plain rmsnorm; every later layer's is fused into the previous
            // layer's residual add (one add_rmsnorm instead of vadd + rmsnorm), and the last layer's
            // residual add is fused with out_norm the same way. 56 fewer launches per token.
            let nl = self.layers.len();
            for (li, l) in self.layers.iter().enumerate() {
                if li == 0 {
                    let (mut x, mut w, mut o) = (self.x, l.attn_norm, self.xn);
                    timed!(0, drv.launch(k[0], 1, 256, &mut p!(x, w, o, d32, eps)));
                }
                timed!(1, { let mut off = 0usize; let mut ok = true;
                    for &(c, a, q6, rows) in &l.qkv { ok &= self.gemv(self.xn, (c, a, q6, rows), d, self.qkv + (off * 4) as u64); off += rows; } ok });
                let (mut qkv, mut qw, mut kw, mut qo, mut ko) = (self.qkv, l.q_norm.unwrap_or(0), l.k_norm.unwrap_or(0), self.q, self.k);
                let (mut nh, mut nkv, mut dh) = (self.nh as u32, self.nkv as u32, self.dh as u32);
                let (mut base, mut posu, mut qoff, mut koff, mut hn) = (self.rope_base, pos as u32, 0u32, self.q_out as u32, self.has_qk_norm as u32);
                let rowb = (self.kv_out * 4) as u64;
                // K and V rows go straight into the cache from the kernel: no D2D copies (class 3 is now 0).
                let (mut kc_row, mut vc_row, mut voff) = (l.k_cache + pos as u64 * rowb, l.v_cache + pos as u64 * rowb, (self.q_out + self.kv_out) as u32);
                let heads = (self.nh + self.nkv) as u32;
                timed!(2, drv.launch(k[5], heads, 128, &mut p!(qkv, qw, kw, qo, ko, nh, nkv, dh, base, posu, eps, qoff, koff, hn, kc_row, vc_row, voff)));
                let (mut q, mut kc, mut vc, mut ao, mut s, mut scale) = (self.q, l.k_cache, l.v_cache, self.attn, (pos + 1) as u32, 1.0f32 / (self.dh as f32).sqrt());
                timed!(4, drv.launch(k[6], self.nh as u32, 128, &mut p!(q, kc, vc, ao, nh, nkv, dh, s, scale)));
                timed!(5, self.gemv(self.attn, l.wo, self.q_out, self.y));
                let (mut x2, mut y2, mut fw, mut xy, mut xn) = (self.x, self.y, l.ffn_norm, self.xy, self.xn);
                timed!(6, drv.launch(k[1], 1, 256, &mut p!(x2, y2, fw, xy, xn, d32, eps)));
                let (mut xn2, mut gc, mut ga, mut h, mut nff, mut din) = (self.xn, l.gate_up.0, l.gate_up.1, self.h, self.n_ff as u32, d32);
                if self.q8x {
                    let (mut qp, mut sp) = (self.xq, self.xs);
                    timed!(7, self.quant_x(&k, self.xn, self.xq, self.xs, d)
                              && drv.launch(k[10], (self.n_ff as u32).div_ceil(4), 128, &mut p!(qp, sp, gc, ga, h, nff, din)));
                } else {
                    timed!(7, drv.launch(k[4], (self.n_ff as u32).div_ceil(4), 128, &mut p!(xn2, gc, ga, h, nff, din)));
                }
                timed!(8, self.gemv(self.h, l.down, self.n_ff, self.dn));
                // x = xy + dn, and in the same kernel xn = rmsnorm(x) * (next attn_norm | out_norm)
                let next_w = if li + 1 < nl { self.layers[li + 1].attn_norm } else { self.out_norm };
                let (mut a, mut b, mut nw, mut xo, mut xno) = (self.xy, self.dn, next_w, self.x, self.xn);
                timed!(9, drv.launch(k[1], 1, 256, &mut p!(a, b, nw, xo, xno, d32, eps)));
            }
            timed!(11, self.gemv(self.xn, self.lm_head, d, self.logits));
        }
        if !drv.sync() { self.prof = prof; return None; }
        let mut out = vec![0f32; self.n_vocab];
        prof.start(&drv);
        if !drv.dtoh(&mut out, self.logits) { self.prof = prof; return None; }
        prof.stop(&drv, 12);
        prof.tick();
        self.prof = prof;
        self.len += 1;
        Some(out)
    }
}

/// **Per-kernel microbench for the native GEMVs** — `iters` back-to-back launches on one weight, one
/// sync, returns (µs per call, the output). Used by `examples/cuda_gemv_bench.rs` to say WHICH decode
/// shape is furthest from the bandwidth floor, so kernel work starts where the bytes are.
pub fn bench_gemv(w: &NativeWeight<'_>, x: &[f32], iters: usize) -> Option<(f64, Vec<f32>)> {
    let drv = driver()?.clone();
    drv.bind();
    let (cols, rows) = (w.cols(), w.rows());
    if x.len() != cols { return None; }
    let g = DecodeGraph { drv: drv.clone(), d: 0, nh: 0, nkv: 0, dh: 0, n_ff: 0, n_vocab: 0, eps: 0.0, rope_base: 0.0,
        has_qk_norm: false, cap: 0, layers: vec![], out_norm: 0, lm_head: (0, 0, false, 0), x: 0, xn: 0, qkv: 0, q: 0, k: 0,
        attn: 0, y: 0, xy: 0, h: 0, dn: 0, logits: 0, q_out: 0, kv_out: 0, len: 0, prof: Prof { on: false, ev: [std::ptr::null_mut(); 2], acc: [0.0; 14], steps: 0 }, q8x: false, xq: 0, xs: 0 };
    let (c, a) = w.ptrs(); let is_q6 = matches!(w, NativeWeight::Q6K { .. });
    let xd = drv.upload_f32(x)?; let od = drv.alloc(rows * 4)?;
    unsafe { if !g.gemv(xd, (c, a, is_q6, rows), cols, od) { return None; } }   // warm (PTX JIT etc.)
    if !drv.sync() { return None; }
    let t0 = std::time::Instant::now();
    for _ in 0..iters { unsafe { if !g.gemv(xd, (c, a, is_q6, rows), cols, od) { return None; } } }
    if !drv.sync() { return None; }
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let mut out = vec![0f32; rows]; if !drv.dtoh(&mut out, od) { return None; }
    unsafe { (drv.cu_mem_free)(xd); (drv.cu_mem_free)(od); }
    Some((us, out))
}
/// Q5_K GEMV with int8 activations (the FERRIC_CUDA_Q8X path): quantise + dp4a GEMV per call.
pub fn bench_gemv_q8(w: &NativeWeight<'_>, x: &[f32], iters: usize) -> Option<(f64, Vec<f32>)> {
    let drv = driver()?.clone(); drv.bind();
    let NativeWeight::Q5K { rows, cols, .. } = *w else { return None };
    if x.len() != cols { return None; }
    let k = drv.decode_kernels()?; let (c, a) = w.ptrs();
    let (xd, xq, xs, od) = (drv.upload_f32(x)?, drv.alloc(cols * 4)?, drv.alloc(cols / 32 * 4)?, drv.alloc(rows * 4)?);
    let run = |d: &Driver| -> bool { unsafe {
        if !DecodeGraph::quant_x_static(d, k, xd, xq, xs, cols) { return false; }
        let (mut qp, mut sp, mut cp, mut ap, mut op, mut o32, mut i32_) = (xq, xs, c, a, od, rows as u32, cols as u32);
        let mut pr: [*mut c_void; 7] = [&mut qp as *mut _ as *mut c_void, &mut sp as *mut _ as *mut c_void, &mut cp as *mut _ as *mut c_void,
            &mut ap as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void, &mut o32 as *mut _ as *mut c_void, &mut i32_ as *mut _ as *mut c_void];
        d.launch(k[9], (rows as u32).div_ceil(4), 128, &mut pr) } };
    if !run(&drv) || !drv.sync() { return None; }
    let t0 = std::time::Instant::now();
    for _ in 0..iters { if !run(&drv) { return None; } }
    if !drv.sync() { return None; }
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let mut out = vec![0f32; rows]; if !drv.dtoh(&mut out, od) { return None; }
    unsafe { (drv.cu_mem_free)(xd); (drv.cu_mem_free)(xq); (drv.cu_mem_free)(xs); (drv.cu_mem_free)(od); }
    Some((us, out))
}
impl DecodeGraph {
    unsafe fn quant_x_static(d: &Driver, k: &[CUfunction; 11], x: CUdeviceptr, xq: CUdeviceptr, xs: CUdeviceptr, n: usize) -> bool {
        let (mut xp, mut qp, mut sp, mut n32) = (x, xq, xs, n as u32);
        let mut pr: [*mut c_void; 4] = [&mut xp as *mut _ as *mut c_void, &mut qp as *mut _ as *mut c_void, &mut sp as *mut _ as *mut c_void, &mut n32 as *mut _ as *mut c_void];
        d.launch(k[8], ((n / 32) as u32).div_ceil(4), 128, &mut pr)
    }
}
/// Fused gate|up+SwiGLU microbench, same contract; `w` must be Q5_K with `2*n_ff` rows.
pub fn bench_swiglu(w: &NativeWeight<'_>, x: &[f32], iters: usize) -> Option<(f64, Vec<f32>)> {
    let drv = driver()?.clone();
    drv.bind();
    let NativeWeight::Q5K { rows, cols, .. } = *w else { return None };
    if x.len() != cols || rows % 2 != 0 { return None; }
    let n_ff = rows / 2; let k = drv.decode_kernels()?;
    let (c, a) = w.ptrs();
    let xd = drv.upload_f32(x)?; let od = drv.alloc(n_ff * 4)?;
    let run = |drv: &Driver| -> bool { unsafe {
        let (mut xp, mut cp, mut ap, mut op, mut nff, mut din) = (xd, c, a, od, n_ff as u32, cols as u32);
        let mut pr: [*mut c_void; 6] = [&mut xp as *mut _ as *mut c_void, &mut cp as *mut _ as *mut c_void, &mut ap as *mut _ as *mut c_void,
                                        &mut op as *mut _ as *mut c_void, &mut nff as *mut _ as *mut c_void, &mut din as *mut _ as *mut c_void];
        drv.launch(k[4], (n_ff as u32).div_ceil(4), 128, &mut pr) } };
    if !run(&drv) || !drv.sync() { return None; }
    let t0 = std::time::Instant::now();
    for _ in 0..iters { if !run(&drv) { return None; } }
    if !drv.sync() { return None; }
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let mut out = vec![0f32; n_ff]; if !drv.dtoh(&mut out, od) { return None; }
    unsafe { (drv.cu_mem_free)(xd); (drv.cu_mem_free)(od); }
    Some((us, out))
}

/// Fused gate|up+SwiGLU with int8 activations (FERRIC_CUDA_Q8X). Same contract as `bench_swiglu`,
/// so the two are directly comparable — the swiglu shape is the LARGEST per-layer Q5_K weight read,
/// and leaving it out of the table let the q8x rows cover under half of that traffic.
pub fn bench_swiglu_q8(w: &NativeWeight<'_>, x: &[f32], iters: usize) -> Option<(f64, Vec<f32>)> {
    let drv = driver()?.clone(); drv.bind();
    let NativeWeight::Q5K { rows, cols, .. } = *w else { return None };
    if x.len() != cols || rows % 2 != 0 { return None; }
    let n_ff = rows / 2; let k = drv.decode_kernels()?; let (c, a) = w.ptrs();
    let (xd, xq, xs, od) = (drv.upload_f32(x)?, drv.alloc(cols * 4)?, drv.alloc(cols / 32 * 4)?, drv.alloc(n_ff * 4)?);
    let run = |d: &Driver| -> bool { unsafe {
        if !DecodeGraph::quant_x_static(d, k, xd, xq, xs, cols) { return false; }
        let (mut qp, mut sp, mut cp, mut ap, mut op, mut nff, mut din) = (xq, xs, c, a, od, n_ff as u32, cols as u32);
        let mut pr: [*mut c_void; 7] = [&mut qp as *mut _ as *mut c_void, &mut sp as *mut _ as *mut c_void, &mut cp as *mut _ as *mut c_void,
            &mut ap as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void, &mut nff as *mut _ as *mut c_void, &mut din as *mut _ as *mut c_void];
        d.launch(k[10], (n_ff as u32).div_ceil(4), 128, &mut pr) } };
    if !run(&drv) || !drv.sync() { return None; }
    let t0 = std::time::Instant::now();
    for _ in 0..iters { if !run(&drv) { return None; } }
    if !drv.sync() { return None; }
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let mut out = vec![0f32; n_ff]; if !drv.dtoh(&mut out, od) { return None; }
    unsafe { (drv.cu_mem_free)(xd); (drv.cu_mem_free)(xq); (drv.cu_mem_free)(xs); (drv.cu_mem_free)(od); }
    Some((us, out))
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

    fn ctx_or_skip(name: &str) -> Option<Arc<ferric_core::Context>> {
        if driver().is_none() { eprintln!("SKIPPED {name}: no CUDA driver / FERRIC_CUDA unset. NOTHING native was checked."); return None; }
        pollster::block_on(ferric_core::Context::new()).ok().map(Arc::new)
    }
    fn q_fixture(ty: u32, bpb: usize, rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        let mut seed = seed; let mut bytes = vec![0u8; rows * (cols / 256) * bpb];
        for b in bytes.iter_mut() { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; *b = (seed >> 40) as u8; }
        for (bi, blk) in bytes.chunks_exact_mut(bpb).enumerate() {
            let d = half::f16::from_f32(0.01 + 0.003 * (bi % 7) as f32);
            if ty == 14 { blk[208..210].copy_from_slice(&d.to_le_bytes()); }
            else { blk[0..2].copy_from_slice(&d.to_le_bytes());
                   blk[2..4].copy_from_slice(&half::f16::from_f32(0.002 + 0.0005 * (bi % 5) as f32).to_le_bytes()); }
        }
        bytes
    }
    fn close(name: &str, want: &[f32], got: &[f32], tol: f32) {
        assert_eq!(want.len(), got.len(), "{name}: length");
        let scale = want.iter().fold(0f32, |a, &v| a.max(v.abs()));
        assert!(scale > 1e-3, "{name}: reference is ~zero; would pass on anything");
        assert!(got.iter().all(|v| v.is_finite()), "{name}: non-finite");
        let worst = want.iter().zip(got).fold(0f32, |a, (&w, &g)| a.max((w - g).abs()));
        eprintln!("{name}: max |Δ| = {worst:.3e} on {scale:.3e}");
        assert!(worst < tol * scale, "{name}: native diverges from WGSL by {worst:.3e}");
        if worst == 0.0 {
            // The #68 tell, kept visible but not made a verdict: two reduction orders CAN coincide on a
            // small row (rmsnorm over 256 values did, on the 4050). The rigorous answer is a third,
            // independent reference — see `cpu_rmsnorm` below — not a rule about zeros.
            eprintln!("⚠ {name}: Δ is EXACTLY zero — fine if an independent reference also agrees; suspect the reference path otherwise (#68)");
        }
    }
    fn cpu_rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let ms = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
        let inv = 1.0 / (ms + eps as f64).sqrt();
        x.iter().zip(w).map(|(&v, &ww)| ((v as f64) * inv * (ww as f64)) as f32).collect()
    }
    fn bare_graph(drv: &Arc<Driver>) -> DecodeGraph {
        DecodeGraph { drv: drv.clone(), d: 0, nh: 0, nkv: 0, dh: 0, n_ff: 0, n_vocab: 0, eps: 0.0, rope_base: 0.0,
            has_qk_norm: false, cap: 0, layers: vec![], out_norm: 0, lm_head: (0, 0, false, 0), x: 0, xn: 0, qkv: 0, q: 0, k: 0,
            attn: 0, y: 0, xy: 0, h: 0, dn: 0, logits: 0, q_out: 0, kv_out: 0, len: 0, prof: Prof { on: false, ev: [std::ptr::null_mut(); 2], acc: [0.0; 14], steps: 0 }, q8x: false, xq: 0, xs: 0 }
    }

    /// Q6_K native GEMV vs the WGSL FLAT kernel through the hermetic seam (never the hooked entry).
    #[test]
    fn q6k_gemv_matches_flat_wgsl() {
        let Some(ctx) = ctx_or_skip("q6k_gemv_matches_flat_wgsl") else { return };
        let (inn, out) = (1024usize, 96usize);
        let bytes = q_fixture(14, 210, out, inn, 0xC0FFEE);
        let xv: Vec<f32> = (0..inn).map(|i| ((i as f32) * 0.29).cos()).collect();
        let x = crate::Tensor::from_vec(&ctx, &xv, &[1, inn]);
        let qm = crate::dtype::QMatrix::from_bytes(&ctx, &bytes, 14, out, inn).expect("Q6_K");
        let want = pollster::block_on(qm.q6k_flat_wgsl(&x).expect("single shard").to_vec());
        let w = qm.native_weight().expect("mirror"); let (c, a) = w.ptrs();
        let drv = driver().unwrap();
        let (xd, od) = (drv.upload_f32(&xv).unwrap(), drv.alloc(out * 4).unwrap());
        let g = bare_graph(drv);
        assert!(unsafe { g.gemv(xd, (c, a, true, out), inn, od) } && drv.sync());
        let mut got = vec![0f32; out]; assert!(drv.dtoh(&mut got, od));
        close("Q6_K gemv vs WGSL FLAT", &want, &got, 2e-4);
    }

    /// The int8-activation Q5_K GEMV vs the f32 WGSL FLAT kernel. ⚠ This one is EXPECTED to differ
    /// beyond the 2e-4 the other tests use: the activation is quantised to int8 per 32 values. The
    /// tolerance here (1.5e-2 relative) is the accuracy trade being measured, and the test prints the
    /// actual number so the policy call can be made on it rather than on the tolerance.
    #[test]
    fn q5k_q8_gemv_vs_flat_wgsl_reports_the_accuracy_trade() {
        let Some(ctx) = ctx_or_skip("q5k_q8_gemv_vs_flat_wgsl_reports_the_accuracy_trade") else { return };
        let (inn, out) = (1024usize, 96usize);
        let bytes = q_fixture(13, 176, out, inn, 0xA11CE);
        let xv: Vec<f32> = (0..inn).map(|i| ((i as f32) * 0.37).sin()).collect();
        let x = crate::Tensor::from_vec(&ctx, &xv, &[1, inn]);
        let qm = crate::dtype::QMatrix::from_bytes(&ctx, &bytes, 13, out, inn).expect("Q5_K");
        let want = pollster::block_on(qm.q5k_flat_wgsl(&x).expect("single shard").to_vec());
        let w = qm.native_weight().expect("mirror");
        let (_, got) = bench_gemv_q8(&w, &xv, 1).expect("q8 path");
        let scale = want.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let worst = want.iter().zip(&got).fold(0f32, |a, (&w, &g)| a.max((w - g).abs()));
        eprintln!("Q5_K int8-activation GEMV vs f32 WGSL FLAT: max |Δ| = {worst:.3e} on {scale:.3e}  (rel {:.2e})", worst / scale);
        assert!(got.iter().all(|v| v.is_finite()));
        assert!(worst > 0.0, "int8 activations cannot match f32 to the bit; an exact match means the f32 path ran");
        assert!(worst < 1.5e-2 * scale, "q8 GEMV diverges more than the int8 budget: {worst:.3e}");
    }

    /// Fused gate|up + SwiGLU vs the WGSL composed path (FLAT matmul via the seam, then swiglu).
    #[test]
    fn q5k_swiglu_matches_wgsl_composed() {
        let Some(ctx) = ctx_or_skip("q5k_swiglu_matches_wgsl_composed") else { return };
        let (inn, n_ff) = (1024usize, 64usize);
        let bytes = q_fixture(13, 176, 2 * n_ff, inn, 0xBEEF);
        let xv: Vec<f32> = (0..inn).map(|i| ((i as f32) * 0.41).sin()).collect();
        let x = crate::Tensor::from_vec(&ctx, &xv, &[1, inn]);
        let qm = crate::dtype::QMatrix::from_bytes(&ctx, &bytes, 13, 2 * n_ff, inn).expect("Q5_K");
        let want = pollster::block_on(qm.q5k_flat_wgsl(&x).expect("single shard").swiglu(n_ff).to_vec());
        let w = qm.native_weight().expect("mirror"); let (c, a) = w.ptrs();
        let drv = driver().unwrap(); let k = drv.decode_kernels().expect("decode ptx");
        let (mut xd, mut od) = (drv.upload_f32(&xv).unwrap(), drv.alloc(n_ff * 4).unwrap());
        let (mut cc, mut aa, mut nff, mut din) = (c, a, n_ff as u32, inn as u32);
        let mut pr: [*mut c_void; 6] = [&mut xd as *mut _ as *mut c_void, &mut cc as *mut _ as *mut c_void, &mut aa as *mut _ as *mut c_void,
                                        &mut od as *mut _ as *mut c_void, &mut nff as *mut _ as *mut c_void, &mut din as *mut _ as *mut c_void];
        assert!(unsafe { drv.launch(k[4], (n_ff as u32).div_ceil(4), 128, &mut pr) } && drv.sync());
        let mut got = vec![0f32; n_ff]; assert!(drv.dtoh(&mut got, od));
        close("Q5_K swiglu vs WGSL composed", &want, &got, 2e-4);
    }

    /// rmsnorm, qk_norm_rope and attn_decode against their WGSL twins.
    #[test]
    fn norm_rope_attention_match_wgsl() {
        let Some(ctx) = ctx_or_skip("norm_rope_attention_match_wgsl") else { return };
        let drv = driver().unwrap(); let k = drv.decode_kernels().expect("decode ptx");
        let (nh, nkv, dh, s_len, eps, base, d) = (4usize, 2usize, 64usize, 37usize, 1e-6f32, 10000.0f32, 256usize);
        let rnd = |n: usize, seed: u64| -> Vec<f32> { (0..n).map(|i| (((i as u64 * 2654435761 + seed) % 1000) as f32 / 500.0 - 1.0)).collect() };
        let up = |v: &[f32]| drv.upload_f32(v).unwrap();
        // rmsnorm
        let (xv, wv) = (rnd(d, 1), rnd(d, 2).iter().map(|v| 1.0 + v * 0.1).collect::<Vec<_>>());
        let want = pollster::block_on(crate::Tensor::from_vec(&ctx, &xv, &[1, d]).rmsnorm(&crate::Tensor::from_vec(&ctx, &wv, &[d]), eps).to_vec());
        let (mut xd, mut wd, mut od, mut d32, mut e) = (up(&xv), up(&wv), drv.alloc(d * 4).unwrap(), d as u32, eps);
        let mut pr: [*mut c_void; 5] = [&mut xd as *mut _ as *mut c_void, &mut wd as *mut _ as *mut c_void, &mut od as *mut _ as *mut c_void, &mut d32 as *mut _ as *mut c_void, &mut e as *mut _ as *mut c_void];
        assert!(unsafe { drv.launch(k[0], 1, 256, &mut pr) } && drv.sync());
        let mut got = vec![0f32; d]; assert!(drv.dtoh(&mut got, od)); close("rmsnorm", &want, &got, 1e-4);
        // Independent f64 reference: agreement here means an exact-zero vs WGSL is coincidence, not circularity.
        close("rmsnorm vs CPU f64 (independent)", &cpu_rmsnorm(&xv, &wv, eps), &got, 1e-5);
        // qk_norm_rope, rows == 1
        let (q_out, kv_out) = (nh * dh, nkv * dh); let width = q_out + 2 * kv_out;
        let qkv = rnd(width, 3);
        let (qw, kw): (Vec<f32>, Vec<f32>) = (rnd(dh, 4).iter().map(|v| 1.0 + v * 0.1).collect(), rnd(dh, 5).iter().map(|v| 1.0 + v * 0.1).collect());
        let pos = 11usize;
        let src = crate::Tensor::from_vec(&ctx, &qkv, &[1, width]);
        let (wq, wk) = (crate::Tensor::from_vec(&ctx, &qw, &[dh]), crate::Tensor::from_vec(&ctx, &kw, &[dh]));
        let (q_ref, k_ref) = crate::Tensor::qk_norm_rope(&src, 0, q_out, &wq, &wk, 1, nh, nkv, dh, base, pos, eps);
        let (q_ref, k_ref) = (pollster::block_on(q_ref.to_vec()), pollster::block_on(k_ref.to_vec()));
        let (mut sd, mut qwd, mut kwd, mut qo, mut ko) = (up(&qkv), up(&qw), up(&kw), drv.alloc(q_out * 4).unwrap(), drv.alloc(kv_out * 4).unwrap());
        let (mut nh32, mut nkv32, mut dh32, mut b, mut p32, mut e2, mut qoff, mut koff, mut hn) = (nh as u32, nkv as u32, dh as u32, base, pos as u32, eps, 0u32, q_out as u32, 1u32);
        let (mut kc0, mut vc0, mut voff0): (CUdeviceptr, CUdeviceptr, u32) = (0, 0, 0);   // no cache write in the unit test
        let mut pr2: [*mut c_void; 17] = [&mut sd as *mut _ as *mut c_void, &mut qwd as *mut _ as *mut c_void, &mut kwd as *mut _ as *mut c_void, &mut qo as *mut _ as *mut c_void, &mut ko as *mut _ as *mut c_void,
            &mut nh32 as *mut _ as *mut c_void, &mut nkv32 as *mut _ as *mut c_void, &mut dh32 as *mut _ as *mut c_void, &mut b as *mut _ as *mut c_void, &mut p32 as *mut _ as *mut c_void,
            &mut e2 as *mut _ as *mut c_void, &mut qoff as *mut _ as *mut c_void, &mut koff as *mut _ as *mut c_void, &mut hn as *mut _ as *mut c_void,
            &mut kc0 as *mut _ as *mut c_void, &mut vc0 as *mut _ as *mut c_void, &mut voff0 as *mut _ as *mut c_void];
        assert!(unsafe { drv.launch(k[5], (nh + nkv) as u32, 128, &mut pr2) } && drv.sync());
        let (mut qg, mut kg) = (vec![0f32; q_out], vec![0f32; kv_out]); assert!(drv.dtoh(&mut qg, qo) && drv.dtoh(&mut kg, ko));
        close("qk_norm_rope q", &q_ref, &qg, 1e-4); close("qk_norm_rope k", &k_ref, &kg, 1e-4);
        // attention vs fused_decode_attention
        let (qv, kv, vv) = (rnd(q_out, 6), rnd(s_len * kv_out, 7), rnd(s_len * kv_out, 8));
        let want = pollster::block_on(crate::Tensor::from_vec(&ctx, &qv, &[1, q_out]).fused_decode_attention(
            &crate::Tensor::from_vec(&ctx, &kv, &[s_len, kv_out]), &crate::Tensor::from_vec(&ctx, &vv, &[s_len, kv_out]), nh, nkv, dh).to_vec());
        let (mut qd, mut kd, mut vd, mut ad) = (up(&qv), up(&kv), up(&vv), drv.alloc(q_out * 4).unwrap());
        let (mut s32, mut sc) = (s_len as u32, 1.0f32 / (dh as f32).sqrt());
        let mut pr3: [*mut c_void; 9] = [&mut qd as *mut _ as *mut c_void, &mut kd as *mut _ as *mut c_void, &mut vd as *mut _ as *mut c_void, &mut ad as *mut _ as *mut c_void,
            &mut nh32 as *mut _ as *mut c_void, &mut nkv32 as *mut _ as *mut c_void, &mut dh32 as *mut _ as *mut c_void, &mut s32 as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void];
        assert!(unsafe { drv.launch(k[6], nh as u32, 128, &mut pr3) } && drv.sync());
        let mut got = vec![0f32; q_out]; assert!(drv.dtoh(&mut got, ad)); close("attn_decode", &want, &got, 2e-4);
        // The real model's head geometry (nh=16, nkv=8, dh=128 -> 4 elements per lane) and a key count
        // that leaves a remainder across the 4 warps (133 = 4*33 + 1), so the warp-strided paths are hit.
        let (nh2, nkv2, dh2, s2) = (16usize, 8usize, 128usize, 133usize);
        let (qo2, kvo2) = (nh2 * dh2, nkv2 * dh2);
        let (qv2, kv2, vv2) = (rnd(qo2, 16), rnd(s2 * kvo2, 17), rnd(s2 * kvo2, 18));
        let want2 = pollster::block_on(crate::Tensor::from_vec(&ctx, &qv2, &[1, qo2]).fused_decode_attention(
            &crate::Tensor::from_vec(&ctx, &kv2, &[s2, kvo2]), &crate::Tensor::from_vec(&ctx, &vv2, &[s2, kvo2]), nh2, nkv2, dh2).to_vec());
        let (mut qd2, mut kd2, mut vd2, mut ad2) = (up(&qv2), up(&kv2), up(&vv2), drv.alloc(qo2 * 4).unwrap());
        let (mut nh32b, mut nkv32b, mut dh32b, mut s32b, mut scb) = (nh2 as u32, nkv2 as u32, dh2 as u32, s2 as u32, 1.0f32 / (dh2 as f32).sqrt());
        let mut pr4: [*mut c_void; 9] = [&mut qd2 as *mut _ as *mut c_void, &mut kd2 as *mut _ as *mut c_void, &mut vd2 as *mut _ as *mut c_void, &mut ad2 as *mut _ as *mut c_void,
            &mut nh32b as *mut _ as *mut c_void, &mut nkv32b as *mut _ as *mut c_void, &mut dh32b as *mut _ as *mut c_void, &mut s32b as *mut _ as *mut c_void, &mut scb as *mut _ as *mut c_void];
        assert!(unsafe { drv.launch(k[6], nh2 as u32, 128, &mut pr4) } && drv.sync());
        let mut got2 = vec![0f32; qo2]; assert!(drv.dtoh(&mut got2, ad2)); close("attn_decode dh=128 S=133", &want2, &got2, 2e-4);
    }
}
