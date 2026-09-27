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
    /// Tier-2 kernels from `cuda_decode.ptx`, loaded once BY NAME (see [`DecodeK`]).
    decode: OnceLock<Option<DecodeK>>,
    /// Tier-3 (prefill) kernels from `cuda_prefill.ptx` (see [`PrefillK`]).
    prefill: OnceLock<Option<PrefillK>>,
}

/// The prefill kernel table, resolved by name from `cuda_prefill.ptx`.
#[derive(Clone, Copy)]
pub(crate) struct PrefillK {
    /// Indexed by `QFmt as usize`: the tensor-core GEMM for each weight format.
    gemm: [CUfunction; 5],
    swiglu_rows: CUfunction, attn_prefill: CUfunction,
}
unsafe impl Send for PrefillK {}
unsafe impl Sync for PrefillK {}

/// The tier-2 kernel table, resolved by name from `cuda_decode.ptx`.
///
/// ⛔ Was a positional `[CUfunction; 11]` indexed as `k[3]`, `k[7]`, … — adding the three Q4_K_M
/// formats would have made it 18 bare indices where one off-by-one launches a Q8_0 kernel on Q4_K
/// words: finite output, wrong model, no error. Named fields make that a compile error instead.
#[derive(Clone, Copy)]
pub(crate) struct DecodeK {
    rmsnorm: CUfunction, add_rmsnorm: CUfunction, qk_norm_rope: CUfunction, attn_decode: CUfunction,
    quant_x_q8: CUfunction, q5k_gemv_q8: CUfunction, q5k_swiglu_gemv_q8: CUfunction,
    /// Indexed by `QFmt as usize`: the GEMV and the fused gate|up + SwiGLU for each weight format.
    gemv: [CUfunction; 5], swiglu: [CUfunction; 5],
}
unsafe impl Send for DecodeK {}
unsafe impl Sync for DecodeK {}

/// The packed weight formats the native tier reads — the ones a real `Q4_K_M` / `Q5_K_M` / `Q8_0`
/// GGUF is made of. ⚠ The discriminant order is the `F` index of `dot_lane<F>` in `cuda_decode.cu`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QFmt { Q4K = 0, Q5K = 1, Q6K = 2, Q8_0 = 3, Q5_0 = 4 }
impl QFmt {
    /// Values per quantisation block: the input width must be a multiple of it.
    pub fn block(self) -> usize { match self { QFmt::Q4K | QFmt::Q5K | QFmt::Q6K => 256, QFmt::Q8_0 | QFmt::Q5_0 => 32 } }
    /// The ggml type id, for messages and for tests that build fixtures from raw GGUF bytes.
    pub fn ggml_type(self) -> u32 { match self { QFmt::Q4K => 12, QFmt::Q5K => 13, QFmt::Q6K => 14, QFmt::Q8_0 => 8, QFmt::Q5_0 => 6 } }
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
            prefill: OnceLock::new(),
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
    fn decode_kernels(&self) -> Option<&DecodeK> {
        self.decode.get_or_init(|| {
            const N: [&[u8]; 17] = [b"rmsnorm\0", b"add_rmsnorm\0", b"qk_norm_rope\0", b"attn_decode\0",
                b"quant_x_q8\0", b"q5k_gemv_q8\0", b"q5k_swiglu_gemv_q8\0",
                // gemv, in QFmt order: Q4_K, Q5_K (the coalesced one, not tier 1's), Q6_K, Q8_0, Q5_0
                b"q4k_gemv\0", b"q5k_gemv\0", b"q6k_gemv\0", b"q8_0_gemv\0", b"q5_0_gemv\0",
                b"q4k_swiglu_gemv\0", b"q5k_swiglu_gemv\0", b"q6k_swiglu_gemv\0", b"q8_0_swiglu_gemv\0", b"q5_0_swiglu_gemv\0"];
            let v = self.load_ptx("cuda_decode.ptx", &N)?;
            Some(DecodeK { rmsnorm: v[0], add_rmsnorm: v[1], qk_norm_rope: v[2], attn_decode: v[3],
                           quant_x_q8: v[4], q5k_gemv_q8: v[5], q5k_swiglu_gemv_q8: v[6],
                           gemv: [v[7], v[8], v[9], v[10], v[11]], swiglu: [v[12], v[13], v[14], v[15], v[16]] })
        }).as_ref()
    }

    fn prefill_kernels(&self) -> Option<&PrefillK> {
        self.prefill.get_or_init(|| {
            const N: [&[u8]; 7] = [b"q4k_gemm\0", b"q5k_gemm\0", b"q6k_gemm\0", b"q8_0_gemm\0", b"q5_0_gemm\0",
                                   b"swiglu_rows\0", b"attn_prefill\0"];
            let v = self.load_ptx("cuda_prefill.ptx", &N)?;
            Some(PrefillK { gemm: [v[0], v[1], v[2], v[3], v[4]], swiglu_rows: v[5], attn_prefill: v[6] })
        }).as_ref()
    }
    /// 2-D grid launch, same contract as [`Driver::launch`].
    unsafe fn launch2(&self, f: CUfunction, gx: u32, gy: u32, block: u32, params: &mut [*mut c_void]) -> bool {
        self.bind();
        let r = unsafe { (self.cu_launch_kernel)(f, gx, gy, 1, block, 1, 1, 0, std::ptr::null_mut(),
                                                 params.as_mut_ptr(), std::ptr::null_mut()) };
        if r != 0 { eprintln!("{}", self.err("cuLaunchKernel", r)); return false; }
        true
    }
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

/// How many whole decode steps the native graph has completed in this process — so a harness can
/// PROVE the tier ran rather than infer it from speed. (A fallback to WGSL produces the same ids; the
/// only other trace of "nothing native ran" is one stderr line.)
static NATIVE_STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn native_steps() -> u64 { NATIVE_STEPS.load(std::sync::atomic::Ordering::Relaxed) }
/// Prompt rows the native PREFILL has completed in this process (see [`native_steps`]).
static PREFILL_ROWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn native_prefill_rows() -> u64 { PREFILL_ROWS.load(std::sync::atomic::Ordering::Relaxed) }

/// A packed weight mirrored into CUDA memory at load time — from the SAME host words the WGSL buffers
/// are built from (`dtype.rs` `from_bytes`), so there is no readback and no second repack; only the
/// kernels differ across fabrics. `fmt` says how the two buffers are laid out, per block:
///
/// | fmt  | codes (u32)          | aux (u32)                               |
/// |------|----------------------|-----------------------------------------|
/// | Q4_K | 32: qs               | 4: d/dmin f16x2, 12 scale bytes         |
/// | Q5_K | 40: qs, qh           | 4: same as Q4_K                         |
/// | Q6_K | 48: ql, qh           | 5: d, 16 int8 scales                    |
/// | Q8_0 | 8: 32 int8           | ½: one f16 per block, TWO per u32       |
/// | Q5_0 | 4: 16 nibble bytes   | 2: qh, d                                |
pub struct QDev { codes: CUdeviceptr, aux: CUdeviceptr, pub fmt: QFmt, drv: Arc<Driver> }
/// Tier-1 names, kept so the `matmul_q5_k` hook and the older call sites read as they did.
pub type Q5KDev = QDev;
pub type Q6KDev = QDev;
unsafe impl Send for QDev {}
unsafe impl Sync for QDev {}
impl QDev {
    pub fn upload(fmt: QFmt, codes: &[u32], aux: &[u32]) -> Option<QDev> {
        let drv = driver()?.clone();
        let c = drv.upload(codes)?;
        let a = match drv.upload(aux) { Some(a) => a, None => { unsafe { (drv.cu_mem_free)(c); } return None; } };
        Some(QDev { codes: c, aux: a, fmt, drv })
    }
    /// Overwrite `codes[cw0..]` and `aux[aw0..]` (word offsets) in place — the device half of a WGSL
    /// weight's `write_rows`. ⛔ Without it an expert streamed into the WGSL buffers leaves this mirror
    /// holding the PREVIOUS expert: finite, plausible, wrong, and only on the native path.
    pub fn write_words(&self, cw0: usize, codes: &[u32], aw0: usize, aux: &[u32]) -> bool {
        self.drv.bind();
        unsafe {
            (self.drv.cu_memcpy_htod)(self.codes + (cw0 * 4) as u64, codes.as_ptr() as *const c_void, codes.len() * 4) == 0
                && (self.drv.cu_memcpy_htod)(self.aux + (aw0 * 4) as u64, aux.as_ptr() as *const c_void, aux.len() * 4) == 0
        }
    }
    /// Tier 1: `out[o] = Σ_k x[k]·W[o,k]` for one activation row, Q5_K only (the `matmul_q5_k` hook).
    /// `None` on any failure or another format (WGSL runs instead).
    pub fn gemv(&self, x: &[f32], o_dim: usize, in_dim: usize) -> Option<Vec<f32>> {
        debug_assert_eq!(x.len(), in_dim);
        if self.fmt != QFmt::Q5K { return None; }
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
impl Drop for QDev {
    fn drop(&mut self) { self.drv.bind(); unsafe { (self.drv.cu_mem_free)(self.codes); (self.drv.cu_mem_free)(self.aux); } }
}

/// A single-shard quantised matrix as the native graph sees it: the device mirror plus `[rows, cols]`.
#[derive(Clone, Copy)]
pub struct NativeWeight<'a> { pub dev: &'a QDev, pub rows: usize, pub cols: usize }
impl NativeWeight<'_> {
    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    pub fn fmt(&self) -> QFmt { self.dev.fmt }
    fn dw(&self) -> DW { DW { codes: self.dev.codes, aux: self.dev.aux, fmt: self.dev.fmt, rows: self.rows, cols: self.cols } }
}
/// A weight by raw handle: what the graph stores (no borrow of the `QMatrix` it came from).
#[derive(Clone, Copy)]
struct DW { codes: CUdeviceptr, aux: CUdeviceptr, fmt: QFmt, rows: usize, cols: usize }

macro_rules! p { ($($e:expr),*) => { [$( &mut $e as *mut _ as *mut c_void ),*] } }

/// Quantise `x[n]` (f32, device) into `xq` (4 int8 per u32) + `xs` (n/32 scales), n % 32 == 0.
unsafe fn quant_x(d: &Driver, k: &DecodeK, x: CUdeviceptr, xq: CUdeviceptr, xs: CUdeviceptr, n: usize) -> bool {
    let (mut xp, mut qp, mut sp, mut n32) = (x, xq, xs, n as u32);
    unsafe { d.launch(k.quant_x_q8, ((n / 32) as u32).div_ceil(4), 128, &mut p!(xp, qp, sp, n32)) }
}
/// `out[w.rows] = W · x` for one activation row, any format. `q8 = Some((xq, xs))` routes a Q5_K weight
/// through the int8-activation dp4a kernel (the FERRIC_CUDA_Q8X numerics trade); other formats ignore it.
unsafe fn launch_gemv(d: &Driver, k: &DecodeK, x: CUdeviceptr, w: DW, out: CUdeviceptr,
                      q8: Option<(CUdeviceptr, CUdeviceptr)>) -> bool {
    let (mut cp, mut ap, mut op, mut o32, mut i32_) = (w.codes, w.aux, out, w.rows as u32, w.cols as u32);
    if let (QFmt::Q5K, Some((xq, xs))) = (w.fmt, q8) {
        if !unsafe { quant_x(d, k, x, xq, xs, w.cols) } { return false; }
        let (mut qp, mut sp) = (xq, xs);
        return unsafe { d.launch(k.q5k_gemv_q8, (w.rows as u32).div_ceil(4), 128, &mut p!(qp, sp, cp, ap, op, o32, i32_)) };
    }
    let mut xp = x;
    unsafe { d.launch(k.gemv[w.fmt as usize], (w.rows as u32).div_ceil(4), 128, &mut p!(xp, cp, ap, op, o32, i32_)) }
}
/// Fused gate|up + SwiGLU: `out[o] = silu(W[o]·x) · W[o + n_ff]·x`, `w.rows == 2·n_ff`. Any format.
unsafe fn launch_swiglu(d: &Driver, k: &DecodeK, x: CUdeviceptr, w: DW, out: CUdeviceptr,
                        q8: Option<(CUdeviceptr, CUdeviceptr)>) -> bool {
    let n_ff = w.rows / 2;
    let (mut cp, mut ap, mut op, mut nff, mut din) = (w.codes, w.aux, out, n_ff as u32, w.cols as u32);
    if let (QFmt::Q5K, Some((xq, xs))) = (w.fmt, q8) {
        if !unsafe { quant_x(d, k, x, xq, xs, w.cols) } { return false; }
        let (mut qp, mut sp) = (xq, xs);
        return unsafe { d.launch(k.q5k_swiglu_gemv_q8, (n_ff as u32).div_ceil(4), 128, &mut p!(qp, sp, cp, ap, op, nff, din)) };
    }
    let mut xp = x;
    unsafe { d.launch(k.swiglu[w.fmt as usize], (n_ff as u32).div_ceil(4), 128, &mut p!(xp, cp, ap, op, nff, din)) }
}

/// `C[m, w.rows] = A[m, w.cols] · Wᵀ` on the tensor cores (f16 in, f32 accumulate; see cuda_prefill.cu).
/// `ldc` lets several weights write side by side into one wider C (the q|k|v parts). An A value past
/// f16 range raises `*ovf`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_gemm(d: &Driver, pk: &PrefillK, a: CUdeviceptr, lda: usize, w: DW, c: CUdeviceptr, ldc: usize,
                      m: usize, ovf: CUdeviceptr) -> bool {
    let (mut ap, mut la, mut cp, mut xp, mut cc, mut lc) = (a, lda as u32, w.codes, w.aux, c, ldc as u32);
    let (mut mm, mut nn, mut kk, mut of) = (m as u32, w.rows as u32, w.cols as u32, ovf);
    unsafe { d.launch2(pk.gemm[w.fmt as usize], (w.rows as u32).div_ceil(64), (m as u32).div_ceil(64), 128,
                       &mut p!(ap, la, cp, xp, cc, lc, mm, nn, kk, of)) }
}

/// **The K/V cache on the device — one per SEQUENCE, owned by the caller's cache, not by the graph.**
///
/// ⛔ It used to live inside `DecodeGraph`, keyed only by its length. One model serves many sequences
/// (`ferric-serve` slots, batched decode), and the moment a second `Cache` stepped through the same
/// model the graph saw a length that was not its own and panicked ("device cache is ahead of the WGSL
/// cache") — a length cannot tell two sequences apart. Here each `Cache` carries its own.
///
/// Rows grow by doubling with a device-to-device carry, so there is no fixed context cap (was 2048).
/// A failed growth changes nothing and returns `false`; the caller then falls back with every row
/// still intact.
pub struct DevKv { drv: Arc<Driver>, k: Vec<CUdeviceptr>, v: Vec<CUdeviceptr>, width: usize, cap: usize,
                   /// Rows valid on the device (== the position of the next token it will write).
                   pub len: usize }
unsafe impl Send for DevKv {}
impl DevKv {
    /// An empty cache for `n_layer` layers of `width` (= n_kv_heads · head_dim) floats per row.
    /// Allocates nothing until [`DevKv::reserve`].
    pub fn new(n_layer: usize, width: usize) -> Option<DevKv> {
        let drv = driver()?.clone();
        Some(DevKv { drv, k: vec![0; n_layer], v: vec![0; n_layer], width, cap: 0, len: 0 })
    }
    pub fn cap(&self) -> usize { self.cap }
    pub fn width(&self) -> usize { self.width }
    pub fn n_layer(&self) -> usize { self.k.len() }
    /// Make room for `need` rows (doubling). On any allocation failure the old buffers are untouched,
    /// what was newly allocated is freed, and `false` is returned.
    pub fn reserve(&mut self, need: usize) -> bool {
        if need <= self.cap { return true; }
        let new_cap = need.max(self.cap * 2).max(256);
        let bytes = new_cap * self.width * 4;
        let mut fresh: Vec<CUdeviceptr> = Vec::with_capacity(2 * self.k.len());
        for _ in 0..2 * self.k.len() {
            match self.drv.alloc(bytes) {
                Some(p) => fresh.push(p),
                None => { self.drv.bind(); for &p in &fresh { unsafe { (self.drv.cu_mem_free)(p); } } return false; }
            }
        }
        let carry = self.len * self.width * 4;
        let n = self.k.len();
        unsafe {
            for il in 0..n {
                if carry > 0 && ((self.drv.cu_memcpy_dtod)(fresh[il], self.k[il], carry) != 0
                                 || (self.drv.cu_memcpy_dtod)(fresh[n + il], self.v[il], carry) != 0) {
                    for &p in &fresh { (self.drv.cu_mem_free)(p); }
                    return false;
                }
            }
            for il in 0..n {
                if self.cap > 0 { (self.drv.cu_mem_free)(self.k[il]); (self.drv.cu_mem_free)(self.v[il]); }
                self.k[il] = fresh[il]; self.v[il] = fresh[n + il];
            }
        }
        self.cap = new_cap;
        true
    }
    /// Host rows → device rows `[at, at + n)` of layer `il` (`k`/`v` are `[n, width]`). Needs `reserve`.
    pub fn write_rows(&mut self, il: usize, at: usize, k: &[f32], v: &[f32]) -> bool {
        let n = k.len() / self.width.max(1);
        if k.len() != v.len() || k.len() % self.width.max(1) != 0 || at + n > self.cap { return false; }
        let off = (at * self.width * 4) as u64;
        self.drv.htod(self.k[il] + off, k) && self.drv.htod(self.v[il] + off, v)
    }
    /// Device rows `[from, from + n)` of layer `il` → host, as `([n, width], [n, width])`.
    pub fn read_rows(&self, il: usize, from: usize, n: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        if from + n > self.len { return None; }
        let off = (from * self.width * 4) as u64;
        let (mut k, mut v) = (vec![0f32; n * self.width], vec![0f32; n * self.width]);
        (self.drv.dtoh(&mut k, self.k[il] + off) && self.drv.dtoh(&mut v, self.v[il] + off)).then_some((k, v))
    }
}
impl Drop for DevKv {
    fn drop(&mut self) {
        if self.cap == 0 { return; }
        self.drv.bind();
        for il in 0..self.k.len() { unsafe { (self.drv.cu_mem_free)(self.k[il]); (self.drv.cu_mem_free)(self.v[il]); } }
    }
}

/// Everything one dense decode layer needs, as device pointers. Built once from host data.
struct LayerDev {
    attn_norm: CUdeviceptr, ffn_norm: CUdeviceptr, q_norm: Option<CUdeviceptr>, k_norm: Option<CUdeviceptr>,
    /// Qwen2's concatenated q|k|v bias `[q_out + 2 kv_out]`, added inside `qk_norm_rope`.
    qkv_bias: Option<CUdeviceptr>,
    /// One per fused-format part, written contiguously into the qkv buffer.
    qkv: Vec<DW>,
    wo: DW,
    gate_up: DW,          // any format, 2·n_ff rows — the fused SwiGLU kernel exists for all five
    down: DW,
}

/// Host-side description of one layer for [`DecodeGraph::build`].
pub struct LayerSpec<'a> {
    pub attn_norm: Vec<f32>, pub ffn_norm: Vec<f32>,
    pub q_norm: Option<Vec<f32>>, pub k_norm: Option<Vec<f32>>,
    /// Concatenated q|k|v bias (Qwen2), or `None`.
    pub qkv_bias: Option<Vec<f32>>,
    pub qkv_parts: Vec<NativeWeight<'a>>,
    pub wo: NativeWeight<'a>,
    /// gate|up stacked: `2·n_ff` rows, any supported format.
    pub gate_up: NativeWeight<'a>,
    pub down: NativeWeight<'a>,
}
pub struct GraphSpec<'a> {
    pub d: usize, pub nh: usize, pub nkv: usize, pub dh: usize, pub n_ff: usize, pub n_vocab: usize,
    pub eps: f32, pub rope_base: f32, pub has_qk_norm: bool,
    /// Per-frequency MULTIPLIER on the inverse frequency, `[dh/2]` — Llama-3 `rope_freqs` (the loader
    /// already inverted ggml's divisors) or a linear factor. `None` = plain rope.
    pub rope_ff: Option<Vec<f32>>,
    /// ggml NORM pairing `(2c, 2c+1)` (`llama`) instead of NEOX `(c, c + dh/2)` (the Qwen family).
    pub norm_pairs: bool,
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
    fn off() -> Prof { Prof { on: false, ev: [std::ptr::null_mut(); 2], acc: [0.0; 14], steps: 0 } }
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

/// **Tier 2: one dense decode step, fully resident.** Weights, activations and scratch live in device
/// memory, and the K/V rows in the caller's [`DevKv`]; per token the host copies ONE `[d]` embedding
/// row in and ONE `[n_vocab]` logits row out. Nothing else crosses the bus.
pub struct DecodeGraph {
    drv: Arc<Driver>,
    d: usize, nh: usize, nkv: usize, dh: usize, n_ff: usize, n_vocab: usize, eps: f32, rope_base: f32,
    has_qk_norm: bool, norm_pairs: bool, rope_ff: Option<CUdeviceptr>,
    layers: Vec<LayerDev>,
    out_norm: CUdeviceptr, lm_head: DW,
    x: CUdeviceptr, xn: CUdeviceptr, qkv: CUdeviceptr, q: CUdeviceptr, k: CUdeviceptr,
    attn: CUdeviceptr, y: CUdeviceptr, xy: CUdeviceptr, h: CUdeviceptr, dn: CUdeviceptr, logits: CUdeviceptr,
    q_out: usize, kv_out: usize,
    prof: Prof,
    /// FERRIC_CUDA_Q8X: int8 activations + dp4a for the Q5_K GEMVs. Changes numerics; opt-in.
    q8x: bool, xq: CUdeviceptr, xs: CUdeviceptr,
    /// Every device allocation this graph made, freed on drop (it used to leak them).
    owned: Vec<CUdeviceptr>,
    /// Prefill scratch, allocated on the first prefill and grown on demand (see [`PrefillBufs`]).
    pf: Option<PrefillBufs>,
}

/// Rows per prefill chunk. A prompt longer than this is run in chunks, each attending to the cache the
/// previous ones wrote — causal attention makes that exact — so scratch stays bounded (≈85 MB at 512
/// rows on Llama-3.2-1B) instead of growing with the prompt. `FERRIC_CUDA_PREFILL_CHUNK` overrides.
fn prefill_chunk() -> usize {
    std::env::var("FERRIC_CUDA_PREFILL_CHUNK").ok().and_then(|v| v.parse().ok()).filter(|&n: &usize| n > 0).unwrap_or(512)
}
/// Prefill scratch for up to `rows` rows (and `logit_rows` rows of logits). Indices into `b`:
const PX: usize = 0; const PXN: usize = 1; const PQKV: usize = 2; const PQ: usize = 3; const PK: usize = 4;
const PATT: usize = 5; const PY: usize = 6; const PXY: usize = 7; const PGU: usize = 8; const PH: usize = 9;
const PDN: usize = 10; const POVF: usize = 11; const PLOG: usize = 12;
struct PrefillBufs { rows: usize, logit_rows: usize, b: [CUdeviceptr; 13] }
/// ⛔ **`ferric-serve` asserts `Engine: Send` at compile time, and `Qwen3` owns a
/// `RefCell<Option<DecodeGraph>>`.** Without this impl the raw handles inside (`CUdeviceptr`,
/// `CUfunction`, the two `CUevent`s in `Prof`) make `Qwen3` — and therefore the whole server —
/// `!Send`, which broke `origin/main`'s CI for five pushes. ⚠ It is invisible on macOS: this module
/// is `#![cfg(linux/windows)]`, so `DecodeGraph` does not exist there and the local Mac gate cannot
/// see the violation. Cross-check with `--target x86_64-unknown-linux-gnu` before pushing.
///
/// **Why it is sound.** These handles belong to the CUDA *context*, not to a thread, and every entry
/// point — `launch`, `sync`, `alloc`, `htod`, `dtoh`, and `step` itself — calls `Driver::bind()`
/// first, which makes the context current on whatever thread is using it (thread_local, once per
/// thread). `Driver` already carries the same pair of impls for the same reason.
/// ⛔ **`Send`, deliberately NOT `Sync`**: the graph owns mutable device scratch (`x`, `xn`, `qkv`,
/// `xq`, …) that two threads must never drive at once. The `RefCell` around it already forbids that;
/// this keeps the type system saying so too.
unsafe impl Send for DecodeGraph {}

impl DecodeGraph {
    /// `None` when the driver or PTX is missing, or when any shape is one the kernels do not cover —
    /// the caller then stays on WGSL. Every refusal is a shape check, never an approximation.
    pub fn build(spec: &GraphSpec<'_>) -> Option<DecodeGraph> {
        let drv = driver()?.clone();
        drv.decode_kernels()?;
        let (d, nh, nkv, dh) = (spec.d, spec.nh, spec.nkv, spec.dh);
        // attn_decode gives each lane dh/32 elements and qk_norm_rope runs one 128-thread block per head.
        if dh > 128 || dh % 32 != 0 || nkv == 0 || nh % nkv != 0 { return None; }
        let q_out = nh * dh; let kv_out = nkv * dh;
        let fits = |w: &NativeWeight<'_>, rows: usize, cols: usize| w.rows == rows && w.cols == cols && cols % w.fmt().block() == 0;
        if !fits(&spec.lm_head, spec.n_vocab, d) { return None; }
        if spec.rope_ff.as_ref().is_some_and(|f| f.len() != dh / 2) { return None; }
        let mut owned: Vec<CUdeviceptr> = Vec::new();
        let mut up = |v: &[f32]| -> Option<CUdeviceptr> { let p = drv.upload_f32(v)?; owned.push(p); Some(p) };
        let mut layers = Vec::with_capacity(spec.layers.len());
        for l in &spec.layers {
            let total: usize = l.qkv_parts.iter().map(|w| w.rows).sum();
            if total != q_out + 2 * kv_out || !l.qkv_parts.iter().all(|w| w.cols == d && d % w.fmt().block() == 0) { return None; }
            if !fits(&l.wo, d, q_out) || !fits(&l.gate_up, 2 * spec.n_ff, d) || !fits(&l.down, d, spec.n_ff) { return None; }
            if l.qkv_bias.as_ref().is_some_and(|b| b.len() != q_out + 2 * kv_out) { return None; }
            layers.push(LayerDev {
                attn_norm: up(&l.attn_norm)?, ffn_norm: up(&l.ffn_norm)?,
                q_norm: match &l.q_norm { Some(v) => Some(up(v)?), None => None },
                k_norm: match &l.k_norm { Some(v) => Some(up(v)?), None => None },
                qkv_bias: match &l.qkv_bias { Some(v) => Some(up(v)?), None => None },
                qkv: l.qkv_parts.iter().map(|w| w.dw()).collect(),
                wo: l.wo.dw(), gate_up: l.gate_up.dw(), down: l.down.dw(),
            });
        }
        let out_norm = up(&spec.out_norm)?;
        let rope_ff = match &spec.rope_ff { Some(v) => Some(up(v)?), None => None };
        let widest = d.max(q_out).max(spec.n_ff);
        let mut al = |bytes: usize| -> Option<CUdeviceptr> { let p = drv.alloc(bytes)?; owned.push(p); Some(p) };
        Some(DecodeGraph {
            d, nh, nkv, dh, n_ff: spec.n_ff, n_vocab: spec.n_vocab, eps: spec.eps, rope_base: spec.rope_base,
            has_qk_norm: spec.has_qk_norm, norm_pairs: spec.norm_pairs, rope_ff, layers,
            out_norm, lm_head: spec.lm_head.dw(),
            x: al(d * 4)?, xn: al(d * 4)?, qkv: al((q_out + 2 * kv_out) * 4)?,
            q: al(q_out * 4)?, k: al(kv_out * 4)?, attn: al(q_out * 4)?,
            y: al(d * 4)?, xy: al(d * 4)?, h: al(spec.n_ff * 4)?, dn: al(d * 4)?,
            logits: al(spec.n_vocab * 4)?,
            q_out, kv_out, prof: Prof::new(&drv),
            q8x: std::env::var("FERRIC_CUDA_Q8X").is_ok(),
            // ⛔ Sized by the WIDEST input a Q5_K GEMV can see. It was `d.max(q_out)`, which a Q5_K
            // ffn_down (input n_ff) would have overrun under FERRIC_CUDA_Q8X — Qwen3-0.6B's down is
            // Q6_K, which is the only reason it never fired.
            xq: al(widest)?, xs: al(widest / 32 * 4 + 4)?,
            owned, drv, pf: None,
        })
    }
    /// A [`DevKv`] shaped for this graph.
    pub fn new_kv(&self) -> Option<DevKv> { DevKv::new(self.layers.len(), self.kv_out) }
    /// Rows per layer of K (= of V): `n_kv_heads · head_dim`.
    pub fn kv_width(&self) -> usize { self.kv_out }
    fn q8(&self) -> Option<(CUdeviceptr, CUdeviceptr)> { self.q8x.then_some((self.xq, self.xs)) }
    /// One decode step: `x_row` is the (already scaled/normed) embedding of the token at position
    /// `kv.len`. Returns the logits row and advances `kv.len`. `None` on any failure — including a
    /// K/V growth that could not allocate — with `kv` unchanged, so the caller can fall back.
    pub fn step(&mut self, kv: &mut DevKv, x_row: &[f32]) -> Option<Vec<f32>> {
        if x_row.len() != self.d || kv.n_layer() != self.layers.len() || kv.width() != self.kv_out { return None; }
        let pos = kv.len;
        if !kv.reserve(pos + 1) { return None; }
        let k = *self.drv.decode_kernels()?;
        let d = self.d;
        let drv = self.drv.clone();
        drv.bind();
        let q8 = self.q8();
        let mut prof = std::mem::replace(&mut self.prof, Prof::off());
        prof.start(&drv);
        if !drv.htod(self.x, x_row) { self.prof = prof; return None; }
        prof.stop(&drv, 13);
        macro_rules! timed { ($cls:expr, $body:expr) => {{ prof.start(&drv); let ok: bool = $body; prof.stop(&drv, $cls); if !ok { self.prof = prof; return None; } }} }
        unsafe {
            let (mut d32, mut eps) = (d as u32, self.eps);
            // Layer 0's attn_norm is a plain rmsnorm; every later layer's is fused into the previous
            // layer's residual add (one add_rmsnorm instead of vadd + rmsnorm), and the last layer's
            // residual add is fused with out_norm the same way. 56 fewer launches per token.
            let nl = self.layers.len();
            for (li, l) in self.layers.iter().enumerate() {
                if li == 0 {
                    let (mut x, mut w, mut o) = (self.x, l.attn_norm, self.xn);
                    timed!(0, drv.launch(k.rmsnorm, 1, 256, &mut p!(x, w, o, d32, eps)));
                }
                timed!(1, { let mut off = 0usize; let mut ok = true;
                    for &w in &l.qkv { ok &= launch_gemv(&drv, &k, self.xn, w, self.qkv + (off * 4) as u64, q8); off += w.rows; } ok });
                let (mut qkv, mut qw, mut kw, mut qo, mut ko) = (self.qkv, l.q_norm.unwrap_or(0), l.k_norm.unwrap_or(0), self.q, self.k);
                let (mut nh, mut nkv, mut dh) = (self.nh as u32, self.nkv as u32, self.dh as u32);
                let (mut base, mut posu, mut qoff, mut koff, mut hn) = (self.rope_base, pos as u32, 0u32, self.q_out as u32, self.has_qk_norm as u32);
                let rowb = (self.kv_out * 4) as u64;
                // K and V rows go straight into the cache from the kernel: no D2D copies (class 3 is 0).
                let (mut kc_row, mut vc_row, mut voff) = (kv.k[li] + pos as u64 * rowb, kv.v[li] + pos as u64 * rowb, (self.q_out + self.kv_out) as u32);
                let (mut bias, mut ff, mut np, mut roww) = (l.qkv_bias.unwrap_or(0), self.rope_ff.unwrap_or(0), self.norm_pairs as u32, (self.q_out + 2 * self.kv_out) as u32);
                let heads = (self.nh + self.nkv) as u32;
                timed!(2, drv.launch(k.qk_norm_rope, heads, 128,
                                     &mut p!(qkv, qw, kw, qo, ko, nh, nkv, dh, base, posu, eps, qoff, koff, hn, kc_row, vc_row, voff, bias, ff, np, roww)));
                let (mut q, mut kc, mut vc, mut ao, mut s, mut scale) = (self.q, kv.k[li], kv.v[li], self.attn, (pos + 1) as u32, 1.0f32 / (self.dh as f32).sqrt());
                timed!(4, drv.launch(k.attn_decode, self.nh as u32, 128, &mut p!(q, kc, vc, ao, nh, nkv, dh, s, scale)));
                timed!(5, launch_gemv(&drv, &k, self.attn, l.wo, self.y, q8));
                let (mut x2, mut y2, mut fw, mut xy, mut xn) = (self.x, self.y, l.ffn_norm, self.xy, self.xn);
                timed!(6, drv.launch(k.add_rmsnorm, 1, 256, &mut p!(x2, y2, fw, xy, xn, d32, eps)));
                timed!(7, launch_swiglu(&drv, &k, self.xn, l.gate_up, self.h, q8));
                timed!(8, launch_gemv(&drv, &k, self.h, l.down, self.dn, q8));
                // x = xy + dn, and in the same kernel xn = rmsnorm(x) * (next attn_norm | out_norm)
                let next_w = if li + 1 < nl { self.layers[li + 1].attn_norm } else { self.out_norm };
                let (mut a, mut b, mut nw, mut xo, mut xno) = (self.xy, self.dn, next_w, self.x, self.xn);
                timed!(9, drv.launch(k.add_rmsnorm, 1, 256, &mut p!(a, b, nw, xo, xno, d32, eps)));
            }
            timed!(11, launch_gemv(&drv, &k, self.xn, self.lm_head, self.logits, q8));
        }
        if !drv.sync() { self.prof = prof; return None; }
        let mut out = vec![0f32; self.n_vocab];
        prof.start(&drv);
        if !drv.dtoh(&mut out, self.logits) { self.prof = prof; return None; }
        prof.stop(&drv, 12);
        prof.tick();
        self.prof = prof;
        kv.len = pos + 1;
        NATIVE_STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(out)
    }
}
impl DecodeGraph {
    fn free_prefill(&mut self) {
        if let Some(pf) = self.pf.take() { self.drv.bind(); for p in pf.b { unsafe { (self.drv.cu_mem_free)(p); } } }
    }
    /// Make sure the prefill scratch holds `rows` rows and `logit_rows` rows of logits.
    fn prefill_bufs(&mut self, rows: usize, logit_rows: usize) -> bool {
        if self.pf.as_ref().is_some_and(|b| b.rows >= rows && b.logit_rows >= logit_rows) { return true; }
        let (rows, logit_rows) = match &self.pf { Some(b) => (rows.max(b.rows), logit_rows.max(b.logit_rows)), None => (rows, logit_rows) };
        self.free_prefill();
        let (d, w, ff) = (self.d, self.q_out + 2 * self.kv_out, self.n_ff);
        let sizes = [rows * d, rows * d, rows * w, rows * self.q_out, rows * self.kv_out, rows * self.q_out,
                     rows * d, rows * d, rows * 2 * ff, rows * ff, rows * d, 1, logit_rows * self.n_vocab];
        let mut b = [0 as CUdeviceptr; 13];
        for (i, n) in sizes.into_iter().enumerate() {
            match self.drv.alloc(n * 4) {
                Some(p) => b[i] = p,
                None => { self.drv.bind(); for &p in &b[..i] { unsafe { (self.drv.cu_mem_free)(p); } } return false; }
            }
        }
        self.pf = Some(PrefillBufs { rows, logit_rows, b });
        true
    }

    /// **Prefill: T prompt rows through every layer on the device**, K/V rows written straight into
    /// `kv` at `[kv.len, kv.len + T)`. `x_rows` is `[T, d]` (already scaled/normed embeddings).
    /// Returns the logits of every row (`all_logits`, `[T, n_vocab]`) or of the last row only.
    ///
    /// The matmuls run on the tensor cores (`cuda_prefill.cu`: f16 in, f32 accumulate); norms, rope,
    /// bias and the residual adds are the decode kernels with a row grid, so the two paths share them.
    /// `None` on any failure — including an activation past f16 range, detected in the GEMM — with
    /// `kv.len` unchanged: rows written past it are dead, and the caller's WGSL prefill runs instead.
    pub fn prefill(&mut self, kv: &mut DevKv, x_rows: &[f32], all_logits: bool) -> Option<Vec<f32>> {
        let d = self.d;
        if x_rows.is_empty() || x_rows.len() % d != 0 || kv.n_layer() != self.layers.len() || kv.width() != self.kv_out { return None; }
        let t = x_rows.len() / d;
        let k = *self.drv.decode_kernels()?;
        let pk = *self.drv.prefill_kernels()?;
        let pos0 = kv.len;
        if !kv.reserve(pos0 + t) { return None; }
        let chunk = prefill_chunk().min(t);
        if !self.prefill_bufs(chunk, if all_logits { chunk } else { 1 }) { return None; }
        let b = self.pf.as_ref().unwrap().b;
        let drv = self.drv.clone();
        drv.bind();
        if !drv.htod(b[POVF], &[0.0]) { return None; }
        let (q_out, kv_out, n_ff, nv) = (self.q_out, self.kv_out, self.n_ff, self.n_vocab);
        let width = q_out + 2 * kv_out;
        let mut out = Vec::with_capacity(if all_logits { t * nv } else { nv });
        let nl = self.layers.len();
        unsafe {
            let (mut d32, mut eps) = (d as u32, self.eps);
            for c0 in (0..t).step_by(chunk) {
                let rows = chunk.min(t - c0);
                let (pos, r32) = (pos0 + c0, rows as u32);
                if !drv.htod(b[PX], &x_rows[c0 * d..(c0 + rows) * d]) { return None; }
                for (li, l) in self.layers.iter().enumerate() {
                    if li == 0 {
                        let (mut x, mut w, mut o) = (b[PX], l.attn_norm, b[PXN]);
                        if !drv.launch(k.rmsnorm, r32, 256, &mut p!(x, w, o, d32, eps)) { return None; }
                    }
                    let mut off = 0usize;
                    for &w in &l.qkv {
                        if !launch_gemm(&drv, &pk, b[PXN], d, w, b[PQKV] + (off * 4) as u64, width, rows, b[POVF]) { return None; }
                        off += w.rows;
                    }
                    let (mut qkv, mut qw, mut kw, mut qo, mut ko) = (b[PQKV], l.q_norm.unwrap_or(0), l.k_norm.unwrap_or(0), b[PQ], b[PK]);
                    let (mut nh, mut nkv, mut dh) = (self.nh as u32, self.nkv as u32, self.dh as u32);
                    let (mut base, mut posu, mut qoff, mut koff, mut hn) = (self.rope_base, pos as u32, 0u32, q_out as u32, self.has_qk_norm as u32);
                    let rowb = (kv_out * 4) as u64;
                    let (mut kc_row, mut vc_row, mut voff) = (kv.k[li] + pos as u64 * rowb, kv.v[li] + pos as u64 * rowb, (q_out + kv_out) as u32);
                    let (mut bias, mut ff, mut np, mut roww) = (l.qkv_bias.unwrap_or(0), self.rope_ff.unwrap_or(0), self.norm_pairs as u32, width as u32);
                    if !drv.launch2(k.qk_norm_rope, (self.nh + self.nkv) as u32, r32, 128,
                                    &mut p!(qkv, qw, kw, qo, ko, nh, nkv, dh, base, posu, eps, qoff, koff, hn, kc_row, vc_row, voff, bias, ff, np, roww)) { return None; }
                    let (mut q, mut kc, mut vc, mut ao, mut tt, mut sc) = (b[PQ], kv.k[li], kv.v[li], b[PATT], r32, 1.0f32 / (self.dh as f32).sqrt());
                    if !drv.launch2(pk.attn_prefill, r32.div_ceil(16), self.nh as u32, 128,
                                    &mut p!(q, kc, vc, ao, nh, nkv, dh, tt, posu, sc)) { return None; }
                    if !launch_gemm(&drv, &pk, b[PATT], q_out, l.wo, b[PY], d, rows, b[POVF]) { return None; }
                    let (mut x2, mut y2, mut fw, mut xy, mut xn) = (b[PX], b[PY], l.ffn_norm, b[PXY], b[PXN]);
                    if !drv.launch(k.add_rmsnorm, r32, 256, &mut p!(x2, y2, fw, xy, xn, d32, eps)) { return None; }
                    if !launch_gemm(&drv, &pk, b[PXN], d, l.gate_up, b[PGU], 2 * n_ff, rows, b[POVF]) { return None; }
                    let (mut gu, mut h, mut nff) = (b[PGU], b[PH], n_ff as u32);
                    if !drv.launch(pk.swiglu_rows, ((rows * n_ff) as u32).div_ceil(256), 256, &mut p!(gu, h, nff, tt)) { return None; }
                    if !launch_gemm(&drv, &pk, b[PH], n_ff, l.down, b[PDN], d, rows, b[POVF]) { return None; }
                    let next_w = if li + 1 < nl { self.layers[li + 1].attn_norm } else { self.out_norm };
                    let (mut a, mut bb, mut nw, mut xo, mut xno) = (b[PXY], b[PDN], next_w, b[PX], b[PXN]);
                    if !drv.launch(k.add_rmsnorm, r32, 256, &mut p!(a, bb, nw, xo, xno, d32, eps)) { return None; }
                }
                if all_logits {
                    if !launch_gemm(&drv, &pk, b[PXN], d, self.lm_head, b[PLOG], nv, rows, b[POVF]) { return None; }
                    let mut part = vec![0f32; rows * nv];
                    if !drv.dtoh(&mut part, b[PLOG]) { return None; }
                    out.extend_from_slice(&part);
                } else if c0 + rows == t {
                    let last = b[PXN] + ((rows - 1) * d * 4) as u64;
                    if !launch_gemv(&drv, &k, last, self.lm_head, b[PLOG], None) { return None; }
                    let mut row = vec![0f32; nv];
                    if !drv.dtoh(&mut row, b[PLOG]) { return None; }
                    out.extend_from_slice(&row);
                }
            }
        }
        if !drv.sync() { return None; }
        let mut flag = [0f32]; if !drv.dtoh(&mut flag, b[POVF]) { return None; }
        if flag[0].to_bits() != 0 {
            eprintln!("cuda: a prefill activation exceeded f16 range (65504); discarding the native prefill — WGSL runs it");
            return None;
        }
        kv.len = pos0 + t;
        PREFILL_ROWS.fetch_add(t as u64, std::sync::atomic::Ordering::Relaxed);
        Some(out)
    }
}
impl Drop for DecodeGraph {
    fn drop(&mut self) {
        self.free_prefill();
        self.drv.bind(); for &p in &self.owned { unsafe { (self.drv.cu_mem_free)(p); } }
    }
}

/// **Per-kernel microbench for the native GEMVs** — `iters` back-to-back launches rotating over `ws`,
/// one sync, returns (µs per call, the output). Used by `examples/cuda_gemv_bench.rs` to say WHICH
/// decode shape is furthest from the bandwidth floor, so kernel work starts where the bytes are.
/// `q8x` routes Q5_K through the int8-activation kernel (quantise included in the time).
fn bench_launch(ws: &[NativeWeight<'_>], x: &[f32], iters: usize, swiglu: bool, q8x: bool) -> Option<(f64, Vec<f32>)> {
    let drv = driver()?.clone(); drv.bind();
    let w0 = ws.first()?;
    let (rows, cols) = (w0.rows, w0.cols);
    if x.len() != cols || (swiglu && rows % 2 != 0) || (q8x && w0.fmt() != QFmt::Q5K) { return None; }
    let n_out = if swiglu { rows / 2 } else { rows };
    let k = *drv.decode_kernels()?;
    let (xd, xq, xs, od) = (drv.upload_f32(x)?, drv.alloc(cols)?, drv.alloc(cols / 32 * 4 + 4)?, drv.alloc(n_out * 4)?);
    let q8 = q8x.then_some((xq, xs));
    let run = |w: &NativeWeight<'_>| unsafe { if swiglu { launch_swiglu(&drv, &k, xd, w.dw(), od, q8) } else { launch_gemv(&drv, &k, xd, w.dw(), od, q8) } };
    if !run(w0) || !drv.sync() { return None; }      // warm (PTX JIT etc.)
    let t0 = std::time::Instant::now();
    for i in 0..iters { if !run(&ws[i % ws.len()]) { return None; } }
    if !drv.sync() { return None; }
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let mut out = vec![0f32; n_out]; if !drv.dtoh(&mut out, od) { return None; }
    unsafe { for p in [xd, xq, xs, od] { (drv.cu_mem_free)(p); } }
    Some((us, out))
}
/// GEMV microbench, any format.
pub fn bench_gemv(ws: &[NativeWeight<'_>], x: &[f32], iters: usize) -> Option<(f64, Vec<f32>)> { bench_launch(ws, x, iters, false, false) }
/// Q5_K GEMV with int8 activations (the FERRIC_CUDA_Q8X path): quantise + dp4a GEMV per call.
pub fn bench_gemv_q8(ws: &[NativeWeight<'_>], x: &[f32], iters: usize) -> Option<(f64, Vec<f32>)> { bench_launch(ws, x, iters, false, true) }
/// Fused gate|up+SwiGLU microbench, same contract; `w` has `2*n_ff` rows, any format.
pub fn bench_swiglu(ws: &[NativeWeight<'_>], x: &[f32], iters: usize) -> Option<(f64, Vec<f32>)> { bench_launch(ws, x, iters, true, false) }
/// Fused gate|up+SwiGLU with int8 activations (FERRIC_CUDA_Q8X). Same contract as `bench_swiglu`,
/// so the two are directly comparable — the swiglu shape is the LARGEST per-layer weight read, and
/// leaving it out of the table let the q8x rows cover under half of that traffic.
pub fn bench_swiglu_q8(ws: &[NativeWeight<'_>], x: &[f32], iters: usize) -> Option<(f64, Vec<f32>)> { bench_launch(ws, x, iters, true, true) }

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
        let ctx = pollster::block_on(ferric_core::Context::new()).ok().map(Arc::new)?;
        // The WGSL twin is a reference only on a real GPU adapter; say which one it was.
        eprintln!("{name}: wgpu adapter = {} [{:?}], CUDA = {}", ctx.adapter_name, ctx.backend, driver().unwrap().name);
        Some(ctx)
    }
    /// Raw GGUF blocks of format `f` for a `[rows, cols]` weight: random bytes with sane f16 scales
    /// written where each format keeps them, so dequantised values are finite and O(0.1–1).
    fn q_fixture(f: QFmt, rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        let (vals, bpb) = crate::dtype::QMatrix::block_bytes(f.ggml_type()).unwrap();
        let mut seed = seed; let mut bytes = vec![0u8; rows * (cols / vals) * bpb];
        for b in bytes.iter_mut() { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; *b = (seed >> 40) as u8; }
        for (bi, blk) in bytes.chunks_exact_mut(bpb).enumerate() {
            let d = half::f16::from_f32(0.01 + 0.003 * (bi % 7) as f32);
            match f {
                QFmt::Q6K => blk[208..210].copy_from_slice(&d.to_le_bytes()),
                QFmt::Q8_0 | QFmt::Q5_0 => blk[0..2].copy_from_slice(&d.to_le_bytes()),
                QFmt::Q4K | QFmt::Q5K => { blk[0..2].copy_from_slice(&d.to_le_bytes());
                    blk[2..4].copy_from_slice(&half::f16::from_f32(0.002 + 0.0005 * (bi % 5) as f32).to_le_bytes()); }
            }
        }
        bytes
    }
    /// **The independent reference**: dequantise the RAW GGUF blocks on the host with ferric-gguf's CPU
    /// dequantiser (shares no code with either GPU kernel or the repack) and accumulate in f64.
    fn host_gemv_f64(f: QFmt, bytes: &[u8], rows: usize, cols: usize, x: &[f32]) -> Vec<f64> {
        let rb = bytes.len() / rows;
        (0..rows).map(|o| {
            let w = ferric_gguf::deq_raw(&bytes[o * rb..(o + 1) * rb], cols, f.ggml_type()).expect("host dequant");
            w.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum()
        }).collect()
    }
    fn max_abs_diff(a: &[f32], b: &[f64]) -> f64 { a.iter().zip(b).fold(0f64, |m, (&x, &y)| m.max((x as f64 - y).abs())) }
    /// Gate a native result against the f64 host reference, with the WGSL kernel's own distance from
    /// that reference as the recorded f32 noise floor. Passing means: within 4x the floor (or 1e-6 of
    /// scale when the floor is smaller than that, e.g. an exact tie).
    fn gate(name: &str, native: &[f32], wgsl: &[f32], host: &[f64]) {
        assert_eq!(native.len(), host.len(), "{name}: length");
        assert_eq!(wgsl.len(), host.len(), "{name}: wgsl length");
        let scale = host.iter().fold(0f64, |a, &v| a.max(v.abs()));
        assert!(scale > 1e-3, "{name}: reference is ~zero; would pass on anything");
        assert!(native.iter().all(|v| v.is_finite()), "{name}: non-finite");
        let (dn, dw) = (max_abs_diff(native, host), max_abs_diff(wgsl, host));
        let tol = 4.0 * dw.max(1e-6 * scale);
        eprintln!("{name}: max|Δ| vs f64 host  CUDA {dn:.3e}   WGSL {dw:.3e} (the f32 floor)   scale {scale:.3e}   tol {tol:.3e}");
        assert!(dn <= tol, "{name}: native diverges from the f64 host reference by {dn:.3e} (> {tol:.3e}; WGSL sits at {dw:.3e})");
    }
    /// The WGSL kernel for a single-shard weight, never through a native hook. Only Q5_K HAS a hook
    /// (`matmul_q5_k`); it takes the FLAT seam, the rest go through `matmul_q`, which for them is WGSL.
    fn wgsl_ref(qm: &crate::dtype::QMatrix, f: QFmt, x: &crate::Tensor) -> Vec<f32> {
        let t = match f { QFmt::Q5K => qm.q5k_flat_wgsl(x).unwrap(), QFmt::Q6K => qm.q6k_flat_wgsl(x).unwrap(), _ => x.matmul_q(qm) };
        pollster::block_on(t.to_vec())
    }
    fn run_native(w: &NativeWeight<'_>, xv: &[f32], swiglu: bool) -> Vec<f32> {
        let drv = driver().unwrap(); let k = *drv.decode_kernels().expect("decode ptx");
        let n_out = if swiglu { w.rows / 2 } else { w.rows };
        let (xd, od) = (drv.upload_f32(xv).unwrap(), drv.alloc(n_out * 4).unwrap());
        let ok = unsafe { if swiglu { launch_swiglu(drv, &k, xd, w.dw(), od, None) } else { launch_gemv(drv, &k, xd, w.dw(), od, None) } };
        assert!(ok && drv.sync(), "launch failed");
        let mut got = vec![0f32; n_out]; assert!(drv.dtoh(&mut got, od));
        unsafe { (drv.cu_mem_free)(xd); (drv.cu_mem_free)(od); }
        got
    }

    /// **Every format's GEMV and fused SwiGLU, against the f64 host reference and the WGSL kernel.**
    ///
    /// Shapes are chosen to hit the lane plans' edges, not just a happy multiple: 96 = 3 blocks of 32 (an
    /// ODD count per row, so Q8_0's two-scales-per-word packing changes parity from row to row), 896 = 28 blocks of 32
    /// (Qwen2.5-0.5B's d; the 16-block warp stride leaves a remainder), 2304 = 9 K-blocks (the 4
    /// block-lanes leave one over), 4864 = Qwen2.5's n_ff, and an output count of 37 (the last
    /// 128-thread block is part-empty, exercising the warp-uniform `o >= o_dim` exit).
    #[test]
    fn every_format_gemv_and_swiglu_match_the_f64_host_reference() {
        let Some(ctx) = ctx_or_skip("every_format_gemv_and_swiglu_match_the_f64_host_reference") else { return };
        let cases: &[(QFmt, usize)] = &[
            (QFmt::Q4K, 1024), (QFmt::Q4K, 2304), (QFmt::Q5K, 2304), (QFmt::Q6K, 2304),
            (QFmt::Q8_0, 896), (QFmt::Q8_0, 4864), (QFmt::Q5_0, 896), (QFmt::Q5_0, 4864), (QFmt::Q8_0, 96), (QFmt::Q5_0, 96),
        ];
        for (ci, &(f, inn)) in cases.iter().enumerate() {
            for &out in &[37usize, 96] {
                let seed = 0x5EED_0000 + (ci * 131 + out) as u64;
                let bytes = q_fixture(f, out, inn, seed);
                let xv: Vec<f32> = (0..inn).map(|i| ((i as f32) * 0.37 + ci as f32).sin()).collect();
                let x = crate::Tensor::from_vec(&ctx, &xv, &[1, inn]);
                let qm = crate::dtype::QMatrix::from_bytes(&ctx, &bytes, f.ggml_type(), out, inn).expect("qmatrix");
                let w = qm.native_weight().unwrap_or_else(|| panic!("{f:?}: no native mirror"));
                assert_eq!(w.fmt(), f, "mirror carries the wrong format tag");
                let host = host_gemv_f64(f, &bytes, out, inn, &xv);
                gate(&format!("{f:?} gemv {out}x{inn}"), &run_native(&w, &xv, false), &wgsl_ref(&qm, f, &x), &host);
                // fused gate|up + SwiGLU on the same weight read as [gate = rows 0..h, up = rows h..2h]
                if out % 2 == 0 {
                    let h = out / 2;
                    let sw: Vec<f64> = (0..h).map(|o| { let g = host[o]; g / (1.0 + (-g).exp()) * host[o + h] }).collect();
                    let wsw = { let wv = wgsl_ref(&qm, f, &x); (0..h).map(|o| { let g = wv[o]; g / (1.0 + (-g).exp()) * wv[o + h] }).collect::<Vec<f32>>() };
                    gate(&format!("{f:?} swiglu {h}x{inn}"), &run_native(&w, &xv, true), &wsw, &sw);
                }
            }
        }
    }

    /// Q6_K native GEMV vs the WGSL FLAT kernel through the hermetic seam (never the hooked entry).
    #[test]
    fn q6k_gemv_matches_flat_wgsl() {
        let Some(ctx) = ctx_or_skip("q6k_gemv_matches_flat_wgsl") else { return };
        let (inn, out) = (1024usize, 96usize);
        let bytes = q_fixture(QFmt::Q6K, out, inn, 0xC0FFEE);
        let xv: Vec<f32> = (0..inn).map(|i| ((i as f32) * 0.29).cos()).collect();
        let x = crate::Tensor::from_vec(&ctx, &xv, &[1, inn]);
        let qm = crate::dtype::QMatrix::from_bytes(&ctx, &bytes, 14, out, inn).expect("Q6_K");
        let want = pollster::block_on(qm.q6k_flat_wgsl(&x).expect("single shard").to_vec());
        let got = run_native(&qm.native_weight().expect("mirror"), &xv, false);
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
        let bytes = q_fixture(QFmt::Q5K, out, inn, 0xA11CE);
        let xv: Vec<f32> = (0..inn).map(|i| ((i as f32) * 0.37).sin()).collect();
        let x = crate::Tensor::from_vec(&ctx, &xv, &[1, inn]);
        let qm = crate::dtype::QMatrix::from_bytes(&ctx, &bytes, 13, out, inn).expect("Q5_K");
        let want = pollster::block_on(qm.q5k_flat_wgsl(&x).expect("single shard").to_vec());
        let w = qm.native_weight().expect("mirror");
        let (_, got) = bench_gemv_q8(std::slice::from_ref(&w), &xv, 1).expect("q8 path");
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
        let bytes = q_fixture(QFmt::Q5K, 2 * n_ff, inn, 0xBEEF);
        let xv: Vec<f32> = (0..inn).map(|i| ((i as f32) * 0.41).sin()).collect();
        let x = crate::Tensor::from_vec(&ctx, &xv, &[1, inn]);
        let qm = crate::dtype::QMatrix::from_bytes(&ctx, &bytes, 13, 2 * n_ff, inn).expect("Q5_K");
        let want = pollster::block_on(qm.q5k_flat_wgsl(&x).expect("single shard").swiglu(n_ff).to_vec());
        let got = run_native(&qm.native_weight().expect("mirror"), &xv, true);
        close("Q5_K swiglu vs WGSL composed", &want, &got, 2e-4);
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
    fn rnd(n: usize, seed: u64) -> Vec<f32> { (0..n).map(|i| ((i as u64 * 2654435761 + seed) % 1000) as f32 / 500.0 - 1.0).collect() }

    /// Launch `qk_norm_rope` for one row and return (q, k, k-cache row, v-cache row).
    #[allow(clippy::too_many_arguments)]
    fn run_rope(qkv: &[f32], qw: Option<&[f32]>, kw: Option<&[f32]>, bias: Option<&[f32]>, ff: Option<&[f32]>,
                norm_pairs: bool, nh: usize, nkv: usize, dh: usize, base: f32, pos: usize, eps: f32)
                -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let drv = driver().unwrap(); let k = *drv.decode_kernels().expect("decode ptx");
        let (q_out, kv_out) = (nh * dh, nkv * dh);
        let up = |v: Option<&[f32]>| v.map(|v| drv.upload_f32(v).unwrap()).unwrap_or(0);
        let (mut sd, mut qwd, mut kwd, mut bd, mut ffd) = (drv.upload_f32(qkv).unwrap(), up(qw), up(kw), up(bias), up(ff));
        let (mut qo, mut ko, mut kc, mut vc) = (drv.alloc(q_out * 4).unwrap(), drv.alloc(kv_out * 4).unwrap(), drv.alloc(kv_out * 4).unwrap(), drv.alloc(kv_out * 4).unwrap());
        let (mut nh32, mut nkv32, mut dh32, mut b, mut p32, mut e2) = (nh as u32, nkv as u32, dh as u32, base, pos as u32, eps);
        let (mut qoff, mut koff, mut hn, mut voff, mut np, mut rw) = (0u32, q_out as u32, qw.is_some() as u32, (q_out + kv_out) as u32, norm_pairs as u32, (q_out + 2 * kv_out) as u32);
        assert!(unsafe { drv.launch(k.qk_norm_rope, (nh + nkv) as u32, 128,
            &mut p!(sd, qwd, kwd, qo, ko, nh32, nkv32, dh32, b, p32, e2, qoff, koff, hn, kc, vc, voff, bd, ffd, np, rw)) } && drv.sync());
        let (mut qg, mut kg, mut kcg, mut vcg) = (vec![0f32; q_out], vec![0f32; kv_out], vec![0f32; kv_out], vec![0f32; kv_out]);
        assert!(drv.dtoh(&mut qg, qo) && drv.dtoh(&mut kg, ko) && drv.dtoh(&mut kcg, kc) && drv.dtoh(&mut vcg, vc));
        (qg, kg, kcg, vcg)
    }
    /// f64 rope of `heads` heads of `x` (already biased/normed), ggml NORM or NEOX pairing.
    fn host_rope(x: &[f64], heads: usize, dh: usize, base: f64, pos: usize, ff: Option<&[f32]>, norm_pairs: bool) -> Vec<f64> {
        let half = dh / 2;
        let mut out = x.to_vec();
        for h in 0..heads {
            for c in 0..half {
                let inv = (-2.0 * c as f64 / dh as f64 * base.ln()).exp() * ff.map_or(1.0, |f| f[c] as f64);
                let (s, co) = (pos as f64 * inv).sin_cos();
                let (p0, p1) = if norm_pairs { (2 * c, 2 * c + 1) } else { (c, c + half) };
                let (x1, x2) = (x[h * dh + p0], x[h * dh + p1]);
                out[h * dh + p0] = x1 * co - x2 * s;
                out[h * dh + p1] = x2 * co + x1 * s;
            }
        }
        out
    }

    /// **Qwen2 biases and Llama-3 rope (NORM pairing + `rope_freqs`) in the fused kernel**, against the
    /// WGSL composed path (`add` then `rope_scaled_interleaved` / `rope`) and an f64 host rope.
    ///
    /// Positions 11 and 3001: the second is past the old 2048 cap, where an f32 angle `pos·inv` has
    /// lost enough bits that a wrong frequency formula and a right one can no longer hide in rounding.
    #[test]
    fn rope_variants_and_qkv_bias_match_wgsl_and_f64() {
        let Some(ctx) = ctx_or_skip("rope_variants_and_qkv_bias_match_wgsl_and_f64") else { return };
        let (nh, nkv, dh, eps) = (8usize, 2usize, 64usize, 1e-6f32);
        let (q_out, kv_out) = (nh * dh, nkv * dh); let width = q_out + 2 * kv_out;
        let qkv = rnd(width, 3); let bias: Vec<f32> = rnd(width, 9).iter().map(|v| v * 0.5).collect();
        // Llama-3.2's shape of factors: 1 on the fast dims, up to 1/32 on the slow ones (already inverted).
        let ff: Vec<f32> = (0..dh / 2).map(|c| if c < dh / 4 { 1.0 } else { 1.0 / (1.0 + (c - dh / 4) as f32) }).collect();
        for &(norm_pairs, with_ff, with_bias, base) in &[(true, true, false, 500000.0f32), (false, false, true, 1e6), (true, false, true, 1e4), (false, true, false, 1e4)] {
            for &pos in &[11usize, 3001] {
                let name = format!("rope pairs={} ff={with_ff} bias={with_bias} base={base} pos={pos}", if norm_pairs { "NORM" } else { "NEOX" });
                let (bo, fo) = (with_bias.then_some(bias.as_slice()), with_ff.then_some(ff.as_slice()));
                let (qg, kg, kcg, vcg) = run_rope(&qkv, None, None, bo, fo, norm_pairs, nh, nkv, dh, base, pos, eps);
                // WGSL composed reference
                let src = crate::Tensor::from_vec(&ctx, &qkv, &[1, width]);
                let src = if with_bias { src.add(&crate::Tensor::from_vec(&ctx, &bias, &[1, width])) } else { src };
                let ft = crate::Tensor::from_vec(&ctx, &ff, &[dh / 2]);
                let wr = |x: crate::Tensor, heads: usize| -> Vec<f32> {
                    let x = x.contiguous();
                    let r = match (with_ff, norm_pairs) {
                        (true, true) => x.rope_scaled_interleaved(&ft, heads, dh, base, pos),
                        (true, false) => x.rope_scaled(&ft, heads, dh, base, pos),
                        (false, true) => x.rope_interleaved(heads, dh, base, pos),
                        (false, false) => x.rope(heads, dh, base, pos),
                    };
                    pollster::block_on(r.to_vec())
                };
                let (qw_, kw_) = (wr(src.narrow(1, 0, q_out), nh), wr(src.narrow(1, q_out, kv_out), nkv));
                // f64 host reference
                let b64 = |i: usize| qkv[i] as f64 + if with_bias { bias[i] as f64 } else { 0.0 };
                let qh = host_rope(&(0..q_out).map(b64).collect::<Vec<_>>(), nh, dh, base as f64, pos, fo, norm_pairs);
                let kh = host_rope(&(q_out..q_out + kv_out).map(b64).collect::<Vec<_>>(), nkv, dh, base as f64, pos, fo, norm_pairs);
                gate(&format!("{name} q"), &qg, &qw_, &qh);
                gate(&format!("{name} k"), &kg, &kw_, &kh);
                assert!(kg.iter().zip(&kcg).all(|(a, b)| a.to_bits() == b.to_bits()), "{name}: K cache row != roped k");
                let vh: Vec<f32> = (q_out + kv_out..width).map(|i| (b64(i)) as f32).collect();
                assert!(vh.iter().zip(&vcg).all(|(a, b)| a.to_bits() == b.to_bits()), "{name}: V cache row != v (+ bias)");
            }
        }
    }

    /// rmsnorm, the Qwen3 qk_norm_rope and attn_decode against their WGSL twins — plus attention past
    /// the old 2048-token cap, which the chunked online softmax had never been run at before (the cap
    /// meant `s` could not exceed one 2048-key chunk).
    #[test]
    fn norm_rope_attention_match_wgsl() {
        let Some(ctx) = ctx_or_skip("norm_rope_attention_match_wgsl") else { return };
        let drv = driver().unwrap(); let k = *drv.decode_kernels().expect("decode ptx");
        let (nh, nkv, dh, s_len, eps, base, d) = (4usize, 2usize, 64usize, 37usize, 1e-6f32, 10000.0f32, 256usize);
        let up = |v: &[f32]| drv.upload_f32(v).unwrap();
        // rmsnorm
        let (xv, wv) = (rnd(d, 1), rnd(d, 2).iter().map(|v| 1.0 + v * 0.1).collect::<Vec<_>>());
        let want = pollster::block_on(crate::Tensor::from_vec(&ctx, &xv, &[1, d]).rmsnorm(&crate::Tensor::from_vec(&ctx, &wv, &[d]), eps).to_vec());
        let (mut xd, mut wd, mut od, mut d32, mut e) = (up(&xv), up(&wv), drv.alloc(d * 4).unwrap(), d as u32, eps);
        assert!(unsafe { drv.launch(k.rmsnorm, 1, 256, &mut p!(xd, wd, od, d32, e)) } && drv.sync());
        let mut got = vec![0f32; d]; assert!(drv.dtoh(&mut got, od)); close("rmsnorm", &want, &got, 1e-4);
        // Independent f64 reference: agreement here means an exact-zero vs WGSL is coincidence, not circularity.
        close("rmsnorm vs CPU f64 (independent)", &cpu_rmsnorm(&xv, &wv, eps), &got, 1e-5);
        // qk_norm_rope, rows == 1, the Qwen3 configuration (QK-norm, NEOX, no bias, no freq factors)
        let (q_out, kv_out) = (nh * dh, nkv * dh); let width = q_out + 2 * kv_out;
        let qkv = rnd(width, 3);
        let (qw, kw): (Vec<f32>, Vec<f32>) = (rnd(dh, 4).iter().map(|v| 1.0 + v * 0.1).collect(), rnd(dh, 5).iter().map(|v| 1.0 + v * 0.1).collect());
        let pos = 11usize;
        let src = crate::Tensor::from_vec(&ctx, &qkv, &[1, width]);
        let (wq, wk) = (crate::Tensor::from_vec(&ctx, &qw, &[dh]), crate::Tensor::from_vec(&ctx, &kw, &[dh]));
        let (q_ref, k_ref) = crate::Tensor::qk_norm_rope(&src, 0, q_out, &wq, &wk, 1, nh, nkv, dh, base, pos, eps);
        let (q_ref, k_ref) = (pollster::block_on(q_ref.to_vec()), pollster::block_on(k_ref.to_vec()));
        let (qg, kg, _, _) = run_rope(&qkv, Some(&qw), Some(&kw), None, None, false, nh, nkv, dh, base, pos, eps);
        close("qk_norm_rope q", &q_ref, &qg, 1e-4); close("qk_norm_rope k", &k_ref, &kg, 1e-4);
        // attention vs fused_decode_attention, and vs an f64 softmax at lengths across the chunk edge
        let attn = |nh: usize, nkv: usize, dh: usize, s: usize, seed: u64| {
            let (qo, kvo) = (nh * dh, nkv * dh);
            let (qv, kv, vv) = (rnd(qo, seed), rnd(s * kvo, seed + 1), rnd(s * kvo, seed + 2));
            let want = pollster::block_on(crate::Tensor::from_vec(&ctx, &qv, &[1, qo]).fused_decode_attention(
                &crate::Tensor::from_vec(&ctx, &kv, &[s, kvo]), &crate::Tensor::from_vec(&ctx, &vv, &[s, kvo]), nh, nkv, dh).to_vec());
            let (mut qd, mut kd, mut vd, mut ad) = (up(&qv), up(&kv), up(&vv), drv.alloc(qo * 4).unwrap());
            let (mut a, mut b, mut c, mut s32, mut sc) = (nh as u32, nkv as u32, dh as u32, s as u32, 1.0f32 / (dh as f32).sqrt());
            assert!(unsafe { drv.launch(k.attn_decode, nh as u32, 128, &mut p!(qd, kd, vd, ad, a, b, c, s32, sc)) } && drv.sync());
            let mut got = vec![0f32; qo]; assert!(drv.dtoh(&mut got, ad));
            let g = nh / nkv;
            let mut host = vec![0f64; qo];
            for h in 0..nh {
                let kvh = h / g;
                let sc: Vec<f64> = (0..s).map(|j| (0..dh).map(|t| qv[h * dh + t] as f64 * kv[j * kvo + kvh * dh + t] as f64).sum::<f64>() / (dh as f64).sqrt()).collect();
                let m = sc.iter().cloned().fold(f64::MIN, f64::max);
                let z: f64 = sc.iter().map(|v| (v - m).exp()).sum();
                for j in 0..s { let p = (sc[j] - m).exp() / z; for e in 0..dh { host[h * dh + e] += p * vv[j * kvo + kvh * dh + e] as f64; } }
            }
            unsafe { for p in [qd, kd, vd, ad] { (drv.cu_mem_free)(p); } }
            gate(&format!("attn_decode nh={nh} nkv={nkv} dh={dh} S={s}"), &got, &want, &host);
        };
        attn(nh, nkv, dh, s_len, 6);
        // The real head geometry (dh=128 -> 4 elements per lane) and a key count that leaves a
        // remainder across the 4 warps (133 = 4*33 + 1), so the warp-strided paths are hit.
        attn(16, 8, 128, 133, 16);
        // Past one 2048-key chunk: 2049 (one key into the second chunk) and 4500 (three chunks).
        attn(14, 2, 64, 2049, 26);
        attn(32, 8, 64, 4500, 36);
    }

    /// **The tensor-core GEMM, every format**, against an f64 host GEMM on the SAME f16-rounded inputs:
    /// A rounded f32 -> f16 and each dequantised weight rounded f32 -> f16 on the host, exactly as the
    /// kernel feeds `mma.sync`, so what remains is accumulation order and the gate can be tight. The
    /// unrounded f64 product is printed beside it — that gap is the f16 numerics trade, measured.
    /// M = 70 and N = 100 are not multiples of the 64x64 tile, so the edge guards are exercised.
    #[test]
    fn prefill_gemm_every_format_matches_f16_rounded_f64_host() {
        let Some(ctx) = ctx_or_skip("prefill_gemm_every_format_matches_f16_rounded_f64_host") else { return };
        let drv = driver().unwrap(); let pk = *drv.prefill_kernels().expect("prefill ptx");
        let h16 = |v: f32| half::f16::from_f32(v).to_f32() as f64;
        for (ci, &(f, k)) in [(QFmt::Q4K, 512usize), (QFmt::Q5K, 512), (QFmt::Q6K, 512), (QFmt::Q8_0, 896), (QFmt::Q5_0, 896), (QFmt::Q8_0, 96), (QFmt::Q5_0, 96)].iter().enumerate() {
            let (m, n) = (70usize, 100usize);
            let bytes = q_fixture(f, n, k, 0xF00D + ci as u64);
            let qm = crate::dtype::QMatrix::from_bytes(&ctx, &bytes, f.ggml_type(), n, k).expect("qmatrix");
            let w = qm.native_weight().expect("mirror");
            let a: Vec<f32> = rnd(m * k, 77 + ci as u64).iter().map(|v| v * 2.0).collect();
            let rb = bytes.len() / n;
            let wrows: Vec<Vec<f32>> = (0..n).map(|o| ferric_gguf::deq_raw(&bytes[o * rb..(o + 1) * rb], k, f.ggml_type()).unwrap()).collect();
            let (mut want, mut exact, mut mag) = (vec![0f64; m * n], vec![0f64; m * n], 0f64);
            for i in 0..m { for o in 0..n {
                let (mut s, mut x, mut g) = (0f64, 0f64, 0f64);
                for j in 0..k { let (av, wv) = (a[i * k + j], wrows[o][j]);
                    s += h16(av) * h16(wv); x += av as f64 * wv as f64; g += (h16(av) * h16(wv)).abs(); }
                want[i * n + o] = s; exact[i * n + o] = x; mag = mag.max(g);
            } }
            let (ad, cd, ovf) = (drv.upload_f32(&a).unwrap(), drv.alloc(m * n * 4).unwrap(), drv.upload_f32(&[0.0]).unwrap());
            assert!(unsafe { launch_gemm(drv, &pk, ad, k, w.dw(), cd, n, m, ovf) } && drv.sync(), "gemm launch");
            let mut got = vec![0f32; m * n]; assert!(drv.dtoh(&mut got, cd));
            let mut flag = [0f32]; assert!(drv.dtoh(&mut flag, ovf));
            unsafe { for p in [ad, cd, ovf] { (drv.cu_mem_free)(p); } }
            let (dr, dx) = (max_abs_diff(&got, &want), max_abs_diff(&got, &exact));
            let tol = 1e-5 * mag;
            eprintln!("{f:?} gemm {m}x{n}x{k}: max|Δ| vs f16-rounded f64 {dr:.3e} (tol {tol:.3e})   vs unrounded f64 {dx:.3e}   Σ|a·w| {mag:.3e}");
            assert!(flag[0].to_bits() == 0, "{f:?}: overflow flag raised on in-range inputs");
            assert!(got.iter().all(|v| v.is_finite()) && dr <= tol, "{f:?}: tensor-core GEMM diverges from the f16-rounded f64 host GEMM");
        }
        // The f16-range guard: one activation past 65504 must raise the flag (the host then runs WGSL).
        let (f, k, m, n) = (QFmt::Q8_0, 96usize, 3usize, 8usize);
        let bytes = q_fixture(f, n, k, 5);
        let qm = crate::dtype::QMatrix::from_bytes(&ctx, &bytes, f.ggml_type(), n, k).unwrap();
        let mut a = rnd(m * k, 9); a[k + 17] = 7.0e4;
        let (ad, cd, ovf) = (drv.upload_f32(&a).unwrap(), drv.alloc(m * n * 4).unwrap(), drv.upload_f32(&[0.0]).unwrap());
        assert!(unsafe { launch_gemm(drv, &pk, ad, k, qm.native_weight().unwrap().dw(), cd, n, m, ovf) } && drv.sync());
        let mut flag = [0f32]; assert!(drv.dtoh(&mut flag, ovf));
        assert!(flag[0].to_bits() != 0, "an activation of 7e4 did NOT raise the f16 overflow flag");
    }

    /// Causal prefill attention with a cache offset (`pos` earlier rows), GQA, row counts that leave
    /// part-empty 16-row query tiles and 32-row key tiles, against an f64 softmax.
    #[test]
    fn prefill_attention_matches_f64_causal_softmax() {
        if driver().is_none() { eprintln!("SKIPPED prefill_attention_matches_f64_causal_softmax: no CUDA driver / FERRIC_CUDA unset."); return; }
        let drv = driver().unwrap(); let pk = *drv.prefill_kernels().expect("prefill ptx");
        for &(nh, nkv, dh, t, pos) in &[(8usize, 2usize, 64usize, 45usize, 37usize), (4, 4, 128, 33, 0), (14, 2, 64, 20, 530), (32, 8, 64, 70, 3)] {
            let (qw, kw) = (nh * dh, nkv * dh); let s = pos + t;
            let (q, kc, vc) = (rnd(t * qw, 1 + t as u64), rnd(s * kw, 2 + s as u64), rnd(s * kw, 3 + s as u64));
            let mut want = vec![0f64; t * qw];
            for i in 0..t { for h in 0..nh {
                let kvh = h / (nh / nkv);
                let sc: Vec<f64> = (0..=pos + i).map(|j| (0..dh).map(|e| q[i * qw + h * dh + e] as f64 * kc[j * kw + kvh * dh + e] as f64).sum::<f64>() / (dh as f64).sqrt()).collect();
                let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
                let z: f64 = sc.iter().map(|v| (v - mx).exp()).sum();
                for (j, v) in sc.iter().enumerate() { let p = (v - mx).exp() / z;
                    for e in 0..dh { want[i * qw + h * dh + e] += p * vc[j * kw + kvh * dh + e] as f64; } }
            } }
            let (mut qd, mut kd, mut vd, mut od) = (drv.upload_f32(&q).unwrap(), drv.upload_f32(&kc).unwrap(), drv.upload_f32(&vc).unwrap(), drv.alloc(t * qw * 4).unwrap());
            let (mut a, mut b, mut c, mut tt, mut pp, mut scl) = (nh as u32, nkv as u32, dh as u32, t as u32, pos as u32, 1.0f32 / (dh as f32).sqrt());
            assert!(unsafe { drv.launch2(pk.attn_prefill, (t as u32).div_ceil(16), nh as u32, 128, &mut p!(qd, kd, vd, od, a, b, c, tt, pp, scl)) } && drv.sync());
            let mut got = vec![0f32; t * qw]; assert!(drv.dtoh(&mut got, od));
            unsafe { for p in [qd, kd, vd, od] { (drv.cu_mem_free)(p); } }
            let d = max_abs_diff(&got, &want);
            eprintln!("attn_prefill nh={nh} nkv={nkv} dh={dh} T={t} pos={pos}: max|Δ| vs f64 {d:.3e}");
            assert!(got.iter().all(|v| v.is_finite()) && d <= 2e-5, "attn_prefill diverges from the f64 causal softmax by {d:.3e}");
        }
    }

    /// The device K/V grows by doubling and CARRIES its rows: write, grow twice, read back bit-exact.
    #[test]
    fn devkv_growth_carries_rows() {
        if driver().is_none() { eprintln!("SKIPPED devkv_growth_carries_rows: no CUDA driver / FERRIC_CUDA unset."); return; }
        let (nl, w) = (3usize, 128usize);
        let mut kv = DevKv::new(nl, w).unwrap();
        assert!(kv.reserve(10) && kv.cap() >= 10);
        let rows = |il: usize, n: usize, s: u64| (rnd(n * w, 100 * il as u64 + s), rnd(n * w, 100 * il as u64 + s + 50));
        for il in 0..nl { let (k, v) = rows(il, 10, 1); assert!(kv.write_rows(il, 0, &k, &v)); }
        kv.len = 10;
        let cap0 = kv.cap();
        assert!(kv.reserve(cap0 + 1) && kv.cap() > cap0, "no growth");
        assert!(kv.reserve(5000) && kv.cap() >= 5000);
        for il in 0..nl {
            let (k, v) = rows(il, 10, 1);
            let (kr, vr) = kv.read_rows(il, 0, 10).expect("read");
            assert!(k.iter().zip(&kr).all(|(a, b)| a.to_bits() == b.to_bits()) && v.iter().zip(&vr).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "layer {il}: rows not carried across growth");
        }
        assert!(kv.read_rows(0, 5, 6).is_none(), "read past len must refuse");
    }
}
