//! **Tiled flash attention for prefill** (portable WGSL, f32 end to end) — the FlashAttention-2 schedule:
//! a workgroup owns a BLOCK of query rows, streams the keys through workgroup memory a tile at a time,
//! and keeps an online softmax per row, so every key/value row read from memory serves BM query rows
//! instead of one.
//!
//! # Why the per-query kernel stopped being enough
//!
//! `FLASH_ATTN_PREFILL_WGSL` (lib.rs) runs one workgroup per (head, query) and re-reads every key and
//! value row for every query: at T = 2,048 on Qwen2.5-0.5B that is 14 heads x 2,048 queries each
//! streaming ~1,024 K and V rows. It ran at ~0.4 TFLOP/s on the M5 Max, and once the prefill GEMMs moved
//! to the matrix units (`FERRIC_QGEMM`, 21-28 TFLOP/s) attention became most of a long prefill.
//!
//! # The schedule
//!
//! * **GQA rows.** Rows are flattened per KV head as `r = i·g + h` (query position `i`, head `h` of the
//!   group of `g = nh/nkv`), so one block holds `BM/g` positions of all `g` heads that share a K/V head —
//!   each K/V tile loaded serves the whole group. It also keeps the causal diagonal thin: a 64-row block
//!   spans ~9 positions at g = 7, not 64.
//! * **Per tile** (BN keys): K tile → workgroup memory; S = Q·Kᵀ on a register micro-tile per thread;
//!   causal mask (key `j` visible to row `r` iff `j <= off + i`); per-row max / rescale / exp / sum; V
//!   tile into the same buffer; O += P·V on a register micro-tile. Four barriers per tile.
//! * **Heaviest blocks first.** Causal blocks late in the sequence have the most keys; the grid is walked
//!   in reverse so the long ones start first and the short ones fill the tail.
//!
//! Tile sizes are chosen per head_dim from the DEVICE's workgroup-storage limit: 32 KiB (Metal, most
//! desktop WebGPU adapters) takes 64- or 32-row blocks; the 16 KiB WebGPU baseline gets smaller ones. The
//! shader is generated with the sizes as constants and the micro-tiles unrolled, so accumulators live in
//! registers (a 64-accumulator array spilled and ran 2-13x slower in the tiled GEMM — see lib.rs).
//!
//! Same contract as `flash_attention_prefill_at`: `q` [T, nh·dh] at positions `off..off+T`, `k`/`v`
//! [off+T, nkv·dh], causal. f32 throughout; the summation order differs from the per-query kernel, so
//! results agree to f32 rounding, not bit for bit (tested against float64 in `flash_tiled_tests`). Also
//! Gemma's extras through [`Tensor::flash_attention_prefill_opts`]: a sliding window, an attention-score
//! softcap, and head_dim up to 256 — shapes that previously ran only on the composed path.
//!
//! # What it buys (examples/flash_prefill_bench.rs, interleaved, min/median)
//!
//! Kernel alone, idle M3 Ultra: Qwen2.5-0.5B shapes 2.5x the per-query kernel at T = 512, 5.1x at 2,048
//! (3.4 TFLOP/s), 8.0x at 8,192 (5.4 TFLOP/s); Qwen3-0.6B / Qwen2.5-1.5B (head_dim 128) 2.2-3.0x;
//! Llama-3.2-1B 6.8x at 2,048. M5 Max under load 40-70: 1.9x / 4.6-5.2x / 6.9-7.2x. Whole-model portable
//! prefill (Qwen2.5-0.5B Q8_0, idle M3 Ultra): 1,384 -> 1,434 tok/s at 512, 1,237 -> 1,430 at 2,048,
//! 836 -> 1,349 at 8,192 — the portable GEMMs dominate below that.
//!
//! # ⛔ Workgroup zero-initialisation was the kernel's floor
//!
//! wgpu defaults to zero-initialising workgroup memory, and naga's MSL backend does it as
//! `if (local_invocation_index == 0) { qs = {}; kvs = {}; ... }` + a barrier: ONE thread clearing the whole
//! 26 KiB tile set while 127 wait, in every workgroup. Measured on the idle M3 Ultra: 0.376-0.378 ms per
//! dispatch with it, 0.057 without, at T = 2 (2 workgroups, trivial work) — the 0.3 ms "floor" that made the
//! tiled kernel lose to the per-query one on every small block. This kernel writes every workgroup location
//! before reading it, so its pipeline is built with the clear OFF (`pipeline`). ⚠ Every other WGSL kernel
//! still pays it through `pipeline_for` — the per-query flash and the decode attention kernels hold 9.2 KiB.

use crate::{empty, unibuf, Tensor};

/// One tile configuration: rows per block, keys per tile, and the two thread layouts (S phase over
/// BM x BN scores, PV phase over BM x dh/4 output vec4s), each `row_threads * col_threads == NT`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Cfg {
    pub bm: usize,
    pub bn: usize,
    srt: usize,
    sct: usize,
    prt: usize,
    pdt: usize,
}

const NT: usize = 128;

/// Split NT threads as rows x cols over an `rows x cols` grid so each thread gets a rectangular,
/// as-square-as-possible micro-tile. None if no power-of-two split divides both.
fn split(rows: usize, cols: usize) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize, usize)> = None;
    let mut ct = 1;
    while ct <= NT {
        let rt = NT / ct;
        if rows % rt == 0 && cols % ct == 0 {
            let (pr, pc) = (rows / rt, cols / ct);
            let score = pr.max(pc) - pr.min(pc);
            if best.is_none_or(|b| score < b.2) { best = Some((rt, ct, score)); }
        }
        ct *= 2;
    }
    best.map(|(r, c, _)| (r, c))
}

/// Workgroup bytes a configuration needs: Q tile + one K/V tile (both padded by one vec4 per row
/// against bank conflicts), the score tile (padded by one float per row), three per-row state arrays.
fn bytes(bm: usize, bn: usize, dh: usize) -> usize {
    let dhp4 = dh / 4 + 1;
    16 * (bm + bn) * dhp4 + 4 * (bm * (bn + 1) + 3 * bm)
}

impl Cfg {
    fn new(bm: usize, bn: usize, dh: usize) -> Option<Cfg> {
        let (srt, sct) = split(bm, bn)?;
        let (prt, pdt) = split(bm, dh / 4)?;
        Some(Cfg { bm, bn, srt, sct, prt, pdt })
    }
}

/// The configuration for this head_dim on a device granting `limit` bytes of workgroup storage, or
/// None (head_dim not a multiple of 4, above 128, or nothing fits) — the caller then keeps the
/// per-query kernel. `FERRIC_FLASH_TILE=BMxBN` forces a size, for measurement.
pub(crate) fn choose(dh: usize, limit: usize) -> Option<Cfg> {
    if dh == 0 || dh % 4 != 0 || dh > 256 { return None; }
    if let Some((bm, bn)) = std::env::var("FERRIC_FLASH_TILE").ok().and_then(|v| {
        let (a, b) = v.split_once('x')?;
        Some((a.parse().ok()?, b.parse().ok()?))
    }) {
        return Cfg::new(bm, bn, dh).filter(|_| bytes(bm, bn, dh) <= limit);
    }
    // Order of preference, the first that fits the device wins. Measured at T = 2,048 on the M5 Max under
    // load (examples/flash_prefill_bench.rs, FERRIC_FLASH_TILE): head_dim 64 at 64x16 2.41 TF/s against
    // 2.05 (32x16), 1.80 (64x8), 1.57 (32x32) for Qwen2.5-0.5B's GQA 7 — Llama-3.2-1B's GQA 4 preferred
    // 32x16 (2.07 vs 1.75), inside that run's noise; head_dim 128 at 32x16 (64x16 does not fit 32 KiB).
    let prefs: &[(usize, usize)] = if dh <= 64 {
        &[(64, 16), (32, 16), (32, 8), (16, 8)]
    } else if dh <= 128 {
        &[(32, 16), (32, 8), (16, 8)]
    } else {
        &[(16, 8)] // head_dim 256 (Gemma): 25.7 KiB, so 32 KiB devices only
    };
    prefs.iter().find_map(|&(bm, bn)| Cfg::new(bm, bn, dh).filter(|_| bytes(bm, bn, dh) <= limit))
}

/// A planted defect, for the tests' negative controls only (production always passes `Fault::None`): each
/// is a plausible way to get this kernel wrong, and the float64 test must see every one of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Fault {
    None,
    /// The online-softmax rescale dropped (`corr = 1`): exact whenever a row's max never moves after its
    /// first tile — which periodic test data can arrange by accident.
    NoRescale,
    /// The causal limit ignores the cache offset (`j <= i` instead of `j <= off + i`).
    CausalNoOffset,
    /// The ragged last key tile dropped (`k0 + BN <= nkeys`).
    DropLastTile,
}

/// Generate the WGSL for `cfg` at head_dim `dh`, micro-tiles unrolled.
pub(crate) fn wgsl(cfg: Cfg, dh: usize, fault: Fault) -> String {
    use std::fmt::Write;
    let (bm, bn) = (cfg.bm, cfg.bn);
    let (dh4, dhp4, bnp) = (dh / 4, dh / 4 + 1, bn + 1);
    let (srt, sct) = (cfg.srt, cfg.sct);
    let (sr, sc) = (bm / srt, bn / sct);
    let (prt, pdt) = (cfg.prt, cfg.pdt);
    let (pr, pd) = (bm / prt, dh4 / pdt);
    let mut s = String::new();
    let w = &mut s;
    let _ = write!(w, r#"// flash_tiled: BM={bm} BN={bn} dh={dh}  S micro-tile {sr}x{sc}  PV micro-tile {pr}x{pd} vec4
@group(0) @binding(0) var<storage,read>        q:   array<vec4<f32>>;   // [T, nh*dh]
@group(0) @binding(1) var<storage,read>        k:   array<vec4<f32>>;   // [off+T, nkv*dh]
@group(0) @binding(2) var<storage,read>        v:   array<vec4<f32>>;   // [off+T, nkv*dh]
@group(0) @binding(3) var<storage,read_write>  out: array<vec4<f32>>;   // [T, nh*dh]
struct Info {{ a: vec4<u32>, b: vec4<u32> }}   // a = (nh, nkv, T, off); b = (scale bits, blocks, window, softcap bits)
@group(0) @binding(4) var<uniform>             info: Info;
var<workgroup> qs:   array<vec4<f32>, {qn}>;
var<workgroup> kvs:  array<vec4<f32>, {kn}>;
var<workgroup> ps:   array<f32, {pn}>;
var<workgroup> mrow: array<f32, {bm}>;
var<workgroup> lrow: array<f32, {bm}>;
var<workgroup> crow: array<f32, {bm}>;
@compute @workgroup_size({NT})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let nh = info.a.x; let nkv = info.a.y; let T = info.a.z; let off = info.a.w;
    let scale = bitcast<f32>(info.b.x); let nblk = info.b.y;
    let win = info.b.z; let cap = bitcast<f32>(info.b.w);   // sliding window (0 = none), softcap (0 = none)
    let g = nh / nkv; let kvh = wg.y;
    let r0 = (nblk - 1u - wg.x) * {bm}u;          // heaviest (latest) blocks first
    let nrows = T * g;
    let i_hi = (min(r0 + {bm}u, nrows) - 1u) / g;  // last query position in this block
    let nkeys = off + i_hi + 1u;                  // causal: keys 0..=off+i_hi
    // Sliding window: no row of this block sees a key at or below off + i_lo - win, so whole tiles
    // below that are skipped (masking handles the partial one).
    let i_lo = min(r0, nrows - 1u) / g;
    var kfirst = 0u;
    if (win > 0u && off + i_lo + 1u > win) {{ kfirst = ((off + i_lo + 1u - win) / {bn}u) * {bn}u; }}
    for (var e = t; e < {bm}u * {dh4}u; e = e + {NT}u) {{
        let lr = e / {dh4}u; let c = e % {dh4}u; let r = r0 + lr;
        var val = vec4<f32>(0.0);
        if (r < nrows) {{ val = q[((r / g) * nh + kvh * g + r % g) * {dh4}u + c]; }}
        qs[lr * {dhp4}u + c] = val;
    }}
    if (t < {bm}u) {{ mrow[t] = -3.0e38; lrow[t] = 0.0; }}
    let sr0 = t / {sct}u; let sc0 = t % {sct}u;
    let pr0 = t / {pdt}u; let pd0 = t % {pdt}u;
"#, qn = bm * dhp4, kn = bn * dhp4, pn = bm * bnp);
    // Each S-phase row's causal limit, fixed for the whole key loop (rows past the end reuse the last).
    for a in 0..sr {
        let base = if fault == Fault::CausalNoOffset { "" } else { "off + " };
        let _ = writeln!(w, "    let lim{a} = {base}min(r0 + sr0 + {}u, nrows - 1u) / g;", a * srt);
    }
    for a in 0..pr {
        for b in 0..pd { let _ = writeln!(w, "    var o{a}_{b} = vec4<f32>(0.0);"); }
    }
    let cond = if fault == Fault::DropLastTile { format!("k0 + {bn}u <= nkeys") } else { "k0 < nkeys".to_string() };
    let corr = if fault == Fault::NoRescale { "1.0" } else { "exp(m_old - m_new)" };
    let _ = write!(w, r#"    for (var k0 = kfirst; {cond}; k0 = k0 + {bn}u) {{
        for (var e = t; e < {bn}u * {dh4}u; e = e + {NT}u) {{
            let lc = e / {dh4}u; let c = e % {dh4}u; let j = k0 + lc;
            var val = vec4<f32>(0.0);
            if (j < nkeys) {{ val = k[(j * nkv + kvh) * {dh4}u + c]; }}
            kvs[lc * {dhp4}u + c] = val;
        }}
        workgroupBarrier();
"#);
    for a in 0..sr {
        for b in 0..sc { let _ = writeln!(w, "        var s{a}_{b} = 0.0;"); }
    }
    let _ = writeln!(w, "        for (var d = 0u; d < {dh4}u; d = d + 1u) {{");
    for a in 0..sr { let _ = writeln!(w, "            let q{a} = qs[(sr0 + {}u) * {dhp4}u + d];", a * srt); }
    for b in 0..sc { let _ = writeln!(w, "            let k{b} = kvs[(sc0 + {}u) * {dhp4}u + d];", b * sct); }
    for a in 0..sr {
        for b in 0..sc { let _ = writeln!(w, "            s{a}_{b} = s{a}_{b} + dot(q{a}, k{b});"); }
    }
    let _ = writeln!(w, "        }}");
    for a in 0..sr {
        for b in 0..sc {
            // Scale, then Gemma-2's softcap (cap·tanh(x/cap), the composed path's order), then the causal and
            // window masks: key j is visible to a row at position p iff j <= p and (no window or p - j < win).
            let _ = writeln!(w, "        {{ let j = k0 + sc0 + {c}u; var x = s{a}_{b} * scale; if (cap > 0.0) {{ x = cap * tanh(x / cap); }}\n          \
                                 ps[(sr0 + {r}u) * {bnp}u + sc0 + {c}u] = select(x, -3.0e38, j > lim{a} || (win > 0u && j + win <= lim{a})); }}",
                             r = a * srt, c = b * sct);
        }
    }
    let _ = write!(w, r#"        workgroupBarrier();
        for (var e = t; e < {bn}u * {dh4}u; e = e + {NT}u) {{
            let lc = e / {dh4}u; let c = e % {dh4}u; let j = k0 + lc;
            var val = vec4<f32>(0.0);
            if (j < nkeys) {{ val = v[(j * nkv + kvh) * {dh4}u + c]; }}
            kvs[lc * {dhp4}u + c] = val;
        }}
        if (t < {bm}u) {{
            var mx = -3.0e38;
            for (var c = 0u; c < {bn}u; c = c + 1u) {{ mx = max(mx, ps[t * {bnp}u + c]); }}
            let m_old = mrow[t]; let m_new = max(m_old, mx);
            var sum = 0.0;
            for (var c = 0u; c < {bn}u; c = c + 1u) {{
                let sv = ps[t * {bnp}u + c];
                let p = select(exp(sv - m_new), 0.0, sv <= -1.0e38);   // masked keys contribute exactly 0
                ps[t * {bnp}u + c] = p; sum = sum + p;
            }}
            let corr = {corr};
            lrow[t] = lrow[t] * corr + sum; mrow[t] = m_new; crow[t] = corr;
        }}
        workgroupBarrier();
"#);
    for a in 0..pr {
        let _ = writeln!(w, "        let cf{a} = crow[pr0 + {}u];", a * prt);
        for b in 0..pd { let _ = writeln!(w, "        o{a}_{b} = o{a}_{b} * cf{a};"); }
    }
    let _ = writeln!(w, "        for (var c = 0u; c < {bn}u; c = c + 1u) {{");
    for b in 0..pd { let _ = writeln!(w, "            let v{b} = kvs[c * {dhp4}u + pd0 + {}u];", b * pdt); }
    for a in 0..pr {
        let _ = writeln!(w, "            let p{a} = ps[(pr0 + {}u) * {bnp}u + c];", a * prt);
        for b in 0..pd { let _ = writeln!(w, "            o{a}_{b} = o{a}_{b} + p{a} * v{b};"); }
    }
    let _ = writeln!(w, "        }}\n        workgroupBarrier();\n    }}");
    for a in 0..pr {
        let _ = writeln!(w, "    {{ let lr = pr0 + {}u; let r = r0 + lr;", a * prt);
        let _ = writeln!(w, "      if (r < nrows) {{ let inv = 1.0 / lrow[lr]; let ob = ((r / g) * nh + kvh * g + r % g) * {dh4}u;");
        for b in 0..pd { let _ = writeln!(w, "        out[ob + pd0 + {}u] = o{a}_{b} * inv;", b * pdt); }
        let _ = writeln!(w, "      }} }}");
    }
    let _ = writeln!(w, "}}");
    s
}

type Pipe = (wgpu::ComputePipeline, std::rc::Rc<wgpu::BindGroupLayout>);
thread_local! {
    // One pipeline per (device, config, head_dim, fault), built on first use.
    static PIPES: std::cell::RefCell<std::collections::HashMap<(usize, Cfg, usize, Fault), Pipe>> =
        std::cell::RefCell::new(Default::default());
}

/// Compile the kernel with workgroup-memory zero-initialisation OFF. `pipeline_for` leaves wgpu's default
/// (on), and naga's MSL backend implements it as `if (local_invocation_index == 0) { qs = {}; ... }` —
/// ONE thread clearing every workgroup array while the other 127 wait at a barrier, in every workgroup.
/// This kernel writes every workgroup location before reading it (Q and K/V tiles are loaded whole, every
/// score slot is written in the S phase, row state is set by `t < BM` before the first read), so the
/// clear is pure cost. `FERRIC_FLASH_ZERO_INIT=1` restores it, for the A/B.
fn pipeline(ctx: &crate::Context, src: &str) -> Pipe {
    let zero = std::env::var("FERRIC_FLASH_ZERO_INIT").is_ok_and(|v| !v.is_empty() && v != "0");
    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("flash_tiled"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(src)),
    });
    let p = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("flash_tiled"), layout: None, module: &module, entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions { zero_initialize_workgroup_memory: zero, ..Default::default() },
        cache: None,
    });
    let bgl = std::rc::Rc::new(p.get_bind_group_layout(0));
    (p, bgl)
}

/// Tiled prefill attention, or None when the shape is outside what the tiled kernel serves (head_dim
/// not a multiple of 4 or above 128, a GQA ratio that does not divide, too many row blocks for one
/// dispatch dimension, or no tile fitting this device's workgroup storage).
pub(crate) fn flash_tiled(q: &Tensor, k: &Tensor, v: &Tensor, nh: usize, nkv: usize, dh: usize, off: usize) -> Option<Tensor> {
    flash_tiled_fault(q, k, v, nh, nkv, dh, off, 0, 0.0, Fault::None)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn flash_tiled_fault(q: &Tensor, k: &Tensor, v: &Tensor, nh: usize, nkv: usize, dh: usize, off: usize,
                                window: usize, softcap: f32, fault: Fault) -> Option<Tensor> {
    let limit = q.ctx.device.limits().max_compute_workgroup_storage_size as usize;
    flash_tiled_cfg(q, k, v, nh, nkv, dh, off, window, softcap, fault, choose(dh, limit)?)
}

/// The kernel at an explicit tile configuration — how the tests reach the configurations a smaller device
/// (the 16 KiB WebGPU baseline) would choose.
#[allow(clippy::too_many_arguments)]
pub(crate) fn flash_tiled_cfg(q: &Tensor, k: &Tensor, v: &Tensor, nh: usize, nkv: usize, dh: usize, off: usize,
                              window: usize, softcap: f32, fault: Fault, cfg: Cfg) -> Option<Tensor> {
    let t = q.shape[0];
    if nkv == 0 || nh % nkv != 0 || t == 0 { return None; }
    let nblk = (t * (nh / nkv)).div_ceil(cfg.bm);
    if nblk > 65_535 || nkv > 65_535 { return None; }
    let ctx = &q.ctx;
    let (pipe, bgl) = PIPES.with(|m| {
        m.borrow_mut().entry((&ctx.device as *const wgpu::Device as usize, cfg, dh, fault))
            .or_insert_with(|| pipeline(ctx, &wgsl(cfg, dh, fault)))
            .clone()
    });
    let (q, k, v) = (q.contiguous(), k.contiguous(), v.contiguous());
    let out = empty(ctx, t * nh * dh);
    let scale = 1.0 / (dh as f32).sqrt();
    if std::env::var_os("FERRIC_CENSUS").is_some() { crate::census_bump("flash_tiled"); }
    crate::record_dispatch(ctx, "flash_tiled", &pipe, &bgl,
        &[q.buf.as_ref(), k.buf.as_ref(), v.buf.as_ref(), &out,
          &unibuf(ctx, &[nh as u32, nkv as u32, t as u32, off as u32, scale.to_bits(), nblk as u32, window as u32,
                         softcap.max(0.0).to_bits()])],
        (nblk as u32, nkv as u32, 1));
    Some(Tensor::from_parts(&q.ctx, out, vec![t, nh * dh]))
}

impl Tensor {
    /// Fused causal prefill attention with Gemma's extras — a sliding `window` (key j visible to the query
    /// at position p iff p - j < window; 0 = none), an attention-score `softcap` (cap·tanh(s/cap) after the
    /// 1/√dh scale; 0 = none) — and head_dim up to 256, on the tiled kernel (or, when the tensor-unit tier
    /// is on and the shape fits it, the matrix-unit kernel). Same block contract as
    /// [`Tensor::flash_attention_prefill_at`]. None when no fused kernel serves the shape on this device,
    /// or when `FERRIC_FLASH=rows` asks for main's routing — the caller then runs its composed path
    /// (`nn::causal_attention_win` / `nn::chunked_attention`), which materialises [nh, T, S] scores.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_prefill_opts(&self, k: &Tensor, v: &Tensor, nh: usize, nkv: usize, dh: usize, off: usize,
                                        window: usize, softcap: f32) -> Option<Tensor> {
        let t = self.shape[0];
        if k.numel() != (off + t) * nkv * dh || v.numel() != k.numel() { return None; }
        match forced() {
            Kernel::Rows => None,
            Kernel::Tiled => flash_tiled_fault(self, k, v, nh, nkv, dh, off, window, softcap, Fault::None),
            kernel => native_opts(self, k, v, nh, nkv, dh, off, window, softcap, kernel == Kernel::Auto)
                .or_else(|| flash_tiled_fault(self, k, v, nh, nkv, dh, off, window, softcap, Fault::None)),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn native_opts(q: &Tensor, k: &Tensor, v: &Tensor, nh: usize, nkv: usize, dh: usize, off: usize, window: usize, softcap: f32,
               auto: bool) -> Option<Tensor> {
    #[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
    { crate::native_attn::attn_opts(q, k, v, nh, nkv, dh, off, window, softcap, auto) }
    #[cfg(not(all(target_os = "macos", not(target_arch = "wasm32"))))]
    { let _ = (q, k, v, nh, nkv, dh, off, window, softcap, auto); None }
}

/// Which prefill-attention kernel [`route`] runs. `Auto` routes by device and size; the others force one
/// (for A/B and tests) and yield None where that kernel cannot serve the shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Auto,
    /// The per-query kernel (`FLASH_ATTN_PREFILL_WGSL`): one workgroup per (head, query).
    Rows,
    /// This module's tiled WGSL kernel.
    Tiled,
    /// The matrix-unit MSL kernel (`native_attn`, Metal with passthrough MSL only).
    Native,
}

/// `FERRIC_FLASH=rows|tiled|native` forces a kernel process-wide (read once, like `FERRIC_QGEMM`: flipping
/// it mid-process would mix numerics). Unset, empty or anything else: `Auto`.
fn forced() -> Kernel {
    static K: std::sync::OnceLock<Kernel> = std::sync::OnceLock::new();
    *K.get_or_init(|| match std::env::var("FERRIC_FLASH").as_deref() {
        Ok("rows") => Kernel::Rows,
        Ok("tiled") => Kernel::Tiled,
        Ok("native") => Kernel::Native,
        _ => Kernel::Auto,
    })
}

/// Run prefill attention on `kernel` (`Auto`: by device and size), or None — the caller then runs the
/// per-query kernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn route(q: &Tensor, k: &Tensor, v: &Tensor, nh: usize, nkv: usize, dh: usize, off: usize, kernel: Kernel) -> Option<Tensor> {
    let t = q.shape[0];
    assert_eq!(k.numel(), (off + t) * nkv * dh, "flash prefill: keys must be the {off} cached + {t} new rows");
    let kernel = if kernel == Kernel::Auto { forced() } else { kernel };
    match kernel {
        Kernel::Rows => None,
        Kernel::Tiled => flash_tiled(q, k, v, nh, nkv, dh, off),
        Kernel::Native => native(q, k, v, nh, nkv, dh, off, false),
        Kernel::Auto => native(q, k, v, nh, nkv, dh, off, true).or_else(|| {
            if nkv > 0 && tiled_pays(t, nh, nkv, off) { flash_tiled(q, k, v, nh, nkv, dh, off) } else { None }
        }),
    }
}

/// Whether the tiled kernel is worth taking at this size. It parallelises over `nkv · ⌈T·g/BM⌉` row blocks
/// where the per-query kernel has `nh · T` workgroups, so a short block continuing a long cache (a
/// prefix-cache suffix, a small chunk) leaves most of the GPU idle on the tiled kernel while each of its
/// few workgroups walks the whole cache. Measured (M3 Ultra, idle, `examples/flash_prefill_bench.rs`
/// BENCH_SWEEP=1, T = 2..256 at off 0 and 2000, three GQA shapes): with a fresh cache (off = 0) tiled ties
/// or wins at every T; continuing a 2,000-key cache it lost 0.5-0.8x up to T·g = 224 rows per KV head and
/// won 1.2-5.2x from 256. The M5 Max (contended) agreed at every point the rule decides.
fn tiled_pays(t: usize, nh: usize, nkv: usize, off: usize) -> bool {
    off <= t || t * (nh / nkv) >= 256
}

#[allow(clippy::too_many_arguments)]
fn native(q: &Tensor, k: &Tensor, v: &Tensor, nh: usize, nkv: usize, dh: usize, off: usize, auto: bool) -> Option<Tensor> {
    #[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
    { crate::native_attn::attn(q, k, v, nh, nkv, dh, off, auto) }
    #[cfg(not(all(target_os = "macos", not(target_arch = "wasm32"))))]
    { let _ = (q, k, v, nh, nkv, dh, off, auto); None }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) mod flash_tiled_tests {
    use super::*;
    use ferric_core::Context;
    use std::sync::Arc;

    /// xorshift: NON-periodic data. A sin() wave repeats, and periodic keys let a kernel that never rescales
    /// its running softmax pass (the CUDA tier's no-rescale mutation did exactly that).
    pub(crate) struct Rng(pub u64);
    impl Rng {
        pub(crate) fn unit(&mut self) -> f32 {
            self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
        }
    }

    /// Queries, keys and values with a DRIFT: every query leans along dimension 0 and the keys' dimension 0
    /// climbs with position, so each row's max score keeps moving to later key tiles — the online-softmax
    /// rescale is exercised on every tile, not once.
    pub(crate) fn data(t: usize, off: usize, nh: usize, nkv: usize, dh: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut r = Rng(seed);
        let s = off + t;
        let mut q: Vec<f32> = (0..t * nh * dh).map(|_| r.unit()).collect();
        for row in 0..t * nh { q[row * dh] += 2.0; }
        let mut k: Vec<f32> = (0..s * nkv * dh).map(|_| r.unit()).collect();
        let mut v: Vec<f32> = (0..s * nkv * dh).map(|_| r.unit()).collect();
        for j in 0..s {
            for h in 0..nkv {
                k[(j * nkv + h) * dh] += 12.0 * j as f32 / s as f32 - 4.0;
                v[(j * nkv + h) * dh + 1] += 2.0 * j as f32 / s as f32;
            }
        }
        (q, k, v)
    }

    /// Causal attention for queries at positions `off..off+t` over `off+t` keys, in float64 on the host.
    pub(crate) fn reference(q: &[f32], k: &[f32], v: &[f32], t: usize, off: usize, nh: usize, nkv: usize, dh: usize) -> Vec<f64> {
        reference_opts(q, k, v, t, off, nh, nkv, dh, 0, 0.0)
    }

    /// [`reference`] with a sliding window (key j visible to position p iff p - j < window; 0 = none) and a
    /// softcap (score = cap·tanh(score/cap) after scaling; 0 = none) — the definitions in nn.rs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reference_opts(q: &[f32], k: &[f32], v: &[f32], t: usize, off: usize, nh: usize, nkv: usize, dh: usize,
                                 window: usize, cap: f64) -> Vec<f64> {
        let (g, scale) = (nh / nkv, 1.0 / (dh as f64).sqrt());
        let mut out = vec![0f64; t * nh * dh];
        let mut e = vec![0f64; off + t];
        for i in 0..t {
            for h in 0..nh {
                let kh = h / g;
                let qr = &q[(i * nh + h) * dh..][..dh];
                let p = off + i;
                let j0 = if window > 0 && p + 1 > window { p + 1 - window } else { 0 };
                let mut m = f64::MIN;
                for (j, ej) in e.iter_mut().enumerate().take(p + 1).skip(j0) {
                    let kr = &k[(j * nkv + kh) * dh..][..dh];
                    *ej = qr.iter().zip(kr).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() * scale;
                    if cap > 0.0 { *ej = cap * (*ej / cap).tanh(); }
                    m = m.max(*ej);
                }
                let mut z = 0f64;
                for ej in e.iter_mut().take(p + 1).skip(j0) { *ej = (*ej - m).exp(); z += *ej; }
                let o = &mut out[(i * nh + h) * dh..][..dh];
                for (j, ej) in e.iter().enumerate().take(p + 1).skip(j0) {
                    let vr = &v[(j * nkv + kh) * dh..][..dh];
                    for d in 0..dh { o[d] += ej * vr[d] as f64; }
                }
                for od in o.iter_mut() { *od /= z; }
            }
        }
        out
    }

    pub(crate) fn max_err(got: &[f32], want: &[f64]) -> f64 {
        assert_eq!(got.len(), want.len());
        got.iter().zip(want).map(|(a, b)| if a.is_finite() { (*a as f64 - b).abs() } else { f64::INFINITY }).fold(0f64, f64::max)
    }

    /// Shapes crossing every boundary the kernel has: GQA ratios 1/2/7/8 (7 does not divide a 64-row
    /// block, so blocks straddle query positions), ragged T, offsets on, before and after a key-tile
    /// edge and past the per-query kernel's 2048-key chunk, T = 1, and head_dim 64 / 80 / 128.
    pub(crate) const SHAPES: &[(usize, usize, usize, usize, usize)] = &[
        // (T, off, nh, nkv, dh)
        (37, 0, 4, 4, 64),
        (40, 2100, 8, 4, 64),
        (150, 1900, 7, 1, 64),
        (65, 15, 16, 2, 128),
        (33, 16, 16, 2, 128),
        (47, 17, 14, 2, 64),
        (1, 511, 14, 2, 64),
        (129, 0, 7, 1, 128),
        (70, 31, 8, 1, 64),
        (21, 5, 6, 2, 80),
    ];

    /// The tiled kernel against float64, gated at 4x the per-query kernel's own f32 distance on the same
    /// data (floored at 1e-6): the portable path's noise floor, measured, not assumed. Then each planted
    /// defect must miss by >= 20x that tolerance on at least the shapes it can affect — or this test could
    /// not have seen it.
    #[test]
    fn tiled_prefill_matches_float64_and_sees_every_planted_defect() {
        let Ok(ctx) = pollster::block_on(Context::new()) else { eprintln!("no GPU — skipping"); return; };
        let ctx = Arc::new(ctx);
        let limit = ctx.device.limits().max_compute_workgroup_storage_size as usize;
        let mut worst_fault = [f64::INFINITY; 3];
        // Every shape on this device's tile AND on the one the 16 KiB WebGPU baseline would choose.
        let cases = SHAPES.iter().enumerate().flat_map(|(n, &s)| {
            let dev = choose(s.4, limit).expect("a tile fits every tested head_dim");
            let base = choose(s.4, 16 * 1024).expect("a 16 KiB tile exists for every tested head_dim");
            std::iter::once((n, s, dev)).chain((base != dev).then_some((n, s, base)))
        });
        for (n, (t, off, nh, nkv, dh), cfg) in cases {
            let (q, k, v) = data(t, off, nh, nkv, dh, 0x9e37_79b9 + n as u64);
            let want = reference(&q, &k, &v, t, off, nh, nkv, dh);
            let qt = Tensor::from_vec(&ctx, &q, &[t, nh * dh]);
            let kt = Tensor::from_vec(&ctx, &k, &[off + t, nkv * dh]);
            let vt = Tensor::from_vec(&ctx, &v, &[off + t, nkv * dh]);
            let rows = pollster::block_on(qt.flash_attention_prefill_with(&kt, &vt, nh, nkv, dh, off, Kernel::Rows).unwrap().to_vec());
            let e_rows = max_err(&rows, &want);
            let tol = 4.0 * e_rows.max(1e-6);
            let run = |f: Fault| pollster::block_on(flash_tiled_cfg(&qt, &kt, &vt, nh, nkv, dh, off, 0, 0.0, f, cfg).expect("tiled").to_vec());
            let e = max_err(&run(Fault::None), &want);
            let faults = [Fault::NoRescale, Fault::CausalNoOffset, Fault::DropLastTile];
            let ef: Vec<f64> = faults.iter().map(|&f| max_err(&run(f), &want)).collect();
            eprintln!("T={t:<4} off={off:<5} nh={nh:<2} nkv={nkv} dh={dh:<3} {}x{}: tiled {e:.2e}  per-query {e_rows:.2e}  \
                       tol {tol:.1e}  | faults {:.1e} {:.1e} {:.1e}", cfg.bm, cfg.bn, ef[0], ef[1], ef[2]);
            assert!(e <= tol, "T={t} off={off} nh={nh} nkv={nkv} dh={dh}: tiled {e:.3e} > {tol:.3e}");
            // Each defect is only visible where its mechanism engages: a rescale needs more than one key
            // tile, an offset needs off > 0, a dropped ragged tile needs keys % BN != 0.
            let engages = [off + t > cfg.bn, off > 0, (off + 1..=off + t).any(|s| s % cfg.bn != 0)];
            for i in 0..3 {
                if engages[i] { worst_fault[i] = worst_fault[i].min(ef[i] / tol); }
            }
        }
        for (i, name) in ["no rescale", "causal offset ignored", "ragged tile dropped"].iter().enumerate() {
            eprintln!("negative control '{name}': worst miss {:.0}x tolerance", worst_fault[i]);
        }
        // ≥ 20x on EVERY shape where the mechanism engages, not on the best one.
        for (i, name) in ["no rescale", "causal offset ignored", "ragged tile dropped"].iter().enumerate() {
            assert!(worst_fault[i] >= 20.0, "negative control '{name}' missed by only {:.1}x — this test cannot see it", worst_fault[i]);
        }
    }

    /// Gemma's extras on the tiled kernel — sliding window, attention softcap, head_dim 256 — against float64,
    /// gated at 4x the COMPOSED path's own f32 distance on the same data (`nn::causal_attention_win`
    /// computes the same function unfused; it is what these shapes ran on before). Windows cross key-tile
    /// edges and the skipped-tile start; blocks continue a cache. Negative controls, each ≥ 20x the tolerance
    /// on every shape where it engages: the window ignored, the softcap ignored, the window one key too wide.
    /// ⚠ The caps are small (2.5-4) on purpose: at Gemma-2's 50, scores of a few units sit where cap·tanh(x/cap)
    /// ≈ x, and dropping the softcap moved the output by LESS than the fp16 bound — a control that could not fail.
    #[test]
    fn window_and_softcap_match_float64() {
        let Ok(ctx) = pollster::block_on(Context::new()) else { eprintln!("no GPU — skipping"); return; };
        let ctx = Arc::new(ctx);
        // (T, off, nh, nkv, dh, window, softcap)
        let shapes: &[(usize, usize, usize, usize, usize, usize, f32)] = &[
            (300, 0, 8, 2, 64, 64, 0.0),
            (100, 700, 14, 2, 64, 129, 4.0),
            (70, 40, 4, 1, 256, 33, 3.0),
            (129, 0, 8, 4, 256, 0, 0.0),
            (65, 450, 4, 1, 256, 512, 0.0),
            (48, 16, 16, 8, 128, 17, 2.5),
        ];
        let mut worst = [f64::INFINITY; 3];
        for (n, &(t, off, nh, nkv, dh, win, cap)) in shapes.iter().enumerate() {
            let (q, k, v) = data(t, off, nh, nkv, dh, 0x5eed + n as u64);
            let want = reference_opts(&q, &k, &v, t, off, nh, nkv, dh, win, cap as f64);
            let qt = Tensor::from_vec(&ctx, &q, &[t, nh * dh]);
            let kt = Tensor::from_vec(&ctx, &k, &[off + t, nkv * dh]);
            let vt = Tensor::from_vec(&ctx, &v, &[off + t, nkv * dh]);
            let comp = pollster::block_on(crate::nn::causal_attention_win(&qt, &kt, &vt, nh, nkv, win, cap).to_vec());
            let tol = 4.0 * max_err(&comp, &want).max(1e-6);
            let run = |w: usize, c: f32| pollster::block_on(
                flash_tiled_fault(&qt, &kt, &vt, nh, nkv, dh, off, w, c, Fault::None).expect("tiled").to_vec());
            let e = max_err(&run(win, cap), &want);
            let ctl = [max_err(&run(0, cap), &want), max_err(&run(win, 0.0), &want), max_err(&run(win + 1, cap), &want)];
            eprintln!("T={t:<4} off={off:<4} nh={nh:<2} nkv={nkv} dh={dh:<3} win={win:<4} cap={cap:<4}: tiled {e:.2e}  composed {:.2e}  \
                       | window ignored {:.1e}  softcap ignored {:.1e}  window+1 {:.1e}", tol / 4.0, ctl[0], ctl[1], ctl[2]);
            assert!(e <= tol, "T={t} off={off} dh={dh} win={win} cap={cap}: tiled {e:.3e} > {tol:.3e}");
            let engages = [win > 0 && off + t > win, cap > 0.0, win > 0 && off + t > win];
            for i in 0..3 { if engages[i] { worst[i] = worst[i].min(ctl[i] / tol); } }
        }
        for (i, name) in ["window ignored", "softcap ignored", "window one key too wide"].iter().enumerate() {
            eprintln!("negative control '{name}': worst miss {:.0}x tolerance", worst[i]);
            assert!(worst[i] >= 20.0, "negative control '{name}' missed by only {:.1}x — this test cannot see it", worst[i]);
        }
    }

    /// The routed entry point takes the tiled kernel where it pays and the per-query kernel where it does
    /// not: a test of the ROUTE, since a kernel verified in isolation buys nothing if the model never
    /// dispatches it — and a size rule that never declines is no rule. Both branches are checked by
    /// comparing the routed output's BITS to each kernel's (they differ in summation order, asserted).
    #[test]
    fn auto_route_takes_the_tiled_kernel_where_it_pays() {
        let Ok(ctx) = pollster::block_on(Context::new()) else { return; };
        let ctx = Arc::new(ctx);
        if std::env::var_os("FERRIC_FLASH").is_some() || std::env::var_os("FERRIC_QGEMM").is_some() { return; }
        // (T, off, expect tiled): a fresh 96-token prefill; a 16-token block over a 2,000-key cache
        // (T·g = 112 < 256: few row blocks, each walking the whole cache — the per-query kernel wins).
        for (t, off, tiled_expected) in [(96usize, 0usize, true), (16, 2000, false), (40, 2000, true)] {
            let (nh, nkv, dh) = (14usize, 2usize, 64usize);
            let (q, k, v) = data(t, off, nh, nkv, dh, 7);
            let qt = Tensor::from_vec(&ctx, &q, &[t, nh * dh]);
            let kt = Tensor::from_vec(&ctx, &k, &[off + t, nkv * dh]);
            let vt = Tensor::from_vec(&ctx, &v, &[off + t, nkv * dh]);
            let auto = pollster::block_on(qt.flash_attention_prefill_at(&kt, &vt, nh, nkv, dh, off).to_vec());
            let tiled = pollster::block_on(qt.flash_attention_prefill_with(&kt, &vt, nh, nkv, dh, off, Kernel::Tiled).unwrap().to_vec());
            let rows = pollster::block_on(qt.flash_attention_prefill_with(&kt, &vt, nh, nkv, dh, off, Kernel::Rows).unwrap().to_vec());
            assert!(tiled != rows, "T={t} off={off}: the kernels agree to the bit — the comparison cannot tell them apart");
            if tiled_expected {
                assert!(auto == tiled, "T={t} off={off}: the route did not take the tiled kernel");
            } else {
                assert!(auto == rows, "T={t} off={off}: the route took the tiled kernel where it loses");
            }
        }
    }
}
