//! **Small-M quantized GEMV: a forward of 2-32 rows dequantizes each weight block once per tile.**
//!
//! # Why
//!
//! The split-K GEMV kernels in `dtype.rs` give every (row, output) pair its own lanes, so a forward of M
//! rows is M independent GEMVs: every weight word is loaded and dequantized M times, and every activation
//! word is re-loaded once per OUTPUT. Above two rows the wide matrices also switch to the `flat` prefill
//! kernel. So a 2-9 row forward — prompt lookup's verify step, a small continuous batch — cost several
//! decode steps (`examples/decode_rows.rs`), and prompt lookup at 7.3 accepted tokens per forward was no
//! faster than the plain loop.
//!
//! Measured first (M3 Ultra, `examples/small_m_bench.rs`), and it corrected the premise: at a 0.5B model's
//! shapes these kernels are NOT DRAM-bound — a 1-row LM head streams 145 MB at 318 GB/s of an 800 GB/s
//! part, and the per-layer matrices (1-10 MB) stay in cache across rows. The cost of the M rows is
//! INSTRUCTIONS: loads and dequantization repeated per row, and the activation re-read per output. The
//! first version of this file moved the row loop inside the one-row kernel's lane and lost on every
//! k-quant shape (Q4_K `ffn_down` 0.54x at M=2): same instruction count per row, 1/M of the workgroups.
//!
//! So each lane group here computes an `R x M` register tile — `R` outputs by `M` rows. Per weight word it
//! loads and dequantizes `R` words once and reuses each for `M` rows; per activation word it loads `M`
//! values once and reuses each for `R` outputs. llama.cpp's `mul_mv_ext` (r1ptg 2-5 rows per threadgroup)
//! and MLX's `qmv` for small batch make the same trade; they are read as the idea, not as oracles. `R`
//! and the rows per workgroup are picked per shape ([`plan`]) so the grid never starves: rows split into
//! z-chunks when outputs alone would leave too few workgroups.
//!
//! # Bit-identity is the design constraint
//!
//! Tiling changes which lane holds which (row, output) accumulator, never the arithmetic an accumulator
//! sees. Each kernel keeps the one-row split-K kernel's lane layout (`__L__` lanes per output, lane `bl`
//! walking blocks `bl, bl + L, ...` in order), each accumulator applies the one-row kernel's expression
//! in the one-row kernel's order, and every accumulator is reduced by the one-row kernel's barrier tree.
//! So a row's result is bit-identical to the one-row kernel's, whatever rows share the dispatch and
//! whatever tile it lands in (`tests::small_m_rows_are_bit_identical_to_one_row_decode`). That is what
//! makes a batched decode step equal solo decode to the bit, and a prompt-lookup verify row equal the
//! plain loop's step in every matmul. It is also why this declines whenever the one-row path would NOT
//! be the default split-K kernel (a forced `flat`, `FERRIC_SUBBLK`, subgroup reduction, the transposed
//! Q4_K layouts): reproducing that path is the contract, so where it changes this steps aside.
//!
//! # Where it applies
//!
//! Q8_0, Q5_0, Q4_K, Q5_K, Q6_K matmuls and the fused Q4_K/Q5_K/Q6_K gate|up+SwiGLU, for
//! `2 <= rows <= FERRIC_MR_MAX` (default 32). With `FERRIC_QGEMM` on, the tensor-unit route takes
//! `rows >= FERRIC_QGEMM_MIN_ROWS` first; this fills `[2, MIN_ROWS)`. `FERRIC_MR=0` (or
//! [`set_small_m`]`(false)`) restores the previous kernels for an A/B in one binary.

use super::*;
use std::sync::atomic::{AtomicU8, Ordering};

static SWITCH: AtomicU8 = AtomicU8::new(0); // 0 = follow FERRIC_MR, 1 = on, 2 = off
static TILE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0); // (R << 16) | M, 0 = plan

/// Force every small-M dispatch to an `(R outputs, M rows)` tile, or `None` for the planner's choice.
/// A sweep hook (`examples/small_m_bench.rs`): the planner's thresholds come from what it measures.
pub fn set_small_m_tile(t: Option<(usize, usize)>) {
    TILE.store(t.map_or(0, |(r, m)| ((r.clamp(1, 8) as u32) << 16) | m.clamp(1, 32) as u32), Ordering::Relaxed);
}

/// Turn the small-M kernels on or off for this process, overriding `FERRIC_MR`. Exists so a benchmark
/// can interleave both arms in ONE process — on a shared machine two launches are not comparable.
pub fn set_small_m(on: bool) { SWITCH.store(if on { 1 } else { 2 }, Ordering::Relaxed); }

fn enabled() -> bool {
    match SWITCH.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var("FERRIC_MR").map(|v| v != "0").unwrap_or(true))
        }
    }
}

/// The most rows routed here (`FERRIC_MR_MAX`, default 32 — the tensor-unit route's floor).
fn max_rows() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("FERRIC_MR_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(32))
}

/// The row window: `[2, FERRIC_MR_MAX]`, and below the tensor-unit route's floor when that route is on
/// (it is tried first and takes `rows >= FERRIC_QGEMM_MIN_ROWS`; this fills the gap under it).
fn in_window(ctx: &Context, rows: usize) -> bool {
    #[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
    let cap = if crate::native_qgemm::enabled(ctx) { max_rows().min(crate::native_qgemm::min_rows().saturating_sub(1)) } else { max_rows() };
    #[cfg(not(all(target_os = "macos", not(target_arch = "wasm32"))))]
    let cap = { let _ = ctx; max_rows() };
    enabled() && rows >= 2 && rows <= cap && std::env::var_os("FERRIC_SUBBLK").is_none()
}

/// Whether `rows` rows of an `[n_out, in_dim]` weight go through a small-M kernel. Only when the
/// ONE-row path is the default split-K kernel, because that is the kernel these reproduce.
fn eligible(ctx: &Context, rows: usize, n_out: usize, in_dim: usize) -> bool {
    in_window(ctx, rows) && q2_0_split_k(1, n_out, in_dim) && !use_subgroup(ctx)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub(crate) enum Fmt { Q8_0, Q5_0, Q4K, Q5K, Q6K }

/// One dispatch's shape: `r` outputs per lane group, `m` rows per workgroup (z-chunks cover the rest).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Plan { pub r: usize, pub m: usize }

/// Pick the tile for `rows` rows over `o_dim` outputs of a `wbytes`-byte weight. Fitted to
/// `examples/small_m_sweep.rs` (M3 Ultra, quiet; 10 shapes x 5 formats x M in 2..16, 13 tiles each):
///   * `R = 2` outputs per lane group for the 32-value formats (Q8_0/Q5_0) when that still leaves >= 256
///     lane groups — 2x2..2x4 won `gate_up`/`qkv`/`ffn_down` by 1.2-3.1x; the k-quants and the fused
///     gate|up (whose lane group already carries two weight rows) did best at `R = 1`;
///   * rows in balanced chunks of about 8 accumulators per lane — a chunk re-reads the weights, but on
///     per-layer matrices that stay cached the extra workgroups pay more than the re-read costs;
///   * EXCEPT a weight too big to stay cached (the LM head, 145 MB): a chunk there is a DRAM pass, so up
///     to 16 rows go in one (M=9: one chunk 1.27 ms, 8+1 chunks 2.05 ms).
/// Balanced means 9 rows run as 5+4, never 8+1: a one-row chunk costs a whole weight pass for one row.
pub(crate) fn plan(fmt: Fmt, rows: usize, o_dim: usize, swiglu: bool, wbytes: usize) -> Plan {
    let forced = TILE.load(Ordering::Relaxed);
    if forced != 0 { return Plan { r: (forced >> 16) as usize, m: ((forced & 0xffff) as usize).clamp(1, rows) }; }
    let r = if matches!(fmt, Fmt::Q8_0 | Fmt::Q5_0) && !swiglu && o_dim.div_ceil(2) >= 256 { 2 } else { 1 };
    let m_target = if wbytes > 32 << 20 { 16 } else { 8 / r };
    let chunks = rows.div_ceil(m_target);
    Plan { r, m: rows.div_ceil(chunks) }
}

/// **The kernel source, generated fully unrolled.** Every (output, row) accumulator is its own named
/// variable and every per-output / per-row value its own `let`, so nothing is an indexed array.
///
/// ⛔ MEASURED, and why: the first tiled version kept `var acc: array<f32, R*M>` and looped over `j`
/// and `r`. naga emits every WGSL loop with a loop-bounding counter (`force_loop_bounding`, on by
/// default in wgpu's Metal path), which stops the Metal compiler from fully unrolling it, so the
/// arrays were indexed dynamically and lived in thread memory: the 4x8 tile ran the LM head 2x SLOWER
/// than one GEMV per row (M3 Ultra, 4.19 vs 2.15 ms at M=8). Generating the unrolled body in Rust
/// puts every accumulator in a register on every backend, whatever its loop policy.
fn source(fmt: Fmt, swiglu: bool, p: Plan, lanes: u32, opw: u32) -> String {
    use std::fmt::Write;
    let (rr, mm) = (p.r, p.m);
    let nr = if swiglu { 2 * rr } else { rr };
    let xt = if fmt == Fmt::Q6K { "f32" } else { "vec4<f32>" };
    let helpers = match fmt { Fmt::Q4K | Fmt::Q5K => Q4_K_HELPERS, Fmt::Q6K => Q6_K_HELPERS, _ => "" };
    let mut s = String::new();
    let js = || 0..nr;
    let rs = || 0..mm;
    let jr = || (0..nr).flat_map(move |j| (0..mm).map(move |r| (j, r)));
    let _ = writeln!(s, "@group(0) @binding(0) var<storage,read> x: array<{xt}>;
@group(0) @binding(1) var<storage,read> codes: array<u32>;
@group(0) @binding(2) var<storage,read> aux: array<u32>;
@group(0) @binding(3) var<storage,read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> info: vec4<u32>;   // rows, o_dim, in_dim, grid_w
var<workgroup> partial: array<f32, {}>;
{helpers}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let rows = info.x; let o_dim = info.y; let in_dim = info.z;
    let t = lid.x; let bl = t % {lanes}u;
    let o0 = ((wg.x + wg.y * info.w) * {opw}u + t / {lanes}u) * {rr}u;
    let r0 = wg.z * {mm}u;", 64 * nr * mm);
    // Loads CLAMP to a valid output/row (duplicate work, discarded at the store): no branches in the body.
    for j in js() {
        let up = if j >= rr { " + o_dim" } else { "" };
        let _ = writeln!(s, "    let or{j} = min(o0 + {}u, o_dim - 1u){up};", j % rr);
    }
    for r in rs() { let _ = writeln!(s, "    let xr{r} = min(r0 + {r}u, rows - 1u) * in_dim;"); }
    for (j, r) in jr() { let _ = writeln!(s, "    var a{j}_{r} = 0.0;"); }
    match fmt {
        // Per word `w` of a 32-value block (lane t: w = t, t+64, ...): `acc + d·dot(x, v)`, MATMUL_Q8_0_SPLITK.
        Fmt::Q8_0 => {
            s.push_str("    let nblk = in_dim / 32u; let nwords = nblk * 8u;
    for (var w: u32 = t; w < nwords; w = w + 64u) {
        let blk = w >> 3u; let xo = (blk * 32u + (w & 7u) * 4u) >> 2u;\n");
            for j in js() {
                let _ = writeln!(s, "        let bi{j} = or{j} * nblk + blk; let sw{j} = unpack2x16float(aux[bi{j} >> 1u]);
        let d{j} = select(sw{j}.y, sw{j}.x, (bi{j} & 1u) == 0u); let c{j} = codes[or{j} * nwords + w];
        let v{j} = vec4<f32>(f32(i32(c{j} << 24u) >> 24u), f32(i32(c{j} << 16u) >> 24u), f32(i32(c{j} << 8u) >> 24u), f32(i32(c{j}) >> 24u));");
            }
            for r in rs() { let _ = writeln!(s, "        let x{r} = x[(xr{r} >> 2u) + xo];"); }
            for (j, r) in jr() { let _ = writeln!(s, "        a{j}_{r} = a{j}_{r} + d{j} * dot(x{r}, v{j});"); }
            s.push_str("    }\n");
        }
        // `acc + (dot(x_lo, lo) + dot(x_hi, hi))·d`, MATMUL_Q5_0_SPLITK.
        Fmt::Q5_0 => {
            s.push_str("    let nblk = in_dim / 32u; let nwords = nblk * 4u;
    for (var w: u32 = t; w < nwords; w = w + 64u) {
        let blk = w >> 2u; let base = (w & 3u) * 4u; let xo = (blk * 32u + base) >> 2u;\n");
            for j in js() {
                let _ = writeln!(s, "        let bi{j} = or{j} * nblk + blk; let qh{j} = aux[bi{j} * 2u]; let d{j} = unpack2x16float(aux[bi{j} * 2u + 1u]).x;
        let c{j} = codes[or{j} * nwords + w];
        let lo{j} = vec4<f32>(f32(i32(( c{j} & 0xfu) | (((qh{j} >> (base + 0u)) & 1u) << 4u)) - 16), f32(i32(((c{j} >> 8u) & 0xfu) | (((qh{j} >> (base + 1u)) & 1u) << 4u)) - 16),
                              f32(i32(((c{j} >> 16u) & 0xfu) | (((qh{j} >> (base + 2u)) & 1u) << 4u)) - 16), f32(i32(((c{j} >> 24u) & 0xfu) | (((qh{j} >> (base + 3u)) & 1u) << 4u)) - 16));
        let hi{j} = vec4<f32>(f32(i32(((c{j} >> 4u) & 0xfu) | (((qh{j} >> (base + 16u)) & 1u) << 4u)) - 16), f32(i32(((c{j} >> 12u) & 0xfu) | (((qh{j} >> (base + 17u)) & 1u) << 4u)) - 16),
                              f32(i32(((c{j} >> 20u) & 0xfu) | (((qh{j} >> (base + 18u)) & 1u) << 4u)) - 16), f32(i32(((c{j} >> 28u) & 0xfu) | (((qh{j} >> (base + 19u)) & 1u) << 4u)) - 16));");
            }
            for r in rs() { let _ = writeln!(s, "        let xa{r} = x[(xr{r} >> 2u) + xo]; let xb{r} = x[(xr{r} >> 2u) + xo + 4u];"); }
            for (j, r) in jr() { let _ = writeln!(s, "        a{j}_{r} = a{j}_{r} + (dot(xa{r}, lo{j}) + dot(xb{r}, hi{j})) * d{j};"); }
            s.push_str("    }\n");
        }
        // Per (block, sub-block s, word w): `acc + ds·dot(x, q) − mm·Σx`, Q4_K_INNER / Q5_K_INNER order.
        Fmt::Q4K | Fmt::Q5K => {
            let _ = writeln!(s, "    let nblk = in_dim / 256u;
    for (var blk: u32 = bl; blk < nblk; blk = blk + {lanes}u) {{");
            for j in js() { let _ = writeln!(s, "        let bi{j} = or{j} * nblk + blk; let dm{j} = unpack2x16float(aux[bi{j} * 4u]);"); }
            s.push_str("        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
            let sh = 4u * (s & 1u); let xs = blk * 256u + 32u * s; let cws = 8u * (s >> 1u);\n");
            for j in js() { let _ = writeln!(s, "            let sm{j} = scmin(bi{j} * 4u, s); let ds{j} = dm{j}.x * f32(sm{j}.x); let mm{j} = dm{j}.y * f32(sm{j}.y);"); }
            for r in rs() { let _ = writeln!(s, "            let xb{r} = (xr{r} + xs) >> 2u;"); }
            s.push_str("            for (var w: u32 = 0u; w < 8u; w = w + 1u) {\n");
            for j in js() {
                if fmt == Fmt::Q4K {
                    let _ = writeln!(s, "                let c{j} = codes[bi{j} * 32u + cws + w];
                let q{j} = vec4<f32>(f32((c{j} >> sh) & 0xfu), f32((c{j} >> (sh + 8u)) & 0xfu), f32((c{j} >> (sh + 16u)) & 0xfu), f32((c{j} >> (sh + 24u)) & 0xfu));");
                } else {
                    let _ = writeln!(s, "                let c{j} = codes[bi{j} * 40u + cws + w]; let h{j} = codes[bi{j} * 40u + 32u + w];
                let q{j} = vec4<f32>(f32((c{j} >> sh) & 0xfu), f32((c{j} >> (sh + 8u)) & 0xfu), f32((c{j} >> (sh + 16u)) & 0xfu), f32((c{j} >> (sh + 24u)) & 0xfu))
                         + vec4<f32>(f32((h{j} >> s) & 1u), f32((h{j} >> (8u + s)) & 1u), f32((h{j} >> (16u + s)) & 1u), f32((h{j} >> (24u + s)) & 1u)) * 16.0;");
                }
            }
            for r in rs() { let _ = writeln!(s, "                let xw{r} = x[xb{r} + w]; let xsum{r} = xw{r}.x + xw{r}.y + xw{r}.z + xw{r}.w;"); }
            for (j, r) in jr() { let _ = writeln!(s, "                a{j}_{r} = a{j}_{r} + ds{j} * dot(xw{r}, q{j}) - mm{j} * xsum{r};"); }
            s.push_str("            }\n        }\n    }\n");
        }
        // Q6_K_BODY's four products per `l`, each `acc + x·d·scale·q`, left to right.
        Fmt::Q6K => {
            let _ = writeln!(s, "    let nblk = in_dim / 256u;
    for (var blk: u32 = bl; blk < nblk; blk = blk + {lanes}u) {{");
            for j in js() { let _ = writeln!(s, "        let bi{j} = or{j} * nblk + blk; let cb{j} = bi{j} * 48u; let ab{j} = bi{j} * 5u; let d{j} = unpack2x16float(aux[ab{j}]).x;"); }
            s.push_str("        for (var hf: u32 = 0u; hf < 2u; hf = hf + 1u) {
            let qlo = 64u * hf; let qho = 32u * hf; let sco = 8u * hf; let xh = blk * 256u + 128u * hf;
            for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                let is = l >> 4u;\n");
            for j in js() {
                let _ = writeln!(s, "                let h{j} = qhb(cb{j}, qho + l); let la{j} = qlb(cb{j}, qlo + l); let lb{j} = qlb(cb{j}, qlo + l + 32u);
                let p{j}a = f32(i32((la{j} & 0xFu) | ((h{j} & 3u) << 4u)) - 32); let p{j}b = f32(i32((lb{j} & 0xFu) | (((h{j} >> 2u) & 3u) << 4u)) - 32);
                let p{j}c = f32(i32((la{j} >> 4u) | (((h{j} >> 4u) & 3u) << 4u)) - 32); let p{j}d = f32(i32((lb{j} >> 4u) | (((h{j} >> 6u) & 3u) << 4u)) - 32);
                let s{j}a = scb(ab{j}, sco + is); let s{j}b = scb(ab{j}, sco + is + 2u); let s{j}c = scb(ab{j}, sco + is + 4u); let s{j}d = scb(ab{j}, sco + is + 6u);");
            }
            for r in rs() { let _ = writeln!(s, "                let xi{r} = xr{r} + xh + l; let x{r}a = x[xi{r}]; let x{r}b = x[xi{r} + 32u]; let x{r}c = x[xi{r} + 64u]; let x{r}d = x[xi{r} + 96u];"); }
            for (j, r) in jr() {
                for q in ["a", "b", "c", "d"] { let _ = writeln!(s, "                a{j}_{r} = a{j}_{r} + x{r}{q} * d{j} * s{j}{q} * p{j}{q};"); }
            }
            s.push_str("            }\n        }\n    }\n");
        }
    }
    // Every accumulator reduced by the one-row kernel's barrier tree (stride L/2 down to 1 over an
    // output's lanes), then stored — plain, or silu(gate)·up.
    let k = |j: usize, r: usize| (j * mm + r) * 64;
    for (j, r) in jr() { let _ = writeln!(s, "    partial[{}u + t] = a{j}_{r};", k(j, r)); }
    let _ = writeln!(s, "    workgroupBarrier();
    for (var st: u32 = {}u; st > 0u; st = st >> 1u) {{
        if (bl < st) {{", lanes / 2);
    for (j, r) in jr() { let _ = writeln!(s, "            partial[{0}u + t] = partial[{0}u + t] + partial[{0}u + t + st];", k(j, r)); }
    s.push_str("        }\n        workgroupBarrier();\n    }\n    if (bl == 0u) {\n");
    for j in 0..rr {
        for r in rs() {
            let val = if swiglu {
                format!("let gg = partial[{}u + t]; out[(r0 + {r}u) * o_dim + o0 + {j}u] = (gg / (1.0 + exp(-gg))) * partial[{}u + t];", k(j, r), k(j + rr, r))
            } else {
                format!("out[(r0 + {r}u) * o_dim + o0 + {j}u] = partial[{}u + t];", k(j, r))
            };
            let _ = writeln!(s, "        if (o0 + {j}u < o_dim && r0 + {r}u < rows) {{ {val} }}");
        }
    }
    s.push_str("    }\n}\n");
    s
}

/// The one-row kernel's lane layout for this format, so the small-M kernel walks blocks identically.
fn lanes(fmt: Fmt, swiglu: bool, in_dim: usize) -> (u32, u32) {
    match (fmt, swiglu) {
        (Fmt::Q8_0 | Fmt::Q5_0, _) => (64, 1),          // the 32-value split-K kernels: 64 lanes, one output
        (Fmt::Q4K | Fmt::Q6K, false) => splitk_lanes_wide(in_dim / 256),
        // matmul_q5_k uses splitk_lanes_sub, which is (l, 1, opw) with FERRIC_SUBBLK unset (required above)
        (Fmt::Q5K, _) => { let (l, _, opw) = splitk_lanes_sub(in_dim / 256); (l, opw) }
        (Fmt::Q4K | Fmt::Q6K, true) => splitk_lanes(in_dim / 256),
    }
}

/// `y = x·Wᵀ` (or the fused SwiGLU, `[rows, n_out/2]`) for all rows in one dispatch.
fn dispatch(x: &Tensor, codes: &wgpu::Buffer, aux: &wgpu::Buffer, n_out: usize, in_dim: usize, fmt: Fmt, swiglu: bool,
            p: Option<Plan>) -> Tensor {
    let ctx = &x.ctx;
    let rows = x.shape[0];
    let o_dim = if swiglu { n_out / 2 } else { n_out };
    let (l, opw) = lanes(fmt, swiglu, in_dim);
    let wbytes = codes.size() as usize + aux.size() as usize;
    let p = p.unwrap_or_else(|| plan(fmt, rows, o_dim, swiglu, wbytes));
    let out = empty(ctx, rows * o_dim);
    let nwg = o_dim.div_ceil(p.r).div_ceil(opw as usize);
    let gw = nwg.min(32768);
    let grid = (gw as u32, nwg.div_ceil(gw) as u32, rows.div_ceil(p.m) as u32);
    let label = match (fmt, swiglu) {
        (Fmt::Q8_0, _) => "matmul_q8_0_mr", (Fmt::Q5_0, _) => "matmul_q5_0_mr",
        (Fmt::Q4K, false) => "matmul_q4_k_mr", (Fmt::Q5K, false) => "matmul_q5_k_mr", (Fmt::Q6K, false) => "matmul_q6_k_mr",
        (Fmt::Q4K, true) => "matmul_q4_k_swiglu_mr", (Fmt::Q5K, true) => "matmul_q5_k_swiglu_mr", (Fmt::Q6K, true) => "matmul_q6_k_swiglu_mr",
    };
    run(ctx, &source(fmt, swiglu, p, l, opw), label,
        &[x.buf.as_ref(), codes, aux, &out, &unibuf(ctx, &[rows as u32, o_dim as u32, in_dim as u32, gw as u32])], grid);
    Tensor::from_parts(ctx, out, vec![rows, o_dim])
}

/// Small-M `x·Wᵀ` if this call qualifies, else `None` (touching nothing). `x` must be contiguous.
pub(super) fn matmul(x: &Tensor, codes: &wgpu::Buffer, aux: &wgpu::Buffer, n_out: usize, in_dim: usize, fmt: Fmt) -> Option<Tensor> {
    if !eligible(&x.ctx, x.shape[0], n_out, in_dim) { return None; }
    Some(dispatch(x, codes, aux, n_out, in_dim, fmt, false, None))
}

/// Small-M fused gate|up + SwiGLU (`[2·n_ff, in]` weight, gate rows first) if it qualifies.
pub(super) fn swiglu(x: &Tensor, codes: &wgpu::Buffer, aux: &wgpu::Buffer, n_out: usize, in_dim: usize, fmt: Fmt) -> Option<Tensor> {
    // The fused one-row kernels have no flat/split choice, so only the row window applies.
    if !in_window(&x.ctx, x.shape[0]) { return None; }
    Some(dispatch(x, codes, aux, n_out, in_dim, fmt, true, None))
}

/// The small-M kernel for exactly these rows with an explicit tile (or the planner's), ignoring every
/// routing gate — the hermetic test seam.
#[cfg(test)]
pub(crate) fn forced(x: &Tensor, codes: &wgpu::Buffer, aux: &wgpu::Buffer, n_out: usize, in_dim: usize, fmt: Fmt,
                     swiglu: bool, p: Option<Plan>) -> Tensor {
    dispatch(&x.contiguous(), codes, aux, n_out, in_dim, fmt, swiglu, p)
}

#[cfg(test)]
mod tests {
    use super::{forced, Fmt, Plan};
    use crate::dtype::{Q4_KWeights, Q5_0Weights, Q5_KWeights, Q6_KWeights, Q8_0Weights};
    use crate::Tensor;
    use std::sync::Arc;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn unit(&mut self) -> f64 { (self.next() % 2_000_001) as f64 / 1e6 - 1.0 }
        fn byte(&mut self) -> u8 { self.next() as u8 }
    }
    fn f16le(v: f64) -> [u8; 2] { half::f16::from_f64(v).to_bits().to_le_bytes() }
    fn h(b: &[u8], o: usize) -> f64 { half::f16::from_bits(u16::from_le_bytes([b[o], b[o + 1]])).to_f64() }

    // ---- The formats as ggml DEFINES them (dequantize_row_q8_0 / _q5_0 / _q4_K / _q5_K / _q6_K), in f64,
    // from the RAW GGUF block bytes — independent of the GPU repack in dtype.rs and of every kernel. ----
    fn deq(fmt: Fmt, raw: &[u8], wrong: bool) -> Vec<f64> {
        let mut w = Vec::new();
        match fmt {
            // wrong: the scale of the NEXT block
            Fmt::Q8_0 => { let nb = raw.len() / 34; for b in 0..nb {
                let blk = &raw[b * 34..]; let d = if wrong { h(raw, ((b + 1) % nb) * 34) } else { h(blk, 0) };
                for l in 0..32 { w.push(d * (blk[2 + l] as i8) as f64); } } }
            // wrong: the fifth bit dropped
            Fmt::Q5_0 => for blk in raw.chunks(22) {
                let d = h(blk, 0);
                let qh = if wrong { 0 } else { u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]) };
                let mut v = [0f64; 32];
                for j in 0..16 {
                    v[j] = d * (((blk[6 + j] & 0xF) as u32 | (((qh >> j) & 1) << 4)) as i32 - 16) as f64;
                    v[j + 16] = d * (((blk[6 + j] >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4)) as i32 - 16) as f64;
                }
                w.extend(v);
            },
            // wrong: low/high nibble roles exchanged
            Fmt::Q4K | Fmt::Q5K => {
                let bs = if fmt == Fmt::Q4K { 144 } else { 176 };
                for blk in raw.chunks(bs) {
                    let (d, dmin, sc) = (h(blk, 0), h(blk, 2), &blk[4..16]);
                    let (qh, qs): (&[u8], &[u8]) = if fmt == Fmt::Q4K { (&[], &blk[16..144]) } else { (&blk[16..48], &blk[48..176]) };
                    let smk = |j: usize| if j < 4 { ((sc[j] & 63) as f64, (sc[j + 4] & 63) as f64) }
                        else { (((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4)) as f64, ((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)) as f64) };
                    for j in 0..4 {
                        let ((s1, m1), (s2, m2)) = (smk(2 * j), smk(2 * j + 1));
                        let mut v = [0f64; 64];
                        for l in 0..32 {
                            let (mut lo, mut hi) = ((qs[32 * j + l] & 0xF) as f64, (qs[32 * j + l] >> 4) as f64);
                            if wrong { std::mem::swap(&mut lo, &mut hi); }
                            if fmt == Fmt::Q5K {
                                if qh[l] & (1 << (2 * j)) != 0 { lo += 16.0; }
                                if qh[l] & (2 << (2 * j)) != 0 { hi += 16.0; }
                            }
                            v[l] = d * s1 * lo - dmin * m1;
                            v[32 + l] = d * s2 * hi - dmin * m2;
                        }
                        w.extend(v);
                    }
                }
            }
            // wrong: the 2 high bits read from the wrong shift
            Fmt::Q6K => for blk in raw.chunks(210) {
                let d = h(blk, 208);
                let mut v = [0f64; 256];
                for hf in 0..2 {
                    let (ql, qh, sc) = (&blk[64 * hf..], &blk[128 + 32 * hf..], &blk[192 + 8 * hf..]);
                    for l in 0..32 {
                        let is = l / 16;
                        let sh = |s: u32| if wrong { (s + 2) % 8 } else { s };
                        let q = [((ql[l] & 0xF) | (((qh[l] >> sh(0)) & 3) << 4)) as i32 - 32,
                                 ((ql[l + 32] & 0xF) | (((qh[l] >> sh(2)) & 3) << 4)) as i32 - 32,
                                 ((ql[l] >> 4) | (((qh[l] >> sh(4)) & 3) << 4)) as i32 - 32,
                                 ((ql[l + 32] >> 4) | (((qh[l] >> sh(6)) & 3) << 4)) as i32 - 32];
                        for (k, qk) in q.iter().enumerate() { v[128 * hf + 32 * k + l] = d * (sc[is + 2 * k] as i8) as f64 * *qk as f64; }
                    }
                }
                w.extend(v);
            },
        }
        w
    }
    fn rand_blocks(rng: &mut Rng, fmt: Fmt, nblk: usize) -> Vec<u8> {
        let mut raw = Vec::new();
        for _ in 0..nblk {
            match fmt {
                Fmt::Q8_0 => { raw.extend(f16le(0.002 + 0.002 * rng.unit())); for _ in 0..32 { raw.push(rng.byte()); } }
                Fmt::Q5_0 => { raw.extend(f16le(0.004 + 0.002 * rng.unit())); for _ in 0..20 { raw.push(rng.byte()); } }
                Fmt::Q4K | Fmt::Q5K => {
                    raw.extend(f16le(0.004 + 0.002 * rng.unit())); raw.extend(f16le(0.002 + 0.001 * rng.unit()));
                    for _ in 0..(if fmt == Fmt::Q4K { 140 } else { 172 }) { raw.push(rng.byte()); }
                }
                Fmt::Q6K => { for _ in 0..208 { raw.push(rng.byte()); } raw.extend(f16le(0.0005 + 0.0002 * rng.unit())); }
            }
        }
        raw
    }
    fn block(fmt: Fmt) -> (usize, usize) {
        match fmt { Fmt::Q8_0 => (32, 34), Fmt::Q5_0 => (32, 22), Fmt::Q4K => (256, 144), Fmt::Q5K => (256, 176), Fmt::Q6K => (256, 210) }
    }

    enum W { Q8(Q8_0Weights), Q50(Q5_0Weights), Q4(Q4_KWeights), Q5(Q5_KWeights), Q6(Q6_KWeights) }
    impl W {
        fn new(ctx: &Arc<ferric_core::Context>, fmt: Fmt, raw: &[u8], n: usize, k: usize) -> W {
            match fmt {
                Fmt::Q8_0 => W::Q8(Q8_0Weights::from_bytes(ctx, raw, n, k)), Fmt::Q5_0 => W::Q50(Q5_0Weights::from_bytes(ctx, raw, n, k)),
                Fmt::Q4K => W::Q4(Q4_KWeights::from_bytes(ctx, raw, n, k)), Fmt::Q5K => W::Q5(Q5_KWeights::from_bytes(ctx, raw, n, k)),
                Fmt::Q6K => W::Q6(Q6_KWeights::from_bytes(ctx, raw, n, k)),
            }
        }
        fn bufs(&self) -> (&wgpu::Buffer, &wgpu::Buffer, usize, usize) {
            match self {
                W::Q8(w) => (&w.codes, &w.scales, w.rows, w.cols), W::Q50(w) => (&w.codes, &w.scales, w.rows, w.cols),
                W::Q4(w) => (&w.codes, &w.aux, w.rows, w.cols), W::Q5(w) => (&w.codes, &w.aux, w.rows, w.cols),
                W::Q6(w) => (&w.codes, &w.aux, w.rows, w.cols),
            }
        }
        /// The ONE-ROW decode path — what serial decode runs today (split-K by default).
        fn one_row(&self, x: &Tensor, swiglu: bool) -> Tensor {
            match (self, swiglu) {
                (W::Q8(w), false) => x.matmul_q8_0(w), (W::Q50(w), false) => x.matmul_q5_0(w),
                (W::Q4(w), false) => x.matmul_q4_k(w), (W::Q5(w), false) => x.matmul_q5_k(w), (W::Q6(w), false) => x.matmul_q6_k(w),
                (W::Q4(w), true) => x.matmul_q4_k_swiglu(w), (W::Q5(w), true) => x.matmul_q5_k_swiglu(w), (W::Q6(w), true) => x.matmul_q6_k_swiglu(w),
                _ => unreachable!(),
            }
        }
    }

    fn ctx() -> Option<Arc<ferric_core::Context>> { pollster::block_on(ferric_core::Context::new()).ok().map(Arc::new) }

    /// ⭐ THE CONTRACT. Every row of a small-M dispatch equals, BIT FOR BIT, that row run alone through the
    /// one-row decode kernel — for every format, plain and fused gate|up, at row counts on both sides of
    /// every tile boundary, under the planner's tile AND forced tiles (R outputs x M rows, including
    /// ragged output counts that leave a partial lane group, and row chunks that put a row in a different
    /// z-slice and tile slot). This is what makes batched decode equal solo decode and a lookup verify row
    /// equal the plain loop's matmul. A different accumulation order, a different reduction tree or a
    /// crossed row fails it on the first mismatching bit.
    #[test]
    fn small_m_rows_are_bit_identical_to_one_row_decode() {
        let Some(ctx) = ctx() else { return };
        let mut rng = Rng(0x5eed_0f_5a11);
        let cases: &[(Fmt, bool, usize, usize)] = &[
            (Fmt::Q8_0, false, 130, 896), (Fmt::Q8_0, false, 67, 224), (Fmt::Q5_0, false, 130, 896), (Fmt::Q5_0, false, 67, 96),
            (Fmt::Q4K, false, 70, 1024), (Fmt::Q4K, false, 33, 4864), (Fmt::Q5K, false, 70, 1024), (Fmt::Q5K, false, 41, 2304),
            (Fmt::Q6K, false, 70, 1024), (Fmt::Q6K, false, 29, 256),
            (Fmt::Q4K, true, 2 * 37, 512), (Fmt::Q5K, true, 2 * 37, 512), (Fmt::Q6K, true, 2 * 21, 768),
        ];
        let tiles: &[Option<Plan>] = &[None, Some(Plan { r: 1, m: 1 }), Some(Plan { r: 4, m: 8 }), Some(Plan { r: 2, m: 3 }), Some(Plan { r: 3, m: 5 })];
        let mut checked = 0usize;
        for &(fmt, sw, n, k) in cases {
            let (bs, bb) = block(fmt);
            let raw = rand_blocks(&mut rng, fmt, n * k / bs); let _ = bb;
            let w = W::new(&ctx, fmt, &raw, n, k);
            let (codes, aux, rows_w, cols_w) = w.bufs();
            let o_dim = if sw { n / 2 } else { n };
            for m in [2usize, 3, 5, 9, 16, 17, 32] {
                let x: Vec<f32> = (0..m * k).map(|_| rng.unit() as f32).collect();
                let solo: Vec<f32> = (0..m).flat_map(|r| {
                    let xr = Tensor::from_vec(&ctx, &x[r * k..(r + 1) * k], &[1, k]);
                    pollster::block_on(w.one_row(&xr, sw).to_vec())
                }).collect();
                let xt = Tensor::from_vec(&ctx, &x, &[m, k]);
                for &p in tiles {
                    let got = pollster::block_on(forced(&xt, codes, aux, rows_w, cols_w, fmt, sw, p).to_vec());
                    assert_eq!(got.len(), m * o_dim);
                    for (i, (a, b)) in got.iter().zip(&solo).enumerate() {
                        assert!(a.to_bits() == b.to_bits(),
                                "{fmt:?} swiglu={sw} n={n} k={k} M={m} tile {p:?}: row {} out {} = {a:e}, one-row decode gives {b:e}",
                                i / o_dim, i % o_dim);
                    }
                    checked += got.len();
                }
            }
        }
        eprintln!("small-M: {checked} outputs bit-identical to one-row decode");
    }

    /// Against the FORMAT DEFINITION (f64, from the raw GGUF bytes): the small-M result sits within f32
    /// accumulation noise of the exact product (1e-6 of Σ|x·w|), no further than the one-row kernel sits,
    /// and each plausible wrong decoding misses by ≥ 20x the tolerance — so this check can see a decoding
    /// defect, not only an ordering one.
    #[test]
    fn small_m_matches_the_format_definition() {
        let Some(ctx) = ctx() else { return };
        let mut rng = Rng(0xdec0_de5);
        for &(fmt, n, k, m) in &[(Fmt::Q8_0, 96, 896, 9), (Fmt::Q5_0, 96, 896, 9), (Fmt::Q4K, 50, 1024, 7), (Fmt::Q5K, 50, 1024, 7), (Fmt::Q6K, 50, 1024, 7)] {
            let (bs, _) = block(fmt);
            let raw = rand_blocks(&mut rng, fmt, n * k / bs);
            let w = W::new(&ctx, fmt, &raw, n, k);
            let (codes, aux, rows_w, cols_w) = w.bufs();
            let x: Vec<f32> = (0..m * k).map(|_| rng.unit() as f32).collect();
            let got = pollster::block_on(forced(&Tensor::from_vec(&ctx, &x, &[m, k]), codes, aux, rows_w, cols_w, fmt, false, None).to_vec());
            let (wd, wbad) = (deq(fmt, &raw, false), deq(fmt, &raw, true));
            let (mut e, mut ebad, mut scale) = (0f64, 0f64, 0f64);
            for i in 0..m {
                for j in 0..n {
                    let (mut a, mut abad, mut s) = (0f64, 0f64, 0f64);
                    for l in 0..k {
                        a += x[i * k + l] as f64 * wd[j * k + l];
                        abad += x[i * k + l] as f64 * wbad[j * k + l];
                        s += (x[i * k + l] as f64 * wd[j * k + l]).abs();
                    }
                    let g = got[i * n + j] as f64;
                    e = e.max((g - a).abs()); ebad = ebad.max((g - abad).abs()); scale = scale.max(s);
                }
            }
            let tol = 1e-6 * scale;
            eprintln!("{fmt:?} M={m} n={n} k={k}: |small-M − f64 definition| {e:.2e} (tol {tol:.2e}); wrong decoding misses by {:.0}x tol", ebad / tol);
            assert!(e <= tol, "{fmt:?}: {e:.3e} > {tol:.3e}");
            assert!(ebad >= 20.0 * tol, "{fmt:?}: the negative control missed by only {ebad:.3e} — this check could not see it");
        }
    }
}
