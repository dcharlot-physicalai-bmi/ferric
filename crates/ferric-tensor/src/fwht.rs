//! **Blockwise fast Walsh–Hadamard transform** on the GPU — the activation-side transform PrismML's
//! Bonsai 2 needs before every folded matmul (`W' · H(s ⊙ x)`; see `ferric_gguf::prism`).
//!
//! `y = H_N(s ⊙ P x) / √N` over consecutive `N`-wide blocks of each row, where `N` is any power of
//! two, `s` an optional ±1 vector over the whole row width and `P` an optional tiled→grouped head
//! permutation (Bonsai 2's `ssm_out`, whose fold was computed in the training V-head order).
//!
//! Arithmetic order is the fork's Metal `kernel_fwht_tg` and `ferric_gguf::prism::fwht_normalized`:
//! prescale by `1/√N` on load, then radix-2 butterflies `(a, b) → (a + b, a − b)` at strides 1, 2, …,
//! N/2. Adds and subtracts only — nothing a compiler may contract into an FMA — so the GPU result is
//! the host reference's bit for bit (asserted in the tests below).
//!
//! Two kernels: stages up to `m = min(N, 4096)` run in workgroup memory (4096 f32 = 16 KiB, WebGPU's
//! guaranteed `maxComputeWorkgroupStorageSize`); strides from `m` to `N/2` — only when `N > 4096` —
//! run one global-memory pass each. Bonsai 2 uses `N = 1024`: one dispatch per transform.

use crate::Tensor;

/// Largest block handled entirely in workgroup memory.
const SHARED_MAX: usize = 4096;

fn shared_wgsl(m: usize) -> String {
    format!(r#"
@group(0) @binding(0) var<storage,read>       x:    array<f32>;
@group(0) @binding(1) var<storage,read>       sg:   array<f32>;   // signs over the row width (or a dummy)
@group(0) @binding(2) var<storage,read_write> y:    array<f32>;
@group(0) @binding(3) var<uniform>            info: array<vec4<u32>, 3>;
// info[0] = rows, width, n (full block), flags (1 = signs, 2 = permutation)
// info[1] = hd, nk, rep, gx (workgroups per grid row)
// info[2] = scale bits, 0, 0, 0
const M: u32 = {m}u;
var<workgroup> sh: array<f32, {m}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {{
    let rows = info[0].x; let width = info[0].y; let flags = info[0].w;
    let hd = info[1].x; let nk = info[1].y; let rep = info[1].z; let gx = info[1].w;
    let scale = bitcast<f32>(info[2].x);
    let per_row = width / M;
    let g = wg.y * gx + wg.x;
    // Uniform control flow: every thread of the workgroup takes the same branch.
    if (g >= rows * per_row) {{ return; }}
    let row = g / per_row;
    let f0 = (g % per_row) * M;
    let rb = row * width;
    for (var e = tid; e < M; e = e + 256u) {{
        let f = f0 + e;
        var src = f;
        if ((flags & 2u) != 0u) {{
            // grouped feature f = d + hd*(r + rep*k)  <-  tiled feature d + hd*(k + nk*r)
            let d = f % hd; let h = f / hd; let r = h % rep; let k = h / rep;
            src = d + hd * (k + nk * r);
        }}
        var v = x[rb + src];
        if ((flags & 1u) != 0u) {{ v = v * sg[f]; }}
        sh[e] = v * scale;
    }}
    workgroupBarrier();
    for (var h = 1u; h < M; h = h * 2u) {{
        for (var p = tid; p < M / 2u; p = p + 256u) {{
            let i = (p / h) * 2u * h + (p % h);
            let a = sh[i]; let b = sh[i + h];
            sh[i] = a + b;
            sh[i + h] = a - b;
        }}
        workgroupBarrier();
    }}
    for (var e = tid; e < M; e = e + 256u) {{ y[rb + f0 + e] = sh[e]; }}
}}
"#)
}

const STAGE_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read_write> y:    array<f32>;
@group(0) @binding(1) var<uniform>            info: array<vec4<u32>, 2>;
// info[0] = rows, width, n, h (this pass's stride); info[1] = row stride of the thread grid, 0, 0, 0
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = info[0].x; let width = info[0].y; let n = info[0].z; let h = info[0].w;
    let p = gid.y * info[1].x + gid.x;
    let half_w = width / 2u;
    if (p >= rows * half_w) { return; }
    let row = p / half_w; let q = p % half_w;
    let blk = q / (n / 2u); let r = q % (n / 2u);
    let i = row * width + blk * n + (r / h) * 2u * h + (r % h);
    let a = y[i]; let b = y[i + h];
    y[i] = a + b;
    y[i + h] = a - b;
}
"#;

/// `H_n(s ⊙ P x)/√n` over consecutive `n`-blocks of the last dimension. `signs` (if any) has length
/// equal to the last dimension; `perm = Some((hd, nk, rep))` gathers tiled head order `[hd, nk, rep]`
/// into grouped `[hd, rep, nk]` first (`hd·nk·rep` must equal the width).
pub fn hadamard_rows(x: &Tensor, n: usize, signs: Option<&Tensor>, perm: Option<(usize, usize, usize)>) -> Tensor {
    let width = *x.shape.last().expect("fwht of a scalar");
    assert!(n.is_power_of_two() && n >= 2, "fwht block {n} is not a power of two >= 2");
    assert!(width % n == 0, "fwht block {n} does not divide the width {width}");
    if let Some(s) = signs { assert_eq!(s.numel(), width, "fwht sign vector length"); }
    if let Some((hd, nk, rep)) = perm { assert_eq!(hd * nk * rep, width, "fwht permutation geometry"); }
    let xc = x.contiguous();
    let rows = xc.numel() / width;
    let ctx = &xc.ctx;
    let out = crate::empty(ctx, rows * width);
    let m = n.min(SHARED_MAX);
    let sc = signs.map(|s| s.contiguous());
    let flags = sc.is_some() as u32 | (perm.is_some() as u32) << 1;
    let (hd, nk, rep) = perm.unwrap_or((1, 1, 1));
    let total = rows * (width / m);
    let gx = total.clamp(1, 32768);
    let gy = total.div_ceil(gx).max(1);
    let scale = 1.0f32 / (n as f32).sqrt();
    let info = crate::unibuf(ctx, &[rows as u32, width as u32, n as u32, flags,
                                    hd as u32, nk as u32, rep as u32, gx as u32,
                                    scale.to_bits(), 0, 0, 0]);
    let sbuf = sc.as_ref().map(|s| s.buf.as_ref()).unwrap_or(xc.buf.as_ref());
    crate::run(ctx, &shared_wgsl(m), "fwht_shared", &[xc.buf.as_ref(), sbuf, &out, &info], (gx as u32, gy as u32, 1));
    let mut h = m;
    while h < n {
        let pairs = rows * width / 2;
        let (grid, rs) = crate::groups2d(pairs);
        let info = crate::unibuf(ctx, &[rows as u32, width as u32, n as u32, h as u32, rs, 0, 0, 0]);
        crate::run(ctx, STAGE_WGSL, "fwht_stage", &[&out, &info], grid);
        h *= 2;
    }
    Tensor::from_parts(ctx, out, xc.shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Host reference, written independently of the kernel: prescale, then butterflies at increasing
    /// stride (the order `ferric_gguf::prism::fwht_normalized` and the fork's Metal kernel use).
    fn host(x: &[f32], width: usize, n: usize, s: Option<&[f32]>, perm: Option<(usize, usize, usize)>) -> Vec<f32> {
        let mut out = Vec::with_capacity(x.len());
        let scale = 1.0 / (n as f32).sqrt();
        for row in x.chunks_exact(width) {
            let mut v: Vec<f32> = (0..width).map(|f| {
                let src = match perm {
                    Some((hd, nk, rep)) => { let (d, h) = (f % hd, f / hd); let (r, k) = (h % rep, h / rep); d + hd * (k + nk * r) }
                    None => f,
                };
                row[src] * s.map_or(1.0, |s| s[f]) * scale
            }).collect();
            for blk in v.chunks_exact_mut(n) {
                let mut h = 1;
                while h < n {
                    for i in (0..n).step_by(2 * h) { for j in i..i + h {
                        let (a, b) = (blk[j], blk[j + h]);
                        blk[j] = a + b; blk[j + h] = a - b;
                    }}
                    h *= 2;
                }
            }
            out.extend(v);
        }
        out
    }

    fn ctx() -> Option<Arc<ferric_core::Context>> { pollster::block_on(ferric_core::Context::new()).ok().map(Arc::new) }

    #[test]
    fn gpu_fwht_is_bit_identical_to_the_host_reference() {
        let Some(ctx) = ctx() else { eprintln!("no GPU; skipped"); return };
        // (rows, width, block, signs?, perm?) — incl. Bonsai 2's 1024 over 5120/6144/17408 widths, the
        // GDN permutation at 48 = 16x3 heads of 128, and blocks past the shared-memory limit (8192,
        // 16384) that exercise the global stage passes.
        let cases: &[(usize, usize, usize, bool, Option<(usize, usize, usize)>)] = &[
            (3, 5120, 1024, true, None), (2, 6144, 1024, true, Some((128, 16, 3))), (1, 17408, 1024, true, None),
            (5, 256, 64, false, None), (2, 16384, 8192, true, None), (1, 16384, 16384, false, None),
            (4, 1024, 2, true, None),
        ];
        for &(rows, width, n, signed, perm) in cases {
            let x: Vec<f32> = (0..rows * width).map(|i| (((i * 2654435761usize) >> 7) % 2001) as f32 / 97.0 - 10.0).collect();
            let s: Vec<f32> = (0..width).map(|i| if (i * 40503) % 7 < 3 { -1.0 } else { 1.0 }).collect();
            let want = host(&x, width, n, signed.then_some(&s[..]), perm);
            let st = Tensor::from_vec(&ctx, &s, &[width]);
            let got = pollster::block_on(hadamard_rows(&Tensor::from_vec(&ctx, &x, &[rows, width]), n,
                                                       signed.then_some(&st), perm).to_vec());
            let bad = got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(bad, 0, "rows {rows} width {width} block {n}: {bad} values differ from the host reference");
        }
    }

    #[test]
    fn the_transform_is_its_own_inverse_and_the_signs_and_permutation_move_the_result() {
        let Some(ctx) = ctx() else { eprintln!("no GPU; skipped"); return };
        let (w, n) = (6144usize, 1024usize);
        let x: Vec<f32> = (0..w).map(|i| (i % 13) as f32 - 6.0).collect();
        let t = Tensor::from_vec(&ctx, &x, &[1, w]);
        let back = pollster::block_on(hadamard_rows(&hadamard_rows(&t, n, None, None), n, None, None).to_vec());
        assert!(back.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-4), "H·H must be the identity");
        let s = Tensor::from_vec(&ctx, &(0..w).map(|i| if i % 5 == 0 { -1.0 } else { 1.0 }).collect::<Vec<_>>(), &[w]);
        let plain = pollster::block_on(hadamard_rows(&t, n, None, None).to_vec());
        let signed = pollster::block_on(hadamard_rows(&t, n, Some(&s), None).to_vec());
        let permuted = pollster::block_on(hadamard_rows(&t, n, None, Some((128, 16, 3))).to_vec());
        assert!(plain.iter().zip(&signed).any(|(a, b)| (a - b).abs() > 1e-3), "signs had no effect");
        assert!(plain.iter().zip(&permuted).any(|(a, b)| (a - b).abs() > 1e-3), "permutation had no effect");
    }
}
