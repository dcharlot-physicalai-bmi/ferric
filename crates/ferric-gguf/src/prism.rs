//! **PrismML Bonsai 2**: the Hadamard weight-fold contract and the fork-private packings.
//!
//! Bonsai 2 stores its ternary weights in a ROTATED basis. For every folded weight `W'` the model
//! computes `W' · H(s ⊙ x)`, where `H` is the normalized blockwise Sylvester–Walsh–Hadamard
//! transform over the INPUT axis and `s` a per-width ±1 sign vector — so the runtime must transform
//! the ACTIVATION before each such matmul. The token-embedding table stores rotated rows and needs
//! the inverse after lookup: `h = s ⊙ (H z)`. Nothing about the bytes says so; only the
//! `prism.hadamard.*` metadata does. A runtime that decodes the weights and skips the transform does
//! not fail — it produces fluent gibberish (PrismML's own warning for mainline llama.cpp on the
//! group-64 `Q2_0` file, BACKEND-SUPPORT.md).
//!
//! Every rule here is taken from the authors' fork, PrismML-Eng/llama.cpp @ `adfffbe`
//! (`prism-b10743`): the contract is validated exactly as `src/llama-model.cpp:1196-1356` does, the
//! transform is `build_lora_mm` in `src/llama-graph.cpp:1545-1580` (optional tiled→grouped GDN
//! permutation, then sign flip, then `llama_mul_mat_hadamard`), the inverse is
//! `src/models/qwen35.cpp:638-650`, and the block layouts are `ggml/src/ggml-common.h:193-220`
//! with the dequantizers of `ggml/src/ggml-quants.c:474-513, 2255-2283`.
//!
//! ⛔ **The reader locks folded tensors.** [`PrismLock`] makes `raw`/`dequant` of any tensor the
//! contract names FAIL until a runtime that applies the transform calls
//! `GgufSource::unlock_prism_hadamard`. That is the only place every loader passes through, so a
//! loader written before this contract existed — or one written later by someone who never heard
//! of it — refuses the file instead of running it without its transforms.

use crate::Meta;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

/// Fork-private group-128 `Q2_0` (`block_pq2_0`: f16 d + 32 code bytes = 34 B / 128). Byte-identical
/// to the legacy PrismML layout this crate has always decoded as id 42, so the parser rewrites it to
/// 42 (see [`crate::parse`]); the file id is kept here for naming and for the tests.
pub const PQ2_0: u32 = 142;
/// Fork-private group-128 ternary (`block_ptq1_0`: 24 B base-3 `qs` + 2 B `qh` + f16 d = 28 B / 128,
/// 1.75 bpw). Same trit packing as upstream TQ1_0, at one scale per 128 instead of per 256.
pub const PTQ1_0: u32 = 143;
/// Internal id for mainline ggml-org `Q2_0` (`block_q2_0`: f16 d + 16 code bytes = 18 B / 64). The
/// FILE says 42; [`crate::resolve_type_42`] maps it here by stride, as it maps `F8_E4M3_B128` to 1042.
pub const Q2_0_G64: u32 = 2042;
pub const PTQ1_0_BYTES: usize = 28;

/// The `prism.hadamard.*` contract, validated.
#[derive(Debug, Clone)]
pub struct HadamardContract {
    pub version: u32,
    pub block_size: usize,
    /// width → ±1 signs. Empty under `sign_mode = identity`.
    pub signs: BTreeMap<usize, Vec<f32>>,
    /// Weights consumed as `W' · H(s ⊙ x)`. Under `tied_output` this also contains
    /// `token_embd.weight` (the fork adds it: the tied head reads the latent table forward).
    pub folded: HashSet<String>,
    /// Row-lookup tables stored rotated; the fork allows only `token_embd.weight`.
    pub inverse: HashSet<String>,
    /// `ssm_out` was folded in the TRAINING (grouped) V-head order, so its activation must be
    /// permuted tiled→grouped before the sign flip and rotation.
    pub gdn_v_grouped: bool,
    pub tied_output: bool,
}

fn meta_u(m: &HashMap<String, Meta>, k: &str) -> Result<Option<u64>, String> {
    match m.get(k) {
        None => Ok(None),
        Some(Meta::U(v)) => Ok(Some(*v)),
        Some(Meta::I(v)) if *v >= 0 => Ok(Some(*v as u64)),
        Some(o) => Err(format!("{k}: expected an unsigned integer, got {o:?}")),
    }
}
fn meta_s<'a>(m: &'a HashMap<String, Meta>, k: &str) -> Result<&'a str, String> {
    match m.get(k) { Some(Meta::Str(s)) => Ok(s), o => Err(format!("{k}: expected a string, got {o:?}")) }
}
fn meta_b(m: &HashMap<String, Meta>, k: &str) -> Result<bool, String> {
    match m.get(k) { None => Ok(false), Some(Meta::Bool(b)) => Ok(*b), Some(o) => Err(format!("{k}: expected a bool, got {o:?}")) }
}
fn meta_strs(m: &HashMap<String, Meta>, k: &str) -> Result<Vec<String>, String> {
    match m.get(k) {
        None => Ok(Vec::new()),
        Some(Meta::Arr(a)) => a.iter().map(|v| match v { Meta::Str(s) => Ok(s.clone()), o => Err(format!("{k}: non-string element {o:?}")) }).collect(),
        Some(o) => Err(format!("{k}: expected an array of strings, got {o:?}")),
    }
}
fn meta_ints(m: &HashMap<String, Meta>, k: &str) -> Result<Vec<i64>, String> {
    match m.get(k) {
        None => Ok(Vec::new()),
        Some(Meta::Arr(a)) => a.iter().map(|v| match v { Meta::I(n) => Ok(*n), Meta::U(n) => Ok(*n as i64), o => Err(format!("{k}: non-integer element {o:?}")) }).collect(),
        Some(o) => Err(format!("{k}: expected an integer array, got {o:?}")),
    }
}

/// The fork's `is_foldable_weight` (`llama-model.cpp:1288-1316`): the tensor kinds its graph routes
/// through the Hadamard-aware matmul helpers. A contract naming anything else is refused there, and
/// here, before any runtime gets to decide whether it can apply it.
fn is_foldable_weight(name: &str) -> bool {
    const KINDS: &[&str] = &[
        "attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "attn_output",
        "ffn_gate", "ffn_up", "ffn_down",
        "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps",
        "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
        "ssm_out",
    ];
    if name == "output.weight" { return true; }
    let Some(rest) = name.strip_prefix("blk.") else { return false };
    let digits = rest.bytes().take_while(|b| b.is_ascii_digit()).count();
    if digits == 0 || rest.as_bytes().get(digits) != Some(&b'.') { return false; }
    let kind = &rest[digits + 1..];
    KINDS.iter().any(|k| kind == format!("{k}.weight"))
}

impl HadamardContract {
    /// Parse and validate the contract. `Ok(None)` means the file has no `prism.hadamard.*` keys at
    /// all. `has_tensor` answers whether the file carries a tensor (the tied-output rules need it).
    ///
    /// ⚠ One deliberate divergence from the fork, in the SAFE direction: the fork ignores every other
    /// `prism.hadamard.*` key when `version` is absent (the model then runs untransformed). Here any
    /// `prism.hadamard.*` key without a version is a malformed contract and refuses.
    pub fn from_meta(m: &HashMap<String, Meta>, has_tensor: impl Fn(&str) -> bool) -> Result<Option<Self>, String> {
        let any = m.keys().any(|k| k.starts_with("prism.hadamard."));
        let Some(version) = meta_u(m, "prism.hadamard.version")? else {
            if any {
                return Err("file carries prism.hadamard.* metadata but no prism.hadamard.version; a \
                            partial Hadamard contract is refused rather than run untransformed".into());
            }
            return Ok(None);
        };
        let version = version as u32;
        let tied_output = meta_b(m, "prism.hadamard.tied_output")?;
        if version != 1 && version != 2 {
            return Err(format!("unsupported prism.hadamard.version: {version}"));
        }
        if (version == 2) != tied_output {
            return Err("prism.hadamard version 2 requires tied_output=true; version 1 forbids it".into());
        }
        if tied_output && has_tensor("output.weight") {
            return Err("prism.hadamard.tied_output requires output.weight to be absent".into());
        }
        let block = meta_u(m, "prism.hadamard.block_size")?.ok_or("missing prism.hadamard.block_size")? as usize;
        if block == 0 || !block.is_power_of_two() {
            return Err(format!("invalid prism.hadamard.block_size: {block}"));
        }
        let transform = meta_s(m, "prism.hadamard.transform")?;
        if transform != "normalized-sylvester-walsh-hadamard" {
            return Err(format!("unsupported prism.hadamard.transform: {transform}"));
        }
        let axis = meta_s(m, "prism.hadamard.axis")?;
        if axis != "input-last-dimension" {
            return Err(format!("unsupported prism.hadamard.axis: {axis}"));
        }
        let sign_mode = meta_s(m, "prism.hadamard.sign_mode")?;
        if sign_mode != "identity" && sign_mode != "explicit" {
            return Err(format!("unsupported prism.hadamard.sign_mode: {sign_mode}"));
        }
        let weight_names = meta_strs(m, "prism.hadamard.weight_names")?;
        if weight_names.is_empty() {
            return Err("prism.hadamard.weight_names is empty".into());
        }
        let mut signs = BTreeMap::new();
        if sign_mode == "explicit" {
            let widths = meta_ints(m, "prism.hadamard.sign_widths")?;
            let values = meta_ints(m, "prism.hadamard.sign_values")?;
            // Explicit mode with no widths would leave the table empty, which reads as identity later
            // and silently changes the model function (the fork's comment, and its refusal).
            if widths.is_empty() {
                return Err("prism.hadamard.sign_mode is explicit but sign_widths is empty".into());
            }
            let mut off = 0usize;
            for &w in &widths {
                if w <= 0 || (w as usize) % block != 0 || off + w as usize > values.len() {
                    return Err(format!("invalid prism.hadamard sign width: {w}"));
                }
                let w = w as usize;
                let v: Vec<f32> = values[off..off + w].iter().map(|&x| match x {
                    1 => Ok(1.0f32), -1 => Ok(-1.0f32),
                    _ => Err("prism.hadamard sign values must be +/-1".to_string()),
                }).collect::<Result<_, _>>()?;
                signs.insert(w, v);
                off += w;
            }
            if off != values.len() {
                return Err("prism.hadamard.sign_values length mismatch".into());
            }
        }
        let gdn_v_grouped = meta_b(m, "prism.hadamard.gdn_v_grouped")?;
        let mut folded = HashSet::new();
        for n in &weight_names {
            if !is_foldable_weight(n) {
                return Err(format!("prism.hadamard: weight '{n}' is not on a verified Hadamard-aware matmul path"));
            }
            if !folded.insert(n.clone()) {
                return Err(format!("duplicate prism.hadamard weight: {n}"));
            }
        }
        let mut inverse = HashSet::new();
        for n in meta_strs(m, "prism.hadamard.inverse_weight_names")? {
            // The fork applies the inverse only to the token-embedding lookup; any other latent table
            // would load and silently stay rotated.
            if n != "token_embd.weight" {
                return Err(format!("prism.hadamard: weight '{n}' is not a verified inverse-after-lookup table"));
            }
            if folded.contains(&n) || !inverse.insert(n.clone()) {
                return Err(format!("duplicate prism.hadamard inverse weight: {n}"));
            }
        }
        if tied_output {
            if !inverse.contains("token_embd.weight") {
                return Err("prism.hadamard.tied_output requires a latent token embedding".into());
            }
            folded.insert("token_embd.weight".into());
        } else if inverse.contains("token_embd.weight") && !has_tensor("output.weight") {
            return Err("a tied Hadamard output requires version 2 and tied_output=true".into());
        }
        Ok(Some(HadamardContract { version, block_size: block, signs, folded, inverse, gdn_v_grouped, tied_output }))
    }

    pub fn is_folded(&self, name: &str) -> bool { self.folded.contains(name) }
    pub fn is_inverse(&self, name: &str) -> bool { self.inverse.contains(name) }

    /// The sign vector for an activation of `width`: `Ok(None)` under identity mode; an ERROR under
    /// explicit mode when the table has no entry — the fork refuses that too (`llama-model.cpp:2080`),
    /// because a missing entry read as "no signs" silently changes the function.
    pub fn signs_for(&self, width: usize) -> Result<Option<&[f32]>, String> {
        if width % self.block_size != 0 {
            return Err(format!("prism.hadamard block size {} does not divide input dimension {width}", self.block_size));
        }
        if self.signs.is_empty() { return Ok(None); }
        self.signs.get(&width).map(|v| Some(v.as_slice()))
            .ok_or_else(|| format!("prism.hadamard has no sign vector for width {width}"))
    }

    /// Every tensor whose stored values live in the rotated basis — what [`PrismLock`] guards.
    pub fn rotated_tensors(&self) -> HashSet<String> {
        self.folded.union(&self.inverse).cloned().collect()
    }
}

/// Blockwise **normalized Sylvester–Walsh–Hadamard** transform, in place, over consecutive blocks of
/// `n` values (`x.len()` a multiple of `n`, `n` a power of two).
///
/// Arithmetic order mirrors the fork's Metal kernel (`kernel_fwht_tg`, `kernels/misc.metal`):
/// prescale by `1/√n` on load, then radix-2 butterflies at strides 1, 2, 4, …, n/2 with
/// `(a, b) → (a + b, a − b)`. Every stage pairs the same two values in the same order, so the result
/// is the same sum tree — for `n = 1024` the prescale is `2⁻⁵`, exact.
pub fn fwht_normalized(x: &mut [f32], n: usize) {
    assert!(n.is_power_of_two() && x.len() % n == 0, "fwht: {} values in blocks of {n}", x.len());
    let scale = 1.0 / (n as f32).sqrt();
    for blk in x.chunks_exact_mut(n) {
        for v in blk.iter_mut() { *v *= scale; }
        let mut h = 1;
        while h < n {
            for i in (0..n).step_by(2 * h) {
                for j in i..i + h {
                    let (a, b) = (blk[j], blk[j + h]);
                    blk[j] = a + b;
                    blk[j + h] = a - b;
                }
            }
            h *= 2;
        }
    }
}

/// The fork's `build_lora_mm` activation transform on the host, row by row over `width`-wide rows:
/// optional tiled→grouped permutation `(hd, nk, rep)`, then `s ⊙`, then blockwise `H`.
///
/// The permutation takes feature `d + hd·(k + nk·r)` (tiled head `k + nk·r`) to `d + hd·(r + rep·k)`
/// (grouped head `k·rep + r`) — ggml's `permute(x[hd,nk,rep,·], 0,2,1,3)`, `llama-graph.cpp:1561-1567`.
pub fn forward_rows(x: &mut [f32], width: usize, block: usize, signs: Option<&[f32]>, perm: Option<(usize, usize, usize)>) {
    assert!(width % block == 0 && x.len() % width == 0);
    let mut tmp = vec![0.0f32; width];
    for row in x.chunks_exact_mut(width) {
        if let Some((hd, nk, rep)) = perm {
            assert_eq!(hd * nk * rep, width, "GDN permutation geometry");
            for k in 0..nk { for r in 0..rep { for d in 0..hd {
                tmp[d + hd * (r + rep * k)] = row[d + hd * (k + nk * r)];
            }}}
            row.copy_from_slice(&tmp);
        }
        if let Some(s) = signs { for (v, s) in row.iter_mut().zip(s) { *v *= *s; } }
        fwht_normalized(row, block);
    }
}

/// The inverse after a latent-table lookup, `h = s ⊙ (H z)` (`qwen35.cpp:645-648`): `H` is its own
/// inverse (normalized, symmetric) and `s` is ±1, so this undoes [`forward_rows`] without a permutation.
pub fn inverse_rows(x: &mut [f32], width: usize, block: usize, signs: Option<&[f32]>) {
    assert!(width % block == 0 && x.len() % width == 0);
    for row in x.chunks_exact_mut(width) {
        fwht_normalized(row, block);
        if let Some(s) = signs { for (v, s) in row.iter_mut().zip(s) { *v *= *s; } }
    }
}

/// **PTQ1_0** — `dequantize_row_ptq1_0` (`ggml-quants.c:2255`), same order and arithmetic: the 24
/// `qs` bytes are consumed in stages of 32/16/8 bytes (only the 16- and 8-byte stages fit in 24),
/// each stage emitting its 5 trits per byte trit-major; then `qh`'s 4 trits per byte. A trit is
/// `((q·3ⁿ mod 256)·3) >> 8`, value `(t − 1)·d`.
pub fn deq_ptq1_0(raw: &[u8], n: usize) -> Vec<f32> {
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    const STAGES: [usize; 3] = [32, 16, 8];
    let mut out = Vec::with_capacity(n);
    for blk in raw.chunks_exact(PTQ1_0_BYTES).take(n / 128) {
        let (qs, qh) = (&blk[0..24], &blk[24..26]);
        let d = crate::rd_f16(&blk[26..28]);
        let mut j = 0usize;
        for &c in &STAGES {
            while j + c <= qs.len() {
                for p in POW3.iter().take(5) {
                    for m in 0..c {
                        let q = qs[j + m].wrapping_mul(*p);
                        out.push((((q as u16 * 3) >> 8) as i32 - 1) as f32 * d);
                    }
                }
                j += c;
            }
        }
        for p in POW3.iter().take(4) {
            for &h in qh {
                let q = h.wrapping_mul(*p);
                out.push((((q as u16 * 3) >> 8) as i32 - 1) as f32 * d);
            }
        }
    }
    out
}

/// Mainline **Q2_0** (group 64) — `dequantize_row_q2_0` (`ggml-quants.c:474`): 18-byte block = f16 d
/// then 16 code bytes; element j is bits `(j%4)·2` of byte `j/4`; value `(q − 1)·d`, q ∈ 0..=3.
pub fn deq_q2_0_g64(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for blk in raw.chunks_exact(18).take(n / 64) {
        let d = crate::rd_f16(&blk[0..2]);
        for j in 0..64 {
            let q = ((blk[2 + j / 4] >> ((j % 4) * 2)) & 3) as i32;
            out.push((q - 1) as f32 * d);
        }
    }
    out
}

/// PTQ1_0 → the group-128 Q2_0 layout (34 B / 128), **losslessly**: the same f16 scale, and each trit
/// `t ∈ {0,1,2}` becomes the 2-bit code `q = t` (Q2_0's value is also `(q − 1)·d`). This is how the
/// GPU runs PTQ1_0 today — through the Q2_0 kernels, at Q2_0's 34-byte footprint rather than
/// PTQ1_0's 28. Element order is the dequantizer's, so decode(transcode(x)) == deq_ptq1_0(x) bit for bit.
pub fn ptq1_0_to_q2_0(raw: &[u8]) -> Result<Vec<u8>, String> {
    if raw.len() % PTQ1_0_BYTES != 0 { return Err(format!("PTQ1_0: {} bytes is not whole 28-byte blocks", raw.len())); }
    let mut out = Vec::with_capacity(raw.len() / PTQ1_0_BYTES * 34);
    let ones = [0x00u8, 0x3C]; // f16 1.0: decode codes with d = 1 → t − 1 exactly
    let mut unit = raw.to_vec();
    for blk in unit.chunks_exact_mut(PTQ1_0_BYTES) { blk[26] = ones[0]; blk[27] = ones[1]; }
    for (bi, blk) in unit.chunks_exact(PTQ1_0_BYTES).enumerate() {
        let t = deq_ptq1_0(blk, 128);
        out.extend_from_slice(&raw[bi * PTQ1_0_BYTES + 26..bi * PTQ1_0_BYTES + 28]);
        let mut qs = [0u8; 32];
        for (j, v) in t.iter().enumerate() {
            let q = (*v as i32 + 1) as u8; // v ∈ {-1, 0, 1}, exactly
            if q > 2 { return Err(format!("PTQ1_0 block {bi}: trit {q} out of range")); }
            qs[j / 4] |= q << ((j % 4) * 2);
        }
        out.extend_from_slice(&qs);
    }
    Ok(out)
}

/// Mainline group-64 Q2_0 → the group-128 layout, **losslessly or not at all**.
///
/// A 128-block needs ONE scale. The two 64-halves merge exactly when their scales are bit-equal, or
/// when one half's codes are all zero-valued (q = 1 → value 0 under any scale, so its scale is free).
/// PrismML say all three Bonsai 2 packings carry the same weights, and a checkpoint that is ternary
/// at group 128 has this property everywhere. Anything else is refused by name — a general
/// group-64 file needs a per-64-scale kernel, which is not written.
pub fn q2_0_g64_to_g128(raw: &[u8]) -> Result<Vec<u8>, String> {
    if raw.len() % 36 != 0 { return Err(format!("Q2_0 (g64): {} bytes is not whole pairs of 18-byte blocks", raw.len())); }
    let mut out = Vec::with_capacity(raw.len() / 36 * 34);
    for (bi, pair) in raw.chunks_exact(36).enumerate() {
        let (a, b) = (&pair[0..18], &pair[18..36]);
        let zero = |h: &[u8]| h[2..18].iter().all(|&c| c == 0x55); // every code q = 1
        let d = if a[0..2] == b[0..2] || zero(b) { [a[0], a[1]] }
                else if zero(a) { [b[0], b[1]] }
                else {
                    return Err(format!(
                        "Q2_0 (group 64) block pair {bi} has two different non-trivial scales ({:#06x} vs \
                         {:#06x}); it cannot be carried at group 128 losslessly and this crate has no \
                         per-64-scale GPU kernel", u16::from_le_bytes([a[0], a[1]]), u16::from_le_bytes([b[0], b[1]])));
                };
        out.extend_from_slice(&d);
        out.extend_from_slice(&a[2..18]);
        out.extend_from_slice(&b[2..18]);
    }
    Ok(out)
}

/// Bytes the group-128 Q2_0 GPU kernels can take, for a tensor stored as `ty` with `cols` inputs per
/// row: `Some(bytes)` for the formats carried there by a lossless transcode, `None` for any other type
/// (the caller's own path applies).
///
/// ⚠ `cols` must be a multiple of 128: the transcodes work on the flat byte stream, and a row whose
/// width is an odd number of 64-blocks would pair its last half-block with the NEXT row's first.
pub fn as_q2_0_g128(ty: u32, raw: &[u8], cols: usize) -> Result<Option<Vec<u8>>, String> {
    if matches!(ty, PTQ1_0 | Q2_0_G64) && cols % 128 != 0 {
        return Err(format!("ggml type {ty}: {cols} inputs per row is not a multiple of 128; no group-128 carrier"));
    }
    match ty {
        PTQ1_0 => ptq1_0_to_q2_0(raw).map(Some),
        Q2_0_G64 => q2_0_g64_to_g128(raw).map(Some),
        _ => Ok(None),
    }
}

/// Reader-side guard over the tensors a Hadamard contract names. See the module docs.
#[derive(Debug, Default)]
pub struct PrismLock { names: HashSet<String>, open: AtomicBool, err: Option<String> }

impl PrismLock {
    pub fn new(meta: &HashMap<String, Meta>, has_tensor: impl Fn(&str) -> bool) -> PrismLock {
        match HadamardContract::from_meta(meta, has_tensor) {
            Ok(None) => PrismLock::default(),
            Ok(Some(c)) => PrismLock { names: c.rotated_tensors(), open: AtomicBool::new(false), err: None },
            // A malformed contract locks EVERY tensor: nothing in the file is safe to run.
            Err(e) => PrismLock { names: HashSet::new(), open: AtomicBool::new(false), err: Some(e) },
        }
    }
    /// Whether this file carries a Hadamard contract (well-formed or not).
    pub fn active(&self) -> bool { !self.names.is_empty() || self.err.is_some() }
    pub fn unlock(&self) -> bool {
        if self.err.is_some() { return false; }
        self.open.store(true, Ordering::SeqCst);
        true
    }
    pub fn check(&self, name: &str) -> Result<(), String> {
        if let Some(e) = &self.err {
            return Err(format!("tensor '{name}': this file's PrismML Hadamard contract is malformed ({e}); refusing every tensor"));
        }
        if self.names.contains(name) && !self.open.load(Ordering::SeqCst) {
            return Err(format!(
                "tensor '{name}' is stored in PrismML's Hadamard-rotated basis (prism.hadamard.* metadata: \
                 Bonsai 2 and later). Its matmul needs the activation transform H(s ⊙ x) — or, for a lookup \
                 table, the inverse s ⊙ (H z) — and this loader has not declared that it applies it. \
                 Running it untransformed produces fluent gibberish, so the read is refused. Runtimes that \
                 implement the transform call GgufSource::unlock_prism_hadamard first."));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_h(n: usize) -> Vec<f32> {
        // The fork's rotation matrix, built exactly as llama-model.cpp:2057-2069 builds it.
        let scale = 1.0 / (n as f32).sqrt();
        let mut m = vec![0.0; n * n];
        for r in 0..n { for c in 0..n {
            m[r * n + c] = if (r & c).count_ones() & 1 == 1 { -scale } else { scale };
        }}
        m
    }

    #[test]
    fn the_butterfly_equals_the_forks_dense_rotation_and_is_its_own_inverse() {
        for &n in &[64usize, 512, 1024, 2048] {
            let x: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0).collect();
            let h = dense_h(n);
            let want: Vec<f64> = (0..n).map(|r| (0..n).map(|c| h[r * n + c] as f64 * x[c] as f64).sum()).collect();
            let mut y = x.clone();
            fwht_normalized(&mut y, n);
            let err = y.iter().zip(&want).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max);
            assert!(err < 1e-5, "n={n}: butterfly vs dense H max|Δ| {err}");
            fwht_normalized(&mut y, n);
            let back = y.iter().zip(&x).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(back < 1e-5, "n={n}: H·H ≠ I, max|Δ| {back}");
        }
    }

    #[test]
    fn inverse_undoes_forward_with_signs_and_the_permutation_is_not_the_identity() {
        let (hd, nk, rep) = (4usize, 2, 3);
        let w = hd * nk * rep; // 24 — use block 8
        let s: Vec<f32> = (0..w).map(|i| if i % 3 == 0 { -1.0 } else { 1.0 }).collect();
        let x: Vec<f32> = (0..w).map(|i| i as f32 * 0.25 - 2.0).collect();
        let mut y = x.clone();
        forward_rows(&mut y, w, 8, Some(&s), None);
        inverse_rows(&mut y, w, 8, Some(&s));
        assert!(y.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-6));
        // The GDN permutation must move something, or a test of it could not fail.
        let mut p = x.clone();
        forward_rows(&mut p, w, 8, None, Some((hd, nk, rep)));
        let mut q = x.clone();
        forward_rows(&mut q, w, 8, None, None);
        assert!(p.iter().zip(&q).any(|(a, b)| (a - b).abs() > 1e-3));
    }

    #[test]
    fn ptq1_0_and_g64_transcodes_decode_bit_identically() {
        // Build a PTQ1_0 block by the fork's own quantizer arithmetic from known trits.
        let trits: Vec<u8> = (0..128).map(|i| ((i * 7 + 3) % 3) as u8).collect();
        let mut blk = vec![0u8; PTQ1_0_BYTES];
        let mut x = 0usize;
        let mut j = 0usize;
        for &c in &[32usize, 16, 8] {
            while j + c <= 24 {
                for m in 0..c {
                    let mut q: u16 = 0;
                    for n in 0..5 { q = q * 3 + trits[x + m + n * c] as u16; }
                    blk[j + m] = ((q * 256 + 242) / 243) as u8;
                }
                x += 5 * c;
                j += c;
            }
        }
        for h in 0..2 {
            let mut q: u16 = 0;
            for m in 0..4 { q = q * 3 + trits[x + h + m * 2] as u16; }
            q *= 3;
            blk[24 + h] = ((q * 256 + 242) / 243) as u8;
        }
        blk[26..28].copy_from_slice(&half::f16::from_f32(0.4375).to_le_bytes());
        let d = deq_ptq1_0(&blk, 128);
        let want: Vec<f32> = trits.iter().map(|&t| (t as i32 - 1) as f32 * 0.4375).collect();
        assert_eq!(d, want, "PTQ1_0 decode");
        let q2 = ptq1_0_to_q2_0(&blk).unwrap();
        assert_eq!(crate::deq_raw(&q2, 128, 42).unwrap(), d, "PTQ1_0 → Q2_0 transcode");

        // g64: two halves with the same scale merge; a zero half merges; different scales refuse.
        let mut g = vec![0u8; 36];
        g[0..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        g[18..20].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        for i in 0..16 { g[2 + i] = (i * 29) as u8; g[20 + i] = (i * 53 + 1) as u8; }
        let m = q2_0_g64_to_g128(&g).unwrap();
        assert_eq!(crate::deq_raw(&m, 128, 42).unwrap(), deq_q2_0_g64(&g, 128));
        g[18..20].copy_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
        assert!(q2_0_g64_to_g128(&g).is_err(), "different non-zero-half scales must refuse");
        for i in 0..16 { g[20 + i] = 0x55; }
        assert_eq!(crate::deq_raw(&q2_0_g64_to_g128(&g).unwrap(), 128, 42).unwrap(), deq_q2_0_g64(&g, 128));
    }

    fn contract_meta() -> HashMap<String, Meta> {
        let mut m = HashMap::new();
        m.insert("prism.hadamard.version".into(), Meta::U(1));
        m.insert("prism.hadamard.block_size".into(), Meta::U(4));
        m.insert("prism.hadamard.transform".into(), Meta::Str("normalized-sylvester-walsh-hadamard".into()));
        m.insert("prism.hadamard.axis".into(), Meta::Str("input-last-dimension".into()));
        m.insert("prism.hadamard.sign_mode".into(), Meta::Str("explicit".into()));
        m.insert("prism.hadamard.weight_names".into(), Meta::Arr(vec![Meta::Str("blk.0.ffn_up.weight".into())]));
        m.insert("prism.hadamard.sign_widths".into(), Meta::Arr(vec![Meta::I(8)]));
        m.insert("prism.hadamard.sign_values".into(), Meta::Arr((0..8).map(|i| Meta::I(if i % 2 == 0 { 1 } else { -1 })).collect()));
        m.insert("prism.hadamard.inverse_weight_names".into(), Meta::Arr(vec![Meta::Str("token_embd.weight".into())]));
        m
    }

    #[test]
    fn the_contract_is_validated_like_the_forks_loader() {
        let has = |n: &str| n == "output.weight";
        let c = HadamardContract::from_meta(&contract_meta(), has).unwrap().unwrap();
        assert!(c.is_folded("blk.0.ffn_up.weight") && c.is_inverse("token_embd.weight"));
        assert!(c.signs_for(8).unwrap().is_some());
        assert!(c.signs_for(16).is_err(), "explicit mode with no entry for a width must refuse");
        let bad = |k: &str, v: Meta| { let mut m = contract_meta(); m.insert(k.into(), v); HadamardContract::from_meta(&m, has).is_err() };
        assert!(bad("prism.hadamard.version", Meta::U(3)));
        assert!(bad("prism.hadamard.block_size", Meta::U(6)));
        assert!(bad("prism.hadamard.transform", Meta::Str("hadamard".into())));
        assert!(bad("prism.hadamard.sign_values", Meta::Arr((0..8).map(|_| Meta::I(2)).collect())));
        assert!(bad("prism.hadamard.weight_names", Meta::Arr(vec![Meta::Str("blk.0.ssm_alpha.weight".into())])));
        assert!(bad("prism.hadamard.inverse_weight_names", Meta::Arr(vec![Meta::Str("output.weight".into())])));
        // A partial contract (no version) is refused, not ignored.
        let mut m = contract_meta(); m.remove("prism.hadamard.version");
        assert!(HadamardContract::from_meta(&m, has).is_err());
        // Tied output (v2) needs no separate head and folds the table forward for the head.
        let mut m = contract_meta();
        m.insert("prism.hadamard.version".into(), Meta::U(2));
        m.insert("prism.hadamard.tied_output".into(), Meta::Bool(true));
        assert!(HadamardContract::from_meta(&m, has).is_err(), "tied output with an output.weight present");
        let c = HadamardContract::from_meta(&m, |_| false).unwrap().unwrap();
        assert!(c.is_folded("token_embd.weight") && c.is_inverse("token_embd.weight"));
    }

    #[test]
    fn the_lock_refuses_rotated_tensors_until_unlocked() {
        let l = PrismLock::new(&contract_meta(), |n| n == "output.weight");
        assert!(l.active());
        assert!(l.check("blk.0.ffn_up.weight").is_err());
        assert!(l.check("token_embd.weight").is_err());
        assert!(l.check("blk.0.attn_norm.weight").is_ok(), "unrotated tensors are not locked");
        assert!(l.unlock());
        assert!(l.check("blk.0.ffn_up.weight").is_ok());
        let mut m = contract_meta(); m.remove("prism.hadamard.version");
        let bad = PrismLock::new(&m, |_| true);
        assert!(bad.check("blk.0.attn_norm.weight").is_err() && !bad.unlock(), "a malformed contract locks everything");
        assert!(!PrismLock::new(&HashMap::new(), |_| true).active());
    }
}
