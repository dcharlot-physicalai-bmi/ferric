//! **GGUF writer** — the byte-for-byte inverse of the reader in [`crate::parse`].
//!
//! `quantize.rs` gave Ferric the ability to produce block-quantized BYTES. This gives it a container
//! to put them in. Without a writer those bytes could only be handed to some other project's tool to
//! be packaged, which is the same dependency-in-the-development-pipeline that `quantize.rs` exists to
//! remove: a container Ferric cannot write is a container Ferric can only produce by asking someone
//! else to produce it.
//!
//! ## The reader is the spec
//!
//! Every field below mirrors [`crate::parse`] and its `Cur` cursor in reverse — same order, same
//! widths, same tags. The round-trip test is against THIS crate's reader, so read that as consistency
//! first (both halves could be wrong in the same way); interop with mainline ggml rests on the
//! constraints called out below, which are taken from the format and from what mainline's loader
//! refuses, not from what our reader happens to tolerate.
//!
//! **Interop measured, 2026-09-04:** a file from this writer (7 KVs across scalar, string and array
//! types; three padded F32 tensors) was read end to end by **llama.cpp's own `llama-gguf … r`**
//! (build 10621, `c1d0e7a00`, ggml 0.22.0) — version 3, alignment 32, all KVs enumerated, tensor
//! offsets 0 / 448 / 864 (i.e. the 440- and 400-byte tensors padded to 32), `ne = (10, 11, 1, 1)`,
//! and its per-element data check passing on every tensor, exit 0. That is the claim the round-trip
//! test cannot make: a reader this crate did not write agrees about where the bytes are.
//!
//! ## Places where two writers with the same shapes produce different files
//!
//! These are the details where the code could plausibly be written the other way, produce a file of
//! exactly the same LENGTH, and be silently wrong:
//!
//! - ⚠ **Everything is little-endian**, the magic included. The reader compares against
//!   `u32::from_le_bytes(*b"GGUF")`, so the four ASCII bytes go out in reading order `G G U F` and
//!   every length, tag, dim and offset is `to_le_bytes`. A big-endian length prefix on a key is the
//!   same four bytes rearranged: same file size, unparseable file.
//! - ⛔ **`TensorInfo::offset` is relative to the DATA SECTION, not to the file.** The reader adds
//!   `data_start` itself (`Gguf::raw`: `self.data_start + t.offset`). The first tensor's offset is
//!   therefore `0`, never the header length. Writing absolute offsets yields a file whose tensor
//!   table looks entirely plausible and whose every tensor reads the wrong bytes.
//! - ⚠ **`dims` go out in GGUF `ne` order — ne0 (fastest-varying) FIRST.** That is the reverse of a
//!   torch/numpy `.shape`: a `[n_out, n_in]` linear weight is stored as `dims = [n_in, n_out]`. This
//!   writer stores what it is given, verbatim, and cannot detect the mistake — the element count is
//!   the product either way, so a reversed pair round-trips through every length check in the format.
//!   Reversing is the caller's job, at the point where a torch shape is converted.
//! - ⚠ **`general.alignment` is simultaneously a KV and a layout decision.** The reader derives
//!   `data_start = align_up(header_end, alignment)` from the KV, so a file whose padding and whose KV
//!   disagree is misread from the first tensor on, with no error. That is why alignment is settable
//!   only through [`GgufWriter::alignment`] — a `kv_u32("general.alignment", …)` is routed there
//!   rather than being stored as an inert number.
//! - ⚠ **Padding goes between tensors as well as before the first.** Each tensor's offset is aligned,
//!   so a tensor whose size is not a multiple of the alignment is followed by pad bytes. Dropping
//!   them still produces a readable, self-consistent-looking file: the tensor table just no longer
//!   describes where the bytes are.
//!
//! ## What the round-trip test CANNOT see
//!
//! ⚠ The reader collapses widths: `Meta::U` holds U8/U16/U32/U64 alike, `Meta::I` holds all four
//! signed widths, `Meta::F` holds F32 and F64. So writing `7u64` where `7u32` was meant produces a
//! different file that reads back identically through [`crate::Meta`], and a round-trip assertion
//! cannot fail on it. The type tags and value widths are therefore pinned separately, by a test that
//! walks the KV section with its own decoder.

use crate::{F8_E4M3_B128, type_size};

// ---- KV type tags. These must equal the match arms of `Cur::val` in lib.rs, the only decoder. ----
const T_U8: u32 = 0;
const T_I8: u32 = 1;
const T_U16: u32 = 2;
const T_I16: u32 = 3;
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;
const T_F64: u32 = 12;

/// Container version this writer emits. The reader accepts any (`let _ver = c.u32()`), but 3 is what
/// the layout above IS: v1 stored tensor counts as u32 and v2 fixed that, so a file claiming 1 or 2
/// with u64 counts is a lie that only mainline's loader would catch.
const VERSION: u32 = 3;

/// The alignment used when the caller sets none, matching the reader's fallback for a file with no
/// `general.alignment` KV.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// ⛔ Mainline ggml refuses a tensor name of `GGML_MAX_NAME` (64) bytes or more, and its own writer
/// truncates into a fixed 64-byte field. A longer name is accepted by this crate's reader and by
/// nothing else, so it is refused here — the failure belongs at the write, not at whichever loader
/// first tries to open the file.
const MAX_NAME: usize = 64;

/// `GGML_MAX_DIMS`. A 5-dimensional tensor has no representation in ggml's own struct.
const MAX_DIMS: usize = 4;

/// One metadata value, in the format's own type vocabulary rather than the reader's widened one.
///
/// Private on purpose: the array variants carry a homogeneous `Vec`, so there is no way to hand the
/// encoder an array whose declared element type disagrees with its elements. GGUF arrays write the
/// element type ONCE, and a heterogeneous array is therefore not a validation failure but an
/// unwritable value.
enum Kv {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    ArrU32(Vec<u32>),
    ArrI32(Vec<i32>),
    ArrU64(Vec<u64>),
    ArrF32(Vec<f32>),
    ArrStr(Vec<String>),
}

impl Kv {
    /// The tag written before the value. For arrays this is [`T_ARRAY`]; the ELEMENT tag is written
    /// after it, inside the value, which is why the two are produced by different functions.
    fn tag(&self) -> u32 {
        match self {
            Kv::U8(_) => T_U8,
            Kv::I8(_) => T_I8,
            Kv::U16(_) => T_U16,
            Kv::I16(_) => T_I16,
            Kv::U32(_) => T_U32,
            Kv::I32(_) => T_I32,
            Kv::U64(_) => T_U64,
            Kv::I64(_) => T_I64,
            Kv::F32(_) => T_F32,
            Kv::F64(_) => T_F64,
            Kv::Bool(_) => T_BOOL,
            Kv::Str(_) => T_STRING,
            Kv::ArrU32(_) | Kv::ArrI32(_) | Kv::ArrU64(_) | Kv::ArrF32(_) | Kv::ArrStr(_) => T_ARRAY,
        }
    }

    /// The value bytes, tag already written by the caller.
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Kv::U8(v) => out.push(*v),
            Kv::I8(v) => out.push(*v as u8),
            Kv::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
            Kv::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
            Kv::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
            Kv::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
            Kv::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
            Kv::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
            // ⚠ `to_le_bytes` on the float, i.e. the IEEE-754 bit pattern little-endian — the
            // reader does `f32::from_bits(self.u32())`, which is the same thing and only the same
            // thing because the bit pattern is what goes on the wire.
            Kv::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
            Kv::F64(v) => out.extend_from_slice(&v.to_le_bytes()),
            // One byte, not four: the reader takes `self.u8() != 0`.
            Kv::Bool(v) => out.push(u8::from(*v)),
            Kv::Str(v) => push_str(out, v),
            Kv::ArrU32(v) => push_arr(out, T_U32, v.len(), |o| v.iter().for_each(|x| o.extend_from_slice(&x.to_le_bytes()))),
            Kv::ArrI32(v) => push_arr(out, T_I32, v.len(), |o| v.iter().for_each(|x| o.extend_from_slice(&x.to_le_bytes()))),
            Kv::ArrU64(v) => push_arr(out, T_U64, v.len(), |o| v.iter().for_each(|x| o.extend_from_slice(&x.to_le_bytes()))),
            Kv::ArrF32(v) => push_arr(out, T_F32, v.len(), |o| v.iter().for_each(|x| o.extend_from_slice(&x.to_le_bytes()))),
            Kv::ArrStr(v) => push_arr(out, T_STRING, v.len(), |o| v.iter().for_each(|x| push_str(o, x))),
        }
    }
}

/// A GGUF string: u64 byte length, then the UTF-8 bytes.
///
/// ⚠ **Length-prefixed, NOT nul-terminated**, and the length counts BYTES not chars — `s.len()` in
/// Rust, which is already bytes. Appending a terminator would shift every subsequent field by one
/// and leave the file the same shape.
fn push_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// An array value: element type tag (u32), count (u64), then the elements back to back with **no
/// per-element tag**. The reader's `val(9)` arm reads exactly this.
fn push_arr(out: &mut Vec<u8>, elem: u32, n: usize, items: impl FnOnce(&mut Vec<u8>)) {
    out.extend_from_slice(&elem.to_le_bytes());
    out.extend_from_slice(&(n as u64).to_le_bytes());
    items(out);
}

/// Round `x` up to a multiple of `a`. Mirrors the reader's `c.p.div_ceil(align) * align` exactly —
/// the same expression rather than a bit-mask equivalent, so alignment can never be a power of two
/// here and a plain multiple there.
fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

/// Elements per block for a ggml type, derived from [`type_size`] rather than tabulated again.
///
/// ⛔ This used to probe [`type_size`] for the smallest element count with a non-zero size, because
/// that function DIVIDED FIRST (`n / 32 * 34`) and answered `Ok(0)` for a partial block —
/// `type_size(Q8_0, 16)` was 0 bytes, not 34. A writer checking only
/// `bytes.len() == type_size(ty, n)` would have accepted an empty `Vec` for a 16-element Q8_0
/// tensor and emitted a file whose tensor was silently zero-length.
///
/// Writing this writer is what surfaced that: nothing in the reader had reason to pass a partial
/// block, so the truncation was only reachable from code that did not exist yet. `type_size` now
/// REFUSES a partial block, and [`crate::block_elems`] is the accessor that made the probe
/// unnecessary. Kept as a thin wrapper so the failure here still reads as a writer-side refusal.
fn block_elems(ty: u32) -> Result<usize, String> {
    // Ask type_size whether it knows the type at all; block_elems answers for anything.
    type_size(ty, crate::block_elems(ty))?;
    Ok(crate::block_elems(ty))
}

struct Tensor {
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
    bytes: Vec<u8>,
}

/// Builds a GGUF file in memory: metadata, then tensors, then [`GgufWriter::finish`].
///
/// Errors are **sticky**, not returned per call — the builder methods chain, and a `Result` on each
/// would make the common path (a dozen KVs and a hundred tensors) unreadable. The first failure is
/// kept and surfaced by `finish`/`write_to`, so a rejected tensor cannot reach a file; [`error`] is
/// there for a caller that wants to look before then.
///
/// [`error`]: GgufWriter::error
pub struct GgufWriter {
    /// Insertion-ordered, and deduplicated on key — see [`GgufWriter::set`].
    kv: Vec<(String, Kv)>,
    tensors: Vec<Tensor>,
    align: u64,
    err: Option<String>,
}

impl GgufWriter {
    /// A writer seeded with the two KVs every GGUF is expected to carry: `general.architecture` and
    /// `general.alignment` (32).
    ///
    /// The architecture string is what a loader dispatches on — it selects the whole `<arch>.*` KV
    /// namespace (`llama.block_count`, `qwen3moe.expert_count`, …) — so it is a constructor argument
    /// rather than one more optional KV that a file can be built without.
    pub fn new(architecture: &str) -> Self {
        let mut w = GgufWriter { kv: Vec::new(), tensors: Vec::new(), align: DEFAULT_ALIGNMENT, err: None };
        w.kv_str("general.architecture", architecture);
        w.kv.push(("general.alignment".to_string(), Kv::U32(DEFAULT_ALIGNMENT as u32)));
        w
    }

    /// The first error, if the build has already failed.
    pub fn error(&self) -> Option<&str> {
        self.err.as_deref()
    }

    /// Record a failure. FIRST one wins: later errors are usually consequences of the first, and the
    /// original is the one that names the caller's actual mistake.
    fn fail(&mut self, msg: String) -> &mut Self {
        if self.err.is_none() {
            self.err = Some(msg);
        }
        self
    }

    /// Insert or REPLACE a metadata key.
    ///
    /// ⛔ Replacement, not append: mainline's loader rejects a file containing the same key twice,
    /// and this crate's reader (a `HashMap`) would silently keep whichever came last. Emitting both
    /// would make a file that reads fine here and nowhere else — the worst of the three outcomes.
    fn set(&mut self, key: &str, v: Kv) -> &mut Self {
        if key == "general.alignment" {
            // ⚠ Alignment is a layout decision, not a number in a table. Route it so that the KV and
            // the padding can never disagree; see `alignment`.
            return match v {
                Kv::U8(a) => self.alignment(a as u64),
                Kv::U16(a) => self.alignment(a as u64),
                Kv::U32(a) => self.alignment(a as u64),
                Kv::U64(a) => self.alignment(a),
                _ => self.fail("general.alignment must be an unsigned integer".to_string()),
            };
        }
        if key.is_empty() {
            return self.fail("a metadata key may not be empty".to_string());
        }
        match self.kv.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = v,
            None => self.kv.push((key.to_string(), v)),
        }
        self
    }

    /// Set the data-section alignment, updating both the padding and the `general.alignment` KV.
    ///
    /// Must be a power of two: mainline computes `offset % alignment` with a mask and rejects
    /// anything else outright. 32 is the default and what essentially every published checkpoint
    /// uses; larger values exist to page- or cache-align the data section.
    pub fn alignment(&mut self, align: u64) -> &mut Self {
        if align == 0 || !align.is_power_of_two() {
            return self.fail(format!("general.alignment must be a power of two, got {align}"));
        }
        self.align = align;
        let v = if align <= u32::MAX as u64 { Kv::U32(align as u32) } else { Kv::U64(align) };
        match self.kv.iter_mut().find(|(k, _)| k == "general.alignment") {
            Some(slot) => slot.1 = v,
            None => self.kv.push(("general.alignment".to_string(), v)),
        }
        self
    }

    pub fn kv_u8(&mut self, key: &str, v: u8) -> &mut Self { self.set(key, Kv::U8(v)) }
    pub fn kv_i8(&mut self, key: &str, v: i8) -> &mut Self { self.set(key, Kv::I8(v)) }
    pub fn kv_u16(&mut self, key: &str, v: u16) -> &mut Self { self.set(key, Kv::U16(v)) }
    pub fn kv_i16(&mut self, key: &str, v: i16) -> &mut Self { self.set(key, Kv::I16(v)) }
    pub fn kv_u32(&mut self, key: &str, v: u32) -> &mut Self { self.set(key, Kv::U32(v)) }
    pub fn kv_i32(&mut self, key: &str, v: i32) -> &mut Self { self.set(key, Kv::I32(v)) }
    pub fn kv_u64(&mut self, key: &str, v: u64) -> &mut Self { self.set(key, Kv::U64(v)) }
    pub fn kv_i64(&mut self, key: &str, v: i64) -> &mut Self { self.set(key, Kv::I64(v)) }
    pub fn kv_f32(&mut self, key: &str, v: f32) -> &mut Self { self.set(key, Kv::F32(v)) }
    pub fn kv_f64(&mut self, key: &str, v: f64) -> &mut Self { self.set(key, Kv::F64(v)) }
    pub fn kv_bool(&mut self, key: &str, v: bool) -> &mut Self { self.set(key, Kv::Bool(v)) }
    pub fn kv_str(&mut self, key: &str, v: &str) -> &mut Self { self.set(key, Kv::Str(v.to_string())) }

    pub fn kv_arr_u32(&mut self, key: &str, v: &[u32]) -> &mut Self { self.set(key, Kv::ArrU32(v.to_vec())) }
    pub fn kv_arr_i32(&mut self, key: &str, v: &[i32]) -> &mut Self { self.set(key, Kv::ArrI32(v.to_vec())) }
    pub fn kv_arr_u64(&mut self, key: &str, v: &[u64]) -> &mut Self { self.set(key, Kv::ArrU64(v.to_vec())) }
    pub fn kv_arr_f32(&mut self, key: &str, v: &[f32]) -> &mut Self { self.set(key, Kv::ArrF32(v.to_vec())) }
    pub fn kv_arr_str(&mut self, key: &str, v: &[String]) -> &mut Self { self.set(key, Kv::ArrStr(v.to_vec())) }

    /// Append a tensor whose bytes are already in the on-disk layout of `ggml_type`.
    ///
    /// `dims` is in **GGUF `ne` order, ne0 first** — the reverse of a torch `.shape`. See the module
    /// header: nothing downstream can catch a reversed pair, because the element count is the product
    /// either way.
    ///
    /// `bytes` must be exactly `type_size(ggml_type, dims.product())` long, and the element count
    /// must be a whole number of that type's blocks. Both are checked here rather than at `finish`
    /// so the error names the tensor while the caller still knows which one it was building.
    pub fn tensor(&mut self, name: &str, dims: &[u64], ggml_type: u32, bytes: Vec<u8>) -> &mut Self {
        if name.is_empty() {
            return self.fail("a tensor name may not be empty".to_string());
        }
        if name.len() >= MAX_NAME {
            return self.fail(format!(
                "tensor name '{name}' is {} bytes; ggml's name field holds {} (GGML_MAX_NAME - 1), so \
                 a longer name is truncated or refused by every mainline loader",
                name.len(), MAX_NAME - 1));
        }
        if self.tensors.iter().any(|t| t.name == name) {
            return self.fail(format!(
                "tensor '{name}' was added twice; duplicate names make the tensor table ambiguous and \
                 a lookup returns whichever copy the reader happens to scan first"));
        }
        if dims.is_empty() || dims.len() > MAX_DIMS {
            return self.fail(format!(
                "tensor '{name}' has {} dimensions; ggml holds 1..={MAX_DIMS} (GGML_MAX_DIMS)", dims.len()));
        }
        if dims.contains(&0) {
            return self.fail(format!("tensor '{name}' has a zero-length dimension in {dims:?}"));
        }
        // ⛔ The internal id, not a file id. `resolve_type_42` hands 1042 to callers for a tensor the
        // FILE labels 42; writing 1042 back out would produce a type id no other reader has heard of.
        if ggml_type == F8_E4M3_B128 {
            return self.fail(format!(
                "tensor '{name}': {F8_E4M3_B128} is this crate's INTERNAL id for F8_E4M3_B128. The \
                 file must say 42 — the reader resolves it back by stride."));
        }
        let n = match dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d)) {
            Some(n) => n as usize,
            None => return self.fail(format!("tensor '{name}': dims {dims:?} overflow a u64 element count")),
        };
        let block = match block_elems(ggml_type) {
            Ok(b) => b,
            Err(e) => return self.fail(format!("tensor '{name}': {e}")),
        };
        if n % block != 0 {
            return self.fail(format!(
                "tensor '{name}' has {n} elements, which is not a whole number of {block}-element \
                 blocks for ggml type {ggml_type}. A partial block has no encoding, and type_size \
                 would report the size of the {} complete ones as if that were the whole tensor.",
                n / block));
        }
        let want = match type_size(ggml_type, n) {
            Ok(w) => w,
            Err(e) => return self.fail(format!("tensor '{name}': {e}")),
        };
        if bytes.len() != want {
            return self.fail(format!(
                "tensor '{name}' (ggml type {ggml_type}, {n} elements as {dims:?}) needs {want} bytes \
                 by this crate's block layout, got {}", bytes.len()));
        }
        self.tensors.push(Tensor { name: name.to_string(), dims: dims.to_vec(), ggml_type, bytes });
        self
    }

    /// F32 convenience: `data` goes out as little-endian `f32`, ggml type 0.
    ///
    /// The length check is the same one [`GgufWriter::tensor`] applies, so a `data` that disagrees
    /// with `dims` is refused there — this does not paper over it by trusting the slice.
    pub fn tensor_f32(&mut self, name: &str, dims: &[u64], data: &[f32]) -> &mut Self {
        let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.tensor(name, dims, 0, bytes)
    }

    /// Serialize to bytes: header, metadata, tensor table, padding, tensor data.
    pub fn finish(self) -> Result<Vec<u8>, String> {
        if let Some(e) = self.err {
            return Err(e);
        }
        let mut out: Vec<u8> = Vec::new();
        // ⚠ The magic is the four ASCII bytes in reading order. The reader compares them as
        // `u32::from_le_bytes(*b"GGUF")`, so writing that u32 with `to_be_bytes` would spell "FUGG".
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&VERSION.to_le_bytes());
        // ⚠ Tensor count FIRST, then KV count. Both u64. Swapping them is invisible in any file where
        // the two happen to be equal, and catastrophic in every other.
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.kv.len() as u64).to_le_bytes());
        for (k, v) in &self.kv {
            push_str(&mut out, k);
            out.extend_from_slice(&v.tag().to_le_bytes());
            v.encode(&mut out);
        }

        // Offsets are DATA-RELATIVE, so they can be laid out before the header's own length is known
        // — there is no fixed point to solve here, which is exactly the property that makes an
        // absolute-offset writer easy to write by mistake.
        let mut offsets = Vec::with_capacity(self.tensors.len());
        let mut at = 0u64;
        for t in &self.tensors {
            offsets.push(at);
            at = align_up(at + t.bytes.len() as u64, self.align);
        }

        for (t, &off) in self.tensors.iter().zip(&offsets) {
            push_str(&mut out, &t.name);
            // n_dims is u32 while each dim is u64. Not a typo: the counts in the header are u64 and
            // this one is not.
            out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for &d in &t.dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&t.ggml_type.to_le_bytes());
            out.extend_from_slice(&off.to_le_bytes());
        }

        // The data section starts at the first aligned position at or after the header's end — the
        // reader recomputes this from the header it just parsed, so the padding here is what makes
        // the two agree.
        let data_start = align_up(out.len() as u64, self.align) as usize;
        out.resize(data_start, 0);
        for (t, &off) in self.tensors.iter().zip(&offsets) {
            debug_assert_eq!(out.len(), data_start + off as usize, "tensor '{}' laid out off-plan", t.name);
            out.extend_from_slice(&t.bytes);
            // Pad up to the next tensor's start. The last tensor is padded too, matching mainline —
            // the reader tolerates trailing bytes, and a data section that is a whole number of
            // alignment units is what a memory-mapping loader expects.
            out.resize(data_start + align_up(off + t.bytes.len() as u64, self.align) as usize, 0);
        }
        Ok(out)
    }

    /// Serialize and write to `path`.
    ///
    /// `Result<_, String>` rather than `io::Result` because the failures are of two kinds — a
    /// rejected tensor and a failed write — and this crate reports both as strings everywhere else
    /// (`GgufFile::open` maps its io errors the same way).
    pub fn write_to(self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        let path = path.as_ref();
        let bytes = self.finish()?;
        std::fs::write(path, &bytes).map_err(|e| format!("{}: {e}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Gguf, GgufFile, Meta, parse};

    fn tmp(tag: &str) -> std::path::PathBuf {
        // No tempfile crate in the vendored set, and none is needed: pid plus a per-call tag is
        // unique within a run, and each test removes its own file.
        std::env::temp_dir().join(format!("ferric-gguf-write-{}-{tag}.gguf", std::process::id()))
    }

    fn u(g: &Gguf, k: &str) -> u64 {
        match g.metadata.get(k) { Some(Meta::U(v)) => *v, o => panic!("{k}: expected U, got {o:?}") }
    }
    fn i(g: &Gguf, k: &str) -> i64 {
        match g.metadata.get(k) { Some(Meta::I(v)) => *v, o => panic!("{k}: expected I, got {o:?}") }
    }
    fn f(g: &Gguf, k: &str) -> f64 {
        match g.metadata.get(k) { Some(Meta::F(v)) => *v, o => panic!("{k}: expected F, got {o:?}") }
    }
    fn b(g: &Gguf, k: &str) -> bool {
        match g.metadata.get(k) { Some(Meta::Bool(v)) => *v, o => panic!("{k}: expected Bool, got {o:?}") }
    }
    fn s<'a>(g: &'a Gguf, k: &str) -> &'a str {
        match g.metadata.get(k) { Some(Meta::Str(v)) => v, o => panic!("{k}: expected Str, got {o:?}") }
    }
    fn arr<'a>(g: &'a Gguf, k: &str) -> &'a [Meta] {
        match g.metadata.get(k) { Some(Meta::Arr(v)) => v, o => panic!("{k}: expected Arr, got {o:?}") }
    }

    /// Deterministic bytes with no repeating period at 32 or 256, so a tensor read at the wrong
    /// offset or with a dropped pad byte cannot coincidentally match.
    fn blob(n: usize, seed: u8) -> Vec<u8> {
        (0..n).map(|k| (k as u32).wrapping_mul(37).wrapping_add(seed as u32) as u8 ^ 0x5A).collect()
    }

    /// A writer with one of every KV type and a spread of tensor types, used by several tests.
    fn fixture() -> GgufWriter {
        let mut w = GgufWriter::new("ferric-test");
        w.kv_u8("t.u8", 200)
            .kv_i8("t.i8", -100)
            .kv_u16("t.u16", 60_000)
            .kv_i16("t.i16", -30_000)
            .kv_u32("t.u32", 4_000_000_000)
            .kv_i32("t.i32", -2_000_000_000)
            .kv_u64("t.u64", u64::MAX / 3)
            .kv_i64("t.i64", i64::MIN + 7)
            .kv_f32("t.f32", -0.15625)
            .kv_f64("t.f64", std::f64::consts::PI)
            .kv_bool("t.bool.t", true)
            .kv_bool("t.bool.f", false)
            .kv_str("t.str", "a string with UTF-8: αβγ ⛔")
            .kv_arr_u32("t.arr.u32", &[1, 2, 3, u32::MAX])
            .kv_arr_i32("t.arr.i32", &[-1, 0, i32::MIN])
            .kv_arr_u64("t.arr.u64", &[u64::MAX, 0])
            .kv_arr_f32("t.arr.f32", &[0.5, -0.25, 1.0])
            .kv_arr_str("t.arr.str", &["<s>".to_string(), "▁the".to_string(), String::new()]);
        // dims deliberately distinct and non-square: a reversed pair changes the tensor table
        // without changing the element count, so only an asymmetric shape can catch it.
        w.tensor_f32("emb.weight", &[3, 5], &(0..15).map(|k| k as f32 * 0.5 - 2.0).collect::<Vec<_>>());
        w.tensor_f32("blk.0.attn_q.weight", &[2, 3, 4], &(0..24).map(|k| -(k as f32)).collect::<Vec<_>>());
        // Q8_0: 34 bytes per 32 elements, so 320 elements is 340 bytes — NOT a multiple of 32, which
        // is what forces real padding between this tensor and the next.
        w.tensor("blk.0.ffn_down.weight", &[64, 5], 8, blob(340, 0x11));
        // F16: 2 bytes an element, block of one.
        w.tensor("output_norm.weight", &[7], 1, blob(14, 0x22));
        w
    }

    /// The primary test: everything written comes back out of this crate's own reader unchanged, both
    /// from memory and from a file on disk.
    ///
    /// ⚠ Consistency, not interop — `parse` and this writer could misread the layout in the same way
    /// and still agree. What it does pin is that the two halves of THIS crate describe one format, and
    /// with the tag/width walk below it pins the format itself against the spec.
    #[test]
    fn everything_written_comes_back_identical_through_this_crates_reader() {
        let bytes = fixture().finish().expect("the fixture must serialize");
        let g = parse(bytes.clone()).expect("our own reader must accept our own file");

        assert_eq!(s(&g, "general.architecture"), "ferric-test");
        assert_eq!(u(&g, "general.alignment"), 32);
        assert_eq!(u(&g, "t.u8"), 200);
        assert_eq!(i(&g, "t.i8"), -100);
        assert_eq!(u(&g, "t.u16"), 60_000);
        assert_eq!(i(&g, "t.i16"), -30_000);
        assert_eq!(u(&g, "t.u32"), 4_000_000_000);
        assert_eq!(i(&g, "t.i32"), -2_000_000_000);
        assert_eq!(u(&g, "t.u64"), u64::MAX / 3);
        assert_eq!(i(&g, "t.i64"), i64::MIN + 7);
        // f32 -> f64 is exact, so this is an equality and not an epsilon.
        assert_eq!(f(&g, "t.f32"), -0.15625f32 as f64);
        assert_eq!(f(&g, "t.f64"), std::f64::consts::PI);
        assert!(b(&g, "t.bool.t"));
        assert!(!b(&g, "t.bool.f"));
        assert_eq!(s(&g, "t.str"), "a string with UTF-8: αβγ ⛔");

        let au: Vec<u64> = arr(&g, "t.arr.u32").iter().map(|m| match m { Meta::U(v) => *v, o => panic!("{o:?}") }).collect();
        assert_eq!(au, vec![1, 2, 3, u32::MAX as u64]);
        let ai: Vec<i64> = arr(&g, "t.arr.i32").iter().map(|m| match m { Meta::I(v) => *v, o => panic!("{o:?}") }).collect();
        assert_eq!(ai, vec![-1, 0, i32::MIN as i64]);
        let a64: Vec<u64> = arr(&g, "t.arr.u64").iter().map(|m| match m { Meta::U(v) => *v, o => panic!("{o:?}") }).collect();
        assert_eq!(a64, vec![u64::MAX, 0]);
        let af: Vec<f64> = arr(&g, "t.arr.f32").iter().map(|m| match m { Meta::F(v) => *v, o => panic!("{o:?}") }).collect();
        assert_eq!(af, vec![0.5, -0.25, 1.0]);
        let as_: Vec<&str> = arr(&g, "t.arr.str").iter().map(|m| match m { Meta::Str(v) => v.as_str(), o => panic!("{o:?}") }).collect();
        assert_eq!(as_, vec!["<s>", "▁the", ""]);
        assert_eq!(g.metadata.len(), 20, "a key was dropped or invented");

        // Tensors: order, dims (in ne order, unreversed), type and every byte.
        let names: Vec<&str> = g.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["emb.weight", "blk.0.attn_q.weight", "blk.0.ffn_down.weight", "output_norm.weight"]);
        assert_eq!(g.tensor("emb.weight").unwrap().dims, vec![3, 5], "dims must survive in ne order");
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().dims, vec![2, 3, 4]);
        assert_eq!(g.tensor("blk.0.ffn_down.weight").unwrap().dims, vec![64, 5]);
        assert_eq!(g.tensor("blk.0.ffn_down.weight").unwrap().ggml_type, 8);
        assert_eq!(g.tensor("output_norm.weight").unwrap().ggml_type, 1);
        assert_eq!(g.raw("blk.0.ffn_down.weight").unwrap(), blob(340, 0x11));
        assert_eq!(g.raw("output_norm.weight").unwrap(), blob(14, 0x22));
        let emb: Vec<f32> = (0..15).map(|k| k as f32 * 0.5 - 2.0).collect();
        assert_eq!(g.dequant("emb.weight").unwrap(), emb);
        assert_eq!(g.dequant("blk.0.attn_q.weight").unwrap(), (0..24).map(|k| -(k as f32)).collect::<Vec<f32>>());

        // Same file through the lazy file-backed reader, which seeks rather than slices — a different
        // path to the same bytes, and the one real loaders use.
        let path = tmp("roundtrip");
        fixture().write_to(&path).expect("write_to");
        assert_eq!(std::fs::read(&path).unwrap(), bytes, "write_to and finish must produce the same file");
        let file = GgufFile::open(&path).expect("GgufFile::open must accept our file");
        assert_eq!(file.tensors.len(), 4);
        assert_eq!(file.raw("blk.0.ffn_down.weight").unwrap(), blob(340, 0x11));
        assert_eq!(file.dequant("emb.weight").unwrap(), emb);
        assert_eq!(file.tensor("blk.0.attn_q.weight").unwrap().dims, vec![2, 3, 4]);
        let _ = std::fs::remove_file(&path);
    }

    /// Every tensor must START on an alignment boundary, which means padding after any tensor whose
    /// size is not a multiple of the alignment — and the data section itself must start on one.
    ///
    /// ⚠ The 340-byte Q8_0 tensor is the subject: 340 = 10·32 + 20, so it is followed by 12 pad bytes
    /// at align 32 and by 44 at align 64. A file with the padding dropped is still parseable and
    /// still self-consistent-looking; only the offsets give it away.
    #[test]
    fn every_tensor_offset_is_a_multiple_of_the_declared_alignment() {
        for align in [32u64, 64, 4096] {
            let mut w = fixture();
            w.alignment(align);
            let path = tmp(&format!("align{align}"));
            w.write_to(&path).expect("write");
            let g = parse(std::fs::read(&path).unwrap()).expect("parse");
            assert_eq!(u(&g, "general.alignment"), align, "the KV must state the alignment actually used");
            for t in &g.tensors {
                assert_eq!(t.offset % align, 0, "tensor '{}' starts at {} which is not {align}-aligned", t.name, t.offset);
            }
            assert_eq!(g.tensors[0].offset, 0, "the first offset is DATA-relative, so it is always 0");
            // Cross-check the absolute position: data_start is itself aligned, and the bytes at
            // data_start + offset are the tensor's. A dropped pad byte moves these.
            //
            // ⛔ EVERY tensor, and the first one above all. Mutation-testing this file found that a
            // writer which pads BETWEEN tensors but not before the FIRST one displaces only tensor 0
            // — the pad-to-next-boundary after it re-syncs everything downstream — so a spot check of
            // the last two tensors passed a file whose first tensor was 24 bytes out of place.
            let file = GgufFile::open(&path).unwrap();
            assert_eq!(file.data_start() % align, 0, "the data section must begin on an alignment boundary");
            assert_eq!(file.dequant("emb.weight").unwrap(),
                       (0..15).map(|k| k as f32 * 0.5 - 2.0).collect::<Vec<f32>>());
            assert_eq!(file.dequant("blk.0.attn_q.weight").unwrap(),
                       (0..24).map(|k| -(k as f32)).collect::<Vec<f32>>());
            assert_eq!(file.raw("blk.0.ffn_down.weight").unwrap(), blob(340, 0x11));
            assert_eq!(file.raw("output_norm.weight").unwrap(), blob(14, 0x22));
            let _ = std::fs::remove_file(&path);
        }
    }

    /// A tensor whose bytes do not match `dims` × the type's block size must be refused, and refused
    /// with the two numbers that let the caller see which side is wrong.
    #[test]
    fn a_tensor_whose_byte_count_disagrees_with_its_dims_is_refused() {
        // F32, 3·5 = 15 elements = 60 bytes. One byte short.
        let mut w = GgufWriter::new("t");
        w.tensor("a", &[3, 5], 0, vec![0u8; 59]);
        let e = w.finish().unwrap_err();
        assert!(e.contains("needs 60 bytes") && e.contains("got 59"), "unexpected: {e}");

        // Q8_0, 64 elements = 2 blocks = 68 bytes. Handed a block too many.
        let mut w = GgufWriter::new("t");
        w.tensor("a", &[64], 8, vec![0u8; 102]);
        let e = w.finish().unwrap_err();
        assert!(e.contains("needs 68 bytes") && e.contains("got 102"), "unexpected: {e}");

        // ⛔ The case a bare `bytes.len() == type_size(...)` check accepts: 16 elements is HALF a
        // Q8_0 block, and type_size divides first, so it reports 0 bytes. An empty Vec would have
        // "matched" and produced a file with a zero-length tensor in it.
        let mut w = GgufWriter::new("t");
        w.tensor("a", &[16], 8, Vec::new());
        let e = w.finish().unwrap_err();
        assert!(e.contains("not a whole number of 32-element blocks"), "unexpected: {e}");

        // The convenience path applies the same check rather than trusting the slice's length.
        let mut w = GgufWriter::new("t");
        w.tensor_f32("a", &[4, 4], &[0.0; 15]);
        assert!(w.finish().unwrap_err().contains("needs 64 bytes"), "tensor_f32 must be checked too");

        // A correct tensor is not refused — otherwise the three assertions above are satisfied by a
        // writer that rejects everything.
        let mut ok = GgufWriter::new("t");
        ok.tensor("a", &[64], 8, vec![0u8; 68]);
        assert!(ok.finish().is_ok(), "a correctly-sized tensor must be accepted");
    }

    /// The malformed inputs that produce a file no other loader will open.
    #[test]
    fn structurally_invalid_tensors_and_alignments_are_refused() {
        let cases: Vec<(&str, Box<dyn Fn(&mut GgufWriter)>, &str)> = vec![
            ("empty name", Box::new(|w: &mut GgufWriter| { w.tensor("", &[32], 8, vec![0; 34]); }), "may not be empty"),
            ("64-byte name", Box::new(|w: &mut GgufWriter| { w.tensor(&"x".repeat(64), &[32], 8, vec![0; 34]); }), "GGML_MAX_NAME"),
            ("duplicate name", Box::new(|w: &mut GgufWriter| { w.tensor("a", &[32], 8, vec![0; 34]); w.tensor("a", &[32], 8, vec![0; 34]); }), "added twice"),
            ("no dims", Box::new(|w: &mut GgufWriter| { w.tensor("a", &[], 0, vec![0; 4]); }), "GGML_MAX_DIMS"),
            ("five dims", Box::new(|w: &mut GgufWriter| { w.tensor("a", &[1, 1, 1, 1, 1], 0, vec![0; 4]); }), "GGML_MAX_DIMS"),
            ("zero dim", Box::new(|w: &mut GgufWriter| { w.tensor("a", &[4, 0], 0, Vec::new()); }), "zero-length dimension"),
            ("unknown type", Box::new(|w: &mut GgufWriter| { w.tensor("a", &[32], 999, vec![0; 34]); }), "unsupported ggml type 999"),
            ("internal f8 id", Box::new(|w: &mut GgufWriter| { w.tensor("a", &[128], 1042, vec![0; 129]); }), "INTERNAL id"),
            ("alignment 0", Box::new(|w: &mut GgufWriter| { w.alignment(0); }), "power of two"),
            ("alignment 48", Box::new(|w: &mut GgufWriter| { w.alignment(48); }), "power of two"),
        ];
        for (what, build, expect) in cases {
            let mut w = GgufWriter::new("t");
            build(&mut w);
            let e = w.finish().unwrap_err();
            assert!(e.contains(expect), "{what}: expected an error naming '{expect}', got: {e}");
        }
    }

    /// ⛔ The reader widens every integer into `Meta::U`/`Meta::I` and every float into `Meta::F`, so
    /// a round-trip through it cannot tell `7u8` from `7u64`. The tags and the value widths are
    /// therefore checked here against an INDEPENDENT walk of the KV section — a decoder written from
    /// the spec rather than a call into the one under test.
    ///
    /// The walk is sequential, so a wrong width does not merely mis-report one value: it desynchronises
    /// everything after it, and the final assertion (that the walk lands exactly on the first tensor
    /// name) is what turns that into a failure.
    #[test]
    fn every_kv_carries_the_type_tag_and_value_width_the_spec_assigns_it() {
        struct Walk<'a> { b: &'a [u8], p: usize }
        impl Walk<'_> {
            fn u32(&mut self) -> u32 {
                let v = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
                self.p += 4;
                v
            }
            fn u64(&mut self) -> u64 {
                let v = u64::from_le_bytes(self.b[self.p..self.p + 8].try_into().unwrap());
                self.p += 8;
                v
            }
            fn string(&mut self) -> String {
                let n = self.u64() as usize;
                let s = String::from_utf8(self.b[self.p..self.p + n].to_vec()).unwrap();
                self.p += n;
                s
            }
            /// Widths straight from the format table, independent of `Kv::encode`.
            fn skip_value(&mut self, tag: u32) {
                match tag {
                    T_U8 | T_I8 | T_BOOL => self.p += 1,
                    T_U16 | T_I16 => self.p += 2,
                    T_U32 | T_I32 | T_F32 => self.p += 4,
                    T_U64 | T_I64 | T_F64 => self.p += 8,
                    T_STRING => { self.string(); }
                    T_ARRAY => {
                        let elem = self.u32();
                        let n = self.u64();
                        for _ in 0..n { self.skip_value(elem); }
                    }
                    other => panic!("no such KV type tag: {other}"),
                }
            }
        }

        let bytes = fixture().finish().unwrap();
        let mut w = Walk { b: &bytes, p: 0 };
        assert_eq!(&bytes[..4], b"GGUF", "the magic is four ASCII bytes in reading order");
        w.p = 4;
        assert_eq!(w.u32(), 3, "version");
        assert_eq!(w.u64(), 4, "tensor count comes FIRST");
        assert_eq!(w.u64(), 20, "then the KV count");

        // Insertion order, with the two constructor keys ahead of the rest.
        let expect: [(&str, u32); 20] = [
            ("general.architecture", T_STRING), ("general.alignment", T_U32),
            ("t.u8", T_U8), ("t.i8", T_I8), ("t.u16", T_U16), ("t.i16", T_I16),
            ("t.u32", T_U32), ("t.i32", T_I32), ("t.u64", T_U64), ("t.i64", T_I64),
            ("t.f32", T_F32), ("t.f64", T_F64), ("t.bool.t", T_BOOL), ("t.bool.f", T_BOOL),
            ("t.str", T_STRING), ("t.arr.u32", T_ARRAY), ("t.arr.i32", T_ARRAY),
            ("t.arr.u64", T_ARRAY), ("t.arr.f32", T_ARRAY), ("t.arr.str", T_ARRAY),
        ];
        for (key, tag) in expect {
            assert_eq!(w.string(), key, "key out of order at byte {}", w.p);
            assert_eq!(w.u32(), tag, "key '{key}' carries the wrong type tag");
            w.skip_value(tag);
        }
        // If any width above was wrong the cursor is now somewhere else entirely, and this is where
        // that shows up.
        assert_eq!(w.string(), "emb.weight", "the KV section must end exactly where the tensor table begins");
        assert_eq!(w.u32(), 2, "n_dims is a u32");
        assert_eq!(w.u64(), 3, "ne0 first");
        assert_eq!(w.u64(), 5, "then ne1");
        assert_eq!(w.u32(), 0, "ggml type F32");
        assert_eq!(w.u64(), 0, "the first tensor's data-relative offset is 0");
    }

    /// A repeated key must replace, not append. Mainline rejects a file with a duplicate key outright
    /// and this crate's `HashMap` reader would silently keep the last — so appending produces a file
    /// that looks correct exactly here and nowhere else.
    #[test]
    fn setting_a_key_twice_replaces_it_rather_than_writing_it_twice() {
        let mut w = GgufWriter::new("t");
        w.kv_u32("n", 1).kv_str("s", "first").kv_u32("n", 2).kv_str("s", "second");
        let bytes = w.finish().unwrap();
        let g = parse(bytes.clone()).unwrap();
        assert_eq!(u(&g, "n"), 2);
        assert_eq!(s(&g, "s"), "second");
        assert_eq!(g.metadata.len(), 4, "architecture, alignment, n, s — and each exactly once");
        let kv_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        assert_eq!(kv_count, 4, "the header count must match what was written, not what was set");
        // Only ONE copy of the key can be present in the bytes.
        let needle: Vec<u8> = 1u64.to_le_bytes().iter().copied().chain(*b"n").collect();
        assert_eq!(bytes.windows(needle.len()).filter(|c| *c == needle.as_slice()).count(), 1,
                   "the key 'n' appears more than once in the file");
    }

    /// `general.alignment` set through the generic KV path must move the LAYOUT too. Storing it as an
    /// inert number would leave the padding at 32 while the reader recomputed `data_start` at 4096.
    #[test]
    fn setting_general_alignment_as_a_plain_kv_still_changes_the_layout() {
        let mut w = GgufWriter::new("t");
        w.kv_u32("general.alignment", 256);
        w.tensor("a", &[32], 8, blob(34, 3));
        w.tensor("b", &[32], 8, blob(34, 4));
        let g = parse(w.finish().unwrap()).unwrap();
        assert_eq!(u(&g, "general.alignment"), 256);
        assert_eq!(g.tensors[1].offset, 256, "the second tensor must sit at the next 256-byte boundary");
        assert_eq!(g.raw("b").unwrap(), blob(34, 4));

        // And a non-integer for that key is a refusal, not a silently ignored setting.
        let mut w = GgufWriter::new("t");
        w.kv_str("general.alignment", "32");
        assert!(w.finish().unwrap_err().contains("unsigned integer"));
    }

    /// A GGUF with metadata and no tensors is legal (llama.cpp's LoRA and vocab-only files are
    /// exactly this), and must not trip the reader's stride checks on an empty tensor table.
    #[test]
    fn a_file_with_no_tensors_at_all_is_still_valid() {
        let mut w = GgufWriter::new("vocab-only");
        w.kv_arr_str("tokenizer.ggml.tokens", &["a".to_string(), "b".to_string()]);
        let g = parse(w.finish().unwrap()).unwrap();
        assert!(g.tensors.is_empty());
        assert_eq!(arr(&g, "tokenizer.ggml.tokens").len(), 2);
    }

    /// Ferric's own quantizer feeding Ferric's own writer, read back through Ferric's own
    /// dequantizer. This is the loop the writer exists to close: before it, producing a quantized
    /// GGUF meant shelling out to another project's tool.
    #[test]
    fn a_tensor_quantized_by_this_crate_survives_the_container_and_dequantizes() {
        let x: Vec<f32> = (0..512).map(|k| ((k as f32) * 0.017).sin() * 0.4).collect();
        let mut raw = Vec::new();
        crate::quantize::quantize_q2_k(&x, &mut raw);
        let direct = crate::deq_raw(&raw, 512, 10).unwrap();

        let mut w = GgufWriter::new("ferric-test");
        w.tensor("blk.0.ffn_up.weight", &[256, 2], 10, raw.clone());
        let g = parse(w.finish().unwrap()).unwrap();
        assert_eq!(g.raw("blk.0.ffn_up.weight").unwrap(), raw, "the container must not touch the bytes");
        assert_eq!(g.dequant("blk.0.ffn_up.weight").unwrap(), direct,
                   "dequantizing out of the file must equal dequantizing the bytes directly");
    }

    /// ⛔ ggml type 42 is claimed by three formats and the reader resolves it BY STRIDE. A writer that
    /// emits 42 at PrismML Q2_0's 34 bytes/block therefore gets 42 back; the point of pinning it is
    /// that the resolution runs on our files too, so the writer's stride and the resolver's table
    /// have to agree or the round-trip silently changes the tensor's type.
    #[test]
    fn a_type_42_tensor_resolves_back_to_the_same_type_it_was_written_as() {
        let n = 128 * 16;
        let mut w = GgufWriter::new("t");
        w.tensor("q", &[n as u64], 42, blob(n / 128 * 34, 9));
        let g = parse(w.finish().unwrap()).unwrap();
        assert_eq!(g.tensors[0].ggml_type, 42, "34 bytes per 128 values is PrismML Q2_0, and stays 42");
        assert_eq!(g.raw("q").unwrap(), blob(n / 128 * 34, 9));
    }
}
