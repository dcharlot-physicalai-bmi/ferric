//! **Unmerged LoRA on the Dense runtime** — `y = W·x + Σ_adapters s·B(A·x)`, chosen per sequence.
//!
//! Merging (`ferric_load::lora::LoraMerged`) costs nothing extra per token only while the merged weight
//! keeps the base's size — on an F32 base. On a QUANTIZED base every adapted weight is widened (to F32,
//! exact, or F16, rounded) and decode pays for the bytes; and either way it bakes ONE adapter into the
//! weights, so serving N fine-tunes costs N copies of the model. Keeping the pairs separate
//! lets every adapter share one (quantized) base, and lets a batched decode give each ROW its own
//! adapter — which is how S-LoRA / Punica-style multi-LoRA serving works, and what PEFT's own
//! `forward(..., adapter_names=[...])` (`_mixed_batch_forward`) defines as the reference.
//!
//! ## Shape of the work
//!
//! The Dense runtime fuses q|k|v into one matmul and gate|up into another, so the adapter is fused to
//! match: per layer and per fused projection, ONE `A` (the adapted parts' `lora_A` stacked, `[Σr, in]`,
//! held as three exact bfloat16 terms — see [`split3_bf16`] for why) and ONE block-diagonal `B`
//! (`[out_q+out_k+out_v, Σr]`, each part's `scaling·lora_B` in its own rows and columns, zeros
//! elsewhere). Two small matmuls then add the whole projection's delta at once —
//! the zero blocks cost `Σr·out` multiply-adds the unfused form would skip, and save a dispatch per part.
//! The zeros contribute exact `+0.0`s, so the result per part is the unfused product.
//!
//! Per-row selection in a batch scales the `[N, Σr]` intermediate by a `[N, 1]` column — `s` on rows
//! using the adapter, `0` elsewhere — before `B`. Every adapter present anywhere in the batch runs over
//! every row: the waste is `(adapters in batch − 1)·N·Σr·(in+out)` flops, small beside the base matmul
//! while ranks are small, and it keeps the whole batch in the same two dispatches per adapter.
//!
//! `scaling` is folded into `B` at upload (the per-request multiplier is applied to the intermediate),
//! so PEFT's `(B·(A·x))·scaling` becomes `(sB)·(A·x)` — the same product, rounded in a different order.

use ferric_core::Context;
use ferric_load::lora::{LoraAdapter, RowOrder};
use ferric_tensor::{QMatrix, Tensor};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Which fused projection of a layer a pair adds to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Slot { Qkv, O, GateUp, Down }

pub(crate) struct Pair {
    /// Stacked `lora_A` as THREE bfloat16 terms, `[3·Σr, in]` — see [`split3_bf16`].
    a: QMatrix,
    /// Block-diagonal `scaling·lora_B`, repeated once per term: `[slot width, 3·Σr]`, float32.
    b: QMatrix,
}

/// Split each f32 into three bfloat16 terms whose sum is EXACTLY the f32: rows `[hi; mid; lo]`.
///
/// ⭐ Why not keep `lora_A` float32: at decode `x·Aᵀ` is a GEMV with a LONG reduction (`in` = 896..4864)
/// and FEW outputs (`Σr` = 2..48), and Ferric's dense f32 `QMatrix` runs the general matmul kernel — one
/// thread per OUTPUT, looping over `in` serially. That is a dozen threads each walking thousands of
/// values: measured 4-5x slower decode on Qwen2.5-0.5B-Q4_K_M with a 7-module adapter (15-29 vs 81-130
/// tok/s). The 16-bit weight kernels split the reduction across a workgroup (split-K). A bfloat16 term
/// carries 8 significant bits and f32's full exponent range, so three of them carry f32's 24: `hi` is
/// `x` rounded, `x - hi` is exact (Sterbenz) with at most 16 significant bits, and so on — the stored
/// value is the adapter's, bit for bit, and only the f32 accumulation order differs.
fn split3_bf16(v: &[f32]) -> Result<[Vec<u16>; 3], String> {
    let bits = ferric_load::hf::f32_to_bf16_bits;
    let val = |b: u16| f32::from_bits((b as u32) << 16);
    let (mut h, mut m, mut l) = (Vec::with_capacity(v.len()), Vec::with_capacity(v.len()), Vec::with_capacity(v.len()));
    for &x in v {
        if !x.is_finite() { return Err(format!("lora_A holds a non-finite value ({x})")); }
        let b1 = bits(x);
        let r1 = x - val(b1);
        let b2 = bits(r1);
        let r2 = r1 - val(b2);
        let b3 = bits(r2);
        if (val(b1) + val(b2)) + val(b3) != x {
            return Err(format!("internal: {x:e} does not split into three bfloat16 terms"));
        }
        h.push(b1); m.push(b2); l.push(b3);
    }
    Ok([h, m, l])
}

#[derive(Default)]
pub(crate) struct LayerLora { qkv: Option<Pair>, o: Option<Pair>, gate_up: Option<Pair>, down: Option<Pair> }

impl LayerLora {
    fn get(&self, s: Slot) -> Option<&Pair> {
        match s { Slot::Qkv => self.qkv.as_ref(), Slot::O => self.o.as_ref(),
                  Slot::GateUp => self.gate_up.as_ref(), Slot::Down => self.down.as_ref() }
    }
    fn slot_mut(&mut self, s: Slot) -> &mut Option<Pair> {
        match s { Slot::Qkv => &mut self.qkv, Slot::O => &mut self.o,
                  Slot::GateUp => &mut self.gate_up, Slot::Down => &mut self.down }
    }
}

/// An adapter resident on the device, shaped for one loaded model. Cheap to share: hold it in an
/// `Arc` and hand clones to as many sequences as use it.
pub struct DeviceLora {
    /// Unique for the life of the process — the identity a prefix cache must key on (see
    /// [`crate::qwen3::Cache::adapter_key`]).
    pub id: u64,
    pub name: String,
    /// Parameters uploaded (both factors, before block-diagonal padding).
    pub params: usize,
    /// Layers the adapter touches.
    pub layer_count: usize,
    layers: Vec<LayerLora>,
}

impl std::fmt::Debug for DeviceLora {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceLora {{ id: {}, name: {:?}, params: {}, layers: {} }}", self.id, self.name, self.params, self.layer_count)
    }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// What the adapter must fit — read from the loaded model, never from the adapter.
pub(crate) struct Geometry {
    pub n_layer: usize,
    pub n_embd: usize,
    pub q_out: usize,
    pub kv_out: usize,
    pub n_ff: usize,
    /// Phi-3: q|k|v is ONE stored weight (`attn_qkv`), so only a pair on the fused weight fits.
    pub fused_qkv: bool,
    /// Phi-3: gate|up is ONE stored weight under the name `ffn_up`, `[2·n_ff, in]`.
    pub fused_gate_up: bool,
}

impl DeviceLora {
    /// Upload `a` for a model of geometry `g`. `a` must already be in the base GGUF's q/k row order.
    ///
    /// Refuses, naming the weight, any pair that does not fit: a layer the model does not have, a
    /// projection it does not store in that form (a pair on `attn_q` for a model whose q|k|v is one
    /// fused weight), or a shape that differs from the model's — a different base model.
    pub(crate) fn build(ctx: &Arc<Context>, a: &LoraAdapter, g: &Geometry) -> Result<DeviceLora, String> {
        if a.qk_rows != RowOrder::Gguf {
            return Err("DeviceLora::build: adapter is not bound to the base's q/k row order".into());
        }
        // (slot, column offset in the slot, expected in, expected out)
        let place = |stem: &str| -> Result<(Slot, usize, usize, usize), String> {
            let (q, kv, d, f) = (g.q_out, g.kv_out, g.n_embd, g.n_ff);
            Ok(match stem {
                "attn_q" if !g.fused_qkv => (Slot::Qkv, 0, d, q),
                "attn_k" if !g.fused_qkv => (Slot::Qkv, q, d, kv),
                "attn_v" if !g.fused_qkv => (Slot::Qkv, q + kv, d, kv),
                "attn_qkv" if g.fused_qkv => (Slot::Qkv, 0, d, q + 2 * kv),
                "attn_output" => (Slot::O, 0, q, d),
                "ffn_gate" if !g.fused_gate_up => (Slot::GateUp, 0, d, f),
                "ffn_up" if !g.fused_gate_up => (Slot::GateUp, f, d, f),
                "ffn_up" if g.fused_gate_up => (Slot::GateUp, 0, d, 2 * f),
                "ffn_down" => (Slot::Down, 0, f, d),
                other => return Err(format!("'{other}' is not stored in that form by this model \
                                             (fused qkv: {}, fused gate|up: {})", g.fused_qkv, g.fused_gate_up)),
            })
        };
        let width = |s: Slot| match s {
            Slot::Qkv => g.q_out + 2 * g.kv_out, Slot::O => g.n_embd, Slot::GateUp => 2 * g.n_ff, Slot::Down => g.n_embd,
        };
        // Group pairs by (layer, slot), keeping column order.
        let mut groups: Vec<Vec<Vec<(usize, &ferric_load::lora::LoraTarget)>>> =
            (0..g.n_layer).map(|_| (0..4).map(|_| Vec::new()).collect()).collect();
        for (name, t) in &a.targets {
            let (il, stem) = name.strip_prefix("blk.").and_then(|r| r.split_once('.'))
                .and_then(|(il, s)| Some((il.parse::<usize>().ok()?, s.strip_suffix(".weight")?)))
                .ok_or_else(|| format!("adapter '{}': '{name}' is not a blk.N.<weight> name", a.name))?;
            if il >= g.n_layer {
                return Err(format!("adapter '{}' adapts layer {il}; the model has {}", a.name, g.n_layer));
            }
            let (slot, off, n_in, n_out) = place(stem).map_err(|e| format!("adapter '{}': {name}: {e}", a.name))?;
            if (t.n_in, t.n_out) != (n_in, n_out) {
                return Err(format!("adapter '{}': {name} is [{} out, {} in]; this model's is [{n_out}, {n_in}] — \
                                    not the base the adapter was trained on", a.name, t.n_out, t.n_in));
            }
            groups[il][slot as usize].push((off, t));
        }
        let mut layers: Vec<LayerLora> = (0..g.n_layer).map(|_| LayerLora::default()).collect();
        let mut touched = 0;
        for (il, per_slot) in groups.iter_mut().enumerate() {
            let mut any = false;
            for (si, parts) in per_slot.iter_mut().enumerate() {
                if parts.is_empty() { continue; }
                parts.sort_by_key(|(off, _)| *off);
                let slot = [Slot::Qkv, Slot::O, Slot::GateUp, Slot::Down][si];
                let n_in = parts[0].1.n_in;
                let r_tot: usize = parts.iter().map(|(_, t)| t.r).sum();
                let w = width(slot);
                let mut a_cat = Vec::with_capacity(r_tot * n_in);
                let mut b_bd = vec![0f32; w * r_tot];
                let mut roff = 0;
                for (off, t) in parts.iter() {
                    a_cat.extend_from_slice(&t.a);
                    for o in 0..t.n_out {
                        for k in 0..t.r {
                            b_bd[(off + o) * r_tot + roff + k] = t.scale * t.b[o * t.r + k];
                        }
                    }
                    roff += t.r;
                }
                // A as three stacked bf16 terms (split-K GEMV, exact values); B repeated three times
                // side by side, so `[x·hiᵀ | x·midᵀ | x·loᵀ] · [B|B|B]ᵀ = (x·Aᵀ)·Bᵀ`. B stays f32: its GEMV
                // reduces over only 3·Σr, which the one-thread-per-output kernel handles well.
                let terms = split3_bf16(&a_cat).map_err(|e| format!("adapter '{}': layer {il}: {e}", a.name))?;
                let a_bytes: Vec<u8> = terms.iter().flatten().flat_map(|b| b.to_le_bytes()).collect();
                let mut b3 = Vec::with_capacity(w * 3 * r_tot);
                for o in 0..w { for _ in 0..3 { b3.extend_from_slice(&b_bd[o * r_tot..(o + 1) * r_tot]); } }
                *layers[il].slot_mut(slot) = Some(Pair {
                    a: QMatrix::from_bytes(ctx, &a_bytes, 30, 3 * r_tot, n_in)?,
                    b: QMatrix::from_dense(ctx, &b3, w, 3 * r_tot),
                });
                any = true;
            }
            touched += any as usize;
        }
        Ok(DeviceLora { id: NEXT_ID.fetch_add(1, Ordering::Relaxed), name: a.name.clone(), params: a.params(),
                        layer_count: touched, layers })
    }
}

/// How strongly one adapter applies to the rows of a forward.
pub(crate) enum RowScale {
    /// Every row, at this multiplier.
    All(f32),
    /// Per row: a `[N, 1]` column, 0 where the row does not use the adapter.
    Rows(Tensor),
}

/// The adapters one forward applies — resolved once per forward from the caches' selections.
pub(crate) struct Active {
    items: Vec<(Arc<DeviceLora>, RowScale)>,
}

impl Active {
    pub(crate) fn none() -> Active { Active { items: Vec::new() } }

    /// Every row of a single sequence uses its cache's selection.
    pub(crate) fn uniform(sel: &[(Arc<DeviceLora>, f32)]) -> Active {
        Active { items: sel.iter().filter(|(_, s)| *s != 0.0).map(|(d, s)| (Arc::clone(d), RowScale::All(*s))).collect() }
    }

    /// Row `i` uses `sels[i]`. Identical selections collapse to [`Active::uniform`], so a batch whose
    /// sequences share an adapter pays nothing for the per-row machinery.
    pub(crate) fn per_row(ctx: &Arc<Context>, sels: &[&[(Arc<DeviceLora>, f32)]]) -> Active {
        let same = |x: &[(Arc<DeviceLora>, f32)], y: &[(Arc<DeviceLora>, f32)]| {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| Arc::ptr_eq(&a.0, &b.0) && a.1 == b.1)
        };
        // `FERRIC_LORA_NEG=rows` is the gate's control for THIS function: every row takes row 0's
        // selection — resolving one selection per batch instead of per sequence, the mistake that
        // serves one request's fine-tune to its neighbours. It must fail the batch comparison.
        let crossed = std::env::var("FERRIC_LORA_NEG").is_ok_and(|v| v.trim() == "rows");
        if crossed || sels.iter().all(|s| same(s, sels[0])) { return Active::uniform(sels[0]); }
        let mut distinct: Vec<Arc<DeviceLora>> = Vec::new();
        for s in sels { for (d, _) in s.iter() { if !distinct.iter().any(|x| Arc::ptr_eq(x, d)) { distinct.push(Arc::clone(d)); } } }
        let items = distinct.into_iter().map(|d| {
            let col: Vec<f32> = sels.iter().map(|s| s.iter().filter(|(x, _)| Arc::ptr_eq(x, &d)).map(|(_, v)| *v).sum()).collect();
            let n = col.len();
            (d, RowScale::Rows(Tensor::from_vec(ctx, &col, &[n, 1])))
        }).collect();
        Active { items }
    }

    /// Whether any active adapter adds to `slot` of layer `il` — the caller must then take the path
    /// that exposes the pre-activation (no fused SwiGLU, no FFN megakernel).
    pub(crate) fn touches(&self, il: usize, slot: Slot) -> bool {
        self.items.iter().any(|(d, _)| d.layers[il].get(slot).is_some())
    }

    /// `Σ s·B(A·x)` for `slot` of layer `il`, or `None` when no active adapter touches it.
    pub(crate) fn delta(&self, x: &Tensor, il: usize, slot: Slot) -> Option<Tensor> {
        let mut acc: Option<Tensor> = None;
        for (d, rs) in &self.items {
            let Some(p) = d.layers[il].get(slot) else { continue };
            let u = x.matmul_q(&p.a);
            let u = match rs {
                RowScale::All(s) if *s == 1.0 => u,
                RowScale::All(s) => u.mul(&u.scalar(*s)),
                RowScale::Rows(m) => u.mul(m),
            };
            let y = u.matmul_q(&p.b);
            acc = Some(match acc { None => y, Some(a) => a.add(&y) });
        }
        acc
    }

    /// `base + delta`, or `base` untouched.
    pub(crate) fn add_to(&self, base: Tensor, x: &Tensor, il: usize, slot: Slot) -> Tensor {
        match self.delta(x, il, slot) { Some(d) => base.add(&d), None => base }
    }
}
