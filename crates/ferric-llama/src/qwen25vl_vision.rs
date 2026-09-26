//! **Qwen2.5-VL's vision tower** — the eyes of MiMo-Embodied-7B, and of every model built on
//! Qwen2.5-VL, read from the checkpoint and from the authors' code (transformers
//! `modeling_qwen2_5_vl.py`) and checked against the latter stage by stage (`examples/qwen25vl_e2e.rs`).
//!
//! It shares its preprocessing, its patch sweep and its vision rope with Qwen3-VL's tower
//! ([`crate::qwen3vl_vision`]), and differs in everything else. Each difference below loads, runs and
//! produces plausible image features if it is carried over wrongly:
//!
//! 1. **WINDOWED attention.** Only `fullatt_block_indexes` (`[7, 15, 23, 31]`) attend over the whole
//!    image. The other 28 blocks attend within windows of `window_size` pixels = 112 / 14 / 2 = **4x4
//!    merged tokens** (8x8 patches), cut from the merged grid in raster order, the last row and column
//!    of windows partial.
//! 2. **The rows are REORDERED into window order before block 0 and put back after the merger.** Both
//!    permutations move whole 2x2 merge groups, so the merger still sees four consecutive rows, and
//!    attention is permutation-equivariant — skipping the reverse leaves every shape and every row
//!    plausible, just attached to the wrong image token.
//! 3. **RMSNorm** (eps 1e-6, fixed in the reference, not read from the config), a **SwiGLU** MLP WITH
//!    biases, a bias-free patch projection, and **no learned position table** — position is rope only.
//! 4. **One merger**, normalising per patch BEFORE the 2x2 merge; no deepstack.
//!
//! ⚠ Scope: ONE still image per call, like the Qwen3-VL tower. Video is refused, not folded.
use crate::qwen3vl_vision::sweep_rc;
use ferric_core::Context;
use ferric_load::SafeTensors;
use ferric_tensor::{MropeMode, Tensor};
use std::sync::Arc;

/// The reference hard-codes this at every RMSNorm in the tower (`Qwen2_5_VLRMSNorm(..., eps=1e-6)`);
/// the config's `rms_norm_eps` (1e-5 here) belongs to the TEXT model.
const VISION_EPS: f32 = 1e-6;

/// Shapes, read from `config.json`'s `vision_config`.
#[derive(Debug, Clone)]
pub struct VisionCfg {
    pub depth: usize,
    pub hidden: usize,
    pub heads: usize,
    pub ff: usize,
    pub patch: usize,
    pub temporal_patch: usize,
    pub merge: usize,
    pub out_hidden: usize,
    /// Pixels per window side — 112. In merged tokens: `window_size / merge / patch`.
    pub window_size: usize,
    /// Blocks that attend over the whole image; every other block is windowed.
    pub full_blocks: Vec<usize>,
}

impl VisionCfg {
    /// Window side in MERGED tokens — 4 for 112 px, patch 14, merge 2.
    pub fn window_merged(&self) -> usize { self.window_size / self.merge / self.patch }
}

/// Mechanism controls for the conformance gate, read ONCE at load. Each removes one mechanism the
/// reference has; the gate requires each to fail at the stage that mechanism lives in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Neg {
    /// Every block attends over the whole image (no windows).
    pub no_window: bool,
    /// Every block is windowed (the full-attention schedule ignored).
    pub all_window: bool,
    /// The merged rows are left in window order (the reverse permutation skipped).
    pub no_reverse: bool,
    /// Row and column swapped in the vision rope.
    pub rope_swap: bool,
    /// The merger's exact erf GELU replaced by the tanh approximation.
    pub gelu_tanh: bool,
}

impl Neg {
    fn from_env() -> Result<Neg, String> {
        let mut n = Neg::default();
        if let Ok(v) = std::env::var("FERRIC_VL_NEG") {
            match v.as_str() {
                "nowindow" => n.no_window = true,
                "allwindow" => n.all_window = true,
                "noreverse" => n.no_reverse = true,
                "rope_swap" => n.rope_swap = true,
                "gelu_tanh" => n.gelu_tanh = true,
                other => return Err(format!("FERRIC_VL_NEG={other} is not a tower control \
                                             (nowindow|allwindow|noreverse|rope_swap|gelu_tanh)")),
            }
        }
        Ok(n)
    }
}

struct Block {
    norm1: Tensor,
    qkv_w: Tensor, qkv_b: Tensor,
    proj_w: Tensor, proj_b: Tensor,
    norm2: Tensor,
    gate_w: Tensor, gate_b: Tensor,
    up_w: Tensor, up_b: Tensor,
    down_w: Tensor, down_b: Tensor,
}

pub struct VisionTower {
    pub cfg: VisionCfg,
    pub neg: Neg,
    patch_w: Tensor, // [3*tps*patch*patch, hidden] — the Conv3d with kernel == stride IS a matmul
    blocks: Vec<Block>,
    ln_q: Tensor,
    fc1_w: Tensor, fc1_b: Tensor,
    fc2_w: Tensor, fc2_b: Tensor,
}

fn t1(ctx: &Arc<Context>, st: &SafeTensors, name: &str) -> Result<Tensor, String> {
    let s = st.get(name)?;
    Ok(Tensor::from_vec(ctx, &s.data, &s.shape.clone()))
}

/// A `[out, in]` weight as the `[in, out]` matrix `matmul` wants. Done once, at load.
fn lin(ctx: &Arc<Context>, st: &SafeTensors, name: &str) -> Result<Tensor, String> {
    let s = st.get(name)?;
    if s.shape.len() != 2 {
        return Err(format!("{name}: expected a 2-D weight, got {:?}", s.shape));
    }
    Ok(Tensor::from_vec(ctx, &s.data, &[s.shape[0], s.shape[1]]).transpose(0, 1).contiguous())
}

/// Window order over the MERGED grid, and each window's length in PATCH rows.
///
/// Transcribed from the reference's `get_window_index`: the `lh x lw` merged grid is cut into
/// `win x win` windows in raster order of windows, each window's tokens in raster order within it,
/// the last window row and column partial. Returns `(order, lens)`: `order[i]` is the merged token
/// (raster index, `y * lw + x`) at window-order slot `i`; `lens` are patch rows per window.
///
/// ⚠ The reference pads by `win - lh % win`, which is a WHOLE EXTRA WINDOW ROW when `lh % win == 0`;
/// that window is empty and `unique_consecutive` then drops its zero length. Counting `ceil(lh / win)`
/// windows, as this does, never makes the empty one — the same partition.
pub fn window_index(gh: usize, gw: usize, merge: usize, win: usize) -> (Vec<usize>, Vec<usize>) {
    let (lh, lw) = (gh / merge, gw / merge);
    let (mut order, mut lens) = (Vec::with_capacity(lh * lw), Vec::new());
    for wy in 0..lh.div_ceil(win) {
        for wx in 0..lw.div_ceil(win) {
            let before = order.len();
            for y in wy * win..((wy + 1) * win).min(lh) {
                for x in wx * win..((wx + 1) * win).min(lw) {
                    order.push(y * lw + x);
                }
            }
            lens.push((order.len() - before) * merge * merge);
        }
    }
    (order, lens)
}

impl VisionTower {
    /// Load from a checkpoint directory (`config.json` + sharded or single safetensors). Tensor names
    /// are the checkpoint's own: `visual.*` at the top level (the pre-transformers-5 layout MiMo ships)
    /// or `model.visual.*` (the 5.x layout) — whichever the file holds, never both.
    pub fn load(ctx: &Arc<Context>, dir: &str) -> Result<VisionTower, String> {
        let cfg_txt = std::fs::read_to_string(format!("{dir}/config.json"))
            .map_err(|e| format!("config.json: {e}"))?;
        let j: serde_json::Value = serde_json::from_str(&cfg_txt).map_err(|e| format!("config: {e}"))?;
        let v = j.get("vision_config").ok_or("config.json has no vision_config")?;
        let u = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize)
            .ok_or_else(|| format!("vision_config: missing {k}"));
        let cfg = VisionCfg {
            depth: u("depth")?, hidden: u("hidden_size")?, heads: u("num_heads")?,
            ff: u("intermediate_size")?, patch: u("patch_size")?,
            temporal_patch: u("temporal_patch_size")?, merge: u("spatial_merge_size")?,
            out_hidden: u("out_hidden_size")?, window_size: u("window_size")?,
            // ⛔ Refused when absent: with no schedule every block would be windowed (or every block
            // full), and both run.
            full_blocks: v.get("fullatt_block_indexes").and_then(|x| x.as_array())
                .ok_or("vision_config: missing fullatt_block_indexes")?
                .iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect(),
        };
        if cfg.window_merged() == 0 {
            return Err(format!("window_size {} is below one merged token", cfg.window_size));
        }
        if v.get("hidden_act").and_then(|x| x.as_str()).is_some_and(|a| a != "silu") {
            return Err(format!("vision hidden_act {:?}: this tower implements SwiGLU (silu) only",
                               v.get("hidden_act")));
        }
        let st = SafeTensors::open(dir)?;
        let root = match (st.info("visual.patch_embed.proj.weight"), st.info("model.visual.patch_embed.proj.weight")) {
            (Some(_), None) => "visual",
            (None, Some(_)) => "model.visual",
            (Some(_), Some(_)) => return Err("checkpoint holds BOTH visual.* and model.visual.* — refusing \
                                              to pick one".into()),
            (None, None) => return Err("no visual.patch_embed.proj.weight in the checkpoint".into()),
        };

        // The Conv3d's kernel equals its stride, so it sees each patch row exactly once: a matmul over
        // the flattened [C, T, p, p] row the preprocessor emits, and bias-free in this checkpoint.
        let pw = st.get(&format!("{root}.patch_embed.proj.weight"))?;
        let in_dim: usize = pw.shape[1..].iter().product();
        if in_dim != 3 * cfg.temporal_patch * cfg.patch * cfg.patch || pw.shape[0] != cfg.hidden {
            return Err(format!("patch weight {:?} is not [hidden, 3, tps, p, p]", pw.shape));
        }
        if st.info(&format!("{root}.patch_embed.proj.bias")).is_some() {
            return Err("the patch projection has a bias this tower does not apply — refusing".into());
        }
        let patch_w = Tensor::from_vec(ctx, &pw.data, &[cfg.hidden, in_dim]).transpose(0, 1).contiguous();

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let p = format!("{root}.blocks.{i}");
            blocks.push(Block {
                norm1: t1(ctx, &st, &format!("{p}.norm1.weight"))?,
                qkv_w: lin(ctx, &st, &format!("{p}.attn.qkv.weight"))?,
                qkv_b: t1(ctx, &st, &format!("{p}.attn.qkv.bias"))?,
                proj_w: lin(ctx, &st, &format!("{p}.attn.proj.weight"))?,
                proj_b: t1(ctx, &st, &format!("{p}.attn.proj.bias"))?,
                norm2: t1(ctx, &st, &format!("{p}.norm2.weight"))?,
                gate_w: lin(ctx, &st, &format!("{p}.mlp.gate_proj.weight"))?,
                gate_b: t1(ctx, &st, &format!("{p}.mlp.gate_proj.bias"))?,
                up_w: lin(ctx, &st, &format!("{p}.mlp.up_proj.weight"))?,
                up_b: t1(ctx, &st, &format!("{p}.mlp.up_proj.bias"))?,
                down_w: lin(ctx, &st, &format!("{p}.mlp.down_proj.weight"))?,
                down_b: t1(ctx, &st, &format!("{p}.mlp.down_proj.bias"))?,
            });
            // An RMSNorm has a weight and no bias; a LayerNorm checkpoint (Qwen2-VL) would carry one,
            // and running it as RMSNorm is silent.
            if st.info(&format!("{p}.norm1.bias")).is_some() {
                return Err(format!("{p}.norm1 has a bias: that is a LayerNorm (Qwen2-VL), not this tower"));
            }
        }
        let wide = cfg.hidden * cfg.merge * cfg.merge;
        let ln_q = t1(ctx, &st, &format!("{root}.merger.ln_q.weight"))?;
        // ⛔ The shape is the evidence for WHERE the merger normalises: [hidden] = per patch, before the
        // 2x2 merge. A [hidden*merge^2] norm would be after it (Qwen3-VL's deepstack kind).
        if ln_q.numel() != cfg.hidden {
            return Err(format!("merger.ln_q is {} wide, expected hidden {} (pre-merge)", ln_q.numel(), cfg.hidden));
        }
        let fc1_w = lin(ctx, &st, &format!("{root}.merger.mlp.0.weight"))?;
        if fc1_w.shape != [wide, wide] {
            return Err(format!("merger.mlp.0 is {:?}, expected [{wide}, {wide}]", fc1_w.shape));
        }
        let fc2_w = lin(ctx, &st, &format!("{root}.merger.mlp.2.weight"))?;
        if fc2_w.shape != [wide, cfg.out_hidden] {
            return Err(format!("merger.mlp.2 is {:?}, expected [{wide}, {}]", fc2_w.shape, cfg.out_hidden));
        }
        Ok(VisionTower {
            neg: Neg::from_env()?,
            patch_w, blocks, ln_q,
            fc1_b: t1(ctx, &st, &format!("{root}.merger.mlp.0.bias"))?, fc1_w,
            fc2_b: t1(ctx, &st, &format!("{root}.merger.mlp.2.bias"))?, fc2_w,
            cfg,
        })
    }

    /// Encode one still image from pre-flattened patch rows `[gh*gw, 3*tps*patch*patch]`, rows in the
    /// spatial-merge sweep the preprocessor emits (see [`crate::qwen3vl_vision::sweep_rc`]).
    ///
    /// Returns the image rows for the language model, `[gh*gw/merge^2, out_hidden]`, in the SAME
    /// (sweep) order as the image tokens. `taps` receives `(stage, tensor)` for `patch_embed` (sweep
    /// order), `block{i}` (WINDOW order — as the reference holds them), `merger_window_order` and
    /// `merger`, so a mismatch can be placed in one stage.
    pub fn encode_patches(&self, px: &Tensor, gh: usize, gw: usize, taps: &mut Vec<(String, Tensor)>)
        -> Result<Tensor, String>
    {
        let c = &self.cfg;
        let (m, m2) = (c.merge, c.merge * c.merge);
        if !gh.is_multiple_of(m) || !gw.is_multiple_of(m) {
            return Err(format!("grid {gh}x{gw} is not a multiple of the merge size {m}"));
        }
        let n = gh * gw;
        if px.shape != [n, 3 * c.temporal_patch * c.patch * c.patch] {
            return Err(format!("expected [{n}, {}] patch rows for a {gh}x{gw} grid, got {:?}",
                               3 * c.temporal_patch * c.patch * c.patch, px.shape));
        }
        let (heads, dh) = (c.heads, c.hidden / c.heads);

        let x = px.matmul(&self.patch_w);
        taps.push(("patch_embed".into(), x.clone()));

        // ---- into WINDOW order ------------------------------------------------------------------
        // Merged token j is sweep rows [j*m2, (j+1)*m2): the sweep emits each 2x2 block as four
        // consecutive rows, block-row-major — so a merged token's sweep index IS its raster index on
        // the merged grid, which is what `window_index` returns.
        let (order, lens) = window_index(gh, gw, m, c.window_merged());
        let rows: Vec<u32> = order.iter().flat_map(|&j| (0..m2).map(move |k| (j * m2 + k) as u32)).collect();
        let mut x = x.gather_rows(&rows);

        // Rope positions: (row, col) per patch in the sweep, then permuted exactly like the rows.
        let rc: Vec<(usize, usize)> = rows.iter().map(|&r| sweep_rc(m, gw, r as usize)).collect();
        let (a, b): (Vec<u32>, Vec<u32>) = rc.iter().map(|&(r, cc)| (r as u32, cc as u32)).unzip();
        let (a, b) = if self.neg.rope_swap { (b, a) } else { (a, b) };
        // Section-major for rope_mrope; components 2 and 3 are never consulted at head_dim/2 sectors
        // but the kernel reads four, so they are filled rather than left to chance.
        let pos: Vec<u32> = [&a, &b, &a, &b].iter().flat_map(|v| v.iter().copied()).collect();
        let sect = (dh / 4) as u32;

        for (il, blk) in self.blocks.iter().enumerate() {
            let full = if self.neg.no_window { true } else if self.neg.all_window { false }
                       else { c.full_blocks.contains(&il) };
            let h = x.rmsnorm(&blk.norm1, VISION_EPS);
            let qkv = h.matmul(&blk.qkv_w).add(&blk.qkv_b);
            let q = qkv.narrow(1, 0, c.hidden).contiguous()
                .rope_mrope(heads, dh, 10_000.0, &pos, [sect; 4], MropeMode::Vision);
            let k = qkv.narrow(1, c.hidden, c.hidden).contiguous()
                .rope_mrope(heads, dh, 10_000.0, &pos, [sect; 4], MropeMode::Vision);
            let v = qkv.narrow(1, 2 * c.hidden, c.hidden).contiguous();
            let att = if full {
                ferric_tensor::nn::bidirectional_attention(&q, &k, &v, heads, heads)
            } else {
                // Each window attends only within itself: the reference's non-flash path splits at
                // `cu_window_seqlens` and concatenates, and this is that loop.
                let mut start = 0;
                let mut out: Option<Tensor> = None;
                for &len in &lens {
                    let w = |t: &Tensor| t.narrow(0, start, len).contiguous();
                    let o = ferric_tensor::nn::bidirectional_attention(&w(&q), &w(&k), &w(&v), heads, heads);
                    out = Some(match out { None => o, Some(acc) => acc.cat(&o, 0) });
                    start += len;
                }
                debug_assert_eq!(start, n, "windows must tile the image");
                out.ok_or("an image with no windows")?
            };
            x = x.add(&att.matmul(&blk.proj_w).add(&blk.proj_b));
            let h2 = x.rmsnorm(&blk.norm2, VISION_EPS);
            let g = h2.matmul(&blk.gate_w).add(&blk.gate_b).silu();
            let u = h2.matmul(&blk.up_w).add(&blk.up_b);
            x = x.add(&g.mul(&u).matmul(&blk.down_w).add(&blk.down_b));
            taps.push((format!("block{il}"), x.clone()));
        }

        // ---- merger: norm per patch, 2x2 merge (a reshape: four consecutive rows), fc, GELU, fc --
        let merged = x.rmsnorm(&self.ln_q, VISION_EPS).reshape(&[n / m2, c.hidden * m2]);
        let hmid = merged.matmul(&self.fc1_w).add(&self.fc1_b);
        let hmid = if self.neg.gelu_tanh { hmid.gelu_tanh() } else { hmid.gelu() };
        let out = hmid.matmul(&self.fc2_w).add(&self.fc2_b);
        taps.push(("merger_window_order".into(), out.clone()));

        // ---- back to token order: merged token j sits at window slot inv[j] ------------------------
        if self.neg.no_reverse {
            taps.push(("merger".into(), out.clone()));
            return Ok(out);
        }
        let mut inv = vec![0u32; order.len()];
        for (slot, &j) in order.iter().enumerate() { inv[j] = slot as u32; }
        let out = out.gather_rows(&inv);
        taps.push(("merger".into(), out.clone()));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::window_index;

    /// The reference's construction written out literally — pad with -100, reshape to windows,
    /// permute, drop the padding — as a second statement of the order `window_index` computes with
    /// bounds. Deliberately a different mechanism: agreeing with itself proves nothing.
    fn padded_reference(gh: usize, gw: usize, m: usize, win: usize) -> (Vec<usize>, Vec<usize>) {
        let (lh, lw) = (gh / m, gw / m);
        let (ph, pw) = (win - lh % win, win - lw % win); // a whole extra window when divisible
        let (hh, ww) = (lh + ph, lw + pw);
        let padded: Vec<i64> = (0..hh * ww).map(|i| {
            let (y, x) = (i / ww, i % ww);
            if y < lh && x < lw { (y * lw + x) as i64 } else { -100 }
        }).collect();
        let (nh, nw) = (hh / win, ww / win);
        let (mut order, mut lens) = (Vec::new(), Vec::new());
        // index_padded.reshape(nh, win, nw, win).permute(0, 2, 1, 3)
        for a in 0..nh {
            for b in 0..nw {
                let mut cnt = 0;
                for c in 0..win {
                    for d in 0..win {
                        let v = padded[(a * win + c) * ww + b * win + d];
                        if v != -100 { order.push(v as usize); cnt += 1; }
                    }
                }
                lens.push(cnt * m * m);
            }
        }
        lens.retain(|&l| l != 0); // unique_consecutive over the cumulative sums drops empty windows
        (order, lens)
    }

    #[test]
    fn window_order_agrees_with_the_padded_reshape_form() {
        for &(gh, gw) in &[(10, 16), (16, 16), (8, 8), (2, 2), (10, 14), (22, 30), (18, 8), (64, 46)] {
            assert_eq!(window_index(gh, gw, 2, 4), padded_reference(gh, gw, 2, 4), "grid {gh}x{gw}");
        }
    }

    /// A third statement, by hand, of the fixture's 10x16 patch grid (5x8 merged): four windows,
    /// two full 4x4 and two partial 1x4 along the bottom.
    #[test]
    fn a_5x8_merged_grid_is_these_four_windows() {
        let (order, lens) = window_index(10, 16, 2, 4);
        assert_eq!(lens, vec![64, 64, 16, 16]);
        assert_eq!(&order[..8], &[0, 1, 2, 3, 8, 9, 10, 11]);
        assert_eq!(&order[16..20], &[4, 5, 6, 7]);
        assert_eq!(&order[32..], &[32, 33, 34, 35, 36, 37, 38, 39]);
        // ⚠ and it is not the identity, or every check above could pass on a tower that never reordered
        assert_ne!(order, (0..40).collect::<Vec<_>>());
    }
}
