//! **Qwen3-VL's vision tower**, straight from the published checkpoint.
//!
//! Every structural fact below was read from the WEIGHTS or from the reference implementation, not
//! from a description of them — three separate details were wrong in the prose spec for this
//! checkpoint, and each would have produced a tower that loads, runs, and attends to the wrong
//! places:
//!
//! 1. **The spec's dimensions are the 8B's.** This 2B carries `hidden 1024, depth 24, heads 16`
//!    (so `head_dim 64`), `ff 4096`, `deepstack [5,11,17]` — from `config.json`, not the document.
//! 2. **Patch embedding is a MATMUL here, not a convolution.** The stored weight is
//!    `[1024, 3, 2, 16, 16]` and `3*2*16*16 = 1536` is exactly the width of one input row, because
//!    this checkpoint's pixels arrive pre-flattened per patch. llama.cpp splits a Conv3d into two
//!    Conv2d only because it consumes raw images and must extract the patches itself.
//! 3. **The two merger kinds normalise at DIFFERENT widths.** `merger.norm` is `[1024]` — per patch,
//!    BEFORE the 2x2 merge. Each `deepstack_merger_list.k.norm` is `[4096]` — AFTER it. Same class in
//!    the reference, one `use_postshuffle_norm` flag apart, and the weight shapes are the only
//!    outside evidence.
//!
//! ⛔ And the two GELUs differ: the block FFN uses `gelu_pytorch_tanh`, the mergers use `nn.GELU()`
//! (exact erf). Reaching for one at both sites is the obvious mistake and is silent.
//!
//! ⚠ Scope: ONE still image per call. Video (`temporal_patch_size` pairing two frames) is not
//! implemented and is refused rather than silently folded into a single frame.
use ferric_core::Context;
use ferric_load::SafeTensors;
use ferric_tensor::{MropeMode, Tensor};
use std::sync::Arc;

/// Shapes the tower is built from. Read from `config.json`'s `vision_config`.
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
    /// Block indices whose output feeds a deepstack merger — `[5, 11, 17]` here.
    pub deepstack: Vec<usize>,
    pub eps: f32,
    /// Side of the learned position grid: `sqrt(pos_embed.rows)` = 48.
    pub pos_side: usize,
}

struct Block {
    ln1_w: Tensor, ln1_b: Tensor,
    qkv_w: Tensor, qkv_b: Tensor,
    proj_w: Tensor, proj_b: Tensor,
    ln2_w: Tensor, ln2_b: Tensor,
    fc1_w: Tensor, fc1_b: Tensor,
    fc2_w: Tensor, fc2_b: Tensor,
}

/// A merger: LayerNorm, then two linears with an exact-erf GELU between.
/// `post_shuffle` says whether the norm runs AFTER the 2x2 merge (deepstack) or before it (main).
struct Merger {
    norm_w: Tensor, norm_b: Tensor,
    fc1_w: Tensor, fc1_b: Tensor,
    fc2_w: Tensor, fc2_b: Tensor,
    post_shuffle: bool,
}

pub struct VisionTower {
    ctx: Arc<Context>,
    pub cfg: VisionCfg,
    patch_w: Tensor,   // [hidden, 3*tps*patch*patch] — a plain linear, see the header
    patch_b: Tensor,
    pos: Tensor,       // [pos_side*pos_side, hidden]
    blocks: Vec<Block>,
    merger: Merger,
    deep: Vec<Merger>,
}

fn t2(ctx: &Arc<Context>, st: &SafeTensors, name: &str) -> Result<Tensor, String> {
    let s = st.get(name)?;
    Ok(Tensor::from_vec(ctx, &s.data, &s.shape.iter().copied().collect::<Vec<_>>()))
}

/// A `[out, in]` weight as the `[in, out]` matrix `matmul` wants. Done once, at load.
fn lin(ctx: &Arc<Context>, st: &SafeTensors, name: &str) -> Result<Tensor, String> {
    let s = st.get(name)?;
    if s.shape.len() != 2 {
        return Err(format!("{name}: expected a 2-D weight, got {:?}", s.shape));
    }
    Ok(Tensor::from_vec(ctx, &s.data, &[s.shape[0], s.shape[1]]).transpose(0, 1).contiguous())
}

impl VisionTower {
    pub fn load(ctx: &Arc<Context>, dir: &str) -> Result<VisionTower, String> {
        let cfg_txt = std::fs::read_to_string(format!("{dir}/config.json"))
            .map_err(|e| format!("config.json: {e}"))?;
        let j: serde_json::Value = serde_json::from_str(&cfg_txt).map_err(|e| format!("config: {e}"))?;
        let v = j.get("vision_config").ok_or("config.json has no vision_config")?;
        let u = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize)
            .ok_or_else(|| format!("vision_config: missing {k}"));
        let st = SafeTensors::open(format!("{dir}/model.safetensors"))?;

        let hidden = u("hidden_size")?;
        let pos_rows = st.info("model.visual.pos_embed.weight")
            .ok_or("no model.visual.pos_embed.weight")?.shape[0];
        let side = (pos_rows as f64).sqrt() as usize;
        if side * side != pos_rows {
            return Err(format!("pos_embed has {pos_rows} rows, not a square grid"));
        }
        let cfg = VisionCfg {
            depth: u("depth")?, hidden, heads: u("num_heads")?, ff: u("intermediate_size")?,
            patch: u("patch_size")?, temporal_patch: u("temporal_patch_size")?,
            merge: u("spatial_merge_size")?, out_hidden: u("out_hidden_size")?,
            deepstack: v.get("deepstack_visual_indexes").and_then(|x| x.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
                .unwrap_or_default(),
            eps: v.get("layer_norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-6) as f32,
            pos_side: side,
        };

        // Patch embedding: [hidden, C, tps, p, p] flattened to [in, hidden] — see the header for why
        // this is a matmul and not a convolution in this checkpoint.
        let pw = st.get("model.visual.patch_embed.proj.weight")?;
        let in_dim: usize = pw.shape[1..].iter().product();
        if in_dim != 3 * cfg.temporal_patch * cfg.patch * cfg.patch {
            return Err(format!("patch weight {:?} does not match C*tps*p*p", pw.shape));
        }
        let patch_w = Tensor::from_vec(ctx, &pw.data, &[pw.shape[0], in_dim]).transpose(0, 1).contiguous();

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let p = format!("model.visual.blocks.{i}");
            blocks.push(Block {
                ln1_w: t2(ctx, &st, &format!("{p}.norm1.weight"))?,
                ln1_b: t2(ctx, &st, &format!("{p}.norm1.bias"))?,
                qkv_w: lin(ctx, &st, &format!("{p}.attn.qkv.weight"))?,
                qkv_b: t2(ctx, &st, &format!("{p}.attn.qkv.bias"))?,
                proj_w: lin(ctx, &st, &format!("{p}.attn.proj.weight"))?,
                proj_b: t2(ctx, &st, &format!("{p}.attn.proj.bias"))?,
                ln2_w: t2(ctx, &st, &format!("{p}.norm2.weight"))?,
                ln2_b: t2(ctx, &st, &format!("{p}.norm2.bias"))?,
                fc1_w: lin(ctx, &st, &format!("{p}.mlp.linear_fc1.weight"))?,
                fc1_b: t2(ctx, &st, &format!("{p}.mlp.linear_fc1.bias"))?,
                fc2_w: lin(ctx, &st, &format!("{p}.mlp.linear_fc2.weight"))?,
                fc2_b: t2(ctx, &st, &format!("{p}.mlp.linear_fc2.bias"))?,
            });
        }
        let merger_at = |pfx: &str, post: bool| -> Result<Merger, String> {
            Ok(Merger {
                norm_w: t2(ctx, &st, &format!("{pfx}.norm.weight"))?,
                norm_b: t2(ctx, &st, &format!("{pfx}.norm.bias"))?,
                fc1_w: lin(ctx, &st, &format!("{pfx}.linear_fc1.weight"))?,
                fc1_b: t2(ctx, &st, &format!("{pfx}.linear_fc1.bias"))?,
                fc2_w: lin(ctx, &st, &format!("{pfx}.linear_fc2.weight"))?,
                fc2_b: t2(ctx, &st, &format!("{pfx}.linear_fc2.bias"))?,
                post_shuffle: post,
            })
        };
        // ⛔ The flag is not cosmetic and the shapes are the evidence: main normalises at `hidden`,
        // deepstack at `hidden * merge^2`. Asserted below so a checkpoint that disagrees is refused
        // rather than reshaped into whatever fits.
        let merger = merger_at("model.visual.merger", false)?;
        let m2 = cfg.merge * cfg.merge;
        if merger.norm_w.numel() != hidden {
            return Err(format!("merger.norm is {} wide, expected hidden {hidden} (pre-shuffle)",
                               merger.norm_w.numel()));
        }
        let mut deep = Vec::new();
        for k in 0..cfg.deepstack.len() {
            let m = merger_at(&format!("model.visual.deepstack_merger_list.{k}"), true)?;
            if m.norm_w.numel() != hidden * m2 {
                return Err(format!("deepstack[{k}].norm is {} wide, expected hidden*merge^2 {} \
                                    (post-shuffle)", m.norm_w.numel(), hidden * m2));
            }
            deep.push(m);
        }
        Ok(VisionTower {
            ctx: ctx.clone(), cfg, patch_w, patch_b: t2(ctx, &st, "model.visual.patch_embed.proj.bias")?,
            pos: t2(ctx, &st, "model.visual.pos_embed.weight")?, blocks, merger, deep,
        })
    }
}


/// `(row, col)` of token `i` in the spatial-merge sweep, for a `gh x gw` patch grid at merge `m`.
///
/// ⛔ This sweep is load-bearing THREE times over and every way of getting it wrong is silent.
/// The merger does `view(-1, hidden*merge^2)`, a RESHAPE, so four CONSECUTIVE rows must already be
/// one `m x m` block. The vision rope positions are emitted in this sweep, so a mismatch rotates
/// each patch by another patch's coordinates. And the interpolated position embedding is gathered in
/// it. None of the three changes a shape.
///
/// ⚠ It is also the order the PIXEL ROWS ALREADY ARRIVE IN — the image processor's
/// `permute(0,3,6,4,7,2,1,5,8)` emits it — so this must never be used to reorder them. Doing that
/// once cost rel 5.7e-1 against the published tower with every assertion still green.
fn sweep_rc(m: usize, gw: usize, i: usize) -> (usize, usize) {
    let by = i / (m * m * (gw / m));
    let bx = (i / (m * m)) % (gw / m);
    let dy = (i / m) % m;
    let dx = i % m;
    (by * m + dy, bx * m + dx)
}

/// Raster index of each token, in sweep order — the gather that turns a raster grid into block order.
fn block_order(m: usize, gh: usize, gw: usize) -> Vec<u32> {
    (0..gh * gw).map(|i| { let (r, c) = sweep_rc(m, gw, i); (r * gw + c) as u32 }).collect()
}

/// Row and column position per token, in the same sweep — section-major for `rope_mrope`.
fn positions(m: usize, gh: usize, gw: usize) -> Vec<u32> {
    let rc: Vec<(usize, usize)> = (0..gh * gw).map(|i| sweep_rc(m, gw, i)).collect();
    let row: Vec<u32> = rc.iter().map(|&(r, _)| r as u32).collect();
    let col: Vec<u32> = rc.iter().map(|&(_, c)| c as u32).collect();
    // sections 2 and 3 are never consulted (only head_dim/2 sectors exist) but the kernel reads
    // four components, so they are filled rather than left to chance.
    let mut p = row.clone();
    p.extend_from_slice(&col);
    p.extend_from_slice(&row);
    p.extend_from_slice(&col);
    p
}

impl VisionTower {
    fn block_order(&self, gh: usize, gw: usize) -> Vec<u32> { block_order(self.cfg.merge, gh, gw) }

    fn positions(&self, gh: usize, gw: usize) -> Vec<u32> { positions(self.cfg.merge, gh, gw) }

    /// The learned 48x48 grid resampled to `gh x gw`, then put into block order.
    ///
    /// ⚠ ALIGN-CORNERS bilinear: the reference computes it by hand with
    /// `linspace(0, side-1, n)` + floor/ceil weights, which IS align-corners. Ferric's
    /// `resize_bilinear` shares one half-pixel convention with image preprocessing, so this asks for
    /// the align-corners form explicitly rather than assuming they agree.
    fn pos_for(&self, gh: usize, gw: usize) -> Tensor {
        let (s, d) = (self.cfg.pos_side, self.cfg.hidden);
        let p = if gh == s && gw == s {
            self.pos.clone()
        } else {
            // ⚠ NO permutes. llama.cpp permutes because a ggml tensor is [n_embd, side, side] with
            // ne0 fastest, so it must rotate the grid axes into place for `ggml_interpolate`. Ferric's
            // `resize_bilinear` already takes [H, W, C] row-major, which is what this reshape gives.
            // Translating the reference's permutes literally resized `[C, H, W]` — i.e. it interpolated
            // the CHANNEL axis against a spatial one, and the shapes only failed to line up by luck.
            self.pos.reshape(&[s, s, d])
                .resize_bilinear_align_corners(gh, gw)
                .reshape(&[gh * gw, d])
        };
        p.gather_rows(&self.block_order(gh, gw))
    }

    /// Encode one still image given pre-flattened patch rows `[gh*gw, 3*tps*patch*patch]`.
    ///
    /// ⛔ `px` rows must ALREADY be in spatial-merge-block order — `(block_row, block_col, in_row,
    /// in_col)`, `in_col` fastest — which is what the image processor's permute emits. This tower
    /// does not reorder them; see `encode_patches`'s body.
    ///
    /// Returns `(image_tokens, deepstack, hidden)`. The first two are projected to the TEXT width,
    /// which is what the language model consumes; `image_tokens` is `[gh*gw/merge^2, out_hidden]`.
    /// `hidden` is the pre-merger rows at VISION width, `[gh*gw, hidden]` — returned so a caller can
    /// check the blocks separately from the merger rather than only seeing the merged result.
    pub fn encode_patches(&self, px: &Tensor, gh: usize, gw: usize)
        -> Result<(Tensor, Vec<Tensor>, Tensor), String> {
        let c = &self.cfg;
        let (m, m2) = (c.merge, c.merge * c.merge);
        if gh % m != 0 || gw % m != 0 {
            return Err(format!("grid {gh}x{gw} is not a multiple of the merge size {m}"));
        }
        let n = gh * gw;
        if px.shape[0] != n {
            return Err(format!("expected {n} patch rows for a {gh}x{gw} grid, got {}", px.shape[0]));
        }
        let (heads, dh) = (c.heads, c.hidden / c.heads);

        // patch embed (a linear here — see the header), then + position.
        //
        // ⛔ NO REORDERING OF THE PATCH ROWS. The rows arrive ALREADY in spatial-merge-block order:
        // the image processor's `permute(0,3,6,4,7,2,1,5,8)` emits (block_row, block_col, in_row,
        // in_col), and the reference tower never touches the token order — it generates its position
        // ids and its pos-embed taps directly in that sweep (`get_vision_position_ids` reshapes to
        // `(h/m, m, w/m, m)` then `transpose(1,2)`). Gathering here permuted rows that were already
        // permuted, so token i got patch `block_order[i]`'s pixels. Attention is permutation-
        // equivariant and the merger reshape still found four consecutive rows, so nothing crashed
        // and nothing was shaped wrong — the tower just described a scrambled image.
        let mut x = px.matmul(&self.patch_w).add(&self.patch_b).add(&self.pos_for(gh, gw));

        let pos = self.positions(gh, gw);
        let sect = (dh / 4) as u32;
        let sections = [sect, sect, sect, sect];
        let mut deep_out = Vec::new();

        for (il, b) in self.blocks.iter().enumerate() {
            let h = x.layernorm(&b.ln1_w, &b.ln1_b, c.eps);
            // one fused projection -> q | k | v, in that order (matches the reference's unbind)
            let qkv = h.matmul(&b.qkv_w).add(&b.qkv_b);
            let q = qkv.narrow(1, 0, c.hidden).contiguous();
            let k = qkv.narrow(1, c.hidden, c.hidden).contiguous();
            let v = qkv.narrow(1, 2 * c.hidden, c.hidden).contiguous();
            // ⛔ VISION rope: head_dim passed whole (the loop and pairing span it) while the
            // frequency denominator is head_dim/2 — the distinction that cost a silent bug once.
            let q = q.rope_mrope(heads, dh, 10_000.0, &pos, sections, MropeMode::Vision);
            let k = k.rope_mrope(heads, dh, 10_000.0, &pos, sections, MropeMode::Vision);
            // ⚠ FULL attention on every layer. Qwen2.5-VL is the one with a windowed pattern;
            // this tower reads no window config and passes no mask.
            let att = ferric_tensor::nn::bidirectional_attention(&q, &k, &v, heads, heads);
            x = x.add(&att.matmul(&b.proj_w).add(&b.proj_b));
            let h2 = x.layernorm(&b.ln2_w, &b.ln2_b, c.eps);
            // tanh GELU here; the mergers use the exact erf one.
            x = x.add(&h2.matmul(&b.fc1_w).add(&b.fc1_b).gelu_tanh().matmul(&b.fc2_w).add(&b.fc2_b));

            if let Some(k) = c.deepstack.iter().position(|&d| d == il) {
                deep_out.push(self.merge_and_project(&x, &self.deep[k], n, m2));
            }
        }
        if deep_out.len() != c.deepstack.len() {
            return Err(format!("expected {} deepstack outputs, produced {}", c.deepstack.len(), deep_out.len()));
        }
        Ok((self.merge_and_project(&x, &self.merger, n, m2), deep_out, x))
    }

    /// Norm (before or after the merge, per the merger kind), 2x2 merge, fc1, exact GELU, fc2.
    fn merge_and_project(&self, x: &Tensor, g: &Merger, n: usize, m2: usize) -> Tensor {
        let wide = self.cfg.hidden * m2;
        let merged = if g.post_shuffle {
            // deepstack: merge first, then normalise at the WIDE width
            x.reshape(&[n / m2, wide]).layernorm(&g.norm_w, &g.norm_b, self.cfg.eps)
        } else {
            // main: normalise per patch, then merge — `view(-1, wide)` in the reference is a reshape,
            // which is only correct because block_order already made each 2x2 four consecutive rows.
            x.layernorm(&g.norm_w, &g.norm_b, self.cfg.eps).reshape(&[n / m2, wide])
        };
        merged.matmul(&g.fc1_w).add(&g.fc1_b).gelu().matmul(&g.fc2_w).add(&g.fc2_b)
    }
}

#[cfg(test)]
mod sweep_tests {
    use super::*;

    /// The sweep, written the OTHER way: four nested loops, which is the plain reading of the
    /// reference's `reshape(h/m, m, w/m, m).transpose(1, 2).flatten()`.
    ///
    /// ⛔ Deliberately NOT the modular-decode form that `sweep_rc` ships. The reference states this
    /// order twice, in two unrelated code paths — `get_vision_position_ids` by reshape/transpose and
    /// `get_vision_interpolation_indices_and_weights` by `i % m`, `(i / m) % m`, … — and the shipped
    /// code follows the second. Checking a modular decode against a modular decode would agree with
    /// itself no matter which one was wrong.
    fn nested(m: usize, gh: usize, gw: usize) -> Vec<(usize, usize)> {
        let mut v = Vec::with_capacity(gh * gw);
        for by in (0..gh).step_by(m) {
            for bx in (0..gw).step_by(m) {
                for dy in 0..m {
                    for dx in 0..m {
                        v.push((by + dy, bx + dx));
                    }
                }
            }
        }
        v
    }

    #[test]
    fn sweep_agrees_with_the_reshape_transpose_form() {
        for &(m, gh, gw) in &[(2, 4, 4), (2, 6, 4), (2, 4, 8), (2, 2, 2), (1, 3, 5), (4, 8, 8)] {
            let want = nested(m, gh, gw);
            let got: Vec<_> = (0..gh * gw).map(|i| sweep_rc(m, gw, i)).collect();
            assert_eq!(got, want, "sweep disagrees at merge {m} on a {gh}x{gw} grid");
        }
    }

    /// A third statement of the same fact, with the indices written out by hand, so that a shared
    /// misreading of the reference cannot make both of the above agree on the wrong answer.
    #[test]
    fn block_order_4x4_merge2_is_this_exact_sequence() {
        assert_eq!(block_order(2, 4, 4),
                   vec![0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]);
        // ⚠ and it is NOT the identity — otherwise every check above would pass on a tower that
        // never applied it.
        assert_ne!(block_order(2, 4, 4), (0..16u32).collect::<Vec<_>>());
        // merge 1 IS raster order; that is the degenerate case the others must not be confused with.
        assert_eq!(block_order(1, 3, 5), (0..15u32).collect::<Vec<_>>());
    }

    #[test]
    fn positions_are_section_major_and_index_the_same_patches() {
        let (m, gh, gw) = (2, 6, 4);
        let n = gh * gw;
        let p = positions(m, gh, gw);
        assert_eq!(p.len(), 4 * n, "rope_mrope reads four components");
        let ord = block_order(m, gh, gw);
        for i in 0..n {
            let (r, c) = (p[i] as usize, p[n + i] as usize);
            assert_eq!(ord[i] as usize, r * gw + c, "token {i} position does not name its own patch");
            assert!(r < gh && c < gw, "token {i} position {r},{c} is off the grid");
        }
        // components 2 and 3 repeat 0 and 1 — filled, not left to chance.
        assert_eq!(&p[2 * n..3 * n], &p[..n]);
        assert_eq!(&p[3 * n..], &p[n..2 * n]);
    }
}
