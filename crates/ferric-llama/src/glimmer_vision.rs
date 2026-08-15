//! **Muse Glimmer's vision tower** — 50-layer ViT + adapter, GGUF mmproj in, image tokens out.
//!
//! Written against `tools/mtmd/models/muse-glimmer.cpp` and `clip.cpp`'s `set_input` branch, read
//! verbatim. Lives in the library rather than an example because two callers need it: the
//! encoder-only check and the full vision-language path.
//!
//! ```text
//! conv2d patchify (patch×patch) + bilinear-resized learned pos-emb
//! → gather(sp_perm): group into pgrid×pgrid windows
//! → n_layer pre-norm ViT layers, 2-D RoPE (width = first half of head, height = second),
//!   window attention EXCEPT global on every 4th layer and the last
//! → gather(inv_perm) → pixel-shuffle merge² → adapter mm_0/1/2 (exact erf GELU between)
//! ```
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_tensor::image::{preprocess, read_ppm, Rgb8};
use ferric_tensor::{nn, QMatrix, Tensor};
use std::sync::Arc;

struct Blk {
    ln1_w: Tensor, ln1_b: Tensor, ln2_w: Tensor, ln2_b: Tensor,
    q: QMatrix, qb: Tensor, k: QMatrix, kb: Tensor, v: QMatrix, vb: Tensor,
    o: QMatrix, ob: Tensor,
    up: QMatrix, upb: Tensor, down: QMatrix, downb: Tensor,
}

/// A loaded Muse Glimmer vision tower. `encode` turns an image into `[n_out, projection_dim]` rows
/// that drop straight into the text sequence via `Qwen3::forward_embeds`.
pub struct VisionTower {
    ctx: Arc<Context>,
    blks: Vec<Blk>,
    patch_w: Tensor, pos_emb: Tensor,
    pre_w: Tensor, pre_b: Tensor, post_w: Tensor, post_b: Tensor,
    mm0: QMatrix, mm1: QMatrix, mm2: QMatrix,
    sp_perm: Vec<u32>, inv_perm: Vec<u32>, ds_perm: Vec<u32>, slens: Vec<usize>,
    pos_w: Vec<u32>, pos_h: Vec<u32>,
    pub n_layer: usize, pub d: usize, pub n_head: usize, pub head_dim: usize,
    pub img_size: usize, pub patch: usize, pub merge: usize, pub grid: usize,
    pub n_tok: usize, pub n_out: usize, pub proj_dim: usize, pub pgrid: usize,
    mean: [f32; 3], std: [f32; 3], eps: f32,
}

impl VisionTower {
    pub fn load(ctx: &Arc<Context>, mmproj: &str) -> Result<VisionTower, String> {
        let g = GgufFile::open(mmproj)?;
        let u = |k: &str| match g.metadata.get(k) { Some(Meta::U(v)) => *v as usize, _ => panic!("missing {k}") };
        let f = |k: &str| match g.metadata.get(k) { Some(Meta::F(v)) => *v as f32, _ => panic!("missing {k}") };
        let arr3 = |k: &str| -> [f32; 3] {
            match g.metadata.get(k) {
                Some(Meta::Arr(v)) if v.len() == 3 => {
                    let mut o = [0f32; 3];
                    for (i, m) in v.iter().enumerate() { if let Meta::F(x) = m { o[i] = *x as f32; } }
                    o
                }
                _ => [0.5; 3],
            }
        };
        let n_layer = u("clip.vision.block_count");
        let d = u("clip.vision.embedding_length");
        let n_head = u("clip.vision.attention.head_count");
        let img_size = u("clip.vision.image_size");
        let patch = u("clip.vision.patch_size");
        let merge = u("clip.vision.spatial_merge_size");
        let ff = u("clip.vision.feed_forward_length");
        let eps = f("clip.vision.attention.layer_norm_epsilon");
        let proj_dim = u("clip.vision.projection_dim");
        let (mean, std) = (arr3("clip.vision.image_mean"), arr3("clip.vision.image_std"));
        let head_dim = d / n_head;
        let grid = img_size / patch;
        let n_tok = grid * grid;
        let n_out = (grid / merge) * (grid / merge);

        let qm = |name: &str| -> QMatrix {
            let t = g.tensor(name).unwrap_or_else(|| panic!("missing {name}"));
            let (ty, rows, cols) = (t.ggml_type, t.dims[1] as usize, t.dims[0] as usize);
            if QMatrix::block_bytes(ty).is_some() {
                QMatrix::from_bytes(ctx, &g.raw(name).unwrap(), ty, rows, cols).unwrap()
            } else {
                QMatrix::from_dense(ctx, &g.dequant(name).unwrap(), rows, cols)
            }
        };
        let ft = |name: &str, shape: &[usize]| Tensor::from_vec(ctx, &g.dequant(name).unwrap(), shape);

        // GGUF [kw,kh,c,o] dequantises row-major to OIHW; conv2d wants HWIO. Permute once, on load.
        let pe = g.dequant("v.patch_embd.weight")?;
        let (kh, kw, c_in, o_out) = (patch, patch, 3usize, d);
        let mut hwio = vec![0f32; kh * kw * c_in * o_out];
        for o in 0..o_out { for c in 0..c_in { for y in 0..kh { for x in 0..kw {
            hwio[((y * kw + x) * c_in + c) * o_out + o] = pe[((o * c_in + c) * kh + y) * kw + x];
        }}}}
        let patch_w = Tensor::from_vec(ctx, &hwio, &[kh, kw, c_in, o_out]);

        let pos_n = g.tensor("v.position_embd.weight").ok_or("no position_embd")?.dims[1] as usize;
        let pgrid = (pos_n as f64).sqrt() as usize;
        let pos_emb = ft("v.position_embd.weight", &[pgrid, pgrid, d])
            .resize_bilinear(grid, grid).reshape(&[n_tok, d]);

        let blks: Vec<Blk> = (0..n_layer).map(|il| {
            let b = |s: &str| format!("v.blk.{il}.{s}");
            Blk {
                ln1_w: ft(&b("ln1.weight"), &[d]), ln1_b: ft(&b("ln1.bias"), &[d]),
                ln2_w: ft(&b("ln2.weight"), &[d]), ln2_b: ft(&b("ln2.bias"), &[d]),
                q: qm(&b("attn_q.weight")), qb: ft(&b("attn_q.bias"), &[d]),
                k: qm(&b("attn_k.weight")), kb: ft(&b("attn_k.bias"), &[d]),
                v: qm(&b("attn_v.weight")), vb: ft(&b("attn_v.bias"), &[d]),
                o: qm(&b("attn_out.weight")), ob: ft(&b("attn_out.bias"), &[d]),
                up: qm(&b("ffn_up.weight")), upb: ft(&b("ffn_up.bias"), &[ff]),
                down: qm(&b("ffn_down.weight")), downb: ft(&b("ffn_down.bias"), &[d]),
            }
        }).collect();

        // Host-side precompute, mirroring clip.cpp's PROJECTOR_TYPE_MUSE_GLIMMER set_input branch.
        let win = pgrid;
        let (nwin_h, nwin_w) = (grid.div_ceil(win), grid.div_ceil(win));
        let mut sp_perm: Vec<u32> = Vec::with_capacity(n_tok);
        let mut slens: Vec<usize> = Vec::new();
        for wy in 0..nwin_h { for wx in 0..nwin_w {
            let mut cnt = 0;
            for hh in 0..win { for ww in 0..win {
                let (gy, gx) = (wy * win + hh, wx * win + ww);
                if gy < grid && gx < grid { sp_perm.push((gy * grid + gx) as u32); cnt += 1; }
            }}
            if cnt > 0 { slens.push(cnt); }
        }}
        let mut inv_perm = vec![0u32; n_tok];
        let (mut pw, mut ph) = (vec![0u32; n_tok], vec![0u32; n_tok]);
        for i in 0..n_tok {
            let orig = sp_perm[i] as usize;
            pw[i] = (orig % grid) as u32 + 1;   // 1-indexed, per the reference
            ph[i] = (orig / grid) as u32 + 1;
            inv_perm[orig] = i as u32;
        }
        let mut ds_perm: Vec<u32> = Vec::with_capacity(n_tok);
        for oy in 0..grid / merge { for ox in 0..grid / merge {
            for ry in 0..merge { for rx in 0..merge {
                ds_perm.push((((oy * merge + ry) * grid) + ox * merge + rx) as u32);
            }}
        }}

        Ok(VisionTower {
            ctx: ctx.clone(), blks, patch_w, pos_emb,
            pre_w: ft("v.pre_ln.weight", &[d]), pre_b: ft("v.pre_ln.bias", &[d]),
            post_w: ft("v.post_ln.weight", &[d]), post_b: ft("v.post_ln.bias", &[d]),
            mm0: qm("mm.0.weight"), mm1: qm("mm.1.weight"), mm2: qm("mm.2.weight"),
            sp_perm, inv_perm, ds_perm, slens, pos_w: pw, pos_h: ph,
            n_layer, d, n_head, head_dim, img_size, patch, merge, grid, n_tok, n_out, proj_dim, pgrid,
            mean, std, eps,
        })
    }

    /// Read a P6 PPM and encode it. Returns `[n_out, projection_dim]`.
    pub fn encode_ppm(&self, bytes: &[u8]) -> Result<Tensor, String> {
        let img = read_ppm(bytes)?;
        Ok(self.encode(&img))
    }

    pub fn encode(&self, img: &Rgb8) -> Tensor {
        let (d, n_tok, n_head, head_dim) = (self.d, self.n_tok, self.n_head, self.head_dim);
        let rope_base = 10000.0f32;
        let sf = 4usize;
        let px = preprocess(&self.ctx, img, self.img_size, self.mean, self.std);
        let mut x = px.reshape(&[1, self.img_size, self.img_size, 3])
            .conv2d(&self.patch_w, (self.patch, self.patch), (0, 0))
            .reshape(&[n_tok, d])
            .add(&self.pos_emb)
            .gather_rows(&self.sp_perm)
            .layernorm(&self.pre_w, &self.pre_b, self.eps);

        for (il, b) in self.blks.iter().enumerate() {
            let is_global = il == self.n_layer - 1 || (il + 1) % sf == 0;
            let h = x.layernorm(&b.ln1_w, &b.ln1_b, self.eps);
            let q = h.matmul_q(&b.q).add(&b.qb);
            let k = h.matmul_q(&b.k).add(&b.kb);
            let v = h.matmul_q(&b.v).add(&b.vb);
            // 2-D RoPE: one rotate-half over the full head would pair a WIDTH dim with a HEIGHT dim,
            // so each half is roped independently with its own per-token positions.
            let half = head_dim / 2;
            let split = |t: &Tensor, pos: &[u32], lo: usize| {
                t.reshape(&[n_tok, n_head, head_dim]).narrow(2, lo, half).contiguous()
                    .reshape(&[n_tok, n_head * half])
                    .rope_at(n_head, half, rope_base, pos)
                    .reshape(&[n_tok, n_head, half])
            };
            let q = split(&q, &self.pos_w, 0).cat(&split(&q, &self.pos_h, half), 2).reshape(&[n_tok, d]);
            let k = split(&k, &self.pos_w, 0).cat(&split(&k, &self.pos_h, half), 2).reshape(&[n_tok, d]);
            // Slice the windows rather than mask them: sp_perm already made each window a contiguous
            // row range, and a [n_head, 4096, 4096] score matrix is 1.07 GB per layer.
            let att = if is_global {
                let (mut parts, mut off) = (Vec::new(), 0usize);
                while off < n_tok {
                    let len = 1024.min(n_tok - off);
                    parts.push(nn::full_attention_kv(&q.narrow(0, off, len).contiguous(), &k, &v, n_head, n_head));
                    off += len;
                }
                parts.iter().skip(1).fold(parts[0].clone(), |a, p| a.cat(p, 0))
            } else {
                let (mut parts, mut off) = (Vec::new(), 0usize);
                for &wl in &self.slens {
                    parts.push(nn::bidirectional_attention(
                        &q.narrow(0, off, wl).contiguous(), &k.narrow(0, off, wl).contiguous(),
                        &v.narrow(0, off, wl).contiguous(), n_head, n_head));
                    off += wl;
                }
                parts.iter().skip(1).fold(parts[0].clone(), |a, p| a.cat(p, 0))
            };
            x = x.add(&att.matmul_q(&b.o).add(&b.ob));
            let h2 = x.layernorm(&b.ln2_w, &b.ln2_b, self.eps);
            x = x.add(&h2.matmul_q(&b.up).add(&b.upb).gelu().matmul_q(&b.down).add(&b.downb));
        }
        x = x.layernorm(&self.post_w, &self.post_b, self.eps).gather_rows(&self.inv_perm);
        let m2 = self.merge * self.merge;
        x = x.gather_rows(&self.ds_perm).reshape(&[self.n_out, m2, d]).permute(&[0, 2, 1]).contiguous()
            .reshape(&[self.n_out, d * m2]);
        x.matmul_q(&self.mm0).gelu().matmul_q(&self.mm1).gelu().matmul_q(&self.mm2)
    }
}
