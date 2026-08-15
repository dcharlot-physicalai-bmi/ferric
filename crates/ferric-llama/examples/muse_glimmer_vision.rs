//! **Muse Glimmer's vision tower from a GGUF mmproj** — 50-layer ViT → 1024 image tokens.
//!
//! Written against `tools/mtmd/models/muse-glimmer.cpp` and `clip.cpp`'s `set_input` branch, read
//! verbatim. Ferric needed no new kernels for this: `gelu()` is already erf-exact, `conv2d` does the
//! patchify, `layernorm` has the bias, `gather_rows` does the permutations, `resize_bilinear` does the
//! position-embedding grid, and additive-mask attention is `scores.add(&mask).softmax(2)`.
//!
//! ```text
//! conv2d patchify (14×14, stride 14) + bilinear-resized learned pos-emb   [4096, 1536]
//! → gather(sp_perm)          group into 32×32 windows  → 4 windows × 1024
//! → 50 layers, pre-norm ViT:
//!     ln1 → q,k,v (+bias) → 2-D RoPE → attn(mask) → out(+bias) → residual
//!     ln2 → ffn_up(+bias) → GELU_ERF → ffn_down(+bias) → residual
//!   mask = block-diagonal window, EXCEPT global on every 4th layer and the last
//! → gather(inv_perm) → pixel-shuffle 2×2 → [1024, 6144]
//! → mm_0 → gelu → mm_1 → gelu → mm_2                                     [1024, 6656]
//! ```
//!
//! ## The 2-D RoPE cannot be one call
//!
//! The first half of each head's dims carries the WIDTH position and the second half the HEIGHT
//! position. A single rotate-half over the full 96-dim head would pair dim `d` with `d+48` — a width
//! dim against a height dim — which is silently wrong. So each 48-dim half is roped independently,
//! with `rope_at`'s per-token positions supplying `pos_w` and `pos_h`.
//!
//!   cargo run -p ferric-llama --example muse_glimmer_vision --release -- <mmproj.gguf> <image.ppm>
use ferric_core::Context;
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_tensor::image::{preprocess, read_ppm};
use ferric_tensor::{nn, QMatrix, Tensor};
use std::sync::Arc;

struct Blk {
    ln1_w: Tensor, ln1_b: Tensor, ln2_w: Tensor, ln2_b: Tensor,
    q: QMatrix, qb: Tensor, k: QMatrix, kb: Tensor, v: QMatrix, vb: Tensor,
    o: QMatrix, ob: Tensor,
    up: QMatrix, upb: Tensor, down: QMatrix, downb: Tensor,
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let mmproj = a.get(1).expect("usage: muse_glimmer_vision <mmproj.gguf> <image.ppm>");
    let imgpath = a.get(2).expect("need an image.ppm (ffmpeg -i x.jpg -pix_fmt rgb24 x.ppm)");

    let g = GgufFile::open(mmproj).expect("open mmproj");
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
    let eps = f("clip.vision.attention.layer_norm_epsilon");
    let proj_dim = u("clip.vision.projection_dim");
    let (mean, std) = (arr3("clip.vision.image_mean"), arr3("clip.vision.image_std"));
    let head_dim = d / n_head;
    let grid = img_size / patch;              // 64
    let n_tok = grid * grid;                  // 4096
    let n_out = (grid / merge) * (grid / merge);
    // Sparse-window layers are every layer EXCEPT every 4th and the last (llama.cpp: sparse_factor 4).
    let sf = 4usize;
    let rope_base = 10000.0f32;

    let ctx = Arc::new(Context::new().await.unwrap());
    let t0 = std::time::Instant::now();
    let qm = |name: &str| -> QMatrix {
        let t = g.tensor(name).unwrap_or_else(|| panic!("missing {name}"));
        let (ty, rows, cols) = (t.ggml_type, t.dims[1] as usize, t.dims[0] as usize);
        if QMatrix::block_bytes(ty).is_some() {
            QMatrix::from_bytes(&ctx, &g.raw(name).unwrap(), ty, rows, cols).unwrap()
        } else {
            QMatrix::from_dense(&ctx, &g.dequant(name).unwrap(), rows, cols)
        }
    };
    let ft = |name: &str, shape: &[usize]| Tensor::from_vec(&ctx, &g.dequant(name).unwrap(), shape);

    // patch_embd is GGUF [kw, kh, c, o] which dequantises row-major to OIHW [o, c, kh, kw];
    // Ferric's conv2d wants HWIO [kh, kw, c, o]. Permute on the host, once.
    let pe = g.dequant("v.patch_embd.weight").unwrap();
    let (kh, kw, c_in, o_out) = (patch, patch, 3usize, d);
    let mut hwio = vec![0f32; kh * kw * c_in * o_out];
    for o in 0..o_out { for c in 0..c_in { for y in 0..kh { for x in 0..kw {
        hwio[((y * kw + x) * c_in + c) * o_out + o] = pe[((o * c_in + c) * kh + y) * kw + x];
    }}}}
    let patch_w = Tensor::from_vec(&ctx, &hwio, &[kh, kw, c_in, o_out]);

    // Learned position embedding is a pgrid×pgrid grid (1024 = 32²), bilinear-resized to the patch grid.
    let pos_n = g.tensor("v.position_embd.weight").unwrap().dims[1] as usize;
    let pgrid = (pos_n as f64).sqrt() as usize;
    let pos_emb = ft("v.position_embd.weight", &[pgrid, pgrid, d]).resize_bilinear(grid, grid)
        .reshape(&[n_tok, d]);

    let pre_w = ft("v.pre_ln.weight", &[d]);  let pre_b = ft("v.pre_ln.bias", &[d]);
    let post_w = ft("v.post_ln.weight", &[d]); let post_b = ft("v.post_ln.bias", &[d]);
    let blks: Vec<Blk> = (0..n_layer).map(|il| {
        let b = |s: &str| format!("v.blk.{il}.{s}");
        Blk {
            ln1_w: ft(&b("ln1.weight"), &[d]), ln1_b: ft(&b("ln1.bias"), &[d]),
            ln2_w: ft(&b("ln2.weight"), &[d]), ln2_b: ft(&b("ln2.bias"), &[d]),
            q: qm(&b("attn_q.weight")), qb: ft(&b("attn_q.bias"), &[d]),
            k: qm(&b("attn_k.weight")), kb: ft(&b("attn_k.bias"), &[d]),
            v: qm(&b("attn_v.weight")), vb: ft(&b("attn_v.bias"), &[d]),
            o: qm(&b("attn_out.weight")), ob: ft(&b("attn_out.bias"), &[d]),
            up: qm(&b("ffn_up.weight")), upb: ft(&b("ffn_up.bias"), &[u("clip.vision.feed_forward_length")]),
            down: qm(&b("ffn_down.weight")), downb: ft(&b("ffn_down.bias"), &[d]),
        }
    }).collect();
    let mm0 = qm("mm.0.weight"); let mm1 = qm("mm.1.weight"); let mm2 = qm("mm.2.weight");

    println!("muse-glimmer vision · {n_layer} layers · d={d} · {n_head}h × {head_dim} · grid {grid}×{grid} = {n_tok} tok");
    println!("  pos-emb {pgrid}×{pgrid} -> {grid}×{grid} bilinear · windows {}×{} · merge {merge} -> {n_out} out tokens",
             grid / pgrid, grid / pgrid);
    println!("  loaded in {:.2?}", t0.elapsed());

    // ---- host-side precompute (clip.cpp PROJECTOR_TYPE_MUSE_GLIMMER set_input) ----
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
    let (mut pos_w, mut pos_h) = (vec![0u32; n_tok], vec![0u32; n_tok]);
    for i in 0..n_tok {
        let orig = sp_perm[i] as usize;
        pos_w[i] = (orig % grid) as u32 + 1;   // 1-indexed, per the reference
        pos_h[i] = (orig / grid) as u32 + 1;
        inv_perm[orig] = i as u32;
    }
    // Block-diagonal window mask in permuted order: 0 inside a window, -inf across.
    let mut mask = vec![f32::NEG_INFINITY; n_tok * n_tok];
    { let mut off = 0usize;
      for &s in &slens { for a in 0..s { for b in 0..s { mask[(off + a) * n_tok + off + b] = 0.0; } } off += s; } }
    let sp_mask = Tensor::from_vec(&ctx, &mask, &[1, n_tok, n_tok]);
    // Pixel-shuffle gather, in ORIGINAL order: the merge×merge neighbours of each output cell.
    let mut ds_perm: Vec<u32> = Vec::with_capacity(n_tok);
    for oy in 0..grid / merge { for ox in 0..grid / merge {
        for ry in 0..merge { for rx in 0..merge {
            ds_perm.push((((oy * merge + ry) * grid) + ox * merge + rx) as u32);
        }}
    }}

    // ---- image ----
    let img = read_ppm(&std::fs::read(imgpath).expect("read image")).expect("ppm");
    let px = preprocess(&ctx, &img, img_size, mean, std);   // [S, S, 3]
    println!("  image {}×{} -> {img_size}×{img_size}", img.w, img.h);

    let t0 = std::time::Instant::now();
    // patchify: [1, S, S, 3] conv2d stride=patch -> [1, grid, grid, d] -> [n_tok, d]
    let mut x = px.reshape(&[1, img_size, img_size, 3])
        .conv2d(&patch_w, (patch, patch), (0, 0))
        .reshape(&[n_tok, d])
        .add(&pos_emb);
    x = x.gather_rows(&sp_perm);
    x = x.layernorm(&pre_w, &pre_b, eps);

    // FERRIC_VBLK truncates the stack — the fastest way to find whether a 50-layer run dies from a
    // per-layer resource ceiling rather than from the math.
    let vlim: usize = std::env::var("FERRIC_VBLK").ok().and_then(|v| v.parse().ok()).unwrap_or(n_layer);
    for (il, b) in blks.iter().take(vlim).enumerate() {
        let is_global = il == n_layer - 1 || (il + 1) % sf == 0;
        let h = x.layernorm(&b.ln1_w, &b.ln1_b, eps);
        let q = h.matmul_q(&b.q).add(&b.qb);
        let k = h.matmul_q(&b.k).add(&b.kb);
        let v = h.matmul_q(&b.v).add(&b.vb);
        // 2-D RoPE: first half of each head = width, second half = height. Two independent 48-dim
        // ropes, because one rotate-half over the full head would pair width against height.
        let half = head_dim / 2;
        let split = |t: &Tensor, pos: &[u32], lo: usize| {
            t.reshape(&[n_tok, n_head, head_dim]).narrow(2, lo, half).contiguous()
                .reshape(&[n_tok, n_head * half])
                .rope_at(n_head, half, rope_base, pos)
                .reshape(&[n_tok, n_head, half])
        };
        let q = split(&q, &pos_w, 0).cat(&split(&q, &pos_h, half), 2).reshape(&[n_tok, d]);
        let k = split(&k, &pos_w, 0).cat(&split(&k, &pos_h, half), 2).reshape(&[n_tok, d]);
        // A block-diagonal mask over 4096 tokens is NOT how to spend memory here. The composed path
        // materialises [n_head, T, T] = 16x4096x4096 f32 = 1.07 GB per layer, which gets the process
        // killed with no error (1- and 2-layer runs succeed; 50 does not). But sp_perm has ALREADY
        // grouped the tokens by window, so each window is a CONTIGUOUS row range and a sparse layer is
        // just `nwin` independent attentions of `win_len` tokens each — 16x1024x1024 = 67 MB, and
        // exactly the computation the mask was describing. Sparsity you can slice is cheaper than
        // sparsity you have to mask.
        let att = if is_global {
            // Global layers still attend over everything; chunk the QUERIES so the scores are
            // [n_head, chunk, T] rather than [n_head, T, T].
            let mut parts: Vec<Tensor> = Vec::new();
            let chunk = 1024usize;
            let mut off = 0usize;
            while off < n_tok {
                let len = chunk.min(n_tok - off);
                let qc = q.narrow(0, off, len).contiguous();
                parts.push(nn::full_attention_kv(&qc, &k, &v, n_head, n_head));
                off += len;
            }
            parts.iter().skip(1).fold(parts[0].clone(), |acc, p| acc.cat(p, 0))
        } else {
            let mut parts: Vec<Tensor> = Vec::new();
            let mut off = 0usize;
            for &wl in &slens {
                let (qw, kw2, vw) = (q.narrow(0, off, wl).contiguous(),
                                     k.narrow(0, off, wl).contiguous(),
                                     v.narrow(0, off, wl).contiguous());
                parts.push(nn::bidirectional_attention(&qw, &kw2, &vw, n_head, n_head));
                off += wl;
            }
            parts.iter().skip(1).fold(parts[0].clone(), |acc, p| acc.cat(p, 0))
        };
        x = x.add(&att.matmul_q(&b.o).add(&b.ob));
        let h2 = x.layernorm(&b.ln2_w, &b.ln2_b, eps);
        let ff = h2.matmul_q(&b.up).add(&b.upb).gelu().matmul_q(&b.down).add(&b.downb);
        x = x.add(&ff);
    }
    x = x.layernorm(&post_w, &post_b, eps).gather_rows(&inv_perm);

    // pixel shuffle: gather merge² neighbours, then concat channel-outer.
    let m2 = merge * merge;
    x = x.gather_rows(&ds_perm).reshape(&[n_out, m2, d]).permute(&[0, 2, 1]).contiguous()
        .reshape(&[n_out, d * m2]);
    let img_tok = x.matmul_q(&mm0).gelu().matmul_q(&mm1).gelu().matmul_q(&mm2);
    let out = img_tok.to_vec().await;
    let el = t0.elapsed();

    let (mut mn, mut mx, mut sum) = (f32::MAX, f32::MIN, 0f64);
    for &z in &out { mn = mn.min(z); mx = mx.max(z); sum += z as f64; }
    println!("\n  {n_out} image tokens × {proj_dim} in {el:.2?}");
    println!("  min {mn:.4}  max {mx:.4}  mean {:.4}  non-finite {}",
             sum / out.len() as f64, out.iter().filter(|z| !z.is_finite()).count());
    assert_eq!(out.len(), n_out * proj_dim, "expected [{n_out}, {proj_dim}]");
    assert!(out.iter().all(|z| z.is_finite()), "vision tower produced non-finite embeddings");
    println!("  ✅ shape and finiteness hold — compare against llama-mtmd-cli for numerics");
}
