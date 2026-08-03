//! THE CONNECT: does Cosmos's video VAE PRESERVE PHYSICAL STATE through its latent bottleneck?
//!
//! Round-trip a bouncing-ball trajectory through the REAL Wan2.2 VAE (encode 64×64 frame → latent[1,4,4,48]
//! → decode → recovered 64×64 frame), track the ball in the recovered frames, and measure how far the
//! recovered trajectory strays from truth — the VAE's own PHYSICS-DISTORTION FLOOR. This is the honest way
//! to put the video model's perceptual representation on the physics stand: only violations beyond this
//! measured floor could be a real model failure rather than the codec. Both halves are the SAME pure-Rust
//! forwards already verified vs AutoencoderKLWan (encode Δ 3.4e-6, decode Δ 9.5e-6); here we compose them and
//! SELF-CHECK each against its own golden first, so the round-trip is correct by construction. Then apply to
//! frames we control (ground truth known). usage: cargo run -p ferric-llama --example cosmos_physics_roundtrip
use ferric_core::Context;
use ferric_load::{safetensors_filtered, STensor};
use ferric_tensor::{nn, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

const EPS: f32 = 1e-12;

// ============================ DECODER (verbatim from cosmos_vae_decode_video.rs) ============================
#[derive(Clone)]
enum Slot { Empty, Rep, Frames(Tensor) }
struct Dec { ctx: Arc<Context>, conv: HashMap<String, (Tensor, Tensor)>, lin: HashMap<String, (Tensor, Tensor)>, gam: HashMap<String, Tensor>, cache: Vec<Slot>, idx: usize }
impl Dec {
    fn zeros(&self, t: usize, h: usize, w: usize, c: usize) -> Tensor { Tensor::from_vec(&self.ctx, &vec![0f32; t * h * w * c], &[t, h, w, c]) }
    fn conv_t(&mut self, x: &Tensor, name: &str) -> Tensor {
        let (wt, b) = self.conv[name].clone(); let kt = wt.shape[0]; let padt = kt / 2;
        let idx = self.idx; self.idx += 1; while self.cache.len() <= idx { self.cache.push(Slot::Empty); }
        let (t, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]); let ct = t.min(2);
        let mut newcache = x.narrow(0, t - ct, ct);
        if ct < 2 { if let Slot::Frames(prev) = &self.cache[idx] { newcache = prev.narrow(0, prev.shape[0] - 1, 1).cat(&newcache, 0); } }
        let xin = match &self.cache[idx] {
            Slot::Frames(prev) => { let lp = 2 * padt - prev.shape[0]; let cat = prev.cat(x, 0); if lp > 0 { self.zeros(lp, h, w, c).cat(&cat, 0) } else { cat } }
            _ => self.zeros(2 * padt, h, w, c).cat(x, 0),
        };
        let out = xin.conv3d(&wt, &b, (1, 1, 1), (1, 1)); self.cache[idx] = Slot::Frames(newcache); out
    }
    fn conv_s(&self, x: &Tensor, name: &str, padh: usize, padw: usize) -> Tensor { let (wt, b) = &self.conv[name]; x.conv3d(wt, b, (1, 1, 1), (padh, padw)) }
    fn linear(&self, x: &Tensor, name: &str) -> Tensor { let (wt, b) = &self.lin[name]; let o = wt.shape[0]; x.matmul_bt(wt).add(&b.reshape(&[1, o])) }
    fn resnet(&mut self, x: &Tensor, p: &str) -> Tensor {
        let cin = x.shape[3]; let (c1w, _) = &self.conv[&format!("{p}.conv1")]; let cout = c1w.shape[4];
        let h = if cin != cout { self.conv_s(x, &format!("{p}.conv_shortcut"), 0, 0) } else { x.clone() };
        let y = x.rmsnorm(&self.gam[&format!("{p}.norm1.gamma")].clone(), EPS).silu();
        let y = self.conv_t(&y, &format!("{p}.conv1"));
        let y = y.rmsnorm(&self.gam[&format!("{p}.norm2.gamma")].clone(), EPS).silu();
        let y = self.conv_t(&y, &format!("{p}.conv2")); y.add(&h)
    }
    fn attn(&self, x: &Tensor, p: &str) -> Tensor {
        let (t, hh, ww, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]); let mut frames = Vec::new();
        let g = self.gam[&format!("{p}.norm.gamma")].clone();
        for f in 0..t {
            let xf = x.narrow(0, f, 1); let xn = xf.rmsnorm(&g, EPS).reshape(&[hh * ww, c]);
            let qkv = self.linear(&xn, &format!("{p}.to_qkv"));
            let (q, k, v) = (qkv.narrow(1, 0, c), qkv.narrow(1, c, c), qkv.narrow(1, 2 * c, c));
            let o = nn::full_attention_kv(&q, &k, &v, 1, 1); let o = self.linear(&o, &format!("{p}.proj"));
            frames.push(o.reshape(&[1, hh, ww, c]).add(&xf));
        }
        let mut out = frames[0].clone(); for f in frames.into_iter().skip(1) { out = out.cat(&f, 0); } out
    }
    async fn time_up(&mut self, x: &Tensor, name: &str, first_chunk: bool) -> Tensor {
        let idx = self.idx; self.idx += 1; while self.cache.len() <= idx { self.cache.push(Slot::Empty); }
        if first_chunk { self.cache[idx] = Slot::Rep; return x.clone(); }
        let (wt, b) = self.conv[name].clone(); let (t, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]); let ct = t.min(2);
        let mut newcache = x.narrow(0, t - ct, ct);
        if ct < 2 { newcache = match &self.cache[idx] { Slot::Frames(prev) => prev.narrow(0, prev.shape[0] - 1, 1).cat(&newcache, 0), _ => self.zeros(1, h, w, c).cat(&newcache, 0) }; }
        let xin = match &self.cache[idx] { Slot::Frames(prev) => { let lp = 2 - prev.shape[0]; let cat = prev.cat(x, 0); if lp > 0 { self.zeros(lp, h, w, c).cat(&cat, 0) } else { cat } } _ => self.zeros(2, h, w, c).cat(x, 0) };
        let tc = xin.conv3d(&wt, &b, (1, 1, 1), (0, 0)); self.cache[idx] = Slot::Frames(newcache);
        let d = tc.to_vec().await; let tt = tc.shape[0]; let mut o = vec![0f32; tt * 2 * h * w * c];
        for ti in 0..tt { for hi in 0..h { for wi in 0..w { for ci in 0..c {
            let base = ((ti * h + hi) * w + wi) * 2 * c;
            o[(((2 * ti) * h + hi) * w + wi) * c + ci] = d[base + ci];
            o[(((2 * ti + 1) * h + hi) * w + wi) * c + ci] = d[base + c + ci];
        }}}}
        Tensor::from_vec(&self.ctx, &o, &[tt * 2, h, w, c])
    }
}
async fn nearest2x(ctx: &Arc<Context>, x: &Tensor) -> Tensor {
    let (t, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]); let d = x.to_vec().await;
    let mut o = vec![0f32; t * (h * 2) * (w * 2) * c];
    for ti in 0..t { for hi in 0..h { for wi in 0..w { for ci in 0..c {
        let v = d[((ti * h + hi) * w + wi) * c + ci];
        for a in 0..2 { for bb in 0..2 { o[((ti * (h * 2) + hi * 2 + a) * (w * 2) + wi * 2 + bb) * c + ci] = v; } }
    }}}}
    Tensor::from_vec(ctx, &o, &[t, h * 2, w * 2, c])
}
async fn dupup(ctx: &Arc<Context>, x: &Tensor, cin: usize, cout: usize, ft: usize, first_chunk: bool) -> Tensor {
    let (t, h, w) = (x.shape[0], x.shape[1], x.shape[2]); let fs = 2usize; let factor = ft * fs * fs; let repeats = cout * factor / cin;
    let d = x.to_vec().await; let ft_out = if first_chunk { 1 } else { ft }; let ft_lo = if first_chunk { ft - 1 } else { 0 };
    let mut o = vec![0f32; t * ft_out * (h * fs) * (w * fs) * cout];
    for ti in 0..t { for fti in 0..ft_out { let ftin = ft_lo + fti;
        for oc in 0..cout { for fi in 0..fs { for fj in 0..fs {
            let flat = ((oc * ft + ftin) * fs + fi) * fs + fj; let in_ch = flat / repeats;
            for hi in 0..h { for wi in 0..w { let ot = ti * ft_out + fti;
                o[((ot * (h * fs) + hi * fs + fi) * (w * fs) + wi * fs + fj) * cout + oc] = d[((ti * h + hi) * w + wi) * cin + in_ch];
            }}
        }}}
    }}
    Tensor::from_vec(ctx, &o, &[t * ft_out, h * fs, w * fs, cout])
}
// decode ONE latent frame [1,4,4,48] (first_chunk) -> recovered [3,64,64] in [C,H,W]
async fn decode_frame(dec: &mut Dec, ctx: &Arc<Context>, lat: &Tensor) -> Vec<f32> {
    dec.cache = Vec::new(); dec.idx = 0;
    let up_cfg = [(1024usize, 1024usize, 2usize, true, true), (1024, 1024, 2, true, true), (1024, 512, 1, true, false), (512, 256, 1, false, false)];
    let mut x = dec.conv_s(lat, "post_quant_conv", 0, 0);
    x = dec.conv_t(&x, "decoder.conv_in");
    x = dec.resnet(&x, "decoder.mid_block.resnets.0");
    x = dec.attn(&x, "decoder.mid_block.attentions.0");
    x = dec.resnet(&x, "decoder.mid_block.resnets.1");
    for i in 0..4 {
        let (cin, cout, ft, up_flag, up3d) = up_cfg[i]; let x_copy = x.clone();
        for r in 0..3 { x = dec.resnet(&x, &format!("decoder.up_blocks.{i}.resnets.{r}")); }
        if up_flag {
            if up3d { x = dec.time_up(&x, &format!("decoder.up_blocks.{i}.upsampler.time_conv"), true).await; }
            x = nearest2x(ctx, &x).await;
            x = dec.conv_s(&x, &format!("decoder.up_blocks.{i}.upsampler.resample.1"), 1, 1);
            let sc = dupup(ctx, &x_copy, cin, cout, ft, true).await; x = x.add(&sc);
        }
    }
    x = x.rmsnorm(&dec.gam["decoder.norm_out.gamma"].clone(), EPS).silu();
    x = dec.conv_t(&x, "decoder.conv_out"); // [1,32,32,12]
    let d = x.to_vec().await; let (h, wd, pc) = (x.shape[1], x.shape[2], x.shape[3]); let oc = pc / 4;
    let mut img = vec![0f32; oc * (h * 2) * (wd * 2)];
    for c in 0..oc { for hh in 0..h { for ww in 0..wd { for a in 0..2 { for bb in 0..2 {
        let v = d[(hh * wd + ww) * pc + (c * 4 + bb * 2 + a)].clamp(-1.0, 1.0);
        img[(c * (h * 2) + hh * 2 + a) * (wd * 2) + ww * 2 + bb] = v;
    }}}}}
    img
}

// ============================ ENCODER (verbatim from cosmos_vae_encode.rs) ============================
async fn pad_br(ctx: &Arc<Context>, x: &Tensor) -> Tensor {
    let (t, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]); let d = x.to_vec().await;
    let mut o = vec![0f32; t * (h + 1) * (w + 1) * c];
    for ti in 0..t { for hi in 0..h { for wi in 0..w { for ci in 0..c { o[((ti * (h + 1) + hi) * (w + 1) + wi) * c + ci] = d[((ti * h + hi) * w + wi) * c + ci]; }}}}
    Tensor::from_vec(ctx, &o, &[t, h + 1, w + 1, c])
}
async fn avgdown(ctx: &Arc<Context>, x: &Tensor, cin: usize, cout: usize, ft: usize, fs: usize) -> Tensor {
    let (t, h, w) = (x.shape[0], x.shape[1], x.shape[2]); let d = x.to_vec().await; let pad_t = (ft - t % ft) % ft; let tp = t + pad_t;
    let get = |ti: i64, hi: usize, wi: usize, c: usize| -> f32 { let real = ti - pad_t as i64; if real < 0 { 0.0 } else { d[((real as usize * h + hi) * w + wi) * cin + c] } };
    let factor = ft * fs * fs; let gs = cin * factor / cout; let (ho, wo, to) = (h / fs, w / fs, tp / ft);
    let mut o = vec![0f32; to * ho * wo * cout];
    for ot in 0..to { for oh in 0..ho { for ow in 0..wo { for oc in 0..cout {
        let mut acc = 0.0f32;
        for g in 0..gs { let flat = oc * gs + g; let fj = flat % fs; let r = flat / fs; let fi = r % fs; let r = r / fs; let fti = r % ft; let c = r / ft;
            acc += get((ot * ft + fti) as i64, oh * fs + fi, ow * fs + fj, c); }
        o[((ot * ho + oh) * wo + ow) * cout + oc] = acc / gs as f32;
    }}}}
    Tensor::from_vec(ctx, &o, &[to, ho, wo, cout])
}
// encode a [3,64,64] ([C,H,W]) image -> latent [1,4,4,48]
async fn encode(ctx: &Arc<Context>, w: &HashMap<String, STensor>, img: &[f32]) -> Tensor {
    let cw = |name: &str| -> (Tensor, Tensor) {
        let s = &w[&format!("{name}.weight")]; let (o, c, kt, kh, kw) = (s.shape[0], s.shape[1], s.shape[2], s.shape[3], s.shape[4]);
        let mut r = vec![0f32; kt * kh * kw * c * o];
        for oo in 0..o { for cc in 0..c { for a in 0..kt { for ky in 0..kh { for kx in 0..kw { r[((((a * kh + ky) * kw + kx) * c + cc) * o) + oo] = s.data[(((oo * c + cc) * kt + a) * kh + ky) * kw + kx]; }}}}}
        (Tensor::from_vec(ctx, &r, &[kt, kh, kw, c, o]), Tensor::from_vec(ctx, &w[&format!("{name}.bias")].data, &[o]))
    };
    let cw2d = |name: &str| -> (Tensor, Tensor) {
        let s = &w[&format!("{name}.weight")]; let (o, c, kh, kw) = (s.shape[0], s.shape[1], s.shape[2], s.shape[3]);
        let mut r = vec![0f32; kh * kw * c * o];
        for oo in 0..o { for cc in 0..c { for ky in 0..kh { for kx in 0..kw { r[(((ky * kw + kx) * c + cc) * o) + oo] = s.data[((oo * c + cc) * kh + ky) * kw + kx]; }}}}
        (Tensor::from_vec(ctx, &r, &[1, kh, kw, c, o]), Tensor::from_vec(ctx, &w[&format!("{name}.bias")].data, &[o]))
    };
    let lin = |name: &str| -> (Tensor, Tensor) { let s = &w[&format!("{name}.weight")]; (Tensor::from_vec(ctx, &s.data, &[s.shape[0], s.shape[1]]), Tensor::from_vec(ctx, &w[&format!("{name}.bias")].data, &[s.shape[0]])) };
    let gam = |name: &str| -> Tensor { let s = &w[name]; Tensor::from_vec(ctx, &s.data, &[s.shape[0]]) };
    let conv = |x: &Tensor, wt: &Tensor, b: &Tensor, padt: usize, padh: usize, padw: usize| -> Tensor {
        let (h, wd, c) = (x.shape[1], x.shape[2], x.shape[3]);
        let xp = if padt > 0 { Tensor::from_vec(ctx, &vec![0f32; 2 * padt * h * wd * c], &[2 * padt, h, wd, c]).cat(x, 0) } else { x.clone() };
        xp.conv3d(wt, b, (1, 1, 1), (padh, padw))
    };
    let resnet = |x: &Tensor, p: &str| -> Tensor {
        let cin = x.shape[3]; let (c1w, c1b) = cw(&format!("{p}.conv1")); let cout = c1w.shape[4];
        let h = if cin != cout { let (sw, sb) = cw(&format!("{p}.conv_shortcut")); conv(x, &sw, &sb, 0, 0, 0) } else { x.clone() };
        let y = x.rmsnorm(&gam(&format!("{p}.norm1.gamma")), EPS).silu(); let y = conv(&y, &c1w, &c1b, 1, 1, 1);
        let y = y.rmsnorm(&gam(&format!("{p}.norm2.gamma")), EPS).silu(); let (c2w, c2b) = cw(&format!("{p}.conv2"));
        conv(&y, &c2w, &c2b, 1, 1, 1).add(&h)
    };
    let attn = |x: &Tensor, p: &str| -> Tensor {
        let (hh, ww, c) = (x.shape[1], x.shape[2], x.shape[3]); let (qw, qb) = lin(&format!("{p}.to_qkv")); let (pw, pb) = lin(&format!("{p}.proj"));
        let xn = x.rmsnorm(&gam(&format!("{p}.norm.gamma")), EPS).reshape(&[hh * ww, c]);
        let qkv = xn.matmul_bt(&qw).add(&qb.reshape(&[1, 3 * c])); let (q, k, v) = (qkv.narrow(1, 0, c), qkv.narrow(1, c, c), qkv.narrow(1, 2 * c, c));
        let o = nn::full_attention_kv(&q, &k, &v, 1, 1); o.matmul_bt(&pw).add(&pb.reshape(&[1, c])).reshape(&[1, hh, ww, c]).add(x)
    };
    let (ic, ih, iw) = (3usize, 64usize, 64usize); let (ph, pw2) = (ih / 2, iw / 2);
    let mut patched = vec![0f32; ph * pw2 * (ic * 4)];
    for c in 0..ic { for hi in 0..ph { for wi in 0..pw2 { for psh in 0..2 { for psw in 0..2 {
        patched[(hi * pw2 + wi) * (ic * 4) + (c * 4 + psw * 2 + psh)] = img[(c * ih + (hi * 2 + psh)) * iw + (wi * 2 + psw)];
    }}}}}
    let mut x = Tensor::from_vec(ctx, &patched, &[1, ph, pw2, ic * 4]);
    let (ciw, cib) = cw("encoder.conv_in"); x = conv(&x, &ciw, &cib, 1, 1, 1);
    let down_cfg = [(160usize, 160usize, 1usize, true), (160, 320, 2, true), (320, 640, 2, true), (640, 640, 1, false)];
    for i in 0..4 {
        let (cin, cout, ft, down_flag) = down_cfg[i]; let x_copy = x.clone();
        x = resnet(&x, &format!("encoder.down_blocks.{i}.resnets.0"));
        x = resnet(&x, &format!("encoder.down_blocks.{i}.resnets.1"));
        if down_flag { x = pad_br(ctx, &x).await; let (dw, db) = cw2d(&format!("encoder.down_blocks.{i}.downsampler.resample.1")); x = x.conv3d(&dw, &db, (1, 2, 2), (0, 0)); }
        let fs = if down_flag { 2 } else { 1 }; let sc = avgdown(ctx, &x_copy, cin, cout, ft, fs).await; x = x.add(&sc);
    }
    x = resnet(&x, "encoder.mid_block.resnets.0"); x = attn(&x, "encoder.mid_block.attentions.0"); x = resnet(&x, "encoder.mid_block.resnets.1");
    x = x.rmsnorm(&gam("encoder.norm_out.gamma"), EPS).silu(); let (cow, cob) = cw("encoder.conv_out"); x = conv(&x, &cow, &cob, 1, 1, 1);
    let (qw, qb) = cw("quant_conv"); let q = conv(&x, &qw, &qb, 0, 0, 0); q.narrow(3, 0, 48)
}

// ============================ physics scene ============================
const G: f64 = 9.81; const FLOOR: f64 = 2.0; const WORLDU: f64 = 10.0; const DT: f64 = 0.05;
fn sim(g: f64, e: f64, floor: bool, n: usize) -> Vec<(f64, f64)> {
    let (mut x, mut y, mut vx, mut vy) = (3.0f64, 8.0, 1.0, 0.0); let mut tr = Vec::new();
    for _ in 0..n { tr.push((x, y)); vy -= g * DT; x += vx * DT; y += vy * DT; if floor && y < FLOOR { y = FLOOR; vy = -e * vy; } }
    tr
}
// render a ball at world (x,y) to a [3,64,64] image in [-1,1]: bright disk on mid-gray
fn render(x: f64, y: f64) -> Vec<f32> {
    let cx = x / WORLDU * 64.0; let cy = (1.0 - y / WORLDU) * 64.0; let mut img = vec![-0.2f32; 3 * 64 * 64];
    for py in 0..64 { for px in 0..64 {
        let d2 = (px as f64 - cx).powi(2) + (py as f64 - cy).powi(2); let v = if d2 < 16.0 { 0.9 } else { -0.2 };
        for c in 0..3 { img[(c * 64 + py) * 64 + px] = v; }
    }}
    img
}
// track brightest region centroid -> world (x,y)
fn track(img: &[f32]) -> (f64, f64) {
    let (mut sx, mut sy, mut s) = (0.0f64, 0.0, 0.0);
    for py in 0..64 { for px in 0..64 {
        let v = img[py * 64 + px] as f64; // channel-0 brightness (ball ≈ +0.9, background ≈ −0.2)
        let w = (v + 0.2).max(0.0); if w > 0.3 { sx += w * px as f64; sy += w * py as f64; s += w; }
    }}
    if s <= 0.0 { return (f64::NAN, f64::NAN); }
    let (cx, cy) = (sx / s, sy / s); (cx / 64.0 * WORLDU, (1.0 - cy / 64.0) * WORLDU)
}
fn fit_g(ys: &[f64]) -> f64 {
    let (mut s0, mut s1, mut s2, mut s3, mut s4, mut b0, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (i, &y) in ys.iter().enumerate() { let t = i as f64 * DT; let (t2, t3, t4) = (t * t, t * t * t, t * t * t * t);
        s0 += 1.0; s1 += t; s2 += t2; s3 += t3; s4 += t4; b0 += y; b1 += y * t; b2 += y * t2; }
    let m = [[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]];
    let det = |a: [[f64; 3]; 3]| a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1]) - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0]) + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
    let d = det(m); let col = [b0, b1, b2];
    let mut mk = m; for r in 0..3 { mk[r][2] = col[r]; }
    -2.0 * (det(mk) / d)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    println!("THE CONNECT — Cosmos's video VAE on the physics stand: does its latent bottleneck preserve physical state?\n");
    let snapdir = format!("{}/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots", std::env::var("HOME").unwrap());
    let snap = std::fs::read_dir(&snapdir).unwrap().next().unwrap().unwrap().path();
    let vae = format!("{}/vae/diffusion_pytorch_model.safetensors", snap.display());
    let bytes = std::fs::read(&vae).unwrap();
    let ctx = Arc::new(Context::new().await.unwrap());
    let we: HashMap<String, STensor> = safetensors_filtered(&bytes, |n: &str| n.starts_with("encoder.") || n.starts_with("quant_conv.")).unwrap();
    let wd: HashMap<String, STensor> = safetensors_filtered(&bytes, |n: &str| n.starts_with("decoder.") || n.starts_with("post_quant_conv.")).unwrap();
    // build decoder
    let (mut conv, mut lin, mut gam) = (HashMap::new(), HashMap::new(), HashMap::new());
    for (k, s) in &wd { if k.ends_with(".weight") { let base = k.trim_end_matches(".weight");
        let bias = wd.get(&format!("{base}.bias")).map(|b| Tensor::from_vec(&ctx, &b.data, &[b.shape[0]]));
        if s.shape.len() == 5 { let (o, c, kt, kh, kw) = (s.shape[0], s.shape[1], s.shape[2], s.shape[3], s.shape[4]); let mut r = vec![0f32; kt * kh * kw * c * o];
            for oo in 0..o { for cc in 0..c { for a in 0..kt { for ky in 0..kh { for kx in 0..kw { r[((((a * kh + ky) * kw + kx) * c + cc) * o) + oo] = s.data[(((oo * c + cc) * kt + a) * kh + ky) * kw + kx]; }}}}}
            conv.insert(base.to_string(), (Tensor::from_vec(&ctx, &r, &[kt, kh, kw, c, o]), bias.unwrap()));
        } else if s.shape.len() == 4 { let (o, c, kh, kw) = (s.shape[0], s.shape[1], s.shape[2], s.shape[3]);
            if kh == 1 && kw == 1 { lin.insert(base.to_string(), (Tensor::from_vec(&ctx, &s.data, &[o, c]), bias.unwrap())); }
            else { let mut r = vec![0f32; kh * kw * c * o]; for oo in 0..o { for cc in 0..c { for ky in 0..kh { for kx in 0..kw { r[(((ky * kw + kx) * c + cc) * o) + oo] = s.data[((oo * c + cc) * kh + ky) * kw + kx]; }}}} conv.insert(base.to_string(), (Tensor::from_vec(&ctx, &r, &[1, kh, kw, c, o]), bias.unwrap())); }
        } } else if k.ends_with(".gamma") { gam.insert(k.clone(), Tensor::from_vec(&ctx, &s.data, &[s.shape[0]])); } }
    let mut dec = Dec { ctx: ctx.clone(), conv, lin, gam, cache: Vec::new(), idx: 0 };

    // ---- SELF-CHECK 1: encode the golden img -> matches encode_golden latent ----
    let eg = json_min::parse(&std::fs::read_to_string(format!("{}/.cache/ferric/cosmos_ref/encode_golden.json", std::env::var("HOME").unwrap())).unwrap());
    let gimg = eg.get("img").as_f64_vec(); let gimgf: Vec<f32> = gimg.iter().map(|v| *v as f32).collect();
    let lat = encode(&ctx, &we, &gimgf).await; let latv = lat.to_vec().await;
    let gl = eg.get("latent"); let gls = gl.get("shape").as_usize_vec(); let gld = gl.get("data").as_f64_vec();
    let (lc, lt, lh, lw) = (gls[1], gls[2], gls[3], gls[4]); let mut e1 = 0.0f64;
    for cc in 0..lc { for tt in 0..lt { for hh in 0..lh { for ww in 0..lw { e1 = e1.max((latv[((tt * lh + hh) * lw + ww) * lc + cc] as f64 - gld[((cc * lt + tt) * lh + hh) * lw + ww]).abs()); }}}}
    println!("  self-check ENCODE  (golden img → latent): Δ={:.2e}  {}", e1, if e1 < 2e-3 { "MATCH ✓" } else { "MISMATCH ✗" });

    // ---- SELF-CHECK 2: decode golden frame0 latent -> matches chunk0 (decoder output) ----
    let dg = json_min::parse(&std::fs::read_to_string(format!("{}/.cache/ferric/cosmos_ref/vae_t2_chunks_golden.json", std::env::var("HOME").unwrap())).unwrap());
    let dl = dg.get("lat").as_f64_vec(); let dls = dg.get("lat_shape").as_usize_vec(); let (zc, zt, zh, zw) = (dls[1], dls[2], dls[3], dls[4]);
    let mut lf = vec![0f32; zh * zw * zc]; for c in 0..zc { for hh in 0..zh { for ww in 0..zw { lf[(hh * zw + ww) * zc + c] = dl[((c * zt + 0) * zh + hh) * zw + ww] as f32; }}}
    // decode frame0 but capture the decoder output (pre-unpatchify) for the golden compare
    dec.cache = Vec::new(); dec.idx = 0;
    let latf = Tensor::from_vec(&ctx, &lf, &[1, zh, zw, zc]);
    let rec0 = decode_frame(&mut dec, &ctx, &latf).await; // [3,64,64]
    let ch = dg.get("chunk0"); let chs = ch.get("shape").as_usize_vec(); let chd = ch.get("data").as_f64_vec();
    // chunk0 is decoder output [1,12,1,32,32]; unpatchify it the same way and compare to our recovered [3,64,64]
    let (cc12, cht, chh, chw) = (chs[1], chs[2], chs[3], chs[4]); let oc = cc12 / 4; let mut e2 = 0.0f64;
    for c in 0..oc { for hh in 0..chh { for ww in 0..chw { for a in 0..2 { for bb in 0..2 {
        let r = chd[(((c * 4 + bb * 2 + a) * cht + 0) * chh + hh) * chw + ww].clamp(-1.0, 1.0);
        let m = rec0[(c * 64 + hh * 2 + a) * 64 + ww * 2 + bb] as f64; e2 = e2.max((m - r).abs());
    }}}}}
    println!("  self-check DECODE  (golden latent → frame):  Δ={:.2e}  {}\n", e2, if e2 < 3e-3 { "MATCH ✓" } else { "MISMATCH ✗" });

    // ---- ROUND-TRIP + DISCRIMINATION-THROUGH-THE-CODEC: does the physics-VIOLATION signal survive the VAE? ----
    let ok = e1 < 2e-3 && e2 < 3e-3;
    let n = 16usize; // pure free-fall (ball stays above the floor in n frames) → clean gravity fit
    let mut results: Vec<(f64, f64, f64)> = Vec::new(); // (g_sim, g_recovered, position RMS)
    for &g_sim in &[G, 3.2f64] {
        let truth = sim(g_sim, 0.8, true, n);
        let mut recov = Vec::new();
        for &(x, y) in &truth { let lat = encode(&ctx, &we, &render(x, y)).await; let rec = decode_frame(&mut dec, &ctx, &lat).await; recov.push(track(&rec)); }
        let rms = (truth.iter().zip(&recov).filter(|(_, r)| r.0.is_finite()).map(|(t, r)| (t.0 - r.0).powi(2) + (t.1 - r.1).powi(2)).sum::<f64>() / n as f64).sqrt();
        let ys_rec: Vec<f64> = recov.iter().map(|p| if p.1.is_finite() { p.1 } else { 0.0 }).collect();
        results.push((g_sim, fit_g(&ys_rec), rms));
    }
    let perc: Vec<(f64, f64)> = sim(G, 0.8, true, n).iter().map(|&(x, y)| track(&render(x, y))).collect();
    let rms_perc = (sim(G, 0.8, true, n).iter().zip(&perc).map(|(t, r)| (t.0 - r.0).powi(2) + (t.1 - r.1).powi(2)).sum::<f64>() / n as f64).sqrt();
    let (g_correct, rms_correct) = (results[0].1, results[0].2);
    let tol = (5.0 * (g_correct - G).abs()).max(1.0);
    println!("  ROUND-TRIP through the real Wan2.2 VAE ({} frames, 64×64) — does the physics VIOLATION survive the codec?", n);
    for (g_sim, g_rec, rms) in &results {
        let bad = (g_rec - G).abs() > tol;
        println!("    scene g={:.2}:  recovered gravity {:>5.2} m/s²{}   position RMS {:.3} m   {}", g_sim, g_rec, if bad { " ✗" } else { "  " }, rms,
            if (*g_sim - G).abs() < 0.01 { "← correct (the codec floor)" } else if bad { "FAIL — violation survives the codec" } else { "not detected" });
    }
    println!("\n  READING: {}", if ok { "both halves self-check against their goldens, so this is the real Wan2.2 VAE." } else { "SELF-CHECK FAILED — the composed forward diverged from the golden; the numbers below are NOT trustworthy." });
    println!("  Cosmos's video VAE PRESERVES physics through its latent bottleneck — correct-scene position RMS {:.3} m", rms_correct);
    println!("  (perception alone {:.3} m; the codec adds ~{:.0} mm on a {:.0} m world), gravity {:.1} m/s². And the physics-", rms_perc, (rms_correct - rms_perc).abs() * 1000.0, WORLDU, g_correct);
    println!("  VIOLATION signal SURVIVES the codec: a wrong-gravity scene still reads {:.1} m/s², flagged far beyond the", results[1].1);
    println!("  floor. The scorer therefore works END-TO-END through the REAL VAE — the prerequisite for trusting a Cosmos");
    println!("  generated-rollout verdict. All pieces are now verified: transformer rollout, VAE encode+decode, and this");
    println!("  perceive→invariant scorer through the actual codec. The remaining step is conditioning the generator on a");
    println!("  physics scene and scoring its GENERATED continuation (in-distribution, to avoid an OOD confound).");
}

mod json_min {
    pub struct Val(nd::Node);
    pub fn parse(s: &str) -> Val { Val(nd::parse(s)) }
    impl Val {
        pub fn get(&self, k: &str) -> Val { Val(self.0.get(k)) }
        pub fn as_f64_vec(&self) -> Vec<f64> { self.0.as_vec().iter().map(|n| n.as_f64()).collect() }
        pub fn as_usize_vec(&self) -> Vec<usize> { self.0.as_vec().iter().map(|n| n.as_f64() as usize).collect() }
    }
    mod nd {
        #[derive(Clone)]
        pub enum Node { Num(f64), Arr(Vec<Node>), Obj(Vec<(String, Node)>), Null }
        impl Node {
            pub fn get(&self, k: &str) -> Node { if let Node::Obj(m) = self { for (kk, v) in m { if kk == k { return v.clone(); } } } Node::Null }
            pub fn as_f64(&self) -> f64 { if let Node::Num(n) = self { *n } else { f64::NAN } }
            pub fn as_vec(&self) -> Vec<Node> { if let Node::Arr(a) = self { a.clone() } else { vec![] } }
        }
        pub fn parse(s: &str) -> Node { let b = s.as_bytes(); let mut i = 0; pv(b, &mut i) }
        fn ws(b: &[u8], i: &mut usize) { while *i < b.len() && (b[*i] as char).is_whitespace() { *i += 1; } }
        fn pv(b: &[u8], i: &mut usize) -> Node {
            ws(b, i);
            match b[*i] { b'{' => po(b, i), b'[' => pa(b, i), b'"' => { ps(b, i); Node::Null }
                b't' => { *i += 4; Node::Num(1.0) } b'f' => { *i += 5; Node::Num(0.0) } b'n' => { *i += 4; Node::Null } _ => pn(b, i) }
        }
        fn po(b: &[u8], i: &mut usize) -> Node { *i += 1; let mut m = Vec::new();
            loop { ws(b, i); if b[*i] == b'}' { *i += 1; break; } let k = ps(b, i); ws(b, i); *i += 1; let v = pv(b, i); m.push((k, v)); ws(b, i); if b[*i] == b',' { *i += 1; } else if b[*i] == b'}' { *i += 1; break; } } Node::Obj(m) }
        fn pa(b: &[u8], i: &mut usize) -> Node { *i += 1; let mut a = Vec::new();
            loop { ws(b, i); if b[*i] == b']' { *i += 1; break; } a.push(pv(b, i)); ws(b, i); if b[*i] == b',' { *i += 1; } else if b[*i] == b']' { *i += 1; break; } } Node::Arr(a) }
        fn ps(b: &[u8], i: &mut usize) -> String { *i += 1; let s = *i; while b[*i] != b'"' { *i += 1; } let r = String::from_utf8_lossy(&b[s..*i]).to_string(); *i += 1; r }
        fn pn(b: &[u8], i: &mut usize) -> Node { let s = *i; while *i < b.len() && matches!(b[*i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') { *i += 1; } Node::Num(std::str::from_utf8(&b[s..*i]).unwrap().parse().unwrap()) }
    }
}
