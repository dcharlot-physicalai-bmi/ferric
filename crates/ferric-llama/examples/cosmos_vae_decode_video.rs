//! MULTI-FRAME Wan2.2 VAE video decode in pure Rust — the streaming causal decoder with feat_cache,
//! time_conv temporal upsampling, and temporal DupUp. Processes a T>1 latent frame-by-frame (each latent
//! frame = one chunk through the whole decoder, sharing feat_cache for cross-chunk causal context),
//! concatenates outputs, unpatchifies. Verified per-chunk + final vs the real AutoencoderKLWan.decode.
//! usage: cargo run -p ferric-llama --example cosmos_vae_decode_video --release -- <wan-vae-safetensors>
use ferric_core::Context;
use ferric_load::{safetensors_filtered, STensor};
use ferric_tensor::{nn, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

const EPS: f32 = 1e-12;

#[derive(Clone)]
enum Slot { Empty, Rep, Frames(Tensor) }

struct Dec {
    ctx: Arc<Context>,
    conv: HashMap<String, (Tensor, Tensor)>, // reordered [kt,kh,kw,C,O] + bias
    lin: HashMap<String, (Tensor, Tensor)>,  // 1x1 as matmul_bt weight [O,C] + bias
    gam: HashMap<String, Tensor>,
    cache: Vec<Slot>,
    idx: usize,
}

impl Dec {
    fn zeros(&self, t: usize, h: usize, w: usize, c: usize) -> Tensor {
        Tensor::from_vec(&self.ctx, &vec![0f32; t * h * w * c], &[t, h, w, c])
    }
    // temporal causal conv (kernel (kt,3,3) pad(_,1,1)) WITH feat_cache; consumes an idx slot
    fn conv_t(&mut self, x: &Tensor, name: &str) -> Tensor {
        let (wt, b) = self.conv[name].clone();
        let kt = wt.shape[0];
        let padt = kt / 2; // 1 for kt=3
        let idx = self.idx; self.idx += 1;
        while self.cache.len() <= idx { self.cache.push(Slot::Empty); }
        let (t, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        let ct = t.min(2);
        let mut newcache = x.narrow(0, t - ct, ct);
        if ct < 2 { if let Slot::Frames(prev) = &self.cache[idx] { newcache = prev.narrow(0, prev.shape[0] - 1, 1).cat(&newcache, 0); } }
        let xin = match &self.cache[idx] {
            Slot::Frames(prev) => {
                let lp = 2 * padt - prev.shape[0];
                let cat = prev.cat(x, 0);
                if lp > 0 { self.zeros(lp, h, w, c).cat(&cat, 0) } else { cat }
            }
            _ => self.zeros(2 * padt, h, w, c).cat(x, 0),
        };
        let out = xin.conv3d(&wt, &b, (1, 1, 1), (1, 1));
        self.cache[idx] = Slot::Frames(newcache);
        out
    }
    // spatial conv (no temporal cache): kt=1 kernels — post_quant/conv_shortcut (pad0) or upsampler Conv2d (pad1)
    fn conv_s(&self, x: &Tensor, name: &str, padh: usize, padw: usize) -> Tensor {
        let (wt, b) = &self.conv[name];
        x.conv3d(wt, b, (1, 1, 1), (padh, padw))
    }
    fn linear(&self, x: &Tensor, name: &str) -> Tensor {
        let (wt, b) = &self.lin[name];
        let o = wt.shape[0];
        x.matmul_bt(wt).add(&b.reshape(&[1, o]))
    }
    fn resnet(&mut self, x: &Tensor, p: &str) -> Tensor {
        let cin = x.shape[3];
        let (c1w, _) = &self.conv[&format!("{p}.conv1")];
        let cout = c1w.shape[4];
        let h = if cin != cout { self.conv_s(x, &format!("{p}.conv_shortcut"), 0, 0) } else { x.clone() };
        let y = x.rmsnorm(&self.gam[&format!("{p}.norm1.gamma")].clone(), EPS).silu();
        let y = self.conv_t(&y, &format!("{p}.conv1"));
        let y = y.rmsnorm(&self.gam[&format!("{p}.norm2.gamma")].clone(), EPS).silu();
        let y = self.conv_t(&y, &format!("{p}.conv2"));
        y.add(&h)
    }
    fn attn(&self, x: &Tensor, p: &str) -> Tensor {
        // per-frame single-head spatial attention (loop over T)
        let (t, hh, ww, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        let mut frames = Vec::new();
        let g = self.gam[&format!("{p}.norm.gamma")].clone();
        for f in 0..t {
            let xf = x.narrow(0, f, 1);
            let xn = xf.rmsnorm(&g, EPS).reshape(&[hh * ww, c]);
            let qkv = self.linear(&xn, &format!("{p}.to_qkv")); // [hw,3c]
            let q = qkv.narrow(1, 0, c);
            let k = qkv.narrow(1, c, c);
            let v = qkv.narrow(1, 2 * c, c);
            let o = nn::full_attention_kv(&q, &k, &v, 1, 1);
            let o = self.linear(&o, &format!("{p}.proj"));
            frames.push(o.reshape(&[1, hh, ww, c]).add(&xf));
        }
        let mut out = frames[0].clone();
        for f in frames.into_iter().skip(1) { out = out.cat(&f, 0); }
        out
    }
    // WanResample upsample3d: time_conv temporal upsample (feat_cache), then caller does spatial
    async fn time_up(&mut self, x: &Tensor, name: &str, first_chunk: bool) -> Tensor {
        let idx = self.idx; self.idx += 1;
        while self.cache.len() <= idx { self.cache.push(Slot::Empty); }
        if first_chunk { self.cache[idx] = Slot::Rep; return x.clone(); } // Rep: skip temporal
        let (wt, b) = self.conv[name].clone();
        let (t, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        let ct = t.min(2);
        let mut newcache = x.narrow(0, t - ct, ct);
        if ct < 2 {
            newcache = match &self.cache[idx] {
                Slot::Frames(prev) => prev.narrow(0, prev.shape[0] - 1, 1).cat(&newcache, 0),
                _ => self.zeros(1, h, w, c).cat(&newcache, 0), // Rep -> zero pad
            };
        }
        let xin = match &self.cache[idx] {
            Slot::Frames(prev) => { let lp = 2 - prev.shape[0]; let cat = prev.cat(x, 0); if lp > 0 { self.zeros(lp, h, w, c).cat(&cat, 0) } else { cat } }
            _ => self.zeros(2, h, w, c).cat(x, 0), // Rep: full left pad
        };
        let tc = xin.conv3d(&wt, &b, (1, 1, 1), (0, 0)); // (3,1,1) -> [T,H,W,2C]
        self.cache[idx] = Slot::Frames(newcache);
        // temporal doubling: out[2t]=tc[t][:,:,0:c], out[2t+1]=tc[t][:,:,c:2c]
        let d = tc.to_vec().await;
        let tt = tc.shape[0];
        let mut o = vec![0f32; tt * 2 * h * w * c];
        for ti in 0..tt { for hi in 0..h { for wi in 0..w { for ci in 0..c {
            let base = ((ti * h + hi) * w + wi) * 2 * c;
            o[(((2 * ti) * h + hi) * w + wi) * c + ci] = d[base + ci];
            o[(((2 * ti + 1) * h + hi) * w + wi) * c + ci] = d[base + c + ci];
        }}}}
        Tensor::from_vec(&self.ctx, &o, &[tt * 2, h, w, c])
    }
}

// nearest-2x spatial over [T,H,W,C]
async fn nearest2x(ctx: &Arc<Context>, x: &Tensor) -> Tensor {
    let (t, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let d = x.to_vec().await;
    let mut o = vec![0f32; t * (h * 2) * (w * 2) * c];
    for ti in 0..t { for hi in 0..h { for wi in 0..w { for ci in 0..c {
        let v = d[((ti * h + hi) * w + wi) * c + ci];
        for a in 0..2 { for bb in 0..2 { o[(((ti * (h * 2) + hi * 2 + a) * (w * 2) + wi * 2 + bb)) * c + ci] = v; } }
    }}}}
    Tensor::from_vec(ctx, &o, &[t, h * 2, w * 2, c])
}

// DupUp3D avg_shortcut over [T,H,W,cin] -> [T*ft(or T if first_chunk),2H,2W,cout]
async fn dupup(ctx: &Arc<Context>, x: &Tensor, cin: usize, cout: usize, ft: usize, first_chunk: bool) -> Tensor {
    let (t, h, w) = (x.shape[0], x.shape[1], x.shape[2]);
    let fs = 2usize;
    let factor = ft * fs * fs;
    let repeats = cout * factor / cin;
    let d = x.to_vec().await;
    let ft_out = if first_chunk { 1 } else { ft };
    let ft_lo = if first_chunk { ft - 1 } else { 0 }; // first_chunk keeps only the last temporal slice
    let mut o = vec![0f32; t * ft_out * (h * fs) * (w * fs) * cout];
    for ti in 0..t { for fti in 0..ft_out {
        let ftin = ft_lo + fti;
        for oc in 0..cout { for fi in 0..fs { for fj in 0..fs {
            let flat = ((oc * ft + ftin) * fs + fi) * fs + fj;
            let in_ch = flat / repeats;
            for hi in 0..h { for wi in 0..w {
                let ot = ti * ft_out + fti;
                o[(((ot * (h * fs) + hi * fs + fi) * (w * fs) + wi * fs + fj)) * cout + oc] = d[((ti * h + hi) * w + wi) * cin + in_ch];
            }}
        }}}
    }}
    Tensor::from_vec(ctx, &o, &[t * ft_out, h * fs, w * fs, cout])
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let vae = std::env::args().nth(1).unwrap_or_else(|| {
        let d = format!("{}/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots", std::env::var("HOME").unwrap());
        let snap = std::fs::read_dir(&d).unwrap().next().unwrap().unwrap().path();
        format!("{}/vae/diffusion_pytorch_model.safetensors", snap.display())
    });
    let gp = format!("{}/.cache/ferric/cosmos_ref/vae_t2_chunks_golden.json", std::env::var("HOME").unwrap());
    let jg = json_min::parse(&std::fs::read_to_string(&gp).unwrap());
    let ctx = Arc::new(Context::new().await.unwrap());
    let w: HashMap<String, STensor> = safetensors_filtered(&std::fs::read(&vae).unwrap(),
        |n: &str| n.starts_with("decoder.") || n.starts_with("post_quant_conv.")).unwrap();

    // preload reordered weights
    let mut conv = HashMap::new();
    let mut lin = HashMap::new();
    let mut gam = HashMap::new();
    for (k, s) in &w {
        if k.ends_with(".weight") {
            let base = k.trim_end_matches(".weight");
            let bias = w.get(&format!("{base}.bias")).map(|b| Tensor::from_vec(&ctx, &b.data, &[b.shape[0]]));
            if s.shape.len() == 5 { // conv3d [O,C,kt,kh,kw]
                let (o, c, kt, kh, kw) = (s.shape[0], s.shape[1], s.shape[2], s.shape[3], s.shape[4]);
                let mut r = vec![0f32; kt * kh * kw * c * o];
                for oo in 0..o { for cc in 0..c { for a in 0..kt { for ky in 0..kh { for kx in 0..kw {
                    r[((((a * kh + ky) * kw + kx) * c + cc) * o) + oo] = s.data[(((oo * c + cc) * kt + a) * kh + ky) * kw + kx];
                }}}}}
                conv.insert(base.to_string(), (Tensor::from_vec(&ctx, &r, &[kt, kh, kw, c, o]), bias.unwrap()));
            } else if s.shape.len() == 4 { // Conv2d [O,C,kh,kw]
                let (o, c, kh, kw) = (s.shape[0], s.shape[1], s.shape[2], s.shape[3]);
                if kh == 1 && kw == 1 { // 1x1 -> matmul
                    lin.insert(base.to_string(), (Tensor::from_vec(&ctx, &s.data, &[o, c]), bias.unwrap()));
                } else {
                    let mut r = vec![0f32; kh * kw * c * o];
                    for oo in 0..o { for cc in 0..c { for ky in 0..kh { for kx in 0..kw {
                        r[(((ky * kw + kx) * c + cc) * o) + oo] = s.data[((oo * c + cc) * kh + ky) * kw + kx];
                    }}}}
                    conv.insert(base.to_string(), (Tensor::from_vec(&ctx, &r, &[1, kh, kw, c, o]), bias.unwrap()));
                }
            } else if k.ends_with(".gamma") {
                gam.insert(k.clone(), Tensor::from_vec(&ctx, &s.data, &[s.shape[0]]));
            }
        } else if k.ends_with(".gamma") {
            gam.insert(k.clone(), Tensor::from_vec(&ctx, &s.data, &[s.shape[0]]));
        }
    }
    let mut dec = Dec { ctx: ctx.clone(), conv, lin, gam, cache: Vec::new(), idx: 0 };

    // latent [1,48,T,4,4] -> per-frame [1,4,4,48]
    let lat = jg.get("lat").as_f64_vec();
    let ls = jg.get("lat_shape").as_usize_vec(); // [1,48,T,4,4]
    let (zc, zt, zh, zw) = (ls[1], ls[2], ls[3], ls[4]);
    let up_cfg = [(1024usize, 1024usize, 2usize, true, true), (1024, 1024, 2, true, true), (1024, 512, 1, true, false), (512, 256, 1, false, false)];

    let mut chunk_outs: Vec<Tensor> = Vec::new();
    for fi in 0..zt {
        let first_chunk = fi == 0;
        dec.idx = 0;
        let mut lf = vec![0f32; zh * zw * zc];
        for c in 0..zc { for hh in 0..zh { for ww in 0..zw {
            lf[(hh * zw + ww) * zc + c] = lat[((c * zt + fi) * zh + hh) * zw + ww] as f32;
        }}}
        let mut x = Tensor::from_vec(&ctx, &lf, &[1, zh, zw, zc]);
        x = dec.conv_s(&x, "post_quant_conv", 0, 0);
        x = dec.conv_t(&x, "decoder.conv_in");
        x = dec.resnet(&x, "decoder.mid_block.resnets.0");
        x = dec.attn(&x, "decoder.mid_block.attentions.0");
        x = dec.resnet(&x, "decoder.mid_block.resnets.1");
        for i in 0..4 {
            let (cin, cout, ft, up_flag, up3d) = up_cfg[i];
            let x_copy = x.clone();
            for r in 0..3 { x = dec.resnet(&x, &format!("decoder.up_blocks.{i}.resnets.{r}")); }
            if up_flag {
                if up3d { x = dec.time_up(&x, &format!("decoder.up_blocks.{i}.upsampler.time_conv"), first_chunk).await; }
                x = nearest2x(&ctx, &x).await;
                x = dec.conv_s(&x, &format!("decoder.up_blocks.{i}.upsampler.resample.1"), 1, 1);
                let sc = dupup(&ctx, &x_copy, cin, cout, ft, first_chunk).await;
                x = x.add(&sc);
            }
            let _ = (cin, cout);
        }
        x = x.rmsnorm(&dec.gam["decoder.norm_out.gamma"].clone(), EPS).silu();
        x = dec.conv_t(&x, "decoder.conv_out");
        chunk_outs.push(x);
    }

    // verify per chunk (decoder output, [T,32,32,12]) vs golden
    let mut worst = 0.0f64;
    for (fi, x) in chunk_outs.iter().enumerate() {
        let g = jg.get(&format!("chunk{fi}"));
        let gs = g.get("shape").as_usize_vec(); // [1,12,T,32,32]
        let gd = g.get("data").as_f64_vec();
        let (c, t, h, wd) = (gs[1], gs[2], gs[3], gs[4]);
        let mine = x.to_vec().await;
        let mut e = 0.0f64;
        for cc in 0..c { for tt in 0..t { for hh in 0..h { for ww in 0..wd {
            let m = mine[((tt * h + hh) * wd + ww) * c + cc] as f64;
            let r = gd[((cc * t + tt) * h + hh) * wd + ww];
            e = e.max((m - r).abs());
        }}}}
        worst = worst.max(e);
        println!("  chunk{fi} decoder-out Δ={e:.2e}  (T={t})");
    }

    // concat chunks in T, unpatchify (patch_size=2) + clamp -> final
    let mut allt: Vec<(Vec<f32>, usize, usize, usize)> = Vec::new(); // (data,H,W,C) per output frame
    for x in &chunk_outs {
        let d = x.to_vec().await;
        let (t, h, wd, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        for tt in 0..t {
            allt.push((d[(tt * h * wd * c)..((tt + 1) * h * wd * c)].to_vec(), h, wd, c));
        }
    }
    let fg = jg.get("final");
    let fs = fg.get("shape").as_usize_vec(); // [1,3,5,64,64]
    let fd = fg.get("data").as_f64_vec();
    let (fc, ft_out, fhh, fww) = (fs[1], fs[2], fs[3], fs[4]);
    let mut ferr = 0.0f64;
    for (of, (data, h, wd, pc)) in allt.iter().enumerate() {
        let oc = pc / 4;
        for c in 0..oc { for hh in 0..*h { for ww in 0..*wd { for a in 0..2 { for bb in 0..2 {
            let v = data[(hh * wd + ww) * pc + (c * 4 + bb * 2 + a)].clamp(-1.0, 1.0) as f64;
            let (fh, fw) = (hh * 2 + a, ww * 2 + bb);
            let r = fd[(((c * ft_out + of) * fhh + fh) * fww) + fw];
            ferr = ferr.max((v - r).abs());
        }}}}}
    }
    let _ = (fc,);
    println!("\nper-chunk worst Δ={worst:.2e}   FINAL video Δ vs AutoencoderKLWan.decode = {ferr:.3e}  ->  {}",
             if worst < 3e-3 && ferr < 3e-3 { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(worst < 3e-3 && ferr < 3e-3, "Ferric video decode diverged");
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
        fn po(b: &[u8], i: &mut usize) -> Node {
            *i += 1; let mut m = Vec::new();
            loop { ws(b, i); if b[*i] == b'}' { *i += 1; break; } let k = ps(b, i); ws(b, i); *i += 1; let v = pv(b, i); m.push((k, v));
                ws(b, i); if b[*i] == b',' { *i += 1; } else if b[*i] == b'}' { *i += 1; break; } }
            Node::Obj(m)
        }
        fn pa(b: &[u8], i: &mut usize) -> Node {
            *i += 1; let mut a = Vec::new();
            loop { ws(b, i); if b[*i] == b']' { *i += 1; break; } a.push(pv(b, i)); ws(b, i);
                if b[*i] == b',' { *i += 1; } else if b[*i] == b']' { *i += 1; break; } }
            Node::Arr(a)
        }
        fn ps(b: &[u8], i: &mut usize) -> String { *i += 1; let s = *i; while b[*i] != b'"' { *i += 1; } let r = String::from_utf8_lossy(&b[s..*i]).to_string(); *i += 1; r }
        fn pn(b: &[u8], i: &mut usize) -> Node {
            let s = *i; while *i < b.len() && matches!(b[*i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') { *i += 1; }
            Node::Num(std::str::from_utf8(&b[s..*i]).unwrap().parse().unwrap())
        }
    }
}
