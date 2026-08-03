//! Wan2.2 VAE ENCODER (single-frame image → latent) in pure Rust, verified stage-by-stage vs the real
//! AutoencoderKLWan.encode. Completes the VAE round-trip (the conditioning input path for Cosmos 3 Edge
//! action modes). Symmetric to the decoder: patchify → conv_in → 4 WanResidualDownBlock (resnets +
//! downsample + AvgDown3D) → mid → norm/conv_out → quant_conv → mode(). usage:
//!   cargo run -p ferric-llama --example cosmos_vae_encode --release -- <wan-vae-safetensors>
use ferric_core::Context;
use ferric_load::{safetensors_filtered, STensor};
use ferric_tensor::{nn, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

const EPS: f32 = 1e-12;

fn main() { pollster::block_on(run()); }
async fn run() {
    let vae = std::env::args().nth(1).unwrap_or_else(|| {
        let d = format!("{}/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots", std::env::var("HOME").unwrap());
        let snap = std::fs::read_dir(&d).unwrap().next().unwrap().unwrap().path();
        format!("{}/vae/diffusion_pytorch_model.safetensors", snap.display())
    });
    let gp = format!("{}/.cache/ferric/cosmos_ref/encode_golden.json", std::env::var("HOME").unwrap());
    let jg = json_min::parse(&std::fs::read_to_string(&gp).unwrap());
    let ctx = Arc::new(Context::new().await.unwrap());
    let w: HashMap<String, STensor> = safetensors_filtered(&std::fs::read(&vae).unwrap(),
        |n: &str| n.starts_with("encoder.") || n.starts_with("quant_conv.")).unwrap();
    println!("loaded {} encoder tensors", w.len());

    let cw = |name: &str| -> (Tensor, Tensor) {
        let s = &w[&format!("{name}.weight")];
        let (o, c, kt, kh, kw) = (s.shape[0], s.shape[1], s.shape[2], s.shape[3], s.shape[4]);
        let mut r = vec![0f32; kt * kh * kw * c * o];
        for oo in 0..o { for cc in 0..c { for a in 0..kt { for ky in 0..kh { for kx in 0..kw {
            r[((((a * kh + ky) * kw + kx) * c + cc) * o) + oo] = s.data[(((oo * c + cc) * kt + a) * kh + ky) * kw + kx];
        }}}}}
        (Tensor::from_vec(&ctx, &r, &[kt, kh, kw, c, o]), Tensor::from_vec(&ctx, &w[&format!("{name}.bias")].data, &[o]))
    };
    let cw2d = |name: &str| -> (Tensor, Tensor) {
        let s = &w[&format!("{name}.weight")];
        let (o, c, kh, kw) = (s.shape[0], s.shape[1], s.shape[2], s.shape[3]);
        let mut r = vec![0f32; kh * kw * c * o];
        for oo in 0..o { for cc in 0..c { for ky in 0..kh { for kx in 0..kw {
            r[(((ky * kw + kx) * c + cc) * o) + oo] = s.data[((oo * c + cc) * kh + ky) * kw + kx];
        }}}}
        (Tensor::from_vec(&ctx, &r, &[1, kh, kw, c, o]), Tensor::from_vec(&ctx, &w[&format!("{name}.bias")].data, &[o]))
    };
    let lin = |name: &str| -> (Tensor, Tensor) {
        let s = &w[&format!("{name}.weight")];
        (Tensor::from_vec(&ctx, &s.data, &[s.shape[0], s.shape[1]]), Tensor::from_vec(&ctx, &w[&format!("{name}.bias")].data, &[s.shape[0]]))
    };
    let gam = |name: &str| -> Tensor { let s = &w[name]; Tensor::from_vec(&ctx, &s.data, &[s.shape[0]]) };
    let conv = |x: &Tensor, wt: &Tensor, b: &Tensor, padt: usize, padh: usize, padw: usize| -> Tensor {
        let (h, wd, c) = (x.shape[1], x.shape[2], x.shape[3]);
        let xp = if padt > 0 { Tensor::from_vec(&ctx, &vec![0f32; 2 * padt * h * wd * c], &[2 * padt, h, wd, c]).cat(x, 0) } else { x.clone() };
        xp.conv3d(wt, b, (1, 1, 1), (padh, padw))
    };
    let resnet = |x: &Tensor, p: &str| -> Tensor {
        let cin = x.shape[3];
        let (c1w, c1b) = cw(&format!("{p}.conv1"));
        let cout = c1w.shape[4];
        let h = if cin != cout { let (sw, sb) = cw(&format!("{p}.conv_shortcut")); conv(x, &sw, &sb, 0, 0, 0) } else { x.clone() };
        let y = x.rmsnorm(&gam(&format!("{p}.norm1.gamma")), EPS).silu();
        let y = conv(&y, &c1w, &c1b, 1, 1, 1);
        let y = y.rmsnorm(&gam(&format!("{p}.norm2.gamma")), EPS).silu();
        let (c2w, c2b) = cw(&format!("{p}.conv2"));
        conv(&y, &c2w, &c2b, 1, 1, 1).add(&h)
    };
    let attn = |x: &Tensor, p: &str| -> Tensor {
        let (hh, ww, c) = (x.shape[1], x.shape[2], x.shape[3]);
        let (qw, qb) = lin(&format!("{p}.to_qkv"));
        let (pw, pb) = lin(&format!("{p}.proj"));
        let xn = x.rmsnorm(&gam(&format!("{p}.norm.gamma")), EPS).reshape(&[hh * ww, c]);
        let qkv = xn.matmul_bt(&qw).add(&qb.reshape(&[1, 3 * c]));
        let (q, k, v) = (qkv.narrow(1, 0, c), qkv.narrow(1, c, c), qkv.narrow(1, 2 * c, c));
        let o = nn::full_attention_kv(&q, &k, &v, 1, 1);
        o.matmul_bt(&pw).add(&pb.reshape(&[1, c])).reshape(&[1, hh, ww, c]).add(x)
    };
    let check = |x: &Tensor, name: &str| -> f64 {
        let st = jg.get(name);
        let shp = st.get("shape").as_usize_vec();
        let data = st.get("data").as_f64_vec();
        let (c, t, h, wd) = (shp[1], shp[2], shp[3], shp[4]);
        let mine = pollster::block_on(x.to_vec());
        let mut e = 0.0f64;
        for cc in 0..c { for tt in 0..t { for hh in 0..h { for ww in 0..wd {
            e = e.max((mine[((tt * h + hh) * wd + ww) * c + cc] as f64 - data[((cc * t + tt) * h + hh) * wd + ww]).abs());
        }}}}
        e
    };

    // ---- patchify(img, 2): [1,3,1,64,64] -> [1,12,1,32,32]  (channel = c*4 + psw*2 + psh) ----
    let img = jg.get("img").as_f64_vec(); // [1,3,1,64,64]
    let (ic, ih, iw) = (3usize, 64usize, 64usize);
    let (ph, pw) = (ih / 2, iw / 2);
    let mut patched = vec![0f32; ph * pw * (ic * 4)];
    for c in 0..ic { for hi in 0..ph { for wi in 0..pw { for psh in 0..2 { for psw in 0..2 {
        let v = img[((c * 1) * ih + (hi * 2 + psh)) * iw + (wi * 2 + psw)] as f32;
        patched[(hi * pw + wi) * (ic * 4) + (c * 4 + psw * 2 + psh)] = v;
    }}}}}
    let mut x = Tensor::from_vec(&ctx, &patched, &[1, ph, pw, ic * 4]);

    // ---- conv_in ----
    let (ciw, cib) = cw("encoder.conv_in");
    x = conv(&x, &ciw, &cib, 1, 1, 1);
    println!("  conv_in  Δ={:.2e}", check(&x, "conv_in"));

    // ---- down blocks (WanResidualDownBlock) ----
    let down_cfg = [(160usize, 160usize, 1usize, true), (160, 320, 2, true), (320, 640, 2, true), (640, 640, 1, false)];
    for i in 0..4 {
        let (cin, cout, ft, down_flag) = down_cfg[i];
        let x_copy = x.clone();
        x = resnet(&x, &format!("encoder.down_blocks.{i}.resnets.0"));
        x = resnet(&x, &format!("encoder.down_blocks.{i}.resnets.1"));
        if down_flag {
            // WanResample downsample (first chunk = spatial only): ZeroPad2d((0,1,0,1)) + Conv2d stride2
            x = pad_br(&ctx, &x).await;
            let (dw, db) = cw2d(&format!("encoder.down_blocks.{i}.downsampler.resample.1"));
            x = x.conv3d(&dw, &db, (1, 2, 2), (0, 0));
        }
        // avg_shortcut = AvgDown3D(cin,cout,ft, fs = 2 if down_flag else 1)
        let fs = if down_flag { 2 } else { 1 };
        let sc = avgdown(&ctx, &x_copy, cin, cout, ft, fs).await;
        x = x.add(&sc);
        println!("  down{i}   Δ={:.2e}", check(&x, &format!("down{i}")));
        let _ = (cin, cout);
    }
    // ---- mid ----
    x = resnet(&x, "encoder.mid_block.resnets.0");
    x = attn(&x, "encoder.mid_block.attentions.0");
    x = resnet(&x, "encoder.mid_block.resnets.1");
    println!("  mid      Δ={:.2e}", check(&x, "mid"));
    // ---- head + quant_conv + mode ----
    x = x.rmsnorm(&gam("encoder.norm_out.gamma"), EPS).silu();
    let (cow, cob) = cw("encoder.conv_out");
    x = conv(&x, &cow, &cob, 1, 1, 1);
    println!("  conv_out Δ={:.2e}", check(&x, "conv_out"));
    let (qw, qb) = cw("quant_conv");
    let q = conv(&x, &qw, &qb, 0, 0, 0); // [1(T),4,4,96]
    // posterior.mode() = mean = first z_dim=48 channels
    let z = 48usize;
    let lat = q.narrow(3, 0, z);
    let lerr = check(&lat, "latent");
    println!("  latent   Δ={lerr:.2e}");
    println!("\nENCODE Δ vs AutoencoderKLWan.encode(mode) = {lerr:.3e}  ->  {}", if lerr < 2e-3 { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(lerr < 2e-3, "Ferric encode diverged");
}

// ZeroPad2d((0,1,0,1)): pad H bottom + W right by 1 (T,C unchanged), [T,H,W,C]->[T,H+1,W+1,C]
async fn pad_br(ctx: &Arc<Context>, x: &Tensor) -> Tensor {
    let (t, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let d = x.to_vec().await;
    let mut o = vec![0f32; t * (h + 1) * (w + 1) * c];
    for ti in 0..t { for hi in 0..h { for wi in 0..w { for ci in 0..c {
        o[(((ti * (h + 1) + hi) * (w + 1) + wi) * c) + ci] = d[((ti * h + hi) * w + wi) * c + ci];
    }}}}
    Tensor::from_vec(ctx, &o, &[t, h + 1, w + 1, c])
}

// AvgDown3D (T=1) : [1,H,W,cin] -> [1,H/fs,W/fs,cout]. temporal pad to multiple of ft, fold + mean.
async fn avgdown(ctx: &Arc<Context>, x: &Tensor, cin: usize, cout: usize, ft: usize, fs: usize) -> Tensor {
    let (t, h, w) = (x.shape[0], x.shape[1], x.shape[2]);
    let d = x.to_vec().await;
    let pad_t = (ft - t % ft) % ft;
    let tp = t + pad_t; // padded T (zeros prepended)
    let get = |ti: i64, hi: usize, wi: usize, c: usize| -> f32 {
        let real = ti - pad_t as i64;
        if real < 0 { 0.0 } else { d[((real as usize * h + hi) * w + wi) * cin + c] }
    };
    let factor = ft * fs * fs;
    let gs = cin * factor / cout;
    let (ho, wo, to) = (h / fs, w / fs, tp / ft);
    let mut o = vec![0f32; to * ho * wo * cout];
    for ot in 0..to { for oh in 0..ho { for ow in 0..wo { for oc in 0..cout {
        let mut acc = 0.0f32;
        for g in 0..gs {
            let flat = oc * gs + g; // flat = ((c*ft+fti)*fs+fi)*fs+fj
            let fj = flat % fs; let r = flat / fs; let fi = r % fs; let r = r / fs; let fti = r % ft; let c = r / ft;
            acc += get((ot * ft + fti) as i64, oh * fs + fi, ow * fs + fj, c);
        }
        o[((ot * ho + oh) * wo + ow) * cout + oc] = acc / gs as f32;
    }}}}
    Tensor::from_vec(ctx, &o, &[to, ho, wo, cout])
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
