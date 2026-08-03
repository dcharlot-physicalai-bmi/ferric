//! FULL Wan2.2 VAE decoder in pure Rust, verified STAGE-BY-STAGE against the real AutoencoderKLWan.
//! For a single-frame latent (T=1) the streaming temporal path is degenerate, so the decoder reduces to
//! a per-frame graph over the verified conv3d. Checks conv_in / mid / up0..up3 / norm_out / conv_out /
//! final against vae_stages_golden.json. usage:
//!   cargo run -p ferric-llama --example cosmos_vae_decode --release -- <wan-vae-dir>
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
    let gp = format!("{}/.cache/ferric/cosmos_ref/vae_stages_golden.json", std::env::var("HOME").unwrap());
    let jg = json_min::parse(&std::fs::read_to_string(&gp).expect("run vae_stages_ref.py first"));
    let ctx = Arc::new(Context::new().await.unwrap());

    let keep = |n: &str| n.starts_with("decoder.") || n.starts_with("post_quant_conv.");
    let w: HashMap<String, STensor> = safetensors_filtered(&std::fs::read(&vae).unwrap(), keep).unwrap();
    println!("loaded {} decoder tensors", w.len());

    // ---- weight accessors (reorder conv [O,C,kt,kh,kw] -> [kt,kh,kw,C,O]) ----
    let cw = |name: &str| -> (Tensor, Tensor) {
        let s = &w[&format!("{name}.weight")];
        let (o, c, kt, kh, kw) = (s.shape[0], s.shape[1], s.shape[2], s.shape[3], s.shape[4]);
        let mut r = vec![0f32; kt * kh * kw * c * o];
        for oo in 0..o { for cc in 0..c { for a in 0..kt { for ky in 0..kh { for kx in 0..kw {
            r[((((a * kh + ky) * kw + kx) * c + cc) * o) + oo] = s.data[(((oo * c + cc) * kt + a) * kh + ky) * kw + kx];
        }}}}}
        (Tensor::from_vec(&ctx, &r, &[kt, kh, kw, c, o]), Tensor::from_vec(&ctx, &w[&format!("{name}.bias")].data, &[o]))
    };
    let cw2d = |name: &str| -> (Tensor, Tensor) { // Conv2d [O,C,kh,kw] -> [1,kh,kw,C,O]
        let s = &w[&format!("{name}.weight")];
        let (o, c, kh, kw) = (s.shape[0], s.shape[1], s.shape[2], s.shape[3]);
        let mut r = vec![0f32; kh * kw * c * o];
        for oo in 0..o { for cc in 0..c { for ky in 0..kh { for kx in 0..kw {
            r[(((ky * kw + kx) * c + cc) * o) + oo] = s.data[((oo * c + cc) * kh + ky) * kw + kx];
        }}}}
        (Tensor::from_vec(&ctx, &r, &[1, kh, kw, c, o]), Tensor::from_vec(&ctx, &w[&format!("{name}.bias")].data, &[o]))
    };
    let lin = |name: &str| -> (Tensor, Tensor) { // Conv2d 1x1 [O,C,1,1] -> matmul_bt weight [O,C]
        let s = &w[&format!("{name}.weight")];
        (Tensor::from_vec(&ctx, &s.data, &[s.shape[0], s.shape[1]]), Tensor::from_vec(&ctx, &w[&format!("{name}.bias")].data, &[s.shape[0]]))
    };
    let gam = |name: &str| -> Tensor { let s = &w[name]; Tensor::from_vec(&ctx, &s.data, &[s.shape[0]]) };
    let has = |name: &str| w.contains_key(&format!("{name}.weight"));

    // causal conv3d: left-pad T by 2*padt, symmetric spatial (padh,padw)
    let conv = |x: &Tensor, wt: &Tensor, b: &Tensor, padt: usize, padh: usize, padw: usize| -> Tensor {
        let (h, wd, c) = (x.shape[1], x.shape[2], x.shape[3]);
        let xp = if padt > 0 { Tensor::from_vec(&ctx, &vec![0f32; 2 * padt * h * wd * c], &[2 * padt, h, wd, c]).cat(x, 0) } else { x.clone() };
        xp.conv3d(wt, b, (1, 1, 1), (padh, padw))
    };
    let mlp_res = |x: &Tensor, p: &str| -> Tensor {  // WanResidualBlock
        let cin = x.shape[3];
        let (c1w, c1b) = cw(&format!("{p}.conv1"));
        let cout = c1w.shape[4];
        let h = if cin != cout { let (sw, sb) = cw(&format!("{p}.conv_shortcut")); conv(x, &sw, &sb, 0, 0, 0) } else { x.clone() };
        let y = x.rmsnorm(&gam(&format!("{p}.norm1.gamma")), EPS).silu();
        let y = conv(&y, &c1w, &c1b, 1, 1, 1);
        let y = y.rmsnorm(&gam(&format!("{p}.norm2.gamma")), EPS).silu();
        let (c2w, c2b) = cw(&format!("{p}.conv2"));
        let y = conv(&y, &c2w, &c2b, 1, 1, 1);
        y.add(&h)
    };
    let attn = |x: &Tensor, p: &str| -> Tensor {  // WanAttentionBlock (T=1, single-head spatial)
        let (hh, ww, c) = (x.shape[1], x.shape[2], x.shape[3]);
        let (qw, qb) = lin(&format!("{p}.to_qkv"));
        let (pw, pb) = lin(&format!("{p}.proj"));
        let xn = x.rmsnorm(&gam(&format!("{p}.norm.gamma")), EPS).reshape(&[hh * ww, c]);
        let qkv = xn.matmul_bt(&qw).add(&qb.reshape(&[1, 3 * c])); // [hw, 3c]
        let q = qkv.narrow(1, 0, c);
        let k = qkv.narrow(1, c, c);
        let v = qkv.narrow(1, 2 * c, c);
        let o = nn::full_attention_kv(&q, &k, &v, 1, 1);            // [hw, c]
        let o = o.matmul_bt(&pw).add(&pb.reshape(&[1, c]));
        o.reshape(&[1, hh, ww, c]).add(x)
    };

    // ---- input: post_quant_conv(latent) ----
    let lat_c = jg.get("lat").as_f64_vec(); // [1,48,1,4,4] channel-first
    let (zc, zt, zh, zw) = (48usize, 1usize, 4usize, 4usize);
    let mut lat = vec![0f32; zt * zh * zw * zc];
    for c in 0..zc { for tt in 0..zt { for hh in 0..zh { for ww in 0..zw {
        lat[((tt * zh + hh) * zw + ww) * zc + c] = lat_c[((c * zt + tt) * zh + hh) * zw + ww] as f32;
    }}}}
    let z = Tensor::from_vec(&ctx, &lat, &[zt, zh, zw, zc]);
    let (pqw, pqb) = cw("post_quant_conv");
    let x = conv(&z, &pqw, &pqb, 0, 0, 0);

    let check = |x: &Tensor, name: &str| -> f64 {
        let st = jg.get(name);
        let shp = st.get("shape").as_usize_vec(); // [1,C,T,H,W]
        let data = st.get("data").as_f64_vec();
        let (c, t, h, wd) = (shp[1], shp[2], shp[3], shp[4]);
        let mine = pollster::block_on(x.to_vec()); // [T,H,W,C]
        let mut e = 0.0f64;
        for cc in 0..c { for tt in 0..t { for hh in 0..h { for ww in 0..wd {
            let m = mine[((tt * h + hh) * wd + ww) * c + cc] as f64;
            let r = data[((cc * t + tt) * h + hh) * wd + ww];
            e = e.max((m - r).abs());
        }}}}
        e
    };

    // ---- conv_in ----
    let (ciw, cib) = cw("decoder.conv_in");
    let mut x = conv(&x, &ciw, &cib, 1, 1, 1);
    println!("  conv_in   Δ={:.2e}", check(&x, "conv_in"));
    // ---- mid_block: resnet, attn, resnet ----
    x = mlp_res(&x, "decoder.mid_block.resnets.0");
    x = attn(&x, "decoder.mid_block.attentions.0");
    x = mlp_res(&x, "decoder.mid_block.resnets.1");
    println!("  mid       Δ={:.2e}", check(&x, "mid"));
    // ---- up_blocks 0..3 (WanResidualUpBlock) ----
    let up_cfg = [(1024usize, 1024usize, 2usize, true), (1024, 1024, 2, true), (1024, 512, 1, true), (512, 256, 0, false)];
    for i in 0..4 {
        let (cin, cout, ft, up_flag) = up_cfg[i];
        let x_copy = x.clone();
        for r in 0..3 { x = mlp_res(&x, &format!("decoder.up_blocks.{i}.resnets.{r}")); }
        if up_flag {
            // upsampler: nearest 2x (host) + Conv2d 3x3 pad1
            x = nearest2x(&ctx, &x).await;
            let (uw, ub) = cw2d(&format!("decoder.up_blocks.{i}.upsampler.resample.1"));
            x = conv(&x, &uw, &ub, 0, 1, 1);
            // avg_shortcut: DupUp3D(cin,cout,ft) on x_copy (first_chunk)
            let sc = dupup(&ctx, &x_copy, cin, cout, ft, 2).await;
            x = x.add(&sc);
        }
        println!("  up{i}       Δ={:.2e}", check(&x, &format!("up{i}")));
        let _ = (cin, cout);
    }
    // ---- head ---- (norm_out golden is pre-silu)
    x = x.rmsnorm(&gam("decoder.norm_out.gamma"), EPS);
    println!("  norm_out  Δ={:.2e}", check(&x, "norm_out"));
    x = x.silu();
    let (cow, cob) = cw("decoder.conv_out");
    x = conv(&x, &cow, &cob, 1, 1, 1);
    println!("  conv_out  Δ={:.2e}", check(&x, "conv_out"));
    // ---- unpatchify (patch_size=2) + clamp ----
    let cvo = x.to_vec().await; // [1,32,32,12]
    let (ph, pw, pc) = (x.shape[1], x.shape[2], x.shape[3]);
    let oc = pc / 4;
    let mut fin = vec![0f32; (ph * 2) * (pw * 2) * oc];
    // unpatchify permute (0,1,4,5,3,6,2): height offset = ps_b, width offset = ps_a,
    // and c_patches = c*4 + ps_a*2 + ps_b  => channel = c*4 + bb*2 + a
    for c in 0..oc { for hh in 0..ph { for ww in 0..pw { for a in 0..2 { for bb in 0..2 {
        let v = cvo[(hh * pw + ww) * pc + (c * 4 + bb * 2 + a)];
        fin[(((hh * 2 + a) * (pw * 2) + (ww * 2 + bb)) * oc) + c] = v.clamp(-1.0, 1.0);
    }}}}}
    let fg = jg.get("final");
    let fs = fg.get("shape").as_usize_vec(); // [1,3,1,64,64]
    let fd = fg.get("data").as_f64_vec();
    let (fc, fhh, fww) = (fs[1], fs[3], fs[4]);
    let mut ferr = 0.0f64;
    for c in 0..fc { for hh in 0..fhh { for ww in 0..fww {
        let m = fin[(hh * fww + ww) * oc + c] as f64;
        let r = fd[(c * fhh + hh) * fww + ww];
        ferr = ferr.max((m - r).abs());
    }}}
    println!("\nFINAL decode Δ vs AutoencoderKLWan.decode = {ferr:.3e}  ->  {}", if ferr < 2e-3 { "MATCH ✓" } else { "MISMATCH ✗" });
}

// nearest-exact 2x spatial upsample on [1,H,W,C] (T=1) via host round-trip
async fn nearest2x(ctx: &Arc<Context>, x: &Tensor) -> Tensor {
    let (h, wd, c) = (x.shape[1], x.shape[2], x.shape[3]);
    let d = x.to_vec().await;
    let mut o = vec![0f32; (h * 2) * (wd * 2) * c];
    for hh in 0..h { for ww in 0..wd { for cc in 0..c {
        let v = d[(hh * wd + ww) * c + cc];
        for a in 0..2 { for b in 0..2 { o[(((hh * 2 + a) * (wd * 2) + (ww * 2 + b)) * c) + cc] = v; } }
    }}}
    Tensor::from_vec(ctx, &o, &[1, h * 2, wd * 2, c])
}

// DupUp3D avg_shortcut (first_chunk, T=1): [1,H,W,in] -> [1,2H,2W,out]
async fn dupup(ctx: &Arc<Context>, x: &Tensor, cin: usize, cout: usize, ft: usize, fs: usize) -> Tensor {
    let (h, wd) = (x.shape[1], x.shape[2]);
    let d = x.to_vec().await; // [1,H,W,cin]
    let factor = ft.max(1) * fs * fs;
    let repeats = cout * factor / cin;
    let ft_i = ft.max(1) - 1; // first_chunk takes last temporal slice
    let mut o = vec![0f32; (h * fs) * (wd * fs) * cout];
    for oc in 0..cout { for fi in 0..fs { for fj in 0..fs {
        let flat = ((oc * ft.max(1) + ft_i) * fs + fi) * fs + fj;
        let in_ch = flat / repeats;
        for hh in 0..h { for ww in 0..wd {
            o[(((hh * fs + fi) * (wd * fs) + (ww * fs + fj)) * cout) + oc] = d[(hh * wd + ww) * cin + in_ch];
        }}
    }}}
    Tensor::from_vec(ctx, &o, &[1, h * fs, wd * fs, cout])
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
            pub fn get(&self, k: &str) -> Node {
                if let Node::Obj(m) = self { for (kk, v) in m { if kk == k { return v.clone(); } } }
                Node::Null
            }
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
            loop { ws(b, i); if b[*i] == b'}' { *i += 1; break; }
                let k = ps(b, i); ws(b, i); *i += 1; let v = pv(b, i); m.push((k, v));
                ws(b, i); if b[*i] == b',' { *i += 1; } else if b[*i] == b'}' { *i += 1; break; } }
            Node::Obj(m)
        }
        fn pa(b: &[u8], i: &mut usize) -> Node {
            *i += 1; let mut a = Vec::new();
            loop { ws(b, i); if b[*i] == b']' { *i += 1; break; }
                a.push(pv(b, i)); ws(b, i);
                if b[*i] == b',' { *i += 1; } else if b[*i] == b']' { *i += 1; break; } }
            Node::Arr(a)
        }
        fn ps(b: &[u8], i: &mut usize) -> String {
            *i += 1; let s = *i; while b[*i] != b'"' { *i += 1; }
            let r = String::from_utf8_lossy(&b[s..*i]).to_string(); *i += 1; r
        }
        fn pn(b: &[u8], i: &mut usize) -> Node {
            let s = *i;
            while *i < b.len() && matches!(b[*i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') { *i += 1; }
            Node::Num(std::str::from_utf8(&b[s..*i]).unwrap().parse().unwrap())
        }
    }
}
