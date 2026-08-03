//! V-JEPA 2 (ViT-L) video encoder in pure Rust, verified stage-by-stage vs the real transformers
//! VJEPA2Model.encoder. conv3d tubelet patch-embed (2×16×16) → 24 pre-norm LayerNorm blocks with 3-axis
//! interleaved RoPE (frame/height/width) + full attention + gelu MLP → final LayerNorm. Meta's JEPA
//! world-model backbone — the ingest target adjacent to EFA's latent-predictive line. usage:
//!   cargo run -p ferric-llama --example vjepa_encode --release -- <vjepa2-vitl-safetensors>
use ferric_core::Context;
use ferric_load::{safetensors_filtered, STensor};
use ferric_tensor::{nn, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

const EPS: f32 = 1e-6;

fn main() { pollster::block_on(run()); }
async fn run() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        let d = format!("{}/.cache/huggingface/hub/models--facebook--vjepa2-vitl-fpc64-256/snapshots", std::env::var("HOME").unwrap());
        let snap = std::fs::read_dir(&d).unwrap().next().unwrap().unwrap().path();
        std::fs::read_dir(snap).unwrap().filter_map(|e| { let p = e.unwrap().path(); if p.extension().and_then(|x| x.to_str()) == Some("safetensors") { Some(p.display().to_string()) } else { None } }).next().unwrap()
    });
    let gp = format!("{}/.cache/ferric/cosmos_ref/vjepa_golden.json", std::env::var("HOME").unwrap());
    let jg = json_min::parse(&std::fs::read_to_string(&gp).unwrap());
    let ctx = Arc::new(Context::new().await.unwrap());
    let w: HashMap<String, STensor> = safetensors_filtered(&std::fs::read(&path).unwrap(), |n: &str| n.starts_with("encoder.")).unwrap();
    let g = |n: &str| { let s = &w[n]; Tensor::from_vec(&ctx, &s.data, &s.shape) };
    let (nh, hd, d) = (16usize, 64usize, 1024usize);

    // ---- patch embed: conv3d tubelet (2,16,16) stride (2,16,16), weight [1024,3,2,16,16] -> [2,16,16,3,1024]
    let vid = jg.get("vid").as_f64_vec(); // [1,4,3,256,256]
    let vs = jg.get("vid_shape").as_usize_vec();
    let (tt, ic, ih, iw) = (vs[1], vs[2], vs[3], vs[4]);
    let mut thwc = vec![0f32; tt * ih * iw * ic];
    for t in 0..tt { for c in 0..ic { for h in 0..ih { for ww in 0..iw {
        thwc[((t * ih + h) * iw + ww) * ic + c] = vid[((t * ic + c) * ih + h) * iw + ww] as f32;
    }}}}
    let pw = &w["encoder.embeddings.patch_embeddings.proj.weight"]; // [1024,3,2,16,16]
    let (o, c, kt, kh, kw) = (pw.shape[0], pw.shape[1], pw.shape[2], pw.shape[3], pw.shape[4]);
    let mut pr = vec![0f32; kt * kh * kw * c * o];
    for oo in 0..o { for cc in 0..c { for a in 0..kt { for ky in 0..kh { for kx in 0..kw {
        pr[((((a * kh + ky) * kw + kx) * c + cc) * o) + oo] = pw.data[(((oo * c + cc) * kt + a) * kh + ky) * kw + kx];
    }}}}}
    let pwt = Tensor::from_vec(&ctx, &pr, &[kt, kh, kw, c, o]);
    let pb = g("encoder.embeddings.patch_embeddings.proj.bias");
    let vt = Tensor::from_vec(&ctx, &thwc, &[tt, ih, iw, ic]);
    let patched = vt.conv3d(&pwt, &pb, (kt, kh, kw), (0, 0)); // [T/2, 16, 16, 1024]
    let n = patched.shape[0] * patched.shape[1] * patched.shape[2];
    let mut x = patched.reshape(&[n, d]);

    let check = |x: &Tensor, name: &str| -> f64 {
        let st = jg.get(name); let sh = st.get("shape").as_usize_vec(); let data = st.get("data").as_f64_vec();
        let mine = pollster::block_on(x.to_vec());
        let mut e = 0.0f64; for k in 0..sh[1] * sh[2] { e = e.max((mine[k] as f64 - data[k]).abs()); } e
    };
    println!("  patch_embed Δ={:.2e}", check(&x, "patch_embed"));

    // ---- 3-axis interleaved RoPE tables [N, head_dim] (frame/height/width, grid_size=16) ----
    let grid = 16usize;
    let slice = 2 * ((hd / 3) / 2); // 20
    let half = slice / 2; // 10
    let omega: Vec<f64> = (0..half).map(|j| 1.0 / 10000f64.powf(j as f64 / half as f64)).collect();
    let mut cos = vec![0f32; n * hd];
    let mut sin = vec![0f32; n * hd];
    for i in 0..n {
        let frame = i / (grid * grid);
        let height = (i % (grid * grid)) / grid;
        let width = i % grid;
        let axes = [(0usize, frame), (slice, height), (2 * slice, width)];
        for (off, pos) in axes {
            for j in 0..half {
                let f = pos as f64 * omega[j];
                let (cv, sv) = (f.cos() as f32, f.sin() as f32);
                cos[i * hd + off + j] = cv; cos[i * hd + off + half + j] = cv; // tile
                sin[i * hd + off + j] = sv; sin[i * hd + off + half + j] = sv;
            }
        }
        for k in (3 * slice)..hd { cos[i * hd + k] = 1.0; sin[i * hd + k] = 0.0; } // unrotated remainder
    }
    let cos_t = Tensor::from_vec(&ctx, &cos, &[n, hd]);
    let sin_t = Tensor::from_vec(&ctx, &sin, &[n, hd]);

    // ---- 24 pre-norm blocks ----
    for il in 0..24 {
        let b = |s: &str| g(&format!("encoder.layer.{il}.{s}"));
        let h = x.layernorm(&b("norm1.weight"), &b("norm1.bias"), EPS);
        let q = h.matmul_bt(&b("attention.query.weight")).add(&b("attention.query.bias").reshape(&[1, d])).apply_rope_interleaved(&cos_t, &sin_t, nh, hd);
        let k = h.matmul_bt(&b("attention.key.weight")).add(&b("attention.key.bias").reshape(&[1, d])).apply_rope_interleaved(&cos_t, &sin_t, nh, hd);
        let v = h.matmul_bt(&b("attention.value.weight")).add(&b("attention.value.bias").reshape(&[1, d]));
        let o = nn::full_attention_kv(&q, &k, &v, nh, nh);
        let o = o.matmul_bt(&b("attention.proj.weight")).add(&b("attention.proj.bias").reshape(&[1, d]));
        x = x.add(&o);
        let h2 = x.layernorm(&b("norm2.weight"), &b("norm2.bias"), EPS);
        let m = h2.matmul_bt(&b("mlp.fc1.weight")).add(&b("mlp.fc1.bias").reshape(&[1, 4096])).gelu()
            .matmul_bt(&b("mlp.fc2.weight")).add(&b("mlp.fc2.bias").reshape(&[1, d]));
        x = x.add(&m);
        if il == 0 { println!("  layer0      Δ={:.2e}", check(&x, "layer0")); }
    }
    x = x.layernorm(&g("encoder.layernorm.weight"), &g("encoder.layernorm.bias"), EPS);
    let ferr = check(&x, "final");
    println!("  final       Δ={ferr:.2e}");
    println!("\nV-JEPA 2 ViT-L encoder Δ vs real VJEPA2Model = {ferr:.3e}  ->  {}", if ferr < 2e-3 { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(ferr < 2e-3, "Ferric V-JEPA 2 encoder diverged");
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
