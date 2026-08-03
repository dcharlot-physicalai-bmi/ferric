//! π0 VLM prefix — the SigLIP vision tower + multi_modal_projector — in pure Rust, verified vs the real
//! PaliGemma get_image_features. image → conv patch-embed(14×14) + learned pos → 27 SigLIP layers
//! (LayerNorm, full attn 16h hd72, gelu_tanh MLP) → post_layernorm → projector(1152→2048) = image tokens
//! that condition the flow expert. usage: cargo run -p ferric-llama --example pi0_vision --release
use ferric_core::Context;
use ferric_load::{safetensors_filtered, STensor};
use ferric_tensor::{nn, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

const EPS: f32 = 1e-6;

fn main() { pollster::block_on(run()); }
async fn run() {
    let home = std::env::var("HOME").unwrap();
    let wp = format!("{home}/.cache/ferric/pi05/pi0_vision.safetensors");
    let jg = json_min::parse(&std::fs::read_to_string(format!("{home}/.cache/ferric/cosmos_ref/pi0_vision_golden.json")).unwrap());
    let ctx = Arc::new(Context::new().await.unwrap());
    let w: HashMap<String, STensor> = safetensors_filtered(&std::fs::read(&wp).unwrap(), |_| true).unwrap();
    let g = |n: &str| { let s = &w[n]; Tensor::from_vec(&ctx, &s.data, &s.shape) };
    let vm = "vision_tower.vision_model";
    let (nh, hd, hidden) = (16usize, 72usize, 1152usize);
    let (ps, imsz) = (14usize, 224usize);
    let grid = imsz / ps; // 16
    let np = grid * grid;  // 256

    // image [1,3,224,224] -> [224,224,3] (T=1) ; conv patch embed via conv3d (kt=1)
    let imgv = jg.get("img").as_f64_vec();
    let mut thwc = vec![0f32; imsz * imsz * 3];
    for c in 0..3 { for h in 0..imsz { for ww in 0..imsz { thwc[(h * imsz + ww) * 3 + c] = imgv[(c * imsz + h) * imsz + ww] as f32; }}}
    // patch weight [1152,3,14,14] -> [1,14,14,3,1152]
    let pw = &w[&format!("{vm}.embeddings.patch_embedding.weight")];
    let (o, c, kh, kw) = (pw.shape[0], pw.shape[1], pw.shape[2], pw.shape[3]);
    let mut pr = vec![0f32; kh * kw * c * o];
    for oo in 0..o { for cc in 0..c { for ky in 0..kh { for kx in 0..kw {
        pr[(((ky * kw + kx) * c + cc) * o) + oo] = pw.data[((oo * c + cc) * kh + ky) * kw + kx];
    }}}}
    let pwt = Tensor::from_vec(&ctx, &pr, &[1, kh, kw, c, o]);
    let pb = g(&format!("{vm}.embeddings.patch_embedding.bias"));
    let vt = Tensor::from_vec(&ctx, &thwc, &[1, imsz, imsz, 3]);
    let patched = vt.conv3d(&pwt, &pb, (1, ps, ps), (0, 0)); // [1,16,16,1152]
    let mut x = patched.reshape(&[np, hidden]).add(&g(&format!("{vm}.embeddings.position_embedding.weight")));

    let check = |x: &Tensor, key: &str| -> f64 {
        let d = jg.get(key).as_f64_vec();
        let m = pollster::block_on(x.to_vec());
        m.iter().zip(&d).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max)
    };

    // 27 SigLIP layers
    for l in 0..27 {
        let b = |s: &str| g(&format!("{vm}.encoder.layers.{l}.{s}"));
        let bb = |s: &str, n: usize| b(s).reshape(&[1, n]);
        let h = x.layernorm(&b("layer_norm1.weight"), &b("layer_norm1.bias"), EPS);
        let q = h.matmul_bt(&b("self_attn.q_proj.weight")).add(&bb("self_attn.q_proj.bias", hidden));
        let k = h.matmul_bt(&b("self_attn.k_proj.weight")).add(&bb("self_attn.k_proj.bias", hidden));
        let v = h.matmul_bt(&b("self_attn.v_proj.weight")).add(&bb("self_attn.v_proj.bias", hidden));
        let o = nn::full_attention_kv(&q, &k, &v, nh, nh).matmul_bt(&b("self_attn.out_proj.weight")).add(&bb("self_attn.out_proj.bias", hidden));
        x = x.add(&o);
        let h2 = x.layernorm(&b("layer_norm2.weight"), &b("layer_norm2.bias"), EPS);
        let m = h2.matmul_bt(&b("mlp.fc1.weight")).add(&bb("mlp.fc1.bias", 4304)).gelu_tanh()
            .matmul_bt(&b("mlp.fc2.weight")).add(&bb("mlp.fc2.bias", hidden));
        x = x.add(&m);
    }
    x = x.layernorm(&g(&format!("{vm}.post_layernorm.weight")), &g(&format!("{vm}.post_layernorm.bias")), EPS);
    // multi_modal_projector 1152 -> 2048
    let feats = x.matmul_bt(&g("multi_modal_projector.linear.weight")).add(&g("multi_modal_projector.linear.bias").reshape(&[1, 2048]));
    let e = check(&feats, "feats");
    println!("π0 SigLIP vision + projector vs real get_image_features: maxΔ={e:.3e}  ->  {}", if e < 2e-3 { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(e < 2e-3, "π0 vision diverged");
}

mod json_min {
    pub struct Val(nd::Node);
    pub fn parse(s: &str) -> Val { Val(nd::parse(s)) }
    impl Val {
        pub fn get(&self, k: &str) -> Val { Val(self.0.get(k)) }
        pub fn as_f64_vec(&self) -> Vec<f64> { self.0.as_vec().iter().map(|n| n.as_f64()).collect() }
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
