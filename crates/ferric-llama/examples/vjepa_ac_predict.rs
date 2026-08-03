//! V-JEPA 2-AC action-conditioned predictor in pure Rust, verified vs Meta's real
//! VisionTransformerPredictorAC. The robot-control world-model head: given T frames of encoder latents +
//! per-frame actions + proprioceptive states, predicts next-frame latents (frame-causally). Interleaves
//! [action, state, patch×H·W] per frame → block-causal frame attention → split RoPE (action/state tokens
//! rotated by frame only; patches 3-axis) → drop cond tokens → proj. usage:
//!   cargo run -p ferric-llama --example vjepa_ac_predict --release
use ferric_core::Context;
use ferric_load::{safetensors_filtered, STensor};
use ferric_tensor::{nn, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

const EPS: f32 = 1e-6;

fn main() { pollster::block_on(run()); }
async fn run() {
    let home = std::env::var("HOME").unwrap();
    let wpath = format!("{home}/.cache/ferric/vjepa2ac/ac_predictor.safetensors");
    let gp = format!("{home}/.cache/ferric/cosmos_ref/vjepa_ac_golden.json");
    let jg = json_min::parse(&std::fs::read_to_string(&gp).unwrap());
    let ctx = Arc::new(Context::new().await.unwrap());
    let w: HashMap<String, STensor> = safetensors_filtered(&std::fs::read(&wpath).unwrap(), |_| true).unwrap();
    let g = |n: &str| { let s = &w[n]; Tensor::from_vec(&ctx, &s.data, &s.shape) };
    let lin = |x: &Tensor, n: &str, out: usize| x.matmul_bt(&g(&format!("{n}.weight"))).add(&g(&format!("{n}.bias")).reshape(&[1, out]));

    let (t, hw, ed) = (jg.get("T").as_usize(), jg.get("HW").as_usize(), jg.get("D").as_usize());
    let (pd, nh, hd, grid) = (1024usize, 16usize, 64usize, 16usize);
    let nt = 2 + hw; // tokens per frame block (action, state, patches)
    let n = t * nt;

    let f2t = |k: &str, d: &[usize]| Tensor::from_vec(&ctx, &jg.get(k).as_f64_vec().iter().map(|v| *v as f32).collect::<Vec<_>>(), d);
    let xin = f2t("x", &[t * hw, ed]);           // encoder latents
    let actions = f2t("actions", &[t, 7]);
    let states = f2t("states", &[t, 7]);

    // predictor_embed + action/state encoders
    let xe = lin(&xin, "predictor_embed", pd);   // [T*HW, 1024]
    let ae = lin(&actions, "action_encoder", pd); // [T, 1024]
    let se = lin(&states, "state_encoder", pd);   // [T, 1024]
    // interleave per frame: [a_t, s_t, patches_t(HW)] -> [N, 1024]  (host assemble)
    let (xed, aed, sed) = (xe.to_vec().await, ae.to_vec().await, se.to_vec().await);
    let mut seq = vec![0f32; n * pd];
    for ti in 0..t {
        for c in 0..pd { seq[(ti * nt) * pd + c] = aed[ti * pd + c]; }            // action token
        for c in 0..pd { seq[(ti * nt + 1) * pd + c] = sed[ti * pd + c]; }        // state token
        for p in 0..hw { for c in 0..pd { seq[(ti * nt + 2 + p) * pd + c] = xed[(ti * hw + p) * pd + c]; } }
    }
    let mut x = Tensor::from_vec(&ctx, &seq, &[n, pd]);

    // block-causal additive mask [N,N]: frame fj <= fi allowed
    let mut mask = vec![0f32; n * n];
    for i in 0..n { for j in 0..n { if (j / nt) > (i / nt) { mask[i * n + j] = -1e30; } } }
    let mask_t = Tensor::from_vec(&ctx, &mask, &[n, n]);

    // split-RoPE cos/sin [N, hd]: action/state -> frame-only (d_dim by frame); patch -> 3-axis
    let slice = 2 * ((hd / 3) / 2); // 20
    let half = slice / 2; // 10
    let omega: Vec<f64> = (0..half).map(|j| 1.0 / 10000f64.powf(j as f64 / half as f64)).collect();
    let mut cos = vec![1f32; n * hd]; // default identity (cos=1, sin=0)
    let mut sin = vec![0f32; n * hd];
    let put = |cos: &mut [f32], sin: &mut [f32], i: usize, off: usize, pos: usize| {
        for j in 0..half {
            let f = pos as f64 * omega[j];
            let (cv, sv) = (f.cos() as f32, f.sin() as f32);
            cos[i * hd + off + j] = cv; cos[i * hd + off + half + j] = cv;
            sin[i * hd + off + j] = sv; sin[i * hd + off + half + j] = sv;
        }
    };
    for i in 0..n {
        let (frame, l) = (i / nt, i % nt);
        if l < 2 {
            put(&mut cos, &mut sin, i, 0, frame); // action/state: d_dim (frame) only
        } else {
            let p = l - 2;
            put(&mut cos, &mut sin, i, 0, frame);
            put(&mut cos, &mut sin, i, slice, p / grid);
            put(&mut cos, &mut sin, i, 2 * slice, p % grid);
        }
    }
    let (cos_t, sin_t) = (Tensor::from_vec(&ctx, &cos, &[n, hd]), Tensor::from_vec(&ctx, &sin, &[n, hd]));

    // 24 blocks
    for il in 0..24 {
        let b = |s: &str| g(&format!("predictor_blocks.{il}.{s}"));
        let h = x.layernorm(&b("norm1.weight"), &b("norm1.bias"), EPS);
        let qkv = h.matmul_bt(&b("attn.qkv.weight")).add(&b("attn.qkv.bias").reshape(&[1, 3 * pd]));
        let q = qkv.narrow(1, 0, pd).apply_rope_interleaved(&cos_t, &sin_t, nh, hd);
        let k = qkv.narrow(1, pd, pd).apply_rope_interleaved(&cos_t, &sin_t, nh, hd);
        let v = qkv.narrow(1, 2 * pd, pd);
        let o = nn::masked_attention_kv(&q, &k, &v, &mask_t, nh, nh).matmul_bt(&b("attn.proj.weight")).add(&b("attn.proj.bias").reshape(&[1, pd]));
        x = x.add(&o);
        let h2 = x.layernorm(&b("norm2.weight"), &b("norm2.bias"), EPS);
        let m = h2.matmul_bt(&b("mlp.fc1.weight")).add(&b("mlp.fc1.bias").reshape(&[1, 4096])).gelu()
            .matmul_bt(&b("mlp.fc2.weight")).add(&b("mlp.fc2.bias").reshape(&[1, pd]));
        x = x.add(&m);
    }
    // drop cond tokens (first 2 per frame), norm, proj
    let xd = x.to_vec().await;
    let mut frames = vec![0f32; t * hw * pd];
    for ti in 0..t { for p in 0..hw { for c in 0..pd { frames[(ti * hw + p) * pd + c] = xd[(ti * nt + 2 + p) * pd + c]; } } }
    let fx = Tensor::from_vec(&ctx, &frames, &[t * hw, pd]);
    let fx = fx.layernorm(&g("predictor_norm.weight"), &g("predictor_norm.bias"), EPS);
    let out = lin(&fx, "predictor_proj", ed).to_vec().await; // [T*HW, 1408]

    let gr = jg.get("out").as_f64_vec();
    let mut e = 0.0f64; let mut se = 0.0f64;
    for k in 0..t * hw * ed { let dd = (out[k] as f64 - gr[k]).abs(); e = e.max(dd); se += dd; }
    let mean = se / (t * hw * ed) as f64;
    println!("V-JEPA 2-AC predictor vs real: maxΔ={e:.3e}  meanΔ={mean:.3e}  ->  {}", if mean < 3e-3 { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(mean < 3e-3, "Ferric V-JEPA 2-AC predictor diverged");
}

mod json_min {
    pub struct Val(nd::Node);
    pub fn parse(s: &str) -> Val { Val(nd::parse(s)) }
    impl Val {
        pub fn get(&self, k: &str) -> Val { Val(self.0.get(k)) }
        pub fn as_usize(&self) -> usize { self.0.as_f64() as usize }
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
