//! π0 (Physical Intelligence) flow-matching ACTION EXPERT in pure Rust, verified vs the real lerobot
//! PI0 denoise_step. The EFA-relevant actuation primitive: given a VLM prefix (captured KV) + robot
//! state + noisy action chunk x_t + flow timestep, the Gemma-300m expert predicts the rectified-flow
//! VELOCITY (noise−actions). embed_suffix (state/action proj + sinusoidal time + silu-MLP) → 18 Gemma
//! layers (GQA 8:1, hd 256, gelu_tanh, (1+w) RMSNorm, RoPE θ1e4) attending to [prefix_KV; suffix] under
//! the pi0 block mask → action_out_proj = velocity. usage: cargo run -p ferric-llama --example pi0_expert --release
use ferric_core::Context;
use ferric_load::{safetensors_filtered, STensor};
use ferric_tensor::{nn, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

const EPS: f32 = 1e-6;
const EMBED_SCALE: bool = false; // verified: the pi0 expert does NOT apply Gemma's sqrt(hidden) embed scale

fn main() { pollster::block_on(run()); }
async fn run() {
    let home = std::env::var("HOME").unwrap();
    let wp = format!("{home}/.cache/ferric/pi05/pi0_expert.safetensors");
    let jg = json_min::parse(&std::fs::read_to_string(format!("{home}/.cache/ferric/cosmos_ref/pi0_golden.json")).unwrap());
    let ctx = Arc::new(Context::new().await.unwrap());
    let w: HashMap<String, STensor> = safetensors_filtered(&std::fs::read(&wp).unwrap(), |n: &str| !n.contains("lm_head")).unwrap();
    let g = |n: &str| { let s = &w[n]; Tensor::from_vec(&ctx, &s.data, &s.shape) };
    // Gemma RMSNorm weight = (1 + w)
    let gnorm = |n: &str| { let s = &w[n]; Tensor::from_vec(&ctx, &s.data.iter().map(|x| x + 1.0).collect::<Vec<_>>(), &s.shape) };
    let lin = |x: &Tensor, n: &str, o: usize| {
        let b = format!("{n}.bias");
        let y = x.matmul_bt(&g(&format!("{n}.weight")));
        if w.contains_key(&b) { y.add(&g(&b).reshape(&[1, o])) } else { y }
    };
    let (pd, nh, nkv, hd) = (1024usize, 8usize, 1usize, 256usize);
    let (chunk, adim, sdim) = (jg.get("chunk").as_usize(), jg.get("adim").as_usize(), jg.get("sdim").as_usize());
    let plen = jg.get("prefix_len").as_usize();
    let ts = jg.get("timestep").as_f64();
    let n = 1 + chunk; // suffix: state + chunk actions

    let f2t = |k: &str, d: &[usize]| Tensor::from_vec(&ctx, &jg.get(k).as_f64_vec().iter().map(|v| *v as f32).collect::<Vec<_>>(), d);
    let state = f2t("state", &[1, sdim]);
    let x_t = f2t("x_t", &[chunk, adim]);

    // ---- embed_suffix ----
    let state_emb = lin(&state, "state_proj", pd); // [1, 1024]
    // sinusoidal time [1024]: period=min*(max/min)^frac; sc=2pi/period; [sin(sc*t); cos(sc*t)]
    let (minp, maxp) = (jg.get("min_period").as_f64(), jg.get("max_period").as_f64());
    let half = pd / 2;
    let mut te = vec![0f32; pd];
    for i in 0..half {
        let frac = i as f64 / (half - 1) as f64;
        let period = minp * (maxp / minp).powf(frac);
        let sc = 1.0 / period * 2.0 * std::f64::consts::PI;
        te[i] = (sc * ts).sin() as f32;
        te[half + i] = (sc * ts).cos() as f32;
    }
    let action_emb = lin(&x_t, "action_in_proj", pd); // [chunk, 1024]
    let time_row = Tensor::from_vec(&ctx, &te, &[1, pd]);
    // broadcast time over chunk via cat then reshape: build [chunk, 2048] on host-friendly path
    let ae = action_emb.to_vec().await;
    let mut at = vec![0f32; chunk * 2 * pd];
    for c in 0..chunk { for j in 0..pd { at[c * 2 * pd + j] = ae[c * pd + j]; at[c * 2 * pd + pd + j] = te[j]; } }
    let action_time = Tensor::from_vec(&ctx, &at, &[chunk, 2 * pd]);
    let action_time = lin(&action_time, "action_time_mlp_in", pd).silu();
    let action_time = lin(&action_time, "action_time_mlp_out", pd); // [chunk, 1024]
    let _ = time_row;
    let mut x = state_emb.cat(&action_time, 0); // [n, 1024]
    if EMBED_SCALE { x = x.mul(&x.scalar((pd as f32).sqrt())); }

    // ---- pi0 block mask [n, plen+n] additive ----
    // prefix (0..plen) all allowed. suffix: state(row0)->state only; action->state+all actions.
    let tk = plen + n;
    let mut mask = vec![0f32; n * tk];
    for qi in 0..n { for kj in 0..n { // suffix-suffix block
        let allowed = if qi == 0 { kj == 0 } else { true };
        if !allowed { mask[qi * tk + (plen + kj)] = -1e30; }
    }}
    let mask_t = Tensor::from_vec(&ctx, &mask, &[n, tk]);

    // ---- 18 Gemma expert layers ----
    for l in 0..18 {
        let b = |s: &str| g(&format!("expert.layers.{l}.{s}"));
        let h = x.rmsnorm(&gnorm(&format!("expert.layers.{l}.input_layernorm.weight")), EPS);
        let q = h.matmul_bt(&b("self_attn.q_proj.weight")).rope(nh, hd, 10000.0, plen);
        let k = h.matmul_bt(&b("self_attn.k_proj.weight")).rope(nkv, hd, 10000.0, plen);
        let v = h.matmul_bt(&b("self_attn.v_proj.weight"));
        // prefix KV for this layer
        let kv = jg.get("kv").idx(l);
        let pk = Tensor::from_vec(&ctx, &kv.get("k").as_f64_vec().iter().map(|x| *x as f32).collect::<Vec<_>>(), &[plen, hd]);
        let pv = Tensor::from_vec(&ctx, &kv.get("v").as_f64_vec().iter().map(|x| *x as f32).collect::<Vec<_>>(), &[plen, hd]);
        let kf = pk.cat(&k, 0);
        let vf = pv.cat(&v, 0);
        let o = nn::masked_attention_kv(&q, &kf, &vf, &mask_t, nh, nkv).matmul_bt(&b("self_attn.o_proj.weight"));
        x = x.add(&o);
        let h2 = x.rmsnorm(&gnorm(&format!("expert.layers.{l}.post_attention_layernorm.weight")), EPS);
        let m = h2.matmul_bt(&b("mlp.gate_proj.weight")).gelu_tanh().mul(&h2.matmul_bt(&b("mlp.up_proj.weight"))).matmul_bt(&b("mlp.down_proj.weight"));
        x = x.add(&m);
    }
    x = x.rmsnorm(&gnorm("expert.norm.weight"), EPS);
    let actions_out = x.narrow(0, 1, chunk); // [chunk, 1024]
    let velocity = lin(&actions_out, "action_out_proj", adim).to_vec().await; // [chunk, 32]

    let vr = jg.get("velocity").as_f64_vec();
    let mut e = 0.0f64; let mut se = 0.0f64;
    for i in 0..chunk * adim { let d = (velocity[i] as f64 - vr[i]).abs(); e = e.max(d); se += d; }
    let mean = se / (chunk * adim) as f64;
    println!("π0 flow-expert velocity vs real lerobot PI0: maxΔ={e:.3e}  meanΔ={mean:.3e}  ->  {}", if mean < 3e-3 { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(mean < 3e-3, "π0 expert diverged");
}

mod json_min {
    pub struct Val(nd::Node);
    pub fn parse(s: &str) -> Val { Val(nd::parse(s)) }
    impl Val {
        pub fn get(&self, k: &str) -> Val { Val(self.0.get(k)) }
        pub fn idx(&self, i: usize) -> Val { Val(self.0.idx(i)) }
        pub fn as_usize(&self) -> usize { self.0.as_f64() as usize }
        pub fn as_f64(&self) -> f64 { self.0.as_f64() }
        pub fn as_f64_vec(&self) -> Vec<f64> { self.0.as_vec().iter().map(|n| n.as_f64()).collect() }
    }
    mod nd {
        #[derive(Clone)]
        pub enum Node { Num(f64), Arr(Vec<Node>), Obj(Vec<(String, Node)>), Null }
        impl Node {
            pub fn get(&self, k: &str) -> Node { if let Node::Obj(m) = self { for (kk, v) in m { if kk == k { return v.clone(); } } } Node::Null }
            pub fn idx(&self, i: usize) -> Node { if let Node::Arr(a) = self { a[i].clone() } else { Node::Null } }
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
