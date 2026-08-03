//! π0 VLM prefix — the Gemma-2B language model — in pure Rust, verified vs the real PaliGemma. Given the
//! merged [image tokens; text embeds] prefix, runs 18 bidirectional Gemma layers and checks the per-layer
//! KV cache (what the flow expert attends to) matches the real model. Completes π0 image→action end-to-end
//! (SigLIP ✓ + this + flow expert ✓). usage: cargo run -p ferric-llama --example pi0_gemma --release
use ferric_core::Context;
use ferric_load::{safetensors_filtered, STensor};
use ferric_tensor::{nn, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

const EPS: f32 = 1e-6;
const EMBED_SCALE: bool = false; // PaliGemma text model scales inputs_embeds by sqrt(hidden)

fn main() { pollster::block_on(run()); }
async fn run() {
    let home = std::env::var("HOME").unwrap();
    let wp = format!("{home}/.cache/ferric/pi05/pi0_gemma2b.safetensors");
    let jg = json_min::parse(&std::fs::read_to_string(format!("{home}/.cache/ferric/cosmos_ref/pi0_gemma_golden.json")).unwrap());
    let ctx = Arc::new(Context::new().await.unwrap());
    let w: HashMap<String, STensor> = safetensors_filtered(&std::fs::read(&wp).unwrap(), |n: &str| !n.contains("embed_tokens") && !n.contains("lm_head")).unwrap();
    let g = |n: &str| { let s = &w[n]; Tensor::from_vec(&ctx, &s.data, &s.shape) };
    let gnorm = |n: &str| { let s = &w[n]; Tensor::from_vec(&ctx, &s.data.iter().map(|x| x + 1.0).collect::<Vec<_>>(), &s.shape) };
    let (hidden, nh, nkv, hd, inter) = (2048usize, 8usize, 1usize, 256usize, 16384usize);
    let plen = jg.get("prefix_len").as_usize();

    let mut x = Tensor::from_vec(&ctx, &jg.get("prefix_embs").as_f64_vec().iter().map(|v| *v as f32).collect::<Vec<_>>(), &[plen, hidden]);
    if EMBED_SCALE { x = x.mul(&x.scalar((hidden as f32).sqrt())); }

    let mut kv_err = 0.0f64;
    for l in 0..18 {
        let b = |s: &str| g(&format!("lm.layers.{l}.{s}"));
        let h = x.rmsnorm(&gnorm(&format!("lm.layers.{l}.input_layernorm.weight")), EPS);
        let q = h.matmul_bt(&b("self_attn.q_proj.weight")).rope(nh, hd, 10000.0, 0);
        let k = h.matmul_bt(&b("self_attn.k_proj.weight")).rope(nkv, hd, 10000.0, 0);
        let v = h.matmul_bt(&b("self_attn.v_proj.weight"));
        // verify this layer's cached k,v vs golden
        let kg = jg.get("kv").idx(l);
        let (mk, mv) = (k.to_vec().await, v.to_vec().await);
        let gk = kg.get("k").as_f64_vec(); let gv = kg.get("v").as_f64_vec();
        let mut le = 0.0f64;
        for i in 0..mk.len() { le = le.max((mk[i] as f64 - gk[i]).abs()); }
        for i in 0..mv.len() { le = le.max((mv[i] as f64 - gv[i]).abs()); }
        if l < 3 || le > 1e-2 { println!("  layer {l} KV Δ={le:.3e}"); }
        kv_err = kv_err.max(le);
        // bidirectional full attention (prefix-lm, all tokens attend to all)
        let o = nn::full_attention_kv(&q, &k, &v, nh, nkv).matmul_bt(&b("self_attn.o_proj.weight"));
        x = x.add(&o);
        let h2 = x.rmsnorm(&gnorm(&format!("lm.layers.{l}.post_attention_layernorm.weight")), EPS);
        let m = h2.matmul_bt(&b("mlp.gate_proj.weight")).gelu_tanh().mul(&h2.matmul_bt(&b("mlp.up_proj.weight"))).matmul_bt(&b("mlp.down_proj.weight"));
        x = x.add(&m);
    }
    let _ = inter;
    println!("π0 Gemma-2B prefix KV cache vs real PaliGemma: maxΔ={kv_err:.3e}  ->  {}", if kv_err < 3e-3 { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(kv_err < 3e-3, "π0 Gemma-2B diverged");
}

mod json_min {
    pub struct Val(nd::Node);
    pub fn parse(s: &str) -> Val { Val(nd::parse(s)) }
    impl Val {
        pub fn get(&self, k: &str) -> Val { Val(self.0.get(k)) }
        pub fn idx(&self, i: usize) -> Val { Val(self.0.idx(i)) }
        pub fn as_usize(&self) -> usize { self.0.as_f64() as usize }
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
