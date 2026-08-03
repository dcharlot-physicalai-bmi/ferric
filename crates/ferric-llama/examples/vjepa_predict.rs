//! V-JEPA 2 base masked PREDICTOR in pure Rust, verified vs the real transformers VJEPA2Predictor
//! (default masks: predict all 512 patches). Completes the base V-JEPA 2 world model (encoder+predictor).
//! Key: for full non-causal attention with per-token RoPE, the predictor's sort→layers→unsort CANCELS,
//! so [context, target] is processed directly with positions [0..N, 0..N]. Fed the real encoder output
//! to isolate predictor error. usage: cargo run -p ferric-llama --example vjepa_predict --release -- <safetensors>
use ferric_core::Context;
use ferric_load::{safetensors_filtered, STensor};
use ferric_tensor::{nn, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

const EPS: f32 = 1e-6;

fn build_rope(ctx: &Arc<Context>, n_ctx: usize, hd: usize, grid: usize) -> (Tensor, Tensor) {
    // 3-axis interleaved RoPE table for tokens [context(0..n_ctx), target(0..n_ctx)] -> pos = i % n_ctx
    let slice = 2 * ((hd / 3) / 2);
    let half = slice / 2;
    let omega: Vec<f64> = (0..half).map(|j| 1.0 / 10000f64.powf(j as f64 / half as f64)).collect();
    let n = 2 * n_ctx;
    let mut cos = vec![0f32; n * hd];
    let mut sin = vec![0f32; n * hd];
    for i in 0..n {
        let pos = i % n_ctx;
        let (frame, height, width) = (pos / (grid * grid), (pos % (grid * grid)) / grid, pos % grid);
        for (off, p) in [(0usize, frame), (slice, height), (2 * slice, width)] {
            for j in 0..half {
                let f = p as f64 * omega[j];
                let (cv, sv) = (f.cos() as f32, f.sin() as f32);
                cos[i * hd + off + j] = cv; cos[i * hd + off + half + j] = cv;
                sin[i * hd + off + j] = sv; sin[i * hd + off + half + j] = sv;
            }
        }
        for k in (3 * slice)..hd { cos[i * hd + k] = 1.0; }
    }
    (Tensor::from_vec(ctx, &cos, &[n, hd]), Tensor::from_vec(ctx, &sin, &[n, hd]))
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        let d = format!("{}/.cache/huggingface/hub/models--facebook--vjepa2-vitl-fpc64-256/snapshots", std::env::var("HOME").unwrap());
        let snap = std::fs::read_dir(&d).unwrap().next().unwrap().unwrap().path();
        std::fs::read_dir(snap).unwrap().filter_map(|e| { let p = e.unwrap().path(); if p.extension().and_then(|x| x.to_str()) == Some("safetensors") { Some(p.display().to_string()) } else { None } }).next().unwrap()
    });
    let gp = format!("{}/.cache/ferric/cosmos_ref/vjepa_pred_golden.json", std::env::var("HOME").unwrap());
    let jg = json_min::parse(&std::fs::read_to_string(&gp).unwrap());
    let ctx = Arc::new(Context::new().await.unwrap());
    let w: HashMap<String, STensor> = safetensors_filtered(&std::fs::read(&path).unwrap(), |n: &str| n.starts_with("predictor.")).unwrap();
    let g = |n: &str| { let s = &w[n]; Tensor::from_vec(&ctx, &s.data, &s.shape) };
    let (ph, nh, hd) = (384usize, 12usize, 32usize); // pred_hidden, pred_heads, head_dim
    let dh = 1024usize;

    // real encoder output [1,512,1024]
    let es = jg.get("enc_shape").as_usize_vec();
    let n_ctx = es[1];
    let enc = Tensor::from_vec(&ctx, &jg.get("enc").as_f64_vec().iter().map(|v| *v as f32).collect::<Vec<_>>(), &[n_ctx, dh]);

    // context = predictor_embeddings(enc)  [512,384]; target = zeros [512,384]; h = cat
    let context = enc.matmul_bt(&g("predictor.embeddings.predictor_embeddings.weight"))
        .add(&g("predictor.embeddings.predictor_embeddings.bias").reshape(&[1, ph]));
    let target = Tensor::from_vec(&ctx, &vec![0f32; n_ctx * ph], &[n_ctx, ph]);
    let mut x = context.cat(&target, 0); // [1024, 384]

    let (cos, sin) = build_rope(&ctx, n_ctx, hd, 16);
    for il in 0..12 {
        let b = |s: &str| g(&format!("predictor.layer.{il}.{s}"));
        let h = x.layernorm(&b("norm1.weight"), &b("norm1.bias"), EPS);
        let q = h.matmul_bt(&b("attention.query.weight")).add(&b("attention.query.bias").reshape(&[1, ph])).apply_rope_interleaved(&cos, &sin, nh, hd);
        let k = h.matmul_bt(&b("attention.key.weight")).add(&b("attention.key.bias").reshape(&[1, ph])).apply_rope_interleaved(&cos, &sin, nh, hd);
        let v = h.matmul_bt(&b("attention.value.weight")).add(&b("attention.value.bias").reshape(&[1, ph]));
        let o = nn::full_attention_kv(&q, &k, &v, nh, nh).matmul_bt(&b("attention.proj.weight")).add(&b("attention.proj.bias").reshape(&[1, ph]));
        x = x.add(&o);
        let h2 = x.layernorm(&b("norm2.weight"), &b("norm2.bias"), EPS);
        let m = h2.matmul_bt(&b("mlp.fc1.weight")).add(&b("mlp.fc1.bias").reshape(&[1, 1536])).gelu()
            .matmul_bt(&b("mlp.fc2.weight")).add(&b("mlp.fc2.bias").reshape(&[1, ph]));
        x = x.add(&m);
    }
    x = x.layernorm(&g("predictor.layernorm.weight"), &g("predictor.layernorm.bias"), EPS);
    // take target tokens [512:1024], proj 384->1024
    let tgt = x.narrow(0, n_ctx, n_ctx);
    let pred = tgt.matmul_bt(&g("predictor.proj.weight")).add(&g("predictor.proj.bias").reshape(&[1, dh]));
    let mine = pred.to_vec().await;

    let pr = jg.get("pred").as_f64_vec(); // [1,512,1024]
    let mut e = 0.0f64;
    for k in 0..n_ctx * dh { e = e.max((mine[k] as f64 - pr[k]).abs()); }
    println!("V-JEPA 2 predictor Δ vs real VJEPA2Predictor = {e:.3e}  (fed real encoder output)  ->  {}", if e < 3e-3 { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(e < 3e-3, "Ferric V-JEPA 2 predictor diverged");
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
