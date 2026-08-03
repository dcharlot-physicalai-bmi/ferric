//! Verify Ferric's `Tensor::conv3d` against the real `WanCausalConv3d` (three kernel shapes the Wan VAE
//! decoder uses). Reorders reference tensors from channels-first [1,C,T,H,W] / [O,C,kT,kH,kW] into
//! Ferric's [T,H,W,C] / [kT,kH,kW,C,O] layout, causal-time-pads, convolves, compares.
//! usage: cargo run -p ferric-llama --example cosmos_conv3d_check
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let path = format!("{}/.cache/ferric/cosmos_ref/conv3d_golden.json", std::env::var("HOME").unwrap());
    let g = json_min::parse(&std::fs::read_to_string(&path).expect("run conv3d_ref.py first"));
    let ctx = Arc::new(Context::new().await.unwrap());
    let mut worst = 0.0f64;
    for name in ["k333", "k311", "k133"] {
        let c = g.get(name);
        let cin = c.get("cin").as_usize();
        let cout = c.get("cout").as_usize();
        let k = c.get("k").as_usize_vec();     // [kt,kh,kw]
        let pad = c.get("pad").as_usize_vec(); // [pt,ph,pw]
        let ins = c.get("in_shape").as_usize_vec();  // [1,C,T,H,W]
        let outs = c.get("out_shape").as_usize_vec(); // [1,O,To,Ho,Wo]
        let (t, h, wd) = (ins[2], ins[3], ins[4]);
        let (kt, kh, kw) = (k[0], k[1], k[2]);
        let (to, ho, wo) = (outs[2], outs[3], outs[4]);
        let xr = c.get("x").as_f64_vec();
        let wr = c.get("w").as_f64_vec();
        let br = c.get("b").as_f64_vec();
        let yr = c.get("y").as_f64_vec();

        // x [1,C,T,H,W] -> causal-time-padded [T+2pt, H, W, C]
        let pt2 = 2 * pad[0];
        let tp = t + pt2;
        let mut xt = vec![0f32; tp * h * wd * cin];
        for cc in 0..cin { for tt in 0..t { for hh in 0..h { for ww in 0..wd {
            let src = ((cc * t + tt) * h + hh) * wd + ww;
            let dst = (((tt + pt2) * h + hh) * wd + ww) * cin + cc;
            xt[dst] = xr[src] as f32;
        }}}}
        // w [O,C,kt,kh,kw] -> [kt,kh,kw,C,O]
        let mut wt = vec![0f32; kt * kh * kw * cin * cout];
        for o in 0..cout { for cc in 0..cin { for a in 0..kt { for ky in 0..kh { for kx in 0..kw {
            let src = (((o * cin + cc) * kt + a) * kh + ky) * kw + kx;
            let dst = ((((a * kh + ky) * kw + kx) * cin + cc) * cout) + o;
            wt[dst] = wr[src] as f32;
        }}}}}
        let x = Tensor::from_vec(&ctx, &xt, &[tp, h, wd, cin]);
        let w = Tensor::from_vec(&ctx, &wt, &[kt, kh, kw, cin, cout]);
        let b = Tensor::from_vec(&ctx, &br.iter().map(|v| *v as f32).collect::<Vec<_>>(), &[cout]);
        let out = x.conv3d(&w, &b, (1, 1, 1), (pad[1], pad[2])).to_vec().await; // [To,Ho,Wo,O]

        let mut err = 0.0f64;
        for o in 0..cout { for tt in 0..to { for hh in 0..ho { for ww in 0..wo {
            let mine = out[((tt * ho + hh) * wo + ww) * cout + o] as f64;
            let re = yr[((o * to + tt) * ho + hh) * wo + ww];
            err = err.max((mine - re).abs());
        }}}}
        worst = worst.max(err);
        println!("  {name}: [1,{cin},{t},{h},{wd}] -> [1,{cout},{to},{ho},{wo}]   maxΔ={err:.2e}");
    }
    println!("\nMAX conv3d Δ vs WanCausalConv3d = {worst:.3e}  ->  {}",
             if worst < 1e-4 { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(worst < 1e-4, "Ferric conv3d diverged from WanCausalConv3d");
}

mod json_min {
    pub struct Val(nd::Node);
    pub fn parse(s: &str) -> Val { Val(nd::parse(s)) }
    impl Val {
        pub fn get(&self, k: &str) -> Val { Val(self.0.get(k)) }
        pub fn as_usize(&self) -> usize { self.0.as_f64() as usize }
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
            match b[*i] { b'{' => po(b, i), b'[' => pa(b, i), b'"' => { ps(b, i); Node::Null } _ => pn(b, i) }
        }
        fn po(b: &[u8], i: &mut usize) -> Node {
            *i += 1; let mut m = Vec::new();
            loop {
                ws(b, i); if b[*i] == b'}' { *i += 1; break; }
                let k = ps(b, i); ws(b, i); *i += 1; let v = pv(b, i); m.push((k, v));
                ws(b, i); if b[*i] == b',' { *i += 1; } else if b[*i] == b'}' { *i += 1; break; }
            }
            Node::Obj(m)
        }
        fn pa(b: &[u8], i: &mut usize) -> Node {
            *i += 1; let mut a = Vec::new();
            loop {
                ws(b, i); if b[*i] == b']' { *i += 1; break; }
                a.push(pv(b, i)); ws(b, i);
                if b[*i] == b',' { *i += 1; } else if b[*i] == b']' { *i += 1; break; }
            }
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
