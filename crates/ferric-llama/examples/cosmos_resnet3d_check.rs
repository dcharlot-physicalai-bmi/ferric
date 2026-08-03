//! Verify Ferric's WanResidualBlock composition (conv3d + WanRMS_norm[=rmsnorm eps→0] + silu) against
//! the real block — both the conv_shortcut (in!=out) and identity (in==out) variants.
//! usage: cargo run -p ferric-llama --example cosmos_resnet3d_check
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;

const EPS: f32 = 1e-12;

// [O,C,kt,kh,kw] -> [kt,kh,kw,C,O]
fn reorder_w(w: &[f64], o: usize, c: usize, kt: usize, kh: usize, kw: usize) -> Vec<f32> {
    let mut r = vec![0f32; kt * kh * kw * c * o];
    for oo in 0..o { for cc in 0..c { for a in 0..kt { for ky in 0..kh { for kx in 0..kw {
        let src = (((oo * c + cc) * kt + a) * kh + ky) * kw + kx;
        r[((((a * kh + ky) * kw + kx) * c + cc) * o) + oo] = w[src] as f32;
    }}}}}
    r
}
// [1,C,T,H,W] -> [T,H,W,C]
fn to_thwc(x: &[f64], c: usize, t: usize, h: usize, w: usize) -> Vec<f32> {
    let mut r = vec![0f32; t * h * w * c];
    for cc in 0..c { for tt in 0..t { for hh in 0..h { for ww in 0..w {
        r[((tt * h + hh) * w + ww) * c + cc] = x[((cc * t + tt) * h + hh) * w + ww] as f32;
    }}}}
    r
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let path = format!("{}/.cache/ferric/cosmos_ref/resnet3d_golden.json", std::env::var("HOME").unwrap());
    let g = json_min::parse(&std::fs::read_to_string(&path).expect("run resnet3d_ref.py first"));
    let ctx = Arc::new(Context::new().await.unwrap());

    // causal 3x3x3 conv: left-pad T by 2, symmetric (1,1) spatial
    let conv333 = |x: &Tensor, w: &Tensor, b: &Tensor, t: usize, h: usize, wd: usize, c: usize| -> Tensor {
        let zeros = Tensor::from_vec(&ctx, &vec![0f32; 2 * h * wd * c], &[2, h, wd, c]);
        let _ = t;
        zeros.cat(x, 0).conv3d(w, b, (1, 1, 1), (1, 1))
    };

    let mut worst = 0.0f64;
    for name in ["shortcut", "identity"] {
        let c = g.get(name);
        let cin = c.get("cin").as_usize();
        let cout = c.get("cout").as_usize();
        let ins = c.get("in_shape").as_usize_vec(); // [1,C,T,H,W]
        let (t, h, wd) = (ins[2], ins[3], ins[4]);
        let has_sc = c.get("has_shortcut").as_f64() != 0.0;

        let x0 = Tensor::from_vec(&ctx, &to_thwc(&c.get("x").as_f64_vec(), cin, t, h, wd), &[t, h, wd, cin]);
        let n1 = Tensor::from_vec(&ctx, &c.get("norm1_g").as_f64_vec().iter().map(|v| *v as f32).collect::<Vec<_>>(), &[cin]);
        let n2 = Tensor::from_vec(&ctx, &c.get("norm2_g").as_f64_vec().iter().map(|v| *v as f32).collect::<Vec<_>>(), &[cout]);
        let c1w = Tensor::from_vec(&ctx, &reorder_w(&c.get("conv1_w").as_f64_vec(), cout, cin, 3, 3, 3), &[3, 3, 3, cin, cout]);
        let c1b = Tensor::from_vec(&ctx, &c.get("conv1_b").as_f64_vec().iter().map(|v| *v as f32).collect::<Vec<_>>(), &[cout]);
        let c2w = Tensor::from_vec(&ctx, &reorder_w(&c.get("conv2_w").as_f64_vec(), cout, cout, 3, 3, 3), &[3, 3, 3, cout, cout]);
        let c2b = Tensor::from_vec(&ctx, &c.get("conv2_b").as_f64_vec().iter().map(|v| *v as f32).collect::<Vec<_>>(), &[cout]);

        // shortcut
        let hsc = if has_sc {
            let scw = Tensor::from_vec(&ctx, &reorder_w(&c.get("sc_w").as_f64_vec(), cout, cin, 1, 1, 1), &[1, 1, 1, cin, cout]);
            let scb = Tensor::from_vec(&ctx, &c.get("sc_b").as_f64_vec().iter().map(|v| *v as f32).collect::<Vec<_>>(), &[cout]);
            x0.conv3d(&scw, &scb, (1, 1, 1), (0, 0))
        } else { x0.clone() };
        // main path
        let x = x0.rmsnorm(&n1, EPS).silu();
        let x = conv333(&x, &c1w, &c1b, t, h, wd, cin);
        let x = x.rmsnorm(&n2, EPS).silu();
        let x = conv333(&x, &c2w, &c2b, t, h, wd, cout);
        let out = x.add(&hsc).to_vec().await; // [T,H,W,O]

        let yr = c.get("y").as_f64_vec(); // [1,O,T,H,W]
        let mut err = 0.0f64;
        for o in 0..cout { for tt in 0..t { for hh in 0..h { for ww in 0..wd {
            let mine = out[((tt * h + hh) * wd + ww) * cout + o] as f64;
            let re = yr[((o * t + tt) * h + hh) * wd + ww];
            err = err.max((mine - re).abs());
        }}}}
        worst = worst.max(err);
        println!("  {name}: [1,{cin},{t},{h},{wd}] -> [1,{cout},..]   maxΔ={err:.2e}");
    }
    println!("\nMAX WanResidualBlock Δ = {worst:.3e}  ->  {}", if worst < 2e-4 { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(worst < 2e-4, "Ferric resnet3d diverged from WanResidualBlock");
}

mod json_min {
    pub struct Val(nd::Node);
    pub fn parse(s: &str) -> Val { Val(nd::parse(s)) }
    impl Val {
        pub fn get(&self, k: &str) -> Val { Val(self.0.get(k)) }
        pub fn as_usize(&self) -> usize { self.0.as_f64() as usize }
        pub fn as_f64(&self) -> f64 { self.0.as_f64() }
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
            match b[*i] {
                b'{' => po(b, i), b'[' => pa(b, i), b'"' => { ps(b, i); Node::Null }
                b't' => { *i += 4; Node::Num(1.0) }   // true
                b'f' => { *i += 5; Node::Num(0.0) }   // false
                b'n' => { *i += 4; Node::Null }       // null
                _ => pn(b, i),
            }
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
