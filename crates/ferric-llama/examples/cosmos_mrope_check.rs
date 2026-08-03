//! Verify Ferric's interleaved 3-axis mRoPE (`cosmos::interleaved_mrope`) against the real diffusers
//! `Cosmos3VLTextRotaryEmbedding`, via the golden emitted by `mrope_ref.py`. usage:
//!   cargo run -p ferric-llama --example cosmos_mrope_check
use ferric_llama::cosmos::interleaved_mrope;

fn main() {
    let path = format!("{}/.cache/ferric/cosmos_ref/mrope_golden.json", std::env::var("HOME").unwrap());
    let g = json_min::parse(&std::fs::read_to_string(&path).expect("run mrope_ref.py first"));
    let head_dim = g.get("head_dim").as_usize();
    let theta = g.get("theta").as_f64();
    let axes_v = g.get("axes").as_f64_vec();
    let axes = (axes_v[0] as usize, axes_v[1] as usize, axes_v[2] as usize);
    let pos = g.get("pos").as_f64_mat(); // [3][N]
    let (pt, ph, pw): (Vec<i64>, Vec<i64>, Vec<i64>) = (
        pos[0].iter().map(|x| *x as i64).collect(),
        pos[1].iter().map(|x| *x as i64).collect(),
        pos[2].iter().map(|x| *x as i64).collect(),
    );
    let cos_ref = g.get("cos").as_f64_mat(); // [N][head_dim]
    let sin_ref = g.get("sin").as_f64_mat();

    let (cos, sin) = interleaved_mrope(&pt, &ph, &pw, head_dim, theta, axes);
    let n = pt.len();
    let mut ce = 0f64;
    let mut se = 0f64;
    for tok in 0..n {
        for j in 0..head_dim {
            ce = ce.max((cos[tok * head_dim + j] as f64 - cos_ref[tok][j]).abs());
            se = se.max((sin[tok * head_dim + j] as f64 - sin_ref[tok][j]).abs());
        }
    }
    println!("mRoPE cos maxΔ={ce:.2e}  sin maxΔ={se:.2e}  ->  {}", if ce.max(se) < 1e-5 { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(ce.max(se) < 1e-5, "Ferric mRoPE diverged from the real Cosmos3VLTextRotaryEmbedding");
}

mod json_min {
    pub struct Val(nd::Node);
    pub fn parse(s: &str) -> Val { Val(nd::parse(s)) }
    impl Val {
        pub fn get(&self, k: &str) -> Val { Val(self.0.get(k)) }
        pub fn as_usize(&self) -> usize { self.0.as_f64() as usize }
        pub fn as_f64(&self) -> f64 { self.0.as_f64() }
        pub fn as_f64_vec(&self) -> Vec<f64> { self.0.as_vec().iter().map(|n| n.as_f64()).collect() }
        pub fn as_f64_mat(&self) -> Vec<Vec<f64>> {
            self.0.as_vec().iter().map(|r| r.as_vec().iter().map(|n| n.as_f64()).collect()).collect()
        }
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
