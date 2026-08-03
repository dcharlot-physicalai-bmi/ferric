//! Verify Ferric's UniPC sampler against the numpy/diffusers golden trajectory.
//! Loads `~/.cache/ferric/cosmos_ref/unipc_golden.json` (schedule + synthetic velocities + the
//! reference sample trajectory, itself matched to the real diffusers scheduler to 1.4e-7) and drives
//! `UniPc::step` with the same velocities, asserting the full trajectory reproduces the reference.
//! usage: cargo run -p ferric-llama --example unipc_check
use ferric_llama::unipc::UniPc;

fn main() {
    let path = format!("{}/.cache/ferric/cosmos_ref/unipc_golden.json", std::env::var("HOME").unwrap());
    let txt = std::fs::read_to_string(&path).expect("run unipc_numpy.py first to emit the golden");
    let g = json_min::parse(&txt);

    let num_steps = g.get("num_steps").as_usize();
    let sigmas_ref = g.get("sigmas").as_f64_vec();
    let timesteps_ref = g.get("timesteps").as_f64_vec();
    let x0_init = g.get("x0_init").as_f64_vec();
    let vels: Vec<Vec<f64>> = g.get("vels").as_f64_mat();
    let traj_ref: Vec<Vec<f64>> = g.get("trajectory").as_f64_mat();

    let mut sched = UniPc::new(num_steps);

    // 1) schedule matches
    let sig_err = max_abs(sched.sigmas(), &sigmas_ref);
    let ts_err = max_abs(sched.timesteps(), &timesteps_ref);
    println!("schedule: sigmasΔ={sig_err:.2e}  timestepsΔ={ts_err:.2e}");
    println!("  sigmas   = {:?}", round(sched.sigmas(), 6));
    println!("  timesteps= {:?}", round(sched.timesteps(), 4));

    // 2) trajectory matches
    let mut x = x0_init.clone();
    let mut max_traj = max_abs(&x, &traj_ref[0]);
    for i in 0..num_steps {
        x = sched.step(&vels[i], &x);
        let e = max_abs(&x, &traj_ref[i + 1]);
        max_traj = max_traj.max(e);
        println!("  step {i}: xΔ={e:.2e}  x[..4]={:?}", round(&x[..4.min(x.len())], 6));
    }
    let ok = sig_err < 1e-6 && ts_err < 1e-4 && max_traj < 1e-6;
    println!("\nMAX trajectory Δ vs golden = {max_traj:.3e}  ->  {}", if ok { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(ok, "UniPC Rust port diverged from the verified numpy/diffusers reference");
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}
fn round(a: &[f64], d: i32) -> Vec<f64> {
    let m = 10f64.powi(d);
    a.iter().map(|x| (x * m).round() / m).collect()
}

/// Tiny dependency-free JSON reader for the flat golden file (numbers, arrays, nested arrays).
mod json_min {
    pub struct Val(serde_like::Node);
    pub fn parse(s: &str) -> Val { Val(serde_like::parse(s)) }
    impl Val {
        pub fn get(&self, k: &str) -> Val { Val(self.0.get(k)) }
        pub fn as_usize(&self) -> usize { self.0.as_f64() as usize }
        pub fn as_f64_vec(&self) -> Vec<f64> { self.0.as_vec().iter().map(|n| n.as_f64()).collect() }
        pub fn as_f64_mat(&self) -> Vec<Vec<f64>> {
            self.0.as_vec().iter().map(|row| row.as_vec().iter().map(|n| n.as_f64()).collect()).collect()
        }
    }
    // minimal recursive-descent JSON, enough for {string: number|array}
    mod serde_like {
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
        pub fn parse(s: &str) -> Node {
            let b = s.as_bytes();
            let mut i = 0;
            let v = parse_val(b, &mut i);
            v
        }
        fn skip_ws(b: &[u8], i: &mut usize) { while *i < b.len() && (b[*i] as char).is_whitespace() { *i += 1; } }
        fn parse_val(b: &[u8], i: &mut usize) -> Node {
            skip_ws(b, i);
            match b[*i] {
                b'{' => parse_obj(b, i),
                b'[' => parse_arr(b, i),
                b'"' => { let _ = parse_str(b, i); Node::Null } // strings only appear as keys
                _ => parse_num(b, i),
            }
        }
        fn parse_obj(b: &[u8], i: &mut usize) -> Node {
            *i += 1; let mut m = Vec::new();
            loop {
                skip_ws(b, i);
                if b[*i] == b'}' { *i += 1; break; }
                let key = parse_str(b, i);
                skip_ws(b, i); *i += 1; // ':'
                let val = parse_val(b, i);
                m.push((key, val));
                skip_ws(b, i);
                if b[*i] == b',' { *i += 1; } else if b[*i] == b'}' { *i += 1; break; }
            }
            Node::Obj(m)
        }
        fn parse_arr(b: &[u8], i: &mut usize) -> Node {
            *i += 1; let mut a = Vec::new();
            loop {
                skip_ws(b, i);
                if b[*i] == b']' { *i += 1; break; }
                a.push(parse_val(b, i));
                skip_ws(b, i);
                if b[*i] == b',' { *i += 1; } else if b[*i] == b']' { *i += 1; break; }
            }
            Node::Arr(a)
        }
        fn parse_str(b: &[u8], i: &mut usize) -> String {
            *i += 1; let start = *i;
            while b[*i] != b'"' { *i += 1; }
            let s = String::from_utf8_lossy(&b[start..*i]).to_string();
            *i += 1; s
        }
        fn parse_num(b: &[u8], i: &mut usize) -> Node {
            let start = *i;
            while *i < b.len() && matches!(b[*i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') { *i += 1; }
            Node::Num(std::str::from_utf8(&b[start..*i]).unwrap().parse().unwrap())
        }
    }
}
