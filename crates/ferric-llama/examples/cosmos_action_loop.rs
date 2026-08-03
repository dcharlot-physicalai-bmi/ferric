//! End-to-end Cosmos 3 Edge ACTION denoising loop in pure Rust, verified against the diffusers
//! `Cosmos3OmniPipeline` action path. Composes the pieces that are already gold-verified — here the
//! velocity is supplied by deterministic synthetic functions (standing in for the generator tower's
//! forward, itself verified exact in `cosmos_gen_forward`) — and runs the exact pipeline assembly:
//!
//!   mask conditioning velocity -> CFG combine -> UniPC step -> zero channels >= raw_action_dim -> slice
//!
//! Verified against `~/.cache/ferric/cosmos_ref/action_loop_golden.json`, whose trajectory was itself
//! confirmed to agree with the REAL diffusers scheduler loop to 2.6e-7. usage:
//!   cargo run -p ferric-llama --example cosmos_action_loop
use ferric_llama::unipc::UniPc;

fn main() {
    let path = format!("{}/.cache/ferric/cosmos_ref/action_loop_golden.json", std::env::var("HOME").unwrap());
    let g = json_min::parse(&std::fs::read_to_string(&path).expect("run action_loop_ref.py first"));
    let num_steps = g.get("num_steps").as_usize();
    let t = g.get("T").as_usize();
    let adim = g.get("adim").as_usize();
    let raw = g.get("raw").as_usize();
    let cfg = g.get("cfg").as_f64();
    let cond_mask = g.get("cond_mask").as_f64_vec();          // [T]
    let init = g.get("init").as_f64_vec();                    // [T*adim]
    let traj_ref = g.get("trajectory").as_f64_mat();          // [num_steps+1][T*adim]
    let final_ref = g.get("final_action").as_f64_vec();       // [T*raw]

    // synthetic cond/uncond velocities — identical formulas to action_loop_ref.py
    let synth = |lat: &[f64], step: usize| -> (Vec<f64>, Vec<f64>) {
        let mut cond = vec![0.0; t * adim];
        let mut uncond = vec![0.0; t * adim];
        for row in 0..t {
            for c in 0..adim {
                let k = row * adim + c;
                let idx = k as f64;
                cond[k] = 0.4 * (0.5 * idx + step as f64 * 0.3 + 0.1 * lat[k]).cos();
                uncond[k] = 0.3 * (0.4 * idx + step as f64 * 0.2 - 0.1 * lat[k]).sin();
            }
        }
        (cond, uncond)
    };
    // _mask_velocity_predictions (action branch) + CFG: zero conditioning rows, zero pad channels
    let mask_pad_cfg = |cond: &[f64], uncond: &[f64]| -> Vec<f64> {
        let mut v = vec![0.0; t * adim];
        for row in 0..t {
            for c in 0..adim {
                let k = row * adim + c;
                let cfg_v = uncond[k] + cfg * (cond[k] - uncond[k]);
                v[k] = if c >= raw { 0.0 } else { cfg_v * (1.0 - cond_mask[row]) };
            }
        }
        v
    };

    let mut sched = UniPc::new(num_steps);
    let mut lat = init.clone();
    let mut max_traj = max_abs(&lat, &traj_ref[0]);
    for i in 0..num_steps {
        let (cond, uncond) = synth(&lat, i);
        let v = mask_pad_cfg(&cond, &uncond);
        lat = sched.step(&v, &lat);
        for row in 0..t { for c in raw..adim { lat[row * adim + c] = 0.0; } } // zero padding after step
        let e = max_abs(&lat, &traj_ref[i + 1]);
        max_traj = max_traj.max(e);
        println!("  step {i}: Δ={e:.2e}");
    }
    // slice to raw_action_dim -> the emitted action chunk
    let mut out = Vec::with_capacity(t * raw);
    for row in 0..t { for c in 0..raw { out.push(lat[row * adim + c]); } }
    let out_err = max_abs(&out, &final_ref);

    println!("\naction chunk [T x raw] (Ferric):");
    for row in 0..t {
        println!("  {:?}", (0..raw).map(|c| (out[row * raw + c] * 1e5).round() / 1e5).collect::<Vec<_>>());
    }
    let ok = max_traj < 1e-6 && out_err < 1e-6;
    println!("\ntrajectory Δ vs golden = {max_traj:.3e}   final-action Δ = {out_err:.3e}   ->  {}",
             if ok { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(ok, "Cosmos action loop diverged from the verified diffusers-pipeline reference");
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) }

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
