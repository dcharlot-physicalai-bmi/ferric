//! EFA loader — the user-facing "download → load → run" for released EFA models. Reads `config.json` + `model.safetensors`
//! from a model directory, reconstructs the policy, and runs it on the bundled environment spec. Dispatches on the
//! `architecture` field: `efa-hybrid-v0` (2-link hybrid: v=−κ∇ₐE+w, potential also verifies) or `efa-flow-v0`
//! (3-link flow policy). Pure-Rust **CPU** inference — no GPU required for these checkpoints (the edge story in one file).
//!
//! Run: `cargo run -p ferric-tensor --example efa_load --release [-- /path/to/model-dir]`
//! Default model dir: /Users/dcharlot/vibe-coding/efa/models/efa-hybrid-arm2
use std::f32::consts::PI;
const H: usize = 96; const DT: f32 = 0.05; const CPL: f32 = 0.5;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }

fn load_safetensors(path: &str) -> std::io::Result<Vec<(String, Vec<usize>, Vec<f32>)>> {
    let raw = std::fs::read(path)?;
    let hl = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&raw[8..8 + hl]).unwrap().to_string();
    let data = &raw[8 + hl..]; let mut out = vec![]; let mut rest = header.as_str();
    while let Some(q) = rest.find("\"dtype\"") {
        let pre = &rest[..q]; let name_end = pre.rfind("\":{").unwrap(); let name_start = pre[..name_end].rfind('"').unwrap() + 1;
        let name = pre[name_start..name_end].to_string(); let after = &rest[q..];
        let sh_s = after.find("\"shape\":[").unwrap() + 9; let sh_e = after[sh_s..].find(']').unwrap() + sh_s;
        let shape: Vec<usize> = after[sh_s..sh_e].split(',').filter(|s| !s.is_empty()).map(|s| s.trim().parse().unwrap()).collect();
        let of_s = after.find("\"data_offsets\":[").unwrap() + 16; let of_e = after[of_s..].find(']').unwrap() + of_s;
        let offs: Vec<usize> = after[of_s..of_e].split(',').map(|s| s.trim().parse().unwrap()).collect();
        let vals: Vec<f32> = data[offs[0]..offs[1]].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        out.push((name, shape, vals)); rest = &after[of_e..]; }
    Ok(out)
}
// tiny MLP evals (relu hidden, linear out) over prebuilt input vectors
fn mlp_scalar(w: &[Vec<f32>], b1: &[f32], w2: &[f32], b2: &[f32], w3: &[f32], b3: f32, f: &[f32]) -> f32 {
    let mut h1 = vec![0.0f32; H]; for j in 0..H { let mut z = b1[j]; for c in 0..f.len() { z += f[c] * w[c][j]; } h1[j] = z.max(0.0); }
    let mut h2 = vec![0.0f32; H]; for j in 0..H { let mut z = b2[j]; for k in 0..H { z += h1[k] * w2[k * H + j]; } h2[j] = z.max(0.0); }
    let mut o = b3; for j in 0..H { o += h2[j] * w3[j]; } o
}
fn mlp_vec(w: &[Vec<f32>], b1: &[f32], w2: &[f32], b2: &[f32], w3: &[f32], b3: &[f32], f: &[f32], nout: usize) -> Vec<f32> {
    let mut h1 = vec![0.0f32; H]; for j in 0..H { let mut z = b1[j]; for c in 0..f.len() { z += f[c] * w[c][j]; } h1[j] = z.max(0.0); }
    let mut h2 = vec![0.0f32; H]; for j in 0..H { let mut z = b2[j]; for k in 0..H { z += h1[k] * w2[k * H + j]; } h2[j] = z.max(0.0); }
    (0..nout).map(|c| { let mut o = b3[c]; for j in 0..H { o += h2[j] * w3[j * nout + c]; } o }).collect()
}
// exact ∂E/∂a for the scalar relu MLP (backprop; action inputs at indices 8,9)
fn grad_a2(w: &[Vec<f32>], b1: &[f32], w2: &[f32], b2: &[f32], w3: &[f32], f: &[f32]) -> (f32, f32) {
    let mut h1 = vec![0.0f32; H]; let mut m1 = vec![false; H];
    for j in 0..H { let mut z = b1[j]; for c in 0..f.len() { z += f[c] * w[c][j]; } m1[j] = z > 0.0; h1[j] = z.max(0.0); }
    let mut m2 = vec![false; H];
    for j in 0..H { let mut z = b2[j]; for k in 0..H { z += h1[k] * w2[k * H + j]; } m2[j] = z > 0.0; }
    let mut d2 = vec![0.0f32; H]; for j in 0..H { if m2[j] { d2[j] = w3[j]; } }
    let mut d1 = vec![0.0f32; H]; for k in 0..H { if m1[k] { let mut z = 0.0; for j in 0..H { z += w2[k * H + j] * d2[j]; } d1[k] = z; } }
    let (mut g1, mut g2) = (0.0f32, 0.0f32); for j in 0..H { g1 += w[8][j] * d1[j]; g2 += w[9][j] * d1[j]; }
    (g1, g2)
}
fn get<'a>(t: &'a [(String, Vec<usize>, Vec<f32>)], n: &str) -> &'a Vec<f32> { &t.iter().find(|(nm, _, _)| nm == n).unwrap().2 }
fn getv(t: &[(String, Vec<usize>, Vec<f32>)], prefix: &str, n: usize) -> Vec<Vec<f32>> { (0..n).map(|c| get(t, &format!("{prefix}.in{c}")).clone()).collect() }
fn cfg_num(cfg: &str, key: &str) -> f32 { let i = cfg.find(&format!("\"{key}\"")).unwrap(); let rest = &cfg[i..]; let c = rest.find(':').unwrap();
    rest[c + 1..].trim_start().split(|ch: char| ch == ',' || ch == '}' || ch == '\n').next().unwrap().trim().parse().unwrap() }

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "/Users/dcharlot/vibe-coding/efa/models/efa-hybrid-arm2".into());
    let cfg = std::fs::read_to_string(format!("{dir}/config.json")).expect("config.json not found — run the release builder first");
    let arch = { let i = cfg.find("\"architecture\"").unwrap(); let rest = &cfg[i + 14..]; let a = rest.find('"').unwrap() + 1; let b = rest[a..].find('"').unwrap() + a; rest[a..b].to_string() };
    let t = load_safetensors(&format!("{dir}/model.safetensors")).expect("model.safetensors not found");
    let nparams: usize = t.iter().map(|(_, _, d)| d.len()).sum();
    println!("  EFA loader — {dir}");
    println!("  architecture: {arch}   tensors: {}   params: {nparams}   (pure-Rust CPU inference)\n", t.len());

    if arch == "efa-hybrid-v0" {
        let umax = cfg_num(&cfg, "umax"); let kappa = cfg_num(&cfg, "kappa");
        let step = |s: [f32; 4], u1: f32, u2: f32| -> [f32; 4] {
            let (t1, t2, o1, o2) = (s[0], s[1], s[2], s[3]); let (c1, c2) = (u1.clamp(-umax, umax), u2.clamp(-umax, umax));
            let no1 = o1 + DT * (-t1.sin() - 0.05 * o1 + CPL * (t2 - t1).sin() + c1);
            let no2 = o2 + DT * (-t2.sin() - 0.05 * o2 + CPL * (t1 - t2).sin() + c2);
            [wrap(t1 + DT * no1), wrap(t2 + DT * no2), no1, no2] };
        let feat = |s: [f32; 4], g: (f32, f32), a1: f32, a2: f32, tt: f32| -> Vec<f32> { let (d1, d2) = (s[0] - g.0, s[1] - g.1);
            vec![d1.cos(), d1.sin(), s[2], d2.cos(), d2.sin(), s[3], s[0].sin(), s[1].sin(), a1, a2, tt] };
        let (ew, eb1, ew2, eb2, ew3, eb3) = (getv(&t, "potential", 11), get(&t, "potential.b1"), get(&t, "potential.w2"), get(&t, "potential.b2"), get(&t, "potential.w3"), get(&t, "potential.b3")[0]);
        let (ww, wb1, ww2, wb2, ww3, wb3) = (getv(&t, "correction", 11), get(&t, "correction.b1"), get(&t, "correction.w2"), get(&t, "correction.b2"), get(&t, "correction.w3"), get(&t, "correction.b3"));
        let act = |s: [f32; 4], g: (f32, f32)| -> (f32, f32) { let (mut a1, mut a2) = (0.0f32, 0.0f32);
            for k in 0..2 { let tt = k as f32 / 2.0;
                let (g1, g2) = grad_a2(&ew, eb1, ew2, eb2, ew3, &feat(s, g, a1, a2, tt));
                let wv = mlp_vec(&ww, wb1, ww2, wb2, ww3, wb3, &feat(s, g, a1, a2, tt), 2);
                a1 += (-kappa * g1 + wv[0]) / 2.0; a2 += (-kappa * g2 + wv[1]) / 2.0; }
            (a1.clamp(-umax, umax), a2.clamp(-umax, umax)) };
        let goals = [(0.8f32, -0.8f32), (-1.0, 0.6), (0.5, 1.0), (-0.6, -0.9)];
        let (mut reach, mut n) = (0, 0);
        for (gi, &g) in goals.iter().enumerate() { for e in 0..40 { let sd = (gi * 40 + e) as u32;
            let mut s = [(u(900 + sd, 7) * 2.0 - 1.0) * PI, (u(900 + sd, 8) * 2.0 - 1.0) * PI, 0.0, 0.0];
            for _ in 0..260 { let (a1, a2) = act(s, g); s = step(s, a1, a2); }
            n += 1; if wrap(s[0] - g.0).abs() < 0.35 && wrap(s[1] - g.1).abs() < 0.35 && s[2].abs() < 0.7 && s[3].abs() < 0.7 { reach += 1; } } }
        // verify demo: the potential scores the policy's own action vs a random action
        let (mut ok, mut vt) = (0, 0);
        for k in 0..2000u32 { let s = [(u(k, 41) * 2.0 - 1.0) * PI, (u(k, 42) * 2.0 - 1.0) * PI, (u(k, 43) * 2.0 - 1.0) * 3.0, (u(k, 44) * 2.0 - 1.0) * 3.0];
            let g = ((u(k, 45) * 2.0 - 1.0) * 1.2, (u(k, 46) * 2.0 - 1.0) * 1.2);
            let (a1, a2) = act(s, g); let (r1, r2) = ((u(k, 47) * 2.0 - 1.0) * umax, (u(k, 48) * 2.0 - 1.0) * umax);
            let ep = mlp_scalar(&ew, eb1, ew2, eb2, ew3, eb3, &feat(s, g, a1, a2, 1.0));
            let en = mlp_scalar(&ew, eb1, ew2, eb2, ew3, eb3, &feat(s, g, r1, r2, 1.0));
            vt += 1; if ep < en { ok += 1; } }
        println!("  ACTUATE  (K=2, 160 episodes × 4 goals):        reach {:>4.0}%", reach as f32 / n as f32 * 100.0);
        println!("  VERIFY   (potential ranks policy action < random): {:>4.1}%", ok as f32 / vt as f32 * 100.0);
        println!("\n  One potential: the action IS energy descent (−κ∇ₐE + w) and validity IS the same energy's value. Loaded from disk.");
    } else if arch == "efa-flow-v0" {
        let umax = cfg_num(&cfg, "umax");
        let step = |s: [f32; 6], uu: [f32; 3]| -> [f32; 6] {
            let (t1, t2, t3, o1, o2, o3) = (s[0], s[1], s[2], s[3], s[4], s[5]);
            let c: Vec<f32> = uu.iter().map(|x| x.clamp(-umax, umax)).collect();
            let no1 = o1 + DT * (-t1.sin() - 0.05 * o1 + CPL * (t2 - t1).sin() + c[0]);
            let no2 = o2 + DT * (-t2.sin() - 0.05 * o2 + CPL * (t1 - t2).sin() + CPL * (t3 - t2).sin() + c[1]);
            let no3 = o3 + DT * (-t3.sin() - 0.05 * o3 + CPL * (t2 - t3).sin() + c[2]);
            [wrap(t1 + DT * no1), wrap(t2 + DT * no2), wrap(t3 + DT * no3), no1, no2, no3] };
        let feat = |s: [f32; 6], g: (f32, f32, f32), a: [f32; 3], tt: f32| -> Vec<f32> { let (d1, d2, d3) = (s[0] - g.0, s[1] - g.1, s[2] - g.2);
            vec![d1.cos(), d1.sin(), s[3], d2.cos(), d2.sin(), s[4], d3.cos(), d3.sin(), s[5], s[0].sin(), s[1].sin(), s[2].sin(), a[0], a[1], a[2], tt] };
        let (fw, fb1, fw2, fb2, fw3, fb3) = (getv(&t, "flow", 16), get(&t, "flow.b1"), get(&t, "flow.w2"), get(&t, "flow.b2"), get(&t, "flow.w3"), get(&t, "flow.b3"));
        let act = |s: [f32; 6], g: (f32, f32, f32), kk: usize| -> [f32; 3] { let mut a = [0.0f32; 3];
            for i in 0..kk { let tt = i as f32 / kk as f32; let v = mlp_vec(&fw, fb1, fw2, fb2, fw3, fb3, &feat(s, g, a, tt), 3);
                for c in 0..3 { a[c] += v[c] / kk as f32; } }
            [a[0].clamp(-umax, umax), a[1].clamp(-umax, umax), a[2].clamp(-umax, umax)] };
        let goals = [(0.8f32, -0.6f32, 0.5f32), (-0.7, 0.5, -0.6), (0.5, 0.9, -0.4), (-0.5, -0.8, 0.7)];
        for kk in [1usize, 2] { let (mut reach, mut n) = (0, 0);
            for (gi, &g) in goals.iter().enumerate() { for e in 0..30 { let sd = (gi * 30 + e) as u32;
                let mut s = [(u(900 + sd, 7) * 2.0 - 1.0) * PI, (u(900 + sd, 8) * 2.0 - 1.0) * PI, (u(900 + sd, 9) * 2.0 - 1.0) * PI, 0.0, 0.0, 0.0];
                for _ in 0..300 { let a = act(s, g, kk); s = step(s, a); }
                n += 1; if wrap(s[0] - g.0).abs() < 0.35 && wrap(s[1] - g.1).abs() < 0.35 && wrap(s[2] - g.2).abs() < 0.35 && s[3].abs() < 0.7 && s[4].abs() < 0.7 && s[5].abs() < 0.7 { reach += 1; } } }
            println!("  ACTUATE (K={kk}, 120 episodes × 4 goal-triples): reach {:>4.0}%", reach as f32 / n as f32 * 100.0); }
        println!("\n  3-DOF flow policy at K forward passes vs a discrete planner's 125–152 evals/decision. Loaded from disk.");
    } else { println!("  unknown architecture: {arch}"); }
}
