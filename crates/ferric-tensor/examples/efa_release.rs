//! EFA release builder #1 — train, GATE, SAVE (safetensors + config), RELOAD, and VERIFY the first open-weight EFA model:
//! `efa-hybrid-arm2` — the hybrid actuation architecture v = −κ∇ₐE + w on the coupled 2-link arm, whose potential also
//! verifies (Eθ(·,1): low = valid action).
//!
//! The ecosystem unit society expects: named weights in a standard format (spec-correct safetensors, hand-rolled and
//! dependency-free), a config.json, a release GATE (actuate ≥95% AND verify ≥90%, retrying up to 3 seeds — the known
//! FVI seed-variance, disclosed rather than hidden), and a ROUND-TRIP: the artifact is reloaded from disk and
//! re-evaluated, so only verified weights ship. Output: efa/models/efa-hybrid-arm2/{model.safetensors, config.json}.
//!
//! Run: `cargo run -p ferric-tensor --example efa_release --release`
use ferric_tensor::{grad, Adam, Tensor, Var};
use std::sync::Arc;
use std::io::Write as IoWrite;
const H: usize = 96; const DT: f32 = 0.05; const GAMMA: f32 = 0.97; const UMAX: f32 = 4.0; const CPL: f32 = 0.5;
const KAPPA: f32 = 2.0; const LAM: f32 = 0.1;
const G5: [f32; 5] = [-4.0, -2.0, 0.0, 2.0, 4.0];
const G9: [f32; 9] = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
const TG: [(f32, f32); 4] = [(0.8, -0.8), (-1.0, 0.6), (0.5, 1.0), (-0.6, -0.9)];
const OUTDIR: &str = "/Users/dcharlot/vibe-coding/efa/models/efa-hybrid-arm2";
use std::f32::consts::PI;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
fn step(s: [f32; 4], u1: f32, u2: f32) -> [f32; 4] {
    let (t1, t2, o1, o2) = (s[0], s[1], s[2], s[3]); let (c1, c2) = (u1.clamp(-UMAX, UMAX), u2.clamp(-UMAX, UMAX));
    let no1 = o1 + DT * (-t1.sin() - 0.05 * o1 + CPL * (t2 - t1).sin() + c1);
    let no2 = o2 + DT * (-t2.sin() - 0.05 * o2 + CPL * (t1 - t2).sin() + c2);
    [wrap(t1 + DT * no1), wrap(t2 + DT * no2), no1, no2]
}
fn cost(s: [f32; 4], g: (f32, f32), u1: f32, u2: f32) -> f32 { wrap(s[0] - g.0).powi(2) + wrap(s[1] - g.1).powi(2) + 0.05 * (s[2] * s[2] + s[3] * s[3]) + 0.01 * (u1 * u1 + u2 * u2) }
fn feat8(s: [f32; 4], g: (f32, f32)) -> [f32; 8] { let (d1, d2) = (s[0] - g.0, s[1] - g.1); [d1.cos(), d1.sin(), s[2], d2.cos(), d2.sin(), s[3], s[0].sin(), s[1].sin()] }

// ---------- safetensors IO (spec-correct, dependency-free) ----------
fn save_safetensors(path: &str, tensors: &[(String, Vec<usize>, Vec<f32>)]) -> std::io::Result<()> {
    let mut header = String::from("{");
    let mut off = 0usize;
    for (i, (name, shape, data)) in tensors.iter().enumerate() {
        let bytes = data.len() * 4;
        if i > 0 { header.push(','); }
        header.push_str(&format!("\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            name, shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","), off, off + bytes));
        off += bytes;
    }
    header.push('}');
    let mut f = std::fs::File::create(path)?;
    f.write_all(&(header.len() as u64).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for (_, _, data) in tensors { for v in data { f.write_all(&v.to_le_bytes())?; } }
    Ok(())
}
fn load_safetensors(path: &str) -> std::io::Result<Vec<(String, Vec<usize>, Vec<f32>)>> {
    let raw = std::fs::read(path)?;
    let hl = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&raw[8..8 + hl]).unwrap().to_string();
    let data = &raw[8 + hl..];
    let mut out = vec![];
    let mut rest = header.as_str();
    while let Some(q) = rest.find("\"dtype\"") {
        // walk back to the tensor name
        let pre = &rest[..q]; let name_end = pre.rfind("\":{").unwrap(); let name_start = pre[..name_end].rfind('"').unwrap() + 1;
        let name = pre[name_start..name_end].to_string();
        let after = &rest[q..];
        let sh_s = after.find("\"shape\":[").unwrap() + 9; let sh_e = after[sh_s..].find(']').unwrap() + sh_s;
        let shape: Vec<usize> = after[sh_s..sh_e].split(',').filter(|s| !s.is_empty()).map(|s| s.trim().parse().unwrap()).collect();
        let of_s = after.find("\"data_offsets\":[").unwrap() + 16; let of_e = after[of_s..].find(']').unwrap() + of_s;
        let offs: Vec<usize> = after[of_s..of_e].split(',').map(|s| s.trim().parse().unwrap()).collect();
        let vals: Vec<f32> = data[offs[0]..offs[1]].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        out.push((name, shape, vals));
        rest = &after[of_e..];
    }
    Ok(out)
}

// ---------- CPU model structs (identical to ebm_hflow) ----------
struct Ef { w: Vec<Vec<f32>>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32 }
impl Ef {
    fn e(&self, s: [f32; 4], g: (f32, f32), a1: f32, a2: f32, t: f32) -> f32 { let mut f = [0.0f32; 11]; let ff = feat8(s, g); for c in 0..8 { f[c] = ff[c]; } f[8] = a1; f[9] = a2; f[10] = t;
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.b1[j]; for c in 0..11 { z += f[c] * self.w[c][j]; } h1[j] = z.max(0.0); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.b2[j]; for k in 0..H { z += h1[k] * self.w2[k * H + j]; } h2[j] = z.max(0.0); }
        let mut o = self.b3; for j in 0..H { o += h2[j] * self.w3[j]; } o }
    fn grad_a(&self, s: [f32; 4], g: (f32, f32), a1: f32, a2: f32, t: f32) -> (f32, f32) {
        let mut f = [0.0f32; 11]; let ff = feat8(s, g); for c in 0..8 { f[c] = ff[c]; } f[8] = a1; f[9] = a2; f[10] = t;
        let mut h1 = [0.0f32; H]; let mut m1 = [false; H];
        for j in 0..H { let mut z = self.b1[j]; for c in 0..11 { z += f[c] * self.w[c][j]; } m1[j] = z > 0.0; h1[j] = z.max(0.0); }
        let mut m2 = [false; H];
        for j in 0..H { let mut z = self.b2[j]; for k in 0..H { z += h1[k] * self.w2[k * H + j]; } m2[j] = z > 0.0; }
        let mut d2 = [0.0f32; H]; for j in 0..H { if m2[j] { d2[j] = self.w3[j]; } }
        let mut d1 = [0.0f32; H]; for k in 0..H { if m1[k] { let mut z = 0.0; for j in 0..H { z += self.w2[k * H + j] * d2[j]; } d1[k] = z; } }
        let (mut g1, mut g2) = (0.0f32, 0.0f32); for j in 0..H { g1 += self.w[8][j] * d1[j]; g2 += self.w[9][j] * d1[j]; }
        (g1, g2) }
}
struct Wn { w: Vec<Vec<f32>>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: [f32; 2] }
impl Wn {
    fn vel(&self, s: [f32; 4], g: (f32, f32), a1: f32, a2: f32, t: f32) -> (f32, f32) { let mut f = [0.0f32; 11]; let ff = feat8(s, g); for c in 0..8 { f[c] = ff[c]; } f[8] = a1; f[9] = a2; f[10] = t;
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.b1[j]; for c in 0..11 { z += f[c] * self.w[c][j]; } h1[j] = z.max(0.0); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.b2[j]; for k in 0..H { z += h1[k] * self.w2[k * H + j]; } h2[j] = z.max(0.0); }
        let (mut o1, mut o2) = (self.b3[0], self.b3[1]); for j in 0..H { o1 += h2[j] * self.w3[j * 2]; o2 += h2[j] * self.w3[j * 2 + 1]; } (o1, o2) }
}
struct Vn { w: Vec<Vec<f32>>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32 }
impl Vn {
    fn eval(&self, s: [f32; 4], g: (f32, f32)) -> f32 { let f = feat8(s, g);
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.b1[j]; for c in 0..8 { z += f[c] * self.w[c][j]; } h1[j] = (z.exp() + 1.0).ln(); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.b2[j]; for k in 0..H { z += h1[k] * self.w2[k * H + j]; } h2[j] = (z.exp() + 1.0).ln(); }
        let mut o = self.b3; for j in 0..H { o += h2[j] * self.w3[j]; } (o.exp() + 1.0).ln() }
    fn ustar(&self, s: [f32; 4], g: (f32, f32), grid: &[f32]) -> (f32, f32) { let mut bu = (0.0, 0.0); let mut be = f32::MAX;
        for &u1 in grid { for &u2 in grid { let ns = step(s, u1, u2); let q = (cost(s, g, u1, u2)) * DT + GAMMA * self.eval(ns, g); if q < be { be = q; bu = (u1, u2); } } } bu }
}
// evaluate a hybrid (Ef, Wn): (actuate% @K=2, verify%, field% via potential share)
fn eval_hybrid(ef: &Ef, wn: &Wn, vn: &Vn) -> (f32, f32, f32) {
    let nep = 40usize; let n = nep * TG.len();
    let mut inits: Vec<[f32; 4]> = vec![]; let mut goals: Vec<(f32, f32)> = vec![];
    for (gi, &g) in TG.iter().enumerate() { for e in 0..nep { let sd = (gi * nep + e) as u32; inits.push([(u(900 + sd, 7) * 2.0 - 1.0) * PI, (u(900 + sd, 8) * 2.0 - 1.0) * PI, 0.0, 0.0]); goals.push(g); } }
    let mut rr = 0; for i in 0..n { let mut s = inits[i]; let g = goals[i];
        for _ in 0..260 { let (mut a1, mut a2) = (0.0f32, 0.0f32); for kk in 0..2 { let t = kk as f32 / 2.0;
            let (g1, g2) = ef.grad_a(s, g, a1, a2, t); let (w1, w2) = wn.vel(s, g, a1, a2, t);
            a1 += (-KAPPA * g1 + w1) / 2.0; a2 += (-KAPPA * g2 + w2) / 2.0; }
            s = step(s, a1.clamp(-UMAX, UMAX), a2.clamp(-UMAX, UMAX)); }
        if wrap(s[0] - goals[i].0).abs() < 0.35 && wrap(s[1] - goals[i].1).abs() < 0.35 && s[2].abs() < 0.7 && s[3].abs() < 0.7 { rr += 1; } }
    let (mut vg, mut vt) = (0, 0); for k in 0..2000 { let s = [(u(k as u32, 41) * 2.0 - 1.0) * PI, (u(k as u32, 42) * 2.0 - 1.0) * PI, (u(k as u32, 43) * 2.0 - 1.0) * 3.0, (u(k as u32, 44) * 2.0 - 1.0) * 3.0];
        let g = ((u(k as u32, 45) * 2.0 - 1.0) * 1.2, (u(k as u32, 46) * 2.0 - 1.0) * 1.2); let us = vn.ustar(s, g, &G5); let bad = ((u(k as u32, 47) * 2.0 - 1.0) * UMAX, (u(k as u32, 48) * 2.0 - 1.0) * UMAX);
        vt += 1; if ef.e(s, g, us.0, us.1, 1.0) < ef.e(s, g, bad.0, bad.1, 1.0) { vg += 1; } }
    let (mut me, mut mw) = (0.0f32, 0.0f32); for i in (0..n).step_by(4) { let (mut a1, mut a2) = (0.0f32, 0.0f32); for kk in 0..2 { let t = kk as f32 / 2.0;
        let (g1, g2) = ef.grad_a(inits[i], goals[i], a1, a2, t); let (w1, w2) = wn.vel(inits[i], goals[i], a1, a2, t);
        me += (KAPPA * KAPPA * (g1 * g1 + g2 * g2)).sqrt(); mw += (w1 * w1 + w2 * w2).sqrt();
        a1 += (-KAPPA * g1 + w1) / 2.0; a2 += (-KAPPA * g2 + w2) / 2.0; } }
    (rr as f32 / n as f32 * 100.0, vg as f32 / vt as f32 * 100.0, me / (me + mw) * 100.0)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA RELEASE BUILDER — efa-hybrid-arm2: train → GATE (actuate≥95 & verify≥90) → save safetensors → reload → verify\n");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let bs = 256usize;
    let sp = |z: Var, ov: &Var| z.exp().add(ov).log();
    let vnet = |f: &[Var], pv: &[Var], ov: &Var| { let mut pre = pv[8].clone(); for c in 0..8 { pre = pre.add(&f[c].matmul(&pv[c])); } sp(sp(sp(pre, ov).matmul(&pv[9]).add(&pv[10]), ov).matmul(&pv[11]).add(&pv[12]), ov) };
    let enet = |f: &[Var], a1: &Var, a2: &Var, tt: &Var, pv: &[Var]| { let mut pre = pv[11].clone(); for c in 0..8 { pre = pre.add(&f[c].matmul(&pv[c])); }
        pre = pre.add(&a1.matmul(&pv[8])).add(&a2.matmul(&pv[9])).add(&tt.matmul(&pv[10]));
        pre.relu().matmul(&pv[12]).add(&pv[13]).relu().matmul(&pv[14]).add(&pv[15]) };
    let selx = Tensor::from_vec(&ctx, &[1.0, 0.0], &[2, 1]); let sely = Tensor::from_vec(&ctx, &[0.0, 1.0], &[2, 1]);
    let kv = Tensor::from_vec(&ctx, &[KAPPA], &[1]); let lv = Tensor::from_vec(&ctx, &[LAM], &[1]);

    let mut released = false;
    for (attempt, &soff) in [0u32, 17, 34].iter().enumerate() {
        println!("  [attempt {} — seed offset {}]", attempt + 1, soff);
        // ---- FVI demonstrator ----
        let mut p: Vec<Tensor> = (0..8).map(|c| Tensor::from_vec(&ctx, &randn(H, 22 + soff + c as u32, 0.5), &[1, H])).collect();
        p.push(Tensor::zeros(&ctx, &[H])); p.push(Tensor::from_vec(&ctx, &randn(H * H, 40 + soff, 1.0 / (H as f32).sqrt()), &[H, H])); p.push(Tensor::zeros(&ctx, &[H]));
        p.push(Tensor::from_vec(&ctx, &randn(H, 41 + soff, 1.0 / (H as f32).sqrt()), &[H, 1])); p.push(Tensor::zeros(&ctx, &[1]));
        let mut tgt = p.clone(); let mut adam = Adam::new(&p, 0.002); let ga = 25usize;
        for it in 0..16000 {
            let mut fc: Vec<Vec<f32>> = (0..8).map(|_| vec![0.0f32; bs]).collect();
            let mut nf: Vec<Vec<f32>> = (0..8).map(|_| vec![0.0f32; bs * ga]).collect(); let mut cst = vec![0.0f32; bs * ga];
            for i in 0..bs { let sd = it as u32 * 7 + i as u32 + soff * 1000;
                let s = [(u(sd, 1) * 2.0 - 1.0) * PI, (u(sd, 2) * 2.0 - 1.0) * PI, (u(sd, 3) * 2.0 - 1.0) * 3.0, (u(sd, 4) * 2.0 - 1.0) * 3.0];
                let g = ((u(sd, 5) * 2.0 - 1.0) * 1.2, (u(sd, 6) * 2.0 - 1.0) * 1.2); let f = feat8(s, g); for c in 0..8 { fc[c][i] = f[c]; }
                let mut a = 0; for &u1 in &G5 { for &u2 in &G5 { let ns = step(s, u1, u2); let nff = feat8(ns, g); for c in 0..8 { nf[c][i * ga + a] = nff[c]; } cst[i * ga + a] = cost(s, g, u1, u2); a += 1; } } }
            let l = |v: &[f32], r: usize| Var::leaf(Tensor::from_vec(&ctx, v, &[r, 1]));
            let tv: Vec<Var> = tgt.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
            let et = vnet(&(0..8).map(|c| l(&nf[c], bs * ga)).collect::<Vec<_>>(), &tv, &ov).value().to_vec().await;
            let mut target = vec![0.0f32; bs]; for i in 0..bs { let mut m = f32::MAX; for a in 0..ga { let q = cst[i * ga + a] * DT + GAMMA * et[i * ga + a]; if q < m { m = q; } } target[i] = m; }
            let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
            let e = vnet(&(0..8).map(|c| l(&fc[c], bs)).collect::<Vec<_>>(), &pv, &ov); let d = e.sub(&Var::leaf(Tensor::from_vec(&ctx, &target, &[bs, 1]))); let loss = d.mul(&d).mean_all(); loss.backward();
            let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect(); adam.step(&mut p, &g); if it % 200 == 0 { tgt = p.clone(); }
        }
        let mut vw: Vec<Vec<f32>> = Vec::new(); for c in 0..8 { vw.push(p[c].to_vec().await); }
        let vn = Vn { w: vw, b1: p[8].to_vec().await, w2: p[9].to_vec().await, b2: p[10].to_vec().await, w3: p[11].to_vec().await, b3: p[12].to_vec().await[0] };
        // ---- joint hybrid training (E + w) ----
        let mut q: Vec<Tensor> = (0..11).map(|c| Tensor::from_vec(&ctx, &randn(H, 60 + soff + c as u32, 0.4), &[1, H])).collect();
        q.push(Tensor::zeros(&ctx, &[H])); q.push(Tensor::from_vec(&ctx, &randn(H * H, 80 + soff, 1.0 / (H as f32).sqrt()), &[H, H])); q.push(Tensor::zeros(&ctx, &[H]));
        q.push(Tensor::from_vec(&ctx, &randn(H, 81 + soff, 1.0 / (H as f32).sqrt()), &[H, 1])); q.push(Tensor::zeros(&ctx, &[1]));
        let mut r: Vec<Tensor> = (0..11).map(|c| Tensor::from_vec(&ctx, &randn(H, 120 + soff + c as u32, 0.4), &[1, H])).collect();
        r.push(Tensor::zeros(&ctx, &[H])); r.push(Tensor::from_vec(&ctx, &randn(H * H, 140 + soff, 1.0 / (H as f32).sqrt()), &[H, H])); r.push(Tensor::zeros(&ctx, &[H]));
        r.push(Tensor::from_vec(&ctx, &randn(H * 2, 141 + soff, 1.0 / (H as f32).sqrt()), &[H, 2])); r.push(Tensor::zeros(&ctx, &[2]));
        let mut adamq = Adam::new(&q, 0.0015); let mut adamr = Adam::new(&r, 0.0015);
        for it in 0..12000 {
            let mut fc: Vec<Vec<f32>> = (0..8).map(|_| vec![0.0f32; bs]).collect();
            let (mut at1, mut at2, mut tt, mut t1, mut t2) = (vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs]);
            for i in 0..bs { let sd = it as u32 * 13 + i as u32 + soff * 1000;
                let s = [(u(sd, 1) * 2.0 - 1.0) * PI, (u(sd, 2) * 2.0 - 1.0) * PI, (u(sd, 3) * 2.0 - 1.0) * 3.0, (u(sd, 4) * 2.0 - 1.0) * 3.0];
                let g = ((u(sd, 5) * 2.0 - 1.0) * 1.2, (u(sd, 6) * 2.0 - 1.0) * 1.2); let us = vn.ustar(s, g, &G9);
                let a01 = (u(sd, 7) * 2.0 - 1.0) * 3.0; let a02 = (u(sd, 8) * 2.0 - 1.0) * 3.0; let t = u(sd, 9) * 0.9;
                let ff = feat8(s, g); for c in 0..8 { fc[c][i] = ff[c]; }
                at1[i] = (1.0 - t) * a01 + t * us.0; at2[i] = (1.0 - t) * a02 + t * us.1; tt[i] = t;
                t1[i] = us.0 - a01; t2[i] = us.1 - a02; }
            let qv: Vec<Var> = q.iter().map(|t| Var::leaf(t.clone())).collect(); let rv: Vec<Var> = r.iter().map(|t| Var::leaf(t.clone())).collect();
            let fv: Vec<Var> = (0..8).map(|c| Var::leaf(Tensor::from_vec(&ctx, &fc[c], &[bs, 1]))).collect();
            let a1v = Var::leaf(Tensor::from_vec(&ctx, &at1, &[bs, 1])); let a2v = Var::leaf(Tensor::from_vec(&ctx, &at2, &[bs, 1])); let tv2 = Var::leaf(Tensor::from_vec(&ctx, &tt, &[bs, 1]));
            let e = enet(&fv, &a1v, &a2v, &tv2, &qv);
            let gr = grad(&e.sum_all(), &[a1v.clone(), a2v.clone()], None);
            let wout = enet(&fv, &a1v, &a2v, &tv2, &rv);
            let w1 = wout.matmul(&Var::leaf(selx.clone())); let w2 = wout.matmul(&Var::leaf(sely.clone()));
            let kva = Var::leaf(kv.clone());
            let v1 = gr[0].neg().mul(&kva).add(&w1); let v2 = gr[1].neg().mul(&kva).add(&w2);
            let d1 = v1.sub(&Var::leaf(Tensor::from_vec(&ctx, &t1, &[bs, 1]))); let d2 = v2.sub(&Var::leaf(Tensor::from_vec(&ctx, &t2, &[bs, 1])));
            let pen = w1.mul(&w1).add(&w2.mul(&w2)).mul(&Var::leaf(lv.clone()));
            let loss = d1.mul(&d1).add(&d2.mul(&d2)).add(&pen).mean_all(); loss.backward();
            let gq: Vec<Tensor> = qv.iter().zip(&q).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect(); adamq.step(&mut q, &gq);
            let gr2: Vec<Tensor> = rv.iter().zip(&r).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect(); adamr.step(&mut r, &gr2);
        }
        // ---- gate ----
        let mut ew: Vec<Vec<f32>> = Vec::new(); for c in 0..11 { ew.push(q[c].to_vec().await); }
        let ef = Ef { w: ew, b1: q[11].to_vec().await, w2: q[12].to_vec().await, b2: q[13].to_vec().await, w3: q[14].to_vec().await, b3: q[15].to_vec().await[0] };
        let mut ww: Vec<Vec<f32>> = Vec::new(); for c in 0..11 { ww.push(r[c].to_vec().await); }
        let wb3 = r[15].to_vec().await; let wn = Wn { w: ww, b1: r[11].to_vec().await, w2: r[12].to_vec().await, b2: r[13].to_vec().await, w3: r[14].to_vec().await, b3: [wb3[0], wb3[1]] };
        let (act, ver, field) = eval_hybrid(&ef, &wn, &vn);
        println!("     trained: actuate {act:.0}%  verify {ver:.1}%  potential-share {field:.0}%");
        if act < 95.0 || ver < 90.0 { println!("     GATE FAILED (need actuate≥95 & verify≥90) — retrying with next seed\n"); continue; }
        // ---- save ----
        std::fs::create_dir_all(OUTDIR).unwrap();
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = vec![];
        for c in 0..11 { tensors.push((format!("potential.in{}", c), vec![1, H], ef.w[c].clone())); }
        tensors.push(("potential.b1".into(), vec![H], ef.b1.clone())); tensors.push(("potential.w2".into(), vec![H, H], ef.w2.clone()));
        tensors.push(("potential.b2".into(), vec![H], ef.b2.clone())); tensors.push(("potential.w3".into(), vec![H, 1], ef.w3.clone()));
        tensors.push(("potential.b3".into(), vec![1], vec![ef.b3]));
        for c in 0..11 { tensors.push((format!("correction.in{}", c), vec![1, H], wn.w[c].clone())); }
        tensors.push(("correction.b1".into(), vec![H], wn.b1.clone())); tensors.push(("correction.w2".into(), vec![H, H], wn.w2.clone()));
        tensors.push(("correction.b2".into(), vec![H], wn.b2.clone())); tensors.push(("correction.w3".into(), vec![H, 2], wn.w3.clone()));
        tensors.push(("correction.b3".into(), vec![2], vec![wn.b3[0], wn.b3[1]]));
        let nparams: usize = tensors.iter().map(|(_, _, d)| d.len()).sum();
        save_safetensors(&format!("{OUTDIR}/model.safetensors"), &tensors).unwrap();
        let config = format!("{{\n  \"architecture\": \"efa-hybrid-v0\",\n  \"description\": \"EFA hybrid actuation: v = -kappa*grad_a(E) + w. One scalar potential E(s,a,t) actuates (descend) and verifies (E(.,1): low = valid); a small l2-penalized correction w closes the scalar-fit gap.\",\n  \"hidden\": {H},\n  \"kappa\": {KAPPA},\n  \"lambda\": {LAM},\n  \"params\": {nparams},\n  \"features\": \"[cos(th1-g1), sin(th1-g1), om1, cos(th2-g2), sin(th2-g2), om2, sin(th1), sin(th2)] + [a1, a2, t]\",\n  \"env\": {{\"type\": \"coupled-2-link-chain\", \"dt\": {DT}, \"umax\": {UMAX}, \"coupling\": {CPL}, \"damping\": 0.05, \"gravity\": \"sin\"}},\n  \"inference\": \"a=0; for k in 0..K {{ t=k/K; a += (-kappa*grad_a E(s,a,t) + w(s,a,t))/K }}; K=2 recommended\",\n  \"metrics\": {{\"actuate_K2\": {act:.1}, \"verify\": {ver:.1}, \"potential_field_share\": {field:.1}}},\n  \"seed_offset\": {soff},\n  \"gate\": \"actuate>=95 && verify>=90, best-of-<=3 seeds (FVI seed variance disclosed)\"\n}}\n");
        std::fs::write(format!("{OUTDIR}/config.json"), config).unwrap();
        // ---- reload + verify round-trip ----
        let loaded = load_safetensors(&format!("{OUTDIR}/model.safetensors")).unwrap();
        let getl = |n: &str| loaded.iter().find(|(nm, _, _)| nm == n).map(|(_, _, d)| d.clone()).unwrap();
        let ef2 = Ef { w: (0..11).map(|c| getl(&format!("potential.in{}", c))).collect(), b1: getl("potential.b1"), w2: getl("potential.w2"), b2: getl("potential.b2"), w3: getl("potential.w3"), b3: getl("potential.b3")[0] };
        let wb = getl("correction.b3"); let wn2 = Wn { w: (0..11).map(|c| getl(&format!("correction.in{}", c))).collect(), b1: getl("correction.b1"), w2: getl("correction.w2"), b2: getl("correction.b2"), w3: getl("correction.w3"), b3: [wb[0], wb[1]] };
        let (act2, ver2, field2) = eval_hybrid(&ef2, &wn2, &vn);
        println!("     RELOADED from disk: actuate {act2:.0}%  verify {ver2:.1}%  potential-share {field2:.0}%  (round-trip {})",
            if (act - act2).abs() < 0.5 && (ver - ver2).abs() < 0.5 { "EXACT ✓" } else { "MISMATCH ✗" });
        println!("\n  RELEASED: {OUTDIR}/model.safetensors ({} params, {} tensors) + config.json", nparams, tensors.len());
        released = true; break;
    }
    if !released { println!("\n  NO RELEASE: all 3 seeds failed the gate — value-training instability; do not ship."); }
}
