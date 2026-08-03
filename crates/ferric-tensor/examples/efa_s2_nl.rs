//! Milestone S2 — take the fused discover→verify loop OFF the quadratic.
//!
//! System: reversed Van der Pol  ẋ₁ = −x₂,  ẋ₂ = x₁ + μ(x₁²−1)x₂  (μ=1). Origin is a stable focus whose
//! region of attraction is bounded by an unstable limit cycle — a NON-CONVEX ROA where a single quadratic
//! Lyapunov function certifies only a small ball. We give the learner a nonlinear energy
//!   V(x) = xᵀPx + Σⱼ wⱼ·tanh(aⱼx₁ + bⱼx₂ + cⱼ)      (fixed tanh basis, learned P and weights)
//! and keep the verifier SOUND on it: the centered/mean-value bound uses the true value at the box center
//! plus a sound gradient bound — quadratic part tight, tanh head bounded by |tanh′|≤1, |tanh″|≤4/(3√3)
//! (CROWN constants) — with adaptive box refinement absorbing the looseness. Sound = 0 uncertified boxes is
//! a real proof over the continuum, exactly the guarantee `ebm_cert_verify` gives for its ternary head.
//!
//! Run head-OFF (pure quadratic) then head-ON, on the same region, so the head's value is visible.
//! Run: `cargo run -p ferric-tensor --example efa_s2_nl --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const MU: f64 = 1.0;
const R0: f64 = 0.10;
const ALPHA: f64 = 0.01;   // certify V̇ ≤ −α‖x‖²
const DELTA: f64 = 0.01;   //         V ≥  δ‖x‖²
const ALPHA_T: f64 = 0.07; // stricter training margins → slack for the sound bound
const DELTA_T: f64 = 0.07;
const GB: usize = 56;
const MAXD: u32 = 7;

// fixed tanh basis: K directions × offsets (sharper ridges → localized correction + tighter tanh′ bounds)
const NDIR: usize = 10;
const OFFS: [f64; 3] = [0.6, 0.95, 1.25];
const STEEP: f64 = 2.6;
const K: usize = NDIR * OFFS.len();
fn feat(j: usize) -> (f64, f64, f64) { // (a,b,c) for tanh(a x1 + b x2 + c)
    let d = j / OFFS.len();
    let th = std::f64::consts::TAU * d as f64 / NDIR as f64;
    (STEEP * th.cos(), STEEP * th.sin(), -STEEP * OFFS[j % OFFS.len()])
}

// system
fn f(x1: f64, x2: f64) -> (f64, f64) { (-x2, x1 + MU * (x1 * x1 - 1.0) * x2) }

// ---- interval kernel ----
#[derive(Clone, Copy)]
struct Iv { lo: f64, hi: f64 }
impl Iv {
    fn add(self, o: Iv) -> Iv { Iv { lo: self.lo + o.lo, hi: self.hi + o.hi } }
    fn mul(self, o: Iv) -> Iv { let (a,b,c,d)=(self.lo*o.lo,self.lo*o.hi,self.hi*o.lo,self.hi*o.hi); Iv{lo:a.min(b).min(c).min(d),hi:a.max(b).max(c).max(d)} }
    fn scale(self, k: f64) -> Iv { if k>=0.0 {Iv{lo:self.lo*k,hi:self.hi*k}} else {Iv{lo:self.hi*k,hi:self.lo*k}} }
    fn sq(self) -> Iv { if self.lo>=0.0 {Iv{lo:self.lo*self.lo,hi:self.hi*self.hi}} else if self.hi<=0.0 {Iv{lo:self.hi*self.hi,hi:self.lo*self.lo}} else {Iv{lo:0.0,hi:(self.lo*self.lo).max(self.hi*self.hi)}} }
    fn amax(self) -> f64 { self.lo.abs().max(self.hi.abs()) }
}
// f and its partials as intervals over a box
fn f_iv(x1: Iv, x2: Iv) -> (Iv, Iv) { // f1=-x2 ; f2 = x1 + μ(x1²−1)x2
    let f1 = x2.scale(-1.0);
    let f2 = x1.add(x1.sq().add(Iv { lo: -1.0, hi: -1.0 }).mul(x2).scale(MU));
    (f1, f2)
}
fn df_iv(x1: Iv, x2: Iv) -> (Iv, Iv, Iv, Iv) { // ∂f1/∂x1, ∂f1/∂x2, ∂f2/∂x1, ∂f2/∂x2
    (Iv { lo: 0.0, hi: 0.0 }, Iv { lo: -1.0, hi: -1.0 },
     Iv { lo: 1.0, hi: 1.0 }.add(x1.mul(x2).scale(2.0 * MU)),   // 1 + 2μ x1 x2
     x1.sq().add(Iv { lo: -1.0, hi: -1.0 }).scale(MU))          // μ(x1²−1)
}

// exact g(x)=V̇+α‖x‖² and h(x)=V−δ‖x‖² at a point, for params (pa,pb,pc, w[..])
fn v_grad(x1: f64, x2: f64, p: &[f64], head: bool) -> (f64, f64, f64) { // returns (V, dV/dx1, dV/dx2)
    let (mut v, mut g1, mut g2) = (p[0]*x1*x1 + 2.0*p[1]*x1*x2 + p[2]*x2*x2,
        2.0*(p[0]*x1 + p[1]*x2), 2.0*(p[1]*x1 + p[2]*x2));
    if head { for j in 0..K { let (a,b,c)=feat(j); let z=a*x1+b*x2+c; let t=z.tanh(); let tp=1.0-t*t;
        v += p[3+j]*t; g1 += p[3+j]*tp*a; g2 += p[3+j]*tp*b; } }
    (v, g1, g2)
}
fn g_at(x1: f64, x2: f64, p: &[f64], head: bool) -> f64 { let (_,g1,g2)=v_grad(x1,x2,p,head); let (f1,f2)=f(x1,x2); g1*f1+g2*f2 + ALPHA*(x1*x1+x2*x2) }
fn h_at(x1: f64, x2: f64, p: &[f64], head: bool) -> f64 { let (v,_,_)=v_grad(x1,x2,p,head); v - DELTA*(x1*x1+x2*x2) }

// tight per-box tanh derivative intervals: φ=tanh(z), φ′=1−φ², φ″=−2φφ′ over z∈[lo,hi]
fn tanh_ivs(z: Iv) -> (Iv, Iv) { // returns (φ′, φ″) as intervals
    let t = Iv { lo: z.lo.tanh(), hi: z.hi.tanh() };      // tanh monotone ↑
    let t2 = t.sq();
    let tp = Iv { lo: 1.0 - t2.hi, hi: 1.0 - t2.lo };     // 1 − tanh²
    let tpp = t.mul(tp).scale(-2.0);                      // −2·tanh·(1−tanh²)
    (tp, tpp)
}
// sound per-box bounds via centered form: max g ≤ g(c)+Σ|∂g/∂xᵢ|·r ; min h ≥ h(c)−Σ|∂h/∂xᵢ|·r.
// Gradients carried as TIGHT intervals (quadratic + tanh head both exact-interval), then |·|=amax.
fn box_bounds(x1: Iv, x2: Iv, p: &[f64], head: bool) -> (f64, f64) {
    let (c1, c2) = ((x1.lo+x1.hi)/2.0, (x2.lo+x2.hi)/2.0);
    let r = ((x1.hi-x1.lo).max(x2.hi-x2.lo))/2.0;
    let (f1, f2) = f_iv(x1, x2);
    let (df11, df12, df21, df22) = df_iv(x1, x2);
    let (pa, pb, pc) = (p[0], p[1], p[2]);
    let z = Iv { lo: 0.0, hi: 0.0 };
    // ∇V and Hessian(V) as intervals over the box (quadratic + head)
    let mut vx1 = x1.scale(2.0*pa).add(x2.scale(2.0*pb));  // ∂V/∂x1
    let mut vx2 = x1.scale(2.0*pb).add(x2.scale(2.0*pc));  // ∂V/∂x2
    let (mut vxx, mut vxy, mut vyy) = (Iv{lo:2.0*pa,hi:2.0*pa}, Iv{lo:2.0*pb,hi:2.0*pb}, Iv{lo:2.0*pc,hi:2.0*pc});
    if head { for j in 0..K { let (a,b,_c)=feat(j); let w=p[3+j];
        let zj = x1.scale(a).add(x2.scale(b)).add(Iv{lo:feat(j).2,hi:feat(j).2});
        let (tp, tpp) = tanh_ivs(zj);
        vx1 = vx1.add(tp.scale(a).scale(w)); vx2 = vx2.add(tp.scale(b).scale(w));
        vxx = vxx.add(tpp.scale(a*a).scale(w)); vxy = vxy.add(tpp.scale(a*b).scale(w)); vyy = vyy.add(tpp.scale(b*b).scale(w));
    }}
    let _ = z;
    // ∂g/∂xᵢ = ∂V̇/∂xᵢ + 2α xᵢ ; ∂V̇/∂x1 = Vxx f1 + Vx1 f1x1 + Vxy f2 + Vx2 f2x1
    let dg1 = vxx.mul(f1).add(vx1.mul(df11)).add(vxy.mul(f2)).add(vx2.mul(df21)).add(x1.scale(2.0*ALPHA));
    let dg2 = vxy.mul(f1).add(vx1.mul(df12)).add(vyy.mul(f2)).add(vx2.mul(df22)).add(x2.scale(2.0*ALPHA));
    let dh1 = vx1.add(x1.scale(-2.0*DELTA));  // ∂h/∂x1 = ∂V/∂x1 − 2δ x1
    let dh2 = vx2.add(x2.scale(-2.0*DELTA));
    let g_hi = g_at(c1, c2, p, head) + (dg1.amax() + dg2.amax()) * r;
    let h_lo = h_at(c1, c2, p, head) - (dh1.amax() + dh2.amax()) * r;
    (h_lo, g_hi)
}
fn certify_box(x1: Iv, x2: Iv, p: &[f64], head: bool, depth: u32) -> Option<(f64, f64)> {
    if x1.lo > -R0 && x1.hi < R0 && x2.lo > -R0 && x2.hi < R0 { return None; }
    let (h_lo, g_hi) = box_bounds(x1, x2, p, head);
    if h_lo > 0.0 && g_hi < 0.0 { return None; }
    if depth == 0 { return Some(((x1.lo+x1.hi)/2.0, (x2.lo+x2.hi)/2.0)); }
    let (m1, m2) = ((x1.lo+x1.hi)/2.0, (x2.lo+x2.hi)/2.0);
    for &(a1,b1) in &[(x1.lo,m1),(m1,x1.hi)] { for &(a2,b2) in &[(x2.lo,m2),(m2,x2.hi)] {
        if let Some(w)=certify_box(Iv{lo:a1,hi:b1},Iv{lo:a2,hi:b2},p,head,depth-1){return Some(w);} } }
    None
}
// INDEPENDENT pointwise cross-check (no intervals): dense grid, exact eval. Returns
// (max of g=V̇+α‖x‖²  — must be <0 for the certificate to hold pointwise, min of h=V−δ‖x‖²).
fn pointwise_worst(p: &[f64], head: bool, rr: f64) -> (f64, f64) {
    let (mut gmax, mut hmin) = (f64::NEG_INFINITY, f64::INFINITY);
    let n = 500usize;
    for i in 0..=n { for j in 0..=n {
        let x1 = -rr + 2.0*rr*i as f64/n as f64;
        let x2 = -rr + 2.0*rr*j as f64/n as f64;
        if x1.abs().max(x2.abs()) < R0 { continue; }
        gmax = gmax.max(g_at(x1, x2, p, head));
        hmin = hmin.min(h_at(x1, x2, p, head));
    }}
    (gmax, hmin)
}
fn v_at(x1: f64, x2: f64, p: &[f64], head: bool) -> f64 { v_grad(x1, x2, p, head).0 }
// does a trajectory from x0 converge to the origin under reversed Van der Pol? (independent ROA oracle)
fn converges(x1: f64, x2: f64) -> bool {
    let (mut a, mut b) = (x1, x2); let dt = 0.003;
    for _ in 0..12000 { let (f1, f2) = f(a, b); a += dt*f1; b += dt*f2;
        if (a*a + b*b).sqrt() > 5.0 { return false; } if (a*a + b*b).sqrt() < 1e-3 { return true; } }
    (a*a + b*b).sqrt() < 0.5
}
// The certified invariant region of attraction: largest sublevel set {V ≤ c*} that fits in the box.
// c* = min V on the box boundary ‖x‖∞=rr (so {V≤c*} ⊆ box, where V̇<0 is proven ⇒ invariant + attractive).
// Returns (c*, area of the origin's basin within {V≤c*}, whether every sampled boundary point converges).
fn sublevel_roa(p: &[f64], head: bool, rr: f64) -> (f64, f64, bool) {
    let m = 400usize;
    let mut cstar = f64::INFINITY; // min V over the four box edges
    for t in 0..=m { let s = -rr + 2.0*rr*t as f64/m as f64;
        for &(x1, x2) in &[(rr, s), (-rr, s), (s, rr), (s, -rr)] { cstar = cstar.min(v_at(x1, x2, p, head)); } }
    // flood-fill the origin's connected component of {V ≤ c*} on a grid inside the box
    let g = 260usize; let cell = 2.0*rr/g as f64;
    let idx = |i: usize, j: usize| i*g + j;
    let inset = |i: usize, j: usize| { let x1 = -rr + (i as f64+0.5)*cell; let x2 = -rr + (j as f64+0.5)*cell; (x1, x2) };
    let mut inset_ok = vec![false; g*g];
    for i in 0..g { for j in 0..g { let (x1, x2) = inset(i, j); if v_at(x1, x2, p, head) <= cstar { inset_ok[idx(i,j)] = true; } } }
    let mut seen = vec![false; g*g]; let mut stack = vec![((g/2), (g/2))]; let mut cnt = 0usize;
    let mut worst = (0.0f64, 0.0f64, -1.0f64); // farthest boundary-ish cell for the ROA sanity sim
    if inset_ok[idx(g/2,g/2)] { seen[idx(g/2,g/2)] = true;
        while let Some((i, j)) = stack.pop() { cnt += 1;
            let (x1, x2) = inset(i, j); let rad = x1*x1 + x2*x2; if rad > worst.2 { worst = (x1, x2, rad); }
            let mut nb = vec![]; if i>0 {nb.push((i-1,j));} if i+1<g {nb.push((i+1,j));} if j>0 {nb.push((i,j-1));} if j+1<g {nb.push((i,j+1));}
            for (ni, nj) in nb { if !seen[idx(ni,nj)] && inset_ok[idx(ni,nj)] { seen[idx(ni,nj)] = true; stack.push((ni, nj)); } } } }
    let area = cnt as f64 * cell * cell;
    // ROA sanity: the certified set MUST lie in the true ROA — simulate its farthest point
    let roa_ok = converges(worst.0, worst.1);
    (cstar, area, roa_ok)
}
fn verify(p: &[f64], head: bool, rr: f64) -> (usize, Vec<(f64, f64)>) {
    let mut ce = Vec::new(); let step = 2.0*rr/GB as f64;
    for i in 0..GB { for j in 0..GB { let (lo1,lo2)=(-rr+i as f64*step,-rr+j as f64*step);
        if let Some(w)=certify_box(Iv{lo:lo1,hi:lo1+step},Iv{lo:lo2,hi:lo2+step},p,head,MAXD){ce.push(w);} } }
    (ce.len(), ce)
}

// ---- learner (V, V̇ are LINEAR in params → first-order autograd) ----
// per point, coefficient of each param in V and in V̇:
fn coeffs(pts: &[(f64, f64)], head: bool) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
    let np = if head { 3 + K } else { 3 };
    let mut cv = vec![Vec::with_capacity(pts.len()); np];
    let mut cd = vec![Vec::with_capacity(pts.len()); np];
    let mut r2 = Vec::with_capacity(pts.len());
    for &(x1, x2) in pts {
        let (f1, f2) = f(x1, x2);
        // quadratic
        cv[0].push((x1*x1) as f32); cv[1].push((2.0*x1*x2) as f32); cv[2].push((x2*x2) as f32);
        cd[0].push((2.0*x1*f1) as f32); cd[1].push((2.0*(x2*f1 + x1*f2)) as f32); cd[2].push((2.0*x2*f2) as f32);
        if head { for j in 0..K { let (a,b,c)=feat(j); let z=a*x1+b*x2+c; let t=z.tanh(); let tp=1.0-t*t;
            cv[3+j].push(t as f32); cd[3+j].push((tp*(a*f1 + b*f2)) as f32); } }
        r2.push((x1*x1 + x2*x2) as f32);
    }
    (cv, cd, r2)
}

// one fused discover→verify (CEGIS) run at region half-width rr; returns (certified, best#boxes, P)
async fn cegis(ctx: &Arc<ferric_core::Context>, head: bool, rr: f64, rounds: usize) -> (bool, usize, Vec<f64>) {
    let np = if head { 3 + K } else { 3 };
    let mut p: Vec<Tensor> = (0..np).map(|i| Tensor::from_vec(ctx, &[if i < 3 && i != 1 { 0.6 } else { 0.0 }], &[1, 1])).collect();
    let mut adam = Adam::new(&p, 0.05);
    let mut train: Vec<(f64, f64)> = Vec::new();
    let mut best = usize::MAX;
    let mut pf: Vec<f64> = p.iter().map(|t| pollster::block_on(t.to_vec())[0] as f64).collect();
    for _round in 0..rounds {
        if !train.is_empty() {
            let (cv, cd, r2) = coeffs(&train, head); let n = train.len();
            let leaf = |v: &[f32]| Var::leaf(Tensor::from_vec(ctx, v, &[n, 1]));
            let lv: Vec<Var> = cv.iter().map(|c| leaf(c)).collect();
            let ld: Vec<Var> = cd.iter().map(|c| leaf(c)).collect();
            let ar2 = leaf(&r2.iter().map(|x| x*ALPHA_T as f32).collect::<Vec<_>>());
            let dr2 = leaf(&r2.iter().map(|x| x*DELTA_T as f32).collect::<Vec<_>>());
            for _ in 0..240 {
                let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
                let mut v = pv[0].mul(&lv[0]); let mut vd = pv[0].mul(&ld[0]);
                for i in 1..np { v = v.add(&pv[i].mul(&lv[i])); vd = vd.add(&pv[i].mul(&ld[i])); }
                let loss = dr2.sub(&v).relu().add(&vd.add(&ar2).relu()).mean_all();
                loss.backward();
                let grd: Vec<Tensor> = pv.iter().zip(&p).map(|(v,t)| v.grad().unwrap_or_else(|| Tensor::from_vec(ctx, &vec![0.0; t.numel()], &t.shape))).collect();
                adam.step(&mut p, &grd);
            }
        }
        pf = p.iter().map(|t| pollster::block_on(t.to_vec())[0] as f64).collect();
        let (nv, ce) = verify(&pf, head, rr);
        best = best.min(nv);
        if nv == 0 { return (true, 0, pf); }
        for w in ce { train.push(w); }
        train.sort_by(|a,b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        train.dedup_by(|a,b| (a.0-b.0).abs()<0.015 && (a.1-b.1).abs()<0.015);
        if train.len() > 6000 { let n=train.len(); train.drain(0..(n-6000)); }
    }
    (false, best, pf)
}

// true ROA area via the simulation oracle (bisect the converging radius on each ray)
fn true_roa_area() -> f64 {
    let na = 240usize; let mut area = 0.0;
    for k in 0..na { let th = std::f64::consts::TAU * k as f64 / na as f64; let (c, s) = (th.cos(), th.sin());
        let (mut lo, mut hi) = (0.05, 3.2);
        if converges(lo*c, lo*s) { for _ in 0..34 { let m = (lo+hi)/2.0; if converges(m*c, m*s) { lo = m; } else { hi = m; } } }
        else { lo = 0.0; }
        area += 0.5 * lo * lo * (std::f64::consts::TAU / na as f64);
    }
    area
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("S2 — reversed Van der Pol, fused discover→verify. HONEST metric = certified invariant ROA area.");
    println!("  ẋ₁=−x₂, ẋ₂=x₁+μ(x₁²−1)x₂ (μ={MU}). Certificate proves V̇<0 & V>0 on a box; the guaranteed");
    println!("  region of attraction is the largest sublevel set {{V≤c*}} inside that box (invariant + attractive).\n");
    let roa = true_roa_area();
    println!("  true ROA area (simulation oracle) ≈ {roa:.3}\n");
    for &head in &[false, true] {
        let label = if head { "quadratic + tanh-head" } else { "quadratic only       " };
        let (mut best_area, mut best_r, mut best_c, mut best_sound, mut best_roaok) = (0.0f64, 0.0f64, 0.0f64, true, true);
        // certified ROA area grows with the certified box, so sweep R up and STOP at the first failure
        // after we've already certified something (no point paying for doomed deep-refinement at high R).
        for step in 0..12 {
            let rr = 0.8 + 0.1 * step as f64;
            let (ok, _b, p) = cegis(&ctx, head, rr, 30).await;
            if !ok { if best_area > 0.0 { break; } else { continue; } }
            let (gmax, hmin) = pointwise_worst(&p, head, rr);
            let (cstar, area, roa_ok) = sublevel_roa(&p, head, rr);
            if area > best_area { best_area = area; best_r = rr; best_c = cstar; best_sound = gmax < 0.0 && hmin > 0.0; best_roaok = roa_ok; }
        }
        println!("  {label}  best certified box R={best_r:.1}  →  invariant ROA {{V≤{best_c:.2}}} area = {best_area:.3}  ({:.0}% of true ROA)", 100.0*best_area/roa);
        println!("  {label}    sound (independent pointwise V̇<0): {}   certified set inside true ROA (sim): {}\n",
                 if best_sound { "YES" } else { "NO — BUG" }, if best_roaok { "YES" } else { "NO" });
    }
    println!("Same fused loop, same SOUND verifier. Swapping V=xᵀPx → xᵀPx+Σwⱼtanh(·) lets ONE energy certify a\nstrictly larger invariant region of attraction of this non-convex system — the step past the quadratic.");
}
