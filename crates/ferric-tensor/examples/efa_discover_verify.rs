//! Milestone 1 — the FUSED discover→verify graph, in ONE deterministic Ferric program.
//!
//! Everyone else runs the learner (a GPU model) and the verifier (an SMT/kernel process) as separate
//! systems with a latency wall between them — every CEGIS step pays a process hop + re-elaboration.
//! Here they are one program: a Lyapunov-function LEARNER (Ferric autograd) and a SOUND interval
//! VERIFIER (dependency-free f64, the trusted kernel) share memory; the certificate violation IS the
//! loss, and the verifier's counterexamples flow straight back into the training batch as data. No
//! process boundary. The certificate is what trains the discoverer.
//!
//! System: damped pendulum  ẋ₁ = x₂,  ẋ₂ = −sin(x₁) − c·x₂.  Origin asymptotically stable.
//! Discover a quadratic Lyapunov function V(x)=xᵀPx, P=[[a,b],[b,c]], certifying the annulus
//! r₀ ≤ ‖x‖∞ ≤ R:   V − δ‖x‖² > 0  (positive)  AND  V̇ + α‖x‖² < 0  (decreasing).
//! The verifier is sound: interval box arithmetic over-approximates each box, so 0 violating boxes =
//! a real certificate over the continuum, not a sampled hope (same principle as `ebm_cert_verify`).
//!
//! Run: `cargo run -p ferric-tensor --example efa_discover_verify --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

// ---- system + certificate constants ----
const CD: f64 = 0.5;      // damping
const R: f64 = 1.0;       // outer radius of certified annulus (‖x‖∞ ≤ R)
const R0: f64 = 0.10;     // inner radius (exclude a ball around the origin, where V̇→0)
const ALPHA: f64 = 0.02;  // decrease margin:   V̇ ≤ −α‖x‖²   (what we CERTIFY)
const DELTA: f64 = 0.02;  // positivity margin: V ≥  δ‖x‖²
const ALPHA_T: f64 = 0.10; // train to a STRICTER margin so the sound bound has slack
const DELTA_T: f64 = 0.10;
const GB: usize = 80;     // verifier grid: GB×GB boxes over [−R,R]²

// ================= the SOUND verifier (trusted kernel, f64 interval arithmetic) =================
#[derive(Clone, Copy)]
struct Iv { lo: f64, hi: f64 }
impl Iv {
    fn n(x: f64) -> Iv { Iv { lo: x, hi: x } }
    fn add(self, o: Iv) -> Iv { Iv { lo: self.lo + o.lo, hi: self.hi + o.hi } }
    fn sub(self, o: Iv) -> Iv { Iv { lo: self.lo - o.hi, hi: self.hi - o.lo } }
    fn mul(self, o: Iv) -> Iv {
        let (a, b, c, d) = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi);
        Iv { lo: a.min(b).min(c).min(d), hi: a.max(b).max(c).max(d) }
    }
    fn scale(self, k: f64) -> Iv { if k >= 0.0 { Iv { lo: self.lo * k, hi: self.hi * k } } else { Iv { lo: self.hi * k, hi: self.lo * k } } }
    fn sq(self) -> Iv { // sound square of an interval
        if self.lo >= 0.0 { Iv { lo: self.lo * self.lo, hi: self.hi * self.hi } }
        else if self.hi <= 0.0 { Iv { lo: self.hi * self.hi, hi: self.lo * self.lo } }
        else { Iv { lo: 0.0, hi: (self.lo * self.lo).max(self.hi * self.hi) } }
    }
}
// sound enclosures of sin/cos over [a,b] (check the extrema inside the range)
fn trig_iv(iv: Iv, off: f64, f: fn(f64) -> f64) -> Iv { // off = phase of the +1 extremum
    let (a, b) = (iv.lo, iv.hi);
    let (mut lo, mut hi) = (f(a).min(f(b)), f(a).max(f(b)));
    let kmin = ((a - off) / std::f64::consts::PI).floor() as i64 - 1;
    let kmax = ((b - off) / std::f64::consts::PI).ceil() as i64 + 1;
    for k in kmin..=kmax {
        let x = off + (k as f64) * std::f64::consts::PI;
        if x >= a && x <= b { let s = f(x); lo = lo.min(s); hi = hi.max(s); }
    }
    Iv { lo, hi }
}
fn sin_iv(iv: Iv) -> Iv { trig_iv(iv, std::f64::consts::FRAC_PI_2, f64::sin) }
fn cos_iv(iv: Iv) -> Iv { trig_iv(iv, 0.0, f64::cos) }

// g(x) = V̇ + α‖x‖²  and  h(x) = V − δ‖x‖²  at a point (exact)
fn g_at(x1: f64, x2: f64, a: f64, b: f64, cc: f64) -> f64 {
    let (f1, f2) = (x2, -x1.sin() - CD * x2);
    2.0 * ((a * x1 + b * x2) * f1 + (b * x1 + cc * x2) * f2) + ALPHA * (x1 * x1 + x2 * x2)
}
fn h_at(x1: f64, x2: f64, a: f64, b: f64, cc: f64) -> f64 {
    (a - DELTA) * x1 * x1 + 2.0 * b * x1 * x2 + (cc - DELTA) * x2 * x2
}
fn absmax(iv: Iv) -> f64 { iv.lo.abs().max(iv.hi.abs()) }

// per-box SOUND bounds via the centered (mean-value) form — tight, O(r²) slack, like ebm_cert_verify:
//   max_box g ≤ g(c) + |∂g/∂x₁|_box·r + |∂g/∂x₂|_box·r ,   min_box h ≥ h(c) − |∂h/∂x₁|·r − |∂h/∂x₂|·r
fn box_bounds(x1: Iv, x2: Iv, a: f64, b: f64, cc: f64) -> (f64, f64) {
    let (c1, c2) = ((x1.lo + x1.hi) / 2.0, (x2.lo + x2.hi) / 2.0);
    let r = ((x1.hi - x1.lo).max(x2.hi - x2.lo)) / 2.0;
    let (s1, co1) = (sin_iv(x1), cos_iv(x1));
    // ∂g/∂x₁ = 2[a·x₂ − b·sin x₁ − b·x₁·cos x₁ − CD·b·x₂ − cc·x₂·cos x₁] + 2α·x₁
    let dgx1 = x2.scale(2.0 * a).sub(s1.scale(2.0 * b)).sub(x1.mul(co1).scale(2.0 * b))
        .sub(x2.scale(2.0 * CD * b)).sub(x2.mul(co1).scale(2.0 * cc)).add(x1.scale(2.0 * ALPHA));
    // ∂g/∂x₂ = 2[a·x₁ + 2b·x₂ − CD·b·x₁ − cc·sin x₁ − 2·CD·cc·x₂] + 2α·x₂
    let dgx2 = x1.scale(2.0 * a).add(x2.scale(4.0 * b)).sub(x1.scale(2.0 * CD * b))
        .sub(s1.scale(2.0 * cc)).sub(x2.scale(4.0 * CD * cc)).add(x2.scale(2.0 * ALPHA));
    let g_hi = g_at(c1, c2, a, b, cc) + (absmax(dgx1) + absmax(dgx2)) * r;
    // ∂h/∂x₁ = 2(a−δ)x₁ + 2b·x₂ ; ∂h/∂x₂ = 2b·x₁ + 2(cc−δ)x₂
    let dhx1 = x1.scale(2.0 * (a - DELTA)).add(x2.scale(2.0 * b));
    let dhx2 = x1.scale(2.0 * b).add(x2.scale(2.0 * (cc - DELTA)));
    let h_lo = h_at(c1, c2, a, b, cc) - (absmax(dhx1) + absmax(dhx2)) * r;
    (h_lo, g_hi)
}
const MAXD: u32 = 7; // adaptive-refinement depth
// certify one box; on failure subdivide. Returns None if certified, else a WITNESS point (the center of
// the first sub-box still failing at max depth) — a real counterexample the learner must fix.
fn certify_box(x1: Iv, x2: Iv, a: f64, b: f64, cc: f64, depth: u32) -> Option<(f64, f64)> {
    if x1.lo > -R0 && x1.hi < R0 && x2.lo > -R0 && x2.hi < R0 { return None; } // excluded inner ball
    let (h_lo, g_hi) = box_bounds(x1, x2, a, b, cc);
    if h_lo > 0.0 && g_hi < 0.0 { return None; } // certified
    if depth == 0 { return Some(((x1.lo + x1.hi) / 2.0, (x2.lo + x2.hi) / 2.0)); }
    let (m1, m2) = ((x1.lo + x1.hi) / 2.0, (x2.lo + x2.hi) / 2.0);
    for &(a1, b1) in &[(x1.lo, m1), (m1, x1.hi)] {
        for &(a2, b2) in &[(x2.lo, m2), (m2, x2.hi)] {
            if let Some(w) = certify_box(Iv { lo: a1, hi: b1 }, Iv { lo: a2, hi: b2 }, a, b, cc, depth - 1) { return Some(w); }
        }
    }
    None
}
// verify the annulus; return (# of top-level boxes that failed, one counterexample witness per failed box)
fn verify(a: f64, b: f64, cc: f64) -> (usize, Vec<(f64, f64)>) {
    let mut ce = Vec::new();
    let step = 2.0 * R / GB as f64;
    for i in 0..GB { for j in 0..GB {
        let (lo1, lo2) = (-R + i as f64 * step, -R + j as f64 * step);
        if let Some(w) = certify_box(Iv { lo: lo1, hi: lo1 + step }, Iv { lo: lo2, hi: lo2 + step }, a, b, cc, MAXD) { ce.push(w); }
    }}
    (ce.len(), ce)
}

// ================= the LEARNER (Ferric autograd; V,V̇ are linear in (a,b,c)) =================
// Per point x: V = a·x1² + 2b·x1x2 + c·x2² ;  V̇ = a·(2 x1 f1) + b·(2(x2 f1 + x1 f2)) + c·(2 x2 f2).
fn coeffs(pts: &[(f64, f64)]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mut ca, mut cb, mut cc, mut da, mut db, mut dc, mut r2) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    for &(x1, x2) in pts {
        let (f1, f2) = (x2, -x1.sin() - CD * x2);
        ca.push((x1 * x1) as f32); cb.push((2.0 * x1 * x2) as f32); cc.push((x2 * x2) as f32);
        da.push((2.0 * x1 * f1) as f32); db.push((2.0 * (x2 * f1 + x1 * f2)) as f32); dc.push((2.0 * x2 * f2) as f32);
        r2.push((x1 * x1 + x2 * x2) as f32);
    }
    (ca, cb, cc, da, db, dc, r2)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context_new().await);
    println!("FUSED discover→verify — damped pendulum, quadratic Lyapunov certificate on annulus [{R0},{R}]");
    println!("  learner: Ferric autograd (a,b,c)   verifier: sound f64 interval boxes ({GB}×{GB})   one program\n");

    // NO training data to start. P begins as a poor 0.6·I (V̇=0 on the x₂ axis → fails the certificate),
    // so round 0 produces REAL counterexamples. From then on the verifier's failing boxes are the ONLY
    // teacher — this is the loop firing, not an anchor doing the work.
    let mut train: Vec<(f64, f64)> = Vec::new();
    let mut p = vec![Tensor::from_vec(&ctx, &[0.6], &[1, 1]),
                     Tensor::from_vec(&ctx, &[0.0], &[1, 1]),
                     Tensor::from_vec(&ctx, &[0.6], &[1, 1])];
    let mut adam = Adam::new(&p, 0.05);

    for round in 0..24 {
        // ---- train on the accumulated counterexamples the verifier has returned (skip if none yet) ----
        if !train.is_empty() {
            let (ca, cb, cc, da, db, dc, r2) = coeffs(&train);
            let n = train.len();
            let leaf = |v: &[f32]| Var::leaf(Tensor::from_vec(&ctx, v, &[n, 1]));
            let (lca, lcb, lcc) = (leaf(&ca), leaf(&cb), leaf(&cc));
            let (lda, ldb, ldc) = (leaf(&da), leaf(&db), leaf(&dc));
            let ar2 = leaf(&r2.iter().map(|x| x * ALPHA_T as f32).collect::<Vec<_>>());
            let dr2 = leaf(&r2.iter().map(|x| x * DELTA_T as f32).collect::<Vec<_>>());
            for _ in 0..250 {
                let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
                let (a, b, c) = (&pv[0], &pv[1], &pv[2]);
                let v = a.mul(&lca).add(&b.mul(&lcb)).add(&c.mul(&lcc));      // V
                let vd = a.mul(&lda).add(&b.mul(&ldb)).add(&c.mul(&ldc));     // V̇
                let pos = dr2.sub(&v).relu();                                 // relu(δ‖x‖² − V)
                let dec = vd.add(&ar2).relu();                               // relu(V̇ + α‖x‖²)
                let loss = pos.add(&dec).mean_all();
                loss.backward();
                let grd: Vec<Tensor> = pv.iter().zip(&p)
                    .map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
                adam.step(&mut p, &grd);
            }
        }
        // ---- read P out, run the SOUND verifier (same program, no boundary) ----
        let rd = |t: &Tensor| pollster::block_on(t.to_vec())[0] as f64;
        let (a, b, c) = (rd(&p[0]), rd(&p[1]), rd(&p[2]));
        let (nviol, ce) = verify(a, b, c);
        println!("  round {round:2}  P=[{a:6.3} {b:6.3}; {b:6.3} {c:6.3}]   train pts ={:5}   uncertified boxes ={:5}{}",
                 train.len(), nviol, if nviol == 0 { "   ✓ CERTIFIED" } else { "" });
        if nviol == 0 {
            println!("\nSOUND certificate found — V(x)=xᵀPx proves the damped pendulum stable on ‖x‖∈[{R0},{R}].");
            println!("The verifier TAUGHT the learner: it began from a poor P = 0.6·I with NO training data, and");
            println!("every candidate it proposed was checked soundly in the SAME Ferric program — the failing");
            println!("boxes became the next batch. Discover and verify, one graph, no process boundary.");
            return;
        }
        // the verifier's counterexamples are the ONLY training signal (accumulate, deduped on a fine grid)
        for w in ce { train.push(w); }
        train.sort_by(|p, q| (p.0.total_cmp(&q.0)).then(p.1.total_cmp(&q.1)));
        train.dedup_by(|p, q| ((p.0 - q.0).abs() < 0.012) && ((p.1 - q.1).abs() < 0.012));
        if train.len() > 6000 { let n = train.len(); train.drain(0..(n - 6000)); }
    }
    println!("\n(did not certify within the round budget — widen α/δ or shrink R)");
}

// tiny local Context bring-up (keeps the example self-contained)
async fn Context_new() -> ferric_core::Context { ferric_core::Context::new().await.unwrap() }
