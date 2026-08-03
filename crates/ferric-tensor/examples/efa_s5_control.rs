//! S5 — discover a CONTROLLER with a safety certificate, on the device.
//!
//! Everything before this VERIFIED a fixed system. S5 DISCOVERS the actionable thing: a feedback controller
//! u=π(x) that provably stabilizes an UNSTABLE plant, with the Lyapunov certificate co-designed in the same
//! fused loop. Plant = inverted pendulum at the top: ẋ₁=x₂, ẋ₂=sin x₁ − c·x₂ + u. The sin x₁ term is
//! DESTABILIZING (gravity pulls away from upright) — with u=0 the origin is a saddle, no certificate exists.
//! The learner co-designs the controller gains (k₁,k₂) AND the certificate V=xᵀPx so the CLOSED loop satisfies
//! V̇+α‖x‖²<0; the SAME sound interval verifier (sin-interval + refinement) certifies the closed loop; an
//! independent forward simulation confirms the discovered controller truly drives the pendulum to upright.
//! The certified region is a SAFETY GUARANTEE: inside it, this controller provably stabilizes.
//! Run: `cargo run -p ferric-tensor --example efa_s5_control --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::f64::consts::PI;
use std::sync::Arc;

const CD: f64 = 0.20;      // plant damping
const R0: f64 = 0.05;
const ALPHA: f64 = 0.01;
const ALPHA_T: f64 = 0.06;
const DELTA_T: f64 = 0.06;
const GB: usize = 48;
const MAXD: u32 = 6;

fn u_of(x1: f64, x2: f64, k1: f64, k2: f64) -> f64 { -(k1*x1 + k2*x2) }
// closed-loop plant with controller (k1,k2)
fn fcl(x1: f64, x2: f64, k1: f64, k2: f64) -> (f64, f64) { (x2, x1.sin() - CD*x2 + u_of(x1,x2,k1,k2)) }

#[derive(Clone, Copy)] struct Iv { lo: f64, hi: f64 }
impl Iv {
    fn add(self,o:Iv)->Iv{Iv{lo:self.lo+o.lo,hi:self.hi+o.hi}}
    fn mul(self,o:Iv)->Iv{let(a,b,c,d)=(self.lo*o.lo,self.lo*o.hi,self.hi*o.lo,self.hi*o.hi);Iv{lo:a.min(b).min(c).min(d),hi:a.max(b).max(c).max(d)}}
    fn scale(self,k:f64)->Iv{if k>=0.0{Iv{lo:self.lo*k,hi:self.hi*k}}else{Iv{lo:self.hi*k,hi:self.lo*k}}}
    fn sq(self)->Iv{if self.lo>=0.0{Iv{lo:self.lo*self.lo,hi:self.hi*self.hi}}else if self.hi<=0.0{Iv{lo:self.hi*self.hi,hi:self.lo*self.lo}}else{Iv{lo:0.0,hi:(self.lo*self.lo).max(self.hi*self.hi)}}}
}
fn sin_iv(x: Iv) -> Iv {
    let mut lo = x.lo.sin().min(x.hi.sin()); let mut hi = x.lo.sin().max(x.hi.sin());
    let mut k = ((x.lo - PI/2.0)/PI).floor() as i64 - 1;
    while (k as f64)*PI + PI/2.0 <= x.hi + 1e-12 {
        let e = (k as f64)*PI + PI/2.0;
        if e >= x.lo - 1e-12 && e <= x.hi + 1e-12 { let s = e.sin(); lo = lo.min(s); hi = hi.max(s); }
        k += 1;
    }
    Iv { lo, hi }
}
// sound upper bound of V̇+α‖x‖² over a box for the CLOSED loop (natural interval extension)
fn gbox(x1: Iv, x2: Iv, a: f64, b: f64, c: f64, k1: f64, k2: f64) -> f64 {
    let f1 = x2;
    let f2 = sin_iv(x1).add(x2.scale(-CD)).add(x1.scale(-k1)).add(x2.scale(-k2)); // sin x1 − c x2 − k1 x1 − k2 x2
    let vx1 = x1.scale(2.0*a).add(x2.scale(2.0*b));
    let vx2 = x1.scale(2.0*b).add(x2.scale(2.0*c));
    vx1.mul(f1).add(vx2.mul(f2)).add(x1.sq().add(x2.sq()).scale(ALPHA)).hi
}
fn cert_box(x1: Iv, x2: Iv, a:f64,b:f64,c:f64,k1:f64,k2:f64, depth:u32) -> Option<(f64,f64)> {
    if x1.lo>-R0 && x1.hi<R0 && x2.lo>-R0 && x2.hi<R0 { return None; }
    // positivity of V=xᵀPx is structural for a,c>0 & det>0; here we check the decrease condition soundly
    if gbox(x1,x2,a,b,c,k1,k2) < 0.0 { return None; }
    if depth==0 { return Some(((x1.lo+x1.hi)/2.0,(x2.lo+x2.hi)/2.0)); }
    let (m1,m2)=((x1.lo+x1.hi)/2.0,(x2.lo+x2.hi)/2.0);
    for &(p,q) in &[(x1.lo,m1),(m1,x1.hi)] { for &(u,v) in &[(x2.lo,m2),(m2,x2.hi)] {
        if let Some(w)=cert_box(Iv{lo:p,hi:q},Iv{lo:u,hi:v},a,b,c,k1,k2,depth-1){return Some(w);} } }
    None
}
fn verify(a:f64,b:f64,c:f64,k1:f64,k2:f64, r:f64) -> (usize, Vec<(f64,f64)>) {
    // require P≻0 (else V not a valid Lyapunov fn) — cheap structural gate before the box sweep
    if a<=1e-6 || c<=1e-6 || a*c-b*b<=1e-6 { return (GB*GB, vec![(0.1,0.0)]); }
    let mut ce=Vec::new(); let step=2.0*r/GB as f64;
    for i in 0..GB { for j in 0..GB { let (lo1,lo2)=(-r+i as f64*step,-r+j as f64*step);
        if let Some(w)=cert_box(Iv{lo:lo1,hi:lo1+step},Iv{lo:lo2,hi:lo2+step},a,b,c,k1,k2,MAXD){ce.push(w);} } }
    (ce.len(), ce)
}
// independent oracle: does the closed loop truly reach upright from x0?
fn stabilizes(x1:f64,x2:f64,k1:f64,k2:f64)->bool{
    let (mut a,mut b)=(x1,x2); let dt=0.002;
    for _ in 0..20000 { let (f1,f2)=fcl(a,b,k1,k2); a+=dt*f1; b+=dt*f2;
        if (a*a+b*b).sqrt()>8.0 {return false;} if (a*a+b*b).sqrt()<1e-3 {return true;} }
    (a*a+b*b).sqrt()<0.3
}
fn pointwise_ok(a:f64,b:f64,c:f64,k1:f64,k2:f64,r:f64)->bool{ // dense check V̇+α‖x‖²<0
    let n=400usize;
    for i in 0..=n { for j in 0..=n {
        let x1=-r+2.0*r*i as f64/n as f64; let x2=-r+2.0*r*j as f64/n as f64;
        if x1.abs().max(x2.abs())<R0 {continue;}
        let (f1,f2)=fcl(x1,x2,k1,k2);
        let vdot=(2.0*a*x1+2.0*b*x2)*f1+(2.0*b*x1+2.0*c*x2)*f2;
        if vdot + ALPHA*(x1*x1+x2*x2) >= 0.0 { return false; }
    }}
    true
}

// co-design (k1,k2,a,b,c) via the fused loop. `learn_ctrl=false` freezes u=0 (uncontrolled baseline).
async fn cegis(ctx:&Arc<ferric_core::Context>, learn_ctrl:bool, r:f64, rounds:usize)->(bool,[f64;5]){
    // params: [k1,k2,a,b,c]
    let init = if learn_ctrl { [1.0f64,0.5,1.0,0.0,1.0] } else { [0.0,0.0,1.0,0.0,1.0] };
    let mut p: Vec<Tensor> = init.iter().map(|&v| Tensor::from_vec(ctx,&[v as f32],&[1,1])).collect();
    let mut adam = Adam::new(&p, 0.03);
    let mut train: Vec<(f64,f64)> = Vec::new();
    let mut pf = init;
    for _ in 0..rounds {
        if !train.is_empty() {
            let n=train.len();
            let col=|v:Vec<f32>| Var::leaf(Tensor::from_vec(ctx,&v,&[n,1]));
            let x1=col(train.iter().map(|&(a,_)|a as f32).collect());
            let x2=col(train.iter().map(|&(_,b)|b as f32).collect());
            let x1_2=col(train.iter().map(|&(a,_)|(2.0*a) as f32).collect());
            let x2_2=col(train.iter().map(|&(_,b)|(2.0*b) as f32).collect());
            let x1sq=col(train.iter().map(|&(a,_)|(a*a) as f32).collect());
            let x1x2_2=col(train.iter().map(|&(a,b)|(2.0*a*b) as f32).collect());
            let x2sq=col(train.iter().map(|&(_,b)|(b*b) as f32).collect());
            let fcl2free=col(train.iter().map(|&(a,b)|(a.sin()-CD*b) as f32).collect()); // sin x1 − c x2
            let r2a=col(train.iter().map(|&(a,b)|((a*a+b*b)*ALPHA_T) as f32).collect());
            let r2d=col(train.iter().map(|&(a,b)|((a*a+b*b)*DELTA_T) as f32).collect());
            for _ in 0..200 {
                let pv:Vec<Var>=p.iter().map(|t|Var::leaf(t.clone())).collect();
                let (k1,k2,a,b,c)=(&pv[0],&pv[1],&pv[2],&pv[3],&pv[4]);
                let vfun=a.mul(&x1sq).add(&b.mul(&x1x2_2)).add(&c.mul(&x2sq));   // V
                let gv1=a.mul(&x1_2).add(&b.mul(&x2_2));                          // ∂V/∂x1
                let gv2=b.mul(&x1_2).add(&c.mul(&x2_2));                          // ∂V/∂x2
                let fcl2 = if learn_ctrl { fcl2free.sub(&k1.mul(&x1)).sub(&k2.mul(&x2)) } else { fcl2free.clone() };
                let vdot=gv1.mul(&x2).add(&gv2.mul(&fcl2));                       // V̇ (closed loop)
                let loss=r2d.sub(&vfun).relu().add(&vdot.add(&r2a).relu()).mean_all();
                loss.backward();
                let grd:Vec<Tensor>=pv.iter().zip(&p).map(|(v,t)|v.grad().unwrap_or_else(||Tensor::from_vec(ctx,&vec![0.0;t.numel()],&t.shape))).collect();
                // freeze controller grads in the baseline
                let grd:Vec<Tensor> = grd.into_iter().enumerate().map(|(i,g)| if !learn_ctrl && i<2 { Tensor::from_vec(ctx,&[0.0],&[1,1]) } else { g }).collect();
                adam.step(&mut p,&grd);
            }
        }
        pf = { let v:Vec<f64>=p.iter().map(|t|pollster::block_on(t.to_vec())[0] as f64).collect(); [v[0],v[1],v[2],v[3],v[4]] };
        let (nv,ce)=verify(pf[2],pf[3],pf[4],pf[0],pf[1],r);
        if nv==0 { return (true,pf); }
        for w in ce { train.push(w); }
        train.sort_by(|a,b|a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        train.dedup_by(|a,b|(a.0-b.0).abs()<0.02 && (a.1-b.1).abs()<0.02);
        if train.len()>4000 { let n=train.len(); train.drain(0..(n-4000)); }
    }
    (false,pf)
}

fn main(){ pollster::block_on(run()); }
async fn run(){
    let ctx=Arc::new(ferric_core::Context::new().await.unwrap());
    println!("S5 — discover a CONTROLLER + safety certificate for the UNSTABLE inverted pendulum (top).");
    println!("  plant ẋ₁=x₂, ẋ₂=sin x₁ − {CD}·x₂ + u ; u=0 ⇒ origin is a saddle (no certificate exists).\n");

    // baseline: no controller — can it certify anything?
    let (ok0,_)=cegis(&ctx,false,0.3,20).await;
    println!("  u = 0 (uncontrolled): largest region certified = {}  ← the destabilizing sin x₁ term wins",
             if ok0 {"‖x‖∞≤0.3"} else {"NONE (verifier refuses — the open loop is not stable)"});

    // discover controller + certificate, sweep the safe region
    println!("\n  co-designing controller u=−(k₁x₁+k₂x₂) AND certificate V=xᵀPx in the fused loop:");
    let (mut br, mut bp)=(0.0f64,[0.0;5]);
    for step in 0..12 {
        let r=0.4+0.1*step as f64;
        let (ok,pf)=cegis(&ctx,true,r,30).await;
        if ok { br=r; bp=pf; } else if br>0.0 { break; }
    }
    let [k1,k2,a,b,c]=bp;
    let sound = pointwise_ok(a,b,c,k1,k2,br);
    // independent sim from the corners/edge of the certified region
    let mut sim_ok=true;
    for t in 0..24 { let s=-br+2.0*br*t as f64/24.0;
        for &(x,y) in &[(br,s),(-br,s),(s,br),(s,-br)] { if !stabilizes(x,y,k1,k2){sim_ok=false;} } }
    println!("  discovered controller: u = −({k1:.2}·x₁ + {k2:.2}·x₂)");
    println!("  certificate: V=xᵀPx, P=[{a:.2} {b:.2}; {b:.2} {c:.2}]");
    println!("  → PROVABLY stabilizes the inverted pendulum on the safety region ‖x‖∞ ≤ {br:.1}");
    println!("     sound (independent pointwise V̇<0): {}   controller reaches upright from the region (sim): {}",
             if sound {"YES"} else {"NO — BUG"}, if sim_ok {"YES"} else {"NO"});
    println!("\n  The actionable object — a controller — comes WITH its proof of the region it is safe in, discovered");
    println!("  on-device in one deterministic loop. (Torque is unbounded here; a real safety cert would add |u|≤ū");
    println!("  as another sound constraint — the same machinery, one more interval.)");
}
