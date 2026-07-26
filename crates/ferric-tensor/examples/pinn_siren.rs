//! A physics-informed neural network (PINN) trained GPU-native on the pure-Rust Ferric fabric —
//! solving the harmonic oscillator u'' + ω²u = 0, u(0)=1, u'(0)=0 (true solution cos ωt) from the
//! PHYSICS ALONE: no solution data, the loss is the ODE residual at collocation points plus the two
//! initial conditions. This is the hardest autodiff case — the loss depends on the network's SECOND
//! derivative wrt its input, u''(t) — which Ferric supplies via its differentiable `grad()` (each op
//! carries a Var-valued VJP), so u' = grad(u, t), u'' = grad(u', t), and the whole residual loss is
//! still differentiable wrt the parameters (training-through-differentiation, one Adam step per epoch).
//! Smooth activation = SIREN (sin), which represents a function AND its derivatives well — the natural
//! PINN activation, and a native Var op here. Everything runs resident on the wgpu device (Metal here).
//! Verified against the analytic cos ωt. No Python, no autodiff library — pure Rust on the fabric.
//!   cargo run --release --example pinn_siren

use ferric_tensor::{deriv, Adam, Siren, Tensor, Var};
use std::sync::Arc;

const W: f32 = 2.0; // ω
const T: f32 = 3.0; // domain [0, T]
const NC: usize = 48; // collocation points

// forward via the sciml library: u(t) = SIREN(t)
fn siren(t: &Var, p: &[Var]) -> Var { Siren::forward(p, t) }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("PINN on the Ferric fabric — backend: {:?}", ctx.backend);
    println!("solving u'' + {W}²u = 0, u(0)=1, u'(0)=0  (true: cos {W}t) from the residual alone\n");

    // SIREN [1,32,32,1] from the sciml library (SIREN init: first layer scaled + random phase)
    let net = Siren::new(&ctx, &[1, 32, 32, 1], 1);
    let mut wp = net.params.clone();
    let mut adam = Adam::new(&wp, 3e-3);

    let tcol: Vec<f32> = (0..NC).map(|i| i as f32 * T / (NC as f32 - 1.0)).collect();
    let w2col = vec![W * W; NC];
    let lambda = 40.0f32;

    for epoch in 0..6000 {
        let pv: Vec<Var> = wp.iter().map(|t| Var::leaf(t.clone())).collect();
        // residual at collocation: u'' + ω²u = 0
        let tv = Var::leaf(Tensor::from_vec(&ctx, &tcol, &[NC, 1]));
        let uu = siren(&tv, &pv);
        let u_t = deriv(&uu, &tv);   // du/dt  (per point)
        let u_tt = deriv(&u_t, &tv); // d²u/dt²
        let res = u_tt.add(&uu.mul(&Var::leaf(Tensor::from_vec(&ctx, &w2col, &[NC, 1]))));
        let loss_res = res.mul(&res).mean_all();
        // initial conditions u(0)=1, u'(0)=0
        let t0 = Var::leaf(Tensor::from_vec(&ctx, &[0.0], &[1, 1]));
        let u0 = siren(&t0, &pv);
        let u0_t = deriv(&u0, &t0);
        let e0 = u0.sub(&Var::leaf(Tensor::from_vec(&ctx, &[1.0], &[1, 1])));
        let loss_ic = e0.mul(&e0).sum_all().add(&u0_t.mul(&u0_t).sum_all());
        let loss = loss_res.add(&loss_ic.mul(&Var::leaf(Tensor::from_vec(&ctx, &[lambda], &[1]))));

        loss.backward(); // ∂loss/∂params — differentiates THROUGH the u''(grad) computation (2nd order)
        let g: Vec<Tensor> = pv.iter().zip(&wp).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::zeros(&ctx, &t.shape))).collect();
        adam.step(&mut wp, &g);

        if epoch % 1000 == 0 || epoch == 5999 {
            let lv = loss.value().to_vec().await[0];
            println!("  epoch {epoch:5}  loss {lv:.6}");
        }
    }

    // ---- verify vs the analytic solution cos(ωt), and CERTIFY on the fabric ----
    let ne = 400usize;
    let te: Vec<f32> = (0..ne).map(|i| i as f32 * T / (ne as f32 - 1.0)).collect();
    let pv: Vec<Var> = wp.iter().map(|t| Var::leaf(t.clone())).collect();
    let tev = Var::leaf(Tensor::from_vec(&ctx, &te, &[ne, 1]));
    let uev = siren(&tev, &pv);
    let uev_t = deriv(&uev, &tev);
    let uev_tt = deriv(&uev_t, &tev);
    let ue = uev.value().to_vec().await;
    let ut = uev_t.value().to_vec().await;
    let resid = uev_tt.add(&uev.mul(&Var::leaf(Tensor::from_vec(&ctx, &vec![W * W; ne], &[ne, 1])))).value().to_vec().await;
    let mut max_err = 0.0f32;
    for i in 0..ne { max_err = max_err.max((ue[i] - (W * te[i]).cos()).abs()); }
    println!("\n  max |u_PINN(t) − cos {W}t| over [0,{T}] = {max_err:.5}");
    println!("  u(1) = {:.5}   (cos {W} = {:.5})", ue[(ne as f32 / T) as usize], (W * 1.0).cos());
    println!("  {}", if max_err < 0.05 { "PASS — a PINN solved the ODE from physics alone, GPU-native on the pure-Rust fabric ✓" } else { "FAIL — did not converge below 0.05" });

    // a-posteriori certificate (computed on the fabric from the net's OWN residual, no true solution):
    // error e obeys e''+ω²e = r(t) with IC mismatches e0,e1  ⇒  ‖e‖∞ ≤ |e0| + |e1|/ω + (1/ω)∫|r|.
    let e0 = (ue[0] - 1.0).abs();
    let e1 = ut[0].abs();
    let mut integ = 0.0f32;
    for i in 1..ne { integ += 0.5 * (resid[i].abs() + resid[i - 1].abs()) * (te[i] - te[i - 1]); }
    let bound = e0 + e1 / W + integ / W;
    println!("\n  a-posteriori certificate (from the residual alone): ‖error‖ ≤ {bound:.5}");
    println!("  actual max error = {max_err:.5}   ⇒   {}", if bound >= max_err { "SOUND (bound ≥ true error) ✓ — the fabric trains AND certifies the PINN" } else { "UNSOUND ✗" });
}
