// UniPC-bh2 sampler (predict_x0, flow sigmas, solver_order=2, lower_order_final) — Rust port of the
// numpy reference (cosmos_ref/unipc_numpy.py), verified against the REAL diffusers scheduler. This is
// the sampler that drives Cosmos-3-Edge denoising: at each step the generator emits a velocity, UniPC
// converts it to x0 and advances the latent. Pure f64, no deps; cross-verified here to the reference
// trajectory, ready to drop onto the verified generator forward in ferric-llama/src/cosmos.rs.

const SIGMAS: [f64; 5] = [0.9950248599052429, 0.9736331105232239, 0.7985969185829163, 0.12816041707992554, 0.0];

fn as_(s: f64) -> (f64, f64) { (1.0 - s, s) }   // flow: alpha_t = 1-sigma, sigma_t = sigma

fn solve(r: &[Vec<f64>], b: &[f64]) -> Vec<f64> { // small Gaussian elimination
    let n = b.len();
    let mut a: Vec<Vec<f64>> = (0..n).map(|i| { let mut row = r[i].clone(); row.push(b[i]); row }).collect();
    for col in 0..n {
        let piv = (col..n).max_by(|&x, &y| a[x][col].abs().partial_cmp(&a[y][col].abs()).unwrap()).unwrap();
        a.swap(col, piv);
        let d = a[col][col];
        for j in col..=n { a[col][j] /= d; }
        for i in 0..n { if i != col { let f = a[i][col]; for j in col..=n { a[i][j] -= f * a[col][j]; } } }
    }
    (0..n).map(|i| a[i][n]).collect()
}

struct UniPc {
    order: usize, lof: bool, n: usize,
    model_outputs: Vec<Option<Vec<f64>>>, lower_order_nums: usize,
    last_sample: Option<Vec<f64>>, this_order: usize, step_index: usize,
}
impl UniPc {
    fn new(order: usize) -> Self {
        UniPc { order, lof: true, n: SIGMAS.len() - 1, model_outputs: vec![None; order],
                lower_order_nums: 0, last_sample: None, this_order: 0, step_index: 0 }
    }
    fn convert(&self, v: &[f64], sample: &[f64]) -> Vec<f64> {
        let (_, sig_t) = as_(SIGMAS[self.step_index]);
        sample.iter().zip(v).map(|(s, vv)| s - sig_t * vv).collect()
    }
    // returns (sigma_t, sigma_s0, alpha_t, h_phi_1, B_h, m0, R, b, D1s)
    fn bh(&self, order: usize, si_curr: usize, si_prev: usize, extra: i64)
        -> (f64, f64, f64, f64, f64, Vec<f64>, Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
        let (alpha_t, sigma_t) = as_(SIGMAS[si_curr]);
        let (alpha_s0, sigma_s0) = as_(SIGMAS[si_prev]);
        let lam_t = alpha_t.ln() - sigma_t.ln();
        let lam_s0 = alpha_s0.ln() - sigma_s0.ln();
        let h = lam_t - lam_s0;
        let m0 = self.model_outputs[self.order - 1].clone().unwrap();
        let (mut rks, mut d1s): (Vec<f64>, Vec<Vec<f64>>) = (Vec::new(), Vec::new());
        for i in 1..order {
            let si = (self.step_index as i64 - (i as i64 + extra)) as usize;
            let mi = self.model_outputs[self.order - 1 - i].clone().unwrap();
            let (a_si, s_si) = as_(SIGMAS[si]);
            let lam_si = a_si.ln() - s_si.ln();
            let rk = (lam_si - lam_s0) / h;
            rks.push(rk);
            d1s.push(mi.iter().zip(&m0).map(|(a, b)| (a - b) / rk).collect());
        }
        rks.push(1.0);
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let mut h_phi_k = h_phi_1 / hh - 1.0;
        let b_h = hh.exp_m1();
        let (mut rmat, mut bvec): (Vec<Vec<f64>>, Vec<f64>) = (Vec::new(), Vec::new());
        let mut fact = 1.0f64;
        for i in 1..=order {
            rmat.push(rks.iter().map(|&rk| rk.powi(i as i32 - 1)).collect());
            bvec.push(h_phi_k * fact / b_h);
            fact *= (i + 1) as f64;
            h_phi_k = h_phi_k / hh - 1.0 / fact;
        }
        (sigma_t, sigma_s0, alpha_t, h_phi_1, b_h, m0, rmat, bvec, d1s)
    }
    fn uni_p(&self, sample: &[f64], order: usize) -> Vec<f64> {
        let (sig_t, sig_s0, alpha_t, h_phi_1, b_h, m0, rmat, bvec, d1s) =
            self.bh(order, self.step_index + 1, self.step_index, 0);
        let c = sample.len();
        let mut xt: Vec<f64> = (0..c).map(|j| sig_t / sig_s0 * sample[j] - alpha_t * h_phi_1 * m0[j]).collect();
        if !d1s.is_empty() {
            let rhos = if order == 2 { vec![0.5] } else {
                let sub_r: Vec<Vec<f64>> = rmat[..order - 1].iter().map(|r| r[..order - 1].to_vec()).collect();
                solve(&sub_r, &bvec[..order - 1])
            };
            for j in 0..c {
                let mut pr = 0.0; for k in 0..rhos.len() { pr += rhos[k] * d1s[k][j]; }
                xt[j] -= alpha_t * b_h * pr;
            }
        }
        xt
    }
    fn uni_c(&self, this_x0: &[f64], last_sample: &[f64], order: usize) -> Vec<f64> {
        let (sig_t, sig_s0, alpha_t, h_phi_1, b_h, m0, rmat, bvec, d1s) =
            self.bh(order, self.step_index, self.step_index - 1, 1);
        let c = last_sample.len();
        let rhos = if order == 1 { vec![0.5] } else { solve(&rmat, &bvec) };
        (0..c).map(|j| {
            let xt_ = sig_t / sig_s0 * last_sample[j] - alpha_t * h_phi_1 * m0[j];
            let mut corr = 0.0; for k in 0..d1s.len() { corr += rhos[k] * d1s[k][j]; }
            let d1_t = this_x0[j] - m0[j];
            xt_ - alpha_t * b_h * (corr + rhos[rhos.len() - 1] * d1_t)
        }).collect()
    }
    fn step(&mut self, v: &[f64], sample: &[f64]) -> Vec<f64> {
        let use_corr = self.step_index > 0 && self.last_sample.is_some();
        let x0 = self.convert(v, sample);
        let mut sample = sample.to_vec();
        if use_corr {
            let last = self.last_sample.clone().unwrap();
            sample = self.uni_c(&x0, &last, self.this_order);
        }
        for i in 0..self.order - 1 { self.model_outputs[i] = self.model_outputs[i + 1].clone(); }
        self.model_outputs[self.order - 1] = Some(x0);
        let this_order = if self.lof { self.order.min(self.n - self.step_index) } else { self.order };
        self.this_order = this_order.min(self.lower_order_nums + 1);
        self.last_sample = Some(sample.clone());
        let prev = self.uni_p(&sample, self.this_order);
        if self.lower_order_nums < self.order { self.lower_order_nums += 1; }
        self.step_index += 1;
        prev
    }
}

fn main() {
    let d = 6;
    let x0: Vec<f64> = (0..d).map(|i| (i as f64 * 0.7).sin()).collect();
    let vels: Vec<Vec<f64>> = (0..4).map(|i| (0..d).map(|j| (j as f64 * 0.5 + i as f64 * 0.3).cos() * 0.4).collect()).collect();
    let mut me = UniPc::new(2);
    let mut x = x0.clone();
    for i in 0..4 { x = me.step(&vels[i], &x); }
    // reference final state (numpy UniPC, verified == real diffusers scheduler to 1e-7)
    let x4_ref = [-0.2988471145495058, 0.5206147200895057, 1.0673532274036543, 1.130566495975874, 0.7223405617296, 0.06172708638989455];
    let err = x.iter().zip(&x4_ref).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    println!("Rust UniPC final x[4] = {:?}", x.iter().map(|v| (v * 1e6).round() / 1e6).collect::<Vec<_>>());
    println!("reference       x[4] = {:?}", x4_ref);
    println!("max |Δ| vs verified reference = {:.3e}  ->  {}", err, if err < 1e-6 { "MATCH ✓" } else { "MISMATCH ✗" });
}
