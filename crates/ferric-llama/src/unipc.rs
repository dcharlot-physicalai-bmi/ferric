//! **UniPC** multistep sampler (B(h) version, `bh2`) — the diffusion solver NVIDIA Cosmos 3 Edge
//! uses for its generator tower (video/action denoising). This is a from-scratch pure-Rust port of
//! diffusers' `UniPCMultistepScheduler` for the exact Cosmos config: rectified-flow prediction,
//! `predict_x0=True`, `use_flow_sigmas=True`, Karras sigmas (ρ=7, σ∈[0.147,200]), `solver_order=2`,
//! `solver_type="bh2"`, `lower_order_final=True`, `final_sigmas_type="zero"`.
//!
//! It runs on the host (tiny CPU vector math in f64); the per-step velocity is produced by the
//! generator tower on the GPU (see `cosmos_gen_forward`). Verified numerically identical to the real
//! diffusers scheduler: the update math was checked against diffusers driven by identical velocities
//! (max trajectory Δ = 1.4e-7 vs the real float32 scheduler), and this Rust port reproduces the f64
//! numpy reference (`unipc_golden.json`) to ~1e-12. See `examples/unipc_check.rs`.
//!
//! The single non-obvious correctness point — invisible without the reference — is that the CORRECTOR
//! bases its update on `last_sample` (the sample at the *start* of the previous step), not on the
//! freshly-predicted `this_sample`; only the extra `D1_t = this_x0 - m0` term carries the new info.

/// UniPC-bh2 solver state. Construct with [`UniPc::new`], read [`UniPc::timesteps`] to drive the
/// network, then call [`UniPc::step`] once per timestep with the network's predicted velocity.
pub struct UniPc {
    sigmas: Vec<f64>,   // length num_steps+1, ends at 0 (final_sigmas_type=zero)
    timesteps: Vec<f64>, // length num_steps — feeds the generator's time embedding
    order: usize,
    lower_order_final: bool,
    n: usize,           // == num_steps
    model_outputs: Vec<Option<Vec<f64>>>, // converted-x0 history, length `order`
    lower_order_nums: usize,
    last_sample: Option<Vec<f64>>,
    this_order: usize,
    step_index: usize,
}

impl UniPc {
    /// Build the sampler for the Cosmos 3 Edge action/video config with `num_steps` denoising steps
    /// (action mode uses 4). Reproduces `set_timesteps` for flow+Karras+config-σ.
    pub fn new(num_steps: usize) -> Self {
        Self::with_config(num_steps, 200.0, 0.147, 7.0, 1000.0, 2, true)
    }

    pub fn with_config(
        num_steps: usize,
        sigma_max: f64,
        sigma_min: f64,
        rho: f64,
        num_train: f64,
        order: usize,
        lower_order_final: bool,
    ) -> Self {
        // Karras ρ=7 ramp over [σ_max, σ_min], then flow transform σ/(σ+1), then append terminal 0.
        let max_inv = sigma_max.powf(1.0 / rho);
        let min_inv = sigma_min.powf(1.0 / rho);
        let mut sigmas = Vec::with_capacity(num_steps + 1);
        let mut timesteps = Vec::with_capacity(num_steps);
        for i in 0..num_steps {
            let ramp = if num_steps == 1 { 0.0 } else { i as f64 / (num_steps as f64 - 1.0) };
            let karras = (max_inv + ramp * (min_inv - max_inv)).powf(rho);
            let flow = karras / (karras + 1.0);
            sigmas.push(flow);
            timesteps.push(flow * num_train);
        }
        sigmas.push(0.0);
        UniPc {
            sigmas,
            timesteps,
            order,
            lower_order_final,
            n: num_steps,
            model_outputs: vec![None; order],
            lower_order_nums: 0,
            last_sample: None,
            this_order: 0,
            step_index: 0,
        }
    }

    /// Per-step timesteps (length `num_steps`) — pass `timesteps()[i]` to the generator's time
    /// embedding at step `i` (Cosmos multiplies by `timestep_scale=1e-3`, recovering the flow σ).
    pub fn timesteps(&self) -> &[f64] { &self.timesteps }
    /// Full sigma schedule (length `num_steps+1`, terminal 0).
    pub fn sigmas(&self) -> &[f64] { &self.sigmas }

    fn convert(&self, v: &[f64], sample: &[f64]) -> Vec<f64> {
        // flow_prediction + predict_x0:  x0 = sample - sigma_t * v
        let sigma = self.sigmas[self.step_index];
        sample.iter().zip(v).map(|(s, vv)| s - sigma * vv).collect()
    }

    /// Shared B(h) coefficient machinery for predictor/corrector. `si_curr`/`si_prev` index sigmas
    /// for (target, source0); `extra_offset` selects the history-index scheme (P: 0, C: 1).
    fn bh_coeffs(&self, order: usize, si_curr: usize, si_prev: usize, extra_offset: usize) -> BhCoeffs {
        let sigma_t = self.sigmas[si_curr];
        let sigma_s0 = self.sigmas[si_prev];
        let alpha_t = 1.0 - sigma_t;
        let alpha_s0 = 1.0 - sigma_s0;
        let lambda_t = alpha_t.ln() - sigma_t.ln();
        let lambda_s0 = alpha_s0.ln() - sigma_s0.ln();
        let h = lambda_t - lambda_s0;
        let m0 = self.model_outputs[order_last_idx(&self.model_outputs)].as_ref().unwrap().clone();
        let mut rks: Vec<f64> = Vec::new();
        let mut d1s: Vec<Vec<f64>> = Vec::new();
        for i in 1..order {
            let si = self.step_index - (i + extra_offset);
            let mi = self.model_outputs[self.model_outputs.len() - (i + 1)].as_ref().unwrap();
            let sigma_si = self.sigmas[si];
            let lambda_si = (1.0 - sigma_si).ln() - sigma_si.ln();
            let rk = (lambda_si - lambda_s0) / h;
            rks.push(rk);
            d1s.push(mi.iter().zip(&m0).map(|(a, b)| (a - b) / rk).collect());
        }
        rks.push(1.0);
        let hh = -h; // predict_x0
        let h_phi_1 = hh.exp_m1();
        let mut h_phi_k = h_phi_1 / hh - 1.0;
        let b_h = hh.exp_m1(); // bh2
        let mut rmat: Vec<Vec<f64>> = Vec::with_capacity(order);
        let mut bvec: Vec<f64> = Vec::with_capacity(order);
        let mut factorial_i = 1.0_f64;
        for i in 1..=order {
            rmat.push(rks.iter().map(|r| r.powi(i as i32 - 1)).collect());
            bvec.push(h_phi_k * factorial_i / b_h);
            factorial_i *= (i + 1) as f64;
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i;
        }
        BhCoeffs { sigma_t, sigma_s0, alpha_t, h_phi_1, b_h, m0, rmat, bvec, d1s }
    }

    fn uni_p(&self, sample: &[f64], order: usize) -> Vec<f64> {
        let c = self.bh_coeffs(order, self.step_index + 1, self.step_index, 0);
        let mut x_t: Vec<f64> = sample
            .iter()
            .zip(&c.m0)
            .map(|(x, m)| c.sigma_t / c.sigma_s0 * x - c.alpha_t * c.h_phi_1 * m)
            .collect();
        if !c.d1s.is_empty() {
            let rhos_p = if order == 2 { vec![0.5] } else { solve(&trim(&c.rmat), &c.bvec[..c.bvec.len() - 1]) };
            for cc in 0..x_t.len() {
                let mut pred = 0.0;
                for (k, r) in rhos_p.iter().enumerate() { pred += r * c.d1s[k][cc]; }
                x_t[cc] -= c.alpha_t * c.b_h * pred;
            }
        }
        x_t
    }

    fn uni_c(&self, this_x0: &[f64], last_sample: &[f64], order: usize) -> Vec<f64> {
        let c = self.bh_coeffs(order, self.step_index, self.step_index - 1, 1);
        let x = last_sample; // <-- diffusers bases the corrector on last_sample, NOT this_sample
        let rhos_c = if order == 1 { vec![0.5] } else { solve(&c.rmat, &c.bvec) };
        x.iter().enumerate().map(|(cc, xv)| {
            let mut x_t = c.sigma_t / c.sigma_s0 * xv - c.alpha_t * c.h_phi_1 * c.m0[cc];
            let mut corr = 0.0;
            for k in 0..rhos_c.len().saturating_sub(1) { corr += rhos_c[k] * c.d1s[k][cc]; }
            let d1_t = this_x0[cc] - c.m0[cc];
            x_t -= c.alpha_t * c.b_h * (corr + rhos_c[rhos_c.len() - 1] * d1_t);
            x_t
        }).collect()
    }

    /// One denoising step: given the network's predicted velocity `v` at the current timestep and the
    /// current `sample`, return the sample at the previous (less-noisy) timestep.
    pub fn step(&mut self, v: &[f64], sample: &[f64]) -> Vec<f64> {
        let use_corrector = self.step_index > 0 && self.last_sample.is_some();
        let x0 = self.convert(v, sample);
        let mut sample = sample.to_vec();
        if use_corrector {
            let last = self.last_sample.clone().unwrap();
            sample = self.uni_c(&x0, &last, self.this_order);
        }
        // history shift, store current converted x0 at the tail
        for i in 0..self.order - 1 { self.model_outputs[i] = self.model_outputs[i + 1].take(); }
        let last = self.order - 1;
        self.model_outputs[last] = Some(x0);
        // order selection (warmup + lower_order_final)
        let this_order = if self.lower_order_final { self.order.min(self.n - self.step_index) } else { self.order };
        self.this_order = this_order.min(self.lower_order_nums + 1);
        self.last_sample = Some(sample.clone());
        let prev = self.uni_p(&sample, self.this_order);
        if self.lower_order_nums < self.order { self.lower_order_nums += 1; }
        self.step_index += 1;
        prev
    }
}

struct BhCoeffs {
    sigma_t: f64,
    sigma_s0: f64,
    alpha_t: f64,
    h_phi_1: f64,
    b_h: f64,
    m0: Vec<f64>,
    rmat: Vec<Vec<f64>>,
    bvec: Vec<f64>,
    d1s: Vec<Vec<f64>>,
}

fn order_last_idx(v: &[Option<Vec<f64>>]) -> usize { v.len() - 1 }

/// Drop the last row and last column (R[:-1,:-1]) for the order≥3 predictor solve.
fn trim(r: &[Vec<f64>]) -> Vec<Vec<f64>> {
    r[..r.len() - 1].iter().map(|row| row[..row.len() - 1].to_vec()).collect()
}

/// Solve the small dense linear system A x = b via Gaussian elimination with partial pivoting.
fn solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a.iter().map(|r| r.clone()).collect();
    let mut y = b.to_vec();
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n { if m[r][col].abs() > m[piv][col].abs() { piv = r; } }
        m.swap(col, piv); y.swap(col, piv);
        let d = m[col][col];
        for r in col + 1..n {
            let f = m[r][col] / d;
            for cc in col..n { m[r][cc] -= f * m[col][cc]; }
            y[r] -= f * y[col];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = y[r];
        for cc in r + 1..n { s -= m[r][cc] * x[cc]; }
        x[r] = s / m[r][r];
    }
    x
}
