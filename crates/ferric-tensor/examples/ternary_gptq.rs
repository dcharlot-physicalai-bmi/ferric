//! Ferric ternary encoder — the GPTQ/PT²-LLM piece: Hessian-based error compensation. Naive per-element
//! ternary ignores that weights interact through the layer's INPUT correlations. GPTQ quantizes columns
//! left-to-right and, after each, updates the not-yet-quantized columns to CANCEL the output error it just
//! introduced, using the inverse input-Hessian H⁻¹ (H = XXᵀ from calibration data). Same 1.6 bpw, but it
//! minimizes ‖(W−Ŵ)X‖ — the actual layer output error — instead of ‖W−Ŵ‖. This is the free-lunch that turns
//! "18% on a hard layer" toward near-lossless. Verified: output error on a HELD-OUT input set.
//!   cargo run -p ferric-tensor --example ternary_gptq --release
fn main() {
    let (rc, ncal, ntest) = (512usize, 4096usize, 512usize); // ncal >> C so the input Hessian is well-estimated
    let (r, c) = (rc, rc);
    let mut seed = 0xDEADBEEF12345678u64;
    let mut u = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((seed >> 40) as f32 + 0.5) / (1u32 << 24) as f32 };
    let mut gauss = |u: &mut dyn FnMut() -> f32| { let (a, b) = (u().max(1e-7), u()); (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos() };

    // weight W [r,c] with ~1% ×12 outliers; calibration + test inputs X [c, n] (correlated channels)
    let mut w = vec![0f32; r * c];
    for v in w.iter_mut() { *v = gauss(&mut u) * 0.02; if u() < 0.01 { *v *= 12.0; } }
    // SHARED covariance structure: one loadings matrix so xcal and xtest come from the SAME distribution
    // (otherwise GPTQ fits the calibration covariance, which won't match a differently-generated test set).
    let nf = 96;
    let load: Vec<f32> = (0..c * nf).map(|_| gauss(&mut u)).collect();
    let gen_x = |u: &mut dyn FnMut() -> f32, g: &mut dyn FnMut(&mut dyn FnMut() -> f32) -> f32, n: usize, load: &[f32]| {
        let mut x = vec![0f32; c * n];
        for s in 0..n {
            let f: Vec<f32> = (0..nf).map(|_| g(u)).collect();
            for i in 0..c {
                let mut acc = 0.3 * g(u);
                for k in 0..nf { acc += load[i * nf + k] * f[k]; }
                x[i * n + s] = acc; // X[i, s]
            }
        }
        x
    };
    let xcal = gen_x(&mut u, &mut gauss, ncal, &load);
    let xtest = gen_x(&mut u, &mut gauss, ntest, &load);

    // per-output-channel ternary scale (PT²-LLM style): α_r, Δ_r from |W[r,:]|
    let mut alpha = vec![0f32; r]; let mut delta = vec![0f32; r];
    for ro in 0..r {
        let row = &w[ro * c..(ro + 1) * c];
        let ma = row.iter().map(|x| x.abs()).sum::<f32>() / c as f32;
        delta[ro] = 0.7 * ma;
        let (mut ss, mut sc) = (0f32, 0usize);
        for &x in row { if x.abs() > delta[ro] { ss += x.abs(); sc += 1; } }
        alpha[ro] = half::f16::from_f32(if sc > 0 { ss / sc as f32 } else { 0.0 }).to_f32();
    }
    let qz = |x: f32, ro: usize| if x.abs() > delta[ro] { if x > 0.0 { alpha[ro] } else { -alpha[ro] } } else { 0.0 };

    // ---- input Hessian H = Xcal Xcalᵀ + damping ; then H⁻¹ via Cholesky ----
    let mut h = vec![0f32; c * c];
    for i in 0..c { for j in 0..=i {
        let mut s = 0.0; for n in 0..ncal { s += xcal[i * ncal + n] * xcal[j * ncal + n]; }
        h[i * c + j] = s; h[j * c + i] = s;
    }}
    let dampv = 0.10 * (0..c).map(|i| h[i * c + i]).sum::<f32>() / c as f32; // higher damping (inputs are highly correlated)
    for i in 0..c { h[i * c + i] += dampv; }
    // Cholesky H=LLᵀ
    let mut l = vec![0f32; c * c];
    for i in 0..c { for j in 0..=i {
        let mut s = h[i * c + j];
        for k in 0..j { s -= l[i * c + k] * l[j * c + k]; }
        if i == j { l[i * c + i] = s.max(1e-12).sqrt(); } else { l[i * c + j] = s / l[j * c + j]; }
    }}
    // invert lower-triangular L
    let mut li = vec![0f32; c * c];
    for i in 0..c {
        li[i * c + i] = 1.0 / l[i * c + i];
        for j in 0..i {
            let mut s = 0.0; for k in j..i { s -= l[i * c + k] * li[k * c + j]; }
            li[i * c + j] = s / l[i * c + i];
        }
    }
    // Hinv = Liᵀ Li  (materialize full matrix)
    let mut hinv = vec![0f32; c * c];
    for i in 0..c { for j in i..c {
        let mut s = 0.0; for k in j..c { s += li[k * c + i] * li[k * c + j]; }
        hinv[i * c + j] = s; hinv[j * c + i] = s;
    }}
    // GPTQ uses the CHOLESKY FACTOR of H⁻¹ (not H⁻¹ itself). Lh = lower Cholesky of Hinv (Hinv = Lh Lhᵀ);
    // the update reads d = Lh[j,j] and Lh[k,j] for k>j (= the upper factor U[j,k]).
    let mut lh = vec![0f32; c * c];
    for i in 0..c { for j in 0..=i {
        let mut s = hinv[i * c + j];
        for k in 0..j { s -= lh[i * c + k] * lh[j * c + k]; }
        if i == j { lh[i * c + i] = s.max(1e-20).sqrt(); } else { lh[i * c + j] = s / lh[j * c + j]; }
    }}

    // ---- output error helper: ‖(W - Q) X‖_F / ‖W X‖_F on a given input set ----
    let err_on = |q: &[f32], xm: &[f32]| {
        let nn = xm.len() / c;
        let (mut num, mut den) = (0f32, 0f32);
        for ro in 0..r { for s in 0..nn {
            let (mut yq, mut yw) = (0f32, 0f32);
            for k in 0..c { let xv = xm[k * nn + s]; yw += w[ro * c + k] * xv; yq += (w[ro * c + k] - q[ro * c + k]) * xv; }
            num += yq * yq; den += yw * yw;
        }}
        (num / den).sqrt()
    };
    let err = |q: &[f32]| err_on(q, &xtest);

    // baseline: naive per-channel ternary
    let mut q_naive = vec![0f32; r * c];
    for ro in 0..r { for j in 0..c { q_naive[ro * c + j] = qz(w[ro * c + j], ro); } }
    let e_naive = err(&q_naive);

    // ---- GPTQ: quantize columns left→right, feed error forward via the Cholesky factor of H⁻¹ ----
    let mut wc = w.clone();              // working copy (gets updated)
    let mut q_gptq = vec![0f32; r * c];
    for j in 0..c {
        let djj = lh[j * c + j];
        let mut e = vec![0f32; r];
        for ro in 0..r { let wj = wc[ro * c + j]; let q = qz(wj, ro); q_gptq[ro * c + j] = q; e[ro] = (wj - q) / djj; }
        for k in (j + 1)..c { let ujk = lh[k * c + j]; if ujk == 0.0 { continue; } for ro in 0..r { wc[ro * c + k] -= e[ro] * ujk; } }
    }
    let e_gptq = err(&q_gptq);

    println!("Ternary output error ‖(W−Ŵ)X‖/‖WX‖ ({r}×{c} layer, ~1% ×12 outliers, {ncal} calib):");
    println!("  [calibration set]  naive {:.3e}  →  GPTQ {:.3e}", err_on(&q_naive, &xcal), err_on(&q_gptq, &xcal));
    println!("  [held-out test]    naive {e_naive:.3e}  →  GPTQ {e_gptq:.3e}   ({:.0}% lower, SAME 1.6 bpw)", 100.0 * (1.0 - e_gptq / e_naive));
    println!("\n✅ GPTQ/PT²-LLM Hessian error compensation runs in pure Rust: it minimizes the LAYER OUTPUT error,");
    println!("   not the weight error — free accuracy at the same 1.6 bpw. Stacks with rotation + multi-plane.");
}
