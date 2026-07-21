//! EFA energy-first #32 — THE WATTS: quantify the compute/energy of edge energy-based control vs LLM inference.
//!
//! The mission (data-center-grade capability at the EDGE) has asserted "performance per watt, not tokens" but
//! never put a number on it. This does — honestly. We can't wattmeter this Mac, so we count the hardware-
//! INDEPENDENT quantity: FLOPs per task, exact from the real model dimensions of the embodied-control builds
//! (ebm_control energy-shaping; ebm_plan MPPI). We contrast with LLM inference (the standard 2·params FLOPs/token),
//! and give an energy estimate at a stated efficiency. Caveats up front and load-bearing: (1) this is FLOP-
//! counting + a cited pJ/FLOP figure, NOT a power measurement; (2) these solve a NANO task (a pendulum) — the
//! CAPABILITY is nano, not yet data-center-grade; the point is the ARCHITECTURE's edge-compute viability, not a
//! capability claim; (3) an LLM cannot even run the continuous control loop — the contrast is about SHAPE + scale.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_watts --release`

// FLOPs for an MLP forward pass over layer sizes (2 FLOP per multiply-accumulate)
fn mlp_flops(layers: &[usize]) -> u64 { let mut f = 0u64; for w in layers.windows(2) { f += 2 * (w[0] as u64) * (w[1] as u64); } f }
fn si(x: f64) -> String { // human-readable magnitude
    let (a, u) = if x >= 1e12 { (x / 1e12, "T") } else if x >= 1e9 { (x / 1e9, "G") } else if x >= 1e6 { (x / 1e6, "M") } else if x >= 1e3 { (x / 1e3, "k") } else { (x, "") };
    format!("{:.2} {}", a, u)
}

fn main() {
    println!("  EFA energy-first — THE WATTS: edge energy-based control vs LLM inference (compute per task)\n");
    let steps = 260u64;                 // control decisions in one swing-up (ebm_control / ebm_plan)
    let e_energy = mlp_flops(&[2, 64, 1]);      // learned energy Ê(θ,ω) forward pass
    let e_dyn = mlp_flops(&[4, 64, 1]);         // learned dynamics ĝ(sinθ,cosθ,ω,u) forward pass
    let (mppi_n, mppi_h) = (300u64, 120u64);    // MPPI samples × horizon (ebm_plan)

    // per-decision compute
    let c_shape = e_energy + 40;                             // energy-shaping: 1 energy eval + O(1) arithmetic
    let c_mppi = mppi_n * mppi_h * (e_dyn + 20);             // MPPI: N·H learned-model rollouts
    // per full swing-up
    let t_shape = c_shape * steps;
    let t_mppi = c_mppi * steps;

    println!("  ON-DEVICE energy-based control — FLOPs (exact, from the real model dims):");
    println!("     energy-shaping (learned energy)   {:>10} FLOP / decision   {:>10} FLOP / swing-up", si(c_shape as f64), si(t_shape as f64));
    println!("     MPPI planning  (learned model)    {:>10} FLOP / decision   {:>10} FLOP / swing-up", si(c_mppi as f64), si(t_mppi as f64));

    // LLM inference: 2·params FLOP per token (standard). A control loop would need reasoning at loop rate.
    println!("\n  LLM INFERENCE for comparison (2·params FLOP / token, standard):");
    for (name, p) in [("7B", 7e9), ("70B", 70e9), ("400B", 400e9)] {
        let per_tok = 2.0 * p;
        println!("     {:>5} model   {:>10} FLOP / token   ({:>10} FLOP for {} tokens)", name, si(per_tok), si(per_tok * steps as f64), steps);
    }

    // energy estimate at a stated efficiency (edge accelerators ~1–10 pJ/FLOP; use 5 pJ = 5e-12 J)
    let pj = 5e-12;
    println!("\n  ENERGY ESTIMATE @ 5 pJ/FLOP (edge-accelerator class; a stated assumption, not a measurement):");
    println!("     energy-shaping swing-up   ≈ {:.2e} J  ({} FLOP)", t_shape as f64 * pj, si(t_shape as f64));
    println!("     MPPI-planned swing-up     ≈ {:.2e} J  ({} FLOP)", t_mppi as f64 * pj, si(t_mppi as f64));
    println!("     ONE 7B-LLM token          ≈ {:.2e} J  (14 G FLOP) — and a datacenter GPU idles at ~10²–10³ W", 14e9 * pj);

    let ratio = (14e9) / (t_shape as f64);
    println!("\n  A whole energy-shaping swing-up costs ~{:.0}× LESS compute than a SINGLE 7B-LLM token.", ratio);
    println!("  Energy-shaping control fits a MICROCONTROLLER; MPPI fits a PHONE; 7B-LLM inference needs a datacenter GPU.");
    println!("\n  HONEST CAVEATS: (1) FLOP-count + a cited pJ/FLOP, NOT a wattmeter. (2) The CAPABILITY here is nano");
    println!("  (a pendulum) — this shows the ARCHITECTURE is edge-compute-viable, NOT that we have data-center-grade");
    println!("  capability yet. (3) An LLM can't run the continuous control loop at all — wrong SHAPE, not just scale.");
}
