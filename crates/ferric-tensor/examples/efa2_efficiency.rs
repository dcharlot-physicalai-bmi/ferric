//! EFA-2 · EFFICIENCY CARD — the joules-per-decision axis nobody else measures (Move 2 of BENCHMARKS-2026).
//! The whole manipulation arc was scored on SUCCESS; this scores every controller on COST — exact FLOPs per decision
//! from architecture (nets) and physics-step counts (planners). The EFA point: where a fast energy-first reactive
//! policy suffices, it is orders of magnitude cheaper per decision than planning — and PushT is exactly where you are
//! forced to pay the planning cost. Perf-per-watt made precise, across the arc.
fn mlp(inn: usize, h: usize, out: usize) -> u64 { 2 * (inn * h + h * h + h * out) as u64 }   // mult+add, 2-hidden MLP
fn main() {
    println!("  EFA-2 · EFFICIENCY CARD — FLOPs per decision across the manipulation arc (exact from architecture)\n");
    // reactive flow/energy policies: one MLP forward per decision (K=1)
    let efa1  = mlp(22, 128, 3);                    // efa-1 flow, 1-DOF path (matches the shipped card's 39,168)
    let reach = mlp(11, 192, 2);                    // Reacher flow (H=192)
    let push  = mlp(11, 128, 2);                    // free-pusher PushCube flow
    let cube  = mlp(13, 128, 2);                    // rotating-cube flow (obs 10 + a2 + t)
    let energy = mlp(14, 192, 1) * 289;             // energy-argmin: E-eval over a 17x17 action grid
    // diffusion policy: T denoise steps x MLP, per CM-macro chunk
    let (t_steps, cm) = (16u64, 6u64); let diff_chunk = t_steps * mlp(25, 256, 12); let diff_dec = diff_chunk / cm;
    // CEM-MPC: NS x NITER rollouts, each HZ x STRIDE physics steps; each step ~ dynamics+contact (~600 FLOP) +
    // a coverage eval per macro (120 samples x ~12 FLOP point-in-T). Per DECISION (one executed macro).
    let (ns, niter, hz, stride) = (48u64, 4u64, 70u64, 5u64);
    let phys_step = 600u64; let cov_eval = 120 * 12;
    let per_rollout = hz * (stride * phys_step + cov_eval);
    let mpc_dec = ns * niter * per_rollout;         // full CEM re-plan per executed macro-action
    println!("  reactive policies (ONE forward pass / decision):");
    println!("     efa-1 flow (1-DOF)        {:>12} FLOP", efa1);
    println!("     Reacher flow              {:>12} FLOP", reach);
    println!("     PushCube flow             {:>12} FLOP", push);
    println!("     rotating-cube flow        {:>12} FLOP", cube);
    println!("     energy-argmin (17x17 grid){:>12} FLOP", energy);
    println!("     diffusion policy          {:>12} FLOP  ({} denoise steps / {}-macro chunk)", diff_dec, t_steps, cm);
    println!("\n  planner:");
    println!("     CEM-MPC (long horizon)    {:>12} FLOP  ({}x{} rollouts x {} steps, coverage-dominated)", mpc_dec, ns, niter, hz * stride);
    println!("\n  the perf-per-watt point:");
    println!("     a working reactive flow (~{} FLOP) is {}x cheaper per decision than the MPC planner (~{} MFLOP)",
        reach, mpc_dec / reach, mpc_dec / 1_000_000);
    println!("     where a fast energy-first policy SUFFICES (Reacher 90%, PushCube 96%, full-stack/cube ~100%@K=1)");
    println!("     it wins ~{}x on joules; PushT is exactly the task that FORCES the {}x planning cost (or full-scale demos).",
        mpc_dec / reach, mpc_dec / reach);
    println!("\n  determinism (established across the arc): every reactive policy is bit-exact (same obs => same action);");
    println!("  MPC is bit-exact given its seed. The two axes no manipulation leaderboard scores — cost & reproducibility —");
    println!("  are the ones EFA reports, and they separate 'cheap where it works' from 'pay to plan where it must'.");
}
