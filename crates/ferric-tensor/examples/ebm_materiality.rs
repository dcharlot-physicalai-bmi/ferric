//! EFA energy-first #38 — MATERIALITY DOES COMPUTATION (Krakauer): physics drops the search clauses.
//!
//! Krakauer (SFI): "physics does for the Soma cube what algorithms do for the Rubik's cube" — the material
//! (exclusion, collision) DROPS clauses out of what is formally an intractable constraint problem, so a child
//! solves it. The EFA reading: encode the constraints as an ENERGY and let descent (the physical dynamics)
//! satisfy them, instead of searching. Clean instance: pack N discs in a box, none overlapping. ABSTRACT search
//! (place at random, reject on overlap) fails exponentially as density rises. PHYSICAL descent on a repulsion
//! energy — the material constraint built into the dynamics — pushes the discs apart and finds a valid packing
//! for free. Same constraint; the physics does the computation.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_materiality --release`
const R: f32 = 0.075; // disc radius (box is the unit square)

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn uu(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }

fn valid(x: &[f32], y: &[f32], n: usize) -> bool {
    for i in 0..n { if x[i] < R - 1e-3 || x[i] > 1.0 - R + 1e-3 || y[i] < R - 1e-3 || y[i] > 1.0 - R + 1e-3 { return false; }
        for j in i + 1..n { let d = ((x[i] - x[j]).powi(2) + (y[i] - y[j]).powi(2)).sqrt(); if d < 2.0 * R - 1e-3 { return false; } } }
    true
}
// ABSTRACT search: uniform random placement, accept iff valid. Return success fraction over `tries`.
fn abstract_success(n: usize, tries: usize, seed0: u32) -> f32 {
    let mut ok = 0; for t in 0..tries {
        let mut x = vec![0.0f32; n]; let mut y = vec![0.0f32; n];
        for i in 0..n { x[i] = R + uu(seed0 + t as u32 * 131 + i as u32 * 2, 1) * (1.0 - 2.0 * R); y[i] = R + uu(seed0 + t as u32 * 131 + i as u32 * 2, 2) * (1.0 - 2.0 * R); }
        if valid(&x, &y, n) { ok += 1; } } ok as f32 / tries as f32
}
// PHYSICAL descent: energy = Σ overlap² + wall² ; descend positions. Return (success fraction, avg steps-to-valid).
fn physical(n: usize, seeds: usize, seed0: u32) -> (f32, f32) {
    let (steps, alpha) = (400usize, 0.6f32); let mut ok = 0; let mut tot_steps = 0u64;
    for sd in 0..seeds {
        let mut x = vec![0.0f32; n]; let mut y = vec![0.0f32; n];
        for i in 0..n { x[i] = R + uu(seed0 + sd as u32 * 977 + i as u32 * 2, 1) * (1.0 - 2.0 * R); y[i] = R + uu(seed0 + sd as u32 * 977 + i as u32 * 2, 2) * (1.0 - 2.0 * R); }
        let mut solved_at = steps;
        for step in 0..steps {
            let mut fx = vec![0.0f32; n]; let mut fy = vec![0.0f32; n];
            for i in 0..n { for j in i + 1..n {
                let (dx, dy) = (x[i] - x[j], y[i] - y[j]); let d = (dx * dx + dy * dy).sqrt() + 1e-6;
                if d < 2.0 * R { let f = (2.0 * R - d); fx[i] += f * dx / d; fy[i] += f * dy / d; fx[j] -= f * dx / d; fy[j] -= f * dy / d; } } }
            for i in 0..n { // walls
                if x[i] < R { fx[i] += R - x[i]; } if x[i] > 1.0 - R { fx[i] -= x[i] - (1.0 - R); }
                if y[i] < R { fy[i] += R - y[i]; } if y[i] > 1.0 - R { fy[i] -= y[i] - (1.0 - R); } }
            for i in 0..n { x[i] = (x[i] + alpha * fx[i]).clamp(0.0, 1.0); y[i] = (y[i] + alpha * fy[i]).clamp(0.0, 1.0); }
            if valid(&x, &y, n) { solved_at = step + 1; break; }
        }
        if valid(&x, &y, n) { ok += 1; tot_steps += solved_at as u64; }
    }
    (ok as f32 / seeds as f32, if ok > 0 { tot_steps as f32 / ok as f32 } else { f32::NAN })
}

fn main() {
    println!("  EFA energy-first — MATERIALITY DOES COMPUTATION: physics drops the search clauses (disc packing)\n");
    println!("  pack N discs (r={}) in the unit box, none overlapping — the same constraint, two ways:", R);
    println!("     N    density   ABSTRACT search (random placement)      PHYSICAL descent (repulsion energy)");
    for &n in &[4usize, 8, 12, 16, 20] {
        let dens = n as f32 * std::f32::consts::PI * R * R * 100.0;
        let a = abstract_success(n, 200_000, 7) * 100.0;
        let (ps, pk) = physical(n, 200, 5000);
        let pkstr = if pk.is_nan() { "—".to_string() } else { format!("~{:.0} steps", pk) };
        println!("     {:<4} {:>4.0}%     valid in {:>6.3}% of random configs        {:>5.0}% solved   {}", n, dens, a, ps * 100.0, pkstr);
    }
    println!("\n  As density rises, abstract random search finds a valid packing in an exponentially vanishing fraction");
    println!("  of configs — while physical descent still solves it in a few steps. The material (repulsion) enforces");
    println!("  the 'no-overlap' clauses by construction: physics does the computation, exactly Krakauer's point.");
}
