//! Synthetic physical processes with known ground truth, parameterized so they form families.
//!
//! Five kinds — a damped oscillation, a thermal decay, a PWM square wave, a chirp, and noise on a
//! drift — each drawn from a family: `variant` selects a deterministic parameter set (a different
//! damping rate, duty cycle, chirp slope) from bounded ranges. Two variants of one kind share
//! structure and differ in parameters, which is exactly the split a generalization experiment
//! needs: train on some variants, hold out others, and the held-out signals are new parameters
//! from the same distribution rather than new physics.
//!
//! Everything is deterministic in `(kind, variant, i)` with no RNG state. The parameter draws are
//! integer hashes and identical on every machine; the waveform samples pass through `f32`
//! transcendentals, so within a platform they are exactly reproducible while cross-platform
//! bit-identity is not claimed.
//!
//! ## Two facts a consumer must not learn the hard way
//!
//! **Per-example normalization erases each family's affine parameters.** A pipeline that fits
//! RevIn per signal sees `(x - mean) / sd`, which cancels every pure offset and pure scale: the
//! damped, square and chirp amplitudes vanish, and the thermal ambient and delta vanish. What
//! survives is the SHAPE parameters — decay rates, frequencies, duty cycles, slopes — and every
//! kind here keeps at least two of them precisely so that the normalized family is never
//! one-dimensional. Even so, two variants CAN land near-identical in normalized space; an
//! experiment that splits train from held-out by variant index must check disjointness in the
//! representation its model actually sees, not in the raw signal. `scaling_curve` does exactly
//! that and reports what it excluded.
//!
//! **The anti-aliasing guarantee has a window.** Parameter ranges keep every process below the
//! 250 Hz Nyquist limit for windows within the first quarter second (the examples use 96 samples
//! at 500 Hz). Sampled far beyond that, a chirp's instantaneous frequency keeps climbing and the
//! output aliases — exactly as an undersampled real sensor would, and with the same consequence:
//! the samples stay finite and deterministic, but they no longer mean "a chirp".
//!
//! The labels these processes carry are written by this generator. Any experiment built on them
//! demonstrates properties of the ARCHITECTURE; it says nothing about real sensors, and callers
//! are expected to say so too.

use std::f32::consts::TAU;

/// Number of process kinds.
pub const KINDS: usize = 5;

/// Human name for a kind. Stable: these are the caption words experiments train against.
pub fn name(kind: usize) -> &'static str {
    ["damped", "thermal", "square", "chirp", "noise"][kind % KINDS]
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic uniform in [0, 1) for one parameter of one (kind, variant).
fn unit(kind: usize, variant: usize, salt: u64) -> f32 {
    let z = mix(((kind as u64) << 32)
        ^ (variant as u64).wrapping_mul(0x9E37_79B9)
        ^ salt.wrapping_mul(0xD134_2543_DE82_EF95));
    ((z >> 40) as f32) / (1u32 << 24) as f32
}

/// One sample of one process. `i` is the sample index at 500 Hz (dt = 2 ms).
///
/// Parameter ranges keep every variant physical within the first quarter second: chirps stay below
/// Nyquist there, duty cycles stay strictly inside (0, 1) so a square wave never degenerates to DC,
/// and decays remain visible within a 96-sample window. Beyond that window a chirp aliases, as the
/// module docs spell out.
pub fn sample(kind: usize, variant: usize, i: usize) -> f32 {
    let t = i as f32 * 0.002;
    let u = |salt: u64| unit(kind, variant, salt);
    match kind % KINDS {
        0 => {
            let amp = 1.0 + 3.0 * u(0);
            let dec = 3.0 + 6.0 * u(1);
            let f = 8.0 + 12.0 * u(2);
            amp * (-dec * t).exp() * (TAU * f * t).sin()
        }
        1 => {
            // A linear drain term rides on the exponential so that TWO shape parameters survive
            // per-example normalization: the decay rate, and the drain-to-delta ratio. Without it
            // this family is one-dimensional after normalization (ambient and delta both cancel),
            // and independent draws on a 1-D axis collide often enough that a train/held-out split
            // by variant index quietly stops holding anything out.
            let ambient = 10.0 + 20.0 * u(0);
            let delta = 20.0 + 40.0 * u(1);
            let rate = 2.0 + 6.0 * u(2);
            let drain = 10.0 + 30.0 * u(3);
            ambient + delta * (-rate * t).exp() - drain * t
        }
        2 => {
            let f = 30.0 + 60.0 * u(0);
            let duty = 0.2 + 0.6 * u(1);
            let amp = 3.0 + 3.0 * u(2);
            if (t * f).fract() < duty { amp } else { 0.0 }
        }
        3 => {
            let f0 = 2.0 + 6.0 * u(0);
            let slope = 100.0 + 300.0 * u(1);
            let amp = 1.0 + 2.0 * u(2);
            (TAU * (f0 * t + 0.5 * slope * t * t)).sin() * amp
        }
        _ => {
            let drift_f = 0.5 + 1.5 * u(0);
            let drift_a = 0.3 + 0.7 * u(1);
            let noise_a = 0.5 + 1.0 * u(2);
            let z = mix(((kind as u64) << 48) ^ ((variant as u64) << 24) ^ (i as u64) ^ 0xABCD);
            let w = ((z >> 40) as f32) / (1u32 << 24) as f32 - 0.5;
            drift_a * (TAU * drift_f * t).sin() + noise_a * w
        }
    }
}

/// A whole signal: `len` samples of one (kind, variant).
pub fn signal(kind: usize, variant: usize, len: usize) -> Vec<f32> {
    (0..len).map(|i| sample(kind, variant, i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_generator_is_deterministic() {
        for kind in 0..KINDS {
            for v in [0usize, 3, 100] {
                assert_eq!(signal(kind, v, 96), signal(kind, v, 96), "kind {kind} v {v}");
            }
        }
    }

    #[test]
    fn every_sample_is_finite_across_a_wide_sweep() {
        for kind in 0..KINDS {
            for v in 0..64 {
                assert!(signal(kind, v, 512).iter().all(|x| x.is_finite()), "kind {kind} v {v}");
            }
        }
    }

    /// Two variants of one kind must actually differ, or "held out" holds nothing out.
    #[test]
    fn variants_within_a_kind_are_distinct() {
        for kind in 0..KINDS {
            for v in 1..8 {
                assert_ne!(signal(kind, 0, 96), signal(kind, v, 96), "kind {kind} variant {v} collides with variant 0");
            }
        }
    }

    #[test]
    fn kinds_are_distinct_from_each_other() {
        for a in 0..KINDS {
            for b in (a + 1)..KINDS {
                assert_ne!(signal(a, 0, 96), signal(b, 0, 96), "kinds {a} and {b} collide");
            }
        }
    }

    /// No variant may degenerate to a constant. A square wave with duty >= 1 becomes DC and stops
    /// being a square wave; the parameter ranges exist to prevent exactly that, so test it rather
    /// than trust it.
    #[test]
    fn no_variant_is_constant() {
        for kind in 0..KINDS {
            for v in 0..64 {
                let s = signal(kind, v, 96);
                let (lo, hi) = s.iter().fold((f32::MAX, f32::MIN), |(l, h), &x| (l.min(x), h.max(x)));
                assert!(hi - lo > 1e-3, "kind {kind} ({}) variant {v} is flat: range {}", name(kind), hi - lo);
            }
        }
    }
}
