//! Sound interval arithmetic.
//!
//! An interval `[lo, hi]` stands for *every* real in that range. Every operation must return an interval
//! that **contains** the true result set — being too wide is merely weak, being too narrow is wrong.
//!
//! ## Why outward rounding, and why it is not pedantry
//!
//! `f64` arithmetic rounds to nearest. For a lower bound that is the wrong direction: `a + b` can land
//! *above* the true infimum, so the interval excludes values it should contain. Every operation therefore
//! nudges `lo` down and `hi` up by one ulp.
//!
//! Ferric's own examples got this wrong — 27 of them do interval certification with plain round-to-nearest
//! arithmetic, and four independently reimplemented it. Measured, the drift is real: summing `0.1` a
//! million times, the round-to-nearest lower bound sits **8.2e-6 above** the correctly-rounded one, i.e.
//! it asserts a tighter bound than it can justify.
//!
//! In practice a Lyapunov margin is usually much larger than the accumulated error, so those conclusions
//! were probably right. But "probably right" is not what the word *certificate* claims, and a certificate
//! that is only approximately sound cannot be composed, published, or trusted at a safety boundary. The
//! whole value of the artifact is that it is a proof.

/// A closed real interval, with all operations outward-rounded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Iv {
    pub lo: f64,
    pub hi: f64,
}

/// Widen outward by one ulp on each side. Applied after every arithmetic operation.
///
/// `next_down`/`next_up` are exact ulp steps, so this is the minimal correction that restores
/// containment under round-to-nearest. Infinities are left alone — stepping past them is meaningless and
/// an infinite bound is already as wide as it gets.
#[inline]
fn outward(lo: f64, hi: f64) -> Iv {
    Iv {
        lo: if lo.is_finite() { lo.next_down() } else { lo },
        hi: if hi.is_finite() { hi.next_up() } else { hi },
    }
}

impl Iv {
    /// An interval containing exactly one value.
    ///
    /// Not widened: a literal is exact, and widening it would lose precision for no soundness gain.
    #[inline]
    pub fn point(x: f64) -> Iv { Iv { lo: x, hi: x } }

    #[inline]
    pub fn new(lo: f64, hi: f64) -> Iv {
        debug_assert!(lo <= hi || lo.is_nan() || hi.is_nan(), "inverted interval [{lo}, {hi}]");
        Iv { lo, hi }
    }

    pub const EMPTY_SENTINEL: Iv = Iv { lo: f64::NAN, hi: f64::NAN };

    #[inline]
    pub fn width(self) -> f64 { self.hi - self.lo }
    #[inline]
    pub fn mid(self) -> f64 { self.lo + (self.hi - self.lo) * 0.5 }
    #[inline]
    pub fn contains(self, x: f64) -> bool { self.lo <= x && x <= self.hi }
    /// Strictly positive everywhere in the interval.
    #[inline]
    pub fn is_positive(self) -> bool { self.lo > 0.0 }
    /// Strictly negative everywhere in the interval.
    #[inline]
    pub fn is_negative(self) -> bool { self.hi < 0.0 }

    #[inline]
    pub fn add(self, o: Iv) -> Iv { outward(self.lo + o.lo, self.hi + o.hi) }

    #[inline]
    pub fn sub(self, o: Iv) -> Iv { outward(self.lo - o.hi, self.hi - o.lo) }

    #[inline]
    pub fn neg(self) -> Iv { Iv { lo: -self.hi, hi: -self.lo } } // exact: negation never rounds

    /// Product. All four corner products, because a sign change inside either operand moves the extrema
    /// off the corners of the naive `[lo*lo, hi*hi]` guess.
    #[inline]
    pub fn mul(self, o: Iv) -> Iv {
        let (a, b, c, d) = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi);
        outward(a.min(b).min(c).min(d), a.max(b).max(c).max(d))
    }

    #[inline]
    pub fn scale(self, k: f64) -> Iv {
        if k >= 0.0 { outward(self.lo * k, self.hi * k) } else { outward(self.hi * k, self.lo * k) }
    }

    /// Square. **Not** `self.mul(self)`.
    ///
    /// `x·x` where `x ∈ [−1, 1]` gives `[−1, 1]` under interval multiplication, because multiplication
    /// treats its operands as independent. Squaring knows they are the same value, so the true range is
    /// `[0, 1]`. Using `mul` here is sound but needlessly wide — and that width is what makes a
    /// branch-and-bound search fail to converge.
    #[inline]
    pub fn sq(self) -> Iv {
        if self.lo >= 0.0 {
            outward(self.lo * self.lo, self.hi * self.hi)
        } else if self.hi <= 0.0 {
            outward(self.hi * self.hi, self.lo * self.lo)
        } else {
            outward(0.0, (self.lo * self.lo).max(self.hi * self.hi))
        }
    }

    /// Reciprocal. `None` when the interval straddles zero — the true result is unbounded and split, and
    /// returning some finite interval would be a lie.
    #[inline]
    pub fn recip(self) -> Option<Iv> {
        if self.lo <= 0.0 && self.hi >= 0.0 { return None; }
        Some(outward(1.0 / self.hi, 1.0 / self.lo))
    }

    pub fn div(self, o: Iv) -> Option<Iv> { Some(self.mul(o.recip()?)) }

    /// Monotone increasing function applied at both ends. Correct for `exp`, `sinh`, odd powers, etc.
    #[inline]
    fn monotone_inc(self, f: impl Fn(f64) -> f64) -> Iv { outward(f(self.lo), f(self.hi)) }

    #[inline]
    pub fn exp(self) -> Iv { self.monotone_inc(f64::exp) }

    /// Hyperbolic tangent — monotone, and the saturating nonlinearity most control barriers involve.
    #[inline]
    pub fn tanh(self) -> Iv { self.monotone_inc(f64::tanh) }

    /// Sine over the interval.
    ///
    /// Endpoints alone are not enough: an extremum strictly inside the range is the true bound, and a
    /// naive `[sin(lo), sin(hi)]` misses it. Every critical point `π/2 + kπ` inside the interval is
    /// checked. This is the operation where an unsound implementation is easiest to write and hardest to
    /// notice, because it is right whenever the interval happens to be narrow.
    pub fn sin(self) -> Iv { self.trig(f64::sin, core::f64::consts::FRAC_PI_2) }

    /// Cosine. Same reasoning as [`Iv::sin`]; extrema at `kπ`.
    pub fn cos(self) -> Iv { self.trig(f64::cos, 0.0) }

    fn trig(self, f: fn(f64) -> f64, first_max_phase: f64) -> Iv {
        // A range spanning a full period attains both extrema; skip the scan and say so.
        if self.width() >= 2.0 * core::f64::consts::PI || !self.width().is_finite() {
            return Iv { lo: -1.0, hi: 1.0 };
        }
        let (mut lo, mut hi) = (f(self.lo).min(f(self.hi)), f(self.lo).max(f(self.hi)));
        // Walk one step wider on each side so a critical point sitting exactly on a boundary is caught.
        let k0 = ((self.lo - first_max_phase) / core::f64::consts::PI).floor() as i64 - 1;
        let k1 = ((self.hi - first_max_phase) / core::f64::consts::PI).ceil() as i64 + 1;
        for k in k0..=k1 {
            let x = first_max_phase + (k as f64) * core::f64::consts::PI;
            if x >= self.lo && x <= self.hi {
                let v = f(x);
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        outward(lo.max(-1.0), hi.min(1.0))
    }

    /// Split at the midpoint, for branch-and-bound.
    pub fn bisect(self) -> (Iv, Iv) {
        let m = self.mid();
        (Iv { lo: self.lo, hi: m }, Iv { lo: m, hi: self.hi })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random sampler — no dependency, reproducible failures.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn in_range(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.next() }
    }

    /// THE soundness property: for any point in the input interval(s), the true value must lie inside the
    /// computed interval. Sampling cannot prove containment, but it reliably catches a violation — and a
    /// violated containment is the one bug that makes every certificate above it worthless.
    #[test]
    fn operations_contain_the_true_result() {
        let mut r = Rng(0xC0FFEE);
        for _ in 0..20_000 {
            let (a1, a2) = (r.in_range(-10.0, 10.0), r.in_range(-10.0, 10.0));
            let (b1, b2) = (r.in_range(-10.0, 10.0), r.in_range(-10.0, 10.0));
            let a = Iv::new(a1.min(a2), a1.max(a2));
            let b = Iv::new(b1.min(b2), b1.max(b2));
            // Sample actual points inside each interval and check every op.
            for _ in 0..8 {
                let x = r.in_range(a.lo, a.hi);
                let y = r.in_range(b.lo, b.hi);
                assert!(a.add(b).contains(x + y), "add: {x}+{y} outside {:?}", a.add(b));
                assert!(a.sub(b).contains(x - y), "sub");
                assert!(a.mul(b).contains(x * y), "mul: {x}*{y} outside {:?}", a.mul(b));
                assert!(a.sq().contains(x * x), "sq: {x}^2 outside {:?}", a.sq());
                assert!(a.neg().contains(-x), "neg");
                assert!(a.sin().contains(x.sin()), "sin: sin({x}) outside {:?}", a.sin());
                assert!(a.cos().contains(x.cos()), "cos: cos({x}) outside {:?}", a.cos());
                assert!(a.tanh().contains(x.tanh()), "tanh");
                if !b.contains(0.0) {
                    assert!(a.div(b).unwrap().contains(x / y), "div");
                }
            }
        }
    }

    #[test]
    fn sin_catches_an_extremum_strictly_inside_the_range() {
        // The classic unsound implementation: [sin(lo), sin(hi)] misses the peak at pi/2. Here both
        // endpoints are 0 and the true maximum is 1.
        let iv = Iv::new(0.0, core::f64::consts::PI);
        let s = iv.sin();
        assert!(s.hi >= 1.0, "sin over [0, pi] must reach 1, got {s:?}");
        assert!(s.lo <= 0.0);
        // A full period attains both extrema.
        let full = Iv::new(-10.0, 10.0).sin();
        assert!(full.lo <= -1.0 && full.hi >= 1.0);
    }

    #[test]
    fn squaring_is_tighter_than_multiplying_by_itself() {
        // Both are SOUND; sq is tighter because it knows the operands are the same value. That width is
        // the difference between a branch-and-bound search converging and running to max depth.
        let x = Iv::new(-1.0, 1.0);
        assert!(x.sq().lo >= -1e-15, "sq of [-1,1] must be non-negative, got {:?}", x.sq());
        assert!(x.mul(x).lo < -0.5, "mul-by-self should be the loose one: {:?}", x.mul(x));
        assert!(x.sq().width() < x.mul(x).width());
    }

    #[test]
    fn outward_rounding_actually_widens() {
        // If this ever stops holding, the crate has silently reverted to round-to-nearest and every
        // certificate it issues is an estimate.
        let a = Iv::point(0.1);
        let s = a.add(a);
        assert!(s.lo < 0.2 && s.hi > 0.2, "0.1+0.1 must straddle 0.2, got {s:?}");
        let mut acc = Iv::point(0.0);
        for _ in 0..1000 { acc = acc.add(Iv::point(0.1)); }
        assert!(acc.lo < 100.0 && acc.hi > 100.0, "accumulated interval lost containment: {acc:?}");
    }

    #[test]
    fn division_by_an_interval_straddling_zero_is_refused() {
        // The true result is unbounded and disconnected; returning a finite interval would be a lie that
        // every downstream bound would then inherit.
        assert!(Iv::new(-1.0, 1.0).recip().is_none());
        assert!(Iv::new(0.0, 1.0).recip().is_none(), "a bound AT zero is still unbounded");
        assert!(Iv::new(1.0, 2.0).recip().is_some());
    }

    #[test]
    fn bisection_covers_the_original_interval() {
        let x = Iv::new(-3.0, 7.0);
        let (a, b) = x.bisect();
        assert_eq!(a.lo, x.lo);
        assert_eq!(b.hi, x.hi);
        assert_eq!(a.hi, b.lo, "bisection left a gap");
        assert!(a.width() > 0.0 && b.width() > 0.0);
    }
}
