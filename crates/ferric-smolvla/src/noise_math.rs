//! Deterministic `ln`, `sin`, and `cos` for [`crate::noise_from_seed`].
//!
//! These are ferric's functions. They use IEEE-754 round-to-nearest operations
//! and do not call platform libm. `f64::sqrt` is the IEEE-754 square root.
//!
//! `ln` follows fdlibm `e_log.c` (`__ieee754_log`). That file's error analysis
//! bounds the error by less than 1 ulp. The SunPro notice from `e_log.c` is
//! preserved below.
//!
//! `k_sin`, `k_cos`, and `rem_pio2` follow the FreeBSD msun sources as revised
//! in musl 1.2.5, not the Sun 5.3 qx form. The cosine return is musl `__cos.c`
//! lines 67–70 (`w + (((1-w)-hz) + (z*r-x*y))`). The reduction constants are
//! musl `__rem_pio2.c` lines 38–39, and the medium-size correction is lines
//! 119–132. FreeBSD `k_cos.c` line 74 is that same return. On `[-π/4, π/4]`
//! the kernel remainder is at most `2^-58`. The medium path covers every
//! Box–Muller angle `2·π·u2`.
//!
//! Copyright (C) 1993 by Sun Microsystems, Inc. Permission to use, copy, modify,
//! and distribute this software is freely granted, provided that this notice is
//! preserved.

const LN2_HI: f64 = 6.93147180369123816490e-01;
const LN2_LO: f64 = 1.90821492927058770002e-10;
const TWO54: f64 = 1.80143985094819840000e16;
const LG1: f64 = 6.666666666666735130e-01;
const LG2: f64 = 3.999999999940941908e-01;
const LG3: f64 = 2.857142874366239149e-01;
const LG4: f64 = 2.222219843214978396e-01;
const LG5: f64 = 1.818357216161805012e-01;
const LG6: f64 = 1.531383769920937332e-01;
const LG7: f64 = 1.479819860511658591e-01;

const S1: f64 = -1.66666666666666324348e-01;
const S2: f64 = 8.33333333332248946124e-03;
const S3: f64 = -1.98412698298579493134e-04;
const S4: f64 = 2.75573137070700676789e-06;
const S5: f64 = -2.50507602534068634195e-08;
const S6: f64 = 1.58969099521155010221e-10;

const C1: f64 = 4.16666666666666019037e-02;
const C2: f64 = -1.38888888888741095749e-03;
const C3: f64 = 2.48015872894767294178e-05;
const C4: f64 = -2.75573143513906633035e-07;
const C5: f64 = 2.08757232129817482790e-09;
const C6: f64 = -1.13596475577881948265e-11;

const INV_PIO2: f64 = 6.36619772367581382433e-01;
const PIO2_1: f64 = 1.57079632673412561417e+00;
const PIO2_1T: f64 = 6.07710050650619224932e-11;
const PIO2_2: f64 = 6.07710050630396597660e-11;
const PIO2_2T: f64 = 2.02226624879595063154e-21;
const PIO2_3: f64 = 2.02226624871116645580e-21;
const PIO2_3T: f64 = 8.47842766036889956997e-32;
const PIO4: f64 = f64::from_bits(0x3fe9_21fb_5444_2d18);

fn high(x: f64) -> i32 {
    (x.to_bits() >> 32) as i32
}

fn from_words(hx: i32, lx: u32) -> f64 {
    f64::from_bits(((hx as u32 as u64) << 32) | u64::from(lx))
}

fn low(x: f64) -> u32 {
    x.to_bits() as u32
}

/// `ln`. `ln(0)` is `-∞`, the same IEEE result as the platform call this replaced.
/// A negative argument or a NaN returns NaN. `+∞` returns `+∞`.
pub fn ln(x: f64) -> f64 {
    let mut x = x;
    let mut hx = high(x);
    let mut lx = low(x);
    let mut k = 0i32;
    if hx < 0x0010_0000 {
        if (((hx as u32) & 0x7fff_ffff) | lx) == 0 {
            return f64::NEG_INFINITY;
        }
        if hx < 0 {
            return f64::NAN;
        }
        k -= 54;
        x *= TWO54;
        hx = high(x);
        lx = low(x);
    }
    if hx >= 0x7ff0_0000 {
        return x + x;
    }
    k += (hx >> 20) - 1023;
    hx &= 0x000f_ffff;
    let i = (hx + 0x9_5f64) & 0x10_0000;
    x = from_words(hx | (i ^ 0x3ff0_0000), lx);
    k += i >> 20;
    let f = x - 1.0;
    if (0x000f_ffff & (2 + hx)) < 3 {
        if f == 0.0 {
            if k == 0 {
                return 0.0;
            }
            let dk = f64::from(k);
            return dk * LN2_HI + dk * LN2_LO;
        }
        let r = f * f * (0.5 - 0.33333333333333333 * f);
        if k == 0 {
            return f - r;
        }
        let dk = f64::from(k);
        return dk * LN2_HI - ((r - dk * LN2_LO) - f);
    }
    let s = f / (2.0 + f);
    let dk = f64::from(k);
    let z = s * s;
    let mut ii = hx - 0x6_147a;
    let w = z * z;
    let j = 0x6_b851 - hx;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    ii |= j;
    let r = t2 + t1;
    if ii > 0 {
        let hfsq = 0.5 * f * f;
        if k == 0 {
            return f - (hfsq - s * (hfsq + r));
        }
        return dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f);
    }
    if k == 0 {
        return f - s * (f - r);
    }
    dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)
}

fn k_sin(x: f64, y: f64, iy: bool) -> f64 {
    let z = x * x;
    let w = z * z;
    let r = S2 + z * (S3 + z * S4) + z * w * (S5 + z * S6);
    let v = z * x;
    if !iy {
        x + v * (S1 + z * r)
    } else {
        x - ((z * (0.5 * y - v * r) - y) - v * S1)
    }
}

fn k_cos(x: f64, y: f64) -> f64 {
    let z = x * x;
    let w = z * z;
    let r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6));
    let hz = 0.5 * z;
    let w1 = 1.0 - hz;
    w1 + (((1.0 - w1) - hz) + (z * r - x * y))
}

fn medium(x: f64, ix: u32, sign: bool) -> (i32, f64, f64) {
    let toint = 1.5 * 2f64.powi(52);
    let mut fn_ = x * INV_PIO2 + toint - toint;
    let mut n = fn_ as i32;
    let mut r = x - fn_ * PIO2_1;
    let mut w = fn_ * PIO2_1T;
    if r - w < -PIO4 {
        n -= 1;
        fn_ -= 1.0;
        r = x - fn_ * PIO2_1;
        w = fn_ * PIO2_1T;
    } else if r - w > PIO4 {
        n += 1;
        fn_ += 1.0;
        r = x - fn_ * PIO2_1;
        w = fn_ * PIO2_1T;
    }
    let mut y0 = r - w;
    let ey = ((y0.to_bits() >> 52) & 0x7ff) as i32;
    let ex = (ix >> 20) as i32;
    if ex - ey > 16 {
        let t = r;
        w = fn_ * PIO2_2;
        r = t - w;
        w = fn_ * PIO2_2T - ((t - r) - w);
        y0 = r - w;
        let ey = ((y0.to_bits() >> 52) & 0x7ff) as i32;
        if ex - ey > 49 {
            let t = r;
            w = fn_ * PIO2_3;
            r = t - w;
            w = fn_ * PIO2_3T - ((t - r) - w);
            y0 = r - w;
        }
    }
    let y1 = (r - y0) - w;
    if sign {
        (-n, -y0, -y1)
    } else {
        (n, y0, y1)
    }
}

fn rem_pio2(x: f64) -> (i32, f64, f64) {
    let bits = x.to_bits();
    let sign = bits >> 63 != 0;
    let ix = ((bits >> 32) as u32) & 0x7fff_ffff;
    if ix <= 0x400f_6a7a {
        if ix & 0xf_ffff == 0x9_21fb {
            return medium(x, ix, sign);
        }
        if ix <= 0x4002_d97c {
            if !sign {
                let z = x - PIO2_1;
                let y0 = z - PIO2_1T;
                return (1, y0, (z - y0) - PIO2_1T);
            }
            let z = x + PIO2_1;
            let y0 = z + PIO2_1T;
            return (-1, y0, (z - y0) + PIO2_1T);
        }
        if !sign {
            let z = x - 2.0 * PIO2_1;
            let y0 = z - 2.0 * PIO2_1T;
            return (2, y0, (z - y0) - 2.0 * PIO2_1T);
        }
        let z = x + 2.0 * PIO2_1;
        let y0 = z + 2.0 * PIO2_1T;
        return (-2, y0, (z - y0) + 2.0 * PIO2_1T);
    }
    if ix <= 0x401c_463b {
        if ix <= 0x4015_fdbc {
            if ix == 0x4012_d97c {
                return medium(x, ix, sign);
            }
            if !sign {
                let z = x - 3.0 * PIO2_1;
                let y0 = z - 3.0 * PIO2_1T;
                return (3, y0, (z - y0) - 3.0 * PIO2_1T);
            }
            let z = x + 3.0 * PIO2_1;
            let y0 = z + 3.0 * PIO2_1T;
            return (-3, y0, (z - y0) + 3.0 * PIO2_1T);
        }
        if ix == 0x4019_21fb {
            return medium(x, ix, sign);
        }
        if !sign {
            let z = x - 4.0 * PIO2_1;
            let y0 = z - 4.0 * PIO2_1T;
            return (4, y0, (z - y0) - 4.0 * PIO2_1T);
        }
        let z = x + 4.0 * PIO2_1;
        let y0 = z + 4.0 * PIO2_1T;
        return (-4, y0, (z - y0) + 4.0 * PIO2_1T);
    }
    medium(x, ix, sign)
}

pub fn sin(x: f64) -> f64 {
    let ix = high(x) as u32 & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        if ix < 0x3e50_0000 {
            return x;
        }
        return k_sin(x, 0.0, false);
    }
    if ix >= 0x7ff0_0000 {
        return x - x;
    }
    let (n, y0, y1) = rem_pio2(x);
    match n & 3 {
        0 => k_sin(y0, y1, true),
        1 => k_cos(y0, y1),
        2 => -k_sin(y0, y1, true),
        _ => -k_cos(y0, y1),
    }
}

pub fn cos(x: f64) -> f64 {
    let ix = high(x) as u32 & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        if ix < 0x3e46_a09e {
            return 1.0;
        }
        return k_cos(x, 0.0);
    }
    if ix >= 0x7ff0_0000 {
        return x - x;
    }
    let (n, y0, y1) = rem_pio2(x);
    match n & 3 {
        0 => k_cos(y0, y1),
        1 => -k_sin(y0, y1, true),
        2 => -k_cos(y0, y1),
        _ => k_sin(y0, y1, true),
    }
}
