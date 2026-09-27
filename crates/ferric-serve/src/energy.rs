//! **Joules per request** — what no serving peer reports (0 of 14 in the parity audit, 27 Sept 2026).
//!
//! A long-lived `macmon` sampler (Apple Silicon, sudoless, 100 ms) runs beside the server. Each generation
//! marks its window; when it finishes, the energy over the window is attributed to it:
//!
//! ```text
//!   joules = ∫ (P(t) − P_idle) / n(t) dt   over the request's window
//! ```
//!
//! - `P(t)` — the ACCELERATOR rails (GPU + DRAM), the boundary that has proved attributable on a shared
//!   machine (the CPU rail swings with other processes and has been seen reading 0). The SoC figure
//!   (CPU + GPU + ANE + DRAM) is reported beside it, and voided when its CPU rail drops out.
//! - `P_idle` — the mean over the last two minutes of samples when NO request was in flight, so a slowly
//!   drifting background is not charged to the model. None yet → only the gross window is reported.
//! - `n(t)` — requests in flight at that instant: concurrent requests share each instant equally.
//!
//! ⚠ `Derived`, not `Measured`: sampled watts, integrated (ferric_joule's own classification). The counters
//! are system-wide, so another process's draw that is not in the idle baseline is inside the figure.
//! ⛔ Guards from the ASR energy study (230edb9): a sample above 1 kW is a meter glitch and voids the
//! figure; it is never clipped into a number.
use serde_json::{json, Value};
use std::sync::Mutex;

pub(crate) struct Energy {
    meter: Option<ferric_joule::Macmon>,
    why_none: String,
    /// (id, t0, t1 or None while in flight)
    log: Mutex<(u64, Vec<(u64, f64, Option<f64>)>)>,
}

/// A request's claim on the meter: its id and when it began.
pub(crate) struct Ticket { id: u64, t0: f64 }

const CEILING_W: f64 = 1000.0;
const IDLE_LOOKBACK_S: f64 = 120.0;

impl Energy {
    pub fn start() -> Energy {
        let off = std::env::var("FERRIC_ENERGY").is_ok_and(|v| v == "0");
        let meter = if off { None } else { ferric_joule::Macmon::start_window(ferric_joule::MacmonScope::Soc, 100, 600.0) };
        let why_none = if off { "disabled (FERRIC_ENERGY=0)".to_string() }
            else if meter.is_none() { "no power meter on this machine (macmon not found; RAPL and NVML are not wired into the server yet)".to_string() }
            else { String::new() };
        Energy { meter, why_none, log: Mutex::new((0, Vec::new())) }
    }

    pub fn available(&self) -> bool { self.meter.is_some() }

    pub fn begin(&self) -> Option<Ticket> {
        let t0 = self.meter.as_ref()?.now_t()?;
        let mut g = self.log.lock().ok()?;
        g.0 += 1;
        let id = g.0;
        g.1.push((id, t0, None));
        // Forget windows that have left the meter's kept span.
        g.1.retain(|&(_, a, b)| b.is_none_or(|b| t0 - b < 900.0) || a > t0 - 900.0);
        Some(Ticket { id, t0 })
    }

    /// Close the window and attribute its energy. `tokens` = generated tokens, for joules per token.
    pub fn end(&self, ticket: Option<Ticket>, tokens: usize) -> Value {
        let Some(meter) = &self.meter else { return json!({"joules": null, "why": self.why_none}) };
        let Some(tk) = ticket else { return json!({"joules": null, "why": "the meter had produced no sample yet"}) };
        let Some(t1) = meter.now_t() else { return json!({"joules": null, "why": "no sample"}) };
        let windows: Vec<(f64, f64)> = {
            let Ok(mut g) = self.log.lock() else { return json!({"joules": null, "why": "lock"}) };
            if let Some(w) = g.1.iter_mut().find(|w| w.0 == tk.id) { w.2 = Some(t1); }
            g.1.iter().map(|&(_, a, b)| (a, b.unwrap_or(f64::INFINITY))).collect()
        };
        let samples = meter.trace();
        attribute(&samples, &windows, tk.t0, t1, tokens)
    }
}

/// The attribution, on a sample trace and the request windows (pure, so it is testable without a meter).
pub(crate) fn attribute(samples: &[ferric_joule::MacmonSample], windows: &[(f64, f64)], t0: f64, t1: f64, tokens: usize) -> Value {
    let accel = |s: &ferric_joule::MacmonSample| s.gpu + s.ram;
    let soc = |s: &ferric_joule::MacmonSample| s.cpu + s.gpu + s.ane + s.ram;
    let secs = t1 - t0;
    if secs <= 0.0 { return json!({"joules": null, "why": "empty window"}); }
    let busy = |t: f64| windows.iter().filter(|&&(a, b)| t >= a && t <= b).count();
    // In the window: every sample inside it, plus the edges (held from the nearest sample).
    let inside: Vec<&ferric_joule::MacmonSample> = samples.iter().filter(|s| s.t >= t0 && s.t <= t1).collect();
    let before = samples.iter().rev().find(|s| s.t < t0);
    let after = samples.iter().find(|s| s.t > t1);
    // Fewer than three samples inside the window is below what 100 ms sampling resolves: a live 0.1 s
    // request came out at −0.004 J, noise around the idle baseline. It is reported as unresolved, with
    // the gross window, rather than as a number that means nothing.
    if inside.len() < 3 || before.is_none() {
        let gross = inside.first().map(|s| (s.gpu + s.ram) * secs);
        return json!({"joules": null, "window_joules": gross.map(|g| (g * 1000.0).round() / 1000.0), "seconds": (secs * 1000.0).round() / 1000.0,
                      "why": format!("{secs:.3} s spans {} meter samples — too short to resolve at 100 ms", inside.len())});
    }
    let span: Vec<&ferric_joule::MacmonSample> = before.into_iter().chain(inside.iter().copied()).chain(after).collect();
    if let Some(bad) = span.iter().find(|s| s.cpu.max(s.gpu).max(s.ram).max(s.sys) > CEILING_W) {
        return json!({"joules": null, "why": format!("meter glitch: a sample read {:.0} W, above any machine this runs on", bad.cpu.max(bad.gpu).max(bad.ram).max(bad.sys))});
    }
    // Idle baseline: recent samples with nothing in flight (a second of settling after each window).
    let settled = |t: f64| !windows.iter().any(|&(a, b)| t >= a - 0.3 && t <= b + 1.0);
    let idle: Vec<&ferric_joule::MacmonSample> = samples.iter()
        .filter(|s| s.t >= t0 - IDLE_LOOKBACK_S && s.t < t1 && settled(s.t) && s.cpu.max(s.gpu).max(s.ram) <= CEILING_W).collect();
    let mean = |v: &[&ferric_joule::MacmonSample], f: &dyn Fn(&ferric_joule::MacmonSample) -> f64|
        if v.is_empty() { None } else { Some(v.iter().map(|s| f(s)).sum::<f64>() / v.len() as f64) };
    let (idle_a, idle_s) = if idle.len() >= 8 { (mean(&idle, &accel), mean(&idle, &soc)) } else { (None, None) };
    // How much the background itself moves between idle samples. On a shared machine another process's
    // GPU work comes and goes; while it is in the baseline but not in the window (or the reverse), the
    // difference is charged to — or credited against — the request.
    let idle_sd = idle_a.map(|m| (idle.iter().map(|s| (accel(s) - m).powi(2)).sum::<f64>() / idle.len() as f64).sqrt());
    // Piecewise-linear power with held edges, integrated by trapezoid; each instant shared by n(t).
    let at = |t: f64, f: &dyn Fn(&ferric_joule::MacmonSample) -> f64| -> f64 {
        let i = span.iter().rposition(|s| s.t <= t);
        let j = span.iter().position(|s| s.t >= t);
        match (i, j) {
            (Some(i), Some(j)) if span[j].t > span[i].t => {
                let f0 = f(span[i]); let f1 = f(span[j]);
                f0 + (t - span[i].t) / (span[j].t - span[i].t) * (f1 - f0)
            }
            (Some(i), _) => f(span[i]),
            (_, Some(j)) => f(span[j]),
            _ => 0.0,
        }
    };
    let mut pts: Vec<f64> = vec![t0];
    pts.extend(inside.iter().map(|s| s.t));
    pts.push(t1);
    let (mut gross_a, mut share_a, mut share_s, mut nmax) = (0.0, 0.0, 0.0, 1usize);
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let dt = b - a;
        if dt <= 0.0 { continue; }
        let n = busy(0.5 * (a + b)).max(1);
        nmax = nmax.max(n);
        let pa = 0.5 * (at(a, &accel) + at(b, &accel));
        let ps = 0.5 * (at(a, &soc) + at(b, &soc));
        gross_a += pa * dt;
        share_a += (pa - idle_a.unwrap_or(0.0)) * dt / n as f64;
        share_s += (ps - idle_s.unwrap_or(0.0)) * dt / n as f64;
    }
    // The SoC figure needs its CPU rail: this machine's macmon has reported it as exactly 0 throughout.
    let cpu_zero = span.iter().filter(|s| s.cpu == 0.0).count() * 2 > span.len();
    let r3 = |x: f64| (x * 1000.0).round() / 1000.0;
    // ⛔ Two refusals, found live with five other jobs compiling and benchmarking on the same GPU: four
    // requests in a row came out at −0.12 to −5.1 J. Energy below idle is not physics, it is a baseline
    // that moved. (1) a window that drew no more than the baseline says nothing; (2) a window whose excess
    // over idle is within twice the idle samples' own spread cannot be told apart from the background.
    let excess_w = idle_a.map(|i| gross_a / secs - i);
    let unresolved = match (excess_w, idle_sd) {
        (Some(x), _) if x <= 0.0 => Some(format!("the window averaged {:.2} W on the accelerator rails, no more than the idle baseline \
            ({:.2} W): the background changed during it, so this request's energy cannot be separated", gross_a / secs, idle_a.unwrap_or(0.0))),
        (Some(x), Some(sd)) if x < 2.0 * sd => Some(format!("this request drew {x:.2} W over idle but the idle samples themselves swing \
            ±{sd:.2} W (another process's load): unresolvable while the machine is shared")),
        _ => None,
    };
    let joules = if unresolved.is_some() { None } else { idle_a.map(|_| r3(share_a)) };
    json!({
        "joules": joules,
        "joules_per_token": joules.map(|j| if tokens > 0 { r3(j / tokens as f64) } else { 0.0 }),
        "window_joules": r3(gross_a),
        "idle_watts": idle_a.map(r3),
        "idle_sd_watts": idle_sd.map(r3),
        "seconds": r3(secs),
        "concurrent_max": nmax,
        "soc_joules": if cpu_zero || idle_s.is_none() { Value::Null } else { json!(r3(share_s)) },
        "boundary": "accelerator: GPU + DRAM rails",
        "class": "derived",
        "meter": "macmon, 100 ms samples, trapezoid",
        "why": if let Some(u) = unresolved { json!(u) } else if idle_a.is_none() { json!("no idle baseline yet (needs 8 samples with nothing in flight in the last 2 min): joules is null, window_joules is gross") }
               else if cpu_zero { json!("the CPU rail read 0 in most samples, so the SoC figure is withheld") } else { Value::Null },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn s(t: f64, gpu: f64) -> ferric_joule::MacmonSample {
        ferric_joule::MacmonSample { t, cpu: 5.0, gpu, ane: 0.0, ram: 1.0, sys: 20.0, cpu_temp: 0.0, gpu_temp: 0.0,
                                     pcpu_mhz: 0.0, ecpu_mhz: 0.0, gpu_mhz: 0.0, cpu_active: 0.0 }
    }

    /// 2 W idle, 12 W while working for 2 s: 20 J above idle — all of it to a lone request, half each to
    /// two that overlapped the whole window.
    #[test]
    fn marginal_joules_and_sharing_between_concurrent_requests() {
        let mut tr: Vec<_> = (0..100).map(|i| s(i as f64 * 0.1, 1.0)).collect(); // idle: gpu 1 + ram 1 = 2 W
        tr.extend((0..=20).map(|i| s(10.0 + i as f64 * 0.1, 11.0)));             // work: 12 W
        tr.extend((1..=20).map(|i| s(12.0 + i as f64 * 0.1, 1.0)));
        let one = attribute(&tr, &[(10.0, 12.0)], 10.0, 12.0, 10);
        assert_eq!(one["idle_sd_watts"].as_f64(), Some(0.0));
        assert!((one["joules"].as_f64().unwrap() - 20.0).abs() < 0.05, "{one}");
        assert_eq!(one["idle_watts"].as_f64().unwrap(), 2.0);
        assert!((one["joules_per_token"].as_f64().unwrap() - 2.0).abs() < 0.01);
        let two = attribute(&tr, &[(10.0, 12.0), (10.0, 12.0)], 10.0, 12.0, 10);
        assert!((two["joules"].as_f64().unwrap() - 10.0).abs() < 0.05, "{two}");
        assert_eq!(two["concurrent_max"], 2);
    }

    #[test]
    fn a_window_under_three_samples_is_unresolved_not_a_number() {
        let tr: Vec<_> = (0..120).map(|i| s(i as f64 * 0.1, 1.0)).collect();
        let v = attribute(&tr, &[(10.02, 10.17)], 10.02, 10.17, 3);
        assert!(v["joules"].is_null() && v["why"].as_str().unwrap().contains("too short"), "{v}");
    }

    #[test]
    fn a_glitch_sample_voids_the_figure_and_no_baseline_leaves_it_null() {
        let mut tr: Vec<_> = (0..100).map(|i| s(i as f64 * 0.1, 1.0)).collect();
        tr.push(s(10.05, 18_779.0));
        tr.extend((1..=10).map(|i| s(10.05 + i as f64 * 0.1, 11.0)));
        let v = attribute(&tr, &[(10.0, 11.0)], 10.0, 11.0, 5);
        assert!(v["joules"].is_null() && v["why"].as_str().unwrap().contains("glitch"), "{v}");
        let fresh: Vec<_> = (0..14).map(|i| s(i as f64 * 0.1, 11.0)).collect();
        let v = attribute(&fresh, &[(0.05, 1.15)], 0.05, 1.15, 5);
        assert!(v["joules"].is_null() && v["window_joules"].as_f64().unwrap() > 0.0, "{v}");
    }

    /// The live failure: a baseline taken while another process loaded the GPU, then a request window
    /// after it stopped — the request "saved" energy. And a background that swings more than the request.
    #[test]
    fn a_window_below_idle_or_inside_the_background_swing_is_refused_not_reported() {
        let mut tr: Vec<_> = (0..100).map(|i| s(i as f64 * 0.1, 19.0)).collect(); // idle 20 W (a neighbour busy)
        tr.extend((0..=20).map(|i| s(10.0 + i as f64 * 0.1, 9.0)));               // window 10 W
        let v = attribute(&tr, &[(10.0, 12.0)], 10.0, 12.0, 10);
        assert!(v["joules"].is_null() && v["why"].as_str().unwrap().contains("no more than the idle"), "{v}");
        assert!(v["window_joules"].as_f64().unwrap() > 0.0, "the gross figure is still reported");
        let mut tr: Vec<_> = (0..100).map(|i| s(i as f64 * 0.1, if i % 2 == 0 { 1.0 } else { 21.0 })).collect(); // 12 ± 10 W
        tr.extend((0..=20).map(|i| s(10.0 + i as f64 * 0.1, 14.0)));             // 15 W: +3 W
        let v = attribute(&tr, &[(10.0, 12.0)], 10.0, 12.0, 10);
        assert!(v["joules"].is_null() && v["why"].as_str().unwrap().contains("swing"), "{v}");
        assert!((v["idle_sd_watts"].as_f64().unwrap() - 10.0).abs() < 0.05, "{v}");
    }
}
