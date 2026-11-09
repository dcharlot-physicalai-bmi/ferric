//! Commands:
//! `infer` refuses MEAN_STD execution unless a pinned embodiment map names
//! stats keys that exist. `network-forward` is the non-executable f32 tensor
//! path and writes the trace the reference compares. `gpu` dispatches that
//! same graph through wgpu and reads it back. `simulate` steps a codec-valid
//! chunk on the ferromotion clock and keeps `ProposedEntryInvoked`.

use ferric_accept::smolvla::ChunkIdentity;
use ferric_smolvla::{
    forward, forward_gpu, noise_from_seed, parse_embodiment_map, parse_joints, require_mean_std,
    sha256_hex, simulation_validation, synthetic_image, tokenize, Sample, ACTION_DIM, CAMERAS,
};
use std::env;
use std::fs;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn main() {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_default();
    let args: Vec<String> = args.collect();
    let result = match cmd.as_str() {
        "infer" => infer(&args),
        "network-forward" => network_forward(&args),
        "gpu" => gpu(&args),
        "simulate" => simulate(&args),
        "gpu-energy" => gpu_energy(&args),
        _ => Err("usage: ferric-smolvla infer|network-forward|gpu|gpu-energy|simulate".into()),
    };
    match result {
        Ok(json) => {
            println!("{json}");
            std::process::exit(0);
        }
        Err(e) => {
            if e.trim_start().starts_with('{') {
                println!("{e}");
            } else {
                eprintln!("{e}");
            }
            std::process::exit(1);
        }
    }
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn parse_u64(s: &str) -> Result<u64, String> {
    if let Some(hex) = s.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).map_err(|e| e.to_string())
    } else {
        s.parse().map_err(|e: std::num::ParseIntError| e.to_string())
    }
}

struct Case {
    weights: String,
    stats: String,
    tokenizer: String,
    task: String,
    seed: u64,
    images: Vec<Vec<f32>>,
    image_sha: String,
    state: Vec<f32>,
    noise: Vec<f32>,
    noise_sha: String,
    token_ids: Vec<u32>,
    token_mask: Vec<bool>,
    weight_sha: String,
    stats_sha: String,
}

fn load_case(args: &[String]) -> Result<Case, String> {
    let weights = flag(args, "--weights").ok_or("--weights")?.to_string();
    let stats = flag(args, "--stats").ok_or("--stats")?.to_string();
    let tokenizer = flag(args, "--tokenizer").ok_or("--tokenizer")?.to_string();
    let task = ferric_smolvla::ensure_newline(flag(args, "--task").unwrap_or("pick the red cube"));
    let seed = parse_u64(flag(args, "--noise-seed").unwrap_or("0x5a17"))?;
    let weight_bytes = fs::read(&weights).map_err(|e| e.to_string())?;
    let weight_sha = sha256_hex(&weight_bytes);
    if weight_sha != "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb" || weight_bytes.len() != 906_712_520 {
        return Err(format!("weights are not the pinned file: {weight_sha}"));
    }
    let stats_bytes = fs::read(&stats).map_err(|e| e.to_string())?;
    let stats_sha = sha256_hex(&stats_bytes);
    let (token_ids, token_mask) = tokenize(&tokenizer, &task)?;
    let mut images = Vec::new();
    let mut image_bytes = Vec::new();
    for cam in 0..CAMERAS {
        let img = synthetic_image(cam, 256, 256);
        image_bytes.extend(ferric_smolvla::f32_bytes(&img));
        images.push(img);
    }
    let noise = noise_from_seed(seed);
    Ok(Case {
        weights,
        stats,
        tokenizer,
        task,
        seed,
        image_sha: sha256_hex(&image_bytes),
        images,
        state: flag(args, "--state").map(parse_state).transpose()?.unwrap_or_else(|| vec![0.0; ACTION_DIM]),
        noise_sha: sha256_hex(&ferric_smolvla::f32_bytes(&noise)),
        noise,
        token_ids,
        token_mask,
        weight_sha,
        stats_sha,
    })
}

fn parse_state(s: &str) -> Result<Vec<f32>, String> {
    let v: Vec<f32> = s.split(',').map(|p| p.parse::<f32>().map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
    if v.len() != ACTION_DIM {
        return Err(format!("state must have {ACTION_DIM} values"));
    }
    Ok(v)
}

fn sample(case: &Case, allow: bool, capture: bool, state_mean: Option<Vec<f32>>, state_std: Option<Vec<f32>>, action_mean: Option<Vec<f32>>, action_std: Option<Vec<f32>>) -> Sample {
    Sample {
        images: case.images.clone(),
        image_hw: vec![(256, 256); CAMERAS],
        state: case.state.clone(),
        token_ids: case.token_ids.clone(),
        token_mask: case.token_mask.clone(),
        noise: case.noise.clone(),
        state_mean,
        state_std,
        action_mean,
        action_std,
        allow_unnormalized_network: allow,
        capture,
    }
}

fn infer(args: &[String]) -> Result<String, String> {
    let case = load_case(args)?;
    let Some(map_path) = flag(args, "--embodiment-map") else {
        return Err(serde_json::json!({
            "command": "ferric-smolvla infer",
            "executable": false,
            "reason": "MEAN_STD execution requires --embodiment-map. No action stats key is assumed.",
            "gpu_energy": false,
            "authorized_rollout": false,
            "robot_actuated": false,
        }).to_string());
    };
    let map = parse_embodiment_map(&fs::read(map_path).map_err(|e| e.to_string())?)?;
    let stats = fs::read(&case.stats).map_err(|e| e.to_string())?;
    if let Err(reason) = require_mean_std(&stats, &map.state_key, &map.action_key) {
        let body = serde_json::json!({
            "command": "ferric-smolvla infer",
            "executable": false,
            "forward_backend": "not_run",
            "input_class": "synthetic_protocol",
            "reason": reason,
            "embodiment_map": map_path,
            "state_key": map.state_key,
            "action_key": map.action_key,
            "weight_sha256": case.weight_sha,
            "stats_sha256": case.stats_sha,
            "gpu_energy": false,
            "robot_actuated": false,
            "authorized_rollout": false,
        });
        return Err(serde_json::to_string_pretty(&body).unwrap());
    }
    let (state_mean, state_std) = ferric_smolvla::load_stats(&case.stats, &map.state_key)?;
    let (action_mean, action_std) = ferric_smolvla::load_stats(&case.stats, &map.action_key)?;
    let store = ferric_smolvla::weights::Store::open(&case.weights)?;
    let result = forward(&store, &sample(&case, false, false, Some(state_mean), Some(state_std), Some(action_mean), Some(action_std)))?;
    if !result.executable {
        return Err("forward returned a non-executable normalization mode".into());
    }
    Ok(serde_json::json!({
        "command": "ferric-smolvla infer",
        "executable": true,
        "forward_backend": "ferric_core::cpu f32",
        "wgpu_dispatch": false,
        "input_class": "synthetic_protocol",
        "embodiment": map.embodiment,
        "state_key": map.state_key,
        "action_key": map.action_key,
        "state_norm": result.state_norm,
        "action_norm": result.action_norm,
        "weight_sha256": case.weight_sha,
        "stats_sha256": case.stats_sha,
        "actions": result.actions,
        "gpu_energy": false,
        "robot_actuated": false,
        "authorized_rollout": false,
    }).to_string())
}

fn network_forward(args: &[String]) -> Result<String, String> {
    let case = load_case(args)?;
    let trace_dir = flag(args, "--trace-dir").unwrap_or("/tmp/smolvla-rust").to_string();
    fs::create_dir_all(&trace_dir).map_err(|e| e.to_string())?;
    let store = ferric_smolvla::weights::Store::open(&case.weights)?;
    let result = forward(&store, &sample(&case, true, true, None, None, None, None))?;
    let trace = result.trace.ok_or("trace missing")?;
    write_f32(&trace_dir, "images", &case.images.concat())?;
    write_f32(&trace_dir, "state", &case.state)?;
    write_f32(&trace_dir, "noise", &case.noise)?;
    write_u32(&trace_dir, "tokens", &case.token_ids)?;
    fs::write(format!("{trace_dir}/mask.u8"), case.token_mask.iter().map(|m| u8::from(*m)).collect::<Vec<_>>()).map_err(|e| e.to_string())?;
    fs::write(format!("{trace_dir}/task.txt"), &case.task).map_err(|e| e.to_string())?;
    write_f32(&trace_dir, "prepared_image0", &trace.prepared_image0)?;
    write_f32(&trace_dir, "patch_token0", &trace.vision.patch_plus_pos_token0)?;
    write_f32(&trace_dir, "vision_l0_token0", &trace.vision.vision_layer0_token0)?;
    write_f32(&trace_dir, "vision_post_token0", &trace.vision.vision_post_token0)?;
    write_f32(&trace_dir, "connector_token0", &trace.vision.connector_token0)?;
    write_f32(&trace_dir, "state_token", &trace.state_token)?;
    write_f32(&trace_dir, "text_l0_k_token0", &trace.text_layer0_k_token0)?;
    write_f32(&trace_dir, "velocity0", &trace.velocity0)?;
    write_f32(&trace_dir, "raw_actions", &trace.raw_actions)?;
    Ok(serde_json::json!({
        "command": "ferric-smolvla network-forward",
        "executable": result.executable,
        "forward_backend": "ferric_core::cpu f32",
        "wgpu_dispatch": false,
        "input_class": "synthetic_protocol",
        "state_norm": result.state_norm,
        "action_norm": result.action_norm,
        "weight_sha256": case.weight_sha,
        "stats_sha256": case.stats_sha,
        "image_sha256": case.image_sha,
        "noise_sha256": case.noise_sha,
        "noise_seed": case.seed,
        "task": case.task,
        "tokenizer": case.tokenizer,
        "token_ids": case.token_ids,
        "trace_dir": trace_dir,
        "raw_action0": &result.raw_actions[..32],
        "gpu_energy": false,
        "robot_actuated": false,
        "authorized_rollout": false,
    }).to_string())
}

fn gpu(args: &[String]) -> Result<String, String> {
    let case = load_case(args)?;
    let store = ferric_smolvla::weights::Store::open(&case.weights)?;
    let cpu = forward(&store, &sample(&case, true, true, None, None, None, None))?;
    let gpu = match pollster::block_on(forward_gpu(&store, &sample(&case, true, true, None, None, None, None))) {
        Ok(v) => v,
        Err(e) => {
            return Err(serde_json::json!({
                "command": "ferric-smolvla gpu",
                "executable": false,
                "wgpu_dispatch_completed": false,
                "wgpu_error": e,
                "forward_backend": "ferric_core::cpu f32 was not labeled as the wgpu result",
                "gpu_energy": false,
                "input_class": "synthetic_protocol",
            }).to_string());
        }
    };
    let profile = error_profile(&cpu.raw_actions, &gpu.raw_actions);
    let cpu_trace = cpu.trace.as_ref().unwrap();
    Ok(serde_json::json!({
        "command": "ferric-smolvla gpu",
        "executable": false,
        "wgpu_dispatch_completed": true,
        "wgpu_adapter": gpu.adapter,
        "wgpu_backend": gpu.backend,
        "wgpu_software_adapter": gpu.software,
        "dispatches": gpu.dispatches,
        "completions": gpu.completions,
        "readbacks": gpu.readbacks,
        "host_index_moves": gpu.host_index_moves,
        "raw_action_max_abs_vs_cpu": profile.0,
        "raw_action_rmse_vs_cpu": profile.1,
        "connector_token0_max_abs_vs_cpu": error_profile(&cpu_trace.vision.connector_token0, &gpu.trace.vision.connector_token0).0,
        "velocity0_max_abs_vs_cpu": error_profile(&cpu_trace.velocity0, &gpu.trace.velocity0).0,
        "weight_sha256": case.weight_sha,
        "input_class": "synthetic_protocol",
        "gpu_energy": false,
        "robot_actuated": false,
        "authorized_rollout": false,
    }).to_string())
}

fn simulate(args: &[String]) -> Result<String, String> {
    let actions_path = flag(args, "--actions").ok_or("--actions")?;
    let joints_path = flag(args, "--joints").ok_or("--joints")?;
    let bytes = fs::read(actions_path).map_err(|e| e.to_string())?;
    let actions: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let joints = parse_joints(&fs::read(joints_path).map_err(|e| e.to_string())?)?;
    let t0 = parse_u64(flag(args, "--t0-ns").unwrap_or("1000000000"))?;
    let step = parse_u64(flag(args, "--step-ns").unwrap_or("20000000"))?;
    let id = ChunkIdentity {
        model_id: "lerobot/smolvla_base".into(),
        weight_sha256: flag(args, "--weight-sha").unwrap_or("7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb").to_string(),
        stats_sha256: flag(args, "--stats-sha").map(|s| s.to_string()).unwrap_or_else(|| "11".repeat(32)),
        noise_sha256: flag(args, "--noise-sha").map(|s| s.to_string()).unwrap_or_else(|| "22".repeat(32)),
        embodiment: flag(args, "--embodiment").unwrap_or("simulation-embodiment").into(),
        t0_ns: t0,
        step_duration_ns: step,
        joints,
    };
    let report = simulation_validation(&actions, id, t0)?;
    let steps: Vec<_> = report.steps.iter().map(|(i, ts, row)| serde_json::json!({"index": i, "timestamp_ns": ts, "command": row})).collect();
    Ok(serde_json::json!({
        "command": "ferric-smolvla simulate",
        "gate": report.gate,
        "authorized_rollout": report.authorized_rollout,
        "authorize_error": report.authorize_error,
        "robot_actuated": report.robot_actuated,
        "steps": steps,
        "gpu_energy": false,
    }).to_string())
}

fn error_profile(a: &[f32], b: &[f32]) -> (f32, f32) {
    let n = a.len().min(b.len()).max(1);
    let mut max_abs = 0.0f32;
    let mut sq = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        let d = (a[i] - b[i]).abs();
        max_abs = max_abs.max(d);
        sq += f64::from(d) * f64::from(d);
    }
    (max_abs, (sq / n as f64).sqrt() as f32)
}

fn write_f32(dir: &str, name: &str, v: &[f32]) -> Result<(), String> {
    fs::write(format!("{dir}/{name}.f32"), ferric_smolvla::f32_bytes(v)).map_err(|e| e.to_string())
}

fn write_u32(dir: &str, name: &str, v: &[u32]) -> Result<(), String> {
    let mut bytes = Vec::new();
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    fs::write(format!("{dir}/{name}.u32"), bytes).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// gpu-energy: the network-forward graph on a hardware wgpu adapter, with
// rootless sysfs counters around drained windows. Not an execution receipt.
// ---------------------------------------------------------------------------

struct Counter {
    label: &'static str,
    path: String,
    range_uj: u64,
}

fn read_u64(path: &str) -> Result<u64, String> {
    fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?.trim().parse::<u64>().map_err(|e| format!("{path}: {e}"))
}

fn find_powercap() -> Result<Vec<Counter>, String> {
    let mut out = Vec::new();
    for (zone, want) in [("/sys/class/powercap/intel-rapl:0", "package-0"), ("/sys/class/powercap/intel-rapl:0:0", "core")] {
        let name = fs::read_to_string(format!("{zone}/name")).map_err(|e| format!("{zone}/name: {e}"))?;
        if name.trim() != want {
            return Err(format!("{zone}/name is `{}`, expected `{want}`", name.trim()));
        }
        let path = format!("{zone}/energy_uj");
        read_u64(&path)?;
        let range_uj = read_u64(&format!("{zone}/max_energy_range_uj"))?;
        out.push(Counter { label: if want == "core" { "rapl-core" } else { "rapl-package" }, path, range_uj });
    }
    Ok(out)
}

fn hwmon_labeled(chip: &str, prefix: &str, label: &str) -> Option<String> {
    ferric_fleet::hwmon_input(std::path::Path::new("/sys/class/hwmon"), chip, prefix, label).map(|p| p.to_string_lossy().into_owned())
}

#[derive(Clone)]
struct Row {
    t_ns: u64,
    pkg_uj: u64,
    core_uj: u64,
    ppt_uw: Option<u64>,
    gpu_edge_mc: Option<u64>,
    cpu_tctl_mc: Option<u64>,
    gpu_sclk_hz: Option<u64>,
}

struct Paths {
    pkg: String,
    core: String,
    ppt: Option<String>,
    gpu_temp: Option<String>,
    cpu_temp: Option<String>,
    sclk: Option<String>,
}

fn sample_row(p: &Paths, origin: Instant) -> Result<Row, String> {
    let t_ns = origin.elapsed().as_nanos() as u64;
    let pkg_uj = read_u64(&p.pkg)?;
    let core_uj = read_u64(&p.core)?;
    let opt = |q: &Option<String>| q.as_ref().and_then(|x| read_u64(x).ok());
    Ok(Row { t_ns, pkg_uj, core_uj, ppt_uw: opt(&p.ppt), gpu_edge_mc: opt(&p.gpu_temp), cpu_tctl_mc: opt(&p.cpu_temp), gpu_sclk_hz: opt(&p.sclk) })
}

struct Window {
    label: String,
    phase: &'static str,
    start: Row,
    end: Row,
    wall_s: f64,
    pkg_j: f64,
    pkg_wrapped: bool,
    core_j: f64,
    core_wrapped: bool,
    ppt_j_sampled: Option<f64>,
    ppt_samples: usize,
    max_gpu_edge_c: Option<f64>,
    max_cpu_tctl_c: Option<f64>,
    mean_sclk_mhz: Option<f64>,
    digest: Option<String>,
    dispatches: Option<u32>,
    completions: Option<u32>,
    readbacks: Option<u32>,
}

fn close_window(label: String, phase: &'static str, start: Row, end: Row, rows: &[Row], pkg_range: u64, core_range: u64) -> Result<Window, String> {
    let (dp, wp) = ferric_fleet::delta_uj(start.pkg_uj, end.pkg_uj, pkg_range)?;
    let (dc, wc) = ferric_fleet::delta_uj(start.core_uj, end.core_uj, core_range)?;
    let inside: Vec<&Row> = rows.iter().filter(|r| r.t_ns >= start.t_ns && r.t_ns <= end.t_ns).collect();
    // Trapezoid over the 10 ms power samples, clipped to the window edges.
    let mut series: Vec<(u64, f64)> = Vec::new();
    if let Some(w) = start.ppt_uw { series.push((start.t_ns, w as f64 * 1e-6)); }
    for r in &inside { if let Some(w) = r.ppt_uw { series.push((r.t_ns, w as f64 * 1e-6)); } }
    if let Some(w) = end.ppt_uw { series.push((end.t_ns, w as f64 * 1e-6)); }
    let ppt_j = ferric_fleet::trapezoid_joules(&series);
    let maxc = |f: &dyn Fn(&Row) -> Option<u64>| inside.iter().filter_map(|r| f(r)).max().map(|v| v as f64 / 1000.0);
    let sclk: Vec<f64> = inside.iter().filter_map(|r| r.gpu_sclk_hz).map(|v| v as f64 / 1e6).collect();
    Ok(Window {
        label,
        phase,
        wall_s: (end.t_ns - start.t_ns) as f64 * 1e-9,
        pkg_j: dp as f64 * 1e-6,
        pkg_wrapped: wp,
        core_j: dc as f64 * 1e-6,
        core_wrapped: wc,
        ppt_j_sampled: ppt_j,
        ppt_samples: series.len(),
        max_gpu_edge_c: maxc(&|r| r.gpu_edge_mc),
        max_cpu_tctl_c: maxc(&|r| r.cpu_tctl_mc),
        mean_sclk_mhz: if sclk.is_empty() { None } else { Some(sclk.iter().sum::<f64>() / sclk.len() as f64) },
        start,
        end,
        digest: None,
        dispatches: None,
        completions: None,
        readbacks: None,
    })
}

fn window_json(w: &Window) -> serde_json::Value {
    serde_json::json!({
        "label": w.label,
        "phase": w.phase,
        "t_start_monotonic_ns": w.start.t_ns,
        "t_end_monotonic_ns": w.end.t_ns,
        "wall_s": w.wall_s,
        "raw": {
            "rapl_package_energy_uj_start": w.start.pkg_uj,
            "rapl_package_energy_uj_end": w.end.pkg_uj,
            "rapl_core_energy_uj_start": w.start.core_uj,
            "rapl_core_energy_uj_end": w.end.core_uj,
            "amdgpu_ppt_power1_input_uw_start": w.start.ppt_uw,
            "amdgpu_ppt_power1_input_uw_end": w.end.ppt_uw,
        },
        "rapl_package_j": w.pkg_j,
        "rapl_package_wrapped": w.pkg_wrapped,
        "rapl_core_j": w.core_j,
        "rapl_core_wrapped": w.core_wrapped,
        "amdgpu_ppt_j_sampled_integral": w.ppt_j_sampled,
        "amdgpu_ppt_samples": w.ppt_samples,
        "max_gpu_edge_c": w.max_gpu_edge_c,
        "max_cpu_tctl_c": w.max_cpu_tctl_c,
        "mean_gpu_sclk_mhz": w.mean_sclk_mhz,
        "raw_action_sha256": w.digest,
        "dispatches": w.dispatches,
        "completions": w.completions,
        "readbacks": w.readbacks,
    })
}

fn stats(v: &[f64]) -> serde_json::Value {
    if v.is_empty() {
        return serde_json::json!({"n": 0});
    }
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = if v.len() > 1 { v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0) } else { f64::NAN };
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    serde_json::json!({"n": v.len(), "mean": mean, "stddev_sample": var.sqrt(), "min": s[0], "max": s[s.len() - 1]})
}

fn write_trace(dir: &str, case: &Case, t: &ferric_smolvla::ActivationTrace) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    write_f32(dir, "images", &case.images.concat())?;
    write_f32(dir, "state", &case.state)?;
    write_f32(dir, "noise", &case.noise)?;
    write_u32(dir, "tokens", &case.token_ids)?;
    fs::write(format!("{dir}/mask.u8"), case.token_mask.iter().map(|m| u8::from(*m)).collect::<Vec<_>>()).map_err(|e| e.to_string())?;
    fs::write(format!("{dir}/task.txt"), &case.task).map_err(|e| e.to_string())?;
    write_f32(dir, "prepared_image0", &t.prepared_image0)?;
    write_f32(dir, "patch_token0", &t.vision.patch_plus_pos_token0)?;
    write_f32(dir, "vision_l0_token0", &t.vision.vision_layer0_token0)?;
    write_f32(dir, "vision_post_token0", &t.vision.vision_post_token0)?;
    write_f32(dir, "connector_token0", &t.vision.connector_token0)?;
    write_f32(dir, "state_token", &t.state_token)?;
    write_f32(dir, "text_l0_k_token0", &t.text_layer0_k_token0)?;
    write_f32(dir, "velocity0", &t.velocity0)?;
    write_f32(dir, "raw_actions", &t.raw_actions)?;
    Ok(())
}

fn stage_profile(a: &ferric_smolvla::ActivationTrace, b: &ferric_smolvla::ActivationTrace) -> serde_json::Value {
    let e = |x: &[f32], y: &[f32]| {
        let (m, r) = error_profile(x, y);
        serde_json::json!({"max_abs": m, "rmse": r, "n": x.len().min(y.len()), "len_a": x.len(), "len_b": y.len()})
    };
    serde_json::json!({
        "prepared_image0": e(&a.prepared_image0, &b.prepared_image0),
        "patch_token0": e(&a.vision.patch_plus_pos_token0, &b.vision.patch_plus_pos_token0),
        "vision_l0_token0": e(&a.vision.vision_layer0_token0, &b.vision.vision_layer0_token0),
        "vision_post_token0": e(&a.vision.vision_post_token0, &b.vision.vision_post_token0),
        "connector_token0": e(&a.vision.connector_token0, &b.vision.connector_token0),
        "state_token": e(&a.state_token, &b.state_token),
        "text_l0_k_token0": e(&a.text_layer0_k_token0, &b.text_layer0_k_token0),
        "velocity0": e(&a.velocity0, &b.velocity0),
        "raw_actions": e(&a.raw_actions, &b.raw_actions),
    })
}

fn gpu_energy(args: &[String]) -> Result<String, String> {
    let out_dir = flag(args, "--out-dir").ok_or("--out-dir")?.to_string();
    let trials: usize = flag(args, "--trials").ok_or("--trials")?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
    let warmup: usize = flag(args, "--warmup").ok_or("--warmup")?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
    let baseline_s: f64 = ferric_fleet::duration_is_explicit(flag(args, "--baseline-s").and_then(|s| s.parse().ok()))?;
    let period_ms: u64 = flag(args, "--sample-ms").ok_or("--sample-ms")?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
    if trials < 2 || warmup < 1 || period_ms == 0 {
        return Err("need --trials >= 2, --warmup >= 1, --sample-ms > 0".into());
    }
    fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    let counters = find_powercap()?;
    let ppt = hwmon_labeled("amdgpu", "power", "PPT").filter(|p| read_u64(p).is_ok());
    let paths = Arc::new(Paths {
        pkg: counters[0].path.clone(),
        core: counters[1].path.clone(),
        ppt: ppt.clone(),
        gpu_temp: hwmon_labeled("amdgpu", "temp", "edge").filter(|p| read_u64(p).is_ok()),
        cpu_temp: hwmon_labeled("k10temp", "temp", "Tctl").filter(|p| read_u64(p).is_ok()),
        sclk: hwmon_labeled("amdgpu", "freq", "sclk").filter(|p| read_u64(p).is_ok()),
    });
    let ppt_label = if paths.ppt.is_some() { Some("PPT".to_string()) } else { None };
    let (pkg_range, core_range) = (counters[0].range_uj, counters[1].range_uj);

    let case = load_case(args)?;
    let store = ferric_smolvla::weights::Store::open(&case.weights)?;
    let origin = Instant::now();
    let wall_epoch_ns = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);

    // Sampler: one row per period for the whole run. Reads only sysfs.
    let rows: Arc<Mutex<Vec<Row>>> = Arc::new(Mutex::new(Vec::new()));
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = {
        let (rows, stop, paths) = (rows.clone(), stop.clone(), paths.clone());
        std::thread::spawn(move || {
            let mut next = Instant::now();
            while !stop.load(Ordering::Relaxed) {
                if let Ok(r) = sample_row(&paths, origin) { rows.lock().unwrap().push(r); }
                next += Duration::from_millis(period_ms);
                let now = Instant::now();
                if next > now { std::thread::sleep(next - now); } else { next = now; }
            }
        })
    };
    let snap = || rows.lock().unwrap().clone();

    eprintln!("cpu f32 reference forward");
    let cpu_t0 = Instant::now();
    let cpu = forward(&store, &sample(&case, true, true, None, None, None, None))?;
    let cpu_wall = cpu_t0.elapsed().as_secs_f64();
    let cpu_trace = cpu.trace.as_ref().ok_or("cpu trace missing")?;
    write_trace(&format!("{out_dir}/trace-cpu"), &case, cpu_trace)?;
    let cpu_digest = ferric_fleet::digest_f32(&cpu.raw_actions);

    let mut session = pollster::block_on(ferric_smolvla::GpuSession::open())?;
    let adapter = session.ctx.adapter_name.clone();
    if session.software {
        return Err(format!("adapter `{adapter}` is software; no GPU energy run"));
    }
    ferric_fleet::gpu_receipt_allowed(&adapter)?;
    let drain = ferric_fleet::Drain { submit_empty: true, poll_wait: true, readback: true };
    ferric_fleet::require_drains(ferric_fleet::Phase::SteadyInfer, drain, drain)?;

    let mut windows: Vec<Window> = Vec::new();
    let s = sample(&case, true, true, None, None, None, None);
    let mut traces = Vec::new();

    // Window: first forward = weight upload + pipeline compile + first inference.
    let mut run = |session: &mut ferric_smolvla::GpuSession, label: String, phase: &'static str| -> Result<(Window, ferric_smolvla::GpuForward), String> {
        session.drain();
        let a = sample_row(&paths, origin)?;
        let g = pollster::block_on(session.forward(&store, &s))?;
        session.drain();
        let b = sample_row(&paths, origin)?;
        let mut w = close_window(label, phase, a, b, &snap(), pkg_range, core_range)?;
        w.digest = Some(ferric_fleet::digest_f32(&g.raw_actions));
        w.dispatches = Some(g.dispatches);
        w.completions = Some(g.completions);
        w.readbacks = Some(g.readbacks);
        Ok((w, g))
    };

    eprintln!("load+compile+first forward on {adapter}");
    let (w, g) = run(&mut session, "load_compile_first".into(), "load+compile")?;
    windows.push(w);
    traces.push(g);
    for i in 0..warmup {
        eprintln!("warmup {i}");
        let (w, g) = run(&mut session, format!("warmup{i}"), "warmup")?;
        windows.push(w);
        traces.push(g);
    }
    // Baseline A1: drained, idle, same sampler.
    let base = |session: &ferric_smolvla::GpuSession, label: &str| -> Result<Window, String> {
        session.drain();
        let a = sample_row(&paths, origin)?;
        session_idle_sleep(baseline_s);
        session.drain();
        let b = sample_row(&paths, origin)?;
        close_window(label.into(), "idle-baseline", a, b, &snap(), pkg_range, core_range)
    };
    eprintln!("baseline A1 {baseline_s} s");
    windows.push(base(&session, "baseline_A1")?);
    for i in 0..trials {
        eprintln!("trial {i}");
        let (w, g) = run(&mut session, format!("trial{i}"), "steady-infer")?;
        windows.push(w);
        traces.push(g);
    }
    eprintln!("baseline A2 {baseline_s} s");
    windows.push(base(&session, "baseline_A2")?);
    stop.store(true, Ordering::Relaxed);
    let _ = sampler.join();
    let all_rows = snap();

    // Raw sampler log.
    let mut csv = fs::File::create(format!("{out_dir}/sampler.csv")).map_err(|e| e.to_string())?;
    writeln!(csv, "t_monotonic_ns,rapl_package_energy_uj,rapl_core_energy_uj,amdgpu_ppt_power1_input_uw,amdgpu_edge_millic,k10temp_tctl_millic,amdgpu_sclk_hz").map_err(|e| e.to_string())?;
    let o = |v: Option<u64>| v.map(|x| x.to_string()).unwrap_or_default();
    for r in &all_rows {
        writeln!(csv, "{},{},{},{},{},{},{}", r.t_ns, r.pkg_uj, r.core_uj, o(r.ppt_uw), o(r.gpu_edge_mc), o(r.cpu_tctl_mc), o(r.gpu_sclk_hz)).map_err(|e| e.to_string())?;
    }

    // Parity: the last measured trial's trace vs CPU f32; digests across all GPU runs.
    let last = traces.last().unwrap();
    write_trace(&format!("{out_dir}/trace-gpu"), &case, &last.trace)?;
    let digests: Vec<String> = windows.iter().filter_map(|w| w.digest.clone()).collect();
    let deterministic = digests.windows(2).all(|p| p[0] == p[1]);

    let trial_w: Vec<&Window> = windows.iter().filter(|w| w.phase == "steady-infer").collect();
    let a1 = windows.iter().find(|w| w.label == "baseline_A1").unwrap();
    let a2 = windows.iter().find(|w| w.label == "baseline_A2").unwrap();
    let bp = |f: &dyn Fn(&Window) -> Option<f64>| -> Option<(f64, f64, f64)> {
        let p1 = f(a1)? / a1.wall_s;
        let p2 = f(a2)? / a2.wall_s;
        Some((p1, p2, (p1 + p2) / 2.0))
    };
    let mut derived = serde_json::Map::new();
    let wall: Vec<f64> = trial_w.iter().map(|w| w.wall_s).collect();
    derived.insert("wall_s_per_inference".into(), stats(&wall));
    for (key, f) in [
        ("rapl_package", Box::new(|w: &Window| Some(w.pkg_j)) as Box<dyn Fn(&Window) -> Option<f64>>),
        ("rapl_core", Box::new(|w: &Window| Some(w.core_j))),
        ("amdgpu_ppt_sampled", Box::new(|w: &Window| w.ppt_j_sampled)),
    ] {
        let gross: Vec<f64> = trial_w.iter().filter_map(|w| f(w)).collect();
        if gross.len() != trial_w.len() { continue; }
        let Some((p1, p2, pb)) = bp(&*f) else { continue };
        let inc: Vec<f64> = trial_w.iter().map(|w| f(w).unwrap() - pb * w.wall_s).collect();
        let avg_w: Vec<f64> = trial_w.iter().map(|w| f(w).unwrap() / w.wall_s).collect();
        derived.insert(key.into(), serde_json::json!({
            "gross_j_per_inference": stats(&gross),
            "mean_power_w_during_inference": stats(&avg_w),
            "baseline_power_w_A1": p1,
            "baseline_power_w_A2": p2,
            "baseline_power_w_mean": pb,
            "baseline_drift_w": p2 - p1,
            "incremental_j_per_inference": stats(&inc),
        }));
    }
    let body = serde_json::json!({
        "command": "ferric-smolvla gpu-energy",
        "executable": false,
        "input_class": "synthetic_protocol",
        "wgpu_adapter": adapter,
        "wgpu_backend": format!("{:?}", session.ctx.backend),
        "wgpu_software_adapter": session.software,
        "wgpu_adapters_listed": session.listed,
        "wgpu_subgroups": session.ctx.subgroups,
        "wgpu_max_storage_binding": session.ctx.max_binding,
        "weight_sha256": case.weight_sha,
        "stats_sha256": case.stats_sha,
        "image_sha256": case.image_sha,
        "noise_sha256": case.noise_sha,
        "noise_seed": case.seed,
        "task": case.task,
        "token_ids": case.token_ids,
        "cpu_f32_wall_s": cpu_wall,
        "cpu_raw_action_sha256": cpu_digest,
        "gpu_raw_action_sha256": digests.last(),
        "gpu_digest_identical_across_all_runs": deterministic,
        "gpu_vs_cpu_stage_profile": stage_profile(cpu_trace, &last.trace),
        "dispatches": last.dispatches,
        "completions": last.completions,
        "readbacks": last.readbacks,
        "host_index_moves": last.host_index_moves,
        "drain": "queue.submit(empty) then device.poll(wait_indefinitely) before and after every window; raw action chunk on host before the closing drain",
        "sampling": {
            "boundary": "RAPL energy_uj read at each window edge after the drain",
            "background_period_ms": period_ms,
            "rows": all_rows.len(),
            "wall_epoch_ns_at_origin": wall_epoch_ns,
        },
        "domains": {
            "rapl_package": {"path": counters[0].path, "max_energy_range_uj": pkg_range, "boundary": ferric_fleet::Domain::RaplPackage.boundary()},
            "rapl_core": {"path": counters[1].path, "max_energy_range_uj": core_range, "boundary": ferric_fleet::Domain::RaplCore.boundary()},
            "amdgpu_ppt": {"path": paths.ppt, "label": ppt_label, "kind": "instantaneous power, trapezoid-integrated at the sampler period; not an energy counter", "boundary": ferric_fleet::Domain::RocmPpt.boundary()},
            "amdgpu_edge_temp": paths.gpu_temp,
            "k10temp_tctl": paths.cpu_temp,
            "amdgpu_sclk": paths.sclk,
            "summed": false,
        },
        "baseline_s": baseline_s,
        "trials": trials,
        "warmup": warmup,
        "windows": windows.iter().map(window_json).collect::<Vec<_>>(),
        "derived": derived,
        "attributable": false,
        "attributable_reason": "gpu-energy does not apply the quiet-host gate; ferric-fleet energy-drive does",
        "gpu_energy": "measured package/core RAPL and sampled PPT around drained wgpu windows; isolated GPU energy is not claimed",
        "robot_actuated": false,
        "authorized_rollout": false,
    });
    let text = serde_json::to_string_pretty(&body).unwrap();
    fs::write(format!("{out_dir}/gpu-energy.json"), &text).map_err(|e| e.to_string())?;
    Ok(text)
}

fn session_idle_sleep(s: f64) {
    std::thread::sleep(Duration::from_secs_f64(s));
}
