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

fn main() {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_default();
    let args: Vec<String> = args.collect();
    let result = match cmd.as_str() {
        "infer" => infer(&args),
        "network-forward" => network_forward(&args),
        "gpu" => gpu(&args),
        "simulate" => simulate(&args),
        _ => Err("usage: ferric-smolvla infer|network-forward|gpu|simulate".into()),
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
