//! `infer` runs the pinned SmolVLA weights and prints one JSON object.
//!
//! The action values come from the f32 forward. The wgpu block is a
//! `state_proj` check on the adapter this process can open. `gpu_energy`
//! is false. `robot_actuated` is false: the joint command is the first
//! row of the chunk after the clock bind, not a motor write.

use ferric_accept::smolvla::ChunkIdentity;
use ferric_smolvla::{
    ACTION_DIM, CAMERAS, Sample, encode_and_read_command, ensure_newline, f32_bytes, forward,
    load_stats, noise_from_seed, parse_joints, sha256_hex, synthetic_image, tokenize,
    wgpu_state_proj,
};
use std::env;

fn main() {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_default();
    if cmd != "infer" {
        eprintln!(
            "usage: ferric-smolvla infer --weights PATH --tokenizer PATH --stats PATH --joints PATH [--task STR] [--noise-seed U64] [--action-stats-key KEY]"
        );
        std::process::exit(2);
    }
    let args: Vec<String> = args.collect();
    let code = match run(&args) {
        Ok(json) => {
            println!("{json}");
            0
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    };
    std::process::exit(code);
}

fn parse_u64(s: &str) -> Result<u64, String> {
    if let Some(hex) = s.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).map_err(|e| e.to_string())
    } else {
        s.parse()
            .map_err(|e: std::num::ParseIntError| e.to_string())
    }
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn run(args: &[String]) -> Result<String, String> {
    let weights = flag(args, "--weights").ok_or("--weights")?;
    let tokenizer = flag(args, "--tokenizer").ok_or("--tokenizer")?;
    let stats = flag(args, "--stats").ok_or("--stats")?;
    let joints_path = flag(args, "--joints").ok_or("--joints")?;
    let task = flag(args, "--task").unwrap_or("pick the red cube");
    let task = ensure_newline(task);
    let seed = parse_u64(flag(args, "--noise-seed").unwrap_or("0x5a17"))?;
    let action_key = flag(args, "--action-stats-key");
    let t0_ns: u64 = flag(args, "--t0-ns")
        .unwrap_or("1000000000")
        .parse()
        .map_err(|e| format!("{e}"))?;
    let step_ns: u64 = flag(args, "--step-ns")
        .unwrap_or("20000000")
        .parse()
        .map_err(|e| format!("{e}"))?;
    let embodiment = flag(args, "--embodiment")
        .unwrap_or("caller-supplied")
        .to_string();

    eprintln!("hashing weights");
    let weight_bytes = std::fs::read(weights).map_err(|e| e.to_string())?;
    let weight_sha = sha256_hex(&weight_bytes);
    let expect = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb";
    if weight_sha != expect || weight_bytes.len() != 906_712_520 {
        return Err(format!(
            "weights are not the pinned file: {weight_sha} bytes {}",
            weight_bytes.len()
        ));
    }
    drop(weight_bytes);
    let stats_bytes = std::fs::read(stats).map_err(|e| e.to_string())?;
    let stats_sha = sha256_hex(&stats_bytes);
    let (token_ids, token_mask) = tokenize(tokenizer, &task)?;
    let mut images = Vec::new();
    let mut image_bytes = Vec::new();
    for cam in 0..CAMERAS {
        let img = synthetic_image(cam, 256, 256);
        image_bytes.extend(f32_bytes(&img));
        images.push(img);
    }
    let noise = if let Some(path) = flag(args, "--noise") {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        if bytes.len() != ferric_smolvla::NOISE_LEN * 4 {
            return Err("noise file must be 50*32 f32 little-endian".into());
        }
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        noise_from_seed(seed)
    };
    let noise_sha = sha256_hex(&f32_bytes(&noise));
    let image_sha = sha256_hex(&image_bytes);
    let (action_mean, action_std) = match action_key {
        Some(key) => {
            let (m, s) = load_stats(stats, key)?;
            (Some(m), Some(s))
        }
        None => (None, None),
    };
    eprintln!("loading weights and running the forward");
    let store = ferric_smolvla::weights::Store::open(weights)?;
    let sample = Sample {
        images,
        image_hw: vec![(256, 256); CAMERAS],
        state: vec![0.0; ACTION_DIM],
        token_ids,
        token_mask,
        noise,
        state_mean: None,
        state_std: None,
        action_mean,
        action_std,
    };
    let result = forward(&store, &sample)?;
    let joints = parse_joints(&std::fs::read(joints_path).map_err(|e| e.to_string())?)?;
    let identity = ChunkIdentity {
        model_id: "lerobot/smolvla_base".into(),
        weight_sha256: weight_sha.clone(),
        stats_sha256: stats_sha.clone(),
        noise_sha256: noise_sha.clone(),
        embodiment,
        t0_ns,
        step_duration_ns: step_ns,
        joints,
    };
    let encoded = encode_and_read_command(&result.actions, identity, t0_ns)?;
    eprintln!("checking state_proj on the wgpu adapter");
    let w = store.tensor("model.state_proj.weight")?;
    let b = store.tensor("model.state_proj.bias")?;
    let mut x = [0.0f32; 32];
    x[0] = 1.0;
    let cpu_proj = ferric_smolvla::project_state(&w, &b, &x);
    let check = match pollster::block_on(wgpu_state_proj(&w, &b, &x)) {
        Ok(c) => c,
        Err(e) => ferric_smolvla::wgpu_absent(e),
    };
    let mut actions = Vec::new();
    for t in 0..5 {
        let row: Vec<String> = result.actions[t * ACTION_DIM..(t + 1) * ACTION_DIM]
            .iter()
            .map(|v| format!("{v:.8}"))
            .collect();
        actions.push(row.join(","));
    }
    let body = serde_json::json!({
        "command": "ferric-smolvla infer",
        "model_id": "lerobot/smolvla_base",
        "weight_sha256": weight_sha,
        "stats_sha256": stats_sha,
        "stats_file_applied_to_action": action_key.is_some(),
        "action_stats_key": action_key,
        "state_norm": result.state_norm,
        "action_norm": result.action_norm,
        "task": task,
        "token_ids_prefix": sample.token_ids[..8],
        "image_sha256": image_sha,
        "image_pattern": "((y*17+x*3+cam*5+c*13)%251)/250 on 3x256x256 in [0,1]",
        "noise_sha256": noise_sha,
        "noise_generator": if flag(args, "--noise").is_some() { "file" } else { "xorshift64-box-muller" },
        "noise_seed": seed,
        "forward_backend": "ferric_core::cpu f32",
        "state_proj_0": cpu_proj[0],
        "wgpu_adapter": check.adapter,
        "wgpu_backend": check.backend,
        "wgpu_adapter_present": check.present,
        "wgpu_software_adapter": check.software,
        "wgpu_adapters": check.listed,
        "wgpu_state_proj_max_abs_diff": check.max_abs,
        "wgpu_error": check.error,
        "gpu_energy": false,
        "action_digest": encoded.digest,
        "action_chunk_bytes": encoded.bytes.len(),
        "n_action_steps": 50,
        "action_dim": ACTION_DIM,
        "actions_first5": actions,
        "actions": result.actions,
        "joint_command_index": 0,
        "joint_command": encoded.command,
        "joint_timestamp_ns": encoded.timestamp_ns,
        "robot_actuated": encoded.robot_actuated,
        "software_joint_command": true,
        "authorize_rollout": "ProposedEntryInvoked remains the ferric-accept refusal and was not called",
    });
    Ok(serde_json::to_string_pretty(&body).map_err(|e| e.to_string())?)
}
