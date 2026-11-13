use ferric_fleet::{JOINT_STEP_EXECUTED, MODEL_RECEIPTS, WGPU_DRAIN, artifacts, url, verify_bytes};
use std::env;
use std::fs;
use std::process::Command;

fn main() {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "plan".into());
    let code = match cmd.as_str() {
        "plan" => plan(),
        "acquire" => acquire(&args.collect::<Vec<_>>()),
        "preflight" => preflight(&args.collect::<Vec<_>>()),
        "check-bytes" => check_bytes(&args.collect::<Vec<_>>()),
        "infer" => infer(&args.collect::<Vec<_>>()),
        "energy-drive" => energy_drive(&args.collect::<Vec<_>>()),
        other => {
            eprintln!("unknown command `{other}`");
            eprintln!(
                "commands: plan | acquire --dir DIR | preflight --adapter-name NAME | check-bytes PATH ARTIFACT | infer --weights PATH | energy-drive [flags] -- WORKLOAD"
            );
            2
        }
    };
    std::process::exit(code);
}

fn plan() -> i32 {
    println!("model_receipts {MODEL_RECEIPTS}");
    println!("joint_step_executed {JOINT_STEP_EXECUTED}");
    println!("wgpu_drain {}", WGPU_DRAIN.join(" ; "));
    println!("windows load | compile | warmup | steady-infer | joint-step");
    println!("steady-infer and joint-step stay refused until a worker attaches a real forward.");
    println!("do not pass llvmpipe, lavapipe, or swiftshader to preflight.");
    for a in artifacts() {
        println!("pin {} {} {} {}", a.path, a.size, a.sha256, url(a.path));
    }
    0
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).map(String::as_str))
}

fn acquire(args: &[String]) -> i32 {
    let Some(dir) = flag(args, "--dir") else {
        eprintln!("acquire needs --dir DIR");
        return 2;
    };
    if let Err(e) = fs::create_dir_all(dir) {
        eprintln!("{e}");
        return 2;
    }
    let mut failed = false;
    for a in artifacts() {
        let dest = format!("{dir}/{}", a.path);
        let status = Command::new("curl")
            .args(["-fsSL", "-o", &dest, &url(a.path)])
            .status();
        match status {
            Ok(s) if s.success() => match fs::read(&dest) {
                Ok(bytes) => match verify_bytes(a.path, &bytes) {
                    Ok(source) => println!("ok {} {} {}", a.path, a.sha256, source),
                    Err(e) => {
                        eprintln!("{e}");
                        failed = true;
                    }
                },
                Err(e) => {
                    eprintln!("{dest}: {e}");
                    failed = true;
                }
            },
            Ok(s) => {
                eprintln!("curl exit {} for {}", s.code().unwrap_or(1), a.path);
                failed = true;
            }
            Err(e) => {
                eprintln!("curl: {e}");
                failed = true;
            }
        }
    }
    if failed { 1 } else { 0 }
}

fn preflight(args: &[String]) -> i32 {
    let Some(name) = flag(args, "--adapter-name") else {
        eprintln!("preflight needs --adapter-name");
        return 2;
    };
    match ferric_fleet::gpu_receipt_allowed(name) {
        Ok(()) => {
            println!("gpu_receipt_allowed adapter={name}");
            println!("this does not read the machine; the worker must pass the wgpu adapter name");
            0
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

fn infer(args: &[String]) -> i32 {
    let Some(path) = flag(args, "--weights") else {
        eprintln!("infer needs --weights PATH");
        return 2;
    };
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{path}: {e}");
            return 2;
        }
    };
    let hash = ferric_fleet::artifacts()
        .into_iter()
        .find(|a| a.path == "model.safetensors")
        .unwrap();
    if bytes.len() as u64 != hash.size {
        eprintln!("weights size {} != pinned {}", bytes.len(), hash.size);
        return 1;
    }
    // Hash the file the worker passed. This is the full-file check.
    let got = {
        // Reuse verify only when the path name matches; size already checked.
        // Compute sha256 via a second read of the pin helper by calling verify_bytes
        // only if we name it model.safetensors.
        match ferric_fleet::verify_bytes("model.safetensors", &bytes) {
            Ok(_) => hash.sha256.to_string(),
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        }
    };
    let mut state = [0f32; 32];
    state[0] = 1.0;
    let y = match ferric_fleet::state_proj(&bytes, &state) {
        Ok(y) => y,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let digest = ferric_fleet::digest_f32(&y);
    println!("backend cpu");
    println!("adapter none");
    println!("gpu_energy false");
    println!("weight_sha256 {got}");
    println!("executed state_proj");
    println!("state_proj_digest {digest}");
    println!("state_proj_0 {:.8}", y[0]);
    println!("action_chunk refused");
    println!("blocker {}", ferric_fleet::ACTION_CHUNK_BLOCKER);
    println!("model_receipts {}", ferric_fleet::MODEL_RECEIPTS);
    println!("joint_step_executed {}", ferric_fleet::JOINT_STEP_EXECUTED);
    1
}

fn energy_drive(args: &[String]) -> i32 {
    let mut flags = Vec::new();
    let mut workload = Vec::new();
    let mut after = false;
    for arg in args {
        if after {
            workload.push(arg.clone());
        } else if arg == "--" {
            after = true;
        } else {
            flags.push(arg.clone());
        }
    }
    let flag = |name: &str| {
        flags.iter().position(|a| a == name).and_then(|i| flags.get(i + 1)).map(String::as_str)
    };
    let Some(out) = flag("--out-dir") else {
        eprintln!("energy-drive needs --out-dir and -- WORKLOAD");
        return 2;
    };
    let meter = flag("--meter").unwrap_or("rapl");
    if meter == "macmon" {
        return energy_drive_macmon(&flags, &workload, out);
    }
    let Some(pkg) = flag("--rapl-package") else {
        eprintln!("energy-drive needs --rapl-package");
        return 2;
    };
    let Some(core) = flag("--rapl-core") else {
        eprintln!("energy-drive needs --rapl-core");
        return 2;
    };
    if workload.is_empty() {
        eprintln!("energy-drive needs a workload after --");
        return 2;
    }
    let pin = flag("--pin-cores").unwrap_or("0,1,2,3,16,17,18,19");
    let cores = match ferric_fleet::parse_cpu_list(pin) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("--pin-cores: {e}");
            return 2;
        }
    };
    let deadline_s: f64 = flag("--deadline-s").and_then(|s| s.parse().ok()).unwrap_or(6.0 * 60.0 * 60.0);
    let retry_s: f64 = flag("--retry-wait-s").and_then(|s| s.parse().ok()).unwrap_or(120.0);
    let settle_s: f64 = flag("--settle-s").and_then(|s| s.parse().ok()).unwrap_or(ferric_fleet::SETTLE_S);
    let load1_max: f64 = flag("--load1-max").and_then(|s| s.parse().ok()).unwrap_or(ferric_fleet::LOAD1_MAX);
    let busy_max: f64 = flag("--other-core-mean").or_else(|| flag("--busy-fraction-max")).and_then(|s| s.parse().ok()).unwrap_or(ferric_fleet::OTHER_CORE_MEAN_MAX);
    let lock = flag("--lock").unwrap_or(ferric_fleet::LOCK_PATH);
    let hwmon = std::path::Path::new("/sys/class/hwmon");
    let cfg = ferric_fleet::LiveConfig {
        rapl_package: pkg.into(),
        rapl_core: core.into(),
        ppt: ferric_fleet::hwmon_input(hwmon, "amdgpu", "power", "PPT"),
        cpu_tctl: ferric_fleet::hwmon_input(hwmon, "k10temp", "temp", "Tctl"),
        gpu_edge: ferric_fleet::hwmon_input(hwmon, "amdgpu", "temp", "edge"),
        settle_s,
        load1_max,
        busy_fraction_max: busy_max,
        pin_cores: cores,
        workload,
    };
    let out_dir = std::path::PathBuf::from(out);
    let lock_wait_s: f64 = flag("--lock-wait-s").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    match ferric_fleet::run_until_quiet(
        std::path::Path::new(lock),
        &out_dir,
        std::time::Duration::from_secs_f64(deadline_s),
        std::time::Duration::from_secs_f64(retry_s),
        lock_wait_s,
        |guard| ferric_fleet::run_locked_attempt(&cfg, guard),
    ) {
        Ok(receipt) => {
            println!("{}", serde_json::to_string_pretty(&ferric_fleet::receipt_json(&receipt)).unwrap());
            if receipt.attributable { 0 } else { 1 }
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

fn energy_drive_macmon(flags: &[String], workload: &[String], out: &str) -> i32 {
    let flag = |name: &str| {
        flags.iter().position(|a| a == name).and_then(|i| flags.get(i + 1)).map(String::as_str)
    };
    if workload.is_empty() {
        eprintln!("energy-drive needs a workload after --");
        return 2;
    }
    let deadline_s: f64 = flag("--deadline-s").and_then(|s| s.parse().ok()).unwrap_or(6.0 * 60.0 * 60.0);
    let retry_s: f64 = flag("--retry-wait-s").and_then(|s| s.parse().ok()).unwrap_or(120.0);
    let settle_s: f64 = flag("--settle-s").and_then(|s| s.parse().ok()).unwrap_or(ferric_fleet::SETTLE_S);
    let load1_max: f64 = flag("--load1-max").and_then(|s| s.parse().ok()).unwrap_or(ferric_fleet::LOAD1_MAX);
    let sample_ms: u64 = flag("--sample-ms").and_then(|s| s.parse().ok()).unwrap_or(1000);
    let lock_wait_s: f64 = flag("--lock-wait-s").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let lock = flag("--lock").unwrap_or(ferric_fleet::LOCK_PATH);
    let cfg = ferric_fleet::MacmonConfig {
        settle_s,
        load1_max,
        sample_ms,
        macmon_bin: flag("--macmon-bin").unwrap_or("macmon").to_string(),
        workload: workload.to_vec(),
    };
    match ferric_fleet::run_until_quiet_macmon(
        std::path::Path::new(lock),
        std::path::Path::new(out),
        std::time::Duration::from_secs_f64(deadline_s),
        std::time::Duration::from_secs_f64(retry_s),
        lock_wait_s,
        &cfg,
    ) {
        Ok(body) => {
            println!("{}", serde_json::to_string_pretty(&body).unwrap());
            if body["attributable"].as_bool().unwrap_or(false) { 0 } else { 1 }
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

fn check_bytes(args: &[String]) -> i32 {
    if args.len() != 2 {
        eprintln!("check-bytes PATH ARTIFACT");
        return 2;
    }
    match fs::read(&args[0]).and_then(|b| {
        verify_bytes(&args[1], &b)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }) {
        Ok(source) => {
            println!("ok {} {source}", args[1]);
            0
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}
