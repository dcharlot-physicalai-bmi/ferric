# Quiet energy driver

`ferric-fleet energy-drive` holds an exclusive non-blocking `flock` on `/tmp/bmi-energy.lock` from preflight through the post-trial idle bracket. bmi-1's ferrotherm harness uses the same path. If the lock is taken, the driver writes a refusal receipt and retries later. It does not block while counters are running, does not use sudo, and does not signal any process except the workload it spawned.

A refusal receipt keeps every phase collected up to the abort and sets `joules` to null. `rapl_package_j`, `rapl_core_j`, and `amdgpu_ppt_j` stay null with it. Those three meters are graded separately and are never added. CPU Tctl and GPU edge temperature are the maximum of the samples in each phase (`phases.<phase>.cpu_tctl_max_c` and `gpu_edge_max_c`), read from the hwmon channel whose label is `Tctl` or `edge`, not from a fixed `hwmonN` index and not from a single reading taken at the refusal. PPT is the amdgpu channel labeled `PPT`.

The idle bracket is 60 s. After it, and before the workload is spawned, every settle load1 sample must be ≤ `--load1-max` and every settle busy-fraction sample on cores outside `--pin-cores` must be ≤ `--busy-fraction-max`. A sample over either limit refuses the attempt, so a busy host does not also receive the trial load. The same busy-fraction cap is checked again while the workload runs. The before and after package-power baselines must agree within 3σ of their pooled sample deviation. `drift` is `{"status":"computed",...}` when both brackets were collected, and `{"status":"not_reached","phase":"<pre_idle|workload|post_idle>"}` otherwise. The RAPL modulus on framework-128gb-0000 is 65532610987 µJ.

`--pin-cores` must include each pinned thread's SMT sibling. On framework-128gb-0000 the siblings of 0–3 are 16–19; a pin of `0,1,2,3` leaves those siblings in the non-pinned set, so the workload's own sibling threads count toward the busy fraction. The driver reads `/sys/devices/system/cpu/cpuN/topology/thread_siblings_list` and refuses before the settle when any sibling is missing. The cpu list accepts commas and ranges.

On that machine, as the user who can read the counters:

```bash
nohup /path/to/ferric-fleet energy-drive \
  --lock /tmp/bmi-energy.lock \
  --out-dir "$HOME/energy" \
  --deadline-s 21600 \
  --retry-wait-s 120 \
  --settle-s 60 \
  --load1-max 2 \
  --busy-fraction-max 0.10 \
  --pin-cores 0-3,16-19 \
  --rapl-package /sys/class/powercap/intel-rapl:0/energy_uj \
  --rapl-core /sys/class/powercap/intel-rapl:0:0/energy_uj \
  -- \
  /path/to/ferric-smolvla gpu-energy \
    --weights "$HOME/art/model.safetensors" \
    --tokenizer /path/to/ferric/crates/ferric-smolvla/tests/fixtures/tokenizer.json \
    --stats "$HOME/art/policy_preprocessor_step_5_normalizer_processor.safetensors" \
    --out-dir "$HOME/energy/gpu-energy" \
    --trials 10 --warmup 2 --baseline-s 30 --sample-ms 10 \
  > "$HOME/energy/driver.log" 2>&1 &
```

`VK_DRIVER_FILES` must not force lavapipe. wgpu's default enumeration on that host selects `Radeon 8060S Graphics (RADV STRIX_HALO)`. `gpu-energy` refuses a software adapter. The same command under a user systemd timer uses `Type=oneshot` and the same `ExecStart=` line, with `User=` set to the counter's owner. No unit should be installed as root for this driver.

`gpu-energy` by itself is not an attributable receipt. The driver is what applies the quiet gates. `docs/fleet/smolvla-wgpu-radv-gfx1151-energy.json` is an earlier measured run with `attributable: false` because the host was loaded. It is not relabeled.

## macOS, macmon

`--meter macmon` samples `macmon pipe -i <sample-ms> -s 0`. The fields are `cpu_power`, `gpu_power`, `ane_power`, and `sys_power` from current macmon `Metrics`, in watts, integrated with the same trapezoid as PPT. `sys_power` is at least `chip_power`, and `chip_power` is at least CPU + GPU + ANE + DRAM, so those meters overlap and are not added. Temperatures are `temp.cpu_temp_avg` and `temp.gpu_temp_avg`.

There is no per-core pin. The receipt says `pinning: unavailable` instead of failing. Per-core busy fraction via `host_processor_info` is not collected; the quiet gate, declared before the run, is load1 (`sysctl -n vm.loadavg` on macOS, `/proc/loadavg` when that file exists) and the idle `sys_power` baseline. The same `/tmp/bmi-energy.lock` is held from preflight through the post bracket. No sudo and no `powermetrics`.

On the M5 Max, as the user who can run `macmon`:

```bash
nohup /path/to/ferric-fleet energy-drive \
  --meter macmon \
  --macmon-bin macmon \
  --lock /tmp/bmi-energy.lock \
  --out-dir "$HOME/energy" \
  --deadline-s 21600 \
  --retry-wait-s 120 \
  --settle-s 60 \
  --load1-max 2 \
  --sample-ms 1000 \
  -- \
  /path/to/ferric-smolvla gpu-energy \
    --meter macmon \
    --macmon-bin macmon \
    --sample-ms 1000 \
    --weights "$HOME/art/model.safetensors" \
    --tokenizer /path/to/ferric/crates/ferric-smolvla/tests/fixtures/tokenizer.json \
    --stats "$HOME/art/policy_preprocessor_step_5_normalizer_processor.safetensors" \
    --out-dir "$HOME/energy/gpu-energy" \
    --trials 10 --warmup 2 --baseline-s 30 \
  > "$HOME/energy/driver.log" 2>&1 &
```
