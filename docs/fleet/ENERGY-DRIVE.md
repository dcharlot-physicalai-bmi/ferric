# Quiet energy driver

The gates below are the ferrotherm harness at `c82c1e96c24d6f6c292af1196cd72199b0838257` (`bench/thrml-tsp-fleet`). Citations are `bench/thrml-tsp/METHODOLOGY.md` and `bench/thrml-tsp/src/energy.rs` in that tree.

- The host lock is `flock` `LOCK_EX` on `/tmp/bmi-energy.lock`. If it is not acquired, the energy grade is refused with reason `host lock not held` (`METHODOLOGY.md` lines 69–85; `energy.rs` lines 887–896). The receipt records `host_lock.held` and `holder_pid`. `--lock-wait-s` polls the non-blocking acquire for that many seconds (`energy.rs` lines 944–1037, default there 3600). Omitting it keeps a single non-blocking try, then the existing retry. A process id attributes or refuses; it is not subtracted from joules (`METHODOLOGY.md` lines 3–4).
- Before each idle bracket the driver waits for 60 consecutive 1 Hz `load1` samples at or under 2, and gives up after 600 samples (`METHODOLOGY.md` lines 28–32; `energy.rs` lines 738–794). Every 1 Hz sample through the bracket must also be finite and at or under 2, and an empty series is a refusal (`METHODOLOGY.md` lines 34–36; `energy.rs` lines 796–811).
- Linux trial windows refuse when the mean busy fraction of cores outside the pin is above 0.05 or any 1 Hz sample is above 0.15 (`METHODOLOGY.md` lines 49–54; `energy.rs` lines 822–858). `load1` is not a trial-window criterion (`METHODOLOGY.md` lines 40–43).
- macOS trial windows use the non-harness CPU fraction, mean at most 0.15 and no sample above 0.50, and the idle brackets must agree within 3σ (`METHODOLOGY.md` lines 57–62; `energy.rs` lines 827–870 and 1092–1133). That fraction is not subtracted from joules.
- Grades are per phase and per meter and are not pooled (`METHODOLOGY.md` lines 6–8 and 87–105). `macmon` `sys_power` is SoC telemetry, not wall power (`energy.rs` lines 429–430 and `meter/src/lib.rs` lines 13–15). `c82c1e96` does not name macmon 0.11.0 or PR 125. That release, checked separately, drops `ram_power` above system power; this driver does not integrate `ram_power`.
- RAPL uses the counter modulus when the register wraps (`energy.rs` `counter_delta_j`, lines 1210–1216). macmon watts use a trapezoid (`meter/src/lib.rs` `trapezoid`, lines 295–353). The uncertainty is the root-sum-square of the finite Student-t half-width, quantization, cadence/edge, and bracket terms (`energy.rs` lines 490–539). A missing term is omitted, not entered as zero.
- Ferrotherm's ABBA order alternates two sampler arms (`energy.rs` `pair_order`, lines 156–165). This driver has one workload, so the order is an idle bracket, the trial, and an idle bracket.
- The harness does not call `sudo` or `powermetrics` (`bench/thrml-tsp/src/main.rs` lines 2788–2804).

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
