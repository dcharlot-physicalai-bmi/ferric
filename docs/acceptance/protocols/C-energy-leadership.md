<!-- ferric-accept-protocol
id: C.energy-leadership.v1
track: C
-->

# Track C — matched-quality energy

## Mission

Leadership is a saving on prefill, on decode, and on training, each at matched quality. Every figure has a gross joule number and an above-idle joule number. The class is measured or derived, in the sense of `ferric_joule::Class`. An estimate is not a claim. A baseline that fails the quality gate is not a baseline. ferrotherm calibrates the window A/B/A before the workload is attributable.

Tolerance id: `C.matched-energy.v1`.

## Independent reference

The quality reference is track A, same git revision, same weight digest. The energy reference is a sensor:

| Source in tree | Class the code assigns | Boundary |
|---|---|---|
| `ferric_joule::Macmon` | measured at the sensor; the server integral is derived | accelerator rails, and the SoC beside them when the CPU rail is not stuck at 0 (`crates/ferric-serve/src/energy.rs`) |
| `ferric_joule::Rapl` | measured | package domain |
| NVML `power.draw`, as named in the `ferric-joule` crate docs | measured | GPU board |
| TDP times a utilisation guess | estimated | not claimable |

ferrotherm is the external metrology instrument. It is not a type in this repository. macmon does not stand in for it on a leadership pass. `scripts/prefill_joules.sh` is the in-tree prefill procedure and it already calls its integral a derived figure.

`scripts/speech_joules_browser.sh` sets `FERRIC=/Users/dcharlot/vibe-coding/ferric`. Do not run that path. Do not read or edit that directory. A browser energy run uses this clone's `web/build.sh` and `web/serve.sh`.

## Preregistered tolerances

These bars are preregistered with the protocol.

- Quality: the track A receipt for the same revision is `outcome: pass` before either arm's joules may enter a comparison. The receipt id is `quality_gate_receipt_id`.
- Gross and above-idle are both finite numbers on a pass. Above-idle below zero fails. The server already returns null when the window is at or under the idle baseline or inside twice the idle spread (`crates/ferric-serve/src/energy.rs`). Null is a withhold.
- Class `estimated` cannot pass. `Class::claimable` is false for it (`crates/ferric-joule/src/lib.rs`, "Estimates may not.").
- A sample above 1000 W voids the window. That ceiling is `CEILING_W` in `energy.rs` and `PREFILL_JOULES_CEILING_W` in `scripts/prefill_joules.sh`. It is not raised to keep a number.
- Prefill attribution follows `scripts/prefill_joules.sh`: each chunk against the idle gaps on either side; gaps that disagree by more than 25% of the work are refused; each chunk's joules per token lies within 2× of the median chunk.
- ferrotherm A/B/A: calibration A1 with no workload, workload B, calibration A2. A1 and A2 must agree within the instrument's own repeatability, recorded on the receipt as `ferrotherm_aba.a1_a2_agree`. Disagreement withholds B. The two A's are not averaged into a pass.
- A saving is two readings. `ferric_joule` has no `Saving::new` for a hand-built delta. Both arms ran, both passed quality, both boundaries match.
- Three workloads, three receipts: prefill, decode, training. One of them does not cover the others.

The field note in `ferric-joule` about a 3.4% model-FLOPs baseline is historical context for why a broken baseline is not a win. This protocol does not adopt 40% as a new numeric gate. The gate is: the baseline passed the same quality predicate.

## Procedure

1. Clean tree, record the revision and the weight-file sha256.
2. Quality: execute track A for that checkpoint. Keep its receipt id.
3. Prefill: `scripts/prefill_joules.sh <model.gguf> <prompt_tokens>`. The script shells out to `macmon` and `cargo build -p ferric-llama --release --example prefill_joules`. Keep the raw sample log as an artifact.
4. Decode: the same A/B/A structure around a decode example that loads and warms outside the window. `scripts/bonsai2_decode_joules.sh` is the in-tree decode-shaped script when that checkpoint is the one under test; otherwise the same idle-gap rules as prefill, named in the command array.
5. Training: `cargo run -p ferric-tensor --release --example train` (or the example whose assert is the quality gate), meter wrapped around the steps, warmup outside the window. The example's existing assert is copied into the quality receipt from the source at this revision. It is not edited in the commit that reports the joules.
6. ferrotherm, when the instrument's revision is pinned: record A1, B, A2 on its clock and on the Ferric meter's clock. If the revision is not pinned, `status` is `withheld`. Do not copy macmon into the ferrotherm fields.
7. Baseline arm: the named competent system, same prompts, same quality predicate, its own meter boundary written next to Ferric's. Different boundaries do not compare (`Saving::claimable` refuses that in `ferric-joule`).

## Receipt

Schema: `docs/acceptance/schemas/receipt.schema.json` and `docs/acceptance/schemas/fabric-energy.schema.json`. Checker: `cargo test -p ferric-accept` and `cargo run -p ferric-accept --example check`.

`energy.class`, `energy.gross_joules`, `energy.above_idle_joules`, `energy.boundary_gross`, `energy.boundary_above_idle`, `quality_gate_receipt_id`, and `fabric_energy`. A leadership close (`closes_track_c` true) also has `energy.ferrotherm_aba.a1_a2_agree` true, all three workloads, and a Pareto latency target. The checker rejects estimated, a model estimate, a negative above-idle, a missing quality id, a hardware counter labeled as a wall outlet, a copied leaderboard joule, and a Zeus production dependency.

## Per-fabric receipt

One receipt is one cell from `docs/acceptance/corpus/fabric-cells.json`, one workload (`prefill`, `decode`, or `training`), and one instrument. `closes_track_c` is false until prefill, decode, and training all exist and the latency-constrained minimum is named. Zeus is not linked into Ferric. The published package on PyPI, `zeus-ml` 0.11.0.post1 (uploaded 2025-02-03), is older than the documentation retrieved on 2026-11-07. The methodology pin is GitHub release `zeus-v0.16.0`, published 2026-07-07: https://github.com/ml-energy/zeus/releases/tag/zeus-v0.16.0

Sources, read for this protocol rather than mirrored:

| Source | What it fixes in the receipt |
|---|---|
| Chung, Ma, Wu, Liu, Kweon, Xia, Wu, and Chowdhury, *The ML.ENERGY Benchmark*, NeurIPS 2025 Datasets and Benchmarks. https://proceedings.neurips.cc/paper_files/paper/2025/file/9dc510e3d7b0b3b2a58ffed7a3ad6b0f-Paper-Datasets_and_Benchmarks_Track.pdf | Steady state is the saturated maximum batch, excluding ramp-up and ramp-down. LLM energy per request is steady-window joules over steady-window output tokens, times tokens. Diffusion energy per request is batch energy over batch size. The recommendation is the time–energy Pareto point that meets a latency target. MLPerf Power uses a physical analyzer; this benchmark uses programmatic counters. |
| Leaderboard v3, December 2025, https://ml.energy/blog/measurement/energy/diagnosing-inference-energy-consumption-with-the-mlenergy-leaderboard-v30/ and https://ml.energy/leaderboard/ | 46 models, 7 tasks, 1,858 configurations on NVIDIA H100 and B200. Those rows are their measurements. `copied_leaderboard_joules_as_own` stays false. |
| Dataset card https://huggingface.co/datasets/ml-energy/benchmark-v3 , Apache-2.0, gated. Fetched 2026-11-07. The card lists `is_stable` and does not publish the numeric stability threshold. | A pass sets `is_stable` true and writes `stability_rule` in its own words. Raw rows were not downloaded. A GPU-only column is not whole-system joules. |
| *Measuring GPU Energy: Best Practices*, https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/ | Do not estimate from TDP. Volta or newer: `nvmlDeviceGetTotalEnergyConsumption` as a counter delta. Older NVIDIA: poll `nvmlDeviceGetPowerUsage` and integrate. Synchronize the device before the second read. |
| Zeus measure page, retrieved 2026-11-07, https://ml.energy/zeus/measure/ | NVIDIA via NVML. AMD via AMDSMI only on ROCm 6.3 or later; earlier versions returned wrong values. CPU and DRAM via RAPL (root on Linux). Apple Silicon via IOReport "Energy Model", documented there as model-based estimates at 1 mJ, not a direct power sensor. Jetson via on-chip rails, millijoules for CPU, GPU, and total chip. `begin_window` / `end_window` synchronize by default. |
| *Thermally Stable Profiling*, https://ml.energy/blog/energy/measurement/thermally-stable-profiling-for-accurate-gpu-energy-measurement/ and https://ml.energy/zeus/reference/profile/ | For a short kernel, repeat it inside a window. Their experiment settled on 5 s of measurement and 5 s of cooldown; the page says those durations are environment-specific and must be profiled, not copied. The profile API's default is 10 trials, 2 throwaway warmup trials, and 10 warmup iterations. `repeated_window` records the durations that were used. |

### Instrument, and what it is not

| `instrument` | Examples | Gross figure | Not this |
|---|---|---|---|
| `hardware_counter` | NVML total energy (millijoules since driver load, two reads, subtract); RAPL `energy_uj`; AMDSMI energy count on ROCm ≥ 6.3 | The counter delta over the window. It includes that domain's static draw. `counter_delta_joules` equals `gross_joules`. `vendor_counter_is_not_wall` is true. | A wall outlet, a facility PUE, or another device's rail |
| `sampled_rail` | `nvmlDeviceGetPowerUsage` integrated; macmon or `powermetrics` rails; Jetson sysfs milliwatts; `scripts/prefill_joules.sh` | Trapezoid or AUC of samples. `sample_period_s` and `sample_count` ≥ 8. Class stays `derived` when the number is an integral of samples, matching `crates/ferric-serve/src/energy.rs`. | A single instantaneous watt reading promoted to joules |
| `wall_meter` | ferrotherm, or any calibrated plug meter | Domain `wall`. `external_meter.name` and `cell_was_the_load` true, because the plug cannot see which device ran. | A GPU-board counter |
| `model_estimate` | TDP times a duty cycle; Apple IOReport Energy Model | Cannot pass. Zeus says those Apple figures are believed to be utilization, frequency, and voltage estimates. | A measured or derived claim |

`above_idle_joules` is `gross_joules` minus `idle_joules` of the same domain, and `subtraction_recorded` is true. Zeus's window delta is the gross of that counter, not the above-idle number. Both stay on the receipt. A GPU-board gross is not rewritten as whole-system joules per token.

### Synchronization, warmup, repetition

The CPU read of a counter happens after the device work in the window has finished. For the CUDA cell that is a stream synchronize before `nvmlDeviceGetTotalEnergyConsumption`. For a wgpu queue the in-tree contract is submit, then poll (`docs/FABRIC.md`). `sync_method` names the call. `warmup_outside_window` is true: weight load, the first compile, and the ML.ENERGY ramp sit outside the window.

Accounting is one of:

- `steady_state_service` for a saturated server. `steady_state_excludes_ramp` and `batch_saturated`. Energy per output token is steady-window joules divided by output tokens in that window.
- `repeated_window` for a kernel or a short step. `n_trials` at least 10. `measurement_duration_s` and `cooldown_duration_s` are the values used on that machine, not the blog's 5 s unless that machine's own sweep selected 5 s.
- `whole_batch` for a batch that runs as one unit. Energy per request is window joules divided by `batch_size`.

`workload_parity` carries the checkpoint sha256, the prompt-set id, the batch or concurrency, and `token_accounting` (`output_tokens`, `prefill_tokens`, or `train_steps`). The other arm of a comparison uses the same four. A latency target is required only when `closes_track_c` is true. The point reported is the minimum energy among configurations that meet the target. One unlabeled minimum is `single_best_case_only` and does not close the track.

### What each Ferric cell can honestly meter

Nothing in this table was executed for this commit. A cell with no row that can pass stays withheld.

| Cell | Counter or meter the method allows | Blocker until a real run |
|---|---|---|
| `cuda.native` | NVML energy counter on Volta or newer, else polled power. Domain `gpu_board`. Host only if RAPL is a second receipt. | An NVIDIA GPU, the driver, and a clean revision. This machine was not that run. |
| `rocm.native` | AMDSMI energy count, and only with `rocm_ge_6_3` true and `stack_version` recorded. | No ROCm kernel is in the tree. Vulkan-on-AMD is `gpu.wgpu.vulkan.hardware`, not this cell. |
| `gpu.wgpu.vulkan.hardware` | The same NVIDIA or AMD counter, named in `counter_or_meter`, after the wgpu queue is submitted and polled. | A hardware Vulkan adapter. Lavapipe does not qualify. |
| `gpu.wgpu.metal`, `metal4.tensor-units` | Sampled macmon or powermetrics rails (`sampled_rail`), or a wall meter. IOReport is a model estimate and cannot pass. | Apple silicon, the rail sampler, and the fp16 contract on Metal4 kept separate from the WGSL cell. |
| `npu.execution-provider` | ANE joules only from a rail or a wall meter. An IOReport `ane_mj` field is the energy model. A CoreML plan names the device and is not a joule figure. | The plan receipt from track B, plus a meter. |
| `cpu.host` | RAPL package, root on Linux, optional DRAM domain as its own receipt. | RAPL present. A package joule is not a GPU joule. |
| `gpu.wgpu.dx12` | No energy API is in the Zeus v0.16.0 hardware-support list for DX12. A pass names a Windows counter explicitly, or uses a wall meter. | A DX12 machine and that named counter. |
| `gpu.wgpu.webgpu.browser`, `browser.worker`, `webnn.browser` | No in-browser counter. A host rail or wall meter, with the browser process inside the boundary, `browser.real` true. | A real browser. The page's adapter must not be lavapipe. |
| `gpu.wgpu.vulkan.lavapipe` | Host package or wall only. `claims_gpu_energy` false. | Software Vulkan has no GPU energy counter. |
| `remote.tcp` | Each node metered, or withhold. An unmetered peer is not zero. | A transport and a meter on the remote, neither of which has been run here. |
| `tenstorrent.host` | `zeus_has_no_device_binding` true. Rail or wall only. | No compute kernel is in the tree. |
| Jetson | Not a Ferric cell. If one is added, the Zeus Jetson path is on-chip rail millijoules, still not a facility wall. | Out of the current cell list. |

`scripts/speech_joules_browser.sh` still points at a developer-machine tree. Do not run it. A browser energy command uses this clone.

## Withhold

Withhold when the meter is absent (the crate returns `None`, not zero), when the idle baseline moved, when the CPU rail reads 0 and a SoC number would hide that, when quality failed, when ferrotherm is unpinned, or when only one of prefill, decode, and training was run and the claim says all three.

## Downgrade refusal

Gross is not loosened into a substitute for above-idle. Above-idle is not loosened into a substitute for gross. A counter delta is not loosened into a wall-outlet figure. A GPU-board joule is not loosened into whole-system joules per token. Derived is not relabeled measured. Estimated, TDP arithmetic, and the Apple IOReport energy model are not relabeled derived. A failed quality gate is not dropped to keep a joule ratio. The 1000 W ceiling is not raised. An ML.ENERGY leaderboard number is not loosened into a Ferric result. The blog's 5 s window is not loosened into a universal duration. The developer-machine path in `scripts/speech_joules_browser.sh` is not used, and its absence is not filled with a made-up browser joule number.
