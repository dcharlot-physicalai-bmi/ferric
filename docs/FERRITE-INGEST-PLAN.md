# Ferrite — the pure-Rust edge-deploy ingest plan

*Synthesized 2026-07-22 from four parallel research sweeps: (1) robotics-native
frameworks (Copper/Viam/LeRobot/Zenoh/dora-rs), (2) Rust building blocks + our own
assets (hwbridge/skillpack/ferric), (3) the 20-platform Wendy-class field survey,
(4) WASM-edge + MCU deploy. Working name **Ferrite** — final name pending approval.*

---

## 1. Thesis

Every platform in the field ships **bytes**. None ships **verified behavior**.
And every platform in the field gates its fleet plane behind a paid cloud.

Ferrite is IPAI @ BMI's play to build the **dominant open-source edge ecosystem
for physical AI** — not a niche beside the incumbents, the whole thing:

1. **Full parity, ungated.** Every table stake (§4) and every killer capability
   the 20-platform survey surfaced (§5a) — implemented, open source, MIT/Apache,
   *including the fleet server*. The field's universal business split (open
   device agent, paid cloud) is its shared weakness: openBalena is gutted,
   Mender gates deltas, Torizon/Foundries/Peridio/Memfault are SaaS. A fully
   open, full-featured stack undercuts every player's model simultaneously.
2. **The moat on top.** The capabilities none of them can follow (§5b), anchored
   by **Ferric's proven cross-fabric bit-identical inference** (Metal ↔ WebGPU ↔
   Vulkan): sign not just an artifact but its *expected eval outputs*, and have
   the device verify behavior after apply. Incumbent inference engines are not
   deterministic across fabrics — they cannot build this.

Pure Rust end-to-end, one typed artifact format from the browser where a policy
was trained, to the Jetson/Pi where it runs, to the MCU that drives the
actuators. It occupies the layer between "trained" and "running" that Copper
(determinism stops at the NN), Viam (models ship like tarballs), Wendy (DX, no
policy concept), and Peridio/Avocado (compliance, no eval gates) all leave open.

## 2. What the four sweeps established

**Robotics-native.** Copper defines the determinism bar (bit-identical replay,
replay-diff CI) but has no fleet/artifact layer. Viam has the best deploy model
(registry + agent reconciliation) but is Go/AGPL and treats models as blobs.
LeRobot deploys pickles over unauthenticated gRPC (CVE-2026-25874) — the incumbent
ML-robotics deploy story is actively unsafe. Zenoh is the settled transport choice.
White space: *the versioned, signed, deterministic, fleet-updatable policy layer.*

**Rust building blocks.** Nothing in the MVP requires inventing anything:
Rugix (pure-Rust A/B OTA, 1.0 Feb 2026, MIT/Apache, powers umbrelOS) ·
`mdns-sd` (discovery) · `nusb` (USB) · `usb-gadget` (CDC-NCM = Wendy's
USB-C-as-network trick) · `wasmtime`/`wasmi` (runtime) · `cargo-zigbuild`
(cross-compile) · `ed25519-dalek` (signing) · `tough`/rust-tuf (TUF later) ·
`landlock`/`seccompiler` (native sandboxing) · rustls (mTLS). Our own assets:
**hwbridge** = 12 codecs × 17 actuator targets as pure, test-vectored byte
functions (mechanical port); **skillpack** = 80% of the app-manifest story
(schema'd manifests, sha256 digests, capability negotiation, runtime-enforced
safety envelope — lacks only author signatures + compiled-payload kinds);
**ferric-serve** = the flagship payload (one static binary → OpenAI endpoint).

**Wendy-class field survey (20 platforms).** The parity bar is 11 table stakes
(§4). Eight killer capabilities worth ingesting, led by Wendy's sub-5s
host-compile→USB-C layer-stream loop, balena's engine-level deltas, Uptane/TUF
root-of-trust (Torizon/Foundries), Memfault's observability-gated rollouts,
Peridio's cohort/channel release algebra, Viam's typed registry. Eight
white-space items **none** of the twenty do (§5b). Competitive note: **Peridio's
Avocado OS 1.0 launched July 8, 2026 as "the production operating system for
Physical AI"** with an NVIDIA Jetson partnership — they claim the compliance
story, Wendy claims the DX story. Both are partial and both gate their fleet
planes. We ingest both stories, ship them open, and add the axis neither can
follow.

**WASM edge.** Viable today: Jetson/Pi-class AOT at 1.3–1.7× native; WASI 0.3
(June 2026) gives typed async capabilities; WasmEdge proves LLM inference through
wasi-nn on Orin. MCU-class viable for control logic in AOT mode (~1.34× native on
Cortex-M4, XIP from flash) — interpreters are disqualifying for hot loops (8–48×).
**WASI typed capabilities beat container entitlements**: an app imports
`robot:motor-bus` and *cannot* call anything else; same artifact runs browser →
Jetson → bare-metal RP2350. Inference belongs in the **host** (wasi-nn pattern):
in-sandbox MCU inference lands 4–15× off SOTA vs vendor SIMD kernels. Open slots:
no pure-Rust MCU AOT path; no pure-Rust host inference engine for the edge.
Wendy Lite's runtime is undocumented (Preview); AkiraOS is C with no ML story;
Wasefire has no capability model yet.

## 3. Architecture

New sibling workspace `~/vibe-coding/ferrite/` (family: ferric, ferromotion).

| Crate | Contents |
|---|---|
| `ferrite-pack` | The artifact format. Skillpack manifest evolved: payload kinds (`wasm-component` \| `native` \| `model` \| `config`), `requires{}` capability grants, safety envelope, sha256 digests, **ed25519 author signatures**, and **signed eval vectors** (input → expected output hashes) for verified-behavior checks. |
| `ferrite-bridge` | hwbridge ported: 12 codecs × 17 targets as pure `fn(&Cmd) -> Vec<u8>` with the JS test vectors as Rust tests. The typed floor of the capability worlds. |
| `ferrite-agent` (bin `ferrited`) | Device daemon: `mdns-sd` advertise · rustls mTLS API (info/deploy/start/stop/logs) · wasmtime runtime — WASI grants derived from manifest `requires{}` · landlock+seccomp for native payloads · staged-dir + atomic-rename app updates · **post-apply self-check** (run signed eval vectors, byte-compare) · telemetry incl. joules-per-task where counters exist (RAPL/tegrastats). |
| `ferrite` (bin) | CLI: `discover` · `ls` · `build` (cargo-zigbuild cross) · `sign` · `deploy` (stream to device, changed-chunks only) · `run` (host-compile → deploy → attach logs, the Wendy inner loop) · `verify` · `promote`. |
| `ferrite-gate` | Promotion gates: eval harness, sim-gate hooks (run the pack against the MuJoCo twin before hardware cohort). v0.3. |
| `ferrite-lite` | MCU runtime (wasmi or wasmtime+Pulley) with the same pack format. v2 — after Linux-class is real. |
| `ferrite-fleet` | Channels/cohorts/TUF fleet plane. v2. |

**Inference boundary:** wasi-nn-shaped WIT world (`ipai:nn`) backed by Ferric as
the host engine. Policies are components that *orchestrate*; kernels run native.
**OS updates:** delegate to Rugix Ctrl — do not rebuild A/B. **Transport:** plain
mTLS/HTTP2 in v1; Zenoh when fleet-scale demands it.

## 4. Parity matrix (table stakes → building block)

| # | Table stake | How we cover it |
|---|---|---|
| 1 | Atomic A/B OS update + rollback | Rugix Ctrl (delegate) |
| 2 | Signed artifacts, device-verified | ed25519-dalek in-pack (v1) → TUF via `tough` (v1.0) |
| 3 | App layer decoupled from OS | wasmtime components; staged-dir + atomic rename |
| 4 | Delta transport | Rugix content-defined chunking (OS); changed-chunk streaming (apps) |
| 5 | Per-device crypto identity | rustls mTLS, per-device keypair at first boot |
| 6 | Cohorts/staged rollout/pin | `ferrite-fleet` (v2); manual pin in v1 |
| 7 | Remote access + logs/vitals | `ferrited` API: logs/attach in v1 |
| 8 | Sub-30s local dev loop | `ferrite run`: host cross-compile → mDNS/USB stream (target: sub-5s redeploy) |
| 9 | Pi + Jetson named targets | cargo-zigbuild aarch64-musl static binaries |
| 10 | Open device side | MIT/Apache-2.0, no feature gating — **and open fleet side** (`ferrite-fleet` self-hostable, deltas + cohorts + dashboard included; the whole field gates at least one of these) |
| 11 | SBOM + reproducible builds | hermetic vendored builds (ferric tradition) + cargo-auditable |

## 5. Beyond parity

### 5a. Killer capabilities ingested from the field (best-in-class, reimplemented open)

- Wendy's **host-compile → mDNS/USB-C changed-chunk streaming** inner loop (sub-5s)
- balena's **engine-level delta transport** (Rugix content-defined chunking both halves)
- Torizon/Foundries' **TUF/Uptane root-of-trust** (`tough` — the first credible pure-Rust Uptane)
- Memfault's **observability-gated rollouts** (success = metric non-regression, on ferric-flow)
- Peridio's **cohort/bundle/channel release algebra** (OS-agnostic, in `ferrite-fleet`)
- Viam's **typed registry of drivers + models** (ferrite-bridge worlds + signed packs)
- Wendy/EVE-OS **entitlement-based hardware permissioning** — done *better* as WASI typed
  imports (deny-by-default at the interface, not the daemon)
- Wendy Lite/Ocre's **one artifact format down to the MCU** (ferrite-lite, v2)

### 5b. The moat (what none of them can build)

1. **Verified-behavior updates** — sign artifact *and* expected eval outputs;
   device self-checks bit-identical after apply. Requires deterministic
   cross-fabric inference → only Ferric has it.
2. **Policy as a first-class channel with eval gates** — skillpack safety
   envelope + shadow/sim-gated promotion. No incumbent treats a robot policy
   differently from a tarball.
3. **Joules-per-task as rollout telemetry** — the EFA perf-per-watt metric as a
   fleet regression signal. Unclaimed by all 20 platforms.
4. **Browser as the dev/ops surface** — WebUSB/WebSerial flash + console,
   simulate-before-deploy in-page, and **browser-trained → deployed continuity**:
   a policy trained at /research/vla ships as a signed pack to a device, same bits.
5. **One typed artifact browser → Jetson → MCU** — WASM component + WIT hardware
   worlds (`ipai:nn`, motor-bus via ferrite-bridge). Wendy needs two substrates;
   we need one.
6. **Pure-Rust end-to-end** — a single static musl agent (<10 MB target), no
   runtime deps. Incumbent agents: Go, C++, Java, Elixir. Only Rugix is Rust and
   it is only the update engine.

## 6. Phasing

**v0.1 — the verified loop (MVP). ✅ SHIPPED 2026-07-22** (same day as this
plan). Workspace `~/vibe-coding/ferrite/`: ferrite-pack + ferrite-runtime +
ferrite CLI + ferrited. 11 tests green (incl. bit-identical pack rebuilds,
behavioral-drift rejection, fuel-bounded runaways). **Live cross-arch demo
completed**: pd-hover (a real PD hover-controller policy, wasm32-wasip1) built
+ signed + vectored on the aarch64 Mac, deployed over Tailscale to the x86_64
RTX 4050 box — device independently re-ran the 3 signed vectors from staged
bytes: **all output digests bit-exact across architectures**; tampered pack
(1 byte) rejected 400 with the differing digest named. Honest finding: wasm
*fuel counts* differ across platforms (repeatable within one) — likely host-I/O
readiness paths — which is why the acceptance contract compares output digests
only; fuel stays telemetry. The Ferric-on-Metal↔Vulkan model-payload variant
(true cross-*fabric* GPU check) is the v0.2 headline.
Cut from v0.1 as planned: OCI, TUF, deltas, BLE, MCU, fleet server, USB gadget.

**v0.2 — the fabric gate + hardware floor. ✅ CORE SHIPPED 2026-07-22 (same
day).** (a) **Engine "ferric"**: model packs whose eval runs on the device's
GPU fabric via ferric-core; ops `matmul-chain` (integer-xorshift weights, no
libm — the proven-deterministic kernel path) and `demo-lm` (full 3-layer
transformer probe). **LIVE cross-FABRIC result**: vectors recorded on Apple
Metal (M5 Max) → `matmul-chain` verified **bit-exact on NVIDIA Vulkan (RTX
4050)** — the moat receipt no incumbent can produce. And the probe proved the
gate's teeth: `demo-lm` (GPU sin/exp = implementation-defined) genuinely
diverges Metal↔Vulkan and was **rejected 400 with the differing digests** —
divergence every other platform would silently ship. The deploy gate now
enforces Ferric's documented determinism boundary at update time.
(b) **ferrite-bridge**: all 12 hwbridge codecs × 17 targets ported (zero-dep,
35 golden byte-vector tests) + `BridgeSpec` manifest stage — payload stdout
(JSON actuator targets) → wire bytes via the named codec, and those wire bytes
are what vectors digest and authors sign: *signed wire-level behavior*.
Still open for v0.2: landlock/seccomp native payloads, Rugix OS A/B, USB-C
CDC-NCM dev link, sub-5s `ferrite run`.

**v0.2 EXTENSION — WHOLE-MODEL DETERMINISM (2026-07-22, same day): the
verifiable envelope now covers FULL TRANSFORMERS.** The demo-lm rejection was
root-caused via a per-kernel fabric probe (ferric-core
examples/fabric_probe.rs) + CPU-IEEE forensics, and fixed at five layers:
(1) det-math WGSL (deterministic exp/sin/cos/ln/rsqrt/recip — Cody-Waite +
Horner + Newton from correctly-rounded ops only; kernels.rs DET_MATH_WGSL,
auto-prepended by Context::pipeline); (2) ferric forks/wgpu-hal:
MTLMathModeSafe (Metal defaulted to fast-math — reassociation); (3) ferric
forks/naga SPIR-V: NoContraction on all float ops (NVIDIA fuses fma when
allowed); (4) ferric forks/naga MSL: `#pragma STDC FP_CONTRACT OFF` (Metal
contracts even in Safe mode); (5) demo weights via integer xorshift (host
sinf differs macOS↔glibc). Result: **all 7 probe rows bit-identical Metal
(M5 Max) ↔ Vulkan (RTX 4050), sqrt verified 0/768 vs plain-IEEE CPU on both —
WGSL now evaluates as-written IEEE on every fabric (same semantics as CPU and
wasm)**. ferric-core validations all green (softmax 3.7e-9, attention 2.2e-8,
matmul exact-0 vs CPU). E2E: demo-lm 0.2.1 (full 3-layer transformer) recorded
on Metal → **ACCEPTED bit-exact on the Vulkan device** through ferrited.
Plumbing lesson: patch tables apply only at the build's workspace root —
ferrite/Cargo.toml now mirrors ferric's fork patches (without them packs
verify against STOCK fast-math wgpu). Upstream-worthy: the wgpu-hal/naga
determinism patches are a legitimate wgpu contribution.

**v0.3 — the gates.** `ferrite-gate`: sim-gated promotion (pack must pass the
MuJoCo twin regression before hardware), joules-per-task telemetry
(RAPL/tegrastats), browser ops page on the site (WebSerial console, WebUSB flash).

**v1.0 — trust + fleet.** TUF root via `tough`, channels/cohorts
(Peridio-style release algebra), delta streaming, `ferrite-lite` MCU preview.

## 7. Honest caveats

- **MCU AOT gap is real**: nothing pure-Rust ships an AOT path for
  Xtensa/riscv32. ferrite-lite v2 starts interpreter-class (wasmi/Pulley) —
  supervisory logic only, not 1 kHz loops. The pure-Rust MCU AOT compiler is a
  genuine open research slot, not a v1 promise.
- **GPU access from WASM on Jetson** is host-plugin-mediated (wasi-nn), not
  standardized beyond it — fine for our architecture (Ferric is the host), but
  in-sandbox GPU compute is not a thing we can promise.
- **Joules-per-task needs counters**: RAPL (x86) and tegrastats (Jetson) exist;
  Pi-class needs an external meter — telemetry is best-effort per platform.
- **Name collisions on crates.io** (`ferrite` variants exist) — irrelevant until
  we publish; pick the public crate prefix then.
- **Peridio/Avocado and Wendy will keep moving** — parity items must track the
  field continuously (re-survey quarterly). The moat axis (determinism / eval
  gates / watts / browser) is where they structurally cannot follow; parity plus
  open-everything is what makes the ecosystem the default choice.

## 8. Naming

Family: Ferric (AI), Ferromotion (kinematics/dynamics), ferric-flow (dataflow).
Proposed: **Ferrite** — the magnetic-core material; small, embedded, carries the
field. CLI `ferrite`, daemon `ferrited`. Alternatives considered: Ferrofleet
(explicit but narrow — the fleet plane is v2, the artifact layer is the soul),
Forgeline (ties to Forge but leaves the Ferric family), ferric-edge (in-workspace
crates — undersells a product this large). Final call: the Dean's.
