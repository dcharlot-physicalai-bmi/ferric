# Ferralloy — the pure-Rust edge-deploy ingest plan

*Synthesized 2026-07-22 from four parallel research sweeps: (1) robotics-native
frameworks (Copper/Viam/LeRobot/Zenoh/dora-rs), (2) Rust building blocks + our own
assets (hwbridge/skillpack/ferric), (3) the 20-platform Wendy-class field survey,
(4) WASM-edge + MCU deploy. Name **Ferralloy** — chosen 2026-07-24 (bare crate name free on crates.io; was working-name Ferrite, which collides with an image viewer).*

---

## 1. Thesis

Every platform in the field ships **bytes**. None ships **verified behavior**.
And every platform in the field gates its fleet plane behind a paid cloud.

 Ferralloy is IPAI @ BMI's play to build the **dominant open-source edge ecosystem
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

New sibling workspace `~/vibe-coding/ferralloy/` (family: ferric, ferromotion).

| Crate | Contents |
|---|---|
| `ferralloy-pack` | The artifact format. Skillpack manifest evolved: payload kinds (`wasm-component` \| `native` \| `model` \| `config`), `requires{}` capability grants, safety envelope, sha256 digests, **ed25519 author signatures**, and **signed eval vectors** (input → expected output hashes) for verified-behavior checks. |
| `ferralloy-bridge` | hwbridge ported: 12 codecs × 17 targets as pure `fn(&Cmd) -> Vec<u8>` with the JS test vectors as Rust tests. The typed floor of the capability worlds. |
| `ferralloy-agent` (bin `ferralloyd`) | Device daemon: `mdns-sd` advertise · rustls mTLS API (info/deploy/start/stop/logs) · wasmtime runtime — WASI grants derived from manifest `requires{}` · landlock+seccomp for native payloads · staged-dir + atomic-rename app updates · **post-apply self-check** (run signed eval vectors, byte-compare) · telemetry incl. joules-per-task where counters exist (RAPL/tegrastats). |
| `ferralloy` (bin) | CLI: `discover` · `ls` · `build` (cargo-zigbuild cross) · `sign` · `deploy` (stream to device, changed-chunks only) · `run` (host-compile → deploy → attach logs, the Wendy inner loop) · `verify` · `promote`. |
| `ferralloy-gate` | Promotion gates: eval harness, sim-gate hooks (run the pack against the MuJoCo twin before hardware cohort). v0.3. |
| `ferralloy-lite` | MCU runtime (wasmi or wasmtime+Pulley) with the same pack format. v2 — after Linux-class is real. |
| `ferralloy-fleet` | Channels/cohorts/TUF fleet plane. v2. |

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
| 6 | Cohorts/staged rollout/pin | `ferralloy-fleet` (v2); manual pin in v1 |
| 7 | Remote access + logs/vitals | `ferralloyd` API: logs/attach in v1 |
| 8 | Sub-30s local dev loop | `ferralloy run`: host cross-compile → mDNS/USB stream (target: sub-5s redeploy) |
| 9 | Pi + Jetson named targets | cargo-zigbuild aarch64-musl static binaries |
| 10 | Open device side | MIT/Apache-2.0, no feature gating — **and open fleet side** (`ferralloy-fleet` self-hostable, deltas + cohorts + dashboard included; the whole field gates at least one of these) |
| 11 | SBOM + reproducible builds | hermetic vendored builds (ferric tradition) + cargo-auditable |

## 5. Beyond parity

### 5a. Killer capabilities ingested from the field (best-in-class, reimplemented open)

- Wendy's **host-compile → mDNS/USB-C changed-chunk streaming** inner loop (sub-5s)
- balena's **engine-level delta transport** (Rugix content-defined chunking both halves)
- Torizon/Foundries' **TUF/Uptane root-of-trust** (`tough` — the first credible pure-Rust Uptane)
- Memfault's **observability-gated rollouts** (success = metric non-regression, on ferric-flow)
- Peridio's **cohort/bundle/channel release algebra** (OS-agnostic, in `ferralloy-fleet`)
- Viam's **typed registry of drivers + models** (ferralloy-bridge worlds + signed packs)
- Wendy/EVE-OS **entitlement-based hardware permissioning** — done *better* as WASI typed
  imports (deny-by-default at the interface, not the daemon)
- Wendy Lite/Ocre's **one artifact format down to the MCU** (ferralloy-lite, v2)

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
   worlds (`ipai:nn`, motor-bus via ferralloy-bridge). Wendy needs two substrates;
   we need one.
6. **Pure-Rust end-to-end** — a single static musl agent (<10 MB target), no
   runtime deps. Incumbent agents: Go, C++, Java, Elixir. Only Rugix is Rust and
   it is only the update engine.
7. **Certificate-gated packs — verified *correctness*, not just verified
   reproduction** (the axis beyond the axis; the science is now proven). §5b.1
   verifies a pack *reproduces* bit-exactly across fabrics — it does not verify
   the pack is *correct*. The Charlot Lab certificate program
   (`bmi-concept/research/certificate-toolchain/`, 24 reproducible artifacts)
   closes that gap: a control policy can ship with a machine-checkable **formal
   behavioral certificate** — a Lyapunov energy, a certified region, and the
   prover + its parameters — proving the *closed-loop behavior* (stability,
   non-divergence, convergence to goal) holds over a **whole continuous region**,
   not at sampled eval points. The pack carries the certificate as a first-class
   facet; `ferralloy-gate` re-checks it for the *deployed* weights (re-run the SOS
   SDP / dReal SMT query, or the sound Taylor+CROWN pass) before the hardware
   cohort — a pack that no longer certifies is 400'd exactly as a drifted eval
   vector is. Grounded receipts, all reproducible: dReal-certified learned
   **ternary** energy beating any quadratic on a non-convex ROA (reversed Van der
   Pol, proven in **2D and 4D**), certified *across* actuator saturation **and**
   contact mode-switches (R=1.2), SOS-certified to **8 states** where SMT chokes
   at 4. And the EFA twist the pack format was built for: **the payload IS the
   certificate** — one ternary energy is simultaneously the controller descended
   and the Lyapunov function proven (8/16 nonzero weights, `Tⱼ·x` = select+negate,
   no matmul), already running native + wasm32 on Ferric
   (`ferric-tensor/examples/ebm_ternary_cert.rs`, cross-verified 4.7e-13). And it
   is nearly free — measured **~1.2 µJ per certified step** on the RAPL telemetry
   axis Ferralloy already ships (§5b.3), 0.0004–0.003% of a per-joint actuation
   budget. No incumbent verifies reproduction; none verifies correctness. The
   §5 law that governs it (a learned energy beats a quadratic *iff* the ROA is
   non-convex) tells the gate when a cheap quadratic certificate suffices and
   when the learned ternary one is required — a deploy-time decision, not a guess.

## 6. Phasing

**v0.1 — the verified loop (MVP). ✅ SHIPPED 2026-07-22** (same day as this
plan). Workspace `~/vibe-coding/ferralloy/`: ferralloy-pack + ferralloy-runtime +
ferralloy CLI + ferralloyd. 11 tests green (incl. bit-identical pack rebuilds,
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
(b) **ferralloy-bridge**: all 12 hwbridge codecs × 17 targets ported (zero-dep,
35 golden byte-vector tests) + `BridgeSpec` manifest stage — payload stdout
(JSON actuator targets) → wire bytes via the named codec, and those wire bytes
are what vectors digest and authors sign: *signed wire-level behavior*.
**2026-07-23 — envelope extension: FULL TRANSFORMERS.** After the ferric
det-math campaign (see ferric/docs/determinism/), the demo-lm pack that the
Vulkan device correctly REJECTED on day one now deploys **accepted,
bit-exact** — a full 3-layer transformer whose Metal-recorded vectors verify
byte-identically on NVIDIA Vulkan. The same digests hold in Chrome's WebGPU,
so browser-trained → device-deployed is a checkable signature end to end.
Also shipped since: `ferralloy run` (verified inner loop: 256 ms warm / 762 ms
code-change, Mac→device over Tailscale with full verification per deploy).
**Native-payload sandbox SHIPPED (2026-07-23):** native ELF packs run
confined — landlock (payload dir rx · per-run scratch rw · loader dirs +
specific resolver files r · TCP denied unless `net` granted) + **seccomp
deny-list** (25 escape/tamper syscalls → EPERM: ptrace, kexec, module load,
mount, bpf, setns/unshare, keyring, …) + no-new-privs + CPU/AS/FSIZE/NOFILE
rlimits + cleared env. E2E-proven on the RTX Linux box: the same binary
reads `/etc/hostname` and unshares namespaces freely unconfined but both are
denied inside the pack, with the deploy behavior verified bit-exact. Engine
`native`, target-gated so macOS/wasm still build.
**Open fleet plane SHIPPED (2026-07-23):** `ferralloy-fleet` — the
self-hostable, ungated fleet server the whole field gates behind a paid
cloud. Channels hold one signed release each (statically verified on upload —
a corrupt PUT is 400'd, channel never created); `ferralloyd` opt-in-subscribes
(FERRALLOY_FLEET_URL/_CHANNEL/_DEVICE_ID), PULLS its target, runs the SAME
on-device accept gate (behavioral verification), applies atomically, reports.
Server never pushes → NAT'd devices update; "rolled out" = behavior verified
on-device, not bytes sent. CLI `ferralloy release … --channel … --fleet …` +
`ferralloy fleet`; live dashboard at /. Agent accept gate refactored to a
shared `accept_pack()`. E2E-proven: release → poll → verify → apply → report;
version bump auto-picked-up; corrupt release rejected CLI- AND server-side.
**CANARY ROLLOUTS SHIPPED (2026-07-24), gated on VERIFIED BEHAVIOR.** A
channel now holds a `current` release plus an optional `rollout` at N% of its
devices, assigned by a stable FNV percentile of the device id. `ferralloy
release … --canary 20` stages a canary; the device poll carries `?device=<id>`
so the server resolves each device to its release; the dashboard shows the
canary's verified-pass rate ("2/2 verified") as the promotion signal;
`ferralloy promote` makes it fleet-wide, `ferralloy abort` reverts. E2E: 4
devices, 50% canary → the right 2 pulled v2 and verified bit-exact while the
other 2 held v1, then promote converged the fleet. "Rolled out" = behavior
verified on a slice of the REAL fleet, not bytes delivered — no incumbent does
this. `ferralloy-fleet` + `ferralloy-agent` published at 0.2.0.

**PUBLISHED to crates.io + fork-free determinism (2026-07-24).** `cargo
install ferralloy` works. The wgpu/naga determinism forks were proven
REDUNDANT — pure-WGSL storage-pinning is bit-identical on STOCK wgpu 30
(Metal + Vulkan, 9/9 golden), so `ferric-core` republished at 0.2 against
stock wgpu, Ferralloy dropped its fork mirror, and every crate builds on the
public registry unmodified. TR-2026-23 revised to Preprint v2 with the
finding.

Still open for v0.2: TUF-rooted keys, Rugix OS A/B, USB-C CDC-NCM dev link.
(A formal-certificate facet — device-side Lyapunov re-verification, verified
CORRECTNESS beyond verified behavior — is in active development in
`ferralloy-pack`.)

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
on Metal → **ACCEPTED bit-exact on the Vulkan device** through ferralloyd.
Plumbing lesson: patch tables apply only at the build's workspace root —
ferralloy/Cargo.toml now mirrors ferric's fork patches (without them packs
verify against STOCK fast-math wgpu). Upstream-worthy: the wgpu-hal/naga
determinism patches are a legitimate wgpu contribution.

**v0.3 — the gates.** `ferralloy-gate`: sim-gated promotion (pack must pass the
MuJoCo twin regression before hardware), joules-per-task telemetry
(RAPL/tegrastats), browser ops page on the site (WebSerial console, WebUSB flash),
and the **certificate gate** (§5b.7). **✅ SHIPPED (2026-07-24):** the certificate
gate is built and workspace-tested end to end —
- `ferralloy-pack::certificate` — a `CertificateSpec` manifest facet (`Manifest.certificate`,
  optional ⇒ pre-certificate packs' canonical bytes unchanged) + the **device-side re-verifier**
  in dependency-free f64 (2nd-order Taylor + per-box CROWN |tanh″|, adaptive box refinement,
  all 6 free/contact×saturation hybrid cases). No SDP/SMT, only `tanh` — a faithful port of
  `ebm_cert_verify.rs` (proven wasm-clean @75KB). Re-proves the R=1.2 certificate: 1486 boxes,
  depth 8, worst bound −0.00000.
- `ferralloy verify-cert <fpack>` — the explicit on-device re-proof CLI.
- **The gate, wired at every hop:** `build --certificate` re-proves at build time (rejects a bad
  cert, naming the offending box); `deploy`/`release` re-prove before shipping/publishing (the
  operator promotion hook); and the **device agent** (`accept_pack`) re-proves before a pack goes
  live — verified *correctness* alongside verified *behavior*. 5 certificate unit tests +
  workspace green; reference fixture in `ferralloy/payloads/certificate-example/`.
SOS/dReal stay a build-time/fleet-server gate (they need SDP/SMT solvers, not on-device).

**v1.0 — trust + fleet.** TUF root via `tough`, channels/cohorts
(Peridio-style release algebra), delta streaming, `ferralloy-lite` MCU preview.

## 7. Honest caveats

- **MCU AOT gap is real**: nothing pure-Rust ships an AOT path for
  Xtensa/riscv32. ferralloy-lite v2 starts interpreter-class (wasmi/Pulley) —
  supervisory logic only, not 1 kHz loops. The pure-Rust MCU AOT compiler is a
  genuine open research slot, not a v1 promise.
- **GPU access from WASM on Jetson** is host-plugin-mediated (wasi-nn), not
  standardized beyond it — fine for our architecture (Ferric is the host), but
  in-sandbox GPU compute is not a thing we can promise.
- **Joules-per-task needs counters**: RAPL (x86) and tegrastats (Jetson) exist;
  Pi-class needs an external meter — telemetry is best-effort per platform.
- **Name collisions on crates.io** (`ferralloy` variants exist) — irrelevant until
  we publish; pick the public crate prefix then.
- **Peridio/Avocado and Wendy will keep moving** — parity items must track the
  field continuously (re-survey quarterly). The moat axis (determinism / eval
  gates / watts / browser) is where they structurally cannot follow; parity plus
  open-everything is what makes the ecosystem the default choice.

## 8. Naming

Family: Ferric (AI), Ferromotion (kinematics/dynamics), ferric-flow (dataflow).
Proposed: ** Ferralloy** — the magnetic-core material; small, embedded, carries the
field. CLI `ferralloy`, daemon `ferralloyd`. Alternatives considered: Ferrofleet
(explicit but narrow — the fleet plane is v2, the artifact layer is the soul),
Forgeline (ties to Forge but leaves the Ferric family), ferric-edge (in-workspace
crates — undersells a product this large). Final call: the Dean's.
