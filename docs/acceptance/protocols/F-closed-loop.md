<!-- ferric-accept-protocol
id: F.closed-loop.v1
track: F
-->

# Track F — closed loop

## Mission

One task, revision-pinned, with reference agreement and synchronized energy, across three interfaces:

- ferrotherm, for metrology
- ferromorphic, for event-NIR
- ferromotion, for dynamics and control

The task id is `f.closed-loop.ferromotion-sin3x.v1`. The in-repo piece of that task is the MLP in `crates/ferric-tensor/examples/ferromotion_mlp.rs`. Passing the example is necessary and not sufficient. The three interface revisions have to be pinned, the second implementation has to agree, and the joules have to cover the same window. A synthetic signal is not an event-NIR camera.

Tolerance id: `F.closed-loop.v1`.

## Independent reference

The reference implementation is a host-side f32 MLP. It does not call `ferric_tensor::Var` or `Adam`. It uses the same init, the same data, and the same Adam update the GPU example uses.

Init, from `ferromotion_mlp.rs`:

- `seq(n, s)[i] = sin(i * 0.7 + s) * 0.5`
- `scaled(n, s, sc) = seq * sc`
- W1 = `scaled(32, 1.0, 1.414)` shape `[1, 32]`, b1 zeros `[1, 32]`
- W2 = `scaled(1024, 2.0, 0.25)` shape `[32, 32]`, b2 zeros `[1, 32]`
- W3 = `scaled(32, 3.0, 0.25)` shape `[32, 1]`, b3 zeros `[1, 1]`

Data: `n = 48`, `x_i = -1 + 2 * i / 47`, `y = sin(3x)`.

Adam, from `crates/ferric-tensor/src/optim.rs`: `lr = 0.01`, `b1 = 0.9`, `b2 = 0.999`, `eps = 1e-8`, bias correction `1 / (1 - b^t)`, update `param -= lr * mhat / (sqrt(vhat) + eps)`. 3000 steps, full batch, loss = mean of squared error, ReLU on the two hidden layers.

Probes: `k = 0..39`, `x = -1 + 2 * k / 39`, compare to `sin(3x)`.

ferromotion's own revision is the dynamics-control pin. This repository does not contain that sibling. `docs/FERRALLOY-INGEST-PLAN.md` names it (`family: ferric, ferromotion`). Until `interfaces.ferromotion_dynamics_control.revision` is a real revision, the interface is unpinned.

ferrotherm's revision is the metrology pin. Nothing in this tree implements that instrument. Track C's A/B/A rules apply, and macmon does not fill the ferrotherm revision field.

ferromorphic's revision is the event-NIR pin. `ferric-signal` can tokenize a deterministic synthetic series (`crates/ferric-signal/src/synth.rs`, kind `thermal` = 1, variant 0, length 96, 500 Hz). The module states that those series say nothing about real sensors. A receipt with `sensor: synthetic` is rejected. The synthetic series may be logged as a fixture so the Ferric side of the digest machinery is specified:

- kind 1, variant 0, 96 samples from `ferric_signal::synth::signal`
- a `TokenReceipt` whose `spec_digest` is recorded
- agreement with a second run is `ferric_signal::agree`: different spec first, then token digest

That fixture does not close the interface.

## Preregistered tolerances

- Quality, both arms, against `sin(3x)`: final MSE < 5e-3 and max abs on the 40 probes < 0.15. This is the example's assert (`let ok = last < 5e-3 && max_err < 0.15`). It is not loosened.
- Cross-implementation agreement: max over the 40 probes of `|pred_ferric - pred_ref|` ≤ 1e-3. This number is preregistered here. It is not an error this workstream measured. It is stricter than 0.15 so that two fits on opposite sides of the oracle cannot both pass and still be called the same answer. A future revision of this predicate happens in a commit that does not also carry `outcome: pass`, and the new predicate stays stricter than 0.15.
- Energy: gross and above-idle, class measured or derived, `window_covers_task` true, `ferrotherm_aba.a1_a2_agree` true. Rules are track C's. Negative above-idle fails.
- All three `interfaces.*.revision` values are non-empty and not `unpinned`.
- `reference_agreement.met` is true only when the 1e-3 bar held.

## Procedure

1. Pin the three revisions in the receipt before the run. If any pin is unknown, stop. The status is withheld. Do not run the MLP and call the track passed.
2. `cargo run -p ferric-tensor --release --example ferromotion_mlp` on the Ferric arm. Record final MSE, the 40 probe values, and the adapter name.
3. Run the independent reference with the formulas above. Record the same probes.
4. Wrap steps 2 and 3 in one meter window each, or one window that covers the Ferric arm and names the reference as quality-only. Gross and above-idle follow `crates/ferric-serve/src/energy.rs` when the server meter is the instrument, or `ferric_joule::measure` when the meter is read directly. ferrotherm A1 and A2 bracket the Ferric window.
5. Event-NIR: at the ferromorphic revision, capture the pinned event stream and record its digest beside the task. A `ferric-signal` synthetic digest may sit in the artifact log labeled fixture. It is not `interfaces.ferromorphic_event_nir.sensor`.
6. Dynamics: the ferromotion revision either re-executes this task or names this Ferric revision as the port of its MLP and attaches that project's reference output. The probe vectors still have to meet 1e-3.

`cargo test -p ferric-accept` is the checker for the receipt shape, not the task.

## Receipt

`task_id` is `f.closed-loop.ferromotion-sin3x.v1`. `tolerance.id` is `F.closed-loop.v1`. Energy fields match track C, plus `window_covers_task` and `ferrotherm_aba`. The checker rejects an unpinned revision and a synthetic sensor.

## Withhold

Withhold when ferrotherm, ferromorphic, or ferromotion has no revision, when the only signal is synthetic, when the two probe vectors were not both recorded, or when the meter could not separate above-idle. The example printing a check mark is not a substitute.

## Downgrade refusal

The 0.15 and 5e-3 gates are not loosened. The 1e-3 agreement bar is not loosened up to 0.15. A synthetic thermal series is not loosened into an event-NIR capture. An unpinned sibling is not loosened into "the interface is specified, so it passes." Energy is not loosened into a step count. The mission stays three interfaces, one task, reference agreement, and synchronized energy.
