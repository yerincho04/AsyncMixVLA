# Deployable trigger v1 — development protocol, 2026-09-16

Target: replace privileged learned triggering with a camera/proprioception/action
history detector. This is an implementation and data-quality milestone, not a
claim that a trained model already improves task success. Historical frozen
results/checkpoints are preserved. No TEST episodes are used here.

## Evidence review

Claude's project memory and the Sept16 calibration-mismatch findings were read.
The old candidate dataset double-processes gripper commands; the older rollout
pipeline differs from deployment; progress features are shifted by20 steps.
An additional audit found whole-episode gripper medians in offline progress
extraction (future information). None of those derived features trains v1.

The suspected task0/trial12 onset discrepancy was a phase-reading error: actual
hook displacement occurs at real steps93–97 (old debug indices113–117), before
the old logger's pre-action snapshot. Across the old373 perturbed traces,372
have hook-side displacement at the expected onset; one never applies and two
apply four increments rather than five. Application is therefore recorded,
not inferred from an episode's 'perturbed' condition.

## Input boundary

The live deployable scorer accepts exactly prepared RGB front/wrist images,
the measured8D robot state, a processed planned7D action, and step/chunk indices.
No env object, raw observation dictionary, task ID, object pose, body identity,
manifest, perturbation direction/onset, outcome or future observation is passed.
Step is an alignment check, not a predictor. The task instruction is available
to the action policy; this initial trigger baseline does not encode it.

Features are235 deterministic float32 robot/action-history and temporal image
change statistics, computed by one shared online/offline module. Images are
orientation-correct64x64 RGB; feature extraction pools them to16x16. Feature
extraction cost is measured per episode. Richer visual representations remain
an experiment, not an assumed requirement or an established win.

## Recording phases

Warmup and idle steps finish before recorder.start. Row t contains O_t and the
raw/planned action before the disturbance hook. The raw gripper is processed
exactly once; that exact action is logged and passed to env.step. The hook's
before/after body positions and diagnostic flags are label-only event records.
Action t yields O_(t+1); a hook edit at t first becomes observable at t+1.
The recorder asserts contiguous steps, exact action identity, state continuity,
terminal consistency and an exact520-action limit. Runtime feature replay uses
the recorded inputs, not environment replay or a cross-run positional join.

Pure Adapter recording calls the canonical run_asyncmixvla path. All legacy
behavior is unchanged without the opt-in recorder/deployable trigger. The
canonical runtime still has historical handoff horizon/counter/perturbation
continuation limitations; these MUST be resolved in a matched evaluation path
before a new final switching table is produced. A new trigger alone does not
repair those benchmark issues.

## Splits and completeness

episodes.json enumerates756 feasible episodes directly from manifests: tasks0–9,
trials12–49, clean plus feasible translation perturbations. The old753-record
cache omitted task2 trials19/35/41 perturbed; these are included explicitly.
Fit12–31; episode-level calibration32–39; DEV40–49 reporting only. No old cached
Adapter success labels are used: every label is this recording's final outcome.

Pilot: tasks0/2/6 trial12, clean and perturbed (six recordings), each with an
uninstrumented canonical Adapter repeat. All action bytes and outcomes must
match. Physical event labels, exactly-once gripper processing, and feature
replay must pass. Missing or failed records stop processing; no automatic
deletion or silent retry. Only expand after the complete pilot passes.

## Initial model and calibration

fit.py defines robot-only and robot+camera L2 logistic baselines (C=1), with at
most64 samples per episode and equal total episode weights before class balance.
Labels are weak Adapter failure risk, NOT measured recovery benefit. Failed
perturbed episodes are positive only after the actual first observable push;
earlier rows are excluded. Clean failures and failed episodes with no observed
push use whole-episode weak labels because their failure onset is unknown.

Threshold and persistence1/3/5 use only calibration trials, controlling clean-
success false triggers per EPISODE to at most10% on that calibration sample.
The highest failure recall then lowest switching burden selects persistence.
The same calibration-only rule chooses between robot-only and camera models,
with robot-only preferred on an exact tie. DEV reports eventual recall and
first-fire timing without choosing parameters or models.
Partial data cannot produce an approved deployment artifact. Even a complete
fit remains deployment_approved=false pending live switching validation.

## Next gates

1. Pass all six pilot recordings and exact matched controls.
2. Collect and validate the entire TRAIN/DEV manifest, then fit both baselines.
3. Verify online scores/first-fire steps against offline input replay.
4. Measure live matched switching success, rescue/retain, false-trigger rate,
   compute and latency; tune only using TRAIN/calibration, report DEV honestly.
5. Freeze before a fresh final evaluation. Oracle/privileged references remain
   separately labeled; no claim of deployability rests on them.

Run CPU checks from openvla-oft:
`python -m unittest discover -s deployable_trigger_v1 -t . -p 'test_*.py'`.
Fit after full collection:
`python -m deployable_trigger_v1.fit --data_dir PATH --out_dir PATH`.
