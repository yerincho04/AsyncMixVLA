# Observable V1/V2 development, 2026-09-16

V1 is the existing frozen visual disturbance detector (fpr10). V2 predicts the
signed success benefit of invoking the frozen AsyncMixVLA handoff at V1's first
candidate. This is a new intervention-utility V2 objective, distinct from the old
Adapter-failure V2. Initial design decides immediately; no extra observation
wait and no additional candidate filter. A wait would consume rescue time and
must be a separate experiment. No TEST trials 0–11 are loaded.

V1 and V2 receive ONLY current/past prepared cameras, measured 8D robot state,
processed 7D Adapter actions, and chunk position. Step is used for alignment.
No environment, object identity/pose, task ID, manifest/onset/direction, outcome,
or future frames enter either predictor. The natural language instruction stays
with the policies; this initial V2 does not encode it. V2 features: the validated
235D observable history, V1's 512D spatial CNN features, 128D past-four-step
embedding change, and V1 score. These are shared between collection and serving.

Pair collection runs a fresh Adapter control and a fresh V1-triggered AsyncMixVLA
continuation. Before either branch decision, exact hashes must match for every
prepared camera/state input and processed action, simulator state at candidate,
queued raw Adapter chunk, V2 features, and physical perturbation bookkeeping.
A failed comparison stops the shard. Deterministic prefix replay preserves the
controller and perturbation-hook state without assuming a MuJoCo snapshot
captures all Python controller state. Simulator state is audit-only. Pilot also
repeats controls and switching arms and checks complete action hashes/outcomes.

Use the corrected EpisodeEnv from research_recovery_v1: 520 actual actions,
including bridge/first OFT action; hook applied before EVERY action including
OFT continuation; bridge capped to leave room for OFT. The remaining prefetched
OFT actions are discarded as in the current handoff (takeover_k=1). All methods
in this new experiment share these rules. Historical DEV numbers are not
interchangeable with this corrected benchmark.

Targets are signed paired success differences: +1 rescue, -1 harm, 0 tie. These
are offline training labels only. A one-seed pair estimates benefit in this
benchmark, not a universal causal success probability. Fit trials12–31,
calibration32–39, development40–49. V1 was already trained on fit trials; V2
candidate distributions are explicitly conditional on that frozen V1.

Fit two regularized logistic heads for rescue and harm, retaining all tied
pairs, with utility p(rescue)-p(harm). No class balancing (probabilities should
reflect pair prevalence). Candidate regularization 0.01/0.1/1 and thresholds
-0.1/0/0.025/0.05/0.1/0.2. Calibration selects maximum paired success subject to
keeping >=90% of V1-only rescues and no more harms; ties prefer fewer switches,
then stronger regularization. Direct V1-only is an explicit fallback. DEV is
reported after selection, never used for threshold selection. Report uncertainty
and paired outcomes, switch burden and timing; no guaranteed improvement.

Six-episode pilot: tasks0/2/6, trial12, clean and perturbed. Validate determinism,
causality and privileged-input rejection before expanding. Full collection is
756 episodes, including clean candidates. If V1 never fires, both policies are
Adapter and a repeat checks identical actions. Do not label such an episode a
V2 negative: V2 is never queried there.

A selected V2 remains deployment_approved=false until online/offline equivalence
and a matched live DEV run pass. Final paper evaluation requires a frozen shared
trigger for sync/naive/ours and a fresh held-out evaluation. Labels from ours may
favor its intervention; this must be disclosed and the shared-trigger baselines
must still be evaluated. Existing TEST outcomes have been seen during historical
method development; trials0–11 are untouched by this trigger's training, not an
entirely unseen project-wide test set.

Pilot 2262244 was cancelled before label acceptance to fix a four-step visual
history off-by-one (deque must hold four previous pooled embeddings). Its logs
are retained. Corrected pilot writes pilot_pairs_v2.
