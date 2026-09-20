# Bounded recovery experiments

Hypothesis: permanent OFT takeover is unnecessarily destructive when policies
are complementary. Let OFT repair the disturbed state for 32 or 96 fresh-feedback
actions, then return to Adapter with a new observation and an empty action queue.
A second candidate caps the stale Adapter bridge at two actions and conditions
the predicted OFT input on exactly those two committed actions.

Screen on TRAIN (trials 12–39), validate selected variants on DEV (40–49).
Trials 0–11 are excluded from tuning. The existing published/frozen artifacts
are read-only. No guarantee of outperforming a baseline is assumed.

The matched harness fixes all methods to 520 actual policy actions, applies the
same disturbance hook before every action (including first OFT and continuation),
records exact policy step counts, and reruns standalone Adapter and OFT controls.
Never-trigger outcomes must reproduce their new Adapter control exactly.

`forced_step` is a handoff-only diagnostic and deliberately also switches on
retain points. It is not an outcome oracle. `oracle_onset` switches only when
the freshly rerun Adapter fails and OFT succeeds; onset step is manifest + 2,
matching the historical oracle convention. `learned_soft_fusion` uses the frozen
learned detector unchanged. Compare every switching baseline with the same
trigger and evaluation harness. Record actual perturbation delivery separately.

Report paired success differences, rescue and retain, timing median/p95, no-stall,
and policy execution counts. Do not compare corrected-harness numbers directly
against historical numbers as if they were matched experiments.
