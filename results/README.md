# Frozen paper artifacts

This directory contains compact, durable outputs used by the paper. Large per-episode traces and intermediate training data are intentionally excluded from the public release.

## Current Table 2

- `table2_learned_k4_jointv2/`: final non-privileged continuous V1+V2 trigger, corrected runtime, joint deploy-matched residual, and `k=4`.
- `table2_oracle_k4_jointv2/`: matched oracle-onset upper-bound evaluation with the same handoff mechanism and `k=4`.

Both directories contain the strict analyzer's JSON and Markdown summaries. Generate new full traces under `runs/` with `jobs/run_table2.sbatch`.

## Frozen evaluation manifests

- `manifests/final_test/test_episode_manifest.json`: exact 238-cell TEST evaluation set retained for hash verification.
- `manifests/final_test/test_episode_manifest_portable.json`: the same cells with repository-relative perturbation paths; use this for new runs.
- `manifests/full10_perturbations/`: perturbation definitions referenced by the TEST manifest.
- `manifests/oracle_trigger/oracle_manifest.json`: privileged oracle-onset decisions used only for the oracle-trigger upper bound.

The TEST manifest preserves its original absolute scratch paths so its SHA-256 remains identical to the evaluated artifact. The matching perturbation files are copied here for archival portability.
