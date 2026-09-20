# Corrected TEST_K4_JOINTV2: v1_v2_cascade_continuous

Matched cells: **238/238**. Integrity findings: **0**.

| Method | Overall | Clean | Perturbed | Rescued Adapter failures | Retain | Invocation | L_handoff median / p95 (ms) | No-stall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Adapter only | 81.1% | 92.5% | 69.5% | 0/45 | 193/193 | 0.0% (0) | -- | -- |
| Full OFT only | 83.2% | 92.5% | 73.7% | 26/45 | 172/193 | 100.0% (238) | -- | -- |
| Synchronous switching | 84.5% | 92.5% | 76.3% | 16/45 | 185/193 | 45.8% (109) | 226.5 / 363.8 | 0.0% |
| Naive asynchronous | 83.2% | 92.5% | 73.7% | 16/45 | 182/193 | 45.8% (109) | 0.1 / 200.1 | 81.7% |
| Ours (AsyncMixVLA) | 84.9% | 92.5% | 77.1% | 17/45 | 185/193 | 45.8% (109) | 0.1 / 224.5 | 76.1% |

## Paired comparisons

- `ours_vs_sync`: ours wins 5, other wins 4, ties 229, exact McNemar p=1.0000
- `ours_vs_naive`: ours wins 5, other wins 1, ties 232, exact McNemar p=0.2188

## Notes (non-blocking)

- perturbation never_reached (known, non-blocking -- matches the ~0.8% base rate already present in fit/calibration/DEV, see comment above): 2/118 perturbed cells (1.7%) -- [(2, 7, 'perturbed'), (8, 6, 'perturbed')]
