# Corrected TEST_K4_JOINTV2: oracle_onset

Matched cells: **238/238**. Integrity findings: **0**.

| Method | Overall | Clean | Perturbed | Rescued Adapter failures | Retain | Invocation | L_handoff median / p95 (ms) | No-stall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Adapter only | 81.1% | 92.5% | 69.5% | 0/45 | 193/193 | 0.0% (0) | -- | -- |
| Full OFT only | 83.2% | 92.5% | 73.7% | 26/45 | 172/193 | 100.0% (238) | -- | -- |
| Synchronous switching | 85.3% | 96.7% | 73.7% | 16/45 | 187/193 | 10.9% (26) | 143.4 / 150.5 | 0.0% |
| Naive asynchronous | 85.3% | 95.0% | 75.4% | 16/45 | 187/193 | 10.9% (26) | 0.1 / 48.8 | 92.3% |
| Ours (AsyncMixVLA) | 85.7% | 95.8% | 75.4% | 17/45 | 187/193 | 10.9% (26) | 0.1 / 214.4 | 84.6% |

## Paired comparisons

- `ours_vs_sync`: ours wins 2, other wins 1, ties 235, exact McNemar p=1.0000
- `ours_vs_naive`: ours wins 2, other wins 1, ties 235, exact McNemar p=1.0000

## Notes (non-blocking)

- perturbation never_reached (known, non-blocking -- matches the ~0.8% base rate already present in fit/calibration/DEV, see comment above): 2/118 perturbed cells (1.7%) -- [(2, 7, 'perturbed'), (8, 6, 'perturbed')]
