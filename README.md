# AsyncMixVLA

AsyncMixVLA overlaps a fast Adapter policy with an OFT policy and aligns the handoff context when a non-privileged visual/proprioceptive trigger requests recovery. This repository is the compact release of the current LIBERO-10 system: corrected runtime, continuous V1+V2 trigger, joint deploy-matched residual, and `k=4` OFT takeover.

The learned trigger receives only deployable observations: front/wrist RGB, robot proprioception, Adapter actions, and their history. The oracle-onset trigger is included only as a privileged upper bound.

## Frozen LIBERO-10 results

The strict learned-trigger report contains 238/238 matched evaluation cells and no integrity findings.

| Method | Overall SR | Clean SR | Perturbed SR | Invocation |
|---|---:|---:|---:|---:|
| Adapter only | 81.09% | 92.50% | 69.49% | 0.0% |
| Full OFT only | 83.19% | 92.50% | 73.73% | 100.0% |
| Synchronous switching | 84.45% | 92.50% | 76.27% | 45.8% |
| Naive asynchronous | 83.19% | 92.50% | 73.73% | 45.8% |
| **AsyncMixVLA** | **84.87%** | **92.50%** | **77.12%** | **45.8%** |

See [`results/table2_learned_k4_jointv2`](results/table2_learned_k4_jointv2) for the complete report and [`results/table2_oracle_k4_jointv2`](results/table2_oracle_k4_jointv2) for the matched oracle-trigger upper bound.

## Repository layout

- `openvla-oft/asyncmixvla`: asynchronous bridge and state/vision alignment.
- `openvla-oft/visual_trigger_v1`: deployable visual disturbance detector.
- `openvla-oft/observable_cascade_v1`: continuous V1+V2 trigger.
- `openvla-oft/research_recovery_v1`: corrected runtime that continues perturbations through OFT execution.
- `openvla-oft/*switching*corrected.py`: final evaluation and strict analysis.
- `VLA-Adapter/serve_adapter_libero10.py`: Adapter inference server.
- `jobs`: portable Slurm launch and analysis scripts.
- `results/manifests`: frozen and portable evaluation manifests.
- `docs/ARTIFACTS.md`: included artifacts and external model requirements.

## Installation

The Adapter and OFT servers require different Python environments because their Transformers dependencies differ. Python 3.11, PyTorch 2.2.0, and CUDA 12.1 match the evaluated environment.

First install LIBERO once and link it into both source trees:

```bash
./scripts/install_libero.sh
```

Create the OFT environment, install a CUDA-compatible PyTorch build, and then install the local source:

```bash
python3.11 -m venv .venv-oft
.venv-oft/bin/pip install --upgrade pip setuptools wheel
.venv-oft/bin/pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121
.venv-oft/bin/pip install -e openvla-oft
.venv-oft/bin/pip install -e third_party/LIBERO
.venv-oft/bin/pip install -r openvla-oft/experiments/robot/libero/libero_requirements.txt
.venv-oft/bin/pip install packaging ninja
.venv-oft/bin/pip install flash-attn==2.5.5 --no-build-isolation
```

Repeat for the Adapter environment:

```bash
python3.11 -m venv .venv-adapter
.venv-adapter/bin/pip install --upgrade pip setuptools wheel
.venv-adapter/bin/pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121
.venv-adapter/bin/pip install -e VLA-Adapter
.venv-adapter/bin/pip install -e third_party/LIBERO
.venv-adapter/bin/pip install -r VLA-Adapter/experiments/robot/libero/libero_requirements.txt
.venv-adapter/bin/pip install packaging ninja
.venv-adapter/bin/pip install flash-attn==2.5.5 --no-build-isolation
```

The large Adapter and OFT policy checkpoints are not stored in Git. Set their locations as described in [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md).

The evaluated environments reported LIBERO 0.1.0 and robosuite 1.4.1. Set `LIBERO_REF` before running the installer if you want to pin a specific LIBERO commit for an archival reproduction.

## Verify the release

The verification command checks all frozen hashes, the 238-cell portable manifest, artifact path resolution, Python syntax, and accidental large files:

```bash
python scripts/verify_release.py
```

For a model-free trigger test in the OFT environment:

```bash
PYTHONPATH=openvla-oft .venv-oft/bin/python scripts/smoke_trigger.py
```

## Run Table 2

Export the two base policy checkpoints and Python executables, then submit the ten-task array:

```bash
export ADAPTER_CHECKPOINT=/path/to/VLA-Adapter-LIBERO-Long
export OFT_CHECKPOINT=/path/to/openvla-7b-oft-finetuned-libero-10
export ADAPTER_PYTHON="$PWD/.venv-adapter/bin/python"
export OFT_PYTHON="$PWD/.venv-oft/bin/python"
sbatch jobs/run_table2.sbatch
```

After all array tasks finish:

```bash
jobs/analyze_table2.sh
```

Run the oracle upper bound with `TRIGGER=oracle_onset sbatch jobs/run_table2.sbatch`. The learned trigger is the default.

## Scope

The trigger has a deployable input contract and contains no simulator-only state. The released calibration is specific to LIBERO-10 and the evaluated translation-style disturbances; deployment on a new robot, camera setup, task suite, or disturbance family requires new calibration and validation.

The continuous V2 policy re-arms after a rejected V1 candidate. Its V2 head was trained on first-candidate examples, so later-candidate scoring is an evaluated benchmark extension of that training distribution. The frozen artifact retains `deployment_approved=false` to record this validation scope.

## Acknowledgements and license

This repository contains modified OpenVLA-OFT and VLA-Adapter source. Both upstream projects use the MIT License. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the preserved licenses inside each source tree.
