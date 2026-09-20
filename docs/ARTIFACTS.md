# Artifacts

## Included frozen artifacts

| Component | Path | SHA-256 |
|---|---|---|
| V1 visual trigger | `openvla-oft/visual_trigger_v1/models/visual_gate.pt` | `63e8c29059e0decd201e9b7232c0b5ed09dd8b7fbfe78f16a5bf2e007b149f56` |
| V1 operating point | `openvla-oft/visual_trigger_v1/models/visual_gate_fpr10.json` | `5a72f0dc624aa9ea19eee7cb3c6b38a4a7c3345fb7823478237c719d87187164` |
| Continuous V2 trigger | `openvla-oft/observable_cascade_v1/models/v2_candidate.json` | `4cbcc007ee99c908a4d3062a14c4334bff2ffc5f5f85f3bc57e933626a3b1e0e` |
| Proprioceptive calibration | `openvla-oft/asyncmixvla/calibration/vlash_gain_calibration.json` | `0f0ed7e9f811df1798c7b45b8a2ee477db0c19afda6f888de2eca839ff8c298f` |
| Joint deploy-matched residual | `openvla-oft/asyncmixvla/calibration/joint_deploy_residual_v2.pt` | `5e3e87be2ad35de5164c560d72b3b02cac6df8bba42050721d2b75be4e032fd1` |

`results/CURRENT_MANIFEST.sha256` verifies the frozen method artifacts, reports, and exact evaluation manifests. Frozen JSON files retain their original absolute paths so their hashes remain identical to the evaluated files. `openvla-oft/artifact_paths.py` resolves those paths within a relocated checkout.

## External base models

The policy checkpoints are excluded because the evaluated directories total approximately 35 GB. The launch scripts require:

- `ADAPTER_CHECKPOINT`: VLA-Adapter LIBERO-Long checkpoint directory. The evaluated directory was approximately 2.6 GB and contained `model.safetensors`, `action_head--checkpoint.pt`, and `proprio_projector--checkpoint.pt`.
- `OFT_CHECKPOINT`: OpenVLA-OFT LIBERO-10 checkpoint directory. The evaluated directory was approximately 15 GB and contained four model safetensor shards, `action_head--150000_checkpoint.pt`, and `proprio_projector--150000_checkpoint.pt`.

Obtain these checkpoints from the corresponding upstream projects or your trained copies. Results are only directly comparable when the base policies match the reported Adapter-only and Full-OFT outcomes.

## Manifests

`test_episode_manifest.json` is the exact evaluated manifest and retains original absolute paths for hash verification. `test_episode_manifest_portable.json` contains the same 238 cells and outcome labels, with only the perturbation-file paths rewritten relative to this repository. The portable file is the default for new runs.
