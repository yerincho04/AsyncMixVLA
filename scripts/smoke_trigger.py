#!/usr/bin/env python3
"""Model-free construction and forward-pass smoke test for the frozen trigger."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OFT = ROOT / "openvla-oft"

import sys
sys.path.insert(0, str(OFT))

from artifact_paths import resolve_artifact_path
from observable_cascade_v1.continuous_trigger import ContinuousObservableCascadeTrigger


def main():
    artifact = OFT / "observable_cascade_v1/models/v2_candidate.json"
    trigger = ContinuousObservableCascadeTrigger(artifact, device="cpu")
    observation = {
        "full_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "state": np.zeros(8, dtype=np.float32),
    }
    result = trigger.update(
        observation=observation,
        adapter_action=np.zeros(7, dtype=np.float32).tolist(),
        chunk_position=0,
        episode_step=0,
    )
    art = json.loads(artifact.read_text())
    assert art["privileged_inputs"] is False
    assert resolve_artifact_path(art["v1_artifact"]).exists()
    assert result.trigger_metadata["privileged_inputs"] is False
    print("trigger smoke test: PASS")


if __name__ == "__main__":
    main()
