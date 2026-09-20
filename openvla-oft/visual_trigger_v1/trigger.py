"""Live visual disturbance gate. Non-privileged: camera + proprio + action only.

Image preprocessing and input packaging are IMPORTED from deployable_trigger_v1.recording
(camera64 / observable) -- the exact functions used when the training data was recorded.
Reimplementing them here is what produced the earlier train/serve calibration bug.
"""
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from deployable_trigger_v1.recording import camera64
from visual_trigger_v1.model import VisualGate
from artifact_paths import resolve_artifact_path


class VisualGateTrigger:
    def __init__(self, artifact_path, device="cpu"):
        artifact_path = resolve_artifact_path(artifact_path)
        art = json.loads(artifact_path.read_text())
        if art.get("input_contract") != "observable_v1" or art.get("privileged_inputs") is not False:
            raise ValueError("Artifact does not certify observable-only inputs")
        self.lags = tuple(art["lags"])
        self.threshold = float(art["threshold"])
        self.persistence = int(art["persistence"])
        if not 0 < self.threshold <= 1 or self.persistence < 1:
            raise ValueError("Invalid calibration in artifact")
        self.device = device
        checkpoint = resolve_artifact_path(art["checkpoint"], anchor=artifact_path)
        ck = torch.load(checkpoint, map_location=device)
        if tuple(ck["lags"]) != self.lags:
            raise ValueError("Checkpoint/artifact lag mismatch")
        self.model = VisualGate().to(device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        hist = max(self.lags) + 1
        self.front, self.wrist = deque(maxlen=hist), deque(maxlen=hist)
        self.consecutive, self.fired = 0, False
        self.trace = []

    def _stack(self):
        chans = []
        for buf in (self.front, self.wrist):
            cur = buf[-1].astype(np.float32) / 255.0
            chans.append(cur)
            for lag in self.lags:
                prev = buf[max(0, len(buf) - 1 - lag)]
                chans.append(cur - prev.astype(np.float32) / 255.0)
        return np.concatenate(chans, axis=2).transpose(2, 0, 1)

    @torch.no_grad()
    def update(self, *, observation, adapter_action, chunk_position, episode_step):
        from asyncmixvla.trigger import TriggerResult
        if set(observation) != {"full_image", "wrist_image", "state"}:
            raise ValueError("Only sanitized camera/proprio observations are accepted")
        self.front.append(camera64(observation["full_image"]))
        self.wrist.append(camera64(observation["wrist_image"]))
        aux = np.concatenate([np.asarray(observation["state"], dtype=np.float32),
                              np.asarray(adapter_action, dtype=np.float32)])
        x = torch.from_numpy(self._stack()[None]).to(self.device)
        a = torch.from_numpy(aux[None]).to(self.device)
        score = float(torch.sigmoid(self.model(x, a))[0])
        self.consecutive = self.consecutive + 1 if score >= self.threshold else 0
        switch = not self.fired and self.consecutive >= self.persistence
        self.fired = self.fired or switch
        self.trace.append(dict(episode_step=episode_step, score=score, should_switch=switch))
        return TriggerResult(should_switch=switch, trigger_score=score,
                             trigger_metadata={"backend": "visual_gate_v1", "privileged_inputs": False})
