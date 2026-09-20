"""Continuous (re-arming) V1+V2 cascade. Uses the ALREADY-FITTED v2_candidate.json
model exactly as-is -- no retraining -- only the runtime POLICY of when V2 is
allowed to look changes.

The one-shot ObservableCascadeTrigger's limit traces to V1 itself: VisualGateTrigger
latches (self.fired) after its first switch decision, so it only ever offers ONE
candidate moment per episode -- V2 never gets a second one to evaluate. This class
reimplements V1's own persistence/threshold rule WITHOUT the permanent latch, so a
declined candidate does not foreclose a later one: after V2 says no, the persistence
counter resets and a fresh run of `persistence` consecutive above-threshold steps
raises a new, independent candidate for V2 to score.

Same inputs as the one-shot cascade (camera + proprio + action only; no privileged
state). Reuses V1's own CNN forward pass and V2's fitted rescue/harm heads verbatim
via features.py/trigger.py -- this file only changes candidate-arming, not scoring.

CAVEAT (disclose in any report): V2's rescue/harm heads were fit ONLY on first-
candidate examples (see PROTOCOL.md / collect.py -- pair collection assumes one
candidate per episode). Applying them to later candidates is extrapolation beyond
the training distribution, not a validated use -- this run is what tells us whether
that extrapolation holds.
"""
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from visual_trigger_v1.model import VisualGate
from deployable_trigger_v1.recording import camera64, observable
from deployable_trigger_v1.features import ObservableFeatures
from .trigger import load_artifact, utility_score
from .features import FEATURE_NAMES
from artifact_paths import resolve_artifact_path


class ContinuousObservableCascadeTrigger:
    def __init__(self, artifact_path, device="cpu", observe_only=False):
        artifact_path = resolve_artifact_path(artifact_path)
        self.art = load_artifact(artifact_path)
        if self.art["policy"] not in ("utility", "failure"):
            raise ValueError("ContinuousObservableCascadeTrigger requires a scored V2 artifact")
        self.observe_only = bool(observe_only)
        p = resolve_artifact_path(self.art["v1_artifact"], anchor=artifact_path)
        # Same pinning as the one-shot ObservableCascadeTrigger (trigger.py): V2's
        # fitted heads are only valid against the exact V1 gate/weights they were
        # trained against. This was missing here -- a real gap, not a stylistic
        # omission -- since silently loading a since-changed V1 would make V2's
        # candidate-time features (which embed V1's own CNN features/score)
        # meaningless without warning.
        if hashlib.sha256(p.read_bytes()).hexdigest() != self.art["v1_artifact_sha256"]:
            raise ValueError("V1 artifact changed after V2 fitting")
        gate = json.loads(p.read_text())
        checkpoint = resolve_artifact_path(gate["checkpoint"], anchor=p)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != self.art["v1_weights_sha256"]:
            raise ValueError("V1 weights changed after V2 fitting")
        self.threshold = float(gate["threshold"])
        self.persistence = int(gate["persistence"])
        if not 0 < self.threshold <= 1 or self.persistence < 1:
            raise ValueError("Invalid V1 calibration in gate artifact")
        self.device = device
        ck = torch.load(checkpoint, map_location=device)
        self.lags = tuple(ck["lags"])
        self.model = VisualGate().to(device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._spatial = None
        self.model.cnn[7].register_forward_hook(lambda m, a, out: setattr(self, "_spatial", out.detach()))

        hist = max(self.lags) + 1
        self.front, self.wrist = deque(maxlen=hist), deque(maxlen=hist)
        self.robot = ObservableFeatures()
        self.emb_history = deque(maxlen=4)
        self.consecutive = 0
        self.switched = False  # still only ONE actual switch: takeover_k=1, no going back to Adapter
        self.trace = []
        self.n_candidates = 0
        self.last_features = None

    def _visual_stack(self):
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
        if self.switched:
            return TriggerResult(should_switch=False)

        self.front.append(camera64(observation["full_image"]))
        self.wrist.append(camera64(observation["wrist_image"]))
        aux = np.concatenate([np.asarray(observation["state"], dtype=np.float32),
                              np.asarray(adapter_action, dtype=np.float32)])
        x = torch.from_numpy(self._visual_stack()[None]).to(self.device)
        a = torch.from_numpy(aux[None]).to(self.device)
        v1_score = float(torch.sigmoid(self.model(x, a))[0])

        robot = self.robot.update(observable(observation, adapter_action, episode_step, chunk_position))
        spatial = torch.nn.functional.adaptive_avg_pool2d(self._spatial, (2, 2))[0].cpu().numpy()
        pooled = spatial.mean(axis=(1, 2))
        previous = self.emb_history[0] if self.emb_history else pooled
        delta = pooled - previous
        self.emb_history.append(pooled.copy())
        v2_features = np.concatenate([robot, spatial.ravel(), delta, [v1_score]]).astype(np.float32)
        if v2_features.shape != (len(FEATURE_NAMES),) or not np.isfinite(v2_features).all():
            raise ValueError("Invalid V2 features")
        self.last_features = v2_features.copy()

        # RE-ARMING persistence: unlike VisualGateTrigger, the counter resets after
        # every candidate (approved or declined) instead of latching forever, so a
        # fresh run of `persistence` steps above threshold raises a NEW candidate.
        self.consecutive = self.consecutive + 1 if v1_score >= self.threshold else 0
        is_candidate = self.consecutive >= self.persistence
        utility = None
        switch = False
        if is_candidate:
            self.consecutive = 0  # re-arm regardless of V2's decision
            self.n_candidates += 1
            utility = utility_score(v2_features, self.art)
            switch = utility >= self.art["threshold"]

        self.trace.append(dict(episode_step=episode_step, v1_score=v1_score, candidate=is_candidate,
                               v2_utility=utility, should_switch=bool(switch),
                               candidate_index=self.n_candidates if is_candidate else None))
        if switch and not self.observe_only:
            self.switched = True
        return TriggerResult(should_switch=bool(switch and not self.observe_only), trigger_score=(utility if is_candidate else v1_score),
            trigger_metadata={"backend": "observable_cascade_v1_continuous", "privileged_inputs": False,
                              "n_candidates_seen": self.n_candidates})
