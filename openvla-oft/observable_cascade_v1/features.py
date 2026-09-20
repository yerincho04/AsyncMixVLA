"""Causal V2 features. No environment, labels, or task/perturbation metadata."""
from collections import deque
import numpy as np
import torch
from deployable_trigger_v1.features import ObservableFeatures
from deployable_trigger_v1.recording import observable
from visual_trigger_v1.trigger import VisualGateTrigger

SCHEMA = 'observable_cascade_spatial_v1'
FEATURE_NAMES = tuple(ObservableFeatures.feature_names) + tuple(
    f'visual.spatial.{i}' for i in range(512)) + tuple(
    f'visual.delta4.{i}' for i in range(128)) + ('v1.score',)

class CandidateFeatures:
    def __init__(self, gate_artifact):
        self.gate = VisualGateTrigger(gate_artifact)
        self.robot = ObservableFeatures()
        self.history = deque(maxlen=4)
        self.last_features = None
        self.first_candidate = None
        # Reuse the exact gate forward pass, avoiding a second CNN evaluation.
        self.spatial = None
        self._handle = self.gate.model.cnn[7].register_forward_hook(self._capture)

    def _capture(self, module, args, output):
        self.spatial = output.detach()

    @torch.no_grad()
    def update(self, *, observation, adapter_action, chunk_position, episode_step):
        if set(observation) != {'full_image','wrist_image','state'}:
            raise ValueError('Only sanitized camera/proprio observations are accepted')
        robot = self.robot.update(observable(observation, adapter_action, episode_step, chunk_position))
        result = self.gate.update(observation=observation, adapter_action=adapter_action,
                                 chunk_position=chunk_position, episode_step=episode_step)
        spatial = torch.nn.functional.adaptive_avg_pool2d(self.spatial, (2,2))[0].cpu().numpy()
        pooled = spatial.mean(axis=(1,2))
        previous = self.history[0] if self.history else pooled
        delta = pooled-previous
        self.history.append(pooled.copy())
        self.last_features = np.concatenate([robot, spatial.ravel(), delta,
                                             [result.trigger_score]]).astype(np.float32)
        if self.last_features.shape != (len(FEATURE_NAMES),) or not np.isfinite(self.last_features).all():
            raise ValueError('Invalid V2 features')
        candidate = bool(result.should_switch)
        if candidate:
            self.first_candidate = episode_step
        return candidate, self.last_features.copy(), result.trigger_score
