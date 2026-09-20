"""Deployable scorer: explicit observable inputs, no simulator dependency."""
import json
from pathlib import Path

import numpy as np

from .features import ObservableFeatures
from .recording import observable


class DeployableTrigger:
    def __init__(self, checkpoint):
        if checkpoint is None:
            raise ValueError('Deployable trigger requires a calibrated checkpoint')
        self.artifact = json.loads(Path(checkpoint).read_text())
        self.extractor = ObservableFeatures()
        if self.artifact.get('input_contract') != 'observable_v1':
            raise ValueError('Checkpoint does not certify observable-only inputs')
        if self.artifact.get('feature_names') != list(self.extractor.feature_names):
            raise ValueError('Feature schema mismatch')
        indices = np.asarray(self.artifact['feature_indices'])
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices):
            raise ValueError('Feature indices must be a nonempty integer vector')
        if np.any(indices < 0) or np.any(indices >= self.extractor.num_features) or len(set(indices.tolist())) != len(indices):
            raise ValueError('Invalid feature indices')
        self.indices = indices.astype(int)
        self.mean = np.asarray(self.artifact['mean'], dtype=np.float64)
        self.scale = np.asarray(self.artifact['scale'], dtype=np.float64)
        self.coef = np.asarray(self.artifact['coef'], dtype=np.float64)
        if any(v.ndim != 1 for v in [self.mean, self.scale, self.coef]) or not (len(self.indices) == len(self.mean) == len(self.scale) == len(self.coef)):
            raise ValueError('Checkpoint dimension mismatch')
        if np.any(self.scale <= 0) or not all(np.isfinite(v).all() for v in [self.mean, self.scale, self.coef]):
            raise ValueError('Invalid checkpoint normalization')
        self.intercept = float(self.artifact['intercept'])
        self.threshold = float(self.artifact['threshold'])
        self.persistence = self.artifact['persistence']
        if isinstance(self.persistence, bool) or not isinstance(self.persistence, int):
            raise ValueError('Persistence must be an integer')
        if not np.isfinite(self.intercept) or not 0 < self.threshold <= 1 or self.persistence < 1:
            raise ValueError('Invalid episode-level calibration')
        self.consecutive = 0
        self.fired = False
        self.trace = []

    def score_features(self, features):
        z = np.clip((features[self.indices]-self.mean)/self.scale, -10, 10)
        logit = np.clip(z @ self.coef + self.intercept, -30, 30)
        return float(1/(1+np.exp(-logit)))

    def update(self, *, observation, adapter_action, chunk_position, episode_step):
        from asyncmixvla.trigger import TriggerResult
        if set(observation) != {'full_image', 'wrist_image', 'state'}:
            raise ValueError('Only sanitized camera/proprio observations are accepted')
        features = self.extractor.update(observable(observation, adapter_action, episode_step, chunk_position))
        score = self.score_features(features)
        self.consecutive = self.consecutive+1 if score >= self.threshold else 0
        switch = not self.fired and self.consecutive >= self.persistence
        self.fired = self.fired or switch
        self.trace.append(dict(episode_step=episode_step, score=score, should_switch=switch))
        return TriggerResult(should_switch=switch, trigger_score=score,
                             trigger_metadata={'backend':'deployable_v1', 'privileged_inputs':False})
