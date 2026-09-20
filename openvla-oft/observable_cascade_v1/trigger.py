"""Observable cascade inference; no simulation or evaluation imports."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .features import CandidateFeatures, FEATURE_NAMES, SCHEMA
from artifact_paths import resolve_artifact_path


def load_artifact(path):
    art = json.loads(Path(path).read_text())
    if (art.get('schema') != SCHEMA or art.get('privileged_inputs') is not False
            or art.get('feature_names') != list(FEATURE_NAMES)):
        raise ValueError('V2 input contract mismatch')
    if art.get('policy') not in ('direct','utility','failure'):
        raise ValueError('Unknown V2 policy')
    if not np.isfinite(art.get('threshold',float('nan'))):
        raise ValueError('Invalid threshold')
    if art['policy'] in ('utility','failure'):
        n=len(FEATURE_NAMES)
        for name in ('mean','scale'):
            values=np.asarray(art.get(name),dtype=float)
            if values.shape!=(n,) or not np.isfinite(values).all():
                raise ValueError(f'Invalid {name}')
        if np.any(np.asarray(art['scale'])<=0):
            raise ValueError('Scale must be positive')
        head_names = ('rescue','harm') if art['policy']=='utility' else ('failure',)
        for name in head_names:
            head=art.get(name,{})
            values=np.asarray(head.get('coef'),dtype=float)
            if values.shape!=(n,) or not np.isfinite(values).all() or not np.isfinite(head.get('intercept',float('nan'))):
                raise ValueError(f'Invalid {name} head')
    return art


def utility_score(features, art):
    if art['policy'] == 'direct':
        return 1.0
    x = np.asarray(features, dtype=np.float64)
    if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
        raise ValueError('Invalid V2 input')
    z = np.clip((x-np.asarray(art['mean']))/np.asarray(art['scale']), -10, 10)
    def probability(head):
        logit = float(z @ np.asarray(head['coef']) + head['intercept'])
        return float(1/(1+np.exp(-np.clip(logit,-30,30))))
    if art['policy'] == 'failure':
        return probability(art['failure'])
    return probability(art['rescue'])-probability(art['harm'])


class ObservableCascadeTrigger:
    def __init__(self, artifact_path):
        artifact_path = resolve_artifact_path(artifact_path)
        self.art = load_artifact(artifact_path)
        p = resolve_artifact_path(self.art['v1_artifact'], anchor=artifact_path)
        if hashlib.sha256(p.read_bytes()).hexdigest() != self.art['v1_artifact_sha256']:
            raise ValueError('V1 artifact changed after V2 fitting')
        gate = json.loads(p.read_text())
        checkpoint = resolve_artifact_path(gate['checkpoint'], anchor=p)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != self.art['v1_weights_sha256']:
            raise ValueError('V1 weights changed after V2 fitting')
        self.features = CandidateFeatures(p)
        self.decided = False
        self.trace = []

    def update(self, *, observation, adapter_action, chunk_position, episode_step):
        from asyncmixvla.trigger import TriggerResult
        if set(observation) != {'full_image','wrist_image','state'}:
            raise ValueError('Only sanitized camera/proprio observations are accepted')
        if self.decided:
            return TriggerResult(should_switch=False)
        candidate, x, score = self.features.update(observation=observation,
            adapter_action=adapter_action, chunk_position=chunk_position, episode_step=episode_step)
        utility = utility_score(x, self.art) if candidate else None
        switch = candidate and (self.art['policy']=='direct' or utility >= self.art['threshold'])
        if candidate:
            self.decided = True
        self.trace.append(dict(episode_step=episode_step,v1_score=score,v2_utility=utility,
                               candidate=candidate,should_switch=bool(switch)))
        return TriggerResult(should_switch=bool(switch), trigger_score=utility if candidate else score,
            trigger_metadata={'backend':'observable_cascade_v1','privileged_inputs':False})
