"""Record one live Adapter episode, with decision inputs separate from labels.

Phase convention: input row t is observation O_t before the perturbation hook
and action a_t. A perturbation applied by hook t is first observable at t+1.
The recorder is installed after warmup/idle; there are no implicit index offsets.
"""
import hashlib
import json
from pathlib import Path
import time
import tempfile

import numpy as np

from .features import ObservableFeatures, ObservableInput


def camera64(rgb):
    x = np.asarray(rgb)
    if x.dtype != np.uint8 or x.ndim != 3 or x.shape[2] != 3:
        raise ValueError('Expected orientation-correct RGB uint8 image')
    # Deterministic nearest-neighbor sampling; also used by the live trigger.
    yy = np.minimum(((np.arange(64) + .5) * x.shape[0] / 64).astype(int), x.shape[0]-1)
    xx = np.minimum(((np.arange(64) + .5) * x.shape[1] / 64).astype(int), x.shape[1]-1)
    return x[yy[:, None], xx[None, :]].copy()


def observable(prepared, action, step, chunk_position):
    return ObservableInput(
        full_image=camera64(prepared['full_image']),
        wrist_image=camera64(prepared['wrist_image']),
        robot_state=np.asarray(prepared['state'], dtype=np.float32).copy(),
        adapter_action=np.asarray(action, dtype=np.float32).copy(),
        episode_step=step, chunk_position=chunk_position)


def dump_json(path, value):
    # Multiple array shards may publish the identical collection gate report.
    # Unique temporary names avoid competing for the same .tmp file.
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                     prefix=path.name+'.', suffix='.tmp', delete=False) as f:
        json.dump(value, f, indent=2)
        temp = Path(f.name)
    temp.replace(path)


class LiveRecorder:
    def __init__(self, prefix, episode):
        self.prefix = Path(prefix)
        self.episode = dict(episode)  # Evaluation labels; never forwarded to features.
        self.extractor = ObservableFeatures()
        self.rows, self.events, self.post_states, self.done = [], [], [], []
        self.started = False
        self.action_digest = hashlib.sha256()
        self.previous_fired = False
        self.feature_times_ms = []

    def start(self, task_description):
        if self.started:
            raise ValueError('Recorder cannot be reused across episodes')
        self.started = True
        self.task_description = task_description

    def observe(self, episode_step, chunk_position, prepared, raw_action, processed_action):
        if not self.started or len(self.rows) != len(self.done):
            raise ValueError('Invalid observation phase')
        if episode_step != len(self.rows):
            raise ValueError('Non-contiguous real episode step')
        raw, executed = np.asarray(raw_action, dtype=np.float32), np.asarray(processed_action, dtype=np.float32)
        expected = raw.copy()
        expected[-1] = -np.sign(2 * raw[-1] - 1)
        if not np.array_equal(executed, expected):
            raise ValueError('Executed action was not processed exactly once')
        inp = observable(prepared, executed, episode_step, chunk_position)
        if self.post_states and not np.array_equal(inp.robot_state, self.post_states[-1]):
            raise ValueError('Observation does not follow the previous executed action')
        t0 = time.perf_counter()
        features = self.extractor.update(inp)
        self.feature_times_ms.append(1000 * (time.perf_counter()-t0))
        self.rows.append((inp, raw.copy(), executed.copy(), features.copy()))

    def event(self, episode_step, diagnostics, hook_body_before, hook_body_after):
        if episode_step != len(self.rows)-1 or len(self.events) != episode_step:
            raise ValueError('Event label not aligned with decision input')
        # Label-only payload. The feature extractor has already run and receives none of it.
        fired = bool(diagnostics and diagnostics.get('fired'))
        delta = None
        if hook_body_before is not None and hook_body_after is not None:
            delta = (np.asarray(hook_body_after)-np.asarray(hook_body_before)).tolist()
        self.events.append(dict(episode_step=episode_step, fired_transition=fired and not self.previous_fired,
                                hook_displacement=delta, diagnostics=dict(diagnostics or {})))
        self.previous_fired = fired

    def executed(self, episode_step, action, done, prepared_after):
        if episode_step != len(self.done) or len(self.events) != episode_step+1:
            raise ValueError('Executed action not aligned with event/input')
        action = np.asarray(action, dtype=np.float32)
        if not np.array_equal(action, self.rows[-1][2]):
            raise ValueError('Logged action differs from actual env.step action')
        self.action_digest.update(action.astype('<f4').tobytes())
        self.post_states.append(np.asarray(prepared_after['state'], dtype=np.float32).copy())
        self.done.append(bool(done))

    def finish(self, final_success, episode_step):
        n = len(self.rows)
        if not n or n != episode_step or n != len(self.done) or n != len(self.events):
            raise ValueError('Episode length/phase mismatch')
        if any(self.done[:-1]) or bool(final_success) != self.done[-1]:
            raise ValueError('Terminal-state mismatch')
        inputs = [r[0] for r in self.rows]
        arrays = dict(
            episode_step=np.arange(n, dtype=np.int32),
            chunk_position=np.asarray([x.chunk_position for x in inputs], dtype=np.int8),
            robot_state=np.stack([x.robot_state for x in inputs]),
            full_image=np.stack([x.full_image for x in inputs]),
            wrist_image=np.stack([x.wrist_image for x in inputs]),
            raw_action=np.stack([r[1] for r in self.rows]),
            executed_action=np.stack([r[2] for r in self.rows]),
            features=np.stack([r[3] for r in self.rows]),
            post_robot_state=np.stack(self.post_states), done=np.asarray(self.done),
        )
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        npz = self.prefix.with_suffix('.npz')
        temp = npz.with_suffix('.npz.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f, **arrays)
        temp.replace(npz)
        labels = dict(schema_version=1, episode=self.episode, task_description=self.task_description,
                      feature_names=list(self.extractor.feature_names),
                      n_steps=n, final_success=bool(final_success), events=self.events,
                      input_phase='pre_hook_pre_action', first_observable_push='application_step + 1',
                      action_sha256=self.action_digest.hexdigest(),
                      arrays_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
                      feature_ms_median=float(np.median(self.feature_times_ms)),
                      feature_ms_p95=float(np.quantile(self.feature_times_ms, .95)))
        dump_json(self.prefix.with_suffix('.labels.json'), labels)
        return labels


def validate(prefix):
    """Recompute from the exact runtime input rows, never simulator replay."""
    prefix = Path(prefix)
    labels = json.loads(prefix.with_suffix('.labels.json').read_text())
    npz = prefix.with_suffix('.npz')
    assert hashlib.sha256(npz.read_bytes()).hexdigest() == labels['arrays_sha256']
    with np.load(npz, allow_pickle=False) as packed:
        a = {k: packed[k] for k in packed.files}
        n = labels['n_steps']
        assert all(len(v) == n for v in a.values())
        assert np.array_equal(a['episode_step'], np.arange(n))
        assert np.array_equal(a['robot_state'][1:], a['post_robot_state'][:-1])
        assert not a['done'][:-1].any() and bool(a['done'][-1]) == labels['final_success']
        expected = a['raw_action'].copy()
        expected[:, -1] = -np.sign(2*expected[:, -1]-1)
        assert np.array_equal(expected, a['executed_action'])
        assert hashlib.sha256(a['executed_action'].astype('<f4').tobytes()).hexdigest() == labels['action_sha256']
        extractor = ObservableFeatures()
        assert labels['feature_names'] == list(extractor.feature_names)
        for i in range(n):
            inp = ObservableInput(full_image=a['full_image'][i], wrist_image=a['wrist_image'][i],
                                  robot_state=a['robot_state'][i], adapter_action=a['executed_action'][i],
                                  episode_step=i, chunk_position=int(a['chunk_position'][i]))
            assert np.array_equal(extractor.update(inp), a['features'][i]), f'Feature mismatch at step {i}'
        assert np.isfinite(a['features']).all()
    assert [e['episode_step'] for e in labels['events']] == list(range(n))
    applied = [e['episode_step'] for e in labels['events']
               if e['hook_displacement'] is not None and np.linalg.norm(e['hook_displacement']) > 1e-10]
    fired = [e['episode_step'] for e in labels['events'] if e['fired_transition']]
    assert len(fired) <= 1
    if labels['episode']['condition'] == 'clean':
        assert not applied and not fired
    if applied:
        assert fired and fired[0] == applied[0], 'Hook firing and physical displacement disagree'
    return dict(n_steps=n, feature_replay_exact=True, action_processing_exact=True,
                physical_application_steps=applied, first_observable_step=applied[0]+1 if applied else None,
                final_success=labels['final_success'])
