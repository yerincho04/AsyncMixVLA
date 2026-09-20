"""Synthetic, simulator-free checks of the live recording phase contract."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .features import ObservableFeatures, ObservableInput
from .recording import LiveRecorder, camera64, validate


def prepared_at(step, push_step=2):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    # Hook application at t first changes the next returned observation, O_(t+1).
    if step > push_step:
        image[:16, :16] = 255
    state = np.zeros(8, dtype=np.float32)
    state[0] = step * 0.001
    return dict(full_image=image, wrist_image=np.zeros_like(image), state=state)


def action_pair(raw_gripper=1.0):
    raw = np.array([0.1, 0, 0, 0, 0, 0, raw_gripper], dtype=np.float32)
    processed = raw.copy()
    processed[-1] = -np.sign(2 * raw[-1] - 1)
    return raw, processed


def record_episode(prefix, push_step=2, raw_grippers=(1, 0, 0, 1, 0, 0, 0)):
    recorder = LiveRecorder(prefix, dict(condition='perturbed', task_id=4, trial_index=20,
                                          perturbation_manifest='evaluation-label-only.json'))
    recorder.start('Put the object away')
    for step, raw_gripper in enumerate(raw_grippers):
        raw, processed = action_pair(raw_gripper)
        recorder.observe(step, step % 8, prepared_at(step, push_step), raw, processed)
        before = np.zeros(3)
        after = np.array([0.02, 0, 0]) if step == push_step else before.copy()
        recorder.event(step, dict(fired=step >= push_step, private_object_pose=[1, 2, 3]), before, after)
        recorder.executed(step, processed, step == len(raw_grippers)-1, prepared_at(step+1, push_step))
    labels = recorder.finish(True, len(raw_grippers))
    return recorder, labels


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.prefix = Path(self.temp.name) / 'episode'

    def new_recorder(self):
        recorder = LiveRecorder(self.prefix, dict(condition='clean'))
        recorder.start('task instruction')
        return recorder

    def test_hook_changes_next_observation_and_features_replay_byte_exactly(self):
        _, labels = record_episode(self.prefix)
        report = validate(self.prefix)
        self.assertEqual(report['physical_application_steps'], [2])
        self.assertEqual(report['first_observable_step'], 3)
        self.assertEqual([event['episode_step'] for event in labels['events'] if event['fired_transition']], [2])
        with np.load(self.prefix.with_suffix('.npz'), allow_pickle=False) as arrays:
            self.assertEqual(int(arrays['full_image'][2].sum()), 0)
            self.assertGreater(int(arrays['full_image'][3].sum()), 0)
            replay = ObservableFeatures()
            for step in range(labels['n_steps']):
                vector = replay.update(ObservableInput(
                    full_image=arrays['full_image'][step], wrist_image=arrays['wrist_image'][step],
                    robot_state=arrays['robot_state'][step], adapter_action=arrays['executed_action'][step],
                    episode_step=step, chunk_position=int(arrays['chunk_position'][step])))
                self.assertEqual(vector.tobytes(), arrays['features'][step].tobytes())
            motion_column = list(replay.feature_names).index('front.lag1.residual_grid.r0c0')
            self.assertEqual(arrays['features'][2, motion_column], 0)
            self.assertGreater(arrays['features'][3, motion_column], 0)
            self.assertEqual(set(arrays.files), {
                'episode_step', 'chunk_position', 'robot_state', 'full_image', 'wrist_image',
                'raw_action', 'executed_action', 'features', 'post_robot_state', 'done',
            })

    def test_raw_zero_half_one_are_processed_exactly_once(self):
        for raw_gripper, expected in ((0.0, 1.0), (0.5, 0.0), (1.0, -1.0)):
            with self.subTest(raw_gripper=raw_gripper):
                recorder = self.new_recorder()
                raw, processed = action_pair(raw_gripper)
                recorder.observe(0, 0, prepared_at(0), raw, processed)
                self.assertEqual(recorder.rows[0][2][-1], expected)
                self.assertEqual(recorder.rows[0][0].adapter_action[-1], expected)

    def test_double_processing_and_changed_motion_action_are_rejected(self):
        raw, processed = action_pair(1)
        twice = processed.copy()
        twice[-1] = -np.sign(2 * twice[-1] - 1)
        recorder = self.new_recorder()
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            recorder.observe(0, 0, prepared_at(0), raw, twice)
        wrong_motion = processed.copy()
        wrong_motion[0] += .1
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            recorder.observe(0, 0, prepared_at(0), raw, wrong_motion)
        recorder.observe(0, 0, prepared_at(0), raw, processed)

    def test_skipped_double_and_misaligned_hooks_are_rejected(self):
        recorder = self.new_recorder()
        raw, processed = action_pair()
        recorder.observe(0, 0, prepared_at(0), raw, processed)
        with self.assertRaises(ValueError):
            recorder.executed(0, processed, False, prepared_at(1))
        with self.assertRaises(ValueError):
            recorder.event(1, {}, None, None)
        recorder.event(0, {}, None, None)
        with self.assertRaises(ValueError):
            recorder.event(0, {}, None, None)
        recorder.executed(0, processed, False, prepared_at(1))
        with self.assertRaises(ValueError):
            recorder.executed(0, processed, False, prepared_at(1))

    def test_executed_action_must_match_the_observed_candidate(self):
        recorder = self.new_recorder()
        raw, processed = action_pair()
        recorder.observe(0, 0, prepared_at(0), raw, processed)
        recorder.event(0, {}, None, None)
        different = processed.copy()
        different[-1] *= -1
        with self.assertRaisesRegex(ValueError, 'actual env.step action'):
            recorder.executed(0, different, False, prepared_at(1))
        recorder.executed(0, processed, False, prepared_at(1))

    def test_observe_requires_completed_previous_step_and_matching_post_state(self):
        recorder = self.new_recorder()
        raw, processed = action_pair()
        recorder.observe(0, 0, prepared_at(0), raw, processed)
        with self.assertRaisesRegex(ValueError, 'phase'):
            recorder.observe(1, 1, prepared_at(1), raw, processed)
        recorder.event(0, {}, None, None)
        recorder.executed(0, processed, False, prepared_at(1))
        with self.assertRaisesRegex(ValueError, 'previous executed action'):
            recorder.observe(1, 1, prepared_at(0), raw, processed)
        recorder.observe(1, 1, prepared_at(1), raw, processed)

    def test_episode_cannot_restart_or_finish_with_missing_phases(self):
        recorder = self.new_recorder()
        with self.assertRaises(ValueError):
            recorder.start('second episode')
        raw, processed = action_pair()
        recorder.observe(0, 0, prepared_at(0), raw, processed)
        with self.assertRaises(ValueError):
            recorder.finish(False, 1)
        recorder.event(0, {}, None, None)
        recorder.executed(0, processed, False, prepared_at(1))
        with self.assertRaises(ValueError):
            recorder.finish(True, 1)

    def test_corrupted_feature_replay_is_detected_even_with_updated_file_hash(self):
        record_episode(self.prefix)
        npz = self.prefix.with_suffix('.npz')
        with np.load(npz, allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        arrays['features'][3, 0] += 1
        np.savez_compressed(npz, **arrays)
        path = self.prefix.with_suffix('.labels.json')
        labels = json.loads(path.read_text())
        labels['arrays_sha256'] = hashlib.sha256(npz.read_bytes()).hexdigest()
        path.write_text(json.dumps(labels))
        with self.assertRaisesRegex(AssertionError, 'Feature mismatch at step 3'):
            validate(self.prefix)

    def test_unmatched_event_and_physical_displacement_is_rejected(self):
        record_episode(self.prefix)
        path = self.prefix.with_suffix('.labels.json')
        labels = json.loads(path.read_text())
        labels['events'][2]['fired_transition'] = False
        labels['events'][1]['fired_transition'] = True
        path.write_text(json.dumps(labels))
        with self.assertRaisesRegex(AssertionError, 'physical displacement disagree'):
            validate(self.prefix)

    def test_camera_sampling_is_deterministic_and_never_reorients(self):
        rgb = np.zeros((128, 128, 3), dtype=np.uint8)
        rgb[:64, :64, 0] = 255
        downsampled = camera64(rgb)
        self.assertEqual(downsampled.shape, (64, 64, 3))
        self.assertEqual(downsampled.dtype, np.uint8)
        self.assertEqual(downsampled[0, 0, 0], 255)
        self.assertEqual(downsampled[-1, -1, 0], 0)
        np.testing.assert_array_equal(camera64(downsampled), downsampled)


if __name__ == '__main__':
    unittest.main()
