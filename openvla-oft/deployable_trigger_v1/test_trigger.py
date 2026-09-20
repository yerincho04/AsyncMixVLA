"""Observable-only scoring and strict checkpoint contract tests without a GPU."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .features import ObservableFeatures
from .test_recording import action_pair, prepared_at, record_episode
from .trigger import DeployableTrigger


def checkpoint_artifact():
    names = list(ObservableFeatures.feature_names)
    indices = [names.index(name) for name in ('action.gripper', 'action.dx', 'front.lag1.residual_grid.r0c0')]
    return dict(
        input_contract='observable_v1', feature_names=names, feature_indices=indices,
        mean=[0, 0, 0], scale=[1, 1, 1], coef=[2, .1, .2],
        intercept=0, threshold=.8, persistence=2,
    )


class DeployableTriggerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkpoint = Path(self.temp.name) / 'checkpoint.json'
        self.checkpoint.write_text(json.dumps(checkpoint_artifact()))

    def replay_npz(self, npz):
        trigger = DeployableTrigger(self.checkpoint)
        output = []
        with np.load(npz, allow_pickle=False) as arrays:
            for step in range(len(arrays['episode_step'])):
                result = trigger.update(
                    observation=dict(full_image=arrays['full_image'][step], wrist_image=arrays['wrist_image'][step], state=arrays['robot_state'][step]),
                    adapter_action=arrays['executed_action'][step], episode_step=step,
                    chunk_position=int(arrays['chunk_position'][step]))
                output.append((result.trigger_score, result.should_switch))
                self.assertFalse(result.trigger_metadata['privileged_inputs'])
        return trigger, output

    def test_offline_features_and_online_scores_match_with_episode_persistence(self):
        prefix = Path(self.temp.name) / 'episode'
        record_episode(prefix)
        online, output = self.replay_npz(prefix.with_suffix('.npz'))
        offline = DeployableTrigger(self.checkpoint)
        with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as arrays:
            expected_scores = [offline.score_features(features) for features in arrays['features']]
        self.assertEqual([item[0] for item in output], expected_scores)
        self.assertEqual([item[1] for item in output], [False, False, True, False, False, False, False])
        self.assertTrue(online.fired)
        self.assertEqual([item['episode_step'] for item in online.trace], list(range(7)))
        # Persistence resets after a low score, and a detector never rearms.
        consecutive, fired = 0, False
        for score, decision in output:
            consecutive = consecutive + 1 if score >= .8 else 0
            expected = not fired and consecutive >= 2
            self.assertEqual(decision, expected)
            fired = fired or expected

    def test_privileged_label_changes_cannot_change_trigger_sequence(self):
        prefix = Path(self.temp.name) / 'episode'
        record_episode(prefix)
        _, original = self.replay_npz(prefix.with_suffix('.npz'))
        labels_path = prefix.with_suffix('.labels.json')
        labels = json.loads(labels_path.read_text())
        labels['episode'].update(task_id=999, adapter_success=False, oracle_recoverable=True,
                                 object_pose=[100, -100, 40], manifest_direction=[-1, 1], onset=-1000)
        labels['task_description'] = 'Privileged label deliberately changed'
        labels['final_success'] = False
        for event in labels['events']:
            event['diagnostics'] = {'future_success': True, 'source_body_xpos': [-99, 99, 99]}
            event['hook_displacement'] = [10, 0, 0]
        labels_path.write_text(json.dumps(labels))
        _, changed = self.replay_npz(prefix.with_suffix('.npz'))
        self.assertEqual(original, changed)
        # Even removing the label file leaves inference operational and identical.
        labels_path.unlink()
        _, label_free = self.replay_npz(prefix.with_suffix('.npz'))
        self.assertEqual(original, label_free)

    def test_observation_dictionary_is_a_strict_whitelist(self):
        trigger = DeployableTrigger(self.checkpoint)
        _, action = action_pair()
        for private_key in ('task_id', 'body_xpos', 'object_pose', 'delta_xy', 'perturbation_step', 'final_success'):
            observation = prepared_at(0)
            observation[private_key] = 0
            with self.subTest(private_key=private_key):
                with self.assertRaisesRegex(ValueError, 'sanitized'):
                    trigger.update(observation=observation, adapter_action=action, episode_step=0, chunk_position=0)
        with self.assertRaises(ValueError):
            trigger.update(observation={'state': np.zeros(8)}, adapter_action=action, episode_step=0, chunk_position=0)
        trigger.update(observation=prepared_at(0), adapter_action=action, episode_step=0, chunk_position=0)

    def test_privileged_arguments_cannot_be_forwarded_as_extra_keywords(self):
        trigger = DeployableTrigger(self.checkpoint)
        _, action = action_pair()
        with self.assertRaises(TypeError):
            trigger.update(observation=prepared_at(0), adapter_action=action, episode_step=0, chunk_position=0, task_id=2)

    def test_checkpoint_requires_explicit_input_contract_and_exact_schema(self):
        for change in ({'input_contract': 'privileged'}, {'feature_names': []}, {'feature_names': list(reversed(ObservableFeatures.feature_names))}):
            artifact = checkpoint_artifact()
            artifact.update(change)
            self.checkpoint.write_text(json.dumps(artifact))
            with self.assertRaises(ValueError):
                DeployableTrigger(self.checkpoint)
        with self.assertRaises(ValueError):
            DeployableTrigger(None)

    def test_invalid_checkpoint_values_are_rejected_at_load(self):
        invalid = [
            {'scale': [1, 0, 1]}, {'scale': [1, float('nan'), 1]},
            {'coef': [1, 1]}, {'coef': [1, float('inf'), 1]},
            {'threshold': 0}, {'threshold': 1.1}, {'threshold': float('nan')},
            {'persistence': 0}, {'persistence': 2.5}, {'persistence': True},
            {'intercept': float('nan')}, {'intercept': float('inf')},
            {'feature_indices': [-1, 0, 1]}, {'feature_indices': [0, 1, 235]},
            {'feature_indices': [.5, 1, 2]},
            {'mean': [[0], [0], [0]]}, {'scale': [[1], [1], [1]]},
            {'coef': [[1], [1], [1]]},
        ]
        for changes in invalid:
            with self.subTest(changes=changes):
                artifact = copy.deepcopy(checkpoint_artifact())
                artifact.update(changes)
                self.checkpoint.write_text(json.dumps(artifact))
                with self.assertRaises(ValueError):
                    DeployableTrigger(self.checkpoint)


if __name__ == '__main__':
    unittest.main()
