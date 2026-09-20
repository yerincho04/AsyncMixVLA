"""CPU contract tests for observable-only causal feature extraction."""

from dataclasses import fields, replace
import unittest

import numpy as np

try:
    from .features import FEATURE_NAMES, ObservableFeatures, ObservableInput
except ImportError:
    from features import FEATURE_NAMES, ObservableFeatures, ObservableInput


def record(step=0, **updates):
    base = ObservableInput(
        full_image=np.zeros((64, 64, 3), dtype=np.uint8),
        wrist_image=np.zeros((64, 64, 3), dtype=np.uint8),
        robot_state=np.zeros(8, dtype=np.float32),
        adapter_action=np.zeros(7, dtype=np.float32),
        episode_step=step,
        chunk_position=step % 8,
    )
    return replace(base, **updates)


def named(vector, name):
    return vector[FEATURE_NAMES.index(name)]


class ObservableFeaturesTests(unittest.TestCase):
    def test_explicit_inputs_and_finite_fixed_schema(self):
        self.assertEqual(
            {field.name for field in fields(ObservableInput)},
            {"full_image", "wrist_image", "robot_state", "adapter_action", "episode_step", "chunk_position"},
        )
        extractor = ObservableFeatures()
        output = extractor.update(record())
        self.assertEqual(len(set(FEATURE_NAMES)), len(FEATURE_NAMES))
        self.assertEqual(output.shape, (extractor.num_features,))
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(np.isfinite(output).all())
        self.assertNotIn("episode_step", FEATURE_NAMES)
        self.assertEqual(named(output, "has_previous"), 0)
        self.assertEqual(named(output, "camera.lag1.available"), 0)

    def test_alignment_errors_do_not_advance_history(self):
        extractor = ObservableFeatures()
        for bad in (record(-1), record(1), record(episode_step=True), record(episode_step=0.0), record(chunk_position=-1), record(chunk_position=8), record(chunk_position=True)):
            with self.subTest(bad=bad.episode_step, position=bad.chunk_position):
                with self.assertRaises(ValueError):
                    extractor.update(bad)
        extractor.update(record())
        with self.assertRaises(ValueError):
            extractor.update(record())
        with self.assertRaises(ValueError):
            extractor.update(record(2))
        extractor.update(record(1))

    def test_shapes_nan_and_pixel_scale_are_rejected(self):
        invalid = [
            record(robot_state=np.zeros(7)), record(robot_state=np.full(8, np.nan)),
            record(adapter_action=np.zeros((1, 7))), record(adapter_action=np.full(7, np.inf)),
            record(full_image=np.zeros((64, 64), dtype=np.uint8)),
            record(full_image=np.zeros((63, 64, 3), dtype=np.uint8)),
            record(full_image=np.full((64, 64, 3), np.nan)),
            record(wrist_image=np.zeros((64, 64, 3), dtype=np.float32)),
        ]
        for bad in invalid:
            extractor = ObservableFeatures()
            with self.assertRaises(ValueError):
                extractor.update(bad)
            extractor.update(record())

    def test_reset_removes_all_previous_episode_information(self):
        extractor = ObservableFeatures()
        extractor.update(record(adapter_action=np.ones(7), full_image=np.full((64, 64, 3), 255, dtype=np.uint8)))
        extractor.update(record(1, robot_state=np.ones(8)))
        extractor.reset()
        np.testing.assert_array_equal(extractor.update(record()), ObservableFeatures().update(record()))

    def test_future_observations_cannot_change_prefix(self):
        rng = np.random.default_rng(12)
        observations = [record(t, robot_state=rng.normal(0, 0.01, 8), adapter_action=rng.uniform(-1, 1, 7), full_image=rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)) for t in range(24)]
        extractor = ObservableFeatures()
        original = [extractor.update(item) for item in observations]
        alternate = ObservableFeatures()
        alternate_outputs = [alternate.update(item) for item in observations[:17]]
        for t in range(17, 24):
            alternate_outputs.append(alternate.update(record(t, robot_state=np.ones(8), adapter_action=-np.ones(7))))
        np.testing.assert_array_equal(np.stack(original[:17]), np.stack(alternate_outputs[:17]))
        self.assertFalse(np.array_equal(original[-1], alternate_outputs[-1]))

    def test_processed_gripper_sign_is_preserved_exactly(self):
        extractor = ObservableFeatures()
        action = np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)
        before = action.copy()
        first = extractor.update(record(adapter_action=action))
        self.assertEqual(named(first, "action.gripper"), -1)
        self.assertEqual(named(first, "history5.action_mean.gripper"), -1)
        np.testing.assert_array_equal(action, before)
        second = extractor.update(record(1, adapter_action=-action))
        self.assertEqual(named(second, "previous_action.gripper"), -1)
        self.assertEqual(named(second, "action.gripper"), 1)
        self.assertEqual(named(second, "action_delta.gripper"), 2)
        self.assertEqual(named(second, "history5.gripper_flip_fraction"), 1)

    def test_motion_is_compared_to_previous_action(self):
        extractor = ObservableFeatures()
        extractor.update(record(adapter_action=np.array([1, 0, 0, 0, 0, 0, -1])))
        state = np.array([0.01, 0, 0, 0, 0, 0, 0.02, -0.02])
        output = extractor.update(record(1, robot_state=state, adapter_action=np.array([-1, 0, 0, 0, 0, 0, -1])))
        self.assertEqual(named(output, "translation_command_alignment"), 1)
        self.assertAlmostEqual(named(output, "translation_motion_per_command"), 0.01)
        self.assertGreater(named(output, "gripper_motion_norm"), 0)

    def test_rotation_delta_handles_axis_angle_branch(self):
        extractor = ObservableFeatures()
        initial = np.zeros(8)
        initial[5] = np.pi - 0.01
        extractor.update(record(robot_state=initial))
        current = initial.copy()
        current[5] = -np.pi + 0.01
        output = extractor.update(record(1, robot_state=current))
        self.assertAlmostEqual(named(output, "rotation_motion_angle"), 0.02, places=6)
        self.assertAlmostEqual(named(output, "rotation_delta.z"), 0.02, places=6)

    def test_local_camera_motion_survives_illumination_correction(self):
        extractor = ObservableFeatures()
        extractor.update(record())
        front = np.full((64, 64, 3), 30, dtype=np.uint8)
        front[:16, :16] = 180
        output = extractor.update(record(1, full_image=front))
        self.assertGreater(named(output, "front.lag1.residual_grid.r0c0"), 0.5)
        self.assertEqual(named(output, "front.lag1.residual_grid.r3c3"), 0)
        self.assertEqual(named(output, "wrist.lag1.residual_mean"), 0)
        self.assertAlmostEqual(named(output, "front.lag1.median_shift.r"), 30 / 255)

    def test_uniform_illumination_is_logged_but_has_zero_local_residual(self):
        extractor = ObservableFeatures()
        extractor.update(record())
        output = extractor.update(record(1, full_image=np.full((64, 64, 3), 40, dtype=np.uint8)))
        self.assertGreater(named(output, "front.lag1.rgb_abs_mean"), 0)
        self.assertEqual(named(output, "front.lag1.residual_mean"), 0)

    def test_temporal_lags_have_explicit_availability(self):
        extractor = ObservableFeatures()
        outputs = [extractor.update(record(t, wrist_image=np.full((64, 64, 3), t, dtype=np.uint8))) for t in range(18)]
        for lag in (1, 4, 16):
            self.assertEqual(named(outputs[lag - 1], f"camera.lag{lag}.available"), 0)
            self.assertEqual(named(outputs[lag], f"camera.lag{lag}.available"), 1)
            self.assertAlmostEqual(named(outputs[17], f"wrist.lag{lag}.rgb_abs_mean"), lag / 255)

    def test_external_input_mutation_cannot_change_recorded_history(self):
        extractor = ObservableFeatures()
        initial = record(adapter_action=np.ones(7))
        extractor.update(initial)
        initial.robot_state[:] = 999
        initial.adapter_action[:] = -999
        initial.full_image[:] = 255
        output = extractor.update(record(1))
        self.assertEqual(named(output, "previous_action.dx"), 1)
        self.assertEqual(named(output, "eef_motion_norm"), 0)
        self.assertEqual(named(output, "front.lag1.rgb_abs_mean"), 0)


if __name__ == "__main__":
    unittest.main()
