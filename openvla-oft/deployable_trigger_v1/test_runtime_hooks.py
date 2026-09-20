"""CPU contract tests for the actual runtime, without loading policy/GL modules.

Only selected definitions are compiled from run_asyncmixvla.py; their simulator
and HTTP dependencies are replaced with deterministic fakes. These tests check
the recording/decision boundary, not model quality or MuJoCo determinism.
"""
import ast
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch


RUNTIME = Path(__file__).resolve().parents[1] / "run_asyncmixvla.py"


class Array(list):
    def copy(self):
        return Array(self)

    def tolist(self):
        return list(self)


class FakeEnv:
    def __init__(self, terminal_after=None):
        self.n = 0
        self.total_calls = 0
        self.actions = []
        self.body = Array([0.0, 0.0, 0.0])
        self.closed = False
        self.terminal_after = terminal_after

    def reset(self):
        self.n = 0
        self.actions = []
        self.body = Array([0.0, 0.0, 0.0])

    def observation(self):
        # The full raw dictionary intentionally contains a privileged field.
        return {"n": self.n, "object_ground_truth": object()}

    def set_init_state(self, state):
        return self.observation()

    def step(self, action):
        self.actions.append(list(action))
        self.total_calls += 1
        self.n += 1
        done = self.terminal_after is not None and self.n >= 10 + self.terminal_after
        return self.observation(), 0, done, {}

    def close(self):
        self.closed = True


class RecordingSpy:
    def __init__(self, env):
        self.env = env
        self.rows = []
        self.finished = None

    def start(self, task_description):
        self.started_at = (self.env.n, self.env.total_calls)
        self.task_description = task_description

    def observe(self, t, chunk_position, observation, raw_action, action):
        self.rows.append({"t": t, "chunk_position": chunk_position,
                          "observation": observation, "raw": raw_action,
                          "action": action, "phases": ["observe"]})

    def event(self, t, diagnostics, before, after):
        row = self.rows[-1]
        assert row["t"] == t
        row.update(diagnostics=diagnostics, hook_before=before, hook_after=after)
        row["phases"].append("event")

    def executed(self, t, action, done, observation_after):
        row = self.rows[-1]
        assert row["t"] == t
        row.update(executed_action=action, done=done, observation_after=observation_after)
        row["phases"].append("executed")

    def finish(self, success, episode_step):
        self.finished = (success, episode_step)


class RuntimeHarness:
    def __init__(self, terminal_after=None):
        self.env = FakeEnv(terminal_after)
        self.process_calls = []
        self.policy_actions = []
        self.trigger_calls = []
        self.creation_calls = 0
        self.trigger = types.SimpleNamespace(inner=object(), update=self.trigger_update)
        self.args = types.SimpleNamespace(
            task_id=0, trial_id=0, condition="perturbed", manifest_path="fake.json",
            mode="adapter_only", trigger="never", max_steps=16, warmup=3,
            debug_trace_out=None, state_alignment="stale", vision_alignment="stale",
            observable_recorder=None,
        )
        self.namespace = {
            "time": time, "get_env_and_task": self.get_env_and_task,
            "NUM_STEPS_WAIT": 10, "CHUNK_SIZE": 8,
            "get_libero_dummy_action": lambda _: [0] * 7, "warm_up": self.warm_up,
            "setup_perturbation": self.setup_perturbation,
            "build_trigger": lambda *a, **kw: self.trigger,
            "MECHANISM_FOR_MODE": {"naive_async_full_oft": "naive_async"},
            "prepare_observation": self.prepare_observation,
            "call_policy": self.call_policy, "process_action": self.process_action,
        }
        tree = ast.parse(RUNTIME.read_text())
        definitions = [node for node in tree.body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                       and node.name in ("run_episode", "_EpisodeTimer")]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(RUNTIME), "exec"), self.namespace)

    def get_env_and_task(self, task_id):
        self.creation_calls += 1
        return self.env, "test task instruction", [0]

    @staticmethod
    def prepare_observation(obs):
        return {"state": Array([obs["n"]]), "full_image": Array([1]), "wrist_image": Array([2])}

    def warm_up(self, env, description, initial_state, count):
        env.reset()
        env.set_init_state(initial_state)
        for _ in range(10):
            env.step([0] * 7)

    def call_policy(self, model, observation, description):
        actions = [Array([0] * 6 + [1]) for _ in range(8)]
        self.policy_actions.extend(actions)
        return actions, 0

    def process_action(self, action):
        self.process_calls.append(list(action))
        copied = action.copy()
        copied[-1] *= -1
        return copied

    def setup_perturbation(self, *args):
        diagnostics = {"body_name": "test_body", "fired": False}

        def hook(t):
            if t == 1:
                self.env.body[0] += 0.01
                diagnostics["fired"] = True

        return hook, diagnostics, {}, {"trigger_step": 1}

    def trigger_update(self, **kwargs):
        self.trigger_calls.append(kwargs)
        return types.SimpleNamespace(should_switch=False)

    def run(self):
        mixed = types.ModuleType("run_mixed_libero10")
        mixed.get_body_position = lambda env, name: env.body.copy()
        with patch.dict(sys.modules, {"run_mixed_libero10": mixed}):
            return self.namespace["run_episode"](self.args)


class RuntimeRecordingTests(unittest.TestCase):
    def test_recording_matches_uninstrumented_trajectory(self):
        baseline = RuntimeHarness()
        baseline_trace = baseline.run()
        recorded = RuntimeHarness()
        recorder = recorded.args.observable_recorder = RecordingSpy(recorded.env)
        trace = recorded.run()
        self.assertEqual(recorded.env.actions, baseline.env.actions)
        self.assertEqual(trace["final_step"], baseline_trace["final_step"])
        self.assertEqual(recorder.started_at, (10, 20))
        self.assertEqual([row["t"] for row in recorder.rows], list(range(16)))
        self.assertEqual(len(recorded.process_calls), 16)
        self.assertTrue(all(a[-1] == 1 for a in recorded.policy_actions))
        for t, row in enumerate(recorder.rows):
            self.assertEqual(row["phases"], ["observe", "event", "executed"])
            self.assertEqual(row["observation"]["state"], [10 + t])
            self.assertEqual(row["observation_after"]["state"], [11 + t])
            self.assertEqual(row["raw"][-1], 1)
            self.assertEqual(row["action"][-1], -1)
            self.assertEqual(row["action"], row["executed_action"])
            self.assertEqual(row["action"], recorded.env.actions[10 + t])
        # Event labels isolate the hook edit; they retain their original value
        # when the live diagnostics dict changes on the following step.
        self.assertFalse(recorder.rows[0]["diagnostics"]["fired"])
        self.assertTrue(recorder.rows[1]["diagnostics"]["fired"])
        self.assertEqual(recorder.rows[1]["hook_before"], [0.0, 0.0, 0.0])
        self.assertEqual(recorder.rows[1]["hook_after"], [0.01, 0.0, 0.0])
        self.assertEqual(recorder.finished, (False, 16))
        self.assertTrue(recorded.env.closed)

    def test_exact_recording_horizon_preserves_legacy_default(self):
        recorded = RuntimeHarness()
        recorded.args.max_steps = 3
        recorded.args.observable_recorder = RecordingSpy(recorded.env)
        self.assertEqual(recorded.run()["final_step"], 3)
        self.assertEqual(len(recorded.process_calls), 3)
        legacy = RuntimeHarness()
        legacy.args.max_steps = 3
        self.assertEqual(legacy.run()["final_step"], 8)
        self.assertEqual(recorded.env.actions, legacy.env.actions[:13])

    def test_terminal_and_clean_recording(self):
        harness = RuntimeHarness(terminal_after=3)
        harness.args.condition = "clean"
        harness.args.warmup = 0
        recorder = harness.args.observable_recorder = RecordingSpy(harness.env)
        trace = harness.run()
        self.assertEqual(recorder.started_at, (10, 10))
        self.assertTrue(trace["final_success"])
        self.assertEqual(recorder.finished, (True, 3))
        self.assertTrue(recorder.rows[-1]["done"])
        self.assertTrue(all(row["diagnostics"] is None and row["hook_before"] is None
                            and row["hook_after"] is None for row in recorder.rows))

    def test_recorder_rejects_switching_before_environment_creation(self):
        for mode, trigger in [("naive_async_full_oft", "never"), ("adapter_only", "deployable")]:
            with self.subTest(mode=mode, trigger=trigger):
                harness = RuntimeHarness()
                harness.args.mode, harness.args.trigger = mode, trigger
                harness.args.observable_recorder = RecordingSpy(harness.env)
                with self.assertRaises(ValueError):
                    harness.run()
                self.assertEqual(harness.creation_calls, 0)

    def test_deployable_runtime_update_receives_only_allowed_inputs(self):
        harness = RuntimeHarness()
        harness.args.mode, harness.args.trigger = "naive_async_full_oft", "deployable"
        harness.run()
        self.assertEqual(len(harness.trigger_calls), 16)
        for call in harness.trigger_calls:
            self.assertEqual(set(call), {"observation", "adapter_action", "chunk_position", "episode_step"})
            self.assertEqual(set(call["observation"]), {"full_image", "wrist_image", "state"})

    def test_deployable_constructor_never_receives_privileged_arguments(self):
        tree = ast.parse(RUNTIME.read_text())
        definitions = [node for node in tree.body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                       and node.name in ("build_trigger", "FrozenTrigger")]
        namespace = {"TriggerResult": object}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(RUNTIME), "exec"), namespace)
        received = []
        trigger_module = types.ModuleType("deployable_trigger_v1.trigger")
        trigger_module.DeployableTrigger = lambda *args, **kwargs: received.append((args, kwargs))
        with patch.dict(sys.modules, {"deployable_trigger_v1.trigger": trigger_module}):
            namespace["build_trigger"](
                types.SimpleNamespace(trigger="deployable", deployable_checkpoint="checkpoint.json"),
                env=object(), manifest_event=object(),
            )
            self.assertEqual(received, [(("checkpoint.json",), {})])
            with self.assertRaisesRegex(ValueError, "requires --deployable_checkpoint"):
                namespace["build_trigger"](types.SimpleNamespace(trigger="deployable"))


if __name__ == "__main__":
    unittest.main()
