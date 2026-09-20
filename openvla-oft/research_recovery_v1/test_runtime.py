"""CPU-only checks of the step boundary, without importing GPU model stacks."""
import ast
import hashlib
from pathlib import Path
import unittest

import numpy as np

source = Path(__file__).with_name('runtime.py')
tree = ast.parse(source.read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EpisodeEnv')
namespace = {'np': np, 'hashlib': hashlib}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
EpisodeEnv = namespace['EpisodeEnv']


class FakeEnv:
    def __init__(self, success_at=None):
        self.n = 0
        self.success_at = success_at

    def step(self, action):
        self.n += 1
        return {'n': self.n}, 0, self.n == self.success_at, {}


class StepBoundaryTests(unittest.TestCase):
    def test_hook_receives_latest_observation_through_handoff_and_return(self):
        calls, holder = [], {}
        env = EpisodeEnv(FakeEnv(), {'n': 0}, 6,
                         lambda s: calls.append((s,holder['obs']['n'])), holder)
        env.oft_starts_at = 2
        for _ in range(4):
            env.step([0]*7)
        env.oft_starts_at = None
        env.policy = 'adapter'
        env.step([1]*7)
        self.assertEqual(calls, [(i,i) for i in range(5)])
        self.assertEqual(env.counts, {'adapter':3,'oft':2})
        self.assertEqual(env.steps, sum(env.counts.values()))

    def test_horizon_has_no_extra_perturbation_or_step(self):
        calls = []
        env = EpisodeEnv(FakeEnv(), {}, 2, calls.append, {})
        env.step([0]*7)
        env.step([0]*7)
        with self.assertRaises(RuntimeError):
            env.step([0]*7)
        self.assertEqual(calls,[0,1])
        self.assertEqual(env.base.n,2)

    def test_success_stops_further_simulation(self):
        env = EpisodeEnv(FakeEnv(success_at=1), {}, 520)
        env.step([0]*7)
        with self.assertRaises(RuntimeError):
            env.step([0]*7)
        self.assertEqual(env.steps,1)


if __name__ == '__main__':
    unittest.main()
