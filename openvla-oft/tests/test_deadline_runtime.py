import numpy as np
import pytest

from asyncmixvla.deadline_runtime import DeadlineAwareEnv, hold_action, missed_control_ticks


@pytest.mark.parametrize(
    "wait,dt,expected",
    [(0.0, 0.05, 0), (0.05, 0.05, 0), (0.051, 0.05, 1), (0.10, 0.05, 1), (0.101, 0.05, 2)],
)
def test_missed_control_ticks(wait, dt, expected):
    assert missed_control_ticks(wait, dt) == expected


def test_zero_arm_hold_preserves_gripper_without_mutating_input():
    source = np.arange(7, dtype=np.float64)
    result = hold_action(source, "zero_arm_hold_gripper")
    assert np.array_equal(result, [0, 0, 0, 0, 0, 0, 6])
    assert np.array_equal(source, np.arange(7, dtype=np.float64))


def test_repeat_last():
    source = np.linspace(-1, 1, 7)
    assert np.array_equal(hold_action(source, "repeat_last"), source)


def test_invalid_hold_mode():
    with pytest.raises(ValueError):
        hold_action(np.zeros(7), "invented")


class _TinyEnv:
    def __init__(self):
        self.steps = 1
        self.horizon = 2
        self.done = False
        self.obs = {"state": "current"}
        self.oft_starts_at = 1

    def step(self, action):
        if self.steps >= self.horizon:
            raise RuntimeError("step beyond horizon")
        self.steps += 1
        return self.obs, 0.0, False, {}


def test_deadline_holds_never_step_past_horizon(monkeypatch):
    env = DeadlineAwareEnv(_TinyEnv(), latency_override_s=1.0)
    env._last_step_end = 0.0
    env._last_action = np.zeros(7)
    monkeypatch.setattr("asyncmixvla.deadline_runtime.get_control_dt", lambda _: 0.05)
    obs, reward, done, info = env.step(np.ones(7))
    assert env.steps == env.horizon == 2
    assert not done
    assert info["deadline_horizon_exhausted"]
    assert not info["oft_action_executed"]
    assert env.deadline_events[0]["horizon_exhausted_during_hold"]
