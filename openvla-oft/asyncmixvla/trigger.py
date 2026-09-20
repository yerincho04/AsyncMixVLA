"""Modular trigger interface for the live AsyncMixVLA runtime.

The async pipeline (asyncmixvla/bridge.py, run_asyncmixvla.py) calls
trigger.update(...) once per Adapter action and only ever reads the returned
TriggerResult -- it never knows or cares which backend produced the
decision. Swapping an oracle trigger for the observable V1+V2 trigger is a configuration
change, never a code change to
the state machine or bridge/handoff logic.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TriggerResult:
    should_switch: bool
    trigger_score: Optional[float] = None
    trigger_metadata: Dict[str, Any] = field(default_factory=dict)


class Trigger:
    """Base interface. A trigger is stateful across one episode (it may
    accumulate evidence step to step) but must never be called again after
    it has returned should_switch=True -- the caller (run_asyncmixvla.py)
    enforces the "exactly one Adapter->OFT escalation, no oscillation" rule
    by freezing the trigger itself once fired, but backends should also
    treat should_switch as a one-way decision internally (never re-arm)."""

    def update(self, *, observation, robot_state, adapter_action, chunk_position,
               remaining_actions, task_id, episode_step) -> TriggerResult:
        raise NotImplementedError


class NeverTrigger(Trigger):
    """adapter_only mode: never evaluates a real decision, always says no.
    Used when the runtime should behave as a pure Adapter baseline with no
    trigger machinery involved at all (Smoke A uses this or an
    OracleOnsetTrigger on a non-RESCUE episode -- both are valid ways to
    produce "trigger never fires")."""

    def update(self, **kwargs) -> TriggerResult:
        return TriggerResult(should_switch=False, trigger_metadata={"backend": "never"})


class OracleOnsetTrigger(Trigger):
    """DEBUG-ONLY backend, for milestone 5 (reproducing Table 2B's validated
    behavior inside the new full-episode live runtime). Reads the SAME
    frozen oracle_manifest.json Table 2B uses (read-only -- this file is
    never modified here) and fires exactly once, at episode_step ==
    T_predict_oracle, for episodes the oracle manifest marks as RESCUE.
    Never fires on NO_SWITCH/UNRECOVERABLE/invalid episodes.

    Unlike Table 2B's oracle_trigger_step bypass (which skips straight to
    the bridge and never runs the episode from step 0), this trigger is
    evaluated at every real step like any other backend -- the runtime
    still executes the full episode from initialization; this backend just
    happens to have hindsight knowledge of *when* to say yes, exactly the
    role Table 2B's oracle timing already played, now expressed through the
    same interface a learned trigger will use."""

    def __init__(self, oracle_manifest_path, task_id, trial_id, condition):
        with open(oracle_manifest_path) as f:
            rows = json.load(f)
        self.row = next(
            (r for r in rows if r["task_id"] == task_id and r["trial_id"] == trial_id
             and r["condition"] == condition),
            None,
        )
        self.will_ever_fire = bool(
            self.row and self.row.get("valid") and self.row.get("oracle_category") == "RESCUE"
        )
        self.t_predict_oracle = self.row["T_predict_oracle"] if self.will_ever_fire else None
        self._fired = False

    def update(self, *, episode_step, **kwargs) -> TriggerResult:
        if self._fired or not self.will_ever_fire:
            return TriggerResult(should_switch=False, trigger_metadata={"backend": "oracle_onset"})
        if episode_step == self.t_predict_oracle:
            self._fired = True
            return TriggerResult(
                should_switch=True, trigger_score=1.0,
                trigger_metadata={"backend": "oracle_onset", "t_predict_oracle": self.t_predict_oracle,
                                   "oracle_category": self.row["oracle_category"]},
            )
        return TriggerResult(should_switch=False, trigger_metadata={"backend": "oracle_onset"})


class ForcedStepTrigger(Trigger):
    """Benchmark backend: fires exactly once, the first time
    episode_step >= forced_step. Needs no manifest and works for any
    (task, trial) -- used for the handoff-only live DEV benchmark, where
    every method under comparison must hand off at the SAME externally-fixed
    step so the handoff itself (not the trigger) is what's being compared.
    Does not touch the learned/oracle_onset trigger paths at all."""

    def __init__(self, forced_step):
        self.forced_step = int(forced_step)
        self._fired = False

    def update(self, *, episode_step, **kwargs) -> TriggerResult:
        if self._fired:
            return TriggerResult(should_switch=False, trigger_metadata={"backend": "forced_step"})
        if episode_step >= self.forced_step:
            self._fired = True
            return TriggerResult(
                should_switch=True, trigger_score=1.0,
                trigger_metadata={"backend": "forced_step", "forced_step": self.forced_step,
                                   "fired_at_step": episode_step},
            )
        return TriggerResult(should_switch=False, trigger_metadata={"backend": "forced_step"})
