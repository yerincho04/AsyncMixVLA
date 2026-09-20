"""Matched recovery experiment. Frozen runtime and artifacts remain untouched.

All methods use 520 actual policy steps, identical live disturbance application,
and freshly run standalone controls. OFT can take over permanently or serve a
bounded recovery burst before Adapter resumes (no second escalation).
"""
import hashlib
import time
from types import SimpleNamespace

import numpy as np

from asyncmixvla.bridge import run_async_bridge, run_sync_handoff
from experiments.robot.libero.libero_utils import get_libero_dummy_action
from experiments.robot.robot_utils import set_seed_everywhere
from measure_async_handoff_latency import warm_up
from run_asyncmixvla import build_trigger, setup_perturbation
from run_test0_switch_timing import call_policy, get_env_and_task, prepare_observation, process_action


class EpisodeEnv:
    """Account for every real step, including the bridge's first OFT action."""

    def __init__(self, env, obs, horizon, hook=None, holder=None):
        self.base, self.obs, self.horizon = env, obs, horizon
        self.hook, self.holder = hook, holder
        self.steps = 0
        self.done = False
        self.policy = 'adapter'
        self.oft_starts_at = None
        self.counts = {'adapter': 0, 'oft': 0}
        self.actions = []
        self.digest = hashlib.sha256()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def step(self, action):
        if self.steps >= self.horizon or self.done:
            raise RuntimeError('step beyond horizon or terminal success')
        if self.hook is not None:
            self.holder['obs'] = self.obs
            self.hook(self.steps)
        policy = self.policy
        if self.oft_starts_at is not None and self.steps >= self.oft_starts_at:
            policy = 'oft'
        a = np.asarray(action, dtype=np.float64)
        self.digest.update(a.tobytes())
        self.actions.append({'step': self.steps, 'policy': policy, 'action': a.tolist()})
        self.obs, reward, self.done, info = self.base.step(action)
        self.counts[policy] += 1
        self.steps += 1
        return self.obs, reward, self.done, info


def run(point, config, trigger_backend='forced_step', oracle_switch=True, horizon=520):
    set_seed_everywhere(7)
    base, language, initial = get_env_and_task(task_id=point['task_id'])
    try:
        warm_up(base, language, initial[point['trial_index']], 3)
        base.reset()
        obs = base.set_init_state(initial[point['trial_index']])
        for _ in range(10):
            obs, _, _, _ = base.step(get_libero_dummy_action('openvla'))
        hook, diag, holder, event = None, None, None, None
        if point['condition'] == 'perturbed':
            hook, diag, holder, event = setup_perturbation(
                point['task_id'], point['trial_index'], point['condition'],
                point['perturbation_manifest'], base, obs)
        env = EpisodeEnv(base, obs, horizon, hook, holder)
        backend = trigger_backend
        if config['mechanism'] in ('adapter', 'oft') or not oracle_switch:
            backend = 'never'
        args = SimpleNamespace(trigger=backend, forced_trigger_step=point['episode_step'],
                               task_id=point['task_id'])
        trigger = build_trigger(args, env=base, manifest_event=event)
        origin = time.perf_counter()
        fired, switch_step, handoff, return_step = False, None, None, None
        active = 'oft' if config['mechanism'] == 'oft' else 'adapter'
        recovery_end = None
        prefix_hash = None
        while not env.done and env.steps < horizon:
            env.policy = active
            env.oft_starts_at = None
            chunk, _ = call_policy(active, prepare_observation(env.obs), language)
            chunk = chunk[:8]
            if len(chunk) != 8:
                raise ValueError(f'expected 8 actions, got {len(chunk)}')
            if active == 'oft':
                execution_steps = config.get('oft_execution_steps', 8)
                assert 1 <= execution_steps <= 8
                chunk = chunk[:execution_steps]
            for pos, raw in enumerate(chunk):
                if env.done or env.steps >= horizon:
                    break
                if active == 'oft' and recovery_end is not None and env.steps >= recovery_end:
                    active, return_step = 'adapter', env.steps
                    break
                decision = None
                if active == 'adapter' and not fired and backend != 'never':
                    decision = trigger.update(
                        env=base, observation=env.obs, robot_state=prepare_observation(env.obs)['state'],
                        adapter_action=process_action(raw).tolist(), chunk_position=pos,
                        remaining_actions=8-pos, task_id=point['task_id'], episode_step=env.steps)
                if decision is not None and decision.should_switch and env.steps < horizon-1:
                    fired, switch_step = True, env.steps
                    prefix_hash = env.digest.hexdigest()
                    bridge = chunk[pos:]
                    cap = config.get('bridge_cap', 8)
                    bridge = bridge[:min(cap, horizon-env.steps-1)]
                    env.oft_starts_at = env.steps + len(bridge)
                    if config['mechanism'] == 'sync':
                        handoff = run_sync_handoff(env, language, env.obs, bridge, origin,
                                                   action_number_offset=env.steps)
                    else:
                        handoff = run_async_bridge(
                            env, language, env.obs, bridge, origin,
                            mechanism='vlash_async' if config['alignment'] == 'ac' else 'naive_async',
                            action_number_offset=env.steps,
                            state_alignment='vlash_gain_corrected' if config['alignment'] == 'ac' else 'stale',
                            vision_alignment='action_consistency' if config['alignment'] == 'ac' else 'stale')
                    env.oft_starts_at = None
                    if handoff['valid']:
                        active = 'oft'
                        budget = config.get('recovery_steps')
                        recovery_end = env.steps - 1 + budget if budget is not None else None
                    break
                env.step(process_action(raw).tolist())
        assert sum(env.counts.values()) == env.steps <= horizon
        return {
            'final_success': bool(env.done), 'final_step': env.steps, 'policy_steps': env.counts,
            'trigger_fired': fired, 'trigger_step': switch_step, 'return_to_adapter_step': return_step,
            'prefix_action_sha256': prefix_hash, 'all_actions_sha256': env.digest.hexdigest(),
            'actions': env.actions, 'perturbation_diagnostics': diag,
            'trigger_debug_trace': getattr(trigger.inner, 'trace', None),
            'bridge_result_valid': handoff.get('valid') if handoff else None,
            'L_handoff': handoff.get('L_handoff_s') if handoff else None,
            'C_async': handoff.get('C_async_s') if handoff else None,
            'B': handoff.get('B_s') if handoff else None,
            'no_stall': handoff.get('no_stall') if handoff else None,
            'wall_s': time.perf_counter()-origin,
        }
    finally:
        base.close()
