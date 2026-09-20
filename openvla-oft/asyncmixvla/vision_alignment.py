"""Modular vision-conditioning for the async OFT request.

Kept separate from proprioception handling (which is always VLASH-style
roll-forward via roll_forward_proprio, reused unmodified from
run_seamless_handoff.py -- see asyncmixvla/bridge.py) so vision prediction
can be swapped without touching trigger/bridge/timing logic (spec section 11).

"stale" is the current, validated, real-deployment mode: the image from the
T_predict observation, used as-is -- causal, real-deployment-viable.

"oracle_future" is an ACAUSAL debug/comparison-only mode (the true
post-bridge frame). The true future image does not exist until the bridge
finishes, so this mode requires the caller to have already obtained it
(future_obs) -- normally by deferring the OFT query until after the bridge
completes (see asyncmixvla/bridge.py's run_async_bridge, which special-cases
this mode for exactly that reason). Never a real-deployment option; used only
as an upper-bound reference, same role "oracle" plays throughout every
offline diagnostic in this project.

"f2f_ap" is the F2F-AP baseline/system-component track's causal future-
visual-LATENT predictor (asyncmixvla/f2f_ap.py) -- it only uses information
available at T_predict (the current latent + the already-committed bridge
actions), so unlike oracle_future it is usable inside the real async overlap.
Rather than returning an image, this mode signals the caller to route the
request through call_policy_f2f_ap()/the server's /act_with_f2f_ap endpoint,
which computes the current latent, predicts the future one, and injects it
server-side -- see run_test0_switch_timing.call_policy_f2f_ap.

"action_consistency" is the frozen, gate-passed AsyncMixVLA visual-handoff
method (asyncmixvla/action_consistency.py; job 2168888). Structurally
identical to "f2f_ap" (same causal inputs, same server-side latent
predict+inject) -- it only differs in the residual's training objective
(action-consistency through frozen OFT vs F2F-AP's latent-L2 reconstruction)
and therefore its checkpoint. Routes through
call_policy_action_consistency()/the server's /act_with_action_consistency
endpoint (server must be started with --action_consistency_checkpoint).
"""
import numpy as np


def align_observation(mode, *, stale_obs, prepare_observation_fn, future_state=None, future_obs=None):
    """Returns a prepared observation dict (full_image/wrist_image/state)
    ready for call_policy("oft", ...) -- except mode="f2f_ap", which returns
    a dict carrying use_f2f_ap=True for the caller to route to
    call_policy_f2f_ap() instead (see asyncmixvla/bridge.py's
    oft_query_pipeline).

    mode="stale": image and (absent an override) state both come from
    stale_obs, i.e. the true T_predict observation -- this is the causal,
    real baseline (Naive Async).

    mode="stale" + future_state given: image stays stale, state is
    overridden with the VLASH-predicted T_switch proprioception -- this is
    the VLASH-style alignment (primary AsyncMixVLA path). future_state is
    computed by the caller via roll_forward_proprio and passed in; this
    function does not compute it, matching the "vision_alignment module
    only decides the image" separation of concerns.

    mode="oracle_future": image comes from future_obs (the true post-bridge
    observation) instead of stale_obs; state still honors a future_state
    override if given, exactly like "stale". Raises if future_obs is not
    supplied -- this mode cannot fabricate a future it hasn't been given.

    mode="f2f_ap": image/state fields are still populated from stale_obs (the
    server needs the current image to compute the current latent), plus
    use_f2f_ap=True so the caller knows to call call_policy_f2f_ap() instead
    of call_policy("oft", ...). bridge_actions are NOT attached here (this
    function only decides the image/state, matching the existing separation
    of concerns) -- the caller already has bridge_actions_raw and passes it
    to call_policy_f2f_ap() directly.

    mode="action_consistency": exactly like "f2f_ap" but sets
    use_action_consistency=True instead, so the caller routes to
    call_policy_action_consistency(). Same causal inputs, same separation of
    concerns.
    """
    if mode == "stale":
        prepared = dict(prepare_observation_fn(stale_obs))
        if future_state is not None:
            prepared["state"] = np.asarray(future_state, dtype=np.float32)
        return prepared
    if mode == "action_consistency":
        prepared = dict(prepare_observation_fn(stale_obs))
        if future_state is not None:
            prepared["state"] = np.asarray(future_state, dtype=np.float32)
        prepared["use_action_consistency"] = True
        return prepared
    if mode == "oracle_context":
        # Experiment-A condition D: the FULL true post-bridge (T_switch) context
        # -- true image AND true proprioception. Unlike "oracle_future" it
        # deliberately IGNORES any predicted future_state, so nothing about the
        # handoff observation is estimated. Acausal upper bound: the true
        # T_switch observation does not exist until the bridge completes, so the
        # caller must defer the OFT query (see run_async_bridge). Never a
        # real-deployment option -- it exists to answer "is there any closed-loop
        # headroom for handoff-context alignment at all, at this takeover_k?".
        if future_obs is None:
            raise ValueError(
                "oracle_context vision alignment requires future_obs (the true post-bridge "
                "observation); the caller must defer the OFT query until the bridge completes."
            )
        return dict(prepare_observation_fn(future_obs))
    if mode == "oracle_future":
        if future_obs is None:
            raise ValueError(
                "oracle_future vision alignment requires future_obs (the true post-bridge "
                "observation) -- the caller must defer the OFT query until after the bridge "
                "completes, since this mode cannot see a future it hasn't been given."
            )
        prepared = dict(prepare_observation_fn(future_obs))
        if future_state is not None:
            prepared["state"] = np.asarray(future_state, dtype=np.float32)
        return prepared
    if mode == "f2f_ap":
        prepared = dict(prepare_observation_fn(stale_obs))
        if future_state is not None:
            prepared["state"] = np.asarray(future_state, dtype=np.float32)
        prepared["use_f2f_ap"] = True
        return prepared
    raise ValueError(f"unknown vision_alignment mode: {mode!r}")
