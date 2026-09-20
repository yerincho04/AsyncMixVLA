"""The frozen, gate-passed AsyncMixVLA visual-handoff residual (2026-09-07,
job 2168888): a tiny MLP trained with an ACTION-CONSISTENCY objective
(minimize |OFT(z_pred) - OFT(z_true)| through a fully frozen OFT) rather than
F2F-AP's latent-L2 reconstruction. On the offline DEV gate it cut first-
action and chunk action L2 vs stale by ~39% (F2F-AP: ~2%), with gripper
agreement at OFT's real 0.5 decision boundary equal to stale on the first
executed action -- see project memory / results/vlash_risk_audit/
gripper_threshold_check_report.json.

This is NOT F2F-AP and must never be labelled as such. It happens to share
F2F-AP's exact architecture (single 2-layer MLP, hidden 64) and checkpoint
FORMAT (produced by convert_C_residual_to_checkpoint.py), so the already-
verified F2FAP load + predict_future_latent path is reused verbatim -- only
the trained weights and the training objective differ. Kept as a distinct
class so logs, config (--vision_alignment action_consistency /
--action_consistency_checkpoint) and reporting never conflate the two, and
so the F2F-AP code path is provably untouched.
"""
from asyncmixvla.f2f_ap import F2FAP


class ActionConsistencyResidual(F2FAP):
    """Alias of F2FAP: identical load + predict_future_latent(z_now_full,
    bridge_actions) -> z_now_full + pooled_delta. Only the checkpoint (hence
    the trained weights and the objective they were trained under) differs.
    Frozen -- do not retrain or tune."""

    pass
