"""F2F-AP: the generic, non-novel future-visual-latent predictor used as the
strongest future-vision BASELINE for AsyncMixVLA -- explicitly NOT the
project's novelty (see the closed Incoming-Policy-Aware Visual Handoff
Alignment track for the action-consistency-trained alternative this is
contrasted against). Trained offline with a pure reconstruction (latent-L2)
objective by train_and_validate_f2f_ap.py, which saves a checkpoint this
module loads.

Predicts OFT's own visual latent (the full un-pooled projected_patch_
embeddings tensor) at T_switch from the current (T_predict) latent plus the
already-committed bridge actions -- causal, uses only information available
at T_predict, so (unlike "oracle_future") it is usable inside the real async
overlap without seeing the future. Injected via the bit-exact-verified
precomputed_projected_patch_embeddings seam on predict_action(); this is
functionally "OFT sees a predicted future image" without requiring actual
pixel synthesis (no raw RGB is cached from this project's extraction, and a
from-scratch pixel-space video generator is not viable at this data scale).
"""
import numpy as np
import torch
import torch.nn as nn

HIDDEN = 64


class F2FAPPredictor(nn.Module):
    """Architecture must match train_and_validate_f2f_ap.py's F2FAPPredictor
    exactly (state_dict loading is purely structural -- shapes/names must
    line up, not the defining file)."""

    def __init__(self, z_dim, bridge_dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim + bridge_dim, hidden), nn.ReLU(), nn.Linear(hidden, z_dim))

    def forward(self, z_now_pooled_norm, bridge_feat_norm):
        return self.net(torch.cat([z_now_pooled_norm, bridge_feat_norm], dim=-1))


def bridge_feature_vector(bridge_actions, max_len=8):
    acts = np.asarray(bridge_actions, dtype=np.float64)
    mean_a, first_a, last_a = acts.mean(axis=0), acts[0], acts[-1]
    length = np.array([len(acts) / max_len])
    return np.concatenate([mean_a, first_a, last_a, length]).astype(np.float32)


class F2FAP:
    """Loaded, ready-to-use F2F-AP predictor: wraps the trained module plus
    the TRAIN-only normalization stats saved alongside it."""

    def __init__(self, checkpoint_path, device="cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.device = device
        self.model = F2FAPPredictor(ckpt["z_dim"], ckpt["bridge_dim"], ckpt.get("hidden", HIDDEN)).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.z_mean, self.z_std = ckpt["z_mean"], ckpt["z_std"]
        self.b_mean, self.b_std = ckpt["b_mean"], ckpt["b_std"]
        self.d_mean, self.d_std = ckpt["d_mean"], ckpt["d_std"]
        # Optional trust region used by deploy-matched checkpoints.  Older
        # checkpoints omit it and retain their byte-for-byte behavior.
        self.max_delta_norm_by_bridge = {
            int(k): float(v) for k, v in ckpt.get("max_delta_norm_by_bridge", {}).items()
        }

    def predict_delta(self, z_now_pooled: np.ndarray, bridge_actions) -> np.ndarray:
        """z_now_pooled: (llm_dim,) mean-pooled current visual latent.
        Returns the predicted (llm_dim,) shift, broadcast-added to every
        patch of the current full latent tensor by the caller."""
        bridge_feat = bridge_feature_vector(bridge_actions)
        with torch.no_grad():
            zn = torch.from_numpy(((z_now_pooled - self.z_mean) / self.z_std)[None]).float().to(self.device)
            br = torch.from_numpy(((bridge_feat - self.b_mean) / self.b_std)[None]).float().to(self.device)
            delta_norm = self.model(zn, br)[0].cpu().numpy()
        delta = delta_norm * self.d_std + self.d_mean
        cap = self.max_delta_norm_by_bridge.get(len(bridge_actions))
        norm = float(np.linalg.norm(delta))
        if cap is not None and norm > cap:
            delta = delta * (cap / max(norm, 1e-12))
        return delta

    def predict_future_latent(self, z_now_full: np.ndarray, bridge_actions) -> np.ndarray:
        """z_now_full: (num_patches, llm_dim) full un-pooled current visual
        latent. Returns the predicted future full latent tensor, same shape,
        ready for injection via predict_action(precomputed_projected_patch_embeddings=...)."""
        z_now_pooled = z_now_full.astype(np.float32).mean(axis=0)
        delta = self.predict_delta(z_now_pooled, bridge_actions)
        return z_now_full.astype(np.float32) + delta[None, :]
