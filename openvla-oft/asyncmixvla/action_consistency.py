"""Frozen action-consistency visual residual used by deployed AsyncMixVLA."""

import numpy as np
import torch
import torch.nn as nn

HIDDEN = 64


class _ResidualPredictor(nn.Module):
    def __init__(self, z_dim, bridge_dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + bridge_dim, hidden), nn.ReLU(), nn.Linear(hidden, z_dim)
        )

    def forward(self, latent, bridge):
        return self.net(torch.cat([latent, bridge], dim=-1))


def _bridge_features(bridge_actions, max_len=8):
    actions = np.asarray(bridge_actions, dtype=np.float64)
    return np.concatenate(
        [actions.mean(axis=0), actions[0], actions[-1], np.array([len(actions) / max_len])]
    ).astype(np.float32)


class ActionConsistencyResidual:
    """Load the trained residual and predict OFT's future visual latent."""

    def __init__(self, checkpoint_path, device="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.device = device
        self.model = _ResidualPredictor(
            checkpoint["z_dim"], checkpoint["bridge_dim"], checkpoint.get("hidden", HIDDEN)
        ).to(device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.z_mean, self.z_std = checkpoint["z_mean"], checkpoint["z_std"]
        self.b_mean, self.b_std = checkpoint["b_mean"], checkpoint["b_std"]
        self.d_mean, self.d_std = checkpoint["d_mean"], checkpoint["d_std"]
        self.max_delta_norm_by_bridge = {
            int(k): float(v) for k, v in checkpoint.get("max_delta_norm_by_bridge", {}).items()
        }

    def predict_delta(self, pooled_latent, bridge_actions):
        bridge = _bridge_features(bridge_actions)
        with torch.no_grad():
            latent_norm = torch.from_numpy(
                ((pooled_latent - self.z_mean) / self.z_std)[None]
            ).float().to(self.device)
            bridge_norm = torch.from_numpy(((bridge - self.b_mean) / self.b_std)[None]).float().to(self.device)
            normalized_delta = self.model(latent_norm, bridge_norm)[0].cpu().numpy()
        delta = normalized_delta * self.d_std + self.d_mean
        cap = self.max_delta_norm_by_bridge.get(len(bridge_actions))
        norm = float(np.linalg.norm(delta))
        if cap is not None and norm > cap:
            delta = delta * (cap / max(norm, 1e-12))
        return delta

    def predict_future_latent(self, current_latent, bridge_actions):
        current = current_latent.astype(np.float32)
        return current + self.predict_delta(current.mean(axis=0), bridge_actions)[None, :]
