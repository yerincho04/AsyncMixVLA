"""Validation and scoring for the deployed observable V2 trigger head."""
import json
from pathlib import Path
import numpy as np
from .features import FEATURE_NAMES, SCHEMA


def load_artifact(path):
    artifact = json.loads(Path(path).read_text())
    if (artifact.get("schema") != SCHEMA or artifact.get("privileged_inputs") is not False
            or artifact.get("feature_names") != list(FEATURE_NAMES)):
        raise ValueError("V2 input contract mismatch")
    if artifact.get("policy") not in ("utility", "failure"):
        raise ValueError("Unsupported V2 policy")
    if not np.isfinite(artifact.get("threshold", float("nan"))):
        raise ValueError("Invalid threshold")
    n_features = len(FEATURE_NAMES)
    for name in ("mean", "scale"):
        values = np.asarray(artifact.get(name), dtype=float)
        if values.shape != (n_features,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name}")
    if np.any(np.asarray(artifact["scale"]) <= 0):
        raise ValueError("Scale must be positive")
    head_names = ("rescue", "harm") if artifact["policy"] == "utility" else ("failure",)
    for name in head_names:
        head = artifact.get(name, {})
        coef = np.asarray(head.get("coef"), dtype=float)
        if (coef.shape != (n_features,) or not np.isfinite(coef).all()
                or not np.isfinite(head.get("intercept", float("nan")))):
            raise ValueError(f"Invalid {name} head")
    return artifact


def utility_score(features, artifact):
    values = np.asarray(features, dtype=np.float64)
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("Invalid V2 input")
    normalized = np.clip(
        (values - np.asarray(artifact["mean"])) / np.asarray(artifact["scale"]), -10, 10
    )

    def probability(head):
        logit = float(normalized @ np.asarray(head["coef"]) + head["intercept"])
        return float(1 / (1 + np.exp(-np.clip(logit, -30, 30))))

    if artifact["policy"] == "failure":
        return probability(artifact["failure"])
    return probability(artifact["rescue"]) - probability(artifact["harm"])
