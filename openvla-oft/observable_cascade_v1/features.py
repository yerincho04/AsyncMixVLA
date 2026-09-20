"""Fixed causal feature contract for the deployed V1+V2 cascade."""
from observable_cascade_v1.observable import ObservableFeatures

SCHEMA = "observable_cascade_spatial_v1"
FEATURE_NAMES = (
    tuple(ObservableFeatures.feature_names)
    + tuple(f"visual.spatial.{i}" for i in range(512))
    + tuple(f"visual.delta4.{i}" for i in range(128))
    + ("v1.score",)
)
