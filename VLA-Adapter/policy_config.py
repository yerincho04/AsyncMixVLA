"""Inference-only configuration for the deployed Adapter server."""
from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class GenerateConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    use_minivlm: bool = True
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    unnorm_key: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    task_suite_name: str = "libero_10"
    seed: int = 7
    save_version: str = "vla-adapter"
    use_pro_version: bool = True
    phase: str = "Inference"


def check_unnorm_key(cfg, model):
    key = cfg.task_suite_name
    if key not in model.norm_stats and f"{key}_no_noops" in model.norm_stats:
        key = f"{key}_no_noops"
    if key not in model.norm_stats:
        raise KeyError(f"Action un-normalization key {key!r} not found in model norm_stats")
    cfg.unnorm_key = key
