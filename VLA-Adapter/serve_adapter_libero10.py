import argparse
import time
from typing import Dict, Any

import msgpack
import msgpack_numpy as m
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request, Response

m.patch()

from policy_config import GenerateConfig, check_unnorm_key
from experiments.robot.openvla_utils import (
    DEVICE,
    get_action_head,
    get_processor,
    get_proprio_projector,
    prepare_images_for_vla,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM


app = FastAPI()
POLICY = {}

def print_param_count(name, *modules):
    total_params = 0
    trainable_params = 0

    for module in modules:
        total_params += sum(p.numel() for p in module.parameters())
        trainable_params += sum(p.numel() for p in module.parameters() if p.requires_grad)

    print(f"[{name}] total params: {total_params / 1e9:.3f}B", flush=True)
    print(f"[{name}] trainable params: {trainable_params / 1e6:.3f}M", flush=True)

def make_cfg(args):
    """
    Match the VLA-Adapter LIBERO eval config as closely as possible.
    """
    model_version = args.model_version
    if model_version is None:
        model_version = "v2" if args.use_pro_version else "v1"

    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        task_suite_name=args.task_suite_name,

        # VLA-Adapter settings
        use_l1_regression=True,
        use_minivlm=True,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=args.chunk_size,

        # Usually unused for VLA-Adapter, but keep explicit.
        load_in_8bit=False,
        load_in_4bit=False,

        # VLA-Adapter repo-specific fields
        save_version=model_version,
        use_pro_version=args.use_pro_version,
        phase="Inference",

        seed=args.seed,
    )
    return cfg




@app.post("/act")
async def act(request: Request):
    t0 = time.perf_counter()

    raw = await request.body()
    payload: Dict[str, Any] = msgpack.unpackb(raw, object_hook=m.decode, raw=False)

    t1 = time.perf_counter()

    cfg = POLICY["cfg"]
    resize_size = POLICY["resize_size"]

    state = np.asarray(payload["state"], dtype=np.float32)

    if state is None:
        raise ValueError("Received state=None from rollout runner.")

    if state.shape[-1] != PROPRIO_DIM:
        raise ValueError(f"Expected proprio state shape [{PROPRIO_DIM}], got {state.shape}")

    # Match the evaluated Adapter preprocessing:
    # client sends raw LIBERO images; server resizes them before get_action(...).
    obs = {
        "full_image": resize_image_for_policy(payload["full_image"], resize_size),
        "wrist_image": resize_image_for_policy(payload["wrist_image"], resize_size),
        "state": state,
    }

    task_description = payload["task_description"]

    t2 = time.perf_counter()

    with torch.inference_mode():
        actions = get_action(
            cfg,
            POLICY["model"],
            obs,
            task_description,
            processor=POLICY["processor"],
            action_head=POLICY["action_head"],
            proprio_projector=POLICY["proprio_projector"],
            noisy_action_projector=None,
            use_film=cfg.use_film,
            use_minivlm=cfg.use_minivlm,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t3 = time.perf_counter()

    actions = np.asarray(actions, dtype=np.float32)

    if actions.ndim != 2 or actions.shape[-1] != 7:
        raise ValueError(f"Expected action chunk shape [T, 7], got {actions.shape}")

    packed = msgpack.packb({"actions": actions}, default=m.encode, use_bin_type=True)

    t4 = time.perf_counter()

    if POLICY["print_timing"]:
        print(
            "[Adapter server] "
            f"recv+unpack={t1 - t0:.4f}s | "
            f"preprocess={t2 - t1:.4f}s | "
            f"infer={t3 - t2:.4f}s | "
            f"pack={t4 - t3:.4f}s | "
            f"total={t4 - t0:.4f}s | "
            f"actions={actions.shape}",
            flush=True,
        )

    return Response(content=packed, media_type="application/msgpack")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/VLA-Adapter-LIBERO-Long-Pro",
        help="Prefer a local checkpoint path, not the HF repo ID.",
    )
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--chunk-size", type=int, default=NUM_ACTIONS_CHUNK)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument(
        "--use-pro-version",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # For LIBERO-Long-Pro, v2 is usually the intended model version.
    parser.add_argument(
        "--model-version",
        default=None,
        help="Model version passed to model.set_version(...). Default: v2 if --use-pro-version, else v1.",
    )

    parser.add_argument("--no-timing", action="store_true")

    args = parser.parse_args()
    cfg = make_cfg(args)

    print("[Adapter] Loading model:", cfg.pretrained_checkpoint, flush=True)
    print("[Adapter] Task suite:", cfg.task_suite_name, flush=True)
    print("[Adapter] use_pro_version:", cfg.use_pro_version, flush=True)
    print("[Adapter] model/save version:", cfg.save_version, flush=True)

    model = get_model(cfg)

    # Match the VLA-Adapter evaluator pattern:
    # initialize_model() calls model.set_version(cfg.save_version).
    if hasattr(model, "set_version"):
        model.set_version(cfg.save_version)
        print("[Adapter] model.version =", getattr(model, "version", None), flush=True)

    # Match direct eval behavior: verify and set cfg.unnorm_key.
    check_unnorm_key(cfg, model)

    processor = get_processor(cfg)
    action_head = get_action_head(cfg, model.llm_dim)
    proprio_projector = get_proprio_projector(
        cfg,
        model.llm_dim,
        proprio_dim=PROPRIO_DIM,
    )

    print_param_count("Adapter model", model)
    print_param_count("Adapter full policy", model, action_head, proprio_projector)

    resize_size = get_image_resize_size(cfg)

    model.eval()
    action_head.eval()
    proprio_projector.eval()

    POLICY.update(
        {
            "cfg": cfg,
            "model": model,
            "processor": processor,
            "action_head": action_head,
            "proprio_projector": proprio_projector,
            "resize_size": resize_size,
            "print_timing": not args.no_timing,
        }
    )

    print("[Adapter] cfg.unnorm_key =", cfg.unnorm_key, flush=True)
    print("[Adapter] cfg.use_proprio =", cfg.use_proprio, flush=True)
    print("[Adapter] cfg.use_minivlm =", cfg.use_minivlm, flush=True)
    print("[Adapter] cfg.use_pro_version =", cfg.use_pro_version, flush=True)
    print("[Adapter] resize_size =", resize_size, flush=True)
    print(f"[Adapter] Ready on http://{args.host}:{args.port}/act", flush=True)

    # Important: workers=1. Multiple workers would load multiple model copies.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        access_log=False,
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
