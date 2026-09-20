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

from experiments.robot.libero.run_libero_eval import GenerateConfig, check_unnorm_key
from experiments.robot.openvla_utils import (
    DEVICE,
    get_action_head,
    get_processor,
    get_proprio_projector,
    normalize_proprio,
    prepare_images_for_vla,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
from asyncmixvla.f2f_ap import F2FAP
from asyncmixvla.action_consistency import ActionConsistencyResidual


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
    Match the direct OpenVLA-OFT LIBERO eval config as closely as possible.
    """
    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        task_suite_name=args.task_suite_name,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=args.chunk_size,
        seed=args.seed,
    )
    return cfg


@app.post("/features")
async def features(request: Request):
    """READ-ONLY diagnostic endpoint, additive -- does not alter /act's
    behavior or the model's weights/inference path in any way. Runs only
    the vision-encoding sub-stage of predict_action (vision_backbone ->
    projector, i.e. model._process_vision_features, called directly and
    unmodified) and returns the mean-pooled projected patch embeddings
    (mean over the patch dimension, one vector per image-stack) instead of
    running the full LLM forward pass / action head. For the cross-policy
    visual handoff diagnostic only (audit_visual_representation_access.py /
    evaluate_visual_latent_diagnostic.py) -- never called from the live
    AsyncMixVLA runtime."""
    raw = await request.body()
    payload: Dict[str, Any] = msgpack.unpackb(raw, object_hook=m.decode, raw=False)

    cfg = POLICY["cfg"]
    resize_size = POLICY["resize_size"]
    obs = {
        "full_image": resize_image_for_policy(payload["full_image"], resize_size),
        "wrist_image": resize_image_for_policy(payload["wrist_image"], resize_size),
    }
    task_description = payload["task_description"]

    with torch.inference_mode():
        all_images = [obs["full_image"], obs["wrist_image"]]
        all_images = prepare_images_for_vla(all_images, cfg)
        primary_image = all_images.pop(0)
        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
        processor = POLICY["processor"]
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)
        if all_images:
            all_wrist_inputs = [processor(prompt, img).to(DEVICE, dtype=torch.bfloat16) for img in all_images]
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [w["pixel_values"] for w in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        model = POLICY["model"]
        input_ids = inputs["input_ids"]
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
        input_embeddings = model.get_input_embeddings()(input_ids)
        # Unmodified sub-call: same _process_vision_features the model's own
        # predict_action uses internally (see modeling_prismatic.py).
        projected_patch_embeddings = model._process_vision_features(inputs["pixel_values"], input_embeddings, use_film=False)
        pooled = projected_patch_embeddings.mean(dim=1).float().cpu().numpy()[0]  # (llm_dim,)

    packed = msgpack.packb({"pooled_features": pooled.astype(np.float32)}, default=m.encode, use_bin_type=True)
    return Response(content=packed, media_type="application/msgpack")


@app.post("/vision_latent_full")
async def vision_latent_full(request: Request):
    """DIAGNOSTIC-ONLY, additive. Same preprocessing as /features, but
    returns the FULL, un-pooled projected_patch_embeddings tensor (shape
    (num_patches*num_images, llm_dim)) instead of a mean-pooled vector.
    Mean-pooling (as /features does) destroys the per-patch structure the
    LLM's attention actually consumes and desyncs predict_action's
    NUM_PATCHES bookkeeping, so it cannot be fed back in through
    /act_from_latent for a numerically faithful injection -- this endpoint
    exists specifically to produce a latent that CAN be. Never called from
    the live AsyncMixVLA runtime; /act is untouched."""
    raw = await request.body()
    payload: Dict[str, Any] = msgpack.unpackb(raw, object_hook=m.decode, raw=False)

    cfg = POLICY["cfg"]
    resize_size = POLICY["resize_size"]
    obs = {
        "full_image": resize_image_for_policy(payload["full_image"], resize_size),
        "wrist_image": resize_image_for_policy(payload["wrist_image"], resize_size),
    }
    task_description = payload["task_description"]

    with torch.inference_mode():
        all_images = [obs["full_image"], obs["wrist_image"]]
        all_images = prepare_images_for_vla(all_images, cfg)
        primary_image = all_images.pop(0)
        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
        processor = POLICY["processor"]
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)
        if all_images:
            all_wrist_inputs = [processor(prompt, img).to(DEVICE, dtype=torch.bfloat16) for img in all_images]
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [w["pixel_values"] for w in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        model = POLICY["model"]
        input_ids = inputs["input_ids"]
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
        input_embeddings = model.get_input_embeddings()(input_ids)
        projected_patch_embeddings = model._process_vision_features(inputs["pixel_values"], input_embeddings, use_film=False)
        z_full = projected_patch_embeddings.float().cpu().numpy()[0]  # (num_patches*num_images, llm_dim)

    packed = msgpack.packb({"latent_full": z_full.astype(np.float32)}, default=m.encode, use_bin_type=True)
    return Response(content=packed, media_type="application/msgpack")


@app.post("/act_from_latent")
async def act_from_latent(request: Request):
    """DIAGNOSTIC-ONLY, additive. Injects a precomputed/intervened
    projected_patch_embeddings tensor at the exact seam predict_action()
    normally computes it (modeling_prismatic.py's new, additive
    precomputed_projected_patch_embeddings kwarg -- defaults to None
    everywhere else, so /act's own call path is untouched). Everything
    downstream of that seam (proprio injection, LLM forward, action head)
    is the SAME code /act uses, unmodified. An anchor full/wrist image is
    still required to build correctly-shaped input_ids/attention_mask/
    pixel_values via the normal processor call -- its pixel content is
    never read once the injected latent is supplied. Never called from the
    live AsyncMixVLA runtime."""
    raw = await request.body()
    payload: Dict[str, Any] = msgpack.unpackb(raw, object_hook=m.decode, raw=False)

    cfg = POLICY["cfg"]
    resize_size = POLICY["resize_size"]
    obs = {
        "full_image": resize_image_for_policy(payload["full_image"], resize_size),
        "wrist_image": resize_image_for_policy(payload["wrist_image"], resize_size),
        "state": np.asarray(payload["state"], dtype=np.float32),
    }
    task_description = payload["task_description"]
    latent_full = np.asarray(payload["latent_full"], dtype=np.float32)

    with torch.inference_mode():
        all_images = [obs["full_image"], obs["wrist_image"]]
        all_images = prepare_images_for_vla(all_images, cfg)
        primary_image = all_images.pop(0)
        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
        processor = POLICY["processor"]
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)
        if all_images:
            all_wrist_inputs = [processor(prompt, img).to(DEVICE, dtype=torch.bfloat16) for img in all_images]
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [w["pixel_values"] for w in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        model = POLICY["model"]
        z = torch.from_numpy(latent_full).unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)

        proprio = obs["state"]
        proprio_norm_stats = model.norm_stats[cfg.unnorm_key]["proprio"]
        proprio = normalize_proprio(proprio, proprio_norm_stats)

        actions, _ = model.predict_action(
            **inputs,
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=POLICY["proprio_projector"],
            noisy_action_projector=None,
            action_head=POLICY["action_head"],
            use_film=cfg.use_film,
            precomputed_projected_patch_embeddings=z,
        )

    actions = np.asarray(actions, dtype=np.float32)
    packed = msgpack.packb({"actions": actions}, default=m.encode, use_bin_type=True)
    return Response(content=packed, media_type="application/msgpack")


@app.post("/act_with_f2f_ap")
async def act_with_f2f_ap(request: Request):
    """F2F-AP baseline/system-component track: vision_alignment="f2f_ap" live
    runtime path. Computes OFT's own visual latent for the CURRENT (stale,
    T_predict-time) image, predicts the T_switch latent via the loaded F2F-AP
    module (causal -- uses only the current latent + the already-committed
    bridge_actions, no future knowledge), injects the predicted latent via
    the same bit-exact-verified precomputed_projected_patch_embeddings seam
    /act_from_latent uses, and returns the resulting action chunk. Requires
    the server to have been started with --f2f_ap_checkpoint; /act itself
    and every other endpoint are untouched. F2F-AP is a baseline, not the
    AsyncMixVLA novelty -- this endpoint exists to measure it fairly
    (including its own inference cost) inside the async compute budget, not
    to replace anything."""
    if POLICY.get("f2f_ap") is None:
        return Response(
            content=msgpack.packb({"error": "server was not started with --f2f_ap_checkpoint"}, default=m.encode, use_bin_type=True),
            media_type="application/msgpack", status_code=400,
        )
    raw = await request.body()
    payload: Dict[str, Any] = msgpack.unpackb(raw, object_hook=m.decode, raw=False)

    cfg = POLICY["cfg"]
    resize_size = POLICY["resize_size"]
    obs = {
        "full_image": resize_image_for_policy(payload["full_image"], resize_size),
        "wrist_image": resize_image_for_policy(payload["wrist_image"], resize_size),
        "state": np.asarray(payload["state"], dtype=np.float32),
    }
    task_description = payload["task_description"]
    bridge_actions = payload["bridge_actions"]

    with torch.inference_mode():
        all_images = [obs["full_image"], obs["wrist_image"]]
        all_images = prepare_images_for_vla(all_images, cfg)
        primary_image = all_images.pop(0)
        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
        processor = POLICY["processor"]
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)
        if all_images:
            all_wrist_inputs = [processor(prompt, img).to(DEVICE, dtype=torch.bfloat16) for img in all_images]
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [w["pixel_values"] for w in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        model = POLICY["model"]
        input_ids = inputs["input_ids"]
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
        input_embeddings = model.get_input_embeddings()(input_ids)
        z_now_full = model._process_vision_features(inputs["pixel_values"], input_embeddings, use_film=False)
        z_now_full_np = z_now_full.float().cpu().numpy()[0]  # (num_patches, llm_dim)

        z_pred_full_np = POLICY["f2f_ap"].predict_future_latent(z_now_full_np, bridge_actions)
        z_pred_full = torch.from_numpy(z_pred_full_np).unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)

        proprio = obs["state"]
        proprio_norm_stats = model.norm_stats[cfg.unnorm_key]["proprio"]
        proprio = normalize_proprio(proprio, proprio_norm_stats)

        actions, _ = model.predict_action(
            **inputs,
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=POLICY["proprio_projector"],
            noisy_action_projector=None,
            action_head=POLICY["action_head"],
            use_film=cfg.use_film,
            precomputed_projected_patch_embeddings=z_pred_full,
        )

    actions = np.asarray(actions, dtype=np.float32)
    packed = msgpack.packb({"actions": actions}, default=m.encode, use_bin_type=True)
    return Response(content=packed, media_type="application/msgpack")


@app.post("/act_with_action_consistency")
async def act_with_action_consistency(request: Request):
    """AsyncMixVLA visual-handoff mode vision_alignment="action_consistency"
    live runtime path. STRUCTURALLY IDENTICAL to /act_with_f2f_ap (compute
    OFT's own visual latent for the CURRENT/T_predict image, apply the loaded
    residual using only the current latent + the already-committed
    bridge_actions -- causal, no future knowledge -- then inject via the same
    bit-exact-verified precomputed_projected_patch_embeddings seam). The ONLY
    difference from /act_with_f2f_ap is which frozen residual checkpoint is
    loaded: the action-consistency-trained one (gate passed 2026-09-07,
    job 2168888), enabled by --action_consistency_checkpoint. /act,
    /act_with_f2f_ap and every other endpoint are untouched.

    Debug-only extras (never sent by the live runtime; used by
    verify_runtime_equivalence_action_consistency.py): if payload carries
    "debug_precomputed_z_now_full" the image-encoding step is skipped and
    that tensor is used as z_now_full directly; if payload["return_z_pred"]
    is truthy the response also includes the post-residual z_pred_full."""
    if POLICY.get("action_consistency") is None:
        return Response(
            content=msgpack.packb({"error": "server was not started with --action_consistency_checkpoint"},
                                  default=m.encode, use_bin_type=True),
            media_type="application/msgpack", status_code=400,
        )
    raw = await request.body()
    payload: Dict[str, Any] = msgpack.unpackb(raw, object_hook=m.decode, raw=False)

    cfg = POLICY["cfg"]
    resize_size = POLICY["resize_size"]
    obs = {
        "full_image": resize_image_for_policy(payload["full_image"], resize_size),
        "wrist_image": resize_image_for_policy(payload["wrist_image"], resize_size),
        "state": np.asarray(payload["state"], dtype=np.float32),
    }
    task_description = payload["task_description"]
    bridge_actions = payload["bridge_actions"]
    debug_z_now = payload.get("debug_precomputed_z_now_full")
    return_z_pred = bool(payload.get("return_z_pred", False))

    with torch.inference_mode():
        all_images = [obs["full_image"], obs["wrist_image"]]
        all_images = prepare_images_for_vla(all_images, cfg)
        primary_image = all_images.pop(0)
        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
        processor = POLICY["processor"]
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)
        if all_images:
            all_wrist_inputs = [processor(prompt, img).to(DEVICE, dtype=torch.bfloat16) for img in all_images]
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [w["pixel_values"] for w in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        model = POLICY["model"]
        if debug_z_now is not None:
            z_now_full_np = np.asarray(debug_z_now, dtype=np.float32)
        else:
            input_ids = inputs["input_ids"]
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
            input_embeddings = model.get_input_embeddings()(input_ids)
            z_now_full = model._process_vision_features(inputs["pixel_values"], input_embeddings, use_film=False)
            z_now_full_np = z_now_full.float().cpu().numpy()[0]  # (num_patches, llm_dim)

        z_pred_full_np = POLICY["action_consistency"].predict_future_latent(z_now_full_np, bridge_actions)
        z_pred_full = torch.from_numpy(z_pred_full_np).unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)

        proprio = obs["state"]
        proprio_norm_stats = model.norm_stats[cfg.unnorm_key]["proprio"]
        proprio = normalize_proprio(proprio, proprio_norm_stats)

        actions, _ = model.predict_action(
            **inputs,
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=POLICY["proprio_projector"],
            noisy_action_projector=None,
            action_head=POLICY["action_head"],
            use_film=cfg.use_film,
            precomputed_projected_patch_embeddings=z_pred_full,
        )

    actions = np.asarray(actions, dtype=np.float32)
    out = {"actions": actions}
    if return_z_pred:
        out["z_pred_full"] = np.asarray(z_pred_full_np, dtype=np.float32)
    packed = msgpack.packb(out, default=m.encode, use_bin_type=True)
    return Response(content=packed, media_type="application/msgpack")


@app.post("/act")
async def act(request: Request):
    t0 = time.perf_counter()

    raw = await request.body()
    payload: Dict[str, Any] = msgpack.unpackb(raw, object_hook=m.decode, raw=False)

    t1 = time.perf_counter()

    cfg = POLICY["cfg"]
    resize_size = POLICY["resize_size"]

    # Match direct run_libero_eval.py preprocessing:
    # get_libero_image/get_libero_wrist_image happens on client,
    # resize_image_for_policy happens here before calling get_action.
    obs = {
        "full_image": resize_image_for_policy(payload["full_image"], resize_size),
        "wrist_image": resize_image_for_policy(payload["wrist_image"], resize_size),
        "state": np.asarray(payload["state"], dtype=np.float32),
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
            "[OFT server] "
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
        default="moojink/openvla-7b-oft-finetuned-libero-10",
    )
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--chunk-size", type=int, default=NUM_ACTIONS_CHUNK)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument(
        "--f2f_ap_checkpoint", default=None,
        help="Path to a checkpoint saved by train_and_validate_f2f_ap.py. Enables /act_with_f2f_ap "
             "(the F2F-AP baseline/system-component track's live endpoint). Omit to leave it disabled "
             "-- every other endpoint, including /act, is unaffected either way.",
    )
    parser.add_argument(
        "--action_consistency_checkpoint", default=None,
        help="Path to the frozen action-consistency residual checkpoint (convert_C_residual_to_checkpoint.py). "
             "Enables /act_with_action_consistency (AsyncMixVLA visual-handoff mode "
             "vision_alignment=action_consistency). Omit to leave it disabled -- every other endpoint, "
             "including /act and /act_with_f2f_ap, is unaffected either way.",
    )

    args = parser.parse_args()

    cfg = make_cfg(args)

    print("[OFT] Loading model:", cfg.pretrained_checkpoint, flush=True)
    print("[OFT] Task suite:", cfg.task_suite_name, flush=True)

    model = get_model(cfg)

    # Match direct eval behavior: verify and set cfg.unnorm_key.
    check_unnorm_key(cfg, model)

    processor = get_processor(cfg)
    action_head = get_action_head(cfg, model.llm_dim)
    proprio_projector = get_proprio_projector(
        cfg,
        model.llm_dim,
        proprio_dim=PROPRIO_DIM,
    )

    print_param_count("OFT model", model)
    print_param_count("OFT full policy", model, action_head, proprio_projector)

    resize_size = get_image_resize_size(cfg)

    model.eval()
    action_head.eval()
    proprio_projector.eval()

    f2f_ap = None
    if args.f2f_ap_checkpoint is not None:
        f2f_ap = F2FAP(args.f2f_ap_checkpoint, device=DEVICE)
        print("[OFT] Loaded F2F-AP checkpoint:", args.f2f_ap_checkpoint, flush=True)

    action_consistency = None
    if args.action_consistency_checkpoint is not None:
        action_consistency = ActionConsistencyResidual(args.action_consistency_checkpoint, device=DEVICE)
        print("[OFT] Loaded action-consistency residual checkpoint:", args.action_consistency_checkpoint, flush=True)

    POLICY.update(
        {
            "cfg": cfg,
            "model": model,
            "processor": processor,
            "action_head": action_head,
            "proprio_projector": proprio_projector,
            "resize_size": resize_size,
            "print_timing": not args.no_timing,
            "f2f_ap": f2f_ap,
            "action_consistency": action_consistency,
        }
    )

    print("[OFT] unnorm_key:", cfg.unnorm_key, flush=True)
    print("[OFT] resize_size:", resize_size, flush=True)
    print(f"[OFT] Ready on http://{args.host}:{args.port}/act", flush=True)

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