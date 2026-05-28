#!/usr/bin/env python3
"""
Extract one 81-frame feature window from a DL3DV scene using Wan-T2V and
save **one safetensor per requested layer**:

    <out_dir>/feature_t{start_idx}_layer{layer_id}.sft

Each file holds a single key  ``"feat"``  whose value is shaped
**(T, H, W, C)** and stored in **FP16**.

Typical wrapper call
--------------------
python -m features.wan.extract_features \
       --scene-dir   data/DL3DV/DL3DV-ALL-960P/1K/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3/images_4 \
       --data-sft    data/DL3DV/DL3DV-processed/1K/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3.sft \
       --out-dir     output/features/wan/1K/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3\
       --model-id    ckpt/Wan2.1-T2V-1.3B-Diffusers \
       --prompt      "" \
       --t           749 \
       --output-layers 20
"""
from __future__ import annotations

import argparse
import glob
import logging
import math
import os
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from .wan_feature import WanFeaturizer, get_wan_featurizer
from .wan_feature_i2v import WanFeaturizerI2V, get_wan_featurizer_i2v

# ---------------------------------------------------------------------------- #
# Helpers                                                                      #
# ---------------------------------------------------------------------------- #


def list_frames(scene_dir: str, ext: str = "png") -> List[str]:
    """Return sorted list of frame PNG paths (frame_00001.png, …)."""
    return sorted(glob.glob(os.path.join(scene_dir, f"frame_*.{ext}")))


def load_frames(paths: List[str]) -> List[Image.Image]:
    return [Image.open(p) for p in paths]


def forward_wan(
    model: WanFeaturizer | WanFeaturizerI2V,
    frames: List[Image.Image],
    prompt: str,
    t: int,
    layer_ids: List[int],
    ensemble: int,
    i2v_condition_frame: str = "first",
) -> Dict[int, torch.Tensor]:
    kwargs = {}
    if isinstance(model, WanFeaturizerI2V):
        kwargs["condition_frame"] = i2v_condition_frame
    with torch.no_grad():
        return model.forward(
            video=frames,
            prompt=prompt,
            t=t,
            output_layer_indices=layer_ids,
            ensemble_size=ensemble,
            **kwargs,
        )


def reshape_to_t_h_w_c(raw: torch.Tensor) -> torch.Tensor:
    """
    Wan (1, N_tokens, C) → (T, H, W, C)

    With 832x480 inputs Wan produces (H_p=30, W_p=52) spatial tokens per
    *output* frame and temporal stride = 4 ⇒ T = 80/4+1 = 21 frames.

    """
    t_tokens, h_tokens, w_tokens = 21, 30, 52
    assert raw.ndim == 3, f"Expected 3D tensor, got {raw.shape}"
    assert raw.shape[0] == 1, f"Expected batch size 1, got {raw.shape[0]}"
    assert (
        raw.shape[1] == t_tokens * h_tokens * w_tokens
    ), f"Expected {t_tokens * h_tokens * w_tokens} tokens, got {raw.shape[1]}"
    return (
        raw.squeeze(0)  # (N_tokens, C)
        .reshape(t_tokens, h_tokens, w_tokens, raw.shape[-1])
        .contiguous()
    )


# ---------------------------------------------------------------------------- #
# Main                                                                         #
# ---------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="WAN feature extractor (one window, one file per layer)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene-dir", required=True, help="Image folder")
    parser.add_argument("--data-sft", required=True, help="Processed .sft file")
    parser.add_argument("--out-dir", required=True, help="Directory to save features")
    parser.add_argument("--image-ext", default="png", help="Image file extension")

    # Wan-specific
    parser.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--t", type=int, default=499, choices=range(0, 1001))
    parser.add_argument(
        "--output-layers",
        nargs="+",
        type=int,
        default=[15],
        help="Transformer block indices to extract",
    )
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument(
        "--i2v-condition-frame",
        choices=["first", "last"],
        default="first",
        help="Frame used as the image condition for Wan I2V models. Ignored for T2V.",
    )
    args, unknown = parser.parse_known_args(argv)

    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime}: [{levelname}] {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )

    if unknown:
        logging.debug(f"[warn] ignored unknown args: {unknown}")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 0 . Determine which layers are still missing (resume support)      #
    # ------------------------------------------------------------------ #
    fname_prefix = "feature_i2v" if "I2V" in args.model_id else "feature"
    if args.model_id.endswith("14B-Diffusers") and "I2V" not in args.model_id:
        fname_prefix = "feature_t2v_14b"
    missing_layers = [
        l
        for l in args.output_layers
        if not os.path.exists(
            os.path.join(out_dir, f"{fname_prefix}_t{args.t}_layer{l}.sft")
        )
    ]

    if not missing_layers:
        logging.info("All requested layer files already exist - nothing to do.")
        return

    # --------------------------------------------------------------------- #
    # 1 . Pick the temporal window                                          #
    # --------------------------------------------------------------------- #
    meta = load_file(args.data_sft)
    start_idx = int(meta["start_idx"].item())  # 0-based index ↔ frame_00001.png
    window_size = 81

    frame_paths = list_frames(args.scene_dir, ext=args.image_ext)
    total = len(frame_paths)

    if start_idx + window_size <= total:
        window_paths = frame_paths[start_idx : start_idx + window_size]
    elif start_idx == 0 and total < window_size:
        # Short scene: pad by repeating the last frame so WAN sees a full
        # 81-frame window. Query-GT alignment stays correct because the
        # dataset clamps effective_context_len to <= gt_num_frames <= total,
        # so `sel` never reaches the padded tokens' real coverage (only the
        # last ~1 WAN token of 21 may blend in padding).
        pad = window_size - total
        first_padded_token = math.ceil(total / 4)
        logging.warning(
            f"[WAN] {os.path.basename(args.scene_dir)}: only {total} frames "
            f"(< {window_size}); padding last frame x{pad}. "
            f"WAN tokens >= {first_padded_token} (of 21) may contain padding."
        )
        window_paths = frame_paths + [frame_paths[-1]] * pad
    else:
        # start_idx > 0 AND start_idx + 81 > total: shouldn't happen if
        # data_sft was produced consistently (gt_num_frames >= 81 implies
        # start_idx + 81 <= start_idx + gt_num_frames <= total). Raise so the
        # underlying data inconsistency isn't silently masked.
        raise RuntimeError(
            f"start_idx={start_idx} + {window_size} > total={total} in "
            f"{args.scene_dir}; stale or inconsistent data_sft"
        )

    frames = load_frames(window_paths)

    # --------------------------------------------------------------------- #
    # 2 . Forward pass                                                     #
    # --------------------------------------------------------------------- #
    logging.debug(f"[WAN] loading model {args.model_id}")
    if "T2V" in args.model_id:
        wan = get_wan_featurizer(model_id=args.model_id)
    else:
        wan = get_wan_featurizer_i2v(model_id=args.model_id)

    feats = forward_wan(
        wan,
        frames,
        prompt=args.prompt,
        t=args.t,
        layer_ids=missing_layers,
        ensemble=args.ensemble,
        i2v_condition_frame=args.i2v_condition_frame,
    )

    # Warn about missing layers
    missing = set(missing_layers) - set(feats.keys())
    if missing:
        logging.warning(
            f"[warn] requested layers {sorted(missing)} not returned by WAN model"
        )

    # --------------------------------------------------------------------- #
    # 3 . Save one .sft per layer                                           #
    # --------------------------------------------------------------------- #
    saved = 0
    for layer_id, raw_feat in feats.items():
        reshaped = reshape_to_t_h_w_c(raw_feat)
        out_path = os.path.join(
            out_dir, f"{fname_prefix}_t{args.t}_layer{layer_id}.sft"
        )
        save_file({"feat": reshaped.half()}, out_path)
        logging.debug(
            f"[WAN] saved layer {layer_id} → {out_path} "
            f"shape {tuple(reshaped.shape)}"
        )
        saved += 1

    if saved == 0:
        logging.error("[err] no layers saved - aborting")
    else:
        logging.info(f"[WAN] done - {saved} layer files written to {out_dir}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
