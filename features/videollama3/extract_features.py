#!/usr/bin/env python3
"""Extract VideoLLaMA3 hidden-state features for probe training.

Example:
    python -m features.videollama3.extract_features \
        --scene-dir data/ScanNet/ScanNet-processed/train/scene0000_00/frames \
        --data-sft data/ScanNet/ScanNet-processed/train/scene0000_00/metadata.sft \
        --out-dir data/ScanNet/FEAT/videollama3/train/scene0000_00 \
        --model-path ckpt/VideoLLaMA3-2B \
        --num-frames 8 \
        --output-layers 7 14 21 28
"""

from __future__ import annotations

import argparse
import builtins
import logging
import os
import sys
from pprint import pprint
from typing import List

import torch
from safetensors.torch import load_file, save_file

from .extractor import get_videollama3_extractor

builtins.pp = pprint

logging.basicConfig(
    level=logging.INFO,
    format="{asctime}: [{levelname}] {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VideoLLaMA3 feature extractor for probe training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--scene-dir", required=True, help="Directory containing frame_*.png/jpg files")
    parser.add_argument("--out-dir", required=True, help="Output directory for feature_layer*.sft files")
    parser.add_argument(
        "--data-sft",
        default=None,
        help="Processed .sft metadata file containing start_idx and gt_num_frames",
    )
    parser.add_argument("--image-ext", default="png", help="Frame extension (alias for --frame-ext)")
    parser.add_argument("--frame-ext", default=None, help="Frame extension")

    parser.add_argument(
        "--model-path",
        required=True,
        help="Local path or HF id for VideoLLaMA3",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of frames to sample (ignored with --use-query-frame-indices)",
    )
    parser.add_argument(
        "--output-layers",
        nargs="+",
        type=int,
        default=[7, 14, 21, 28],
        help="LLM hidden-state layer indices to save (VideoLLaMA3-2B/7B: 28 layers)",
    )
    parser.add_argument("--prompt", default="", help="Neutral prompt for the multimodal forward pass")
    parser.add_argument("--device", default="cuda:0", help="Device to load the model on")
    parser.add_argument(
        "--target-size",
        nargs=2,
        type=int,
        default=[960, 540],
        metavar=("W", "H"),
        help="Resize frames to (W, H) before the VideoLLaMA3 processor. Use 0 0 to keep original size.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager", "none"],
        help="Attention implementation passed to from_pretrained; use 'none' to omit the argument.",
    )
    parser.add_argument(
        "--use-token-compression",
        action="store_true",
        help="Keep VideoLLaMA3's content-dependent visual token compression on. "
             "This usually breaks [T,H,W,C] reshape and is intended only for debugging.",
    )

    parser.add_argument(
        "--use-query-frame-indices",
        action="store_true",
        help="Use query frame indices matching the ScanNet probe dataset.",
    )
    parser.add_argument("--context-len", type=int, default=76)
    parser.add_argument("--query-idx-divisor", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Overwrite existing feature files")

    return parser.parse_args(argv)


def check_existing_files(out_dir: str, layers: List[int]) -> List[int]:
    missing = []
    for layer in layers:
        path = os.path.join(out_dir, f"feature_layer{layer}.sft")
        if not os.path.exists(path):
            missing.append(layer)
    return missing


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.force:
        missing_layers = check_existing_files(args.out_dir, args.output_layers)
        if not missing_layers:
            logger.info("All layer files exist. Use --force to overwrite.")
            return
        if len(missing_layers) < len(args.output_layers):
            logger.info("Extracting only missing layers: %s", missing_layers)
            args.output_layers = missing_layers

    if not os.path.isdir(args.scene_dir):
        logger.error("Scene directory not found: %s", args.scene_dir)
        sys.exit(1)

    frame_ext = args.frame_ext if args.frame_ext else args.image_ext

    start_idx = 0
    gt_num_frames = None
    if args.data_sft and os.path.exists(args.data_sft):
        meta = load_file(args.data_sft)
        start_idx = int(meta["start_idx"].item())
        if "gt_num_frames" in meta:
            gt_num_frames = int(meta["gt_num_frames"].item())
        elif "images" in meta:
            gt_num_frames = meta["images"].shape[0]
        else:
            raise KeyError(
                f"{args.data_sft} has neither 'gt_num_frames' nor 'images'; cannot infer clip length."
            )
        logger.info("Loaded metadata: start_idx=%d, gt_num_frames=%s", start_idx, gt_num_frames)
    else:
        logger.warning("No data_sft provided, using default frame range (may cause misalignment)")

    target_size = tuple(args.target_size) if args.target_size[0] > 0 else None
    attn_impl = None if args.attn_implementation == "none" else args.attn_implementation

    logger.info("Loading VideoLLaMA3 from %s", args.model_path)
    logger.info(
        "Device=%s, layers=%s, target_size=%s, attn=%s, token_compression=%s",
        args.device,
        args.output_layers,
        target_size,
        attn_impl,
        args.use_token_compression,
    )
    extractor = get_videollama3_extractor(
        model_path=args.model_path,
        select_layers=tuple(args.output_layers),
        question=args.prompt,
        device=args.device,
        target_size=target_size,
        attn_implementation=attn_impl or "",
        use_token_compression=args.use_token_compression,
    )

    logger.info("Extracting features from %s", args.scene_dir)
    if args.use_query_frame_indices:
        logger.info(
            "Using query frame indices mode (context_len=%d, divisor=%d)",
            args.context_len,
            args.query_idx_divisor,
        )
    else:
        logger.info(
            "Sampling %d frames from GT range [%d, %s)",
            args.num_frames,
            start_idx,
            start_idx + gt_num_frames if gt_num_frames is not None else "all",
        )

    try:
        features = extractor.extract(
            frame_dir=args.scene_dir,
            num_frames=args.num_frames,
            frame_ext=frame_ext,
            start_idx=start_idx,
            gt_num_frames=gt_num_frames,
            use_query_frame_indices=args.use_query_frame_indices,
            context_len=args.context_len,
            query_idx_divisor=args.query_idx_divisor,
        )
    except Exception as exc:
        logger.error("Feature extraction failed: %s", exc)
        raise

    saved = 0
    for layer, feat in features.items():
        out_path = os.path.join(args.out_dir, f"feature_layer{layer}.sft")
        feat_half = feat.half().cpu()
        logger.info("Saving layer %s: shape %s -> %s", layer, tuple(feat_half.shape), out_path)
        save_file({"feat": feat_half}, out_path)
        saved += 1

    logger.info("Done. Saved %d layer files to %s", saved, args.out_dir)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
