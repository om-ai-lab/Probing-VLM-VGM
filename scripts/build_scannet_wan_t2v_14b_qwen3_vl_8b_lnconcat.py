"""Build offline ScanNet features by branch-LN concat of WAN-14B and Qwen3VL.

Default input layout:
  <feat_root>/wan-t2v-14b/<split>/<scene>/feature_t749_layer18.sft
  <feat_root>/qwen3-vl-8b/<split>/<scene>/feature_layer22.sft

Default output layout:
  <feat_root>/wan-t2v-14b-qwen3-vl-8b-lnconcat/<split>/<scene>/feature_layer22.sft

WAN is cropped to Qwen's temporal length, spatially resized to Qwen's token
grid, then each branch is LayerNormed per token along the channel dimension
before channel concat.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        default="data/ScanNet/ScanNet-processed",
        help="Root containing ScanNet train.json / val.json split files.",
    )
    parser.add_argument(
        "--feat-root",
        default="data/ScanNet/FEAT",
        help="Root containing per-VFM ScanNet feature trees.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--wan-vfm-name", default="wan-t2v-14b")
    parser.add_argument("--qwen-vfm-name", default="qwen3-vl-8b")
    parser.add_argument("--out-vfm-name", default="wan-t2v-14b-qwen3-vl-8b-lnconcat")
    parser.add_argument("--wan-postfix", default="_t749_layer18")
    parser.add_argument("--qwen-postfix", default="_layer22")
    parser.add_argument("--out-postfix", default="_layer22")
    parser.add_argument(
        "--resize-mode",
        choices=["avg", "bilinear"],
        default="avg",
        help="How to resize WAN's spatial grid to Qwen3VL's grid.",
    )
    parser.add_argument("--ln-eps", type=float, default=1e-6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _iter_split_scenes(processed_root: Path, split: str) -> Iterable[tuple[str, str]]:
    split_file = processed_root / f"{split}.json"
    with split_file.open("r") as f:
        pairs = json.load(f)
    for subset, scene_id in pairs:
        yield str(subset), str(scene_id)


def _resize_wan_to_qwen(
    wan_feat: torch.Tensor,
    qwen_t: int,
    qwen_hw: tuple[int, int],
    mode: str,
) -> torch.Tensor:
    if wan_feat.ndim != 4:
        raise ValueError(f"WAN feat must be (T,H,W,C), got {tuple(wan_feat.shape)}")
    if wan_feat.shape[0] < qwen_t:
        raise ValueError(
            f"WAN feat must have at least qwen_t={qwen_t} frames, got {wan_feat.shape[0]}"
        )
    wan_feat = wan_feat[:qwen_t]
    if tuple(wan_feat.shape[1:3]) == qwen_hw:
        return wan_feat.contiguous()

    x = wan_feat.permute(0, 3, 1, 2).float()
    if mode == "avg":
        x = F.adaptive_avg_pool2d(x, qwen_hw)
    elif mode == "bilinear":
        x = F.interpolate(x, size=qwen_hw, mode="bilinear", align_corners=False)
    else:
        raise ValueError(f"Unknown resize mode: {mode}")
    return x.permute(0, 2, 3, 1).contiguous().to(wan_feat.dtype)


def _per_token_layer_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = F.layer_norm(x.float(), (x.shape[-1],), eps=eps)
    return x.to(dtype)


def _build_one(
    feat_root: Path,
    subset: str,
    scene_id: str,
    args: argparse.Namespace,
) -> str:
    wan_path = (
        feat_root
        / args.wan_vfm_name
        / subset
        / scene_id
        / f"feature{args.wan_postfix}.sft"
    )
    qwen_path = (
        feat_root
        / args.qwen_vfm_name
        / subset
        / scene_id
        / f"feature{args.qwen_postfix}.sft"
    )
    out_dir = feat_root / args.out_vfm_name / subset / scene_id
    out_path = out_dir / f"feature{args.out_postfix}.sft"

    if out_path.exists() and not args.overwrite:
        return "skip"
    if not wan_path.is_file():
        return f"missing WAN: {wan_path}"
    if not qwen_path.is_file():
        return f"missing Qwen: {qwen_path}"
    if args.dry_run:
        return "dry-run"

    qwen_feat = load_file(str(qwen_path))["feat"]
    wan_feat = load_file(str(wan_path))["feat"]
    if qwen_feat.ndim != 4:
        raise ValueError(f"Qwen feat must be (T,H,W,C), got {tuple(qwen_feat.shape)}")
    if qwen_feat.shape[0] > 20:
        raise ValueError(f"Qwen feat must have at most 20 frames, got {qwen_feat.shape[0]}")

    wan_feat = _resize_wan_to_qwen(
        wan_feat,
        qwen_t=int(qwen_feat.shape[0]),
        qwen_hw=(int(qwen_feat.shape[1]), int(qwen_feat.shape[2])),
        mode=args.resize_mode,
    )
    wan_feat = _per_token_layer_norm(wan_feat, eps=float(args.ln_eps))
    qwen_feat = _per_token_layer_norm(qwen_feat, eps=float(args.ln_eps))
    fused = torch.cat([wan_feat, qwen_feat], dim=-1).contiguous()

    expected_c = int(wan_feat.shape[-1] + qwen_feat.shape[-1])
    if fused.shape != (*qwen_feat.shape[:3], expected_c):
        raise RuntimeError(
            f"Bad fused shape for {subset}/{scene_id}: {tuple(fused.shape)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    save_file({"feat": fused}, str(tmp_path))
    os.replace(tmp_path, out_path)
    return "write"


def main() -> None:
    args = _parse_args()
    processed_root = Path(args.processed_root)
    feat_root = Path(args.feat_root)

    for split in args.splits:
        stats: dict[str, int] = {}
        scenes = list(_iter_split_scenes(processed_root, split))
        if args.limit is not None:
            scenes = scenes[: args.limit]
        total = len(scenes)
        print(f"[{split}] scenes={total}")
        for i, (subset, scene_id) in enumerate(scenes, start=1):
            status = _build_one(feat_root, subset, scene_id, args)
            stats[status] = stats.get(status, 0) + 1
            if i == 1 or i % 25 == 0 or i == total:
                print(f"[{split}] {i}/{total} {subset}/{scene_id}: {status}")
        print(f"[{split}] summary: {stats}")


if __name__ == "__main__":
    main()
