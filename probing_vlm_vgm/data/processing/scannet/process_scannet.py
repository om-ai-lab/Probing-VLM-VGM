"""Preprocess raw ScanNet scenes into the 81-frame clip layout used by Setup A.

For each scene under ``--raw_root/scans_{train,val}/<scene>/`` we produce:

    <out_root>/<split>/<scene>/
        frames/frame_00000.jpg ... frame_00080.jpg  (81 frames, resized to target HxW)
        instance_masks.npy                (81, H, W) uint16, multi-view consistent IDs
        poses.npy                         (81, 4, 4) float32 camera-to-world
        intrinsic.txt                     4x4 color intrinsic (copied from exported/)
        metadata.sft                      start_idx=0, gt_num_frames=81  (matches DL3DV)

Frame sampling: ``np.linspace(0, len(color)-1, 81).round().astype(int)``. Scenes
with fewer than MIN_FRAMES frames are skipped. Short scenes (20 <= n < 81) get
repeated frames.

Instance masks come from ``<scene>_2d-instance-filt.zip`` which already encodes
multi-view consistent object IDs (ScanNet's 3D→2D projection, refined). We
resize masks with ``Image.NEAREST`` to preserve integer IDs.

Example:
    python -m probing_vlm_vgm.data.processing.scannet.process_scannet \\
        --raw_root /data24/shz/project/3detr/votenet/scannet/scans \\
        --out_root data/ScanNet/ScanNet-processed \\
        --split train --num_scenes 100
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

DEFAULT_NUM_FRAMES = 81
DEFAULT_TARGET_H = 540
DEFAULT_TARGET_W = 720
DEFAULT_MIN_FRAMES = 20
DEFAULT_WINDOW_RATIO = 0.50  # center-window fraction of exported frames used to source 81 clip
DEFAULT_MIN_WINDOW_FRAMES = 60  # fall back to full scene if the ratio-window has fewer exported frames


def discover_scenes(raw_root: Path, split: str) -> List[Path]:
    """Return sorted list of scene directories under scans_{split}/."""
    split_dir = raw_root / f"scans_{split}"
    if not split_dir.is_dir():
        raise FileNotFoundError(f"ScanNet split directory not found: {split_dir}")
    return sorted([d for d in split_dir.iterdir() if d.is_dir()])


def list_color_frames(scene_dir: Path) -> List[int]:
    """Return sorted list of frame indices (ints from filename stems) under exported/color/."""
    color_dir = scene_dir / "exported" / "color"
    if not color_dir.is_dir():
        return []
    idx_list = []
    for f in color_dir.iterdir():
        if f.suffix.lower() != ".jpg":
            continue
        try:
            idx_list.append(int(f.stem))
        except ValueError:
            continue
    idx_list.sort()
    return idx_list


def resize_jpg(src: Path, dst: Path, target_hw: tuple) -> None:
    """Resize an RGB JPG using bilinear, save to dst."""
    img = Image.open(src).convert("RGB")
    img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)  # PIL: (W, H)
    img.save(dst, quality=90)


def read_mask_from_zip(zip_ref: zipfile.ZipFile, orig_frame_idx: int) -> Image.Image:
    """Read a single instance mask PNG from the zip."""
    name = f"instance-filt/{orig_frame_idx}.png"
    data = zip_ref.read(name)
    return Image.open(io.BytesIO(data))


def resize_mask(mask: Image.Image, target_hw: tuple) -> np.ndarray:
    """Resize an instance mask with NEAREST (preserve integer IDs).

    Returns uint16 numpy array.
    """
    mask = mask.resize((target_hw[1], target_hw[0]), Image.NEAREST)
    arr = np.array(mask)
    return arr.astype(np.uint16)


def process_scene(
    scene_dir: Path,
    out_dir: Path,
    split: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
    target_hw: tuple = (DEFAULT_TARGET_H, DEFAULT_TARGET_W),
    min_frames: int = DEFAULT_MIN_FRAMES,
    window_ratio: float = DEFAULT_WINDOW_RATIO,
    min_window_frames: int = DEFAULT_MIN_WINDOW_FRAMES,
    skip_existing: bool = True,
) -> str:
    """Process a single scene. Returns one of {'ok', 'skipped', 'missing', 'short'}."""
    scene_id = scene_dir.name

    if skip_existing:
        sentinel = out_dir / "metadata.sft"
        if sentinel.exists():
            return "skipped"

    exported = scene_dir / "exported"
    color_dir = exported / "color"
    pose_dir = exported / "pose"
    intrinsic_color_file = exported / "intrinsic" / "intrinsic_color.txt"
    inst_zip = scene_dir / f"{scene_id}_2d-instance-filt.zip"

    if not color_dir.is_dir() or not pose_dir.is_dir() or not inst_zip.is_file():
        return "missing"

    orig_indices = list_color_frames(scene_dir)
    if len(orig_indices) < min_frames:
        return "short"

    # Select center window of exported frames. Denser sampling in a smaller
    # temporal span keeps more instances visible across multiple sampled views,
    # which strengthens the MVC pull signal. Fall back to the full scene when
    # the window would be too short (preserves coverage for short scenes).
    n_total = len(orig_indices)
    window_len = int(round(n_total * window_ratio))
    if window_len < min_window_frames:
        window_indices = orig_indices
    else:
        start = (n_total - window_len) // 2
        window_indices = orig_indices[start:start + window_len]

    # Uniform linear sampling within the (windowed) frame pool.
    sel_positions = np.linspace(0, len(window_indices) - 1, num_frames).round().astype(int)
    sel_orig = [window_indices[p] for p in sel_positions]

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_out = out_dir / "frames"
    frames_out.mkdir(exist_ok=True)

    # 1. Frames: resize and save
    for i, orig_idx in enumerate(sel_orig):
        src = color_dir / f"{orig_idx}.jpg"
        if not src.is_file():
            # Should not happen if orig_indices was built correctly
            raise FileNotFoundError(f"Missing frame: {src}")
        dst = frames_out / f"frame_{i:05d}.jpg"
        resize_jpg(src, dst, target_hw)

    # 2. Instance masks: read from zip + resize nearest
    masks = np.empty((num_frames, target_hw[0], target_hw[1]), dtype=np.uint16)
    with zipfile.ZipFile(inst_zip, "r") as z:
        for i, orig_idx in enumerate(sel_orig):
            mask_img = read_mask_from_zip(z, orig_idx)
            masks[i] = resize_mask(mask_img, target_hw)
    np.save(out_dir / "instance_masks.npy", masks)

    # 3. Poses: camera-to-world 4x4
    poses = np.empty((num_frames, 4, 4), dtype=np.float32)
    for i, orig_idx in enumerate(sel_orig):
        p_path = pose_dir / f"{orig_idx}.txt"
        if not p_path.is_file():
            # Some ScanNet scenes have missing/-inf poses; fill identity as a safe default.
            poses[i] = np.eye(4, dtype=np.float32)
            continue
        p = np.loadtxt(p_path, dtype=np.float32)
        if p.shape != (4, 4) or not np.all(np.isfinite(p)):
            poses[i] = np.eye(4, dtype=np.float32)
        else:
            poses[i] = p
    np.save(out_dir / "poses.npy", poses)

    # 4. Intrinsic
    if intrinsic_color_file.is_file():
        shutil.copy(intrinsic_color_file, out_dir / "intrinsic.txt")

    # 5. Metadata .sft (mirrors DL3DV convention for VFM extractors)
    save_file(
        {
            "start_idx": torch.tensor(0, dtype=torch.int32),
            "gt_num_frames": torch.tensor(num_frames, dtype=torch.int32),
        },
        str(out_dir / "metadata.sft"),
    )

    return "ok"


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw_root",
        type=Path,
        default=Path("/data24/shz/project/3detr/votenet/scannet/scans"),
        help="Root of raw ScanNet (containing scans_train/ and scans_val/).",
    )
    p.add_argument(
        "--out_root",
        type=Path,
        default=Path("data/ScanNet/ScanNet-processed"),
    )
    p.add_argument("--split", choices=["train", "val", "both"], default="both")
    p.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    p.add_argument("--target_h", type=int, default=DEFAULT_TARGET_H)
    p.add_argument("--target_w", type=int, default=DEFAULT_TARGET_W)
    p.add_argument("--min_frames", type=int, default=DEFAULT_MIN_FRAMES,
                   help="Skip scenes with fewer exported frames than this.")
    p.add_argument("--window_ratio", type=float, default=DEFAULT_WINDOW_RATIO,
                   help="Fraction of exported frames (centered) to source the 81 clip from. "
                        "1.0 = full scene, 0.5 = center half (denser temporal sampling).")
    p.add_argument("--min_window_frames", type=int, default=DEFAULT_MIN_WINDOW_FRAMES,
                   help="If the ratio-window has fewer frames than this, fall back to full scene. "
                        "Ensures short scenes keep full coverage.")
    p.add_argument(
        "--num_scenes", type=int, default=None,
        help="Max scenes per split (for quick tests). Default: process all.",
    )
    p.add_argument("--start", type=int, default=0, help="Start scene index.")
    p.add_argument("--overwrite", action="store_true", help="Reprocess even if metadata.sft exists.")
    p.add_argument("--scene_id", type=str, default=None, help="Process a single scene by ID (debug).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    splits = ["train", "val"] if args.split == "both" else [args.split]
    target_hw = (args.target_h, args.target_w)

    total = {"ok": 0, "skipped": 0, "missing": 0, "short": 0}
    wall_start = time.time()

    for split in splits:
        scenes = discover_scenes(args.raw_root, split)
        if args.scene_id is not None:
            scenes = [s for s in scenes if s.name == args.scene_id]
        else:
            if args.start > 0:
                scenes = scenes[args.start:]
            if args.num_scenes is not None:
                scenes = scenes[: args.num_scenes]

        logger.info(f"[{split}] processing {len(scenes)} scenes → {args.out_root}/{split}")

        for i, scene_dir in enumerate(scenes):
            out_dir = args.out_root / split / scene_dir.name
            t0 = time.time()
            try:
                status = process_scene(
                    scene_dir,
                    out_dir,
                    split=split,
                    num_frames=args.num_frames,
                    target_hw=target_hw,
                    min_frames=args.min_frames,
                    window_ratio=args.window_ratio,
                    min_window_frames=args.min_window_frames,
                    skip_existing=not args.overwrite,
                )
            except Exception as e:
                logger.error(f"[{split}] {scene_dir.name}: FAILED ({type(e).__name__}: {e})")
                continue

            total[status] = total.get(status, 0) + 1
            dt = time.time() - t0
            if args.verbose or (i % 10 == 0 and status == "ok"):
                logger.info(
                    f"[{split} {i+1}/{len(scenes)}] {scene_dir.name}: {status} ({dt:.1f}s)"
                )

    elapsed = time.time() - wall_start
    logger.info(
        f"Done in {elapsed:.1f}s. Stats: "
        + ", ".join(f"{k}={v}" for k, v in total.items())
    )


if __name__ == "__main__":
    main()
