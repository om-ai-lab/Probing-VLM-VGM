"""Build video-level multi-label tag GT for Exp-B (semantic tagging).

For each processed scene under ``<processed_root>/<split>/<scene>/`` we produce:

    tag_pixel_counts_{num_classes}.npy   (T, C) int32 — per-frame pixel count
                                          per class, T = clip length (e.g. 81).
                                          THIS is the source of truth that
                                          the tagging dataset reads at training
                                          time, so it can apply the same two
                                          thresholds against the SAMPLED views
                                          (not the full clip). Solves the
                                          81-frame-label / 8-frame-input
                                          label-noise issue (research_plan §5.2.3).

    tag_labels_{num_classes}.npy         (C,) uint8 — clip-level multi-label.
                                          y_c = 1 iff class c passes BOTH
                                          ``min_pixels_per_video`` and
                                          ``min_frames_present`` thresholds
                                          over the FULL clip. Kept for
                                          debugging + train_pos_rate; the
                                          dataset does NOT use this.

Class assignment uses the multi-view-consistent ``instance_masks.npy`` (already
produced by ``process_scannet.py``) plus the per-scene ``aggregation.json``
mapping (objectId → raw class label) plus the official ScanNet200 / 20
taxonomy mapping (scannet_constants.py).

After processing both splits, also writes:
    <processed_root>/train_pos_rate_{num_classes}.npy   (C,) float32 — per-class
                                                         positive rate over train
    <processed_root>/class_names_{num_classes}.json     [str, ...] length C
    <processed_root>/coverage_stats_{num_classes}.json  diagnostics (per-class
                                                         positive scene count for
                                                         train and val)

Example:
    python -m probing_vlm_vgm.data.processing.scannet.build_tag_labels \\
        --raw_root /data24/shz/project/3detr/votenet/scannet/scans \\
        --processed_root data/ScanNet/ScanNet-processed \\
        --label_map_tsv /data24/shz/project/3detr/votenet/scannet/meta_data/scannetv2-labels.combined.tsv \\
        --num_classes 200 \\
        --split both
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from probing_vlm_vgm.data.processing.scannet.scannet_constants import (
    build_raw_to_class_map,
    get_class_labels,
)

logger = logging.getLogger(__name__)

DEFAULT_MIN_PIXELS = 200
DEFAULT_MIN_FRAMES = 2


def load_instance_to_raw_label(agg_path: Path) -> Dict[int, str]:
    """Build {instance_mask_pixel_value → raw_category_str} from a scene's aggregation.json.

    ScanNet convention:
        - aggregation.json's segGroups[i].objectId is 0-indexed
        - 2d-instance-filt masks store 1-indexed values (0 = background/ignore)
      → mask pixel value v ≥ 1 corresponds to objectId = v - 1.
    """
    with open(agg_path, "r") as f:
        agg = json.load(f)
    inst_to_raw: Dict[int, str] = {}
    for seg in agg.get("segGroups", []):
        # objectId is 0-indexed in the JSON, +1 to match mask pixel value space
        inst_to_raw[int(seg["objectId"]) + 1] = str(seg["label"])
    return inst_to_raw


def compute_per_frame_pixel_counts(
    masks: np.ndarray,                # (T, H, W) uint16, instance IDs (0 = bg)
    inst_to_raw: Dict[int, str],      # mask value → raw label
    raw_to_class: Dict[str, int],     # raw label → class index ∈ [0, C)
    num_classes: int,
) -> np.ndarray:
    """Per-frame pixel count for each class.

    This is the **canonical** scene-level statistic. The dataset reads it and
    applies thresholds against the SAMPLED frames at training time; the
    `aggregate_to_clip_label` helper applies the same thresholds against the
    full clip for the debug / train_pos_rate path.

    Returns:
        counts: (T, C) int32 — counts[t, c] = pixels of class c in frame t.
    """
    T = masks.shape[0]
    counts = np.zeros((T, num_classes), dtype=np.int32)

    for t in range(T):
        # Unique pixel values + counts within this frame
        vals, c = np.unique(masks[t], return_counts=True)
        for v, n in zip(vals.tolist(), c.tolist()):
            if v == 0:
                continue  # background / ignore
            raw_label = inst_to_raw.get(int(v))
            if raw_label is None:
                continue  # mask has this instance ID but aggregation.json doesn't
            cls = raw_to_class.get(raw_label)
            if cls is None:
                continue  # raw label not in chosen taxonomy
            counts[t, cls] += int(n)

    return counts


def aggregate_to_clip_label(
    counts: np.ndarray,                # (T, C) int — output of compute_per_frame_pixel_counts
    min_pixels_per_video: int = DEFAULT_MIN_PIXELS,
    min_frames_present: int = DEFAULT_MIN_FRAMES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the two thresholds to a (T, C) count tensor → clip-level y.

    The dataset calls this same function at training time on `counts[sel]`
    (rows for the sampled views only). This module also calls it on the full
    counts during preprocessing to produce the clip-level debug label.
    """
    total_pixels = counts.sum(axis=0)                       # (C,)
    frames_present = (counts > 0).sum(axis=0)               # (C,)
    y = (
        (total_pixels >= min_pixels_per_video)
        & (frames_present >= min_frames_present)
    ).astype(np.uint8)
    return y, total_pixels, frames_present


# Back-compat wrapper for the existing test signature.
def compute_tag_labels(
    masks: np.ndarray,
    inst_to_raw: Dict[int, str],
    raw_to_class: Dict[str, int],
    num_classes: int,
    min_pixels_per_video: int = DEFAULT_MIN_PIXELS,
    min_frames_present: int = DEFAULT_MIN_FRAMES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper: per-frame counts + threshold aggregation."""
    counts = compute_per_frame_pixel_counts(
        masks, inst_to_raw, raw_to_class, num_classes
    )
    return aggregate_to_clip_label(counts, min_pixels_per_video, min_frames_present)


def process_scene(
    scene_dir: Path,
    raw_scene_dir: Path,
    raw_to_class: Dict[str, int],
    num_classes: int,
    min_pixels_per_video: int,
    min_frames_present: int,
    skip_existing: bool,
) -> Tuple[str, Optional[np.ndarray]]:
    """Process one scene. Returns (status, y_or_None).

    Writes two files:
        tag_pixel_counts_{C}.npy  (T, C) int32  — canonical, dataset reads this
        tag_labels_{C}.npy        (C,)  uint8   — clip-level debug label

    status ∈ {'ok', 'skipped', 'no_agg', 'no_mask', 'all_zero'}.
    """
    scene_id = scene_dir.name
    counts_path = scene_dir / f"tag_pixel_counts_{num_classes}.npy"
    label_path = scene_dir / f"tag_labels_{num_classes}.npy"

    # Sentinel for "fully processed" is the per-frame counts file — it's the
    # new canonical output. If counts exist but tag_labels doesn't (re-running
    # after a code change), we still re-aggregate.
    if skip_existing and counts_path.is_file() and label_path.is_file():
        return "skipped", np.load(label_path)

    masks_path = scene_dir / "instance_masks.npy"
    if not masks_path.is_file():
        return "no_mask", None

    agg_path = raw_scene_dir / f"{scene_id}.aggregation.json"
    if not agg_path.is_file():
        return "no_agg", None

    inst_to_raw = load_instance_to_raw_label(agg_path)
    masks = np.load(masks_path, mmap_mode="r")

    # 1) Per-frame pixel counts (canonical — dataset reads this).
    counts = compute_per_frame_pixel_counts(
        masks=np.asarray(masks),  # materialize the mmap slice once
        inst_to_raw=inst_to_raw,
        raw_to_class=raw_to_class,
        num_classes=num_classes,
    )
    np.save(counts_path, counts)

    # 2) Clip-level label (debug + train_pos_rate). Same thresholds applied
    # over the FULL clip, not over sampled frames — slight over-estimate vs
    # what the dataset will compute at training time, but used only for
    # head/mid/tail bucketing (coarse ranking, robust to this slack).
    y, _, _ = aggregate_to_clip_label(
        counts, min_pixels_per_video, min_frames_present
    )
    np.save(label_path, y)

    status = "all_zero" if y.sum() == 0 else "ok"
    return status, y


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw_root",
        type=Path,
        default=Path("/data24/shz/project/3detr/votenet/scannet/scans"),
        help="Raw ScanNet root (contains scans_train/ and scans_val/).",
    )
    p.add_argument(
        "--processed_root",
        type=Path,
        default=Path("data/ScanNet/ScanNet-processed"),
        help="Where process_scannet.py wrote instance_masks.npy etc.",
    )
    p.add_argument(
        "--label_map_tsv",
        type=Path,
        default=Path("/data24/shz/project/3detr/votenet/scannet/meta_data/scannetv2-labels.combined.tsv"),
    )
    p.add_argument("--num_classes", type=int, choices=[20, 200], default=200)
    p.add_argument("--min_pixels_per_video", type=int, default=DEFAULT_MIN_PIXELS)
    p.add_argument("--min_frames_present", type=int, default=DEFAULT_MIN_FRAMES)
    p.add_argument("--split", choices=["train", "val", "both"], default="both")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even if tag_labels_*.npy already exists.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    splits = ["train", "val"] if args.split == "both" else [args.split]

    raw_to_class = build_raw_to_class_map(str(args.label_map_tsv), args.num_classes)
    logger.info(
        f"Loaded {len(raw_to_class)} raw_category → class mappings "
        f"(num_classes={args.num_classes})"
    )

    # Collected per-split stats for the post-loop aggregation
    train_labels: List[np.ndarray] = []   # accumulate to build train_pos_rate
    coverage: Dict[str, np.ndarray] = {}  # per-split (C,) counts of positive scenes

    wall_start = time.time()

    for split in splits:
        split_dir = args.processed_root / split
        if not split_dir.is_dir():
            logger.warning(f"Processed split dir missing: {split_dir}")
            continue
        raw_split_dir = args.raw_root / f"scans_{split}"

        scenes = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        logger.info(f"[{split}] processing {len(scenes)} scenes")

        counts = {"ok": 0, "skipped": 0, "all_zero": 0, "no_agg": 0, "no_mask": 0}
        pos_count = np.zeros(args.num_classes, dtype=np.int64)

        for i, scene_dir in enumerate(scenes):
            raw_scene_dir = raw_split_dir / scene_dir.name
            t0 = time.time()
            try:
                status, y = process_scene(
                    scene_dir=scene_dir,
                    raw_scene_dir=raw_scene_dir,
                    raw_to_class=raw_to_class,
                    num_classes=args.num_classes,
                    min_pixels_per_video=args.min_pixels_per_video,
                    min_frames_present=args.min_frames_present,
                    skip_existing=not args.overwrite,
                )
            except Exception as e:  # noqa: BLE001
                logger.error(f"[{split}] {scene_dir.name}: FAILED {type(e).__name__}: {e}")
                continue

            counts[status] = counts.get(status, 0) + 1
            if y is not None:
                pos_count += y.astype(np.int64)
                if split == "train":
                    train_labels.append(y)

            dt = time.time() - t0
            if args.verbose or i % 50 == 0:
                pos_classes = int(y.sum()) if y is not None else 0
                logger.info(
                    f"[{split} {i+1}/{len(scenes)}] {scene_dir.name}: "
                    f"{status} (pos={pos_classes}/{args.num_classes}, {dt:.2f}s)"
                )

        logger.info(f"[{split}] stats: {counts}")
        coverage[split] = pos_count

    # --------------------------- post-aggregation ---------------------- #
    # 1) train_pos_rate.npy — per-class positive rate over the train split
    if train_labels:
        train_arr = np.stack(train_labels, axis=0).astype(np.float32)  # (N, C)
        train_pos_rate = train_arr.mean(axis=0)
        out_rate = args.processed_root / f"train_pos_rate_{args.num_classes}.npy"
        np.save(out_rate, train_pos_rate)
        logger.info(f"Wrote {out_rate}: shape={train_pos_rate.shape}, "
                    f"min={train_pos_rate.min():.4f}, max={train_pos_rate.max():.4f}")

    # 2) class_names.json
    class_names = list(get_class_labels(args.num_classes))
    out_names = args.processed_root / f"class_names_{args.num_classes}.json"
    with open(out_names, "w") as f:
        json.dump({"num_classes": args.num_classes, "names": class_names}, f, indent=2)
    logger.info(f"Wrote {out_names}")

    # 3) coverage_stats.json — diagnostics for threshold tuning
    cov: Dict[str, List[int]] = {k: v.tolist() for k, v in coverage.items()}
    cov["class_names"] = class_names
    cov["thresholds"] = {
        "min_pixels_per_video": args.min_pixels_per_video,
        "min_frames_present": args.min_frames_present,
    }
    out_cov = args.processed_root / f"coverage_stats_{args.num_classes}.json"
    with open(out_cov, "w") as f:
        json.dump(cov, f, indent=2)
    logger.info(f"Wrote {out_cov}")

    elapsed = time.time() - wall_start
    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
