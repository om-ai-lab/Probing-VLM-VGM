"""ScanNet tagging-probe dataset (Exp-B: video-level multi-label).

Returns for each __getitem__:
    vfm_feat:    (num_views, H_f, W_f, C) if feat_pixalign=True, else
                 (T_feat, H_f, W_f, C)
    vfm_idx:     (num_views,) long — feat-frame index per sampled view
    tag_labels:  (num_classes,) float32 ∈ {0, 1} — computed **online** by
                 indexing ``tag_pixel_counts_{C}.npy[sel]`` and applying the
                 two thresholds against the SAMPLED views (not the full 81
                 frames). This avoids the label-noise pathology where a class
                 visible only in frames 30-40 gets labelled positive but the
                 model receives 8 sampled views that miss it entirely.
    vfm_name:    str
    scene_id:    str

Unlike ScanNetInstanceDataset this dataset does NOT load:
    - instance_masks.npy (we use tag_labels_*.npy directly)
    - valid_mask, poses, intrinsic, images
which keeps per-sample IO light and lets us run larger batch sizes than the
instance line.

VFM feature loading + frame sampling are inherited from
VFMFeatureLoaderMixin (shared with the instance dataset).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from probing_vlm_vgm.data.components._vfm_feat_mixin import VFMFeatureLoaderMixin
from probing_vlm_vgm.dust3r.datasets.base.easy_dataset import EasyDataset

logger = logging.getLogger(__name__)


class ScanNetTaggingDataset(VFMFeatureLoaderMixin, EasyDataset):
    """Tagging-probe dataset backed by ScanNet-processed/{split}/<scene>/."""

    def __init__(
        self,
        root: str,
        root_vfm: str,
        split: str = "train",
        subset=None,  # str / list / None — same convention as instance dataset
        vfm_name: str = "wan",
        feat_postfix: str = "_t749_layer20",
        feat_pixalign: bool = True,
        num_views: int = 8,
        min_view_interval: Optional[int] = 5,
        context_len: int = 76,
        query_idx_divisor: Optional[int] = 4,
        target_spatial_size: Optional[Sequence[int]] = None,
        pool_in_worker: bool = True,
        seed: Optional[int] = None,
        gt_num_frames: int = 81,
        num_classes: int = 200,
        # Online thresholding — applied against the N sampled views, NOT the
        # full 81 frames. Same numeric defaults as the preprocessor's
        # full-clip label aggregation, but the operational meaning differs:
        # here they say "the class must be robustly visible in the model's
        # input"; in the preprocessor they say "the class is present in the
        # scene overall". See research_plan v3 §5.2.3.
        min_pixels_per_video: int = 200,
        min_frames_present: int = 1,
        **kwargs,
    ):
        if vfm_name not in self.SUPPORTED_VFMS:
            raise ValueError(
                f"Unknown vfm_name={vfm_name!r}; supported: {self.SUPPORTED_VFMS}"
            )
        if num_classes not in (20, 200):
            raise ValueError(f"num_classes must be 20 or 200, got {num_classes}")

        # Attributes consumed by the mixin
        self.vfm_name = vfm_name
        self.feat_postfix = feat_postfix
        self.feat_pixalign = feat_pixalign
        self.num_views = num_views
        self.min_view_interval = min_view_interval
        self.context_len = context_len
        self.query_idx_divisor = query_idx_divisor
        self.gt_num_frames = gt_num_frames

        # Local config
        self.target_spatial_size = (
            tuple(target_spatial_size) if target_spatial_size is not None else None
        )
        self.pool_in_worker = bool(pool_in_worker)
        self.seed = seed
        self.split = split
        self.root = root
        self.root_vfm = root_vfm
        self.num_classes = int(num_classes)
        self.min_pixels_per_video = int(min_pixels_per_video)
        self.min_frames_present = int(min_frames_present)
        self.kwargs = kwargs

        # Resolve subset (mirrors instance dataset for consistency)
        if subset is None:
            subset_list = [split]
        elif isinstance(subset, str):
            subset_list = [subset]
        else:
            subset_list = list(subset)
        self._subset_set = set(subset_list)

        split_file = os.path.join(root, f"{split}.json")
        with open(split_file, "r") as f:
            pairs = json.load(f)

        # Scene filtering: must have metadata.sft AND
        # tag_pixel_counts_{C}.npy (the per-frame counts needed for online
        # thresholding). Scenes missing the counts file are skipped with a
        # rank-aware log line so the user knows to rerun build_tag_labels.py.
        counts_filename = f"tag_pixel_counts_{self.num_classes}.npy"
        self.scenes: list = []
        n_seen, n_missing_tag = 0, 0
        for sub, scene_id in pairs:
            if sub not in self._subset_set:
                continue
            n_seen += 1
            scene_dir = os.path.join(root, sub, scene_id)
            if not os.path.isfile(os.path.join(scene_dir, "metadata.sft")):
                continue
            if not os.path.isfile(os.path.join(scene_dir, counts_filename)):
                n_missing_tag += 1
                continue
            self.scenes.append((sub, scene_id, scene_dir))

        logger.info(
            f"ScanNetTaggingDataset: {len(self.scenes)} usable scenes in "
            f"{split_file} (num_classes={self.num_classes}, "
            f"missing-tag={n_missing_tag}/{n_seen})"
        )

    def __len__(self):
        return len(self.scenes)

    def get_stats(self):
        return f"{len(self)} scenes (C={self.num_classes})"

    def __repr__(self):
        return (
            f"{type(self).__name__}({self.get_stats()}, split={self.split!r}, "
            f"vfm={self.vfm_name!r}, num_views={self.num_views})"
        )

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx):
        # Deterministic per-index RNG for val; per-worker RNG for train.
        if self.seed is not None:
            rng = np.random.default_rng(self.seed + idx)
        elif not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(torch.initial_seed())
            rng = self._rng
        else:
            rng = self._rng

        subset, scene_id, scene_dir = self.scenes[idx]

        # 1) Sample frame indices within the valid context window
        effective_context_len = min(self.context_len, self.gt_num_frames)
        sel = self._sample_query_frames(rng, self.num_views, effective_context_len, 0)

        if self.query_idx_divisor is not None:
            sel = (
                torch.floor((sel - 1) / self.query_idx_divisor) * self.query_idx_divisor
                + 1
            )
            sel = sel.clamp(min=0).long()

        # 2) Compute video-level tag labels ONLINE from per-frame counts.
        # The preprocessor stored (T, C) int32 counts; we slice the rows
        # corresponding to the sampled views, apply both thresholds, and
        # produce a (C,) {0,1} label that reflects what's actually visible
        # to the model rather than what's in the entire 81-frame clip.
        counts_path = os.path.join(
            scene_dir, f"tag_pixel_counts_{self.num_classes}.npy"
        )
        counts_all = np.load(counts_path, mmap_mode="r")           # (T, C) int32
        counts_sel = np.asarray(counts_all[sel.numpy()])           # (S, C)
        total_pixels = counts_sel.sum(axis=0)                       # (C,)
        frames_present = (counts_sel > 0).sum(axis=0)               # (C,)
        y = (
            (total_pixels >= self.min_pixels_per_video)
            & (frames_present >= self.min_frames_present)
        ).astype(np.float32)
        tag_labels = torch.from_numpy(y)
        assert tag_labels.shape == (self.num_classes,), (
            f"tag_pixel_counts at {counts_path} has C={counts_all.shape[1]}, "
            f"expected {self.num_classes}"
        )

        # 3) Load VFM feature using the shared mixin path.
        vfm_feat_path = os.path.join(
            self.root_vfm,
            self.vfm_name,
            subset,
            scene_id,
            f"feature{self.feat_postfix}.sft",
        )
        vfm_feat, vfm_idx = self._load_vfm_feat(vfm_feat_path, sel)

        # 4) CPU-side adaptive pool — mirrors instance dataset for parity.
        if (
            self.pool_in_worker
            and self.target_spatial_size is not None
            and vfm_feat.ndim == 4
            and tuple(vfm_feat.shape[1:3]) != self.target_spatial_size
        ):
            S, H, W, C = vfm_feat.shape
            target_h, target_w = self.target_spatial_size
            if target_h <= H and target_w <= W:
                orig_dtype = vfm_feat.dtype
                x = vfm_feat.permute(0, 3, 1, 2).float()
                x = F.adaptive_avg_pool2d(x, (target_h, target_w))
                vfm_feat = x.permute(0, 2, 3, 1).contiguous().to(orig_dtype)

        output = {
            "vfm_feat": vfm_feat.contiguous(),
            "vfm_idx": vfm_idx,
            "tag_labels": tag_labels,
            "vfm_name": self.vfm_name,
            "scene_id": scene_id,
        }
        if self.target_spatial_size is not None:
            output["target_spatial_size"] = torch.tensor(
                self.target_spatial_size, dtype=torch.long
            )
        return output
