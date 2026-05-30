"""ScanNet instance-probe dataset (Setup A).

Returns for each __getitem__:
    vfm_feat:       (T_feat, H_f, W_f, C) if feat_pixalign=False, else
                    (num_views, H_f, W_f, C)   (already indexed by vfm_idx)
    vfm_idx:        (num_views,) long  — feat-frame index for each sampled view
    instance_masks: (num_views, H_m, W_m) int64  — multi-view consistent IDs
    valid_mask:     (num_views, H_m, W_m) bool   — exclude ignore_ids
    images:         (num_views, 3, H_im, W_im) uint8 (optional, for viz)
    vfm_name:       str
    scene_id:       str

Unlike VideoProbeDataset we do not load pmap/dmap/cmap/pose/intrinsic — Setup A
is instance-only and those fields would be dead weight that inflates IO.

Sampling logic for frame indices mirrors VideoProbeDataset._sample_query_frames
(same min-gap slack allocation). VFM-idx mapping for WAN/OpenSora/VLM is also
identical — we only need the 'wan', 'opensora', 'internvl*', 'qwen3vl*' paths
since cogvideox/aether/vjepa are irrelevant for the VLM vs Video-Model
comparison in Setup A (can be added later if needed).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from probing_vlm_vgm.data.components._vfm_feat_mixin import VFMFeatureLoaderMixin
from probing_vlm_vgm.dust3r.datasets.base.easy_dataset import EasyDataset

logger = logging.getLogger(__name__)


class ScanNetInstanceDataset(VFMFeatureLoaderMixin, EasyDataset):
    """Instance-probe dataset backed by ScanNet-processed/{split}/<scene>/ scenes.

    VFM feature loading + frame sampling come from VFMFeatureLoaderMixin
    (shared with ScanNetTaggingDataset). This class only adds the
    instance-mask + valid-mask logic specific to Exp-C.
    """

    def __init__(
        self,
        root: str,
        root_vfm: str,
        split: str = "train",
        subset=None,  # str ('train'/'val') or list or None → defaults to [split]
        vfm_name: str = "wan",
        feat_postfix: str = "_t749_layer20",
        feat_pixalign: bool = True,
        num_views: int = 8,
        min_view_interval: Optional[int] = 5,
        context_len: int = 76,
        query_idx_divisor: Optional[int] = 4,
        target_spatial_size: Optional[Sequence[int]] = None,
        pool_in_worker: bool = True,
        ignore_ids: Sequence[int] = (0,),
        load_images: bool = False,
        seed: Optional[int] = None,
        gt_num_frames: int = 81,
        **kwargs,
    ):
        if vfm_name not in self.SUPPORTED_VFMS:
            raise ValueError(
                f"Unknown vfm_name={vfm_name!r}; supported: {self.SUPPORTED_VFMS}"
            )
        self.vfm_name = vfm_name
        self.feat_postfix = feat_postfix
        self.feat_pixalign = feat_pixalign
        self.num_views = num_views
        self.min_view_interval = min_view_interval
        self.context_len = context_len
        self.query_idx_divisor = query_idx_divisor
        self.target_spatial_size = (
            tuple(target_spatial_size) if target_spatial_size is not None else None
        )
        # If True AND target_spatial_size is set, the worker pools (H, W) →
        # target_hw on CPU before returning. Cuts feature-tensor payload 4x
        # for opensora (3072-dim hidden) — shrinks every downstream step
        # (worker contiguous, shared-mem transfer, pin_memory, H2D, GPU
        # permute+pool) proportionally. ~60 ms/sample CPU pool is hidden by
        # 8-worker parallelism. Real training throughput 2.04x on opensora
        # train split (scripts/test_pool_in_worker.py). Default True since
        # the model.forward's (H, W) != target_hw branch makes pre-pooled
        # input a GPU-side no-op — no caller has to change anything.
        self.pool_in_worker = bool(pool_in_worker)
        self.ignore_ids = set(int(x) for x in ignore_ids)
        self.load_images = load_images
        self.seed = seed
        self.split = split
        self.root = root
        self.root_vfm = root_vfm
        self.gt_num_frames = gt_num_frames
        self.kwargs = kwargs

        # Resolve subset list (filter on the first column of {split}.json pairs).
        if subset is None:
            subset_list = [split]
        elif isinstance(subset, str):
            subset_list = [subset]
        else:
            subset_list = list(subset)
        self._subset_set = set(subset_list)

        # Load split JSON (format matches DL3DV: [[subset, scene_id], ...]).
        split_file = os.path.join(root, f"{split}.json")
        with open(split_file, "r") as f:
            pairs = json.load(f)

        self.scenes = []  # list of (subset, scene_id, scene_dir)
        for sub, scene_id in pairs:
            if sub not in self._subset_set:
                continue
            scene_dir = os.path.join(root, sub, scene_id)
            if os.path.isfile(os.path.join(scene_dir, "metadata.sft")):
                self.scenes.append((sub, scene_id, scene_dir))
        logger.info(
            f"ScanNetInstanceDataset: found {len(self.scenes)} scenes in {split_file}"
        )

    def __len__(self):
        return len(self.scenes)

    def get_stats(self):
        return f"{len(self)} scenes"

    def __repr__(self):
        return (
            f"{type(self).__name__}({self.get_stats()}, split={self.split!r}, "
            f"vfm={self.vfm_name!r}, num_views={self.num_views})"
        )

    # _sample_query_frames, _load_vfm_feat, _load_vlm_feat_layers are
    # inherited from VFMFeatureLoaderMixin and shared with the tagging
    # dataset. Anything instance-specific lives below.

    # ---------------------------------------------------------------- #
    # __getitem__                                                      #
    # ---------------------------------------------------------------- #
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

        # 2) Load instance masks for the selected frames (memory-map the full array).
        masks_path = os.path.join(scene_dir, "instance_masks.npy")
        masks_all = np.load(masks_path, mmap_mode="r")  # (81, H, W) uint16
        masks_sel = np.array(masks_all[sel.numpy()])  # copy out of mmap
        instance_masks = torch.from_numpy(masks_sel.astype(np.int64))  # (S, H, W)

        # 3) Build valid_mask (exclude ignore_ids, e.g. ID 0 background/unannotated)
        if self.ignore_ids:
            valid_mask = torch.ones_like(instance_masks, dtype=torch.bool)
            for ig in self.ignore_ids:
                valid_mask &= instance_masks != ig
        else:
            valid_mask = torch.ones_like(instance_masks, dtype=torch.bool)

        # 4) Load VFM feature
        vfm_feat_path = os.path.join(
            self.root_vfm,
            self.vfm_name,
            subset,
            scene_id,
            f"feature{self.feat_postfix}.sft",
        )
        vfm_feat, vfm_idx = self._load_vfm_feat(vfm_feat_path, sel)

        # 5) CPU-side adaptive pool to target_spatial_size (on by default).
        # For wide-channel features (e.g. opensora C=3072) shipping the
        # un-pooled tensor through worker→main→GPU dominates throughput;
        # pre-pooling here cuts that payload ~4x and gives ~2x end-to-end
        # training speedup. The historic comment that CPU pool was "the
        # dominant per-worker cost" predates feat_pixalign (which already
        # slices to num_views frames before we get here), so this pool
        # operates on 8 frames not 21 — well within the budget that 8
        # parallel workers can absorb.
        if (
            self.pool_in_worker
            and self.target_spatial_size is not None
            and vfm_feat.ndim == 4  # (S, H, W, C)
            and tuple(vfm_feat.shape[1:3]) != self.target_spatial_size
        ):
            S, H, W, C = vfm_feat.shape
            target_h, target_w = self.target_spatial_size
            if target_h <= H and target_w <= W:
                orig_dtype = vfm_feat.dtype
                x = vfm_feat.permute(0, 3, 1, 2).float()  # (S, C, H, W)
                x = F.adaptive_avg_pool2d(x, (target_h, target_w))
                vfm_feat = x.permute(0, 2, 3, 1).contiguous().to(orig_dtype)

        # 5b) Pool defers to model.forward when pool_in_worker=False (default).
        output = {
            "vfm_feat": vfm_feat.contiguous(),
            "vfm_idx": vfm_idx,
            "instance_masks": instance_masks,
            "valid_mask": valid_mask,
            "vfm_name": self.vfm_name,
            "scene_id": scene_id,
        }
        if self.target_spatial_size is not None:
            output["target_spatial_size"] = torch.tensor(
                self.target_spatial_size, dtype=torch.long
            )

        # 6) Optional image loading (only for viz / debugging)
        if self.load_images:
            frames_dir = os.path.join(scene_dir, "frames")
            imgs = []
            for s in sel.tolist():
                img = Image.open(os.path.join(frames_dir, f"frame_{s:05d}.jpg")).convert("RGB")
                imgs.append(torch.from_numpy(np.array(img)).permute(2, 0, 1))  # (3,H,W) uint8
            output["images"] = torch.stack(imgs, dim=0)

        return output

    # Per-VFM feature loading + vfm_idx mapping inherited from
    # VFMFeatureLoaderMixin — see data/components/_vfm_feat_mixin.py.
