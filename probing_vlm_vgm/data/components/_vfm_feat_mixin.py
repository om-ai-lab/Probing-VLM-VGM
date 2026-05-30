"""Shared VFM feature loading + frame sampling logic for ScanNet probe datasets.

Both ScanNetInstanceDataset (Exp-C) and ScanNetTaggingDataset (Exp-B) consume
frozen VFM features the same way:
  - sample N frames from the 81-frame clip with optional min-gap
  - load the per-VFM .sft feature file
  - map sampled frame indices → feature indices using VFM-specific layout
    (different for video diffusion vs VLM vs vjepa vs dino)

This mixin centralizes that logic. Subclasses must set the following
attributes (typically in __init__):

    self.vfm_name          : str
    self.feat_postfix      : str
    self.feat_pixalign     : bool
    self.context_len       : int
    self.query_idx_divisor : Optional[int]
    self.min_view_interval : Optional[int]
    self.gt_num_frames     : int

The mixin provides:
    SUPPORTED_VFMS         : tuple of supported VFM names
    _sample_query_frames   : produce N frame indices with min-gap
    _load_vfm_feat         : dispatch to per-VFM loader + idx mapping
    _load_vlm_feat_layers  : concat multi-layer VLM features

Keep this file behavior-identical to the original implementation in
``scannet_instance_dataset.py``; the refactor is purely structural.
"""
from __future__ import annotations

import math
import os
from typing import Tuple

import numpy as np
import torch
from safetensors.torch import load_file


class VFMFeatureLoaderMixin:
    SUPPORTED_VFMS: Tuple[str, ...] = (
        "wan-t2v-1.3b", "wan-t2v-14b", "wan-i2v-14b", "opensora",
        "internvl3-1b", "internvl3-2b", "internvl3-8b", "internvl3.5-4b", "internvl3.5-8b", "internvl3-8b-sensenova",
        "qwen3-vl-8b", "qwen3-vl-8b-sensenova", "qwen3-vl-4b", "qwen3-vl-2b",
        "qwen2.5-vl-7b", "qwen2.5-vl-3b", "videollama3-7b", "videollama3-2b",
        "llavaov15-4b", "llavaov15-8b", "mimo-vl-7b",
        "wan-t2v-14b-qwen3-vl-8b-concat", "wan-t2v-14b-qwen3-vl-8b-lnconcat",
        "cogvideox-i2v-5b", "cogvideox-t2v-2b", "cogvideox-t2v-5b", "aether",
        "vjepa",
        "dino",
    )

    # ------------------------------------------------------------------ #
    # Frame sampling
    # ------------------------------------------------------------------ #
    def _sample_query_frames(self, rng, n: int, win_len: int, local_start: int = 0) -> torch.Tensor:
        """Sample N frame indices in [local_start, local_start+win_len)."""
        min_gap = getattr(self, "min_view_interval", None) or 0
        if min_gap <= 0:
            return (
                torch.linspace(
                    local_start, local_start + win_len - 1, n, dtype=torch.float32
                )
                .round()
                .to(torch.long)
            )
        needed = (n - 1) * min_gap + 1
        if needed > win_len:
            raise ValueError(
                f"Cannot sample {n} views with min_gap={min_gap} in win_len={win_len}"
            )
        slack = win_len - needed
        cuts = np.sort(rng.integers(0, slack + 1, size=n - 1, dtype=int))
        extras = np.diff(np.concatenate(([0], cuts, [slack])))
        idxs = [local_start]
        for extra in extras[:-1]:
            idxs.append(idxs[-1] + min_gap + int(extra))
        assert len(idxs) == n
        return torch.as_tensor(idxs, dtype=torch.long)

    # ------------------------------------------------------------------ #
    # Per-VFM feature loading
    # ------------------------------------------------------------------ #
    def _load_vfm_feat(self, vfm_feat_path: str, sel: torch.Tensor):
        """Load frozen VFM feature and produce (sel → feature-index) mapping.

        Returns (vfm_feat, vfm_idx):
            vfm_feat: (T, H, W, C) if not pixalign else (num_views, H, W, C)
            vfm_idx:  (num_views,)  — feature-index for each sampled view;
                                       if pixalign, becomes arange(num_views).
        """
        if self.vfm_name in (
            "wan-t2v-1.3b", "wan-t2v-14b", "wan-i2v-14b", "opensora",
        ):
            vfm_feat = load_file(vfm_feat_path)["feat"]  # (21, H, W, C)
            if vfm_feat.shape[0] != 21:
                raise RuntimeError(
                    f"Expected 21 feat frames for {self.vfm_name}, got {vfm_feat.shape[0]}"
                )
            if self.context_len < 81:
                end_idx = math.ceil((self.context_len - 1) / 80.0 * 20.0)
                vfm_feat = vfm_feat[: end_idx + 1]
            vfm_idx = (
                torch.floor((sel - 1).float() / 80.0 * 20.0).long() + 1
            )
            vfm_idx = vfm_idx.clamp(min=0, max=vfm_feat.shape[0] - 1)

        elif self.vfm_name in (
            "internvl3-1b", "internvl3-2b", "internvl3-8b", "internvl3.5-4b", "internvl3.5-8b", "internvl3-8b-sensenova",
            "qwen3-vl-8b", "qwen3-vl-8b-sensenova", "qwen3-vl-4b", "qwen3-vl-2b",
            "qwen2.5-vl-7b", "qwen2.5-vl-3b", "videollama3-7b", "videollama3-2b",
            "llavaov15-4b", "llavaov15-8b", "mimo-vl-7b",
            "wan-t2v-14b-qwen3-vl-8b-concat", "wan-t2v-14b-qwen3-vl-8b-lnconcat",
        ):
            # Features stored at query_frame positions
            # [0, 1, 5, 9, ..., context_len) → 20 positions with context_len=76, div=4.
            vfm_feat = self._load_vlm_feat_layers(vfm_feat_path)  # (T, H, W, C_total)
            T = vfm_feat.shape[0]
            divisor = self.query_idx_divisor if self.query_idx_divisor else 4
            query_frame_seq = [0, 1]
            i = 1 + divisor
            while i < self.context_len:
                query_frame_seq.append(i)
                i += divisor
            query_frame_seq = query_frame_seq[:T]
            frame_to_feat = {f: j for j, f in enumerate(query_frame_seq)}
            vfm_idx = torch.tensor(
                [frame_to_feat.get(int(s), 0) for s in sel], dtype=torch.long
            )

        elif self.vfm_name in (
            "cogvideox-i2v-5b", "cogvideox-t2v-2b", "cogvideox-t2v-5b", "aether",
        ):
            vfm_feat_5d = load_file(vfm_feat_path)["feat"]  # (N, T, H, W, C)
            n_frames = vfm_feat_5d.shape[1]
            assert n_frames in (11, 13), (
                f"Expected 11 (Aether) or 13 (CogVideoX) feature frames per chunk, "
                f"got {n_frames} in {vfm_feat_path}"
            )
            n_chunks = vfm_feat_5d.shape[0]
            vfm_feat = vfm_feat_5d.reshape(-1, *vfm_feat_5d.shape[2:])  # (N*T, H, W, C)

            vfm_idx = torch.zeros_like(sel)
            mask0 = sel == 0
            vfm_idx[mask0] = 0
            mask_rest = ~mask0
            if mask_rest.any():
                offs = sel[mask_rest] - 1
                clip_idx = offs % n_chunks
                pos_in_clip = offs // n_chunks + 1
                token_id = (pos_in_clip - 1) // 4 + 1
                vfm_idx[mask_rest] = clip_idx * n_frames + token_id
            vfm_idx = vfm_idx.clamp(min=0, max=vfm_feat.shape[0] - 1)

        elif self.vfm_name == "vjepa":
            vfm_feat_5d = load_file(vfm_feat_path)["feat"]  # (N, 8, H, W, C)
            n_chunks = vfm_feat_5d.shape[0]
            vfm_feat = vfm_feat_5d.reshape(-1, *vfm_feat_5d.shape[2:])  # (N*8, H, W, C)

            vfm_idx = torch.zeros_like(sel)
            mask0 = sel == 0
            vfm_idx[mask0] = 0
            mask_rest = ~mask0
            if mask_rest.any():
                offs = sel[mask_rest] - 1
                clip_idx = offs % n_chunks
                pos_in_clip = offs // n_chunks + 1
                token_id = pos_in_clip // 2  # stride-2 → 0…7
                vfm_idx[mask_rest] = clip_idx * 8 + token_id
            vfm_idx = vfm_idx.clamp(min=0, max=vfm_feat.shape[0] - 1)

        elif self.vfm_name == "dino":
            vfm_feat = load_file(vfm_feat_path)["feat"]  # (T, H, W, C)
            T = vfm_feat.shape[0]
            vfm_idx = (
                torch.floor(sel.float() / (self.gt_num_frames - 1) * (T - 1)).long()
            )
            vfm_idx = vfm_idx.clamp(min=0, max=T - 1)

        else:
            raise NotImplementedError(self.vfm_name)

        if self.feat_pixalign:
            vfm_feat = vfm_feat[vfm_idx]
            vfm_idx = torch.arange(vfm_feat.shape[0])

        return vfm_feat, vfm_idx

    def _load_vlm_feat_layers(self, vfm_feat_path: str) -> torch.Tensor:
        """Handle possibly multi-layer VLM features by concatenating on channel dim."""
        base_dir = os.path.dirname(vfm_feat_path)
        if "layer" in self.feat_postfix:
            layer_str = self.feat_postfix.split("layer")[-1]
            layer_indices = [int(x) for x in layer_str.split("_") if x.isdigit()]
        else:
            layer_indices = []
        if len(layer_indices) <= 1:
            return load_file(vfm_feat_path)["feat"]
        feats = []
        for li in layer_indices:
            p = os.path.join(base_dir, f"feature_layer{li}.sft")
            feats.append(load_file(p)["feat"])
        return torch.cat(feats, dim=-1)
