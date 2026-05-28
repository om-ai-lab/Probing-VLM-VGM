"""Unified probe model — shared BackbonePA + active_heads routing.

Supports five head types covering the three research-plan experiments:

    Exp-A (pure geometry, §5.1):    "depth"        DPTHead
                                     "point"        DPTHead
                                     "camera"       CameraHead
    Exp-C (mid: instance, §5.3):     "instance"     InstanceHead (1×1 Conv)
    Exp-B (pure semantics, §5.2):    "semantic_tag" SemanticTagHead (DETR-style)

All heads share the same frozen-VFM-feature input and the same BackbonePA
(N-layer alternating frame+global attention). Setting `backbone_depth=0`
turns the backbone into pure input_proj + special-token concat — this is
the read-out-purity ablation (Abl-1).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from probing_vlm_vgm.models.components.backbone_pixalign import BackbonePA
from probing_vlm_vgm.models.components.dpt_head import DPTHead
from probing_vlm_vgm.models.components.instance_head import InstanceHead, InstanceHeadMLP
from probing_vlm_vgm.models.components.semantic_tag_head import SemanticTagHead
from probing_vlm_vgm.vggt.heads.camera_head import CameraHead

logger = logging.getLogger(__name__)


def _auto_intermediate_layer_idx(backbone_depth: int) -> List[int]:
    """Pick 4 evenly-spaced layer indices in [0, backbone_depth) for DPT.

    backbone_depth=0  → [0, 0, 0, 0]  (depth=0 branch emits 4 copies anyway)
    backbone_depth=1  → [0, 0, 0, 0]
    backbone_depth=2  → [0, 0, 1, 1]
    backbone_depth=4  → [0, 1, 2, 3]
    backbone_depth=8  → [1, 3, 5, 7]   (matches the historical default)
    """
    if backbone_depth <= 1:
        return [0, 0, 0, 0]
    return [
        max(0, min(backbone_depth - 1, int(round((i + 1) * (backbone_depth - 1) / 4))))
        for i in range(4)
    ]


class ProbeModelPA(nn.Module, PyTorchModelHubMixin):
    """Unified probe — backbone + head bank, selected by `active_heads`.

    Args:
        video_channels:   input frozen-VFM channel dim.
        embed_dim:        backbone width.
        backbone_depth:   N alt-attn layers. 0 = read-out-purity ablation.
        gradient_checkpointing: enabled inside backbone.
        active_heads:     subset of {"depth", "point", "camera", "instance",
                          "semantic_tag"} — only these heads are instantiated.

        DPT head args (used by depth/point):
          dpt_dim, dpt_stage_channels, dpt_intermediate_layer_idx (None
          ⇒ auto-derive from backbone_depth).

        InstanceHead args (used by "instance"):
          instance_out_channels (default 32)
          instance_mlp (default False — use InstanceHeadMLP instead)

        SemanticTagHead args (used by "semantic_tag"):
          num_classes, semantic_embed_dim, semantic_num_layers,
          semantic_num_heads, semantic_per_class_linear, clip_init_embeds.

        with_mask: inherited foreground-mask tokenizer flag (Exp-A only).
    """

    def __init__(
        self,
        video_channels: int = 1024,
        embed_dim: int = 512,
        backbone_depth: int = 8,
        backbone_num_heads: int = 16,
        backbone_dropout: float = 0.0,  # proj + MLP dropout inside every backbone block
        gradient_checkpointing: bool = False,
        # DPT head config (depth/point heads)
        dpt_dim: int = 128,
        dpt_stage_channels: List[int] = [128, 256, 512, 512],
        dpt_intermediate_layer_idx: Optional[List[int]] = None,
        # Head selection
        active_heads: List[str] = ("depth", "point"),
        # Instance head config
        instance_out_channels: int = 32,
        instance_mlp: bool = False,
        instance_hidden_channels: Optional[int] = None,
        # SemanticTag head config
        num_classes: int = 200,
        semantic_embed_dim: int = 512,
        semantic_num_layers: int = 2,
        semantic_num_heads: int = 8,
        semantic_mlp_ratio: float = 4.0,
        semantic_dropout: float = 0.0,
        semantic_classifier_mode: str = "per_class",  # "per_class" | "q2l" | "open_vocab"
        clip_init_embeds: Optional[torch.Tensor] = None,
        clip_init_path: Optional[str] = None,
        # Mask tokenizer (Exp-A optional)
        with_mask: bool = False,
    ) -> None:
        super().__init__()

        self.active_heads = set(active_heads)
        self.backbone_depth = backbone_depth
        self.embed_dim = embed_dim

        # Auto-derive DPT layer indices if not explicitly given. Always do
        # this when an explicit idx would reference layers that don't exist
        # under the current backbone_depth (e.g. dropping from 8 → 4 layers).
        if dpt_intermediate_layer_idx is None:
            dpt_intermediate_layer_idx = _auto_intermediate_layer_idx(backbone_depth)
        elif backbone_depth > 0 and max(dpt_intermediate_layer_idx) >= backbone_depth:
            # Existing yaml writes intermediate_layer_idx explicitly assuming
            # depth=8 (e.g. [1,3,5,7]). Auto-clamp instead of crashing — this
            # makes the Abl-1 depth sweep work without per-yaml overrides.
            old_idx = list(dpt_intermediate_layer_idx)
            dpt_intermediate_layer_idx = _auto_intermediate_layer_idx(backbone_depth)
            logger.warning(
                "ProbeModelPA: dpt_intermediate_layer_idx=%s references layers "
                ">= backbone_depth=%d. Auto-clamping to %s. Pass an explicit "
                "in-range value to silence this warning.",
                old_idx, backbone_depth, dpt_intermediate_layer_idx,
            )

        # ----- Shared backbone -----
        self.backbone = BackbonePA(
            in_channels=video_channels,
            embed_dim=embed_dim,
            depth=backbone_depth,
            num_heads=backbone_num_heads,
            dropout=backbone_dropout,
            gradient_checkpointing=gradient_checkpointing,
            with_mask=with_mask,
        )

        # ----- Geometry heads (Exp-A) -----
        head_in = embed_dim * 2  # frame+global concat dim

        self.camera_head = (
            CameraHead(dim_in=head_in) if "camera" in self.active_heads else None
        )

        self.point_head = (
            DPTHead(
                dim_in=head_in,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
                features=dpt_dim,
                out_channels=dpt_stage_channels,
                intermediate_layer_idx=dpt_intermediate_layer_idx,
            )
            if "point" in self.active_heads
            else None
        )

        self.depth_head = (
            DPTHead(
                dim_in=head_in,
                output_dim=2,
                activation="exp",
                conf_activation="expp1",
                features=dpt_dim,
                out_channels=dpt_stage_channels,
                intermediate_layer_idx=dpt_intermediate_layer_idx,
            )
            if "depth" in self.active_heads
            else None
        )

        # ----- Instance head (Exp-C) -----
        if "instance" in self.active_heads:
            if instance_mlp:
                self.instance_head = InstanceHeadMLP(
                    in_channels=head_in,
                    out_channels=instance_out_channels,
                    hidden_channels=instance_hidden_channels,
                )
            else:
                self.instance_head = InstanceHead(
                    in_channels=head_in,
                    out_channels=instance_out_channels,
                )
        else:
            self.instance_head = None

        # ----- Semantic-tag head (Exp-B) -----
        if "semantic_tag" in self.active_heads:
            # yaml-friendly path: if clip_init_path is given, load the .npy
            # here so the rest of the model doesn't have to know about file
            # I/O. clip_init_embeds (a Tensor) takes precedence — useful for
            # in-process construction (e.g. tests).
            if clip_init_embeds is None and clip_init_path is not None:
                arr = np.load(clip_init_path)
                if arr.shape[0] != num_classes:
                    raise ValueError(
                        f"clip_init_path={clip_init_path!r} has {arr.shape[0]} rows "
                        f"but num_classes={num_classes}. Did you build the embeds "
                        f"for the right vocabulary?"
                    )
                clip_init_embeds = torch.from_numpy(arr).float()

            self.semantic_tag_head = SemanticTagHead(
                dim_in=head_in,
                num_classes=num_classes,
                embed_dim=semantic_embed_dim,
                num_layers=semantic_num_layers,
                num_heads=semantic_num_heads,
                mlp_ratio=semantic_mlp_ratio,
                dropout=semantic_dropout,
                clip_init_embeds=clip_init_embeds,
                classifier_mode=semantic_classifier_mode,
            )
        else:
            self.semantic_tag_head = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _strip_special_tokens(
        last_tokens: torch.Tensor, patch_start_idx: int, Hf: int, Wf: int
    ) -> torch.Tensor:
        """(B,S,P_full,2C) → (B,S,P_patch,2C) with camera+register removed."""
        return last_tokens[..., patch_start_idx:, :]

    @staticmethod
    def _tokens_to_spatial(
        patch_tokens: torch.Tensor, Hf: int, Wf: int
    ) -> torch.Tensor:
        """(B,S,P_patch,2C) → (B,S,2C,Hf,Wf)."""
        B, S, P, C = patch_tokens.shape
        assert P == Hf * Wf, f"P={P} != Hf*Wf={Hf*Wf}"
        return patch_tokens.permute(0, 1, 3, 2).reshape(B, S, C, Hf, Wf).contiguous()

    # ------------------------------------------------------------------ #
    def forward(
        self,
        video_features: torch.Tensor,
        video_shape: tuple,
        fg_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            video_features: (B, S, C, Hf, Wf) frozen VFM features.
            video_shape:    (B, S, 3, H, W) original frame shape — DPT needs
                            it for upsampling target.
            fg_mask:        optional (B, S, 1, H, W) foreground mask for
                            backbone-side tokenization (Exp-A only).

        Returns:
            dict with whichever keys correspond to active_heads.
        """
        assert len(video_shape) == 5, "video_shape must be (B,S,3,H,W)"
        aggregated_tokens_list, patch_start_idx = self.backbone(video_features, fg_mask)

        Hf, Wf = video_features.shape[3], video_features.shape[4]
        predictions: Dict[str, Any] = {}

        with torch.amp.autocast("cuda", enabled=False):
            # ----- Geometry heads -----
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    n_patches=(Hf, Wf),
                    frames_shape=video_shape,
                    patch_start_idx=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list,
                    n_patches=(Hf, Wf),
                    frames_shape=video_shape,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            # Instance / semantic heads only need the FINAL backbone stage,
            # with the camera + register tokens stripped out.
            need_last_patch = (
                self.instance_head is not None or self.semantic_tag_head is not None
            )
            if need_last_patch:
                last_tokens = aggregated_tokens_list[-1]
                patch_tokens = self._strip_special_tokens(
                    last_tokens, patch_start_idx, Hf, Wf
                )

                if self.instance_head is not None:
                    spatial = self._tokens_to_spatial(patch_tokens, Hf, Wf)
                    predictions["instance"] = self.instance_head(spatial)

                if self.semantic_tag_head is not None:
                    predictions["tag_logits"] = self.semantic_tag_head(patch_tokens)

        return predictions
