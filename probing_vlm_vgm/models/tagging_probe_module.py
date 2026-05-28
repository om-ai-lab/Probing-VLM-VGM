"""LightningModule for Exp-B: video-level multi-label semantic tagging.

Training path:
    vfm_feat  ──►  ProbeModelPA (shared backbone + SemanticTagHead)
                    │
                    ▼
              tag_logits [B, num_classes]
                    │
                    ▼
              AsymmetricLoss(logits, video_label)

Validation/test path:
    Accumulate (logits, labels) across the full val/test loader, then
    once per epoch compute the full Exp-B metric bundle:
        val/mAP, val/AP_head, val/AP_mid, val/AP_tail, val/OF1, val/CF1

OF1/CF1 use a fixed sigmoid probability threshold of 0.5, matching the
closed-vocabulary tagging eval convention used for this project.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from probing_vlm_vgm.eval.tagging_metric import _bucket_indices, compute_tagging_metrics
from probing_vlm_vgm.losses import AsymmetricLoss
from probing_vlm_vgm.models.base_probe_module import BaseProbeModule
from probing_vlm_vgm.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class TaggingProbeLitModule(BaseProbeModule):
    """Wraps a ProbeModelPA (with semantic_tag head active) for Exp-B."""

    def __init__(
        self,
        probe: nn.Module,                              # ProbeModelPA
        optimizer: Any,
        scheduler: Any,
        compile: bool = False,
        # Loss config
        asl_gamma_neg: float = 4.0,
        asl_gamma_pos: float = 0.0,
        asl_clip: float = 0.05,
        # Metric config
        train_pos_rate_path: Optional[str] = None,   # .npy path with (C,) train positive rates
        # Misc
        pretrained: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            compile=compile,
            pretrained=pretrained,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        self.save_hyperparameters(logger=False, ignore=["probe"])
        self.kwargs = kwargs
        self.output_path = kwargs.get("output_path", None)
        self.case_output_root = kwargs.get("case_output_root", "")
        self.case_model_name = kwargs.get("case_model_name", "")

        self.probe = probe
        self.criterion = AsymmetricLoss(
            gamma_neg=asl_gamma_neg,
            gamma_pos=asl_gamma_pos,
            clip=asl_clip,
        )

        # Pre-computed training positive rate (used for head/mid/tail bucketing
        # in eval). If None, the metric falls back to the eval-set's own
        # positive rate, which is fine for smoke tests but less principled.
        self._train_pos_rate: Optional[np.ndarray] = None
        if train_pos_rate_path is not None:
            try:
                self._train_pos_rate = np.load(train_pos_rate_path)
                log.info(
                    f"loaded train_pos_rate from {train_pos_rate_path} "
                    f"shape={self._train_pos_rate.shape}"
                )
            except Exception as e:  # noqa: BLE001 — keep training running
                log.warning(f"failed to load train_pos_rate: {e}")

        self._class_names: Optional[list[str]] = None
        num_classes = 200
        head = getattr(self.probe, "semantic_tag_head", None)
        if head is not None and hasattr(head, "num_classes"):
            num_classes = int(head.num_classes)
        elif hasattr(self.probe, "num_classes"):
            num_classes = int(self.probe.num_classes)
        data_root = kwargs.get("data_root", "data/ScanNet/ScanNet-processed")
        class_path = os.path.join(data_root, f"class_names_{num_classes}.json")
        if os.path.isfile(class_path):
            try:
                with open(class_path, "r") as f:
                    class_data = json.load(f)
                self._class_names = class_data.get("names", class_data)
            except Exception as e:  # noqa: BLE001
                log.warning(f"failed to load class names from {class_path}: {e}")
        self._tag_buckets: Dict[str, set[int]] = {}
        if self._train_pos_rate is not None:
            self._tag_buckets = {
                name: set(int(i) for i in idxs)
                for name, idxs in _bucket_indices(self._train_pos_rate).items()
            }

        # Eval-time accumulators (reset per epoch).
        self._val_logits: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def _write_case_outputs(
        self,
        batch: Dict[str, Any],
        logits: torch.Tensor,
        labels: torch.Tensor,
        topk: int = 20,
    ) -> None:
        if not self.case_output_root:
            return
        probs = torch.sigmoid(logits.detach().float()).cpu()
        labels_cpu = labels.detach().float().cpu()
        scene_ids = batch.get("scene_id", [])
        model_name = self.case_model_name or os.path.basename(str(self.output_path or "model"))
        mid_classes = self._tag_buckets.get("mid", set())
        for i, scene_id in enumerate(scene_ids):
            scene_id = str(scene_id)
            out_dir = os.path.join(str(self.case_output_root), scene_id, model_name)
            os.makedirs(out_dir, exist_ok=True)
            k = min(topk, probs.shape[1])
            vals, idxs = torch.topk(probs[i], k=k)
            pos = torch.nonzero(labels_cpu[i] > 0.5, as_tuple=False).flatten().tolist()
            pos_set = set(pos)
            mid_pos = [cls_idx for cls_idx in pos if cls_idx in mid_classes]
            top5 = set(idxs[: min(5, len(idxs))].tolist())
            top10 = set(idxs[: min(10, len(idxs))].tolist())

            def class_name(cls_idx: int) -> str:
                return (
                    self._class_names[cls_idx]
                    if self._class_names and cls_idx < len(self._class_names)
                    else f"class_{cls_idx}"
                )

            def mean_prob(indices: list[int]) -> Optional[float]:
                if not indices:
                    return None
                return float(probs[i, indices].mean().item())

            score_payload = {
                "scene_id": scene_id,
                "model": model_name,
                "positive_mean_prob": mean_prob(pos),
                "mid_positive_mean_prob": mean_prob(mid_pos),
                "num_positive": len(pos),
                "num_mid_positive": len(mid_pos),
                "mid_hit_at_5": None if not mid_pos else len(set(mid_pos) & top5) / len(mid_pos),
                "mid_hit_at_10": None if not mid_pos else len(set(mid_pos) & top10) / len(mid_pos),
                "positive_labels": [
                    {
                        "index": cls_idx,
                        "name": class_name(cls_idx),
                        "prob": float(probs[i, cls_idx].item()),
                        "bucket": "mid" if cls_idx in mid_classes else "other",
                    }
                    for cls_idx in pos
                ],
                "top_predictions": [
                    {
                        "rank": rank,
                        "index": cls_idx,
                        "name": class_name(cls_idx),
                        "prob": float(score),
                        "is_positive": cls_idx in pos_set,
                        "bucket": "mid" if cls_idx in mid_classes else "other",
                    }
                    for rank, (cls_idx, score) in enumerate(
                        zip(idxs.tolist(), vals.tolist()), start=1
                    )
                ],
                "class_probs": [
                    {
                        "index": cls_idx,
                        "name": class_name(cls_idx),
                        "prob": float(probs[i, cls_idx].item()),
                        "is_positive": cls_idx in pos_set,
                        "bucket": "mid" if cls_idx in mid_classes else "other",
                    }
                    for cls_idx in range(probs.shape[1])
                ],
            }
            with open(os.path.join(out_dir, "tagging_scores.json"), "w") as f:
                json.dump(score_payload, f, indent=2)

            with open(os.path.join(out_dir, "tagging.txt"), "w") as f:
                f.write(f"scene_id: {scene_id}\n")
                f.write(f"model: {model_name}\n")
                f.write(f"positive_mean_prob: {score_payload['positive_mean_prob']}\n")
                f.write(f"mid_positive_mean_prob: {score_payload['mid_positive_mean_prob']}\n")
                f.write(f"mid_hit_at_5: {score_payload['mid_hit_at_5']}\n")
                f.write(f"mid_hit_at_10: {score_payload['mid_hit_at_10']}\n")
                f.write("top_predictions:\n")
                for rank, (cls_idx, score) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
                    name = class_name(cls_idx)
                    gt_mark = "*" if cls_idx in pos else ""
                    f.write(f"{rank:02d}. {name}\t{score:.6f}{gt_mark}\n")
                f.write("positive_labels:\n")
                for cls_idx in pos:
                    bucket = "mid" if cls_idx in mid_classes else "other"
                    f.write(f"- {class_name(cls_idx)}\t{probs[i, cls_idx].item():.6f}\t{bucket}\n")

    # ------------------------------------------------------------------ #
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Returns raw logits [B, num_classes] from the semantic_tag head."""
        vfm_feat = batch["vfm_feat"]
        # Dataset returns (B, S, H, W, C); ProbeModelPA wants (B, S, C, H, W).
        vfm_feat = vfm_feat.permute(0, 1, 4, 2, 3).contiguous()

        # Optional spatial pool (mirrors the path in InstanceProbeLitModule).
        if "target_spatial_size" in batch:
            target_hw = tuple(batch["target_spatial_size"][0].tolist())
            B, S, C, H, W = vfm_feat.shape
            if (H, W) != target_hw:
                assert target_hw[0] <= H and target_hw[1] <= W, (
                    f"target {target_hw} must be <= source ({H},{W})"
                )
                x = vfm_feat.reshape(B * S, C, H, W)
                x = F.adaptive_avg_pool2d(x, output_size=target_hw)
                vfm_feat = x.reshape(B, S, C, target_hw[0], target_hw[1])

        # video_shape is needed by DPT heads but not by SemanticTagHead. We
        # pass a sentinel so ProbeModelPA can still assert 5D; the geometry
        # heads are not instantiated under this config.
        B, S, C, H, W = vfm_feat.shape
        video_shape = (B, S, 3, H * 14, W * 14)

        preds = self.probe(vfm_feat, video_shape)
        return preds["tag_logits"]

    # ------------------------------------------------------------------ #
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch)
        labels = batch["tag_labels"].to(dtype=logits.dtype)
        loss = self.criterion(logits, labels)

        bs = logits.shape[0]
        self.log(
            "train/loss", loss, on_step=True, on_epoch=False, prog_bar=True,
            batch_size=bs, sync_dist=True,
        )
        self.log(
            "train/loss_asl", loss, on_step=True, on_epoch=True, prog_bar=False,
            batch_size=bs, sync_dist=True,
        )
        self.log(
            "trainer/lr",
            self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0],
            on_step=True, on_epoch=False, prog_bar=True,
        )
        return loss

    # ------------------------------------------------------------------ #
    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self._val_logits = []
        self._val_labels = []

    @torch.no_grad()
    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        logits = self(batch)
        labels = batch["tag_labels"].to(dtype=logits.dtype)
        if self.case_output_root and getattr(self.trainer, "testing", False):
            self._write_case_outputs(batch, logits, labels)

        loss = self.criterion(logits, labels)
        bs = logits.shape[0]
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
            batch_size=bs, sync_dist=True,
        )
        self.log(
            "val/loss_asl", loss, on_step=False, on_epoch=True,
            batch_size=bs, sync_dist=True,
        )
        # Accumulate for epoch-end metric.
        self._val_logits.append(logits.detach().float().cpu())
        self._val_labels.append(labels.detach().float().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_logits:
            return

        # Concat local-rank batches.
        local_logits = torch.cat(self._val_logits, dim=0).to(self.device)  # (N_local, C)
        local_labels = torch.cat(self._val_labels, dim=0).to(self.device)  # (N_local, C)

        # DDP-safe: gather across all ranks so the metric is computed on the
        # FULL validation set, not the local shard. AP / F1 are not
        # linearly-averageable; computing them per-rank and averaging via
        # sync_dist produces incorrect numbers (e.g. a rank with zero
        # positives for class c contributes NaN AP that pollutes the mean).
        gathered_logits = self.all_gather(local_logits)
        gathered_labels = self.all_gather(local_labels)

        # all_gather output shape:
        #   single GPU  → (N_local, C)            — same as input
        #   multi-rank  → (world_size, N_local, C) — prepend gathering dim
        if gathered_logits.dim() == 3:
            gathered_logits = gathered_logits.flatten(0, 1)
            gathered_labels = gathered_labels.flatten(0, 1)

        logits_np = gathered_logits.float().cpu().numpy()  # (N_global, C)
        labels_np = gathered_labels.float().cpu().numpy()  # (N_global, C)

        # from_logits=True: the head returns raw scores (we don't sigmoid in
        # forward). Be explicit so threshold tuning isn't fooled by an
        # early-training stretch where logits happen to fall in [0, 1].
        m = compute_tagging_metrics(
            labels_np, logits_np,
            train_pos_rate=self._train_pos_rate,
            from_logits=True,
        )
        # Every rank holds the SAME global metric (identical all_gather'd
        # data → identical compute). Log on ALL ranks with sync_dist=False:
        #   - the values are already global, so no reduction is needed
        #   - logging on every rank (not rank_zero_only) keeps `val/mAP`
        #     visible to the EarlyStopping callback on every rank, which
        #     needs a consistent monitor value to make its all-reduced
        #     stop/continue decision under DDP.
        self.log("val/mAP", float(m.mAP), sync_dist=False)
        self.log("val/AP_head", float(m.AP_head), sync_dist=False)
        self.log("val/AP_mid", float(m.AP_mid), sync_dist=False)
        self.log("val/AP_tail", float(m.AP_tail), sync_dist=False)
        self.log("val/OF1", float(m.OF1), sync_dist=False)
        self.log("val/CF1", float(m.CF1), sync_dist=False)

        # Reset to free memory; on_validation_epoch_start re-empties anyway.
        self._val_logits = []
        self._val_labels = []

    # test_step / on_test_epoch_end share the val path. Keeping them as
    # thin wrappers means `python train.py test=true` produces the same
    # `val/*` keys parse_results.py already expects.
    @torch.no_grad()
    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> None:
        self.validation_step(batch, batch_idx, dataloader_idx)

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()
