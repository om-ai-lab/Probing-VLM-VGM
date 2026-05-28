"""LightningModule for Setup A: instance probe on frozen VFM features.

Training path:
    vfm_feat  ──►  InstanceHead (1×1 Conv + normalize)  ──►  feats (B, S, D, Hf, Wf)
        │                                                        │
        │                                                        ▼
        └──────────────►  upsample to mask resolution (bilinear + renormalize)
                          │
                          ▼
                          mvc_loss(feats_up, instance_masks, valid_mask)

Evaluation path:
    vfm_feat  ──►  feats (B, S, D, Hf, Wf)
        │
        ▼  for each scene in batch:
    cluster at *feature* resolution via HDBSCAN on (S*Hf*Wf, D)
        │   (cheap: ~5K-30K points vs millions at mask res)
        ▼
    upsample cluster labels (S, Hf, Wf) → (S, H_mask, W_mask) via nearest
        │
        ▼
    instance_metric.t_miou_t_sr  → val/t_miou, val/t_sr

Why cluster at feature resolution:
    HDBSCAN cost scales with N log N. On 540×720×8 mask pixels per scene
    (~3M points) clustering would dominate wall-clock. At the feature grid
    (~5K-30K points) it's near-instant, and the nearest-neighbor upsample to
    mask resolution preserves the semantic boundaries the clusters encode.

Metrics logged:
    train/loss_mvc, train/loss_pull, train/loss_push, train/num_pull_pairs, ...
    val/loss_mvc, val/loss_pull, val/loss_push,
    val/t_miou, val/t_sr, val/n_clusters, val/n_gt_instances
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from PIL import Image
from torchmetrics import MeanMetric

from probing_vlm_vgm.eval.instance_metric import (
    aggregate_scene_metrics,
    hdbscan_cluster,
    t_miou_t_sr,
)
from probing_vlm_vgm.losses import mvc_loss
from probing_vlm_vgm.models.base_probe_module import BaseProbeModule
from probing_vlm_vgm.utils.vis_utils import vfm_pca_images
from probing_vlm_vgm.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


# ---------------------------------------------------------------------- #
# Module-level worker for ProcessPoolExecutor.
#
# Must be top-level (not a method) so it pickles cleanly for forkserver.
# Takes plain numpy arrays + scalars; returns the same dict shape as the
# sync path produces, so the calling LitModule can append directly into
# self._val_per_scene without any post-processing.
# ---------------------------------------------------------------------- #
def _cluster_and_score_one(
    feat_b: np.ndarray,             # (S*Hf*Wf, D) float32
    masks_b: np.ndarray,            # (S, H_m, W_m) int64
    valid_b: np.ndarray,            # (S, H_m, W_m) bool
    fr_shape: tuple,                # (S, Hf, Wf) — to reshape labels
    mask_hw: tuple,                 # (H_m, W_m) — NN upsample target
    min_cluster_size: int,
    min_samples: int,
    pca_dim: Optional[int],
    iou_thresh: float,
    ignore_ids: tuple,
) -> Dict[str, float]:
    # Lazy imports keep worker spawn cost low; torch is heavy but we only
    # need it for F.interpolate (NN nearest upsample).
    import numpy as np
    import torch
    import torch.nn.functional as F
    from threadpoolctl import threadpool_limits

    from probing_vlm_vgm.eval.instance_metric import hdbscan_cluster, t_miou_t_sr

    # Pin BLAS/OMP to 1 thread inside this worker: each scene is its own
    # process, and with N workers × default-96-BLAS-threads we'd oversubscribe
    # the box and watch wall-clock get WORSE than sync. threadpool_limits is
    # set per-process and reverts on exit; safe to invoke unconditionally
    # because the pool calls this function once per submitted task.
    with threadpool_limits(limits=1):
        labels = hdbscan_cluster(
            feat_b,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            pca_dim=pca_dim,
        ).reshape(fr_shape)
        labels_up = (
            F.interpolate(
                torch.from_numpy(labels).unsqueeze(1).float(),
                size=mask_hw, mode="nearest",
            )
            .long()
            .squeeze(1)
            .numpy()
        )
        scene_metrics = t_miou_t_sr(
            labels_up, masks_b, valid_b,
            iou_thresh=iou_thresh, ignore_ids=ignore_ids,
        )
        scene_metrics["n_clusters"] = int((np.unique(labels) >= 0).sum())
    return scene_metrics


class InstanceProbeLitModule(BaseProbeModule):
    """Lightning wrapper around the instance head + MVC loss + eval metrics."""

    def __init__(
        self,
        probe: nn.Module,                # ProbeModelPA(active_heads=["instance"])
        optimizer,
        scheduler,
        compile: bool = False,
        # loss config
        mvc_num_samples: int = 1024,
        mvc_margin: float = 1.0,
        mvc_lambda_pull: float = 1.0,
        mvc_lambda_push: float = 1.0,
        # eval config
        hdbscan_min_cluster_size: int = 30,
        hdbscan_min_samples: int = 5,
        hdbscan_pca_dim: Optional[int] = 8,
        eval_iou_thresh: float = 0.5,
        eval_ignore_ids: tuple = (0,),
        weight_metrics_by_instances: bool = False,
        # Parallel eval: when >0, HDBSCAN+T-mIoU per scene is dispatched to
        # a persistent ProcessPoolExecutor (forkserver) so the GPU forward
        # in test_step doesn't wait for clustering. Per-scene work is pure
        # CPU numpy; deterministic; identical aggregate metrics (within fp
        # rounding) vs sync. Tune to leave headroom for the dataloader.
        num_eval_workers: int = 0,
        # viz config (test_step only)
        viz_scene_ids: Optional[Sequence[str]] = None,
        viz_random_n: int = 0,        # if >0 and viz_scene_ids is empty,
        viz_random_seed: int = 0,     # randomly sample N scene_ids from the
                                       # val set (fixed seed → same N scenes
                                       # picked across every ckpt eval).
        viz_output_subdir: str = "viz/test",
        viz_max_frames: int = 8,
        viz_match_pred_to_gt: bool = True,
        viz_save_individual: bool = False,
        # misc
        pretrained: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            compile=compile,
            pretrained=pretrained,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        # Lightning saves self.hparams; we ignore the probe because it's a
        # nn.Module (it'll be tracked via self.probe anyway).
        self.save_hyperparameters(logger=False, ignore=["probe"])
        self.kwargs = kwargs
        self.output_path = kwargs.get("output_path", None)

        # The probe is the unified ProbeModelPA configured with
        # active_heads=["instance"]. Plan v3 §5.3 — instance shares the same
        # BackbonePA as Exp-A/B so the three tasks are read out under
        # identical capacity, making the cross-task comparison apples-to-apples.
        self.probe = probe

        # MVC / eval config
        self.mvc_num_samples = mvc_num_samples
        self.mvc_margin = mvc_margin
        self.mvc_lambda_pull = mvc_lambda_pull
        self.mvc_lambda_push = mvc_lambda_push
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.hdbscan_pca_dim = hdbscan_pca_dim
        self.eval_iou_thresh = eval_iou_thresh
        self.eval_ignore_ids = tuple(eval_ignore_ids)
        self.weight_metrics_by_instances = weight_metrics_by_instances
        self.num_eval_workers = int(num_eval_workers)
        # Created in on_test_epoch_start, torn down in on_test_epoch_end so
        # workers don't survive across multiple .test() invocations.
        self._eval_pool = None
        self._eval_futures: List = []
        # One-time guard for the masks int64 → int16 cast in test_step.
        self._mask_dtype_checked = False

        self.pretrained = pretrained

        # Viz config: only scenes whose batch["scene_id"] is in this set get
        # rendered. Empty set ⇒ no viz, zero overhead. Requires the dataset
        # to be constructed with load_images=True (otherwise we silently skip
        # because RGB frames are absent from the batch).
        self.viz_scene_ids = set(viz_scene_ids or [])
        self.viz_random_n = int(viz_random_n)
        self.viz_random_seed = int(viz_random_seed)
        self.viz_output_subdir = viz_output_subdir
        self.viz_max_frames = viz_max_frames
        self.viz_match_pred_to_gt = viz_match_pred_to_gt
        self.viz_save_individual = viz_save_individual
        self._viz_palette = self._build_instance_palette()
        self._viz_warned_no_images = False

        # Running meters
        self.val_loss = MeanMetric()
        self._val_per_scene: List[Dict[str, float]] = []

    # ---------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------- #
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Run the unified probe (BackbonePA + instance head) on frozen VFM feat.

        batch['vfm_feat']: (B, S, Hf, Wf, C) — our dataset returns [H, W, C] last.
        Returns: (B, S, D, Hf, Wf) L2-normalized instance embedding.
        """
        vfm_feat = batch["vfm_feat"]  # (B, S, Hf, Wf, C)
        # Move channel to dim 2 for the probe's expected (B, S, C, H, W) layout.
        vfm_feat = vfm_feat.permute(0, 1, 4, 2, 3).contiguous()
        # Optional spatial pool — moved here from the dataset worker. CPU-side
        # adaptive_avg_pool2d on bf16 was the dominant per-worker cost and
        # starved the GPU; on GPU it is essentially free.
        if "target_spatial_size" in batch:
            target_hw = tuple(batch["target_spatial_size"][0].tolist())
            B, S, C, H, W = vfm_feat.shape
            if (H, W) != target_hw:
                assert target_hw[0] <= H and target_hw[1] <= W, (
                    f"target {target_hw} must be <= source ({H},{W}); "
                    f"upsampling latents is not well-defined for this "
                    f"downsampling-only path."
                )
                x = vfm_feat.reshape(B * S, C, H, W)
                x = F.adaptive_avg_pool2d(x, output_size=target_hw)
                vfm_feat = x.reshape(B, S, C, target_hw[0], target_hw[1])

        # video_shape is required by ProbeModelPA's signature but only used
        # by the DPT/Camera heads, which are not instantiated here. Pass a
        # sentinel (Hf*14, Wf*14) — the actual image resolution is irrelevant
        # to the instance head.
        B, S, C, Hf, Wf = vfm_feat.shape
        video_shape = (B, S, 3, Hf * 14, Wf * 14)
        preds = self.probe(vfm_feat, video_shape)
        return preds["instance"]  # (B, S, D, Hf, Wf)

    # ---------------------------------------------------------------- #
    # Training
    # ---------------------------------------------------------------- #
    def _upsample_and_renormalize(
        self, feats: torch.Tensor, target_hw: tuple
    ) -> torch.Tensor:
        """Bilinear upsample (B, S, D, Hf, Wf) → (B, S, D, H, W) then L2-renorm."""
        B, S, D, Hf, Wf = feats.shape
        x = feats.reshape(B * S, D, Hf, Wf)
        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        x = F.normalize(x, dim=1)
        return x.reshape(B, S, D, target_hw[0], target_hw[1])

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        feats = self(batch)  # (B, S, D, Hf, Wf)
        masks = batch["instance_masks"]  # (B, S, H_m, W_m) int64
        valid = batch["valid_mask"]  # (B, S, H_m, W_m) bool
        mask_hw = masks.shape[-2:]

        # Upsample feats to mask resolution (training in mask space preserves
        # rich instance-boundary signal from GT).
        feats_up = self._upsample_and_renormalize(feats, mask_hw)

        out = mvc_loss(
            feats_up,
            masks,
            valid_mask=valid,
            num_samples=self.mvc_num_samples,
            margin=self.mvc_margin,
            lambda_pull=self.mvc_lambda_pull,
            lambda_push=self.mvc_lambda_push,
        )

        loss = out["loss"]
        bs = feats.shape[0]
        # `train/loss` is the canonical key the EarlyStopping callback monitors
        # (configs/callbacks/default.yaml). Log it as an alias for `loss_mvc`
        # so the same callback works for both video-probe and instance-probe.
        self.log("train/loss", loss, on_step=True, on_epoch=False,
                 prog_bar=False, batch_size=bs, sync_dist=True)
        self.log("train/loss_mvc", loss, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=bs, sync_dist=True)
        self.log("train/loss_pull", out["loss_pull"], on_step=False, on_epoch=True,
                 batch_size=bs, sync_dist=True)
        self.log("train/loss_push", out["loss_push"], on_step=False, on_epoch=True,
                 batch_size=bs, sync_dist=True)
        return loss

    # ---------------------------------------------------------------- #
    # Validation (training-time): MVC loss only, NO HDBSCAN.
    # HDBSCAN + T-mIoU/T-SR live in the test path so train-time val stays
    # cheap (clustering at mask res or feat res ≫ training step time).
    # ---------------------------------------------------------------- #
    def on_validation_epoch_start(self) -> None:
        # Base class handles set_epoch on dataset/sampler.
        super().on_validation_epoch_start()
        self.val_loss.reset()

    @torch.no_grad()
    def _step_loss(
        self,
        batch: Dict[str, Any],
        log_prefix: str = "val",
    ) -> torch.Tensor:
        """Shared loss path for both validation_step and test_step.

        Computes the MVC pull/push loss at mask resolution and logs the two
        components plus aggregates the running mean for `<log_prefix>/loss_mvc`.
        Returns the per-batch loss tensor (also captured by self.val_loss).
        """
        feats = self(batch)  # (B, S, D, Hf, Wf)
        masks = batch["instance_masks"]  # (B, S, H_m, W_m)
        valid = batch["valid_mask"]
        mask_hw = masks.shape[-2:]

        feats_up = self._upsample_and_renormalize(feats, mask_hw)
        out = mvc_loss(
            feats_up,
            masks,
            valid_mask=valid,
            num_samples=self.mvc_num_samples,
            margin=self.mvc_margin,
            lambda_pull=self.mvc_lambda_pull,
            lambda_push=self.mvc_lambda_push,
        )
        bs = feats.shape[0]
        self.val_loss(out["loss"])
        self.log(f"{log_prefix}/loss_pull", out["loss_pull"],
                 on_step=False, on_epoch=True,
                 batch_size=bs, sync_dist=True)
        self.log(f"{log_prefix}/loss_push", out["loss_push"],
                 on_step=False, on_epoch=True,
                 batch_size=bs, sync_dist=True)
        return out["loss"]

    @torch.no_grad()
    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Lightweight: only the MVC loss, no HDBSCAN clustering.
        # Use test_step (e.g. via `train=false test=true`) for the full
        # T-mIoU / T-SR evaluation.
        self._step_loss(batch, log_prefix="val")

    def on_validation_epoch_end(self) -> None:
        # `val/loss` is the canonical key the ModelCheckpoint callback monitors
        # (configs/callbacks/default.yaml). Alias to `val/loss_mvc` for
        # checkpoint selection compatibility with the video-probe pipeline.
        # Compute once (MeanMetric.compute resets state on next reset()), then
        # log to both keys as a Python float so neither call mutates state.
        val_loss = self.val_loss.compute()
        self.log("val/loss", val_loss, prog_bar=False, sync_dist=True)
        self.log("val/loss_mvc", val_loss, prog_bar=True, sync_dist=True)

    # ---------------------------------------------------------------- #
    # Test (heavyweight): MVC loss + HDBSCAN + T-mIoU/T-SR.
    # ---------------------------------------------------------------- #
    def on_test_epoch_start(self) -> None:
        self._val_per_scene = []
        self.val_loss.reset()
        self._eval_futures = []
        self._mask_dtype_checked = False

        # Resolve random viz_scene_ids: deterministic across ckpts because we
        # sample from the dataset's `scenes` list (built by walking val.json
        # in fixed order) with a user-provided seed. Skipped if the user has
        # already set explicit viz_scene_ids.
        if not self.viz_scene_ids and self.viz_random_n > 0:
            ds = self._get_test_dataset()
            if ds is not None and hasattr(ds, "scenes") and len(ds.scenes) > 0:
                all_ids = [s[1] for s in ds.scenes]
                rng = np.random.default_rng(self.viz_random_seed)
                n = min(self.viz_random_n, len(all_ids))
                idx = sorted(int(i) for i in rng.choice(len(all_ids), size=n, replace=False))
                self.viz_scene_ids = {all_ids[i] for i in idx}
                log.info(
                    f"viz_random: picked {n} scene(s) from {len(all_ids)} "
                    f"(seed={self.viz_random_seed}): {sorted(self.viz_scene_ids)}"
                )
            else:
                log.warning(
                    "viz_random_n>0 but couldn't reach the test dataset's scene "
                    "list; falling back to no viz this run."
                )

        if self.num_eval_workers > 0:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            # `forkserver` (not fork) avoids the OpenMP/libgomp deadlock that
            # hits HDBSCAN/sklearn workers spawned from a fork()-after-import
            # process; same rationale as VideoProbeDataModule's dataloader.
            self._eval_pool = ProcessPoolExecutor(
                max_workers=self.num_eval_workers,
                mp_context=mp.get_context("forkserver"),
            )
            log.info(
                f"InstanceProbeLitModule: started eval pool "
                f"(workers={self.num_eval_workers}, forkserver)"
            )

    @torch.no_grad()
    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # 1) Loss path (logs val/loss_pull + val/loss_push + accumulates val_loss).
        # We keep the "val/" prefix so existing parse_results.py and wandb
        # panels work unchanged across train-time val and test.
        self._step_loss(batch, log_prefix="val")

        # 2) HDBSCAN + per-scene metrics at feature resolution.
        feats = self(batch)
        masks = batch["instance_masks"]
        valid = batch["valid_mask"]
        mask_hw = masks.shape[-2:]
        B, S, D, Hf, Wf = feats.shape

        feats_cpu = feats.detach().to(dtype=torch.float32).cpu().numpy()
        # Cast masks int64 → int16 here: source data on disk is uint16 (see
        # ScanNetInstanceDataset.__getitem__), and instance IDs in practice
        # stay ≤ ~100 (verified on val split). Going int16 cuts IPC pickle
        # bandwidth 4x when submitting per-scene work to the eval pool. Hot
        # path, so we only do the safety check ONCE per .test() run.
        masks_int64 = masks.cpu().numpy()
        if not self._mask_dtype_checked:
            mx = int(masks_int64.max())
            if mx >= 32_768:
                raise ValueError(
                    f"instance_masks max id {mx} >= int16 limit (32768). "
                    f"Disable the int16 cast in test_step or widen to int32."
                )
            self._mask_dtype_checked = True
        masks_np = masks_int64.astype(np.int16, copy=False)
        valid_np = valid.cpu().numpy()

        fr_shape = (S, Hf, Wf)
        mask_hw_t = tuple(mask_hw)

        for b in range(B):
            feat_b = feats_cpu[b]  # (S, D, Hf, Wf)
            feat_b = np.transpose(feat_b, (0, 2, 3, 1)).reshape(-1, D)  # (N, D)
            scene_id = batch["scene_id"][b]
            wants_viz = (
                scene_id in self.viz_scene_ids and "images" in batch
            )

            # Async path: submit clustering + metric to the pool. The future
            # is drained in on_test_epoch_end. Skipped for viz scenes because
            # the visualizer needs the upsampled labels in-process.
            if self._eval_pool is not None and not wants_viz:
                fut = self._eval_pool.submit(
                    _cluster_and_score_one,
                    feat_b,
                    masks_np[b],
                    valid_np[b],
                    fr_shape,
                    mask_hw_t,
                    int(self.hdbscan_min_cluster_size),
                    int(self.hdbscan_min_samples),
                    self.hdbscan_pca_dim,
                    float(self.eval_iou_thresh),
                    tuple(self.eval_ignore_ids),
                )
                self._eval_futures.append(fut)
                continue

            # Sync path: pool disabled OR scene needs viz (we need labels_up
            # available locally to render the grid).
            labels = hdbscan_cluster(
                feat_b,
                min_cluster_size=self.hdbscan_min_cluster_size,
                min_samples=self.hdbscan_min_samples,
                metric="euclidean",
                pca_dim=self.hdbscan_pca_dim,
            )  # (N,) int64, -1 = noise
            labels = labels.reshape(S, Hf, Wf)

            labels_t = torch.from_numpy(labels).unsqueeze(1).float()
            labels_up = F.interpolate(
                labels_t, size=mask_hw, mode="nearest"
            ).long().squeeze(1).numpy()

            scene_metrics = t_miou_t_sr(
                labels_up,
                masks_np[b],
                valid_np[b],
                iou_thresh=self.eval_iou_thresh,
                ignore_ids=self.eval_ignore_ids,
            )
            scene_metrics["n_clusters"] = int((np.unique(labels) >= 0).sum())
            self._val_per_scene.append(scene_metrics)

            # ---- Optional per-scene visualization. ----
            if scene_id in self.viz_scene_ids:
                if "images" not in batch:
                    if not self._viz_warned_no_images:
                        log.warning(
                            "viz_scene_ids set but batch has no 'images' — set "
                            "load_images=True on the val dataset to enable viz."
                        )
                        self._viz_warned_no_images = True
                else:
                    # Pass feat-res labels + per-pixel head feature so the
                    # visualizer can render smooth, pixel-level pred masks
                    # via centroid 1-NN at mask resolution. The metric path
                    # above (NN-upsampled labels_up) is unchanged.
                    self._visualize_instance_grid(
                        scene_id=scene_id,
                        images=batch["images"][b],
                        gt_masks=masks[b],
                        pred_labels_fr=torch.from_numpy(labels),
                        head_feat=torch.from_numpy(feats_cpu[b]),
                        valid_mask=valid[b],
                        vfm_feat=batch["vfm_feat"][b],
                        vfm_idx=batch["vfm_idx"][b],
                    )

    def on_test_epoch_end(self) -> None:
        # Same val/loss + val/loss_mvc dual-logging as validation_epoch_end so
        # downstream parse_results.py and ModelCheckpoint see the canonical key.
        val_loss = self.val_loss.compute()
        self.log("val/loss", val_loss, prog_bar=False, sync_dist=True)
        self.log("val/loss_mvc", val_loss, prog_bar=True, sync_dist=True)

        # Drain any pending parallel-eval futures, then tear down the pool.
        # `as_completed` orders by finish time, not submission — fine because
        # aggregate_scene_metrics computes an order-invariant mean.
        if self._eval_pool is not None:
            from concurrent.futures import as_completed
            n_pending = len(self._eval_futures)
            log.info(f"Draining {n_pending} parallel-eval futures...")
            for fut in as_completed(self._eval_futures):
                self._val_per_scene.append(fut.result())
            self._eval_pool.shutdown(wait=True)
            self._eval_pool = None
            self._eval_futures = []

        if not self._val_per_scene:
            return

        agg = aggregate_scene_metrics(
            self._val_per_scene,
            weight_by_instances=self.weight_metrics_by_instances,
        )
        n_clusters_mean = float(
            np.mean([p.get("n_clusters", 0) for p in self._val_per_scene])
        )
        self.log("val/t_miou", agg["t_miou"], prog_bar=True, sync_dist=True)
        self.log("val/t_sr", agg["t_sr"], prog_bar=True, sync_dist=True)
        self.log("val/n_clusters_mean", n_clusters_mean, sync_dist=True)
        self.log("val/n_gt_instances_mean",
                 float(agg["n_gt_instances_total"]) / max(agg["n_scenes"], 1),
                 sync_dist=True)

        self._val_per_scene = []

    def _get_test_dataset(self):
        """Best-effort access to the underlying test/val Dataset.

        Lightning wraps the val loaders in a CombinedLoader (mode=sequential)
        for this datamodule, so we have to unwrap one layer to get the
        DataLoader and another to get its .dataset. Falls back to
        eval()-ing the datamodule's validation_datasets[0] string if the
        trainer-side handles aren't set up yet.
        """
        for attr in ("test_dataloaders", "val_dataloaders"):
            loader = getattr(self.trainer, attr, None) if self.trainer else None
            if loader is None:
                continue
            # Unwrap CombinedLoader
            dl = loader
            if hasattr(dl, "flattened"):
                seq = dl.flattened
                if seq:
                    dl = seq[0]
            elif hasattr(dl, "iterables"):
                it = dl.iterables
                dl = it[0] if isinstance(it, (list, tuple)) else next(iter(it.values()))
            elif isinstance(dl, (list, tuple)) and dl:
                dl = dl[0]
            ds = getattr(dl, "dataset", None)
            if ds is not None:
                return ds
        # Fallback: instantiate from the datamodule's string spec.
        try:
            dm = getattr(self.trainer, "datamodule", None) if self.trainer else None
            if dm is not None and hasattr(dm, "hparams"):
                strs = dm.hparams.get("validation_datasets", [])
                if strs:
                    # The dataset string references ScanNetInstanceDataset by name
                    # — must be in scope. Import it locally so the eval works
                    # regardless of how the LitModule was instantiated.
                    from probing_vlm_vgm.data.components.scannet_instance_dataset import (  # noqa: F401
                        ScanNetInstanceDataset,
                    )
                    return eval(strs[0])
        except Exception:
            return None
        return None

    # ---------------------------------------------------------------- #
    # Visualization helpers (test-time only).
    #
    # Layout per scene (one PNG, rows = sampled frames):
    #     | RGB | GT instance mask | Pred cluster mask | Feature PCA |
    #
    # GT and Pred share a single random palette. With viz_match_pred_to_gt
    # the predicted cluster IDs are remapped to GT IDs via per-scene
    # Hungarian on IoU, so the same physical instance gets the same colour
    # on both sides → at-a-glance correctness.
    # ---------------------------------------------------------------- #
    @staticmethod
    def _build_instance_palette(max_id: int = 512, seed: int = 42) -> np.ndarray:
        """Random distinct RGB colours indexed by instance id. Index 0 = black."""
        rng = np.random.default_rng(seed)
        palette = rng.integers(40, 255, size=(max_id, 3), dtype=np.uint8)
        palette[0] = np.array([0, 0, 0], dtype=np.uint8)  # background / noise
        return palette

    @staticmethod
    def _colorize_labels(
        label_map: torch.Tensor,
        palette: np.ndarray,
        valid: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """(H, W) int64 → (H, W, 3) BGR uint8. -1/0 ⇒ black; ~valid ⇒ grey."""
        arr = label_map.cpu().numpy()
        arr = np.where(arr < 0, 0, arr).astype(np.int64) % len(palette)
        rgb = palette[arr]  # (H, W, 3) RGB
        bgr = rgb[..., ::-1].copy()
        if valid is not None:
            invalid = ~valid.cpu().numpy()
            bgr[invalid] = np.array([60, 60, 60], dtype=np.uint8)
        return bgr

    @staticmethod
    def _match_pred_to_gt_global(
        pred: torch.Tensor,  # (S, H, W) int64
        gt: torch.Tensor,    # (S, H, W) int64
        valid: torch.Tensor, # (S, H, W) bool
    ) -> torch.Tensor:
        """Per-scene Hungarian on IoU; remap pred cluster ids → matched GT ids.

        Predicted clusters that don't match any GT keep their original id
        (offset above the GT id range so colours stay distinct).
        """
        from scipy.optimize import linear_sum_assignment

        pred_np = pred.cpu().numpy()
        gt_np = gt.cpu().numpy()
        valid_np = valid.cpu().numpy()

        pred_ids = sorted(int(x) for x in np.unique(pred_np) if x >= 0)
        gt_ids = sorted(int(x) for x in np.unique(gt_np) if x > 0)
        if not pred_ids or not gt_ids:
            return pred.clone()

        iou = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float64)
        for i, pi in enumerate(pred_ids):
            pmask = (pred_np == pi) & valid_np
            for j, gj in enumerate(gt_ids):
                gmask = (gt_np == gj) & valid_np
                inter = np.logical_and(pmask, gmask).sum()
                if inter == 0:
                    continue
                union = np.logical_or(pmask, gmask).sum()
                if union > 0:
                    iou[i, j] = inter / union

        row_ind, col_ind = linear_sum_assignment(-iou)
        remap: Dict[int, int] = {}
        for r, c in zip(row_ind, col_ind):
            if iou[r, c] > 0:
                remap[pred_ids[r]] = gt_ids[c]

        # Unmatched pred ids → offset above max GT id so they stay distinct
        # but don't collide with any matched colour.
        offset = max(gt_ids) + 1
        for pi in pred_ids:
            if pi not in remap:
                remap[pi] = offset + pi

        out = np.zeros_like(pred_np)
        for pi, new_id in remap.items():
            out[pred_np == pi] = new_id
        return torch.from_numpy(out).long()

    @staticmethod
    def _make_pca_tensor(
        vfm_feat: torch.Tensor,  # (T, Hf, Wf, C)
        vfm_idx: torch.Tensor,   # (S,)
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Build ONE global PCA basis from the clip and return (S, 3, H, W) ∈ [0,1]."""
        T, hp, wp, C = vfm_feat.shape
        pca_imgs = vfm_pca_images(
            vfm_feat.reshape(-1, C).cpu(),
            tp=T, hp=hp, wp=wp, hi=H, wi=W,
            return_pil=False,
        )
        sel = [pca_imgs[int(t)] for t in vfm_idx.cpu()]
        tensors = [
            torch.from_numpy(im).permute(2, 0, 1).to(torch.float32) / 255.0
            for im in sel
        ]
        return torch.stack(tensors, dim=0)

    @staticmethod
    @torch.no_grad()
    def _pixel_level_labels(
        head_feat: torch.Tensor,    # (S, D, Hf, Wf) float32 L2-normalized
        labels_fr: np.ndarray,      # (S, Hf, Wf) int64; -1 = HDBSCAN noise
        mask_hw: tuple,             # (H_m, W_m) — target resolution
    ) -> np.ndarray:
        """Promote feat-res cluster labels to mask res via centroid 1-NN.

        1. Compute per-cluster centroid in head-embedding space (D-dim).
        2. Bilinear-upsample head_feat → mask res, re-L2-norm.
        3. For each pixel, assign nearest centroid by cosine similarity
           (= dot product on unit vectors).

        Viz-only: metric path uses the cheaper NN-upsampled feat-res labels
        from HDBSCAN. This helper yields smooth pixel boundaries that
        follow the head's learned embedding gradient. Memory bounded by
        per-frame compute (one (H_m*W_m, K) sim matrix at a time).
        """
        S, D, Hf, Wf = head_feat.shape
        H_m, W_m = mask_hw

        cluster_ids = sorted(int(x) for x in np.unique(labels_fr) if x >= 0)
        if not cluster_ids:
            # All noise → fall back to NN-upsample of labels_fr (still -1).
            labels_t = torch.from_numpy(labels_fr).unsqueeze(1).float()
            return (
                F.interpolate(labels_t, size=mask_hw, mode="nearest")
                .long().squeeze(1).numpy()
            )

        # Per-cluster centroid: mean of head feature over feat-res pixels
        # that HDBSCAN assigned to that cluster.
        feat_flat = head_feat.permute(0, 2, 3, 1).reshape(-1, D)  # (S*Hf*Wf, D)
        labels_flat = torch.from_numpy(labels_fr.reshape(-1))
        centroids = torch.stack([
            feat_flat[labels_flat == cid].mean(dim=0)
            for cid in cluster_ids
        ])  # (K, D)
        centroids = F.normalize(centroids, dim=1)

        # Bilinear-upsample features to mask resolution, then per-frame 1-NN.
        feat_up = F.interpolate(
            head_feat, size=mask_hw, mode="bilinear", align_corners=False
        )
        feat_up = F.normalize(feat_up, dim=1)  # (S, D, H_m, W_m)

        cid_array = np.array(cluster_ids, dtype=np.int64)
        pixel_labels = np.zeros((S, H_m, W_m), dtype=np.int64)
        for s in range(S):
            feat_s = feat_up[s].permute(1, 2, 0).reshape(-1, D)  # (H_m*W_m, D)
            sims = feat_s @ centroids.t()                          # (H_m*W_m, K)
            nearest = sims.argmax(dim=1).numpy()
            pixel_labels[s] = cid_array[nearest].reshape(H_m, W_m)
        return pixel_labels

    @rank_zero_only
    @torch.no_grad()
    def _visualize_instance_grid(
        self,
        scene_id: str,
        images: torch.Tensor,         # (S, 3, H, W) uint8
        gt_masks: torch.Tensor,       # (S, H, W) int64
        pred_labels_fr: torch.Tensor, # (S, Hf, Wf) int64 — HDBSCAN at feat res
        head_feat: torch.Tensor,      # (S, D, Hf, Wf) float32 — L2-norm head output
        valid_mask: torch.Tensor,     # (S, H, W) bool
        vfm_feat: torch.Tensor,       # (T, Hf, Wf, C)
        vfm_idx: torch.Tensor,        # (S,)
    ) -> None:
        S, _, H, W = images.shape

        # Subsample frames if too many — keep first/last for context.
        if S > self.viz_max_frames:
            idx = torch.linspace(0, S - 1, self.viz_max_frames).long()
            images = images[idx]
            gt_masks = gt_masks[idx]
            pred_labels_fr = pred_labels_fr[idx]
            head_feat = head_feat[idx]
            valid_mask = valid_mask[idx]
            vfm_idx = vfm_idx[idx]
            S = self.viz_max_frames

        # Promote feat-res labels to pixel-res via centroid 1-NN. Without
        # this we'd nearest-upsample the feat-res HDBSCAN labels straight
        # to mask res, producing visibly blocky (Hf/H_m × Wf/W_m) tiles in
        # the viz — the metric path stays at feat res either way.
        pred_masks = self._pixel_level_labels(
            head_feat=head_feat,
            labels_fr=pred_labels_fr.cpu().numpy(),
            mask_hw=(H, W),
        )
        pred_masks = torch.from_numpy(pred_masks)

        # Per-scene Hungarian remap so colours align across GT and Pred.
        if self.viz_match_pred_to_gt:
            pred_masks = self._match_pred_to_gt_global(
                pred_masks, gt_masks, valid_mask
            )

        # Feature PCA: vfm_feat is the original clip-level feature; we still
        # build the basis from the whole clip and select the rendered frames.
        pca_maps = self._make_pca_tensor(vfm_feat, vfm_idx, H, W)  # (S, 3, H, W)

        # Optionally save individual columns per view (handy for paper figures).
        save_dir = os.path.join(self.output_path, self.viz_output_subdir)
        os.makedirs(save_dir, exist_ok=True)
        ind_dir = None
        if self.viz_save_individual:
            ind_dir = os.path.join(save_dir, f"{scene_id}_individual")
            os.makedirs(ind_dir, exist_ok=True)

        rows = []
        for s in range(S):
            rgb = images[s].permute(1, 2, 0).cpu().numpy()  # (H, W, 3) uint8
            rgb_bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

            gt_bgr = self._colorize_labels(
                gt_masks[s], self._viz_palette, valid=valid_mask[s]
            )
            pr_bgr = self._colorize_labels(pred_masks[s], self._viz_palette)

            pca_rgb = (pca_maps[s].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pca_bgr = cv2.cvtColor(pca_rgb, cv2.COLOR_RGB2BGR)

            if ind_dir is not None:
                vd = os.path.join(ind_dir, f"view_{s:02d}")
                os.makedirs(vd, exist_ok=True)
                cv2.imwrite(os.path.join(vd, "rgb.png"), rgb_bgr)
                cv2.imwrite(os.path.join(vd, "gt.png"), gt_bgr)
                cv2.imwrite(os.path.join(vd, "pred.png"), pr_bgr)
                cv2.imwrite(os.path.join(vd, "pca.png"), pca_bgr)

            rows.append(np.concatenate([rgb_bgr, gt_bgr, pr_bgr, pca_bgr], axis=1))

        grid = np.concatenate(rows, axis=0)
        save_path = os.path.join(save_dir, f"{scene_id}_grid.png")
        cv2.imwrite(save_path, grid)

        # Mirror video_probe: also push to wandb if configured.
        for logger_ in self.loggers:
            if isinstance(logger_, WandbLogger):
                import wandb

                final_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(final_rgb)
                if pil.height > 0:
                    pil = pil.resize((int(pil.width * 384 / pil.height), 384))
                logger_.experiment.log(
                    {f"test-viz/{scene_id}": wandb.Image(pil)}
                )
                break

    # configure_optimizers and pretrained loading inherited from
    # BaseProbeModule (which uses strict=False).
