#!/usr/bin/env python
"""Shard-only ScanNet instance evaluation without Lightning Trainer.

This evaluator replaces the old "one Lightning process per checkpoint" eval
pool with one checkpoint at a time, sharded over all requested GPUs. HDBSCAN is
served by a single global CPU worker pool, so adding GPUs does not multiply the
number of HDBSCAN workers.

The first-pass optimisation is intentionally conservative: control process and
thread concurrency, but keep the existing feature, HDBSCAN, metric, viz, and
wandb semantics.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Thread limits must be set before importing numpy/sklearn/torch in spawned
# workers. setdefault lets callers override deliberately.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Subset

from probing_vlm_vgm.data.components.scannet_instance_dataset import ScanNetInstanceDataset
from probing_vlm_vgm.eval.instance_metric import aggregate_scene_metrics, hdbscan_cluster, t_miou_t_sr
from probing_vlm_vgm.losses import mvc_loss
from probing_vlm_vgm.utils.vis_utils import vfm_pca_images


VIDEO_MODEL_TOKENS = ("wan-t2v-1.3b", "opensora", "cogvideox-i2v-5b", "aether", "vjepa")


def _set_torch_thread_limits() -> None:
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _normalize_tss(value: str) -> str:
    value = value.strip()
    value = value[1:] if value.startswith("[") else value
    value = value[:-1] if value.endswith("]") else value
    return value.replace(" ", "")


def _run_regex(views: int) -> re.Pattern[str]:
    return re.compile(rf"_views{views}_bs([0-9]+)_(None|\[([0-9, ]+)\])$")


def _parse_run_name(run_name: str, views: int) -> Optional[Tuple[int, Optional[List[int]], str]]:
    """Legacy parser for old sweep-style run names.

    New release configs use short run names, so collect_runs first reads
    .hydra/config.yaml. This parser remains as a fallback for very old runs.
    """
    match = _run_regex(views).search(run_name)
    if not match:
        return None
    batch_size = int(match.group(1))
    tss_token = match.group(2)
    if tss_token == "None":
        return batch_size, None, "None"
    raw = match.group(3)
    target_spatial_size = [int(x) for x in raw.replace(" ", "").split(",") if x]
    return batch_size, target_spatial_size, f"[{raw}]"


def _is_video_model_run(run_name: str) -> bool:
    return any(f"_{tok}_" in run_name for tok in VIDEO_MODEL_TOKENS)


def _is_video_model_name(vfm_name: str) -> bool:
    return any(vfm_name == tok or vfm_name.startswith(tok) for tok in VIDEO_MODEL_TOKENS)


def _matches_vfm_filter(run_name: str, vfm_name: str, vfms: Sequence[str]) -> bool:
    if not vfms:
        return True
    return any(
        vfm_name == v
        or run_name.startswith(f"{v}_")
        or f"_{v}_" in run_name
        for v in vfms
    )


def _matches_layer_filter(run_name: str, layers: Sequence[int]) -> bool:
    if not layers:
        return True
    return any(f"_layer{layer}_" in run_name for layer in layers)


def _matches_bd_filter(run_name: str, bds: Sequence[int]) -> bool:
    if not bds:
        return True
    return any(f"_bd{bd}_" in run_name for bd in bds)


def _select_checkpoint(run_dir: Path) -> Optional[Path]:
    ckpt_dir = run_dir / "checkpoints"
    best = ckpt_dir / "best.ckpt"
    if best.is_file():
        return best
    epoch_ckpts = sorted(
        p for p in ckpt_dir.iterdir()
        if p.is_file() and p.name.startswith("epoch_") and p.suffix == ".ckpt"
    ) if ckpt_dir.is_dir() else []
    if len(epoch_ckpts) == 1:
        return epoch_ckpts[0]
    last = ckpt_dir / "last.ckpt"
    if last.is_file():
        return last
    return None


def _eval_run_name(train_run_name: str) -> str:
    return train_run_name.replace("scannet-instance", "scannet-instance-eval", 1)


def _load_run_cfg(run_dir: Path):
    return OmegaConf.load(run_dir / ".hydra" / "config.yaml")


def _run_params_from_cfg(
    run_dir: Path, run_name: str, views: int
) -> Optional[Tuple[int, Optional[List[int]], str, str]]:
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        parsed = _parse_run_name(run_name, views)
        if parsed is None:
            return None
        batch_size, target_spatial_size, tss_token = parsed
        return batch_size, target_spatial_size, tss_token, ""

    cfg = OmegaConf.load(cfg_path)
    if int(cfg.gt_num_frames) != views:
        return None
    batch_size = int(cfg.batch_size)
    vfm_name = str(getattr(cfg, "vfm_name", ""))
    target_spatial_size = OmegaConf.select(cfg, "target_spatial_size")
    if target_spatial_size is None:
        return batch_size, None, "None", vfm_name
    target_spatial_size = [int(x) for x in target_spatial_size]
    tss_token = "[" + ",".join(str(x) for x in target_spatial_size) + "]"
    return batch_size, target_spatial_size, tss_token, vfm_name


def _to_plain_dict(conf: Any) -> Dict[str, Any]:
    return OmegaConf.to_container(conf, resolve=True)  # type: ignore[return-value]


def collect_runs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"runs dir not found: {runs_dir}")

    wanted_tss = {_normalize_tss(v) for v in args.video_tss}
    found: List[Dict[str, Any]] = []
    skipped_done: List[str] = []
    skipped_vfm: List[str] = []
    skipped_layer: List[str] = []
    skipped_bd: List[str] = []
    skipped_tss: List[str] = []
    skipped_run_name: List[str] = []
    matched_run_names: List[str] = []

    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        run_name = run_dir.name
        parsed = _run_params_from_cfg(run_dir, run_name, args.views)
        if parsed is None:
            continue
        batch_size, target_spatial_size, tss_token, vfm_name = parsed

        if args.run_name:
            if run_name not in args.run_name:
                skipped_run_name.append(run_name)
                continue
            matched_run_names.append(run_name)

        if not _matches_vfm_filter(run_name, vfm_name, args.vfm):
            skipped_vfm.append(run_name)
            continue

        if not _matches_layer_filter(run_name, args.layer):
            skipped_layer.append(run_name)
            continue

        if not _matches_bd_filter(run_name, args.bd):
            skipped_bd.append(run_name)
            continue

        is_video_run = _is_video_model_name(vfm_name) or _is_video_model_run(run_name)
        if wanted_tss and is_video_run:
            got = "" if target_spatial_size is None else ",".join(str(x) for x in target_spatial_size)
            if got not in wanted_tss:
                skipped_tss.append(run_name)
                continue

        done_path = Path(args.done_dir) / f"{run_name}.done"
        if args.skip_done and not args.force and done_path.is_file():
            skipped_done.append(run_name)
            continue

        ckpt = _select_checkpoint(run_dir)
        if ckpt is None:
            print(f"[skip] {run_name}: no best/epoch/last checkpoint", file=sys.stderr)
            continue

        found.append(
            {
                "run_dir": str(run_dir),
                "run_name": run_name,
                "eval_run_name": _eval_run_name(run_name),
                "ckpt_path": str(ckpt),
                "batch_size": batch_size,
                "target_spatial_size": target_spatial_size,
                "tss_token": tss_token,
                "vfm_name": vfm_name,
            }
        )

    if args.limit_runs is not None:
        found = found[: args.limit_runs]

    if args.run_name:
        missing = [name for name in args.run_name if name not in matched_run_names]
        if missing:
            raise RuntimeError(
                "Missing requested --run-name value(s) under "
                f"{runs_dir} for views={args.views}: {missing}"
            )

    print(f"Found {len(found)} run(s) for views={args.views}")
    for item in found:
        print(f"  - {item['run_name']}")
    if skipped_done:
        print(f"Skipped done: {len(skipped_done)}")
    if skipped_vfm:
        print(f"Skipped by --vfm: {len(skipped_vfm)}")
    if skipped_layer:
        print(f"Skipped by --layer: {len(skipped_layer)}")
    if skipped_bd:
        print(f"Skipped by --bd: {len(skipped_bd)}")
    if skipped_tss:
        print(f"Skipped video-model by --video-tss: {len(skipped_tss)}")
    if skipped_run_name:
        print(f"Skipped by --run-name: {len(skipped_run_name)}")
    return found


def build_dataset_from_cfg(
    cfg: Any,
    *,
    target_spatial_size: Optional[List[int]],
    load_images: bool,
    pool_in_worker: bool = True,
) -> ScanNetInstanceDataset:
    return ScanNetInstanceDataset(
        root=str(cfg.data.data_root),
        root_vfm=str(cfg.data.feat_root),
        split="val",
        vfm_name=str(cfg.vfm_name),
        feat_postfix=str(cfg.feat_postfix),
        feat_pixalign=True,
        num_views=int(cfg.gt_num_frames),
        min_view_interval=5,
        context_len=76,
        query_idx_divisor=4,
        seed=0,
        target_spatial_size=target_spatial_size,
        pool_in_worker=pool_in_worker,
        load_images=load_images,
    )


def _build_instance_palette(max_id: int = 512, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    palette = rng.integers(40, 255, size=(max_id, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    return palette


def _colorize_labels(
    label_map: torch.Tensor,
    palette: np.ndarray,
    valid: Optional[torch.Tensor] = None,
) -> np.ndarray:
    arr = label_map.cpu().numpy()
    arr = np.where(arr < 0, 0, arr).astype(np.int64) % len(palette)
    rgb = palette[arr]
    bgr = rgb[..., ::-1].copy()
    if valid is not None:
        invalid = ~valid.cpu().numpy()
        bgr[invalid] = np.array([60, 60, 60], dtype=np.uint8)
    return bgr


def _match_pred_to_gt_global(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
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

    offset = max(gt_ids) + 1
    for pi in pred_ids:
        if pi not in remap:
            remap[pi] = offset + pi

    out = np.zeros_like(pred_np)
    for pi, new_id in remap.items():
        out[pred_np == pi] = new_id
    return torch.from_numpy(out).long()


def _pixel_level_labels(
    head_feat: torch.Tensor,
    labels_fr: np.ndarray,
    mask_hw: Tuple[int, int],
) -> np.ndarray:
    S, D, Hf, Wf = head_feat.shape
    cluster_ids = sorted(int(x) for x in np.unique(labels_fr) if x >= 0)
    if not cluster_ids:
        labels_t = torch.from_numpy(labels_fr).unsqueeze(1).float()
        return (
            F.interpolate(labels_t, size=mask_hw, mode="nearest")
            .long()
            .squeeze(1)
            .numpy()
        )

    feat_flat = head_feat.permute(0, 2, 3, 1).reshape(-1, D)
    labels_flat = torch.from_numpy(labels_fr.reshape(-1))
    centroids = torch.stack([
        feat_flat[labels_flat == cid].mean(dim=0)
        for cid in cluster_ids
    ])
    centroids = F.normalize(centroids, dim=1)

    feat_up = F.interpolate(head_feat, size=mask_hw, mode="bilinear", align_corners=False)
    feat_up = F.normalize(feat_up, dim=1)

    cid_array = np.array(cluster_ids, dtype=np.int64)
    pixel_labels = np.zeros((S, mask_hw[0], mask_hw[1]), dtype=np.int64)
    for s in range(S):
        feat_s = feat_up[s].permute(1, 2, 0).reshape(-1, D)
        sims = feat_s @ centroids.t()
        nearest = sims.argmax(dim=1).numpy()
        pixel_labels[s] = cid_array[nearest].reshape(mask_hw)
    return pixel_labels


def _make_pca_tensor(
    vfm_feat: torch.Tensor,
    vfm_idx: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    T, hp, wp, C = vfm_feat.shape
    pca_imgs = vfm_pca_images(
        vfm_feat.reshape(-1, C).cpu(),
        tp=T,
        hp=hp,
        wp=wp,
        hi=H,
        wi=W,
        return_pil=False,
    )
    tensors = [
        torch.from_numpy(pca_imgs[int(t)].copy()).permute(2, 0, 1).to(torch.float32) / 255.0
        for t in vfm_idx.cpu()
    ]
    return torch.stack(tensors, dim=0)


def save_instance_viz_grid(
    *,
    scene_id: str,
    images: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_labels_fr: torch.Tensor,
    head_feat: torch.Tensor,
    valid_mask: torch.Tensor,
    vfm_feat: torch.Tensor,
    vfm_idx: torch.Tensor,
    output_dir: Path,
    viz_output_subdir: str,
    viz_max_frames: int,
    viz_match_pred_to_gt: bool,
    viz_save_individual: bool,
    case_output_root: Optional[Path] = None,
    case_model_name: Optional[str] = None,
) -> Path:
    palette = _build_instance_palette()
    S, _, H, W = images.shape
    if S > viz_max_frames:
        idx = torch.linspace(0, S - 1, viz_max_frames).long()
        images = images[idx]
        gt_masks = gt_masks[idx]
        pred_labels_fr = pred_labels_fr[idx]
        head_feat = head_feat[idx]
        valid_mask = valid_mask[idx]
        vfm_idx = vfm_idx[idx]
        S = viz_max_frames

    pred_masks = torch.from_numpy(
        _pixel_level_labels(head_feat=head_feat, labels_fr=pred_labels_fr.cpu().numpy(), mask_hw=(H, W))
    )
    if viz_match_pred_to_gt:
        pred_masks = _match_pred_to_gt_global(pred_masks, gt_masks, valid_mask)

    pca_maps = _make_pca_tensor(vfm_feat, vfm_idx, H, W)

    save_dir = output_dir / viz_output_subdir
    save_dir.mkdir(parents=True, exist_ok=True)
    ind_dir = save_dir / f"{scene_id}_individual"
    if viz_save_individual:
        ind_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in range(S):
        rgb = images[s].permute(1, 2, 0).cpu().numpy()
        rgb_bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        gt_bgr = _colorize_labels(gt_masks[s], palette, valid=valid_mask[s])
        pr_bgr = _colorize_labels(pred_masks[s], palette)
        pca_rgb = (
            pca_maps[s].permute(1, 2, 0).cpu().numpy() * 255
        ).clip(0, 255).astype(np.uint8)
        pca_bgr = cv2.cvtColor(pca_rgb, cv2.COLOR_RGB2BGR)

        if viz_save_individual:
            vd = ind_dir / f"view_{s:02d}"
            vd.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vd / "rgb.png"), rgb_bgr)
            cv2.imwrite(str(vd / "gt.png"), gt_bgr)
            cv2.imwrite(str(vd / "pred.png"), pr_bgr)
            cv2.imwrite(str(vd / "pca.png"), pca_bgr)

        rows.append(np.concatenate([rgb_bgr, gt_bgr, pr_bgr, pca_bgr], axis=1))

    grid = np.concatenate(rows, axis=0)
    save_path = save_dir / f"{scene_id}_grid.png"
    cv2.imwrite(str(save_path), grid)
    if case_output_root is not None and case_model_name:
        case_dir = case_output_root / scene_id / case_model_name
        case_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(case_dir / "instance_grid.png"), grid)
        if viz_save_individual and ind_dir.is_dir():
            dst = case_dir / "instance_individual"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(ind_dir, dst)
    return save_path


def cluster_and_score(
    *,
    feat_flat: np.ndarray,
    masks: np.ndarray,
    valid: np.ndarray,
    fr_shape: Tuple[int, int, int],
    mask_hw: Tuple[int, int],
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    hdbscan_pca_dim: Optional[int],
    eval_iou_thresh: float,
    eval_ignore_ids: Tuple[int, ...],
    return_labels: bool = False,
) -> Dict[str, Any]:
    from threadpoolctl import threadpool_limits

    timings: Dict[str, float] = {}
    with threadpool_limits(limits=1):
        t0 = time.perf_counter()
        labels = hdbscan_cluster(
            feat_flat,
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            metric="euclidean",
            pca_dim=hdbscan_pca_dim,
        ).reshape(fr_shape)
        timings["cluster_sec"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        labels_up = (
            F.interpolate(
                torch.from_numpy(labels).unsqueeze(1).float(),
                size=mask_hw,
                mode="nearest",
            )
            .long()
            .squeeze(1)
            .numpy()
        )
        scene_metrics = t_miou_t_sr(
            labels_up,
            masks,
            valid,
            iou_thresh=eval_iou_thresh,
            ignore_ids=eval_ignore_ids,
        )
        timings["metric_sec"] = time.perf_counter() - t1
        scene_metrics["n_clusters"] = int((np.unique(labels) >= 0).sum())

    out: Dict[str, Any] = {"metrics": scene_metrics, "timings": timings}
    if return_labels:
        out["labels_fr"] = labels
    return out


def hdbscan_worker(worker_id: int, task_queue: Any, result_queue: Any) -> None:
    _set_torch_thread_limits()
    while True:
        task = task_queue.get()
        if task is None:
            break
        try:
            out = cluster_and_score(**task["payload"])
            result_queue.put(
                {
                    "type": "metric",
                    "source": "hdbscan",
                    "scene_id": task["scene_id"],
                    "metrics": out["metrics"],
                    "timings": out["timings"],
                }
            )
        except Exception:
            result_queue.put(
                {
                    "type": "error",
                    "source": f"hdbscan-{worker_id}",
                    "scene_id": task.get("scene_id"),
                    "traceback": traceback.format_exc(),
                }
            )


def _dataloader_worker_init(_: int) -> None:
    _set_torch_thread_limits()


def load_probe_from_checkpoint(probe_cfg: Dict[str, Any], ckpt_path: str, device: torch.device):
    probe = instantiate(OmegaConf.create(probe_cfg))
    # Lightning checkpoints include Hydra/OmegaConf objects, which PyTorch 2.6
    # rejects under the default weights_only=True loader.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    probe_state = {
        key[len("probe."):]: value
        for key, value in state_dict.items()
        if key.startswith("probe.")
    }
    missing, unexpected = probe.load_state_dict(probe_state, strict=False)
    if unexpected:
        print(f"[warn] unexpected checkpoint keys for probe: {unexpected[:8]}", file=sys.stderr)
    if missing:
        print(f"[warn] missing checkpoint keys for probe: {missing[:8]}", file=sys.stderr)
    if hasattr(probe, "backbone") and hasattr(probe.backbone, "gradient_checkpointing"):
        probe.backbone.gradient_checkpointing = False
    probe.eval().requires_grad_(False)
    probe.to(device)
    return probe


def forward_instance(
    probe: torch.nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    use_bf16: bool,
) -> torch.Tensor:
    vfm_feat = batch["vfm_feat"].to(device, non_blocking=True)
    vfm_feat = vfm_feat.permute(0, 1, 4, 2, 3).contiguous()
    if "target_spatial_size" in batch:
        target_hw = tuple(int(x) for x in batch["target_spatial_size"][0].tolist())
        B, S, C, H, W = vfm_feat.shape
        if (H, W) != target_hw:
            x = vfm_feat.reshape(B * S, C, H, W)
            x = F.adaptive_avg_pool2d(x, output_size=target_hw)
            vfm_feat = x.reshape(B, S, C, target_hw[0], target_hw[1])

    B, S, _, Hf, Wf = vfm_feat.shape
    video_shape = (B, S, 3, Hf * 14, Wf * 14)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)
    with autocast:
        preds = probe(vfm_feat, video_shape)
    return preds["instance"]


def gpu_worker(
    *,
    rank: int,
    gpu_id: str,
    run_info: Dict[str, Any],
    cfg_path: str,
    probe_cfg: Dict[str, Any],
    dataset_kwargs: Dict[str, Any],
    indices: List[int],
    viz_scene_ids: List[str],
    output_dir: str,
    task_queue: Any,
    result_queue: Any,
    batch_size: int,
    num_workers: int,
    use_bf16: bool,
    compute_loss: bool,
    metric_cfg: Dict[str, Any],
    viz_cfg: Dict[str, Any],
    case_output_root: str = "",
    case_model_name: str = "",
) -> None:
    try:
        _set_torch_thread_limits()
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)

        probe = load_probe_from_checkpoint(probe_cfg, run_info["ckpt_path"], device)
        dataset = ScanNetInstanceDataset(**dataset_kwargs)
        subset = Subset(dataset, indices)

        loader_kwargs: Dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": False,
            "drop_last": False,
            "num_workers": num_workers,
            "pin_memory": device.type == "cuda",
            "worker_init_fn": _dataloader_worker_init,
        }
        if num_workers > 0:
            loader_kwargs.update(
                {
                    "persistent_workers": True,
                    "multiprocessing_context": "forkserver",
                    "prefetch_factor": 2,
                }
            )
        loader = DataLoader(subset, **loader_kwargs)

        viz_set = set(viz_scene_ids)
        loss_items: List[Dict[str, float]] = []
        forward_sec = 0.0
        enqueue_wait_sec = 0.0
        n_batches = 0

        for batch in loader:
            n_batches += 1
            t_forward = time.perf_counter()
            with torch.no_grad():
                feats = forward_instance(probe, batch, device, use_bf16=use_bf16)
                if compute_loss:
                    masks_dev = batch["instance_masks"].to(device, non_blocking=True)
                    valid_dev = batch["valid_mask"].to(device, non_blocking=True)
                    B, S, D, Hf, Wf = feats.shape
                    mask_hw = tuple(int(x) for x in masks_dev.shape[-2:])
                    x = feats.reshape(B * S, D, Hf, Wf)
                    x = F.interpolate(x, size=mask_hw, mode="bilinear", align_corners=False)
                    x = F.normalize(x, dim=1).reshape(B, S, D, mask_hw[0], mask_hw[1])
                    out = mvc_loss(
                        x,
                        masks_dev,
                        valid_mask=valid_dev,
                        num_samples=int(metric_cfg["mvc_num_samples"]),
                        margin=float(metric_cfg["mvc_margin"]),
                        lambda_pull=float(metric_cfg["mvc_lambda_pull"]),
                        lambda_push=float(metric_cfg["mvc_lambda_push"]),
                    )
                    loss_items.append(
                        {
                            "loss": float(out["loss"].detach().cpu()),
                            "loss_pull": float(out["loss_pull"].detach().cpu()),
                            "loss_push": float(out["loss_push"].detach().cpu()),
                        }
                    )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_sec += time.perf_counter() - t_forward

            feats_cpu = feats.detach().to(dtype=torch.float32).cpu().numpy()
            masks_np64 = batch["instance_masks"].numpy()
            max_mask = int(masks_np64.max()) if masks_np64.size else 0
            if max_mask >= 32768:
                raise ValueError(f"instance mask id {max_mask} exceeds int16 range")
            masks_np = masks_np64.astype(np.int16, copy=False)
            valid_np = batch["valid_mask"].numpy()
            B, S, D, Hf, Wf = feats_cpu.shape
            fr_shape = (S, Hf, Wf)
            mask_hw_np = tuple(int(x) for x in masks_np.shape[-2:])

            for b in range(B):
                scene_id = str(batch["scene_id"][b])
                feat_b = np.transpose(feats_cpu[b], (0, 2, 3, 1)).reshape(-1, D)
                payload = {
                    "feat_flat": feat_b,
                    "masks": masks_np[b],
                    "valid": valid_np[b],
                    "fr_shape": fr_shape,
                    "mask_hw": mask_hw_np,
                    **metric_cfg["hdbscan_metric_kwargs"],
                }

                if scene_id in viz_set and "images" in batch:
                    out = cluster_and_score(**payload, return_labels=True)
                    viz_path = save_instance_viz_grid(
                        scene_id=scene_id,
                        images=batch["images"][b],
                        gt_masks=batch["instance_masks"][b],
                        pred_labels_fr=torch.from_numpy(out["labels_fr"]),
                        head_feat=torch.from_numpy(feats_cpu[b]),
                        valid_mask=batch["valid_mask"][b],
                        vfm_feat=batch["vfm_feat"][b],
                        vfm_idx=batch["vfm_idx"][b],
                        output_dir=Path(output_dir),
                        case_output_root=Path(case_output_root) if case_output_root else None,
                        case_model_name=case_model_name,
                        **viz_cfg,
                    )
                    result_queue.put(
                        {
                            "type": "metric",
                            "source": f"gpu-{gpu_id}-viz",
                            "scene_id": scene_id,
                            "metrics": out["metrics"],
                            "timings": out["timings"],
                            "viz_path": str(viz_path),
                        }
                    )
                else:
                    t_put = time.perf_counter()
                    task_queue.put({"scene_id": scene_id, "payload": payload})
                    enqueue_wait_sec += time.perf_counter() - t_put

        result_queue.put(
            {
                "type": "worker_done",
                "source": f"gpu-{gpu_id}",
                "rank": rank,
                "n_indices": len(indices),
                "n_batches": n_batches,
                "loss_items": loss_items,
                "forward_sec": forward_sec,
                "enqueue_wait_sec": enqueue_wait_sec,
            }
        )
    except Exception:
        result_queue.put(
            {
                "type": "error",
                "source": f"gpu-{gpu_id}",
                "traceback": traceback.format_exc(),
            }
        )


def shard_indices(indices: Sequence[int], n_shards: int) -> List[List[int]]:
    return [list(indices[i::n_shards]) for i in range(n_shards)]


def choose_viz_scene_ids(dataset: ScanNetInstanceDataset, indices: Sequence[int], n: int, seed: int) -> List[str]:
    if n <= 0:
        return []
    if not indices:
        return []
    rng = np.random.default_rng(seed)
    n = min(n, len(indices))
    picked_positions = sorted(int(i) for i in rng.choice(len(indices), size=n, replace=False))
    return [dataset.scenes[indices[pos]][1] for pos in picked_positions]


def mean_or_nan(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def evaluate_one_run(args: argparse.Namespace, run_info: Dict[str, Any]) -> Dict[str, Any]:
    t_run = time.perf_counter()
    run_dir = Path(run_info["run_dir"])
    cfg = _load_run_cfg(run_dir)
    output_dir = Path(args.output_root) / "runs" / run_info["eval_run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    target_spatial_size = run_info["target_spatial_size"]
    dataset_for_index = build_dataset_from_cfg(
        cfg,
        target_spatial_size=target_spatial_size,
        load_images=False,
        pool_in_worker=True,
    )
    all_indices = list(range(len(dataset_for_index)))
    if args.scene_id:
        wanted_scene_ids = set(args.scene_id)
        all_indices = [
            idx for idx in all_indices
            if dataset_for_index.scenes[idx][1] in wanted_scene_ids
        ]
    if args.limit_scenes is not None:
        all_indices = all_indices[: args.limit_scenes]
    if not all_indices:
        raise RuntimeError("No scenes selected for evaluation")

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        gpu_ids = ["0"]
    shards = shard_indices(all_indices, len(gpu_ids))

    viz_scene_ids = []
    if args.load_images:
        if args.viz_scene_id:
            selected = {dataset_for_index.scenes[idx][1] for idx in all_indices}
            viz_scene_ids = [scene_id for scene_id in args.viz_scene_id if scene_id in selected]
        elif args.scene_id:
            viz_scene_ids = list(args.scene_id)
        else:
            viz_scene_ids = choose_viz_scene_ids(
                dataset_for_index,
                all_indices,
                args.viz_random_n,
                args.viz_random_seed,
            )

    probe_cfg = _to_plain_dict(cfg.model.probe)
    dataset_kwargs = {
        "root": str(cfg.data.data_root),
        "root_vfm": str(cfg.data.feat_root),
        "split": "val",
        "vfm_name": str(cfg.vfm_name),
        "feat_postfix": str(cfg.feat_postfix),
        "feat_pixalign": True,
        "num_views": int(cfg.gt_num_frames),
        "min_view_interval": 5,
        "context_len": 76,
        "query_idx_divisor": 4,
        "seed": 0,
        "target_spatial_size": target_spatial_size,
        "pool_in_worker": True,
        "load_images": bool(args.load_images),
    }

    metric_cfg = {
        "mvc_num_samples": int(cfg.model.mvc_num_samples),
        "mvc_margin": float(cfg.model.mvc_margin),
        "mvc_lambda_pull": float(cfg.model.mvc_lambda_pull),
        "mvc_lambda_push": float(cfg.model.mvc_lambda_push),
        "hdbscan_metric_kwargs": {
            "hdbscan_min_cluster_size": int(cfg.model.hdbscan_min_cluster_size),
            "hdbscan_min_samples": int(cfg.model.hdbscan_min_samples),
            "hdbscan_pca_dim": None if cfg.model.hdbscan_pca_dim is None else int(cfg.model.hdbscan_pca_dim),
            "eval_iou_thresh": float(cfg.model.eval_iou_thresh),
            "eval_ignore_ids": tuple(int(x) for x in cfg.model.eval_ignore_ids),
        },
    }
    viz_cfg = {
        "viz_output_subdir": str(cfg.model.viz_output_subdir),
        "viz_max_frames": int(args.viz_max_frames),
        "viz_match_pred_to_gt": bool(cfg.model.viz_match_pred_to_gt),
        "viz_save_individual": bool(args.viz_save_individual),
    }

    if args.dry_run:
        print(json.dumps({
            "run": run_info["run_name"],
            "ckpt": run_info["ckpt_path"],
            "output_dir": str(output_dir),
            "n_scenes": len(all_indices),
            "gpus": gpu_ids,
            "shards": [len(s) for s in shards],
            "viz_scene_ids": viz_scene_ids,
        }, indent=2))
        return {}

    ctx = mp.get_context(args.mp_context)
    queue_size = args.hdbscan_queue_size or max(args.hdbscan_workers * 2, len(gpu_ids))
    task_queue = ctx.Queue(maxsize=queue_size)
    result_queue = ctx.Queue()

    hdbscan_procs = [
        ctx.Process(target=hdbscan_worker, args=(i, task_queue, result_queue), daemon=False)
        for i in range(args.hdbscan_workers)
    ]
    for proc in hdbscan_procs:
        proc.start()

    gpu_procs = []
    for rank, (gpu_id, shard) in enumerate(zip(gpu_ids, shards)):
        if not shard:
            continue
        proc = ctx.Process(
            target=gpu_worker,
            kwargs={
                "rank": rank,
                "gpu_id": gpu_id,
                "run_info": run_info,
                "cfg_path": str(run_dir / ".hydra" / "config.yaml"),
                "probe_cfg": probe_cfg,
                "dataset_kwargs": dataset_kwargs,
                "indices": shard,
                "viz_scene_ids": viz_scene_ids,
                "output_dir": str(output_dir),
                "task_queue": task_queue,
                "result_queue": result_queue,
                "batch_size": int(args.batch_size or run_info["batch_size"]),
                "num_workers": int(args.num_workers),
                "use_bf16": args.precision == "bf16-mixed",
                "compute_loss": not args.skip_loss,
                "metric_cfg": metric_cfg,
                "viz_cfg": viz_cfg,
                "case_output_root": str(args.case_output_root or ""),
                "case_model_name": run_info["run_name"],
            },
            daemon=False,
        )
        gpu_procs.append(proc)
        proc.start()

    expected = len(all_indices)
    per_scene: List[Dict[str, float]] = []
    per_scene_records: List[Dict[str, Any]] = []
    viz_paths: List[str] = []
    errors: List[Dict[str, Any]] = []
    worker_done = 0
    loss_items: List[Dict[str, float]] = []
    worker_stats: List[Dict[str, Any]] = []
    n_metrics = 0
    last_print = time.perf_counter()

    try:
        while n_metrics < expected:
            try:
                msg = result_queue.get(timeout=10)
            except queue.Empty:
                live = [p.is_alive() for p in gpu_procs]
                if not any(live) and worker_done < len(gpu_procs):
                    errors.append({"type": "error", "source": "main", "traceback": "all GPU workers exited before reporting done"})
                    break
                print(f"[progress] {run_info['run_name']} {n_metrics}/{expected} scenes...")
                continue

            if msg["type"] == "metric":
                n_metrics += 1
                metrics = msg["metrics"]
                per_scene.append(metrics)
                per_scene_records.append(
                    {
                        "scene_id": msg.get("scene_id"),
                        "source": msg.get("source"),
                        "metrics": metrics,
                        "timings": msg.get("timings", {}),
                        "viz_path": msg.get("viz_path"),
                    }
                )
                if msg.get("viz_path"):
                    viz_paths.append(msg["viz_path"])
                now = time.perf_counter()
                if now - last_print >= args.progress_interval:
                    last_print = now
                    elapsed = now - t_run
                    rate = n_metrics / max(elapsed, 1e-9)
                    print(
                        f"[progress] {run_info['run_name']} "
                        f"{n_metrics}/{expected} scenes ({rate:.2f} scene/s)"
                    )
            elif msg["type"] == "worker_done":
                worker_done += 1
                loss_items.extend(msg.get("loss_items", []))
                worker_stats.append(msg)
                print(
                    f"[worker-done] {msg['source']} "
                    f"scenes={msg.get('n_indices')} batches={msg.get('n_batches')} "
                    f"forward={msg.get('forward_sec', 0):.1f}s "
                    f"queue_wait={msg.get('enqueue_wait_sec', 0):.1f}s"
                )
            elif msg["type"] == "error":
                errors.append(msg)
                print(f"[error] {msg.get('source')} scene={msg.get('scene_id')}", file=sys.stderr)
                print(msg.get("traceback", ""), file=sys.stderr)
                break

        for proc in gpu_procs:
            proc.join(timeout=5)
            if proc.exitcode not in (0, None):
                errors.append({"source": f"gpu-proc-{proc.pid}", "traceback": f"exitcode={proc.exitcode}"})

        # Metrics can arrive before the worker_done bookkeeping message,
        # especially for tiny tests or viz-only scenes. Drain cheap status
        # messages after all GPU workers have exited so timings are complete.
        while True:
            try:
                msg = result_queue.get_nowait()
            except queue.Empty:
                break
            if msg.get("type") == "worker_done":
                worker_done += 1
                loss_items.extend(msg.get("loss_items", []))
                worker_stats.append(msg)
            elif msg.get("type") == "error":
                errors.append(msg)
    finally:
        for _ in hdbscan_procs:
            task_queue.put(None)
        for proc in hdbscan_procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()

    if errors:
        err_path = output_dir / "errors.json"
        with err_path.open("w") as f:
            json.dump(errors, f, indent=2)
        raise RuntimeError(f"{run_info['run_name']} failed; see {err_path}")

    if n_metrics != expected:
        raise RuntimeError(f"{run_info['run_name']} incomplete: {n_metrics}/{expected} scene metrics")

    agg = aggregate_scene_metrics(
        per_scene,
        weight_by_instances=bool(cfg.model.weight_metrics_by_instances),
    )
    n_clusters_mean = float(np.mean([p.get("n_clusters", 0) for p in per_scene]))
    metrics_out = {
        "val/t_miou": agg["t_miou"],
        "val/t_sr": agg["t_sr"],
        "val/n_clusters_mean": n_clusters_mean,
        "val/n_gt_instances_mean": float(agg["n_gt_instances_total"]) / max(agg["n_scenes"], 1),
        "val/n_scenes": agg["n_scenes"],
        "val/n_gt_instances_total": agg["n_gt_instances_total"],
    }
    if loss_items:
        metrics_out.update(
            {
                "val/loss": mean_or_nan([x["loss"] for x in loss_items]),
                "val/loss_mvc": mean_or_nan([x["loss"] for x in loss_items]),
                "val/loss_pull": mean_or_nan([x["loss_pull"] for x in loss_items]),
                "val/loss_push": mean_or_nan([x["loss_push"] for x in loss_items]),
            }
        )

    timing_out = {
        "total_sec": time.perf_counter() - t_run,
        "scene_sec_mean_cluster": mean_or_nan([r["timings"].get("cluster_sec", float("nan")) for r in per_scene_records]),
        "scene_sec_mean_metric": mean_or_nan([r["timings"].get("metric_sec", float("nan")) for r in per_scene_records]),
        "worker_stats": worker_stats,
        "hdbscan_workers": args.hdbscan_workers,
        "gpus": gpu_ids,
        "shards": [len(s) for s in shards],
    }

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics_out, f, indent=2)
    with (output_dir / "timings.json").open("w") as f:
        json.dump(timing_out, f, indent=2)
    with (output_dir / "per_scene_metrics.jsonl").open("w") as f:
        for rec in per_scene_records:
            f.write(json.dumps(rec) + "\n")

    if args.use_wandb:
        log_to_wandb(args, cfg, run_info, output_dir, metrics_out, timing_out, viz_paths)

    done_dir = Path(args.done_dir)
    done_dir.mkdir(parents=True, exist_ok=True)
    done_path = done_dir / f"{run_info['run_name']}.done"
    with done_path.open("w") as f:
        f.write(f"run_name={run_info['run_name']}\n")
        f.write(f"eval_run_name={run_info['eval_run_name']}\n")
        f.write(f"finished_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        f.write(f"output_dir={output_dir}\n")
        f.write(f"ckpt_path={run_info['ckpt_path']}\n")

    print(
        f"[done] {run_info['run_name']} "
        f"t_miou={metrics_out['val/t_miou']:.4f} "
        f"t_sr={metrics_out['val/t_sr']:.4f} "
        f"time={timing_out['total_sec']:.1f}s"
    )
    return {"metrics": metrics_out, "timings": timing_out, "output_dir": str(output_dir)}


def log_to_wandb(
    args: argparse.Namespace,
    cfg: Any,
    run_info: Dict[str, Any],
    output_dir: Path,
    metrics: Dict[str, Any],
    timings: Dict[str, Any],
    viz_paths: Sequence[str],
) -> None:
    import wandb

    project = args.project or str(cfg.logger.wandb.project)
    run = wandb.init(
        project=project,
        name=run_info["eval_run_name"],
        group="scannet-instance-eval",
        dir=str(output_dir),
        config={
            "train_run_name": run_info["run_name"],
            "ckpt_path": run_info["ckpt_path"],
            "gpus": args.gpus,
            "hdbscan_workers": args.hdbscan_workers,
            "sharded_eval": True,
        },
    )
    try:
        wandb.log({**metrics, "runtime/total_sec": timings["total_sec"]})
        for path in viz_paths:
            scene_id = Path(path).name.replace("_grid.png", "")
            img = Image.open(path)
            if img.height > 0:
                img = img.resize((int(img.width * 384 / img.height), 384))
            wandb.log({f"test-viz/{scene_id}": wandb.Image(img)})
    finally:
        run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=int, required=True)
    parser.add_argument("--runs-dir", default="logs/scannet-instance/runs")
    parser.add_argument("--output-root", default="logs/scannet-instance-eval-sharded")
    parser.add_argument("--done-dir", default=None)
    parser.add_argument("--video-tss", action="append", default=[])
    parser.add_argument("--run-name", action="append", default=[], help="Exact run directory basename to evaluate. Pass multiple times.")
    parser.add_argument("--vfm", action="append", default=[])
    parser.add_argument("--layer", action="append", type=int, default=[])
    parser.add_argument("--bd", action="append", type=int, default=[], help="Only include runs with matching _bdN_ backbone depth suffix")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--hdbscan-workers", type=int, default=8)
    parser.add_argument("--hdbscan-queue-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers per GPU worker")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-GPU batch size")
    parser.add_argument("--precision", choices=["bf16-mixed", "fp32"], default="bf16-mixed")
    parser.add_argument("--skip-loss", action="store_true", help="Skip MVC loss and only compute clustering metrics")
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-runs", type=int, default=None)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument(
        "--scene-id",
        action="append",
        default=[],
        help="Only evaluate these scene_id values. Pass multiple times.",
    )
    parser.add_argument("--project", default="")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--load-images", action="store_true")
    parser.add_argument(
        "--viz-scene-id",
        action="append",
        default=[],
        help="Explicit scene_id to visualize. Pass multiple times. Requires --load-images.",
    )
    parser.add_argument("--viz-random-n", type=int, default=3)
    parser.add_argument("--viz-random-seed", type=int, default=0)
    parser.add_argument("--viz-max-frames", type=int, default=8)
    parser.add_argument("--viz-save-individual", action="store_true")
    parser.add_argument(
        "--case-output-root",
        default="",
        help="Optional extra output root organized as <scene_id>/<run_name>/ for case-study assets",
    )
    parser.add_argument("--mp-context", choices=["spawn", "forkserver"], default="spawn")
    parser.add_argument("--progress-interval", type=float, default=30.0)
    args = parser.parse_args()
    if args.done_dir is None:
        args.done_dir = str(Path(args.output_root) / "done")
    args.use_wandb = not args.no_wandb and (bool(args.project) or not args.dry_run)
    return args


def main() -> None:
    _set_torch_thread_limits()
    args = parse_args()
    runs = collect_runs(args)
    if not runs:
        print("No runs to evaluate.")
        return

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    all_results = []
    for run_info in runs:
        all_results.append(evaluate_one_run(args, run_info))

    if not args.dry_run:
        summary_path = Path(args.output_root) / "last_summary.json"
        with summary_path.open("w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
