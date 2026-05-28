"""Fine-grained evaluation metrics for the L1 dense 3D probe (DL3DV / CO3D).

These are stratified breakdowns of the standard mean-error metrics. They expose
*where* a model's error concentrates (near vs far depth, frame distance,
percentile of failure, etc.) so we can build the abstraction-ladder /
training-bias story in §5.4 of vlm_vs_videomodel_3d_refine.md.

Conventions
-----------
- All inputs are torch tensors on any device. Returns are CPU floats / Python
  dicts of floats.
- Inputs are PER-SAMPLE (a single clip's S frames). Caller (test_step) loops
  over the batch.
- Empty buckets return NaN. Lightning's MeanMetric / wandb will skip NaN rows
  when aggregating across batches as long as we register them with
  nan_strategy="ignore" in self.log (Lightning default for `reduce_fx="mean"`).
- Alignment is the CALLER's responsibility:
    * Point-map metrics expect Umeyama-aligned `pred_pmap` (use
      `align_pmaps`).
    * Depth metrics do their own per-sample median scale alignment.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def _safe_mean(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(x.mean().item())


def _ensure_bool(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return mask
    return mask > 0


# --------------------------------------------------------------------------- #
# G1: depth bucket                                                            #
# --------------------------------------------------------------------------- #


def depth_bucket_error(
    aligned_pred_pmap: torch.Tensor,  # [S, 3, H, W]  Umeyama-aligned to GT
    gt_pmap: torch.Tensor,            # [S, 3, H, W]
    gt_depth: torch.Tensor,           # [S, 1, H, W]  positive depths
    valid_mask: torch.Tensor,         # [S, 1, H, W]  bool / float
    n_bins: int = 10,
    mode: str = "quantile",           # "quantile" | "log"
    log_bin_edges: Optional[Sequence[float]] = None,
    prefix: str = "depth_bin",
) -> Dict[str, float]:
    """Stratify per-pixel point-map L2 error by GT depth value.

    Quantile mode: split the sample's valid pixels into n_bins equal-population
    bins by GT depth percentile. This is the default — gives "near 10% / mid /
    far 10%" semantics that aggregate cleanly across samples with different
    depth scales.

    Log mode: fixed log-spaced edges (caller passes log_bin_edges), or
    auto-computed from the per-sample [min, max] if None. Use this when you
    want absolute-distance buckets for a specific scene type.
    """
    assert mode in ("quantile", "log"), f"unknown mode: {mode}"

    err = (aligned_pred_pmap - gt_pmap).norm(dim=1, keepdim=True)  # [S,1,H,W]
    valid = _ensure_bool(valid_mask)

    err_flat = err[valid]
    depth_flat = gt_depth[valid]
    if err_flat.numel() == 0:
        return {f"{prefix}_{i:02d}_err": float("nan") for i in range(n_bins)}

    if mode == "quantile":
        qs = torch.linspace(0, 1, n_bins + 1, device=depth_flat.device)
        edges = torch.quantile(depth_flat.float(), qs)
    else:
        if log_bin_edges is not None:
            edges = torch.tensor(
                list(log_bin_edges), device=depth_flat.device, dtype=torch.float
            )
        else:
            d_min = depth_flat.min().clamp_min(1e-6).item()
            d_max = max(depth_flat.max().item(), d_min + 1e-6)
            edges = torch.logspace(
                math.log10(d_min),
                math.log10(d_max),
                n_bins + 1,
                device=depth_flat.device,
            )

    out: Dict[str, float] = {}
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            sel = (depth_flat >= lo) & (depth_flat <= hi)
        else:
            sel = (depth_flat >= lo) & (depth_flat < hi)
        out[f"{prefix}_{i:02d}_err"] = _safe_mean(err_flat[sel])
    return out


# --------------------------------------------------------------------------- #
# D3: KITTI-style monocular depth metrics                                     #
# --------------------------------------------------------------------------- #


def standard_depth_metrics(
    pred_depth: torch.Tensor,         # [S, 1, H, W]
    gt_depth: torch.Tensor,           # [S, 1, H, W]
    valid_mask: torch.Tensor,         # [S, 1, H, W]  bool / float
    scale_align: str = "median",      # "median" | "none"
    eps: float = 1e-6,
    prefix: str = "depth",
) -> Dict[str, float]:
    """Standard monocular depth metrics with per-sample median scale alignment.

    Returns AbsRel, SqRel, RMSE, RMSE_log, log10, and δ < 1.25 / 1.25^2 / 1.25^3.
    These are the canonical KITTI/NYU metrics — "free credibility" relative to
    just reporting MSE.
    """
    keys = (
        "abs_rel", "sq_rel", "rmse", "rmse_log", "log10",
        "delta_1.25", "delta_1.25_sq", "delta_1.25_cu",
    )

    valid = _ensure_bool(valid_mask)
    p = pred_depth[valid].float().clamp_min(eps)
    g = gt_depth[valid].float().clamp_min(eps)
    if p.numel() == 0:
        return {f"{prefix}_{k}": float("nan") for k in keys}

    if scale_align == "median":
        s = (g.median() / p.median()).clamp(eps, 1.0 / eps)
        p = p * s

    ratio = torch.maximum(p / g, g / p)
    abs_rel = ((p - g).abs() / g).mean()
    sq_rel = ((p - g).pow(2) / g).mean()
    rmse = (p - g).pow(2).mean().sqrt()
    rmse_log = (p.log() - g.log()).pow(2).mean().sqrt()
    log10 = (p.log10() - g.log10()).abs().mean()
    d1 = (ratio < 1.25).float().mean()
    d2 = (ratio < 1.25 ** 2).float().mean()
    d3 = (ratio < 1.25 ** 3).float().mean()

    return {
        f"{prefix}_abs_rel": float(abs_rel.item()),
        f"{prefix}_sq_rel": float(sq_rel.item()),
        f"{prefix}_rmse": float(rmse.item()),
        f"{prefix}_rmse_log": float(rmse_log.item()),
        f"{prefix}_log10": float(log10.item()),
        f"{prefix}_delta_1.25": float(d1.item()),
        f"{prefix}_delta_1.25_sq": float(d2.item()),
        f"{prefix}_delta_1.25_cu": float(d3.item()),
    }


# --------------------------------------------------------------------------- #
# C2: error percentiles                                                       #
# --------------------------------------------------------------------------- #


def error_percentiles(
    err_per_pixel: torch.Tensor,      # any shape; per-pixel L2 error
    valid_mask: torch.Tensor,         # broadcastable to err_per_pixel
    qs: Sequence[float] = (0.5, 0.9, 0.99),
    prefix: str = "err",
) -> Dict[str, float]:
    """Quantiles (P50/P90/P99...) of per-pixel error within the valid mask.

    P50 is the median; P99 captures the heavy-tail behavior that mean-error
    hides. Big gap between P90 and P99 ⇒ catastrophic failure mode (a few
    pixels carrying huge error).
    """
    valid = _ensure_bool(valid_mask).expand_as(err_per_pixel)
    flat = err_per_pixel[valid].float()
    if flat.numel() == 0:
        return {f"{prefix}_p{int(q * 100):02d}": float("nan") for q in qs}

    out: Dict[str, float] = {}
    for q in qs:
        qt = torch.tensor(q, device=flat.device, dtype=flat.dtype)
        out[f"{prefix}_p{int(q * 100):02d}"] = float(torch.quantile(flat, qt).item())
    return out


# --------------------------------------------------------------------------- #
# T1: frame-distance bucket                                                   #
# --------------------------------------------------------------------------- #


def frame_distance_bucket(
    per_frame_err: torch.Tensor,      # [S]  scalar error per frame
    vfm_idx: torch.Tensor,            # [S]  frame indices in original clip
    ref_pos: int = 0,                 # which position in vfm_idx is the reference
    n_buckets: int = 4,
    prefix: str = "frame_dist",
) -> Dict[str, float]:
    """Bucket frames by |vfm_idx - vfm_idx[ref_pos]|; report mean error per bucket.

    The probe uses frame 0 as the reference frame for pose / point-map
    coordinates. As we move further from the reference, error is expected to
    grow — but the rate of growth differs across model classes (VLM vs Video).
    Plot the resulting fan-out for §5.4.3.
    """
    err = per_frame_err.float().reshape(-1)
    idx = vfm_idx.long().reshape(-1)
    if err.numel() == 0:
        return {f"{prefix}_b{i}_err": float("nan") for i in range(n_buckets)}

    ref = idx[ref_pos]
    dist = (idx - ref).abs().float()
    d_min, d_max = float(dist.min().item()), float(dist.max().item())
    if d_max - d_min < 1e-6:
        return {
            f"{prefix}_b0_err": float(err.mean().item()),
            **{f"{prefix}_b{i}_err": float("nan") for i in range(1, n_buckets)},
        }

    edges = torch.linspace(d_min, d_max, n_buckets + 1, device=dist.device)
    out: Dict[str, float] = {}
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        if i == n_buckets - 1:
            sel = (dist >= lo) & (dist <= hi)
        else:
            sel = (dist >= lo) & (dist < hi)
        out[f"{prefix}_b{i}_err"] = _safe_mean(err[sel])
    return out


# --------------------------------------------------------------------------- #
# convenience: run everything we have on one sample                           #
# --------------------------------------------------------------------------- #


def all_finegrained_for_sample(
    aligned_pred_pmap: torch.Tensor,   # [S, 3, H, W]
    gt_pmap: torch.Tensor,             # [S, 3, H, W]
    pred_depth: Optional[torch.Tensor],  # [S, 1, H, W]
    gt_depth: torch.Tensor,            # [S, 1, H, W]
    valid_mask: torch.Tensor,          # [S, 1, H, W]
    vfm_idx: torch.Tensor,             # [S]
    n_depth_bins: int = 10,
    n_frame_buckets: int = 4,
) -> Dict[str, float]:
    """Run all currently-implemented per-sample finegrained metrics.

    Returns a flat dict suitable for `self.log` in test_step. NaNs indicate
    empty buckets / invalid samples; keep them so per-batch keys stay aligned.
    """
    out: Dict[str, float] = {}

    # G1: depth bucket on Umeyama-aligned point-map L2 error
    out.update(
        depth_bucket_error(
            aligned_pred_pmap,
            gt_pmap,
            gt_depth,
            valid_mask,
            n_bins=n_depth_bins,
            mode="quantile",
            prefix="depth_bin",
        )
    )

    # C2: percentile of per-pixel point-map error
    err_pp = (aligned_pred_pmap - gt_pmap).norm(dim=1, keepdim=True)  # [S,1,H,W]
    out.update(
        error_percentiles(
            err_pp,
            valid_mask,
            qs=(0.5, 0.9, 0.99),
            prefix="pmap_err",
        )
    )

    # T1: frame-distance bucket on per-frame mean error
    valid = _ensure_bool(valid_mask)
    weights = valid.float()
    per_frame_err = (err_pp * weights).sum(dim=(1, 2, 3)) / (
        weights.sum(dim=(1, 2, 3)) + 1e-6
    )  # [S]
    out.update(
        frame_distance_bucket(
            per_frame_err,
            vfm_idx,
            ref_pos=0,
            n_buckets=n_frame_buckets,
            prefix="frame_dist",
        )
    )

    # D3: KITTI-style depth metrics (only if depth head was active)
    if pred_depth is not None:
        out.update(
            standard_depth_metrics(
                pred_depth,
                gt_depth,
                valid_mask,
                scale_align="median",
                prefix="depth",
            )
        )

    # Derived summary scalars per group. Single numbers per run for cross-model
    # sort / wandb tables; complement the per-bin / per-bucket fan-out.
    #   far_minus_near / far_over_near: absolute / relative near-to-far degradation
    #   mean / var: overall level / spread across all buckets
    _add_derived_summary(out, prefix="depth_bin", key_suffix="_err")
    _add_derived_summary(out, prefix="frame_dist", key_suffix="_err")

    # C2 tail-heaviness: P99 / P50 ratio. Bigger = longer tail (more catastrophic
    # outliers). Scale-invariant across models. Gaussian baseline ≈ 2.55.
    p50 = out.get("pmap_err_p50", float("nan"))
    p99 = out.get("pmap_err_p99", float("nan"))
    if not (math.isnan(p50) or math.isnan(p99)) and p50 > 1e-9:
        out["pmap_err_p99_over_p50"] = p99 / p50

    return out


def _add_derived_summary(
    out: Dict[str, float], prefix: str, key_suffix: str = "_err"
) -> None:
    """Mutate `out` in place: add `<prefix>_{far_minus_near, far_over_near,
    mean, var}` scalars derived from sorted `<prefix>_*<key_suffix>` keys.

    The `_err` suffix filter naturally excludes any derived scalars added
    here (which end in `_mean` / `_var` / `_minus_near` / `_over_near`).
    Skips emission if fewer than 2 valid (non-NaN) bucket values exist.
    """
    keys = sorted(
        k for k in out if k.startswith(f"{prefix}_") and k.endswith(key_suffix)
    )
    vals = [out[k] for k in keys if not math.isnan(out[k])]
    if len(vals) < 2:
        return

    near, far = out[keys[0]], out[keys[-1]]
    if not (math.isnan(near) or math.isnan(far)):
        out[f"{prefix}_far_minus_near"] = far - near
        out[f"{prefix}_far_over_near"] = (
            far / near if near > 1e-9 else float("nan")
        )
    mean = sum(vals) / len(vals)
    out[f"{prefix}_mean"] = mean
    out[f"{prefix}_var"] = sum((v - mean) ** 2 for v in vals) / len(vals)
