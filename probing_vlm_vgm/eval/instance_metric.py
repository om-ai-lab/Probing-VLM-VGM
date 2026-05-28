"""Instance-clustering metrics for Setup A, aligned with IGGT (Li et al. 2025).

Metrics:
    - T-mIoU (Temporal mIoU):   mean IoU of matched (pred cluster, GT instance)
                                 pairs, computed on cross-view aggregate masks.
    - T-SR   (Temporal Success Rate): fraction of GT instances whose matched
                                 pred cluster has per-view IoU > threshold in
                                 *every view where the GT instance appears*.

HDBSCAN wrapper defaults mirror IGGT-style scene clustering (min_cluster_size=30,
min_samples=5) but are configurable.

Notes on pred↔GT matching:
    We match each GT instance to the pred cluster with highest global
    (cross-view) IoU. The same pred cluster can be chosen by multiple GT
    instances (greedy, non-unique). This is the simplest interpretation and
    matches what IGGT effectively does in their pipeline. A Hungarian option
    can be added later if needed.

On GT filtering:
    `ignore_ids` lists GT IDs that should not be counted as instances (e.g.
    ID 0 if it denotes background / unannotated region in ScanNet). These
    pixels are still counted in the denominator of each pred IoU (they reduce
    the IoU of predictions that happen to land there), which is the standard
    convention.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# HDBSCAN wrapper
# ---------------------------------------------------------------------------
def hdbscan_cluster(
    feats: np.ndarray,
    min_cluster_size: int = 30,
    min_samples: int = 5,
    metric: str = "euclidean",
    pca_dim: Optional[int] = None,
) -> np.ndarray:
    """Cluster per-pixel features with HDBSCAN.

    Args:
        feats: (N, D) float array. Should be L2-normalized if using euclidean
            distance over features from the unit hypersphere.
        min_cluster_size: HDBSCAN's min_cluster_size.
        min_samples: HDBSCAN's min_samples.
        metric: HDBSCAN distance metric.
        pca_dim: If set and < D, PCA-reduce feats to this dim before
            clustering. HDBSCAN's kd-tree degrades to O(N²) above D≈16, so
            for our MVC-trained 32-dim embeddings (whose discriminative
            structure lives in a much lower-rank subspace) projecting to
            ~8 dims gives ~2x wall-clock speedup at <0.002 t_miou drop —
            verified via scripts/compare_hdbscan_pca.py on Qwen3VL
            (Δ=-0.0001 over 50 scenes) and OpenSora (Δ=-0.0018).

    Returns:
        labels: (N,) int array with cluster IDs; -1 denotes HDBSCAN noise.
    """
    try:
        import hdbscan
    except ImportError as e:
        raise ImportError(
            "hdbscan is required for instance clustering. "
            "Install via `pip install hdbscan`."
        ) from e

    if pca_dim is not None and feats.shape[1] > pca_dim:
        from sklearn.decomposition import PCA
        feats = PCA(n_components=pca_dim, random_state=0).fit_transform(feats)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    labels = clusterer.fit_predict(feats)
    return labels.astype(np.int64)


# ---------------------------------------------------------------------------
# Matching + IoU primitives
# ---------------------------------------------------------------------------
def _pair_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Boolean IoU between two flat masks. Returns 0 if union is empty."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _build_masks_per_view(
    pred_labels: np.ndarray,
    gt_ids: np.ndarray,
    valid_mask: np.ndarray,
    ignore_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """Shape labels into per-view masks.

    Args:
        pred_labels: (S, H, W) int
        gt_ids:      (S, H, W) int
        valid_mask:  (S, H, W) bool
        ignore_ids:  GT IDs to ignore (filtered out of GT instance list)

    Returns:
        pred_per_view: (S, P) where P = num unique pred IDs (excluding -1).
                       Bool mask per (view, pred_cluster).
        gt_per_view:   (S, G) where G = num unique valid GT IDs.
        pred_ids:  list of P pred IDs (order matches pred_per_view axis 1)
        gt_ids_list: list of G GT IDs (order matches gt_per_view axis 1)
    """
    S, H, W = pred_labels.shape
    assert gt_ids.shape == (S, H, W)
    assert valid_mask.shape == (S, H, W)

    # Collect unique IDs (exclude -1 / ignored)
    pred_unique = np.unique(pred_labels[valid_mask])
    pred_unique = pred_unique[pred_unique != -1]

    gt_unique = np.unique(gt_ids[valid_mask])
    gt_unique = np.array([g for g in gt_unique if g not in ignore_ids], dtype=gt_unique.dtype)

    P = len(pred_unique)
    G = len(gt_unique)

    # Build per-view flat masks: shape (S, H*W) bool
    valid_flat = valid_mask.reshape(S, -1)

    pred_per_view = np.zeros((S, P, H * W), dtype=bool)
    for p_idx, p_id in enumerate(pred_unique):
        mask = (pred_labels.reshape(S, -1) == p_id) & valid_flat
        pred_per_view[:, p_idx] = mask

    gt_per_view = np.zeros((S, G, H * W), dtype=bool)
    for g_idx, g_id in enumerate(gt_unique):
        mask = (gt_ids.reshape(S, -1) == g_id) & valid_flat
        gt_per_view[:, g_idx] = mask

    return pred_per_view, gt_per_view, list(pred_unique), list(gt_unique)


def _cross_view_iou_matrix(
    pred_per_view: np.ndarray, gt_per_view: np.ndarray
) -> np.ndarray:
    """Compute (G, P) IoU matrix using cross-view aggregate masks.

    Args:
        pred_per_view: (S, P, HW) bool
        gt_per_view:   (S, G, HW) bool

    Returns:
        iou: (G, P) float
    """
    # Aggregate across views (OR)
    pred_agg = pred_per_view.any(axis=0)  # (P, HW)
    gt_agg = gt_per_view.any(axis=0)  # (G, HW)

    # Intersection and union via matrix multiplication on bools -> int
    pred_agg_i = pred_agg.astype(np.int64)
    gt_agg_i = gt_agg.astype(np.int64)

    inter = gt_agg_i @ pred_agg_i.T  # (G, P)
    pred_area = pred_agg_i.sum(axis=1)[None, :]  # (1, P)
    gt_area = gt_agg_i.sum(axis=1)[:, None]  # (G, 1)
    union = gt_area + pred_area - inter

    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return iou.astype(np.float64)


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------
def t_miou_t_sr(
    pred_labels: np.ndarray,
    gt_ids: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    iou_thresh: float = 0.5,
    ignore_ids: Sequence[int] = (0,),
) -> Dict[str, float]:
    """Compute T-mIoU and T-SR for a single scene.

    Args:
        pred_labels: (S, H, W) int — HDBSCAN cluster IDs (-1 = noise).
        gt_ids:      (S, H, W) int — per-pixel instance IDs (per-scene scoped).
        valid_mask:  (S, H, W) bool — pixels to include. If None, all pixels used.
        iou_thresh:  threshold on per-view IoU for T-SR "success".
        ignore_ids:  GT IDs to ignore (e.g. 0 = background in many datasets).

    Returns:
        {
            't_miou': float,    # mean cross-view IoU of best-matched pairs
            't_sr':   float,    # fraction of GT instances tracked in every view
            'n_gt_instances': int,
        }
    If there are no valid GT instances, returns zeros and n_gt_instances=0.
    """
    if valid_mask is None:
        valid_mask = np.ones_like(gt_ids, dtype=bool)
    assert pred_labels.shape == gt_ids.shape == valid_mask.shape

    S, H, W = pred_labels.shape
    HW = H * W
    pred_flat = pred_labels.reshape(S, HW)
    gt_flat = gt_ids.reshape(S, HW)
    valid_flat = valid_mask.reshape(S, HW).astype(bool, copy=False)

    pred_unique = np.unique(pred_flat[valid_flat])
    pred_unique = pred_unique[pred_unique != -1]

    gt_unique = np.unique(gt_flat[valid_flat])
    ignore_set = set(int(x) for x in ignore_ids)
    gt_unique = np.array(
        [g for g in gt_unique if int(g) not in ignore_set],
        dtype=gt_unique.dtype,
    )

    P = int(len(pred_unique))
    G = int(len(gt_unique))

    if G == 0:
        return {"t_miou": 0.0, "t_sr": 0.0, "n_gt_instances": 0}

    if P == 0:
        # No predicted clusters at all — all GT instances are missed
        return {"t_miou": 0.0, "t_sr": 0.0, "n_gt_instances": int(G)}

    # Compact-id representation:
    #   - aggregate masks are only (P/G, HW), not the old (S, P/G, HW) block;
    #   - per-view overlaps are dense bincount confusion tables.
    # This keeps the exact aggregate matching semantics while avoiding the
    # large per-view boolean tensors and Python loops over every GT instance.
    pred_agg = np.zeros((P, HW), dtype=bool)
    gt_agg = np.zeros((G, HW), dtype=bool)
    pred_area = np.zeros((S, P), dtype=np.int64)
    gt_area = np.zeros((S, G), dtype=np.int64)
    inter = np.zeros((S, G, P), dtype=np.int64)

    positions = np.arange(HW, dtype=np.int64)
    for s in range(S):
        valid_s = valid_flat[s]
        if not valid_s.any():
            continue

        pred_s = pred_flat[s, valid_s]
        gt_s = gt_flat[s, valid_s]
        pos_s = positions[valid_s]

        pred_found = pred_s != -1
        if pred_found.any():
            pred_idx = np.searchsorted(pred_unique, pred_s[pred_found])
            pred_agg[pred_idx, pos_s[pred_found]] = True
            pred_area[s] = np.bincount(pred_idx, minlength=P)

        gt_found = np.isin(gt_s, gt_unique, assume_unique=False)
        if gt_found.any():
            gt_idx = np.searchsorted(gt_unique, gt_s[gt_found])
            gt_agg[gt_idx, pos_s[gt_found]] = True
            gt_area[s] = np.bincount(gt_idx, minlength=G)

        both_found = pred_found & gt_found
        if both_found.any():
            pred_idx = np.searchsorted(pred_unique, pred_s[both_found])
            gt_idx = np.searchsorted(gt_unique, gt_s[both_found])
            inter[s] = np.bincount(
                gt_idx * P + pred_idx,
                minlength=G * P,
            ).reshape(G, P)

    # Match GT ↔ pred using cross-view aggregate IoU (stable, picks the cluster
    # that best represents this GT instance globally across all views).
    pred_agg_i = pred_agg.astype(np.int32)
    gt_agg_i = gt_agg.astype(np.int32)
    inter_agg = gt_agg_i @ pred_agg_i.T  # (G, P)
    pred_agg_area = pred_agg_i.sum(axis=1, dtype=np.int64)[None, :]  # (1, P)
    gt_agg_area = gt_agg_i.sum(axis=1, dtype=np.int64)[:, None]  # (G, 1)
    union_agg = gt_agg_area + pred_agg_area - inter_agg
    iou_mat = np.divide(
        inter_agg,
        union_agg,
        out=np.zeros((G, P), dtype=np.float64),
        where=union_agg > 0,
    )
    best_pred_idx = iou_mat.argmax(axis=1)  # (G,)

    # For each matched pair, compute per-view IoU in the views where the GT
    # instance is visible, then average. This is the IGGT T-mIoU convention:
    # penalize missing the object in any single view.
    gt_idx_range = np.arange(G)
    matched_inter = inter[:, gt_idx_range, best_pred_idx]  # (S, G)
    matched_pred_area = pred_area[:, best_pred_idx]  # (S, G)
    union = gt_area + matched_pred_area - matched_inter
    per_view_iou = np.divide(
        matched_inter,
        union,
        out=np.zeros((S, G), dtype=np.float64),
        where=union > 0,
    )
    gt_view_has = gt_area > 0
    visible_counts = gt_view_has.sum(axis=0)
    counted = visible_counts > 0
    n_counted = int(counted.sum())

    if n_counted == 0:
        return {"t_miou": 0.0, "t_sr": 0.0, "n_gt_instances": 0}

    per_instance_miou = np.divide(
        (per_view_iou * gt_view_has).sum(axis=0),
        visible_counts,
        out=np.zeros(G, dtype=np.float64),
        where=visible_counts > 0,
    )

    # T-SR: success iff all visible views clear iou_thresh.
    successes = ((per_view_iou >= iou_thresh) | ~gt_view_has).all(axis=0)

    t_miou = float(per_instance_miou[counted].mean())
    t_sr = float(successes[counted].mean())

    return {"t_miou": t_miou, "t_sr": t_sr, "n_gt_instances": int(G)}


def aggregate_scene_metrics(
    per_scene: Sequence[Dict[str, float]],
    weight_by_instances: bool = False,
) -> Dict[str, float]:
    """Average per-scene metrics into a single mean.

    Args:
        per_scene: list of dicts from `t_miou_t_sr`.
        weight_by_instances: if True, weight each scene's metric by its
            n_gt_instances (IGGT-style macro vs. instance-weighted).

    Returns:
        {'t_miou': float, 't_sr': float, 'n_scenes': int, 'n_gt_instances_total': int}
    """
    per_scene = [p for p in per_scene if p.get("n_gt_instances", 0) > 0]
    if not per_scene:
        return {"t_miou": 0.0, "t_sr": 0.0, "n_scenes": 0, "n_gt_instances_total": 0}

    if weight_by_instances:
        weights = np.array([p["n_gt_instances"] for p in per_scene], dtype=np.float64)
        weights = weights / weights.sum()
    else:
        weights = np.full(len(per_scene), 1.0 / len(per_scene))

    t_miou = float(sum(w * p["t_miou"] for w, p in zip(weights, per_scene)))
    t_sr = float(sum(w * p["t_sr"] for w, p in zip(weights, per_scene)))
    n_total = int(sum(p["n_gt_instances"] for p in per_scene))

    return {
        "t_miou": t_miou,
        "t_sr": t_sr,
        "n_scenes": len(per_scene),
        "n_gt_instances_total": n_total,
    }
