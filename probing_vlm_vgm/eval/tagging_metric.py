"""Multi-label tagging metrics for Exp-B.

Computes:
    mAP                    macro-AP across classes (main metric)
    AP_head, AP_mid, AP_tail  AP averaged within frequency buckets (long-tail diagnostic)
    OF1, CF1               F1 at a fixed 0.5 probability threshold
    per_class_threshold    fixed threshold used for F1 (0.5 for valid classes)

Implementation notes:
  - AP is computed by integrating precision over recall using the
    "step function" convention (sklearn's `average_precision_score`).
  - When a class has zero positives, its AP is undefined; we skip it for
    macro averaging (consistent with COCO mAP).
  - Frequency buckets ("head / mid / tail") are taken from the *training*
    positive-rate, passed in as `train_pos_rate`. We split at the 33rd /
    66th percentiles by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------- #
def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Single-class AP.

    Equivalent to sklearn.metrics.average_precision_score for binary labels,
    but without the sklearn dependency. y_true: (N,) {0,1}; y_score: (N,) float.
    Returns NaN if y_true has zero positives.
    """
    y_true = np.asarray(y_true).astype(np.float64).reshape(-1)
    y_score = np.asarray(y_score).astype(np.float64).reshape(-1)
    n_pos = y_true.sum()
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="stable")
    y_true_sorted = y_true[order]
    tp_cum = np.cumsum(y_true_sorted)
    fp_cum = np.cumsum(1 - y_true_sorted)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall = tp_cum / n_pos
    # Step-function AP: sum over the precision delta-recall steps.
    delta_recall = np.diff(np.concatenate([[0], recall]))
    return float((precision * delta_recall).sum())


# ---------------------------------------------------------------------- #
def per_class_best_threshold(
    y_true: np.ndarray, y_score: np.ndarray, grid: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the F1-optimal threshold for each class.

    Args:
        y_true:  (N, C) {0, 1}
        y_score: (N, C) sigmoid-applied [0, 1] floats
        grid:    threshold grid (default linspace(0.05, 0.95, 19))

    Returns:
        thresholds: (C,) best threshold per class (NaN if class has no
                    positives — caller should ignore those classes).
        f1_at_best: (C,) F1 at that threshold (NaN if no positives).
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    y_true = np.asarray(y_true).astype(np.float64)
    y_score = np.asarray(y_score).astype(np.float64)
    N, C = y_true.shape
    out_th = np.full(C, np.nan)
    out_f1 = np.full(C, np.nan)
    for c in range(C):
        yt = y_true[:, c]
        if yt.sum() == 0:
            continue
        ys = y_score[:, c]
        best_f1 = -1.0
        best_th = 0.5
        for th in grid:
            pred = (ys >= th).astype(np.float64)
            tp = (pred * yt).sum()
            fp = (pred * (1 - yt)).sum()
            fn = ((1 - pred) * yt).sum()
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        out_th[c] = best_th
        out_f1[c] = best_f1
    return out_th, out_f1


# ---------------------------------------------------------------------- #
def _bucket_indices(
    train_pos_rate: np.ndarray, q_low: float = 1.0 / 3, q_high: float = 2.0 / 3
) -> Dict[str, np.ndarray]:
    """Split classes into head/mid/tail by training positive-rate percentile."""
    rates = np.asarray(train_pos_rate).astype(np.float64)
    # rank by rate; high rate = head
    order = np.argsort(-rates, kind="stable")
    n = len(rates)
    n_head = max(1, int(np.floor(n * q_low)))
    n_mid = max(1, int(np.floor(n * (q_high - q_low))))
    head = order[:n_head]
    mid = order[n_head : n_head + n_mid]
    tail = order[n_head + n_mid :]
    return {"head": head, "mid": mid, "tail": tail}


# ---------------------------------------------------------------------- #
@dataclass
class TaggingMetrics:
    mAP: float
    AP_head: float
    AP_mid: float
    AP_tail: float
    OF1: float                 # overall F1 at fixed 0.5 threshold
    CF1: float                 # macro F1 = mean of per-class F1 at fixed 0.5 threshold
    per_class_ap: np.ndarray   # (C,) raw per-class AP (NaN for zero-pos)
    per_class_threshold: np.ndarray  # (C,) fixed threshold, NaN for zero-pos
    per_class_f1: np.ndarray   # (C,) F1 at fixed threshold, NaN for zero-pos


def compute_tagging_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    train_pos_rate: Optional[np.ndarray] = None,
    from_logits: Optional[bool] = None,
) -> TaggingMetrics:
    """Compute the full Exp-B metric bundle.

    Args:
        y_true:         (N, C) {0, 1} ground-truth video-level labels
        y_score:        (N, C) raw logits OR probabilities — see `from_logits`
        train_pos_rate: (C,) per-class positive rate from the TRAINING split,
                        used to bucket head/mid/tail. If None, head/mid/tail
                        are derived from the eval-set's own positive rates
                        (less ideal but useful for smoke tests).
        from_logits:    explicitly state whether y_score is raw logits.
                          True  → apply sigmoid before thresholding.
                          False → y_score is already in [0, 1].
                          None  → heuristic (apply sigmoid iff min<0 or max>1).
                          The heuristic has an edge case: raw logits that
                          happen to land entirely within [0, 1] (rare but
                          possible early in training) would be treated as
                          probabilities, biasing fixed-0.5 F1.
                          Pass from_logits=True from training code to be safe;
                          AP is unaffected either way (order-invariant).

    Returns:
        TaggingMetrics dataclass.
    """
    y_true = np.asarray(y_true).astype(np.float64)
    y_score = np.asarray(y_score).astype(np.float64)
    assert y_true.shape == y_score.shape, (y_true.shape, y_score.shape)
    N, C = y_true.shape

    if from_logits is None:
        from_logits = bool(y_score.min() < 0.0 or y_score.max() > 1.0)
    if from_logits:
        y_prob = 1.0 / (1.0 + np.exp(-y_score))
    else:
        y_prob = y_score

    # ---- per-class AP ----
    per_class_ap = np.array([average_precision(y_true[:, c], y_prob[:, c]) for c in range(C)])
    valid_ap = per_class_ap[~np.isnan(per_class_ap)]
    mAP = float(valid_ap.mean()) if valid_ap.size > 0 else float("nan")

    # ---- bucketed AP ----
    if train_pos_rate is None:
        train_pos_rate = y_true.mean(axis=0)
    buckets = _bucket_indices(train_pos_rate)
    bucket_ap = {}
    for name, idx in buckets.items():
        vals = per_class_ap[idx]
        vals = vals[~np.isnan(vals)]
        bucket_ap[name] = float(vals.mean()) if vals.size > 0 else float("nan")

    # ---- fixed-threshold F1 ----
    per_class_th = np.full(C, 0.5, dtype=np.float64)
    per_class_f1 = np.full(C, np.nan, dtype=np.float64)
    for c in range(C):
        yt = y_true[:, c]
        if yt.sum() == 0:
            per_class_th[c] = np.nan
            continue
        pred_c = (y_prob[:, c] >= 0.5).astype(np.float64)
        tp_c = (pred_c * yt).sum()
        fp_c = (pred_c * (1 - yt)).sum()
        fn_c = ((1 - pred_c) * yt).sum()
        denom_c = 2 * tp_c + fp_c + fn_c
        per_class_f1[c] = (2 * tp_c / denom_c) if denom_c > 0 else 0.0
    valid_f1 = per_class_f1[~np.isnan(per_class_f1)]
    CF1 = float(valid_f1.mean()) if valid_f1.size > 0 else float("nan")

    # Overall F1 — predict 1 wherever probability ≥ 0.5.
    pred = (y_prob >= 0.5).astype(np.float64)
    tp = (pred * y_true).sum()
    fp = (pred * (1 - y_true)).sum()
    fn = ((1 - pred) * y_true).sum()
    denom = 2 * tp + fp + fn
    OF1 = float(2 * tp / denom) if denom > 0 else 0.0

    return TaggingMetrics(
        mAP=mAP,
        AP_head=bucket_ap.get("head", float("nan")),
        AP_mid=bucket_ap.get("mid", float("nan")),
        AP_tail=bucket_ap.get("tail", float("nan")),
        OF1=OF1,
        CF1=CF1,
        per_class_ap=per_class_ap,
        per_class_threshold=per_class_th,
        per_class_f1=per_class_f1,
    )
