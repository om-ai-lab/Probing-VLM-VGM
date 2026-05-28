"""Asymmetric Loss for Multi-Label Classification (Exp-B tagging head).

Ridnik et al., ICCV 2021 — https://arxiv.org/abs/2009.14119

The standard multi-label BCE treats all positives and negatives symmetrically,
which is disastrous when the negative-to-positive ratio is large (ScanNet200:
one video has ~5 positive classes out of 200 → 1:40 imbalance). ASL fixes
this by:

  - Down-weighting easy negatives with a focal-style factor (gamma_neg)
  - NOT down-weighting positives by default (gamma_pos=0) since they are rare
  - Optionally shifting the negative probability by `clip` to fully discard
    very-low-confidence negatives (probability < clip → loss = 0)

Defaults (gamma_neg=4, gamma_pos=0, clip=0.05) come from the paper's MS-COCO
recipe and are the recommended starting point.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss with optional probability shift for negatives.

    Args:
        gamma_neg: focal exponent for negative samples (default 4).
        gamma_pos: focal exponent for positive samples (default 0 = no focal).
        clip:      probability shift applied to negatives; predictions below
                   `clip` are clamped to 0 contribution. Set to 0 to disable.
        eps:       numerical floor for log.
        reduction: 'mean' | 'sum' | 'none' (reduces over batch only; per-class
                   loss is always summed within a sample first).

    forward(logits, targets):
        logits:  (B, C) raw scores
        targets: (B, C) {0, 1} float
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        assert reduction in ("mean", "sum", "none"), reduction
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Probability of positive class
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # Optionally shift the negative probability — predictions weaker than
        # `clip` contribute zero, which prunes easy negatives entirely.
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Log probabilities (positive class for y=1, negative class for y=0)
        log_pos = torch.log(xs_pos.clamp(min=self.eps))
        log_neg = torch.log(xs_neg.clamp(min=self.eps))

        loss_pos = targets * log_pos
        loss_neg = (1 - targets) * log_neg

        # Focal-style modulation: down-weight easy samples.
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            # pt = xs_pos for positives, xs_neg for negatives
            pt0 = xs_pos * targets                          # 0 for negatives
            pt1 = xs_neg * (1 - targets)                    # 0 for positives
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            # Detach the modulation so gradient flows only through the log term
            # (this matches the official ASL reference implementation).
            one_sided_w = one_sided_w.detach()
            loss = (loss_pos + loss_neg) * one_sided_w
        else:
            loss = loss_pos + loss_neg

        # Per-sample loss = sum over classes; then reduce over batch.
        per_sample = -loss.sum(dim=1)
        if self.reduction == "mean":
            return per_sample.mean()
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample
