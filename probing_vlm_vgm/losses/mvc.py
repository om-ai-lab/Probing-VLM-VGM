"""Multi-view contrastive (MVC) loss for instance probing.

Implements the pull-push contrastive objective from IGGT (Li et al. 2025, eq. 4):
  - pull: pixels sharing an instance ID should have close features
  - push: pixels with different instance IDs should have feature distance >= margin

Design notes:
  - Pairs are formed only within the same batch sample (per-scene scoped).
    Different scenes in the batch have disjoint instance-ID spaces, so
    cross-scene pairs are never formed. `torch.cdist` over (B, P, D) does this
    automatically.
  - Pixel sampling reduces memory: we pick P pixels per scene instead of
    materializing a full (S*H*W)^2 pairwise matrix.
  - A valid_mask lets us skip background / no-instance pixels (ID 0 in ScanNet
    is typically "no annotation" or wall/floor — user can decide semantics
    at call site).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def sample_valid_pixels(
    valid_mask: torch.Tensor,
    num_samples: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """For each batch item, sample `num_samples` indices into flattened valid pixels.

    Args:
        valid_mask: (B, N) bool mask over flattened pixels.
        num_samples: number of pixel indices per batch item.
        generator: optional torch Generator for reproducible sampling.

    Returns:
        idx: (B, num_samples) long tensor, indices into the flat pixel dim.
             If a batch item has fewer than `num_samples` valid pixels, indices
             are sampled with replacement. If it has zero valid pixels, indices
             are all 0 (caller should guard with `valid_count`).
    """
    B, N = valid_mask.shape
    device = valid_mask.device
    idx = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    for b in range(B):
        vb = valid_mask[b]
        valid_idx = torch.nonzero(vb, as_tuple=False).squeeze(-1)  # (nv,)
        if valid_idx.numel() == 0:
            continue  # leave as zeros; caller filters via valid_count
        replace = valid_idx.numel() < num_samples
        if replace:
            picks = torch.randint(
                0, valid_idx.numel(), (num_samples,),
                device=device, generator=generator,
            )
        else:
            picks = torch.randperm(
                valid_idx.numel(), device=device, generator=generator,
            )[:num_samples]
        idx[b] = valid_idx[picks]
    return idx


def mvc_loss(
    feats: torch.Tensor,
    gt_ids: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    num_samples: int = 1024,
    margin: float = 1.0,
    lambda_pull: float = 1.0,
    lambda_push: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Compute pull-push multi-view contrastive loss.

    Args:
        feats: (B, S, D, H, W) — L2-normalized instance features (unit sphere).
        gt_ids: (B, S, H, W) int — instance ID per pixel, *per-scene scoped*.
        valid_mask: (B, S, H, W) bool — pixels eligible for sampling (exclude
            background / unannotated). If None, all pixels are valid.
        num_samples: pixels sampled per batch item (P in the paper).
        margin: hinge margin for the push term.
        lambda_pull / lambda_push: loss weights.
        generator: optional torch.Generator for reproducibility.

    Returns:
        Dict with keys:
            loss: scalar total loss
            loss_pull: pull component (mean distance of same-instance pairs)
            loss_push: push component (mean hinge of different-instance pairs)
            num_pull_pairs / num_push_pairs: counts (for logging)
    """
    assert feats.ndim == 5, f"feats must be (B,S,D,H,W), got {feats.shape}"
    assert gt_ids.shape == feats.shape[:2] + feats.shape[3:], (
        f"gt_ids shape {gt_ids.shape} must match feats {feats.shape} minus D dim"
    )

    B, S, D, H, W = feats.shape
    N = S * H * W

    # Flatten: (B, N, D) and (B, N)
    feats_flat = feats.permute(0, 1, 3, 4, 2).reshape(B, N, D)
    ids_flat = gt_ids.reshape(B, N)

    if valid_mask is None:
        valid_flat = torch.ones(B, N, dtype=torch.bool, device=feats.device)
    else:
        assert valid_mask.shape == gt_ids.shape
        valid_flat = valid_mask.reshape(B, N)

    # Sample P valid pixels per batch item
    idx = sample_valid_pixels(valid_flat, num_samples, generator=generator)  # (B, P)

    # Gather sampled features and IDs
    gather_f = idx.unsqueeze(-1).expand(-1, -1, D)  # (B, P, D)
    f_sampled = torch.gather(feats_flat, 1, gather_f)  # (B, P, D)
    m_sampled = torch.gather(ids_flat, 1, idx)  # (B, P)

    # Track which sampled slots actually had any valid pixel in that batch entry
    # (if a scene has zero valid pixels, its P sampled indices are all 0 with
    # ID gt_ids[b, 0] which may or may not be meaningful — we mask it out)
    has_valid = valid_flat.any(dim=1, keepdim=True)  # (B, 1)
    slot_valid = has_valid.expand(B, num_samples)  # (B, P)

    # Pairwise L2 distance: (B, P, P). Since feats are unit-normalized,
    # cdist^2 = 2 - 2*cos_sim, cdist in [0, 2].
    d = torch.cdist(f_sampled, f_sampled, p=2)  # (B, P, P)

    same = m_sampled.unsqueeze(-1) == m_sampled.unsqueeze(-2)  # (B, P, P)

    # Pair validity: both endpoints in valid slots, exclude self-pair
    pair_valid = slot_valid.unsqueeze(-1) & slot_valid.unsqueeze(-2)  # (B, P, P)
    eye = torch.eye(num_samples, device=feats.device, dtype=torch.bool).unsqueeze(0)
    pair_valid = pair_valid & ~eye

    pull_mask = same & pair_valid  # (B, P, P)
    push_mask = (~same) & pair_valid

    num_pull = pull_mask.sum().clamp(min=1)
    num_push = push_mask.sum().clamp(min=1)

    loss_pull = (d * pull_mask.float()).sum() / num_pull.float()
    loss_push = (F.relu(margin - d) * push_mask.float()).sum() / num_push.float()

    loss = lambda_pull * loss_pull + lambda_push * loss_push

    return {
        "loss": loss,
        "loss_pull": loss_pull.detach(),
        "loss_push": loss_push.detach(),
        "num_pull_pairs": num_pull.detach(),
        "num_push_pairs": num_push.detach(),
    }
