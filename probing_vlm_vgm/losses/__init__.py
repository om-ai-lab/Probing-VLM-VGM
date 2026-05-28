"""Canonical home for all loss functions in probing_vlm_vgm.

Three families:
    - geometry: depth / point / camera (Exp-A) — confidence-weighted ℓ2 + Huber
    - mvc:      multi-view contrastive pull-push (Exp-C) — instance grouping
    - asymmetric: multi-label classification (Exp-B) — long-tail tagging

Old import paths (`probing_vlm_vgm.utils.loss`, `probing_vlm_vgm.utils.mvc_loss`) still work
via thin re-export shims, so existing experiment yamls and modules keep
running without changes.
"""
from probing_vlm_vgm.losses.asymmetric import AsymmetricLoss
from probing_vlm_vgm.losses.geometry import (
    camera_loss,
    depth_loss,
    point_loss,
)
from probing_vlm_vgm.losses.mvc import mvc_loss

__all__ = [
    "AsymmetricLoss",
    "camera_loss",
    "depth_loss",
    "point_loss",
    "mvc_loss",
]
