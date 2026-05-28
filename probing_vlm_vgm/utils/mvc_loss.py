"""Backward-compat shim. Canonical home moved to probing_vlm_vgm.losses.mvc.

New code should import from `probing_vlm_vgm.losses`:
    from probing_vlm_vgm.losses import mvc_loss
"""
from probing_vlm_vgm.losses.mvc import mvc_loss, sample_valid_pixels  # noqa: F401
