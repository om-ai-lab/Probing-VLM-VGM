"""Backward-compat shim. Canonical home moved to probing_vlm_vgm.losses.geometry.

New code should import from `probing_vlm_vgm.losses`:
    from probing_vlm_vgm.losses import camera_loss, depth_loss, point_loss
"""
from probing_vlm_vgm.losses.geometry import *  # noqa: F401,F403
from probing_vlm_vgm.losses.geometry import (  # noqa: F401  (explicit re-exports)
    camera_loss,
    camera_loss_single,
    check_and_fix_inf_nan,
    conf_loss,
    depth_loss,
    gradient_loss,
    gradient_loss_multi_scale,
    normal_loss,
    normalize_pointcloud,
    point_loss,
    point_map_to_normal,
    reg_loss,
)
