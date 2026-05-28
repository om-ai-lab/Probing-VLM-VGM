"""Instance head for Setup A: frozen VFM → per-pixel instance embedding.

Design: minimalist by intent.
  - Single 1x1 Conv projects from the VFM feature channel dim to D_ins (default 32).
  - L2-normalized output lies on the unit hypersphere so the MVC loss sees
    bounded distances in [0, 2] and the push margin is well-defined.
  - No DPT pyramid, no cross-modal fusion, no attention backbone. The whole
    point of Setup A is to minimize what the head adds so the measured
    clustering quality reflects the frozen VFM feature's instance-level
    semantic content.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceHead(nn.Module):
    """1x1 Conv + L2 normalize.

    Input shape convention follows VideoProbeDataset: (B, S, C, H, W).
    """

    def __init__(self, in_channels: int, out_channels: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize instance features.

        Args:
            feat: (B, S, C, H, W) frozen VFM feature.

        Returns:
            (B, S, D, H, W) L2-normalized along the D axis (dim=2 at 5D).
        """
        assert feat.ndim == 5, f"expected 5D (B,S,C,H,W), got {feat.shape}"
        B, S, C, H, W = feat.shape
        assert C == self.in_channels, (
            f"input channel {C} != configured in_channels {self.in_channels}"
        )

        x = self.proj(feat.reshape(B * S, C, H, W))
        x = F.normalize(x, dim=1)  # unit sphere on channel dim
        return x.reshape(B, S, self.out_channels, H, W)


class InstanceHeadMLP(nn.Module):
    """Setup A' — per-pixel 3-layer MLP + L2 normalize.

    Adds two non-linearities vs InstanceHead so the readout can express
    pixel-local non-linear functions of the frozen feature without introducing
    cross-pixel / cross-frame mixing. This preserves the "no-backbone /
    read-out purity" framing (still a per-pixel function of frozen features),
    just relaxing the linear constraint of InstanceHead.

    Architecture: 1x1 Conv → GeLU → 1x1 Conv → GeLU → 1x1 Conv → L2 normalize.
    Hidden dim defaults to 2 * out_channels (=64 for out_channels=32) so the
    parameter count stays in the ~100K-1M range instead of blowing up to
    backbone-scale.

    Same input convention as InstanceHead: (B, S, C, H, W).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        hidden_channels: int | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Hidden dim: default = 2 * out_channels. Caller can override if larger
        # capacity is desired. Keep it << in_channels so total params stay small.
        self.hidden_channels = (
            hidden_channels if hidden_channels is not None else out_channels * 2
        )

        # 3-layer MLP: in → hidden → hidden → out
        self.proj1 = nn.Conv2d(
            in_channels, self.hidden_channels, kernel_size=1, bias=True
        )
        self.proj2 = nn.Conv2d(
            self.hidden_channels, self.hidden_channels, kernel_size=1, bias=True
        )
        self.proj3 = nn.Conv2d(
            self.hidden_channels, out_channels, kernel_size=1, bias=True
        )
        self.act = nn.GELU()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        assert feat.ndim == 5, f"expected 5D (B,S,C,H,W), got {feat.shape}"
        B, S, C, H, W = feat.shape
        assert C == self.in_channels, (
            f"input channel {C} != configured in_channels {self.in_channels}"
        )

        x = feat.reshape(B * S, C, H, W)
        x = self.act(self.proj1(x))
        x = self.act(self.proj2(x))
        x = self.proj3(x)
        x = F.normalize(x, dim=1)
        return x.reshape(B, S, self.out_channels, H, W)
