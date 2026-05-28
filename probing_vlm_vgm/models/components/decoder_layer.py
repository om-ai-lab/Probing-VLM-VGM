"""Transformer decoder layer: self-attn → cross-attn → FFN.

Used by SemanticTagHead (Exp-B) to let class queries first exchange
information among themselves (self-attn lets the head learn class
co-occurrence priors) and then read from backbone patch tokens via
cross-attention.

Pre-norm formulation, GeLU FFN, no positional encoding on queries
(class queries are a set — order is fixed and identifies the class).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DecoderLayer(nn.Module):
    """Single transformer decoder block.

    Args:
        embed_dim:   query / kv channel dim
        num_heads:   attention heads
        mlp_ratio:   FFN hidden = embed_dim * mlp_ratio
        dropout:     applied inside attention + FFN
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm3 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        kv_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: (B, N_q, D)
            kv:      (B, N_kv, D)
            kv_key_padding_mask: (B, N_kv) bool — True for positions to ignore.

        Returns:
            (B, N_q, D)
        """
        # Self-attention among queries (pre-norm)
        q = self.norm1(queries)
        sa_out, _ = self.self_attn(q, q, q, need_weights=False)
        queries = queries + sa_out

        # Cross-attention: queries → kv
        q = self.norm2(queries)
        ca_out, _ = self.cross_attn(
            q, kv, kv,
            key_padding_mask=kv_key_padding_mask,
            need_weights=False,
        )
        queries = queries + ca_out

        # FFN
        queries = queries + self.ffn(self.norm3(queries))
        return queries
