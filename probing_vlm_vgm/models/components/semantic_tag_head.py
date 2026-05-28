"""SemanticTagHead — DETR-decoder-style multi-label classification head (Exp-B).

Architecture (matches research_plan.md §6.3.3):

    patch_tokens [B, S, P_patch, dim_in]    ← from BackbonePA (last stage)
            │
            │  reshape + in_proj
            ▼
    KV  [B, S*P_patch, embed_dim]

    class_queries [num_classes, embed_dim]   ← learnable OR CLIP-text-embed init
            │  expand to [B, num_classes, embed_dim]
            ▼
    × num_layers of DecoderLayer (self-attn → cross-attn → FFN)
            │
            ▼
    classifier (3 modes — see `classifier_mode`)  →  logits [B, num_classes]

Query initialization:
  - Random init   (clip_init_embeds=None): queries are pure learnable params,
                  initialized to N(0, 0.02²).
  - CLIP init     (clip_init_embeds=Tensor[C, clip_dim]): a frozen buffer of
                  CLIP text embeddings + a small learnable `clip_proj` lifts
                  them to the head's embed_dim. Each forward re-derives
                  queries = clip_proj(buffer), so swapping the buffer at
                  inference time swaps the **query** prototypes.

Classifier modes (`classifier_mode`):
  - "per_class":   one Linear(embed_dim, 1) per class. Closed-vocab only
                   (the classifier weights are class-specific and learned
                   for the training vocabulary; swapping the CLIP buffer
                   would leave them mismatched). This is the default and
                   the right choice for Exp-B's closed-vocab ScanNet200
                   tagging.
  - "q2l":         Q2L-style with a learnable (C, embed_dim) weight matrix;
                   each query c is scored against row c. Same closed-vocab
                   property — class-specific weights.
  - "open_vocab":  vocab-agnostic readout. After the decoder stack, score
                   each query against ITS OWN initial CLIP embedding
                   (`<q_c, clip_proj(clip_init_c)>` + shared bias). The
                   classifier has zero per-class params — swapping the
                   CLIP buffer truly swaps the vocabulary without
                   re-training. Required when reusing this head on a
                   different vocabulary (L3 open-vocab extension).
                   Requires `clip_init_embeds` to be provided.

NOTE on closed-vocab vs open-vocab: research_plan §5.2.2 says CLIP init
"directly extends to open-vocab L3". That's only true with
`classifier_mode="open_vocab"`. For Exp-B's main results, we stick with
"per_class" (better closed-vocab accuracy); "open_vocab" is the L3
hand-off path.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from probing_vlm_vgm.models.components.decoder_layer import DecoderLayer


class SemanticTagHead(nn.Module):
    def __init__(
        self,
        dim_in: int,
        num_classes: int = 200,
        embed_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        clip_init_embeds: Optional[torch.Tensor] = None,
        classifier_mode: str = "per_class",
        # Back-compat: per_class_linear=True/False used to map to
        # "per_class"/"q2l". Honour it if classifier_mode is left default.
        per_class_linear: Optional[bool] = None,
    ) -> None:
        """
        Args:
            dim_in:           input channel dim (= 2*embed_dim of BackbonePA)
            num_classes:      number of categories (e.g. 200 for ScanNet200)
            embed_dim:        head's internal width
            num_layers:       K decoder layers (paper recommends K=2)
            num_heads:        attention heads per decoder layer
            mlp_ratio:        FFN expansion in DecoderLayer
            dropout:          shared dropout for attn + FFN
            clip_init_embeds: optional (num_classes, clip_dim) tensor; if
                              given, queries are derived as clip_proj(embed)
                              every forward. Required for "open_vocab".
            classifier_mode:  "per_class" | "q2l" | "open_vocab" — see the
                              module docstring for the closed/open-vocab
                              implications. Default "per_class" matches
                              Exp-B's closed-vocab ScanNet200 setting.
            per_class_linear: DEPRECATED — kept for backward compat. True →
                              "per_class", False → "q2l". Set classifier_mode
                              directly in new code.
        """
        super().__init__()

        # Back-compat translation for per_class_linear.
        if per_class_linear is not None:
            classifier_mode = "per_class" if per_class_linear else "q2l"
        if classifier_mode not in ("per_class", "q2l", "open_vocab"):
            raise ValueError(
                f"classifier_mode must be one of "
                f"'per_class' | 'q2l' | 'open_vocab', got {classifier_mode!r}"
            )

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.classifier_mode = classifier_mode

        # Project backbone output (2*embed_dim of backbone) into head width.
        # If dims match we still wrap an Identity so external code can
        # introspect a uniform attribute name.
        self.in_proj = (
            nn.Identity() if dim_in == embed_dim else nn.Linear(dim_in, embed_dim)
        )

        # ----- Query initialization -----
        if clip_init_embeds is not None:
            assert clip_init_embeds.dim() == 2 and clip_init_embeds.shape[0] == num_classes, (
                f"clip_init_embeds must be (num_classes={num_classes}, clip_dim), "
                f"got {tuple(clip_init_embeds.shape)}"
            )
            clip_dim = clip_init_embeds.shape[1]
            self.register_buffer("clip_init", clip_init_embeds.clone(), persistent=False)
            self.clip_proj: Optional[nn.Linear] = nn.Linear(clip_dim, embed_dim)
            self.queries: Optional[nn.Parameter] = None
        else:
            self.clip_init = None  # type: ignore[assignment]
            self.clip_proj = None
            self.queries = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)

        # ----- Decoder stack -----
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # ----- Classifier -----
        if classifier_mode == "per_class":
            # 200 separate Linear(embed_dim, 1) — closed-vocab.
            self.cls_heads = nn.ModuleList(
                [nn.Linear(embed_dim, 1) for _ in range(num_classes)]
            )
        elif classifier_mode == "q2l":
            # Q2L-style: each query takes inner product with its own row of
            # a shared (num_classes, embed_dim) matrix. Still closed-vocab
            # (cls_weight[c] is trained for class c).
            self.cls_weight = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
            self.cls_bias = nn.Parameter(torch.zeros(num_classes))
        else:  # "open_vocab"
            if clip_init_embeds is None:
                raise ValueError(
                    "classifier_mode='open_vocab' requires clip_init_embeds — "
                    "the classifier scores each query against ITS OWN CLIP "
                    "prototype, so without CLIP init there's nothing to score "
                    "against. For closed-vocab tagging use 'per_class' or 'q2l'."
                )
            # Single shared scalar bias + temperature. Zero per-class params,
            # so swapping the clip_init buffer fully swaps the vocabulary.
            # `logit_scale` mirrors the CLIP convention of a learnable temp.
            self.logit_scale = nn.Parameter(torch.tensor(2.6592))  # ln(1/0.07)
            self.cls_bias = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------ #
    def get_queries(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return (B, num_classes, embed_dim) query tensor."""
        if self.queries is not None:
            q = self.queries
        else:
            q = self.clip_proj(self.clip_init.to(device))
        return q.unsqueeze(0).expand(batch_size, -1, -1)

    # ------------------------------------------------------------------ #
    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, S, P_patch, dim_in) — backbone last stage,
                          with camera/register tokens already stripped.

        Returns:
            logits: (B, num_classes) raw scores (apply sigmoid externally).
        """
        assert patch_tokens.dim() == 4, (
            f"expected (B,S,P,dim_in), got {tuple(patch_tokens.shape)}"
        )
        B, S, P, _ = patch_tokens.shape

        kv = self.in_proj(patch_tokens).reshape(B, S * P, self.embed_dim)

        # Cache the initial-prototype queries — needed by "open_vocab" so we
        # can score the *post-decoder* queries against their *pre-decoder*
        # CLIP-derived prototype. For "per_class" / "q2l" we only need the
        # post-decoder queries.
        queries = self.get_queries(B, patch_tokens.device)  # (B, C, D)
        if self.classifier_mode == "open_vocab":
            init_queries = queries  # snapshot the prototype before mixing

        for layer in self.layers:
            queries = layer(queries, kv)

        if self.classifier_mode == "per_class":
            outs = [self.cls_heads[c](queries[:, c]) for c in range(self.num_classes)]
            logits = torch.cat(outs, dim=1)
        elif self.classifier_mode == "q2l":
            # (B, C, D) ⊙ (C, D) → sum over D + bias → (B, C)
            logits = (queries * self.cls_weight.unsqueeze(0)).sum(dim=-1) + self.cls_bias
        else:  # "open_vocab"
            # Cosine sim between the post-decoder query and its initial CLIP
            # prototype, scaled by a learnable temperature (CLIP convention).
            # Zero per-class learnable params → vocab-agnostic.
            q_norm = F.normalize(queries, dim=-1)
            p_norm = F.normalize(init_queries, dim=-1)
            cos = (q_norm * p_norm).sum(dim=-1)              # (B, C)
            logits = cos * self.logit_scale.exp() + self.cls_bias
        return logits
