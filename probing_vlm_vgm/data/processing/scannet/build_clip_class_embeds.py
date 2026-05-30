"""Pre-compute CLIP text embeddings for ScanNet20 / ScanNet200 class names.

Used to initialize SemanticTagHead.queries (via clip_init_path in
configs/model/probe_tagging.yaml). Computed once, reused across all VFM
training runs.

Each class label is encoded under 7 prompt templates (CLIP-style ensemble);
the per-template embeddings are L2-normalized, averaged, then re-normalized.
This is the same recipe as CLIP's zero-shot ImageNet evaluation.

Default backbone: ``ViT-L-14`` (openai weights), output dim 768.

Example:
    python -m probing_vlm_vgm.data.processing.scannet.build_clip_class_embeds \\
        --model ViT-L-14 \\
        --pretrained openai \\
        --num_classes 200 \\
        --out_path data/ScanNet/clip_class_embeds_200_vitl14.npy
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from probing_vlm_vgm.data.processing.scannet.scannet_constants import get_class_labels

logger = logging.getLogger(__name__)


# Indoor-scene-tilted CLIP ensemble. The "in a room" / "indoor" templates are
# helpful for ScanNet specifically (most ImageNet templates assume outdoor
# scenes). Order matters only insofar as we average so it doesn't.
TEMPLATES: List[str] = [
    "a photo of a {}",
    "a photo of the {}",
    "a photo of a small {}",
    "a photo of a large {}",
    "a photo of one {}",
    "a {} in a room",
    "an indoor photo of a {}",
]


def encode_class_names(
    class_names: List[str],
    model_name: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    templates: Optional[List[str]] = None,
) -> np.ndarray:
    """Encode each class name under the ensemble → (C, D) L2-normalized.

    Backend: HuggingFace ``transformers`` CLIPTextModel. We prefer it over
    open_clip because it ships with the probing_vlm_vgm env and the OpenAI CLIP
    weights are bit-for-bit identical between the two libraries.
    Recommended model names (matching the OpenAI CLIP family):
        openai/clip-vit-large-patch14   # ViT-L/14, 768d  (default)
        openai/clip-vit-base-patch32    # ViT-B/32, 512d
    """
    from transformers import CLIPTextModel, CLIPTokenizer

    if templates is None:
        templates = TEMPLATES

    logger.info(f"Loading CLIP text model: {model_name} on {device}")
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name).to(device).eval()

    embeds = []
    with torch.no_grad():
        for c in class_names:
            prompts = [t.format(c) for t in templates]
            tok = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            out = model(**tok)
            # CLIPTextModel returns last_hidden_state + pooler_output. The
            # original CLIP zero-shot recipe uses the EOS token embedding
            # (= pooler_output in HF's CLIPTextModel).
            feats = out.pooler_output                       # (T, D)
            feats = F.normalize(feats.float(), dim=-1)
            embeds.append(feats.mean(dim=0))                # (D,)
    embeds_t = torch.stack(embeds, dim=0)                   # (C, D)
    embeds_t = F.normalize(embeds_t, dim=-1)                # re-norm after averaging
    return embeds_t.cpu().numpy().astype(np.float32)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num_classes", type=int, choices=[20, 200], default=200)
    p.add_argument(
        "--model", type=str, default="openai/clip-vit-large-patch14",
        help="HuggingFace CLIP model name. Defaults to ViT-L/14 (768d). For "
             "smaller experiments use 'openai/clip-vit-base-patch32' (512d).",
    )
    p.add_argument(
        "--out_path", type=Path, default=None,
        help="Output .npy path. Default: "
             "data/ScanNet/clip_class_embeds_<C>_<model>.npy",
    )
    p.add_argument(
        "--device", type=str, default="cuda",
        help="cuda or cpu. ViT-L/14 on CPU takes <1 min for 200 classes — OK fallback.",
    )
    p.add_argument(
        "--class_names_path", type=Path, default=None,
        help="Optional JSON {names: [str, ...]} to override the built-in "
             "ScanNet20/200 list. Useful for the LVIS-pseudo long-tail ablation.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Resolve class names
    if args.class_names_path is not None:
        import json
        with open(args.class_names_path, "r") as f:
            class_names = json.load(f)["names"]
        logger.info(f"Loaded {len(class_names)} class names from {args.class_names_path}")
    else:
        class_names = list(get_class_labels(args.num_classes))
        logger.info(f"Using built-in ScanNet{args.num_classes} class names")

    if args.num_classes != len(class_names):
        logger.warning(
            f"num_classes={args.num_classes} but class_names has {len(class_names)} "
            f"entries — the file name will use --num_classes for consistency."
        )

    # Resolve output path
    if args.out_path is None:
        # Tag derived from the model name's last path component, kept short.
        # e.g. "openai/clip-vit-large-patch14" → "vitl14".
        last = args.model.split("/")[-1].lower()
        if "large" in last and "patch14" in last:
            model_tag = "vitl14"
        elif "base" in last and "patch32" in last:
            model_tag = "vitb32"
        elif "base" in last and "patch16" in last:
            model_tag = "vitb16"
        else:
            model_tag = last.replace("-", "")
        args.out_path = Path(
            f"data/ScanNet/clip_class_embeds_{args.num_classes}_{model_tag}.npy"
        )
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    # Encode
    embeds = encode_class_names(
        class_names=class_names,
        model_name=args.model,
        device=args.device,
    )
    logger.info(f"Embeddings shape: {embeds.shape}, dtype: {embeds.dtype}")

    np.save(args.out_path, embeds)
    logger.info(f"Wrote {args.out_path}")

    # Quick sanity check: cosine similarity between a few pairs of classes
    sim = embeds @ embeds.T
    np.fill_diagonal(sim, 0.0)
    logger.info(
        f"Off-diagonal cosine sim — min: {sim.min():.3f}, "
        f"max: {sim.max():.3f}, mean: {sim.mean():.3f}"
    )


if __name__ == "__main__":
    main()
