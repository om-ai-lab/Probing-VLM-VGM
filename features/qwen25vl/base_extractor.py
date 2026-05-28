"""
Base class for Qwen2.5-VL feature extraction.

This module provides the base class for extracting hidden states from Qwen2.5-VL models
for probe training and layer-wise analysis.

Differences vs. Qwen3-VL:
- Model class is Qwen2_5_VLForConditionalGeneration.
- ViT patch_size = 14 (vs 16 in Qwen3-VL), so the spatial factor in smart_resize
  is patch_size * spatial_merge_size = 28 (vs 32).
- Native video_processor budget defaults to {shortest_edge=128*28*28, longest_edge=28*28*768}
  in the Qwen2.5-VL source. The actual checkpoint preprocessor_config.json may override
  these (e.g. min/max_pixels in Qwen2.5-VL-7B-Instruct), and we always read live values
  from `processor.video_processor.size` rather than hard-coding them.
- Qwen2.5-VL ViT uses windowed self-attention with a few full-attention layers
  (`fullatt_block_indexes`); Qwen3-VL ViT removed the windowing. This is internal to the
  ViT and does not affect LLM hidden state extraction.

Both Qwen2.5-VL and Qwen3-VL configs use the same nested `text_config.hidden_size` /
`text_config.num_hidden_layers` paths and top-level `image_token_id` / `video_token_id`,
so the runtime introspection code is identical.
"""

from __future__ import annotations

import glob
import logging
import math
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

logger = logging.getLogger(__name__)


def get_query_frame_indices(context_len: int = 76, query_idx_divisor: int = 4) -> List[int]:
    """
    Generate query frame indices matching video_probe_dataset_'s query mechanism.

    Rule: 0, 1, then 1 + k*query_idx_divisor (k=1,2,3,...), until < context_len
    Result for context_len=76, divisor=4: [0, 1, 5, 9, 13, ..., 73]

    Args:
        context_len: Maximum frame index (exclusive), default 76
        query_idx_divisor: Alignment divisor, default 4

    Returns:
        List of frame indices (20 frames for default params)
    """
    indices = [0, 1]
    idx = 1 + query_idx_divisor
    while idx < context_len:
        indices.append(idx)
        idx += query_idx_divisor
    return indices


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 28,                       # Qwen2.5-VL: patch_size 14 * merge_size 2
    min_pixels: int = 128 * 28 * 28,        # 100,352 — matches Qwen2.5-VL source default
    max_pixels: int = 28 * 28 * 768,        # 602,112 — matches Qwen2.5-VL source default
) -> Tuple[int, int]:
    """
    Reference smart_resize for Qwen2.5-VL.

    NOTE: this is a documentation/utility function. At runtime, the HuggingFace processor
    handles resizing internally according to its own (min_pixels, max_pixels) budget.
    See `_compute_per_image_pixels` for how we feed a video-parity budget to the processor.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


class BaseQwen2_5_VLExtractor(ABC):
    """
    Base class for extracting hidden states from Qwen2.5-VL models.

    Handles:
    - Model loading
    - Frame loading and preprocessing via AutoProcessor
    - Hidden state extraction from LLM layers
    - Reshape of visual token hidden states to spatial [T, H, W, C]
    """

    def __init__(
        self,
        model_path: str,
        select_layers: List[int],
        question: str = "",
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        target_size: Optional[Tuple[int, int]] = (960, 540),
    ):
        """
        Args:
            model_path: Path to the Qwen2.5-VL model (local or HF Hub)
            select_layers: LLM layer indices to extract hidden states from
            question: Question/prompt to use for feature extraction
            device: Device to load model on
            torch_dtype: Data type for model weights
            target_size: (width, height) to resize all frames to before processing.
                         Set to None to keep original resolution.
                         Default (960, 540) matches DL3DV-960P.
        """
        self.model_path = model_path
        self.select_layers = select_layers
        self.question = question
        self.torch_dtype = torch_dtype
        self.device = torch.device(device)
        self.target_size = target_size

        logger.info(f"Loading Qwen2.5-VL model from {model_path}")
        logger.info(f"Target device: {self.device}")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            device_map=None,
            low_cpu_mem_usage=False,
        ).eval().to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_path)

        self.config = self.model.config
        # Both Qwen2.5-VL and Qwen3-VL store text params under `text_config`.
        # Qwen2.5-VL config has a backward-compat path that auto-builds text_config
        # even when the on-disk config.json is flat (verified for the
        # Qwen2.5-VL-7B-Instruct checkpoint).
        self.hidden_size = self.config.text_config.hidden_size
        self.num_layers = self.config.text_config.num_hidden_layers
        self.image_token_id = self.config.image_token_id
        self.video_token_id = self.config.video_token_id

        for layer in select_layers:
            resolved = layer if layer >= 0 else self.num_layers + layer + 1
            if resolved < 0 or resolved > self.num_layers:
                raise ValueError(
                    f"Layer {layer} out of range. Model has {self.num_layers} layers (valid: 0-{self.num_layers})."
                )

        logger.info(f"Model loaded. Hidden size: {self.hidden_size}, Layers: {self.num_layers}")
        logger.info(f"Extracting layers: {select_layers}")

    def load_frames(
        self,
        frame_dir: str,
        num_frames: int,
        frame_ext: str = "png",
        start_idx: int = 0,
        gt_num_frames: int = None,
        use_query_frame_indices: bool = False,
        context_len: int = 76,
        query_idx_divisor: int = 4,
    ) -> List[Image.Image]:
        """
        Load frames from a directory.

        Args:
            frame_dir: Directory containing frame_*.{ext} files
            num_frames: Number of frames to sample (ignored if use_query_frame_indices=True)
            frame_ext: Frame file extension
            start_idx: Starting frame index (from GT data_sft)
            gt_num_frames: Number of GT frames (for alignment)
            use_query_frame_indices: If True, use query frame indices matching dataset
            context_len: Context length for query frame indices
            query_idx_divisor: Divisor for query frame alignment
        """
        frame_paths = sorted(glob.glob(f"{frame_dir}/frame_*.{frame_ext}"))
        total_frames = len(frame_paths)

        if total_frames == 0:
            raise ValueError(f"No frames found in {frame_dir}")

        if gt_num_frames is not None:
            end_idx = min(start_idx + gt_num_frames, total_frames)
            gt_range_paths = frame_paths[start_idx:end_idx]
            range_len = len(gt_range_paths)

            if range_len == 0:
                raise ValueError(
                    f"No frames in GT range [{start_idx}, {end_idx}), "
                    f"total frames: {total_frames}"
                )

            if use_query_frame_indices:
                query_indices = get_query_frame_indices(context_len, query_idx_divisor)
                if range_len < context_len:
                    mapped_indices = []
                    for qi in query_indices:
                        if qi == 0:
                            mapped_idx = 0
                        else:
                            mapped_idx = int(np.floor(qi / (context_len - 1) * (range_len - 1)))
                        if mapped_idx < range_len:
                            mapped_indices.append(mapped_idx)
                    seen = set()
                    unique_indices = []
                    for idx in mapped_indices:
                        if idx not in seen:
                            seen.add(idx)
                            unique_indices.append(idx)
                    selected_paths = [gt_range_paths[i] for i in unique_indices]
                else:
                    valid_indices = [i for i in query_indices if i < range_len]
                    selected_paths = [gt_range_paths[i] for i in valid_indices]

                logger.info(f"Using query frame indices: {len(selected_paths)} frames "
                           f"(context_len={context_len}, divisor={query_idx_divisor})")
            elif range_len <= num_frames:
                selected_paths = gt_range_paths
            else:
                indices = np.linspace(0, range_len - 1, num_frames).round().astype(int)
                selected_paths = [gt_range_paths[i] for i in indices]
        else:
            if use_query_frame_indices:
                query_indices = get_query_frame_indices(context_len, query_idx_divisor)
                valid_indices = [i for i in query_indices if i < total_frames]
                selected_paths = [frame_paths[i] for i in valid_indices]
                logger.info(f"Using query frame indices: {len(selected_paths)} frames")
            elif total_frames <= num_frames:
                selected_paths = frame_paths
            else:
                indices = np.linspace(0, total_frames - 1, num_frames).round().astype(int)
                selected_paths = [frame_paths[i] for i in indices]

        images = [Image.open(path).convert("RGB") for path in selected_paths]
        if self.target_size is not None:
            w, h = self.target_size
            images = [img.resize((w, h), Image.BICUBIC) for img in images]
        return images

    @abstractmethod
    def build_messages(self, images: List[Image.Image]) -> List[dict]:
        """Build the messages list for the processor."""
        raise NotImplementedError

    def _compute_per_image_pixels(
        self,
        num_frames: int,
    ) -> Tuple[int, int]:
        """
        Compute per-image min/max pixels that match video mode behavior.

        Video processor applies min/max to T*H*W total pixels.
        For multi-image, we divide the budget by num_frames so each image
        gets the same effective resolution as it would in video mode.

        Reads `patch_size`, `merge_size`, `temporal_patch_size`, `size["shortest_edge"|"longest_edge"]`
        from the live `processor.video_processor` — these attributes exist on Qwen2.5-VL's
        Qwen2VLVideoProcessor (verified against the loaded Qwen2.5-VL-7B-Instruct processor).
        """
        video_proc = self.processor.video_processor
        video_min = video_proc.size["shortest_edge"]
        video_max = video_proc.size["longest_edge"]

        t_bar = math.ceil(num_frames / video_proc.temporal_patch_size) * video_proc.temporal_patch_size

        per_image_min = max(video_proc.patch_size * video_proc.merge_size, video_min // t_bar)
        per_image_max = video_max // t_bar

        return per_image_min, per_image_max

    @torch.no_grad()
    def forward_with_hidden_states(
        self,
        images: List[Image.Image],
        video_parity: bool = True,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns hidden states from specified layers.

        Returns:
            (hidden_states_by_layer, input_ids, visual_mask, image_grid_thw)
        """
        messages = self.build_messages(images)

        extra_kwargs = {}
        if video_parity and len(images) > 1:
            per_min, per_max = self._compute_per_image_pixels(len(images))
            extra_kwargs["min_pixels"] = per_min
            extra_kwargs["max_pixels"] = per_max

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **extra_kwargs,
        )

        logger.debug(f"Processor returned keys: {inputs.keys()}")

        inputs_on_device = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs_on_device[k] = v.to(self.device)
            else:
                inputs_on_device[k] = v

        has_image_grid = "image_grid_thw" in inputs_on_device
        has_video_grid = "video_grid_thw" in inputs_on_device
        if not has_image_grid and not has_video_grid:
            logger.warning(
                "Neither image_grid_thw nor video_grid_thw found in processor output. "
                f"Available keys: {list(inputs.keys())}"
            )

        outputs = self.model.model(
            **inputs_on_device,
            output_hidden_states=True,
            return_dict=True,
        )

        all_hidden_states = outputs.hidden_states
        result = {}
        for layer in self.select_layers:
            layer_idx = layer if layer >= 0 else len(all_hidden_states) + layer
            result[layer] = all_hidden_states[layer_idx]

        input_ids = inputs_on_device["input_ids"]
        visual_mask = (input_ids == self.image_token_id) | (input_ids == self.video_token_id)
        visual_mask = visual_mask.reshape(-1)

        image_grid_thw = inputs_on_device.get("image_grid_thw", None)

        return result, input_ids, visual_mask, image_grid_thw

    def extract_visual_hidden_states(
        self,
        hidden_states: Dict[int, torch.Tensor],
        visual_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract visual token hidden states and reshape to [T, H, W, C].

        image_grid_thw rows are [T_i, H_patches, W_patches] in patch units (before merge).
        We assume all frames in the multi-image input share the same H_patches and W_patches
        (true under the video-parity budget) so we can index row 0 for spatial dims.
        """
        merge_size = self.processor.image_processor.merge_size
        num_frames = image_grid_thw.shape[0]
        h = image_grid_thw[0, 1].item() // merge_size
        w = image_grid_thw[0, 2].item() // merge_size

        result = {}
        for layer, hs in hidden_states.items():
            B, seq_len, hidden_size = hs.shape
            hs_flat = hs.reshape(B * seq_len, hidden_size)
            visual_hs = hs_flat[visual_mask]
            visual_features = visual_hs.reshape(num_frames, h, w, hidden_size)
            result[layer] = visual_features

        return result

    @torch.no_grad()
    def extract(
        self,
        frame_dir: str,
        num_frames: int,
        frame_ext: str = "png",
        start_idx: int = 0,
        gt_num_frames: int = None,
        video_parity: bool = True,
        use_query_frame_indices: bool = False,
        context_len: int = 76,
        query_idx_divisor: int = 4,
    ) -> Dict[int, torch.Tensor]:
        """Main extraction method: load frames → forward → extract visual hidden states."""
        images = self.load_frames(
            frame_dir, num_frames, frame_ext,
            start_idx=start_idx, gt_num_frames=gt_num_frames,
            use_query_frame_indices=use_query_frame_indices,
            context_len=context_len, query_idx_divisor=query_idx_divisor,
        )

        actual_num_frames = len(images)
        logger.info(f"Loaded {actual_num_frames} frames")

        hidden_states, input_ids, visual_mask, image_grid_thw = self.forward_with_hidden_states(
            images, video_parity=video_parity
        )

        visual_features = self.extract_visual_hidden_states(
            hidden_states, visual_mask, image_grid_thw
        )

        return visual_features


@lru_cache(maxsize=4)
def get_qwen2_5_vl_extractor(
    model_path: str,
    model_type: str = "qwen25vl",
    select_layers: Tuple[int, ...] = (7, 14, 21, 28),
    question: str = "",
    device: str = "cuda:0",
    target_size: Optional[Tuple[int, int]] = (960, 540),
) -> BaseQwen2_5_VLExtractor:
    """Get a cached Qwen2.5-VL extractor instance."""
    from .qwen25vl_extractor import Qwen2_5_VLExtractor

    if model_type == "qwen25vl":
        return Qwen2_5_VLExtractor(
            model_path=model_path,
            select_layers=list(select_layers),
            question=question,
            device=device,
            target_size=target_size,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
