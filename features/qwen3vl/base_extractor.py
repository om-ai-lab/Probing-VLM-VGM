"""
Base class for Qwen3-VL feature extraction.

This module provides the base class for extracting hidden states from Qwen3-VL models
for probe training and layer-wise analysis.
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
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)


def get_query_frame_indices(context_len: int = 76, query_idx_divisor: int = 4) -> List[int]:
    """
    Generate query frame indices matching video_probe_dataset_'s query mechanism.
    
    Rule: 0, 1, then 1 + k*query_idx_divisor (k=1,2,3,...), until < context_len
    Result for context_len=76, divisor=4: [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73]
    
    Args:
        context_len: Maximum frame index (exclusive), default 76
        query_idx_divisor: Alignment divisor, default 4
        
    Returns:
        List of frame indices (20 frames for default params)
    """
    indices = [0, 1]  # Frame 0 and frame 1
    idx = 1 + query_idx_divisor  # Start from 5
    while idx < context_len:
        indices.append(idx)
        idx += query_idx_divisor
    return indices


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 128 * 32 * 32,      # 131,072
    max_pixels: int = 32 * 32 * 768,       # 786,432
) -> Tuple[int, int]:
    """
    Calculate resized dimensions based on frame count (matches Qwen3-VL video processor).
    
    This ensures multi-image input uses the same resolution strategy as video input,
    where more frames result in smaller per-frame resolution to stay within pixel budget.
    
    Args:
        num_frames: Number of frames
        height: Original height
        width: Original width
        temporal_factor: Temporal patch size (default 2)
        factor: Spatial factor = patch_size * merge_size (default 32)
        min_pixels: Minimum total pixels
        max_pixels: Maximum total pixels
        
    Returns:
        Tuple of (resized_height, resized_width)
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


class BaseQwen3VLExtractor(ABC):
    """
    Base class for extracting hidden states from Qwen3-VL models.
    
    This class handles:
    - Model loading with multi-GPU support
    - Frame loading and preprocessing via AutoProcessor
    - Hidden states extraction from LLM layers
    - Reshaping to spatial format for probe training
    """
    
    def __init__(
        self,
        model_path: str,
        select_layers: List[int],
        question: str = "",
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        target_size: Optional[Tuple[int, int]] = (960, 540),
        attn_implementation: str = "sdpa",
    ):
        """
        Initialize the feature extractor.
        
        Args:
            model_path: Path to the Qwen3-VL model (local or HuggingFace hub)
            select_layers: List of layer indices to extract hidden states from
            question: The question/prompt to use for feature extraction
            device: Device to load model on (e.g., "cuda:0", "cuda:1", "cpu")
            torch_dtype: Data type for model weights
            target_size: (width, height) to resize all frames to before processing.
                         Set to None to keep original resolution.
                         Default (960, 540) matches DL3DV-960P.
            attn_implementation: Attention backend passed to Transformers.
                         Use "sdpa" for broad compatibility, or
                         "flash_attention_2" when a compatible flash-attn build
                         is available.
        """
        self.model_path = model_path
        self.select_layers = select_layers
        self.question = question
        self.torch_dtype = torch_dtype
        self.device = torch.device(device)
        self.target_size = target_size
        self.attn_implementation = attn_implementation
        
        logger.info(f"Loading Qwen3-VL model from {model_path}")
        logger.info(f"Target device: {self.device}")
        logger.info(f"Attention implementation: {self.attn_implementation}")
        
        # Load model
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=None,
            low_cpu_mem_usage=False,
        ).eval().to(self.device)
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # Get model configuration
        self.config = self.model.config
        self.hidden_size = self.config.text_config.hidden_size
        self.num_layers = self.config.text_config.num_hidden_layers
        self.image_token_id = self.config.image_token_id
        self.video_token_id = self.config.video_token_id
        
        # Validate select_layers
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
            gt_num_frames: Number of GT frames (for alignment with GT data)
            use_query_frame_indices: If True, use query frame indices matching dataset's
                query mechanism (e.g., [0,1,5,9,...,73]) instead of uniform sampling
            context_len: Context length for query frame indices (default 76)
            query_idx_divisor: Divisor for query frame alignment (default 4)
            
        Returns:
            List of PIL Images
        """
        frame_paths = sorted(glob.glob(f"{frame_dir}/frame_*.{frame_ext}"))
        total_frames = len(frame_paths)
        
        if total_frames == 0:
            raise ValueError(f"No frames found in {frame_dir}")
        
        # Determine the frame range to sample from
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
                # Use query frame indices matching dataset's query mechanism
                query_indices = get_query_frame_indices(context_len, query_idx_divisor)
                # Map query indices to GT range
                if range_len < context_len:
                    mapped_indices = []
                    for qi in query_indices:
                        if qi == 0:
                            mapped_idx = 0
                        else:
                            mapped_idx = int(np.floor(qi / (context_len - 1) * (range_len - 1)))
                        if mapped_idx < range_len:
                            mapped_indices.append(mapped_idx)
                    # Remove duplicates while preserving order
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
        
        # Load and optionally resize images
        images = [Image.open(path).convert("RGB") for path in selected_paths]
        if self.target_size is not None:
            w, h = self.target_size
            images = [img.resize((w, h), Image.BICUBIC) for img in images]
        return images
    
    @abstractmethod
    def build_messages(self, images: List[Image.Image]) -> List[dict]:
        """
        Build the messages list for the processor.
        
        Different Qwen3-VL variants may use different message formats.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Messages in OpenAI format for processor.apply_chat_template()
        """
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
        
        Args:
            num_frames: Number of frames
            
        Returns:
            (min_pixels_per_image, max_pixels_per_image)
        """
        video_proc = self.processor.video_processor
        video_min = video_proc.size["shortest_edge"]
        video_max = video_proc.size["longest_edge"]
        
        t_bar = math.ceil(num_frames / video_proc.temporal_patch_size) * video_proc.temporal_patch_size
        
        per_image_min = max(video_proc.patch_size * video_proc.merge_size, video_min // t_bar)
        per_image_max = video_max // t_bar
        
        # logger.info(
        #     f"Video parity: {num_frames} frames, t_bar={t_bar}, "
        #     f"per-image pixels: min={per_image_min}, max={per_image_max}"
        # )
        return per_image_min, per_image_max
    
    @torch.no_grad()
    def forward_with_hidden_states(
        self,
        images: List[Image.Image],
        video_parity: bool = True,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns hidden states from specified layers.
        
        Args:
            images: List of PIL Images
            video_parity: If True, set per-image min/max_pixels to match video mode
            
        Returns:
            Tuple of:
            - Dictionary mapping layer index to hidden states tensor
            - input_ids tensor
            - visual_mask (boolean tensor for visual token positions)
        """
        # Build messages
        messages = self.build_messages(images)
        
        # Compute per-image pixel limits for video parity
        extra_kwargs = {}
        if video_parity and len(images) > 1:
            per_min, per_max = self._compute_per_image_pixels(len(images))
            extra_kwargs["min_pixels"] = per_min
            extra_kwargs["max_pixels"] = per_max
        
        # Process with processor
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **extra_kwargs,
        )
        
        # Debug: log what keys processor returned
        logger.debug(f"Processor returned keys: {inputs.keys()}")
        
        # Move tensors to device
        inputs_on_device = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs_on_device[k] = v.to(self.device)
            else:
                inputs_on_device[k] = v
        
        # Ensure we have the required grid_thw parameters
        # The processor should return image_grid_thw or video_grid_thw
        # If not present, we need to handle this case
        has_image_grid = "image_grid_thw" in inputs_on_device
        has_video_grid = "video_grid_thw" in inputs_on_device
        
        if not has_image_grid and not has_video_grid:
            logger.warning(
                "Neither image_grid_thw nor video_grid_thw found in processor output. "
                f"Available keys: {list(inputs.keys())}"
            )
        
        # Forward with hidden states
        outputs = self.model.model(
            **inputs_on_device,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Extract hidden states for selected layers
        all_hidden_states = outputs.hidden_states
        result = {}
        
        for layer in self.select_layers:
            layer_idx = layer if layer >= 0 else len(all_hidden_states) + layer
            result[layer] = all_hidden_states[layer_idx]
        
        # Get visual token mask
        input_ids = inputs_on_device["input_ids"]
        visual_mask = (input_ids == self.image_token_id) | (input_ids == self.video_token_id)
        visual_mask = visual_mask.reshape(-1)
        
        # Get image_grid_thw for spatial reshape
        image_grid_thw = inputs_on_device.get("image_grid_thw", None)
        
        return result, input_ids, visual_mask, image_grid_thw
    
    def extract_visual_hidden_states(
        self,
        hidden_states: Dict[int, torch.Tensor],
        visual_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract only the visual token hidden states and reshape to spatial format.
        
        Args:
            hidden_states: Dict of layer -> hidden states [B, seq_len, hidden_size]
            visual_mask: Boolean mask for visual token positions
            image_grid_thw: Tensor of shape [N, 3], each row [T, H_patches, W_patches]
                            from processor output. H/W are in patch units (before merge).
            
        Returns:
            Dict of layer -> visual hidden states [T, H, W, C]
        """
        merge_size = self.processor.image_processor.merge_size
        num_frames = image_grid_thw.shape[0]
        # H, W in token units (after merge)
        h = image_grid_thw[0, 1].item() // merge_size
        w = image_grid_thw[0, 2].item() // merge_size
        
        result = {}
        for layer, hs in hidden_states.items():
            B, seq_len, hidden_size = hs.shape
            hs_flat = hs.reshape(B * seq_len, hidden_size)
            visual_hs = hs_flat[visual_mask]  # [num_visual_tokens, hidden_size]
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
        """
        Main extraction method: load frames, forward, extract visual hidden states.
        
        Args:
            frame_dir: Directory containing frames
            num_frames: Number of frames to sample (ignored if use_query_frame_indices=True)
            frame_ext: Frame file extension
            start_idx: Starting frame index (from GT data_sft)
            gt_num_frames: Number of GT frames (for alignment)
            video_parity: If True, resize images to match video mode resolution
            use_query_frame_indices: If True, use query frame indices matching dataset
            context_len: Context length for query frame indices
            query_idx_divisor: Divisor for query frame alignment
            
        Returns:
            Dict of layer -> visual features [T, H, W, C]
        """
        # Load frames
        images = self.load_frames(
            frame_dir, num_frames, frame_ext,
            start_idx=start_idx, gt_num_frames=gt_num_frames,
            use_query_frame_indices=use_query_frame_indices,
            context_len=context_len, query_idx_divisor=query_idx_divisor,
        )
        
        actual_num_frames = len(images)
        logger.info(f"Loaded {actual_num_frames} frames")
        
        # Forward pass (with optional video parity resize)
        hidden_states, input_ids, visual_mask, image_grid_thw = self.forward_with_hidden_states(
            images, video_parity=video_parity
        )
        
        # Extract and reshape visual hidden states
        visual_features = self.extract_visual_hidden_states(
            hidden_states, visual_mask, image_grid_thw
        )
        
        return visual_features


@lru_cache(maxsize=4)
def get_qwen3vl_extractor(
    model_path: str,
    model_type: str = "qwen3vl",
    select_layers: Tuple[int, ...] = (8, 16, 24, 32),
    question: str = "",
    device: str = "cuda:0",
    target_size: Optional[Tuple[int, int]] = (960, 540),
    attn_implementation: str = "sdpa",
) -> BaseQwen3VLExtractor:
    """
    Get a cached Qwen3-VL extractor instance.
    
    Args:
        model_path: Path to the model
        model_type: "qwen3vl" or "sensenova"
        select_layers: Tuple of layers to extract (tuple for hashability)
        question: The question/prompt
        device: Device to load model on (e.g., "cuda:0", "cuda:1")
        target_size: (width, height) to resize frames to, or None for original
        attn_implementation: Transformers attention backend.
        
    Returns:
        The extractor instance
    """
    from .qwen3vl_extractor import Qwen3VLExtractor
    from .sensenova_extractor import SenseNovaQwen3VLExtractor
    
    if model_type == "qwen3vl":
        return Qwen3VLExtractor(
            model_path=model_path,
            select_layers=list(select_layers),
            question=question,
            device=device,
            target_size=target_size,
            attn_implementation=attn_implementation,
        )
    elif model_type == "sensenova":
        return SenseNovaQwen3VLExtractor(
            model_path=model_path,
            select_layers=list(select_layers),
            question=question,
            device=device,
            target_size=target_size,
            attn_implementation=attn_implementation,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
