"""
Base class for InternVL feature extraction.

This module provides the base class for extracting hidden states from InternVL models
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
import torch.nn as nn
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .utils import build_transform, dynamic_preprocess

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


# Import conversation template utilities from model's remote code
# These will be imported dynamically when the model is loaded
_conv_template_module = None


def _get_conv_template(model_path: str, template_name: str):
    """
    Get conversation template from the model's remote code.
    
    Args:
        model_path: Path to the model (to locate conversation.py)
        template_name: Name of the template to use
        
    Returns:
        Conversation template object
    """
    global _conv_template_module
    
    if _conv_template_module is None:
        import importlib.util
        import os
        
        # Try to load conversation.py from model path
        conv_path = os.path.join(model_path, "conversation.py")
        if os.path.exists(conv_path):
            spec = importlib.util.spec_from_file_location("conversation", conv_path)
            _conv_template_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_conv_template_module)
        else:
            raise FileNotFoundError(
                f"conversation.py not found at {conv_path}. "
                "Please ensure the model path contains the remote code files."
            )
    
    return _conv_template_module.get_conv_template(template_name)


class BaseInternVLExtractor(ABC):
    """
    Base class for extracting hidden states from InternVL models.
    
    This class handles:
    - Model loading with multi-GPU support
    - Frame loading and preprocessing
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
    ):
        """
        Initialize the feature extractor.
        
        Args:
            model_path: Path to the InternVL model (local or HuggingFace hub)
            select_layers: List of layer indices to extract hidden states from
            question: The question/prompt to use for feature extraction
            device: Device to load model on (e.g., "cuda:0", "cuda:1", "cpu")
            torch_dtype: Data type for model weights
        """
        self.model_path = model_path
        self.select_layers = select_layers
        self.question = question
        self.torch_dtype = torch_dtype
        self.device = torch.device(device)
        
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Target device: {self.device}")
        
        # Load model directly to specified device
        # Note: Must set device_map=None and low_cpu_mem_usage=False to avoid
        # meta tensor creation, which breaks InternVL's vision model init
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            device_map=None,
            low_cpu_mem_usage=False,
        ).eval().to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        
        # Get model configuration
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.num_image_token = self.model.num_image_token
        self.hidden_size = self.config.llm_config.hidden_size
        self.num_layers = self.config.llm_config.num_hidden_layers
        
        # Validate select_layers
        # Note: hidden_states has num_layers+1 elements (embeddings + decoder layers)
        # Layer 0 = embeddings, Layer N = final decoder layer
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
        max_num: int = 1,
        frame_ext: str = "png",
        input_size: int = 448,
        start_idx: int = 0,
        gt_num_frames: int = None,
        use_query_frame_indices: bool = False,
        context_len: int = 76,
        query_idx_divisor: int = 4,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Load and preprocess frames from a directory.
        
        Args:
            frame_dir: Directory containing frame_*.{ext} files
            num_frames: Number of frames to sample (ignored if use_query_frame_indices=True)
            max_num: Maximum number of tiles per frame (1 for video)
            frame_ext: Frame file extension
            input_size: Input image size
            start_idx: Starting frame index (from GT data_sft)
            gt_num_frames: Number of GT frames (for alignment with GT data)
            use_query_frame_indices: If True, use query frame indices matching dataset's
                query mechanism (e.g., [0,1,5,9,...,73]) instead of uniform sampling
            context_len: Context length for query frame indices (default 76)
            query_idx_divisor: Divisor for query frame alignment (default 4)
            
        Returns:
            pixel_values: Tensor of shape [total_tiles, 3, input_size, input_size]
            num_patches_list: List of tile counts per frame
        """
        # Get all frame paths
        frame_paths = sorted(glob.glob(f"{frame_dir}/frame_*.{frame_ext}"))
        total_frames = len(frame_paths)
        
        if total_frames == 0:
            raise ValueError(f"No frames found in {frame_dir}")
        
        # Determine the frame range to sample from
        if gt_num_frames is not None:
            # Sample from GT range [start_idx, start_idx + gt_num_frames)
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
                # This ensures extracted features align with video_probe_dataset_'s query sampling
                query_indices = get_query_frame_indices(context_len, query_idx_divisor)
                # Map query indices to GT range (query indices are relative to context window)
                # GT range is downsampled from original video, so we need to map:
                # query_idx in [0, context_len) -> GT frame idx in [0, range_len)
                # Using the same mapping as dataset: floor((idx) / (context_len - 1) * (range_len - 1))
                if range_len < context_len:
                    # GT is shorter than context, need to map query indices
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
                    # GT range covers context window, use query indices directly
                    valid_indices = [i for i in query_indices if i < range_len]
                    selected_paths = [gt_range_paths[i] for i in valid_indices]
                
                logger.info(f"Using query frame indices: {len(selected_paths)} frames "
                           f"(context_len={context_len}, divisor={query_idx_divisor})")
            else:
                # Uniformly sample num_frames from GT range
                # Use round() to match VideoProbeDataset's gt_num_frames downsampling
                if range_len <= num_frames:
                    selected_paths = gt_range_paths
                else:
                    indices = np.linspace(0, range_len - 1, num_frames).round().astype(int)
                    selected_paths = [gt_range_paths[i] for i in indices]
        else:
            # Fallback: sample from entire video (legacy behavior)
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
        
        # Process each frame
        transform = build_transform(input_size=input_size)
        pixel_values_list = []
        num_patches_list = []
        
        for path in selected_paths:
            img = Image.open(path).convert("RGB")
            
            # Dynamic preprocess: split into tiles
            tiles = dynamic_preprocess(
                img,
                image_size=input_size,
                use_thumbnail=True,
                max_num=max_num,
            )
            
            pixel_values = torch.stack([transform(tile) for tile in tiles])
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list
    
    @abstractmethod
    def build_prompt(self, num_frames: int) -> str:
        """
        Build the prompt string for the model.
        
        Different InternVL variants use different prompt formats:
        - Original: "Frame1: <image>\nFrame2: <image>\n..."
        - SenseNova: "Image-1: <image>\nImage-2: <image>\n..."
        
        Args:
            num_frames: Number of frames in the input
            
        Returns:
            The formatted prompt string
        """
        raise NotImplementedError
    
    def _build_query_with_template(
        self,
        question: str,
        num_patches_list: List[int],
    ) -> str:
        """
        Build the full query string using the model's conversation template.
        
        This replicates the logic from InternVL's chat() method:
        1. Get conversation template
        2. Add user message
        3. Format with get_prompt()
        4. Replace <image> with proper IMG_CONTEXT token sequences
        
        Args:
            question: The question/prompt containing <image> placeholders
            num_patches_list: Number of patches per image
            
        Returns:
            Fully formatted query string ready for tokenization
        """
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        
        # Get conversation template from model
        template_name = getattr(self.model, "template", "internlm2-chat")
        template = _get_conv_template(self.model_path, template_name)
        
        # Set system message if available
        if hasattr(self.model, "system_message"):
            template.system_message = self.model.system_message
        
        # Add user message and assistant placeholder
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        
        # Get formatted prompt
        query = template.get_prompt()
        
        # Replace <image> placeholders with IMG_CONTEXT token sequences
        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
        
        return query

    @torch.no_grad()
    def forward_with_hidden_states(
        self,
        pixel_values: torch.Tensor,
        num_patches_list: List[int],
    ) -> Dict[int, torch.Tensor]:
        """
        Forward pass that returns hidden states from specified layers.
        
        This method replicates InternVL's chat() tokenization logic to ensure
        correct IMG_CONTEXT token placement and conversation formatting.
        
        Args:
            pixel_values: Preprocessed image tensor [N, 3, H, W]
            num_patches_list: Number of patches per frame
            
        Returns:
            Dictionary mapping layer index to hidden states tensor
        """
        pixel_values = pixel_values.to(device=self.device, dtype=self.torch_dtype)
        
        # Build prompt with <image> placeholders
        num_frames = len(num_patches_list)
        question = self.build_prompt(num_frames)
        
        # Build full query using conversation template (same as InternVL's chat method)
        query = self._build_query_with_template(question, num_patches_list)
        
        # Tokenize
        model_inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)
        
        # Get IMG_CONTEXT token id
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id
        
        # Extract visual features
        vit_embeds = self.model.extract_feature(pixel_values)
        
        # Build input embeddings
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        
        # Find IMG_CONTEXT token positions
        input_ids_flat = input_ids.reshape(B * N)
        selected = (input_ids_flat == img_context_token_id)
        
        # Verify visual token count matches expectation
        expected_visual_tokens = sum(n * self.num_image_token for n in num_patches_list)
        actual_visual_tokens = selected.sum().item()
        
        if actual_visual_tokens != expected_visual_tokens:
            raise ValueError(
                f"Visual token mismatch: expected {expected_visual_tokens}, "
                f"got {actual_visual_tokens}. Query length: {N}"
            )
        
        if actual_visual_tokens == 0:
            raise ValueError(
                "No IMG_CONTEXT tokens found in input. "
                "Check that conversation template is being applied correctly."
            )
        
        # Replace IMG_CONTEXT tokens with visual embeddings
        vit_embeds_flat = vit_embeds.reshape(-1, vit_embeds.shape[-1])
        n_visual = min(selected.sum(), vit_embeds_flat.size(0))
        selected_indices = torch.where(selected)[0][:n_visual]
        input_embeds[selected_indices] = vit_embeds_flat[:n_visual].to(input_embeds.dtype)
        
        input_embeds = input_embeds.reshape(B, N, C)
        
        # Forward with hidden states
        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Extract hidden states for selected layers
        all_hidden_states = outputs.hidden_states
        result = {}
        
        for layer in self.select_layers:
            # Handle negative indices
            layer_idx = layer if layer >= 0 else len(all_hidden_states) + layer
            result[layer] = all_hidden_states[layer_idx]
        
        return result, input_ids, selected
    
    def extract_visual_hidden_states(
        self,
        hidden_states: Dict[int, torch.Tensor],
        input_ids: torch.Tensor,
        visual_mask: torch.Tensor,
        num_patches_list: List[int],
    ) -> Dict[int, torch.Tensor]:
        """
        Extract only the visual token hidden states and reshape to spatial format.
        
        Args:
            hidden_states: Dict of layer -> hidden states [B, seq_len, hidden_size]
            input_ids: Input token IDs
            visual_mask: Boolean mask for visual token positions
            num_patches_list: Number of patches per frame
            
        Returns:
            Dict of layer -> visual hidden states [T, H, W, C]
        """
        result = {}
        
        for layer, hs in hidden_states.items():
            # hs shape: [B, seq_len, hidden_size]
            B, seq_len, hidden_size = hs.shape
            
            # Extract visual tokens
            hs_flat = hs.reshape(B * seq_len, hidden_size)
            visual_hs = hs_flat[visual_mask]  # [num_visual_tokens, hidden_size]
            
            # Reshape to spatial format per frame
            visual_features = []
            start_idx = 0
            
            for num_patches in num_patches_list:
                # Each patch contributes num_image_token tokens
                num_tokens = num_patches * self.num_image_token
                frame_hs = visual_hs[start_idx:start_idx + num_tokens]
                
                # Reshape to spatial: assuming square patches
                # When max_num=1, we have 1 patch with num_image_token tokens
                # The tokens are arranged as H x W grid
                h = w = int(math.sqrt(self.num_image_token))
                if h * w != self.num_image_token:
                    # Non-square, use original token count
                    h = w = int(math.sqrt(num_tokens))
                
                if num_patches == 1:
                    frame_hs = frame_hs.reshape(h, w, hidden_size)
                else:
                    # Multiple patches: reshape each patch then concatenate
                    # This is more complex, for now just reshape flat
                    total_h = int(math.sqrt(num_tokens))
                    total_w = num_tokens // total_h
                    frame_hs = frame_hs.reshape(total_h, total_w, hidden_size)
                
                visual_features.append(frame_hs)
                start_idx += num_tokens
            
            # Stack frames: [T, H, W, C]
            result[layer] = torch.stack(visual_features, dim=0)
        
        return result
    
    @torch.no_grad()
    def extract(
        self,
        frame_dir: str,
        num_frames: int,
        max_num: int = 1,
        frame_ext: str = "png",
        start_idx: int = 0,
        gt_num_frames: int = None,
        use_query_frame_indices: bool = False,
        context_len: int = 76,
        query_idx_divisor: int = 4,
    ) -> Dict[int, torch.Tensor]:
        """
        Main extraction method: load frames, forward, extract visual hidden states.
        
        Args:
            frame_dir: Directory containing frames
            num_frames: Number of frames to sample (ignored if use_query_frame_indices=True)
            max_num: Max tiles per frame
            frame_ext: Frame file extension
            start_idx: Starting frame index (from GT data_sft)
            gt_num_frames: Number of GT frames (for alignment)
            use_query_frame_indices: If True, use query frame indices matching dataset
            context_len: Context length for query frame indices
            query_idx_divisor: Divisor for query frame alignment
            
        Returns:
            Dict of layer -> visual features [T, H, W, C]
        """
        # Load frames
        pixel_values, num_patches_list = self.load_frames(
            frame_dir, num_frames, max_num, frame_ext,
            start_idx=start_idx, gt_num_frames=gt_num_frames,
            use_query_frame_indices=use_query_frame_indices,
            context_len=context_len, query_idx_divisor=query_idx_divisor,
        )
        
        logger.info(f"Loaded {len(num_patches_list)} frames with patches: {num_patches_list}")
        
        # Forward pass
        hidden_states, input_ids, visual_mask = self.forward_with_hidden_states(
            pixel_values, num_patches_list
        )
        
        # Extract and reshape visual hidden states
        visual_features = self.extract_visual_hidden_states(
            hidden_states, input_ids, visual_mask, num_patches_list
        )
        
        return visual_features


@lru_cache(maxsize=4)
def get_internvl_extractor(
    model_path: str,
    model_type: str = "internvl3",
    select_layers: Tuple[int, ...] = (8, 16, 24, 32),
    question: str = "",
    device: str = "cuda:0",
) -> BaseInternVLExtractor:
    """
    Get a cached InternVL extractor instance.
    
    Args:
        model_path: Path to the model
        model_type: "internvl3", "internvl35", or "sensenova"
        select_layers: Tuple of layers to extract (tuple for hashability)
        question: The question/prompt
        device: Device to load model on (e.g., "cuda:0", "cuda:1")
        
    Returns:
        The extractor instance
    """
    from .internvl3_extractor import InternVL3Extractor
    from .sensenova_extractor import SenseNovaInternVLExtractor
    
    if model_type in {"internvl3", "internvl35"}:
        return InternVL3Extractor(
            model_path=model_path,
            select_layers=list(select_layers),
            question=question,
            device=device,
        )
    elif model_type == "sensenova":
        return SenseNovaInternVLExtractor(
            model_path=model_path,
            select_layers=list(select_layers),
            device=device,
            question=question,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
