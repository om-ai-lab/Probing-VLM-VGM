"""
SenseNova SI InternVL3 feature extractor.

Uses the SenseNova prompt format:
"Image-1: <image>\nImage-2: <image>\n..." + question

This is for the spatial intelligence enhanced InternVL3 models.
"""

from __future__ import annotations

from typing import List, Optional

from .base_extractor import BaseInternVLExtractor


class SenseNovaInternVLExtractor(BaseInternVLExtractor):
    """
    Feature extractor for SenseNova Spatial Intelligence InternVL3 models.
    
    Uses the Image-N based prompt format for multi-image input.
    """
    
    def __init__(
        self,
        model_path: str,
        select_layers: List[int],
        question: str = "",
        **kwargs,
    ):
        """
        Initialize the SenseNova InternVL extractor.
        
        Args:
            model_path: Path to the SenseNova InternVL3 model
            select_layers: List of layer indices to extract
            question: Optional question to append to the prompt
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(
            model_path=model_path,
            select_layers=select_layers,
            question=question,
            **kwargs,
        )
    
    def build_prompt(self, num_frames: int) -> str:
        """
        Build prompt using SenseNova format.
        
        Format for single image: "<image>\n" + question
        Format for multiple images: "Image-1: <image>\nImage-2: <image>\n..." + question
        
        This follows the reorganize_prompt logic from sensenova_si/utils.py
        
        Args:
            num_frames: Number of frames in the input
            
        Returns:
            Formatted prompt string
        """
        if num_frames == 1:
            # Single image format
            prompt = "<image>\n"
            if self.question:
                prompt += self.question
        else:
            # Multi-image format
            image_prefix = "".join(
                [f"Image-{i+1}: <image>\n" for i in range(num_frames)]
            )
            if self.question:
                prompt = image_prefix + self.question
            else:
                prompt = image_prefix.rstrip("\n")
        
        return prompt
