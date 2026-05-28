"""
Original InternVL3 feature extractor.

Uses the standard InternVL3 prompt format:
"Frame1: <image>\nFrame2: <image>\n..." + question
"""

from __future__ import annotations

from typing import List, Optional

from .base_extractor import BaseInternVLExtractor


class InternVL3Extractor(BaseInternVLExtractor):
    """
    Feature extractor for original InternVL3 models.
    
    Uses the standard frame-based prompt format for video input.
    """
    
    def __init__(
        self,
        model_path: str,
        select_layers: List[int],
        question: str = "",
        **kwargs,
    ):
        """
        Initialize the InternVL3 extractor.
        
        Args:
            model_path: Path to the InternVL3 model
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
        Build prompt using original InternVL3 format.
        
        Format: "Frame1: <image>\nFrame2: <image>\n..." + question
        
        Args:
            num_frames: Number of frames in the input
            
        Returns:
            Formatted prompt string
        """
        # Build frame prefix
        frame_prefix = "".join(
            [f"Frame{i+1}: <image>\n" for i in range(num_frames)]
        )
        
        # Combine with question
        if self.question:
            prompt = frame_prefix + self.question
        else:
            prompt = frame_prefix.rstrip("\n")
        
        return prompt
