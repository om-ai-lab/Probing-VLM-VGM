"""
Standard Qwen3-VL feature extractor.

Uses multi-image format with Frame-N labels, indicating this is a continuous video sequence.
Each frame is processed independently for precise spatial alignment with GT.
"""

from __future__ import annotations

from typing import List

from PIL import Image

from .base_extractor import BaseQwen3VLExtractor


class Qwen3VLExtractor(BaseQwen3VLExtractor):
    """
    Feature extractor for standard Qwen3-VL models.
    
    Uses multi-image format with explicit frame labels.
    This allows precise per-frame spatial alignment for probe training.
    """
    
    def __init__(
        self,
        model_path: str,
        select_layers: List[int],
        question: str = "",
        **kwargs,
    ):
        """
        Initialize the Qwen3-VL extractor.
        
        Args:
            model_path: Path to the Qwen3-VL model
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
    
    def build_messages(self, images: List[Image.Image]) -> List[dict]:
        """
        Build messages using multi-image format with video context.
        
        Format:
        - Prefix: "The following images are consecutive frames from a video."
        - Each frame: "Frame1: <image>\nFrame2: <image>\n..."
        - Question at the end
        
        Args:
            images: List of PIL Images (frames)
            
        Returns:
            Messages in OpenAI format
        """
        content = []
        
        if len(images) == 1:
            # Single image - no video context needed
            content.append({"type": "image", "image": images[0]})
            if self.question:
                content.append({"type": "text", "text": self.question})
            else:
                content.append({"type": "text", "text": "Describe this image."})
        else:
            # Multiple frames - add video context
            content.append({
                "type": "text",
                "text": "The following images are consecutive frames from a video.\n"
            })
            
            for i, img in enumerate(images):
                content.append({"type": "text", "text": f"Frame{i+1}: "})
                content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": "\n"})
            
            # Add question
            if self.question:
                content.append({"type": "text", "text": self.question})
            else:
                content.append({"type": "text", "text": "Describe this video."})
        
        messages = [{"role": "user", "content": content}]
        return messages
