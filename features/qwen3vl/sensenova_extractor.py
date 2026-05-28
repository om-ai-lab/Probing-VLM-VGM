"""
SenseNova SI Qwen3-VL feature extractor.

Matches the SenseNova Qwen inference format (to_openai_format):
images and text are interleaved without extra labels like "Image-N:".
"""

from __future__ import annotations

from typing import List

from PIL import Image

from .base_extractor import BaseQwen3VLExtractor


class SenseNovaQwen3VLExtractor(BaseQwen3VLExtractor):
    """
    Feature extractor for SenseNova Spatial Intelligence Qwen3-VL models.
    
    Matches the to_openai_format used in sensenova_si/qwen.py:
    images are placed inline without Image-N labels.
    """
    
    def __init__(
        self,
        model_path: str,
        select_layers: List[int],
        question: str = "",
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            select_layers=select_layers,
            question=question,
            **kwargs,
        )
    
    def build_messages(self, images: List[Image.Image]) -> List[dict]:
        """
        Build messages matching SenseNova Qwen format.
        
        Images are placed inline followed by the question text,
        consistent with the training data format:
            <image>\n<image>\n...question
        
        Args:
            images: List of PIL Images (frames)
            
        Returns:
            Messages in OpenAI format
        """
        content = []
        
        for img in images:
            content.append({"type": "image", "image": img})
        
        if self.question:
            content.append({"type": "text", "text": self.question})
        else:
            text = "Describe this image." if len(images) == 1 else "Describe these images."
            content.append({"type": "text", "text": text})
        
        return [{"role": "user", "content": content}]
