"""
Standard Qwen2.5-VL feature extractor.

Uses multi-image format with Frame-N labels, indicating this is a continuous video sequence.
Each frame is processed independently for precise spatial alignment with GT.
"""

from __future__ import annotations

from typing import List

from PIL import Image

from .base_extractor import BaseQwen2_5_VLExtractor


class Qwen2_5_VLExtractor(BaseQwen2_5_VLExtractor):
    """
    Feature extractor for standard Qwen2.5-VL models.

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
        """
        content = []

        if len(images) == 1:
            content.append({"type": "image", "image": images[0]})
            if self.question:
                content.append({"type": "text", "text": self.question})
            else:
                content.append({"type": "text", "text": "Describe this image."})
        else:
            content.append({
                "type": "text",
                "text": "The following images are consecutive frames from a video.\n"
            })

            for i, img in enumerate(images):
                content.append({"type": "text", "text": f"Frame{i+1}: "})
                content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": "\n"})

            if self.question:
                content.append({"type": "text", "text": self.question})
            else:
                content.append({"type": "text", "text": "Describe this video."})

        messages = [{"role": "user", "content": content}]
        return messages
