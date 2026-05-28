"""VideoLLaMA3 feature extractor.

The extractor follows the same contract as the other VLM extractors in this
repo: run a multimodal forward pass, pull LLM hidden states at visual-token
positions, reshape them to ``[T, H, W, C]``, and let ``extract_features.py``
save each selected layer as ``feature_layer{L}.sft`` with key ``feat``.

VideoLLaMA3's default token compression is disabled by default here. The
compression mask is content dependent, so leaving it on would produce ragged
visual token sequences that cannot be reshaped into a stable patch grid.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module

logger = logging.getLogger(__name__)


def _ensure_transformers_video_input_alias() -> None:
    """Patch older transformers builds for VideoLLaMA3 remote processor code.

    The VideoLLaMA3 processor imports ``VideoInput`` from ``transformers.image_utils``.
    Some installed transformers builds in this environment do not expose that typing
    alias, although the runtime image/video helpers it uses are present. The alias is
    only used for annotations, so adding it is enough to let ``trust_remote_code`` load.
    """
    import typing

    import transformers.image_utils as image_utils

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = typing.Any


def _processor_class_ref(model_path: str) -> str:
    for filename in ("processor_config.json", "preprocessor_config.json", "tokenizer_config.json"):
        path = os.path.join(model_path, filename)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        ref = cfg.get("auto_map", {}).get("AutoProcessor")
        if isinstance(ref, (list, tuple)):
            ref = ref[0]
        if ref:
            return ref
    return "processing_videollama3.Videollama3Qwen2Processor"


def _load_videollama3_processor(model_path: str):
    """Load the remote VideoLLaMA3 processor with local compatibility patches."""
    _ensure_transformers_video_input_alias()
    processor_class = get_class_from_dynamic_module(
        _processor_class_ref(model_path),
        model_path,
    )

    original = processor_class._get_arguments_from_pretrained.__func__

    def compat_get_arguments(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):
        return original(cls, pretrained_model_name_or_path, **kwargs)

    processor_class._get_arguments_from_pretrained = classmethod(compat_get_arguments)
    return processor_class.from_pretrained(model_path, trust_remote_code=True)


def get_query_frame_indices(context_len: int = 76, query_idx_divisor: int = 4) -> List[int]:
    """Generate the query-frame sequence used by the ScanNet probe datasets."""
    indices = [0, 1]
    idx = 1 + query_idx_divisor
    while idx < context_len:
        indices.append(idx)
        idx += query_idx_divisor
    return indices


class VideoLLaMA3Extractor:
    """Extract LLM-layer visual token features from VideoLLaMA3."""

    def __init__(
        self,
        model_path: str,
        select_layers: List[int],
        question: str = "",
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        target_size: Optional[Tuple[int, int]] = (960, 540),
        attn_implementation: str = "flash_attention_2",
        use_token_compression: bool = False,
    ):
        self.model_path = model_path
        self.select_layers = list(select_layers)
        self.question = question
        self.torch_dtype = torch_dtype
        self.device = torch.device(device)
        self.target_size = target_size
        self.attn_implementation = attn_implementation
        self.use_token_compression = use_token_compression

        logger.info("Loading VideoLLaMA3 model from %s", model_path)
        logger.info("Target device: %s", self.device)
        _ensure_transformers_video_input_alias()

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.device.type == "cuda":
            model_kwargs["device_map"] = {"": str(self.device)}
        else:
            model_kwargs["device_map"] = None
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        ).eval()
        if model_kwargs["device_map"] is None:
            self.model = self.model.to(self.device)
        self.processor = _load_videollama3_processor(model_path)

        self.config = self.model.config
        self.hidden_size = int(self.config.hidden_size)
        self.num_layers = int(self.config.num_hidden_layers)
        self.image_token_id = int(self.config.image_token_index)

        # Keep the model behavior explicit. Content-dependent compression breaks
        # the rectangular visual grid expected by the probe datasets.
        self.model.config.use_token_compression = bool(use_token_compression)

        for layer in self.select_layers:
            resolved = self._resolve_layer_index(layer, self.num_layers + 1)
            if resolved < 0 or resolved > self.num_layers:
                raise ValueError(
                    f"Layer {layer} out of range. Model has {self.num_layers} layers "
                    f"(valid hidden-state indices: 0-{self.num_layers})."
                )

        logger.info(
            "Model loaded. LLM hidden=%d, layers=%d, image_token_id=%d, "
            "use_token_compression=%s",
            self.hidden_size,
            self.num_layers,
            self.image_token_id,
            self.model.config.use_token_compression,
        )
        logger.info("Extracting layers: %s", self.select_layers)

    @staticmethod
    def _resolve_layer_index(layer: int, num_hidden_states: int) -> int:
        """Map a requested layer to the hidden_states tuple index."""
        return layer if layer >= 0 else num_hidden_states + layer

    def load_frames(
        self,
        frame_dir: str,
        num_frames: int,
        frame_ext: str = "png",
        start_idx: int = 0,
        gt_num_frames: int | None = None,
        use_query_frame_indices: bool = False,
        context_len: int = 76,
        query_idx_divisor: int = 4,
    ) -> List[Image.Image]:
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
                    f"No frames in GT range [{start_idx}, {end_idx}), total frames: {total_frames}"
                )

            if use_query_frame_indices:
                query_indices = get_query_frame_indices(context_len, query_idx_divisor)
                if range_len < context_len:
                    mapped_indices = []
                    for qi in query_indices:
                        mapped_idx = 0 if qi == 0 else int(np.floor(qi / (context_len - 1) * (range_len - 1)))
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
                    selected_paths = [gt_range_paths[i] for i in query_indices if i < range_len]
                logger.info(
                    "Using query frame indices: %d frames (context_len=%d, divisor=%d)",
                    len(selected_paths),
                    context_len,
                    query_idx_divisor,
                )
            elif range_len <= num_frames:
                selected_paths = gt_range_paths
            else:
                indices = np.linspace(0, range_len - 1, num_frames).round().astype(int)
                selected_paths = [gt_range_paths[i] for i in indices]
        else:
            if use_query_frame_indices:
                query_indices = get_query_frame_indices(context_len, query_idx_divisor)
                selected_paths = [frame_paths[i] for i in query_indices if i < total_frames]
                logger.info("Using query frame indices: %d frames", len(selected_paths))
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

    def _build_conversation(self, images: List[Image.Image]) -> List[dict]:
        text = self.question.strip() if self.question else "Describe this video."
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": images,
                        "num_frames": len(images),
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]

    def _downsampled_grid_sizes(self, image_inputs: dict) -> List[torch.Tensor]:
        grid_sizes = []
        for grid_size, merge_size in zip(image_inputs.get("grid_sizes", []), image_inputs.get("merge_sizes", [])):
            if not torch.all(grid_size[1:] % merge_size == 0):
                raise RuntimeError(
                    f"Grid size {grid_size.tolist()} is not divisible by merge_size={int(merge_size)}"
                )
            if int(grid_size[0].item()) == 1:
                grid_sizes.append(grid_size[1:] // merge_size)
            elif int(grid_size[0].item()) > 1:
                grid_sizes.extend([grid_size[1:] // merge_size] * int(grid_size[0].item()))
        return grid_sizes

    def _expand_visual_tokens(self, text: str, image_inputs: dict) -> str:
        image_token = "<image>"
        placeholder = "<placeholder>"
        image_idx = 0
        grid_sizes = self._downsampled_grid_sizes(image_inputs)
        while image_token in text:
            if image_idx >= len(grid_sizes):
                raise RuntimeError(
                    f"Prompt has more {image_token} tokens than visual grids: {len(grid_sizes)}"
                )
            num_tokens = int(grid_sizes[image_idx].prod().item())
            text = text.replace(image_token, placeholder * num_tokens, 1)
            image_idx += 1
        if image_idx != len(grid_sizes):
            raise RuntimeError(
                f"Prompt expanded {image_idx} visual items, but processor produced {len(grid_sizes)} grids"
            )
        return text.replace(placeholder, image_token)

    def _prepare_inputs(self, images: List[Image.Image]) -> dict:
        """Prepare VideoLLaMA3 inputs without calling the version-sensitive processor __call__."""
        conversation = self._build_conversation(images)
        image_inputs = self.processor.image_processor(
            images=[images],
            merge_size=[self.processor.video_merge_size],
            return_tensors="pt",
        )
        image_inputs["modals"] = ["video"]

        prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = self._expand_visual_tokens(prompt, image_inputs)
        text_inputs = self.processor.tokenizer(prompt, return_tensors="pt")
        return {**text_inputs, **image_inputs}

    def _move_inputs_to_device(self, inputs: dict) -> dict:
        result = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                value = value.to(self.device)
                if key == "pixel_values":
                    value = value.to(self.torch_dtype)
            result[key] = value
        return result

    @torch.no_grad()
    def forward_with_hidden_states(
        self,
        images: List[Image.Image],
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self._prepare_inputs(images)

        if "input_ids" not in inputs:
            raise RuntimeError(f"Processor output has no input_ids. Keys: {list(inputs.keys())}")
        if "grid_sizes" not in inputs or "merge_sizes" not in inputs:
            raise RuntimeError(f"Processor output has no grid_sizes/merge_sizes. Keys: {list(inputs.keys())}")

        grid_sizes = inputs["grid_sizes"]
        merge_sizes = inputs["merge_sizes"]
        visual_mask = (inputs["input_ids"] == self.image_token_id).reshape(-1)
        expected_tokens = int((grid_sizes.prod(dim=1) // (merge_sizes ** 2)).sum().item())
        actual_tokens = int(visual_mask.sum().item())
        if actual_tokens != expected_tokens:
            raise RuntimeError(
                f"Processor produced {actual_tokens} visual tokens, expected {expected_tokens} "
                f"from grid_sizes={grid_sizes.tolist()} merge_sizes={merge_sizes.tolist()}. "
                "Check conversation formatting and token compression settings."
            )

        inputs_on_device = self._move_inputs_to_device(dict(inputs))
        outputs = self.model(
            **inputs_on_device,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        all_hidden = outputs.hidden_states
        result = {}
        for layer in self.select_layers:
            layer_idx = self._resolve_layer_index(layer, len(all_hidden))
            result[layer] = all_hidden[layer_idx]

        return (
            result,
            inputs_on_device["input_ids"],
            visual_mask.to(self.device),
            inputs_on_device["grid_sizes"],
            inputs_on_device["merge_sizes"],
        )

    def extract_visual_hidden_states(
        self,
        hidden_states: Dict[int, torch.Tensor],
        visual_mask: torch.Tensor,
        grid_sizes: torch.Tensor,
        merge_sizes: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        if grid_sizes.shape[0] != 1:
            raise RuntimeError(
                f"Expected one video item, got grid_sizes shape {tuple(grid_sizes.shape)}"
            )
        grid_t, grid_h, grid_w = [int(x) for x in grid_sizes[0].tolist()]
        merge_size = int(merge_sizes[0].item())
        if grid_h % merge_size != 0 or grid_w % merge_size != 0:
            raise RuntimeError(
                f"grid_sizes {grid_sizes[0].tolist()} not divisible by merge_size={merge_size}"
            )
        h = grid_h // merge_size
        w = grid_w // merge_size
        expected_tokens = grid_t * h * w

        result = {}
        for layer, hs in hidden_states.items():
            batch, seq_len, hidden_size = hs.shape
            hs_flat = hs.reshape(batch * seq_len, hidden_size)
            visual_hs = hs_flat[visual_mask]
            if visual_hs.shape[0] != expected_tokens:
                raise RuntimeError(
                    f"Layer {layer}: got {visual_hs.shape[0]} visual states, "
                    f"expected {expected_tokens} ({grid_t}x{h}x{w})"
                )
            result[layer] = visual_hs.reshape(grid_t, h, w, hidden_size)
        return result

    @torch.no_grad()
    def extract(
        self,
        frame_dir: str,
        num_frames: int,
        frame_ext: str = "png",
        start_idx: int = 0,
        gt_num_frames: int | None = None,
        use_query_frame_indices: bool = False,
        context_len: int = 76,
        query_idx_divisor: int = 4,
    ) -> Dict[int, torch.Tensor]:
        images = self.load_frames(
            frame_dir,
            num_frames,
            frame_ext,
            start_idx=start_idx,
            gt_num_frames=gt_num_frames,
            use_query_frame_indices=use_query_frame_indices,
            context_len=context_len,
            query_idx_divisor=query_idx_divisor,
        )
        logger.info("Loaded %d frames", len(images))

        hidden_states, input_ids, visual_mask, grid_sizes, merge_sizes = self.forward_with_hidden_states(images)
        logger.info(
            "VideoLLaMA3 visual grid: grid_sizes=%s merge_sizes=%s visual_tokens=%d seq_len=%d",
            grid_sizes.detach().cpu().tolist(),
            merge_sizes.detach().cpu().tolist(),
            int(visual_mask.sum().item()),
            int(input_ids.shape[-1]),
        )
        return self.extract_visual_hidden_states(hidden_states, visual_mask, grid_sizes, merge_sizes)


@lru_cache(maxsize=4)
def get_videollama3_extractor(
    model_path: str,
    select_layers: Tuple[int, ...] = (7, 14, 21, 28),
    question: str = "",
    device: str = "cuda:0",
    target_size: Optional[Tuple[int, int]] = (960, 540),
    attn_implementation: str = "flash_attention_2",
    use_token_compression: bool = False,
) -> VideoLLaMA3Extractor:
    return VideoLLaMA3Extractor(
        model_path=model_path,
        select_layers=list(select_layers),
        question=question,
        device=device,
        target_size=target_size,
        attn_implementation=attn_implementation,
        use_token_compression=use_token_compression,
    )
