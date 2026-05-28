#!/usr/bin/env python3
"""ScanNet → feature extraction wrapper (mirrors features/run_dl3dv.py).

Unlike DL3DV which organizes scenes under subsets (1K, 2K, ...), ScanNet uses
split directories (train, val). We adapt the DL3DV driver to walk
``<scannet_processed_root>/<split>/<scene_id>/`` and dispatch each scene to
the requested VFM extractor, exactly like run_dl3dv.py does for hash dirs.

Subset/path convention for output:
    <out_root>/<vfm>/<split>/<scene_id>/feature_*.sft

This matches the pattern ``<out_root>/<vfm>/<subset>/<hash>/…`` used by
run_dl3dv.py, with ``subset=split`` and ``hash=scene_id``.

Example (WAN, train split, first 10 scenes):
CUDA_VISIBLE_DEVICES=4 python -m features.run_scannet \\
    --vfm wan \\
    --split both \\
    --scannet-root data/ScanNet/ScanNet-processed \\
    --out-root data/ScanNet/FEAT \\
    --model-id ckpt/Wan2.1-T2V-1.3B-Diffusers \\
    --prompt "" --output-layers 20 --t 749 \\
    --end 10

Example (InternVL3, query-frame-indices, matches DL3DV convention):
CUDA_VISIBLE_DEVICES=1 HF_HOME=/tmp/hf_cache python -m features.run_scannet \\
    --vfm internvl --split train \\
    --scannet-root data/ScanNet/ScanNet-processed \\
    --out-root data/ScanNet/FEAT \\
    --model-path ckpt/InternVL3-8B --model-type internvl3 \\
    --use-query-frame-indices --context-len 76 --query-idx-divisor 4 \\
    --output-layers 12 15 18 21 24





# Wan
CUDA_VISIBLE_DEVICES=6 python -m features.run_scannet \
    --vfm wan \
    --split both \
    --scannet-root data/ScanNet/ScanNet-processed \
    --out-root data/ScanNet/FEAT \
    --model-id ckpt/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "" --output-layers 10 12 15 --t 749

CUDA_VISIBLE_DEVICES=5 python -m features.run_scannet \
    --vfm wan --vfm-name wan-14b \
    --split both \
    --scannet-root data/ScanNet/ScanNet-processed \
    --out-root data/ScanNet/FEAT \
    --model-id ckpt/Wan2.1-T2V-14B-Diffusers \
    --prompt "" --output-layers 10 12 15 18 20 --t 749

CUDA_VISIBLE_DEVICES=5 python -m features.run_scannet \
    --vfm wan --vfm-name wan-i2v-14b-480p \
    --split both \
    --scannet-root data/ScanNet/ScanNet-processed \
    --out-root data/ScanNet/FEAT \
    --model-id ckpt/Wan2.1-I2V-14B-480P-Diffusers \
    --prompt "" --output-layers 10 12 15 18 20 --t 749 \
    --i2v-condition-frame first

# OpenSora
CUDA_VISIBLE_DEVICES=5 python -m features.run_scannet \
  --vfm opensora --split both \
  --out-root data/ScanNet/FEAT \
  --t 0.25 --output-layers 10 12 15

# InternVL3
CUDA_VISIBLE_DEVICES=0 HF_HOME=/tmp/hf_cache python -m features.run_scannet \
  --vfm internvl  --split both \
  --model-path ckpt/InternVL3-8B --model-type internvl3 \
  --use-query-frame-indices --context-len 76 --query-idx-divisor 4 \
  --output-layers 12 15 18 21 24

# InternVL3-1B 24 layers
CUDA_VISIBLE_DEVICES=7 HF_HOME=/tmp/hf_cache python -m features.run_scannet \
  --vfm internvl --vfm-name internvl-1b --split both \
  --model-path ckpt/InternVL3-1B --model-type internvl3 \
  --use-query-frame-indices --context-len 76 --query-idx-divisor 4 \
  --output-layers 9 12 15 18 21

# InternVL3-2B 28 layers
CUDA_VISIBLE_DEVICES=4 HF_HOME=/tmp/hf_cache python -m features.run_scannet \
  --vfm internvl --vfm-name internvl-2b --split both \
  --model-path ckpt/InternVL3-2B --model-type internvl3 \
  --use-query-frame-indices --context-len 76 --query-idx-divisor 4 \
  --output-layers 12 15 18 21 24

# InternVL3 SenseNova (注意 --out-root 换个目录避免覆盖)
CUDA_VISIBLE_DEVICES=3 HF_HOME=/tmp/hf_cache python -m features.run_scannet \
  --vfm internvl --split both \
  --model-path ckpt/SenseNova-SI-1.3-InternVL3-8B --model-type sensenova \
  --use-query-frame-indices --context-len 76 --query-idx-divisor 4 \
  --output-layers 12 15 18 21 24 \
  --out-root data/ScanNet/FEAT_sensenova

# InternVL3.5-4B 36 layers
CUDA_VISIBLE_DEVICES=0 HF_HOME=/tmp/hf_cache python -m features.run_scannet \
  --vfm internvl --vfm-name internvl35-4b --split both \
  --model-path ckpt/InternVL3_5-4B --model-type internvl35 \
  --use-query-frame-indices --context-len 76 --query-idx-divisor 4 \
  --output-layers 19 22 25 28 31

# InternVL3.5-8B 36 layers (requires all 4 checkpoint shards)
CUDA_VISIBLE_DEVICES=0 HF_HOME=/tmp/hf_cache python -m features.run_scannet \
  --vfm internvl --vfm-name internvl35-8b --split both \
  --model-path ckpt/InternVL3_5-8B --model-type internvl35 \
  --use-query-frame-indices --context-len 76 --query-idx-divisor 4 \
  --output-layers 19 22 25 28 31

# Qwen3-VL
CUDA_VISIBLE_DEVICES=6 python -m features.run_scannet \
  --vfm qwen3vl --split both \
  --model-path ckpt/Qwen3-VL-8B-Instruct --model-type qwen3vl \
  --use-query-frame-indices --context-len 76 --query-idx-divisor 4 \
  --output-layers 13 16 19 22 25 28 \
  --out-root data/ScanNet/FEAT

# Qwen3-VL SenseNova
CUDA_VISIBLE_DEVICES=0 python -m features.run_scannet \
  --vfm qwen3vl --split both \
  --model-path ckpt/SenseNova-SI-1.1-Qwen3-VL-8B --model-type sensenova \
  --use-query-frame-indices --context-len 76 --query-idx-divisor 4 \
  --output-layers 13 16 19 22 25 28 \
  --out-root data/ScanNet/FEAT_sensenova

# Qwen3-VL 4B
CUDA_VISIBLE_DEVICES=3 python -m features.run_scannet \
    --vfm qwen3vl \
    --vfm-name qwen3vl-4b \
    --split both \
    --scannet-root data/ScanNet/ScanNet-processed \
    --model-path ckpt/Qwen3-VL-4B-Instruct \
    --model-type qwen3vl \
    --use-query-frame-indices --context-len 76 --query-idx-divisor 4 \
    --output-layers  13 16 19 22 25 28

# Qwen3-VL 2B
CUDA_VISIBLE_DEVICES=3 python -m features.run_scannet \
    --vfm qwen3vl \
    --vfm-name qwen3vl-2b \
    --split both \
    --scannet-root data/ScanNet/ScanNet-processed \
    --model-path ckpt/Qwen3-VL-2B-Instruct \
    --model-type qwen3vl \
    --use-query-frame-indices --context-len 76 --query-idx-divisor 4 \
    --output-layers  12 15 18 21 24

# Qwen2.5-VL (7B, 28 layers, with query frame indices):
CUDA_VISIBLE_DEVICES=7 python -m features.run_scannet \
        --vfm qwen25vl \
        --vfm-name qwen25vl-7b \
        --split both \
        --scannet-root data/ScanNet/ScanNet-processed \
        --model-path ckpt/Qwen2.5-VL-7B-Instruct \
        --model-type qwen25vl \
        --use-query-frame-indices \
        --context-len 76 \
        --query-idx-divisor 4 \
        --output-layers 12 15 18 21 24
        
# Qwen2.5-VL-3B 36layers
CUDA_VISIBLE_DEVICES=7 python -m features.run_scannet \
        --vfm qwen25vl \
        --split both \
        --vfm-name qwen25vl-3b \
        --scannet-root data/ScanNet/ScanNet-processed \
        --model-path ckpt/Qwen2.5-VL-3B-Instruct \
        --model-type qwen25vl \
        --use-query-frame-indices \
        --context-len 76 \
        --query-idx-divisor 4 \
        --output-layers 12 15 18 21 24

# LLaVA-OneVision-1.5-4B native video mode
CUDA_VISIBLE_DEVICES=3 python -m features.run_scannet \
        --vfm llavaov15 \
        --vfm-name llavaov15-4b \
        --split both \
        --scannet-root data/ScanNet/ScanNet-processed \
        --out-root data/ScanNet/FEAT \
        --model-path ckpt/LLaVA-OneVision-1.5-4B-Instruct \
        --use-query-frame-indices \
        --context-len 76 \
        --query-idx-divisor 4 \
        --target-size 448 448 \
        --output-layers 12 15 18 21 24


# LLaVA-OneVision-1.5-8B native video mode
CUDA_VISIBLE_DEVICES=3 python -m features.run_scannet \
        --vfm llavaov15 \
        --vfm-name llavaov15-8b \
        --split both \
        --scannet-root data/ScanNet/ScanNet-processed \
        --out-root data/ScanNet/FEAT \
        --model-path ckpt/LLaVA-OneVision-1.5-8B-Instruct \
        --use-query-frame-indices \
        --context-len 76 \
        --query-idx-divisor 4 \
        --target-size 448 448 \
        --output-layers 13 16 19 22 25 28

# MiMo-VL-7B native video mode
CUDA_VISIBLE_DEVICES=2 python -m features.run_scannet \
        --vfm mimo \
        --vfm-name mimo-vl-7b \
        --split both \
        --scannet-root data/ScanNet/ScanNet-processed \
        --out-root data/ScanNet/FEAT \
        --model-path ckpt/MiMo-VL-7B-SFT \
        --use-query-frame-indices \
        --context-len 76 \
        --query-idx-divisor 4 \
        --target-size 448 448 \
        --output-layers 12 15 18 21 24

# VJEPA (single-layer self-supervised video encoder, no --output-layers / --t)
CUDA_VISIBLE_DEVICES=4 python -m features.run_scannet \
    --vfm vjepa --split both \
    --scannet-root data/ScanNet/ScanNet-processed \
    --out-root data/ScanNet/FEAT


# CogVideoX-2B (T2V, DIFT-style; supports multi-layer extraction)
CUDA_VISIBLE_DEVICES=0 python -m features.run_scannet \
    --vfm cogvideox --vfm-name cogvideox-2b --split both \
    --scannet-root data/ScanNet/ScanNet-processed \
    --out-root data/ScanNet/FEAT \
    --model-id ckpt/CogVideoX-2b \
    --t 749 --output-layers 10 12 15 18 20


# CogVideoX-5B (T2V, DIFT-style; supports multi-layer extraction)
CUDA_VISIBLE_DEVICES=0 python -m features.run_scannet \
    --vfm cogvideox --vfm-name cogvideox-5b --split both \
    --scannet-root data/ScanNet/ScanNet-processed \
    --out-root data/ScanNet/FEAT \
    --model-id ckpt/CogVideoX-5b \
    --t 749 --output-layers 10 12 15 18 20


# CogVideoX-5B-I2V (DIFT-style; supports multi-layer extraction)
CUDA_VISIBLE_DEVICES=0 python -m features.run_scannet \
    --vfm cogvideox --vfm-name cogvideox --split both \
    --scannet-root data/ScanNet/ScanNet-processed \
    --out-root data/ScanNet/FEAT \
    --model-id ckpt/CogVideoX-5b-I2V \
    --t 749 --output-layers 10 12 15 18 20


# Aether (3D-finetuned CogVideoX; --task videogen mirrors the DL3DV setup)
CUDA_VISIBLE_DEVICES=0 python -m features.run_scannet \
    --vfm aether --split both \
    --scannet-root data/ScanNet/ScanNet-processed \
    --out-root data/ScanNet/FEAT \
    --t 749 --output-layers 10 12 15 18 20 --task videogen


"""
import argparse
import importlib
import logging
import os
import sys
import time
from datetime import timedelta


# --------------------------------------------------------------------------- #
# Helpers (copied from run_dl3dv.py)                                          #
# --------------------------------------------------------------------------- #
def list_scene_dirs(split_dir):
    """Return scene dirs that have both frames/ and metadata.sft."""
    out = []
    if not os.path.isdir(split_dir):
        return out
    for d in sorted(os.listdir(split_dir)):
        scene_dir = os.path.join(split_dir, d)
        if not os.path.isdir(scene_dir):
            continue
        if os.path.isdir(os.path.join(scene_dir, "frames")) and os.path.isfile(
            os.path.join(scene_dir, "metadata.sft")
        ):
            out.append(d)
    return out


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def call_extractor(extractor_main, argv):
    start = time.time()
    try:
        with HiddenPrints():
            extractor_main(argv)
        return time.time() - start, True
    except SystemExit as exc:
        return time.time() - start, exc.code == 0
    except Exception as exc:
        logging.exception("Extractor crashed: %s", exc)
        return time.time() - start, False


def nice_td(seconds):
    return str(timedelta(seconds=int(seconds)))


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="ScanNet → feature extraction wrapper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--scannet-root",
        default="data/ScanNet/ScanNet-processed",
        help="Root of processed ScanNet (contains train/ and val/).",
    )
    parser.add_argument(
        "--out-root",
        default="data/ScanNet/FEAT",
        help="Root to store extracted features <out_root>/<vfm>/<split>/<scene>/",
    )
    parser.add_argument(
        "--image-ext",
        default="jpg",
        help="Image file extension in each scene's frames/ directory.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "both"],
        help="ScanNet split (mirrors 'subset' in run_dl3dv.py).",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument(
        "--vfm",
        default="wan",
        choices=[
            "wan", "dino", "vjepa", "opensora", "cogvideox", "aether",
            "f3r", "internvl", "qwen3vl", "qwen25vl", "videollama3", "llavaov15", "mimo",
        ],
        help="Which extractor module to invoke.",
    )
    parser.add_argument(
        "--vfm-name",
        default=None,
        help="Override the directory name used under --out-root (defaults to --vfm). "
             "Use this to differentiate model-size variants of the same extractor, "
             "e.g. --vfm qwen3vl --vfm-name qwen3vl-4b for the 4B checkpoint, so its "
             "features land at <out_root>/qwen3vl-4b/<split>/<scene_id>/ instead of "
             "colliding with the 8B output at <out_root>/qwen3vl/...",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args, extractor_args = parser.parse_known_args()
    extractor_mod = importlib.import_module(f"features.{args.vfm}.extract_features")
    extractor_main = extractor_mod.main

    out_vfm_name = args.vfm_name or args.vfm

    # ------------------------------------------------------------------ #
    # Logging                                                            #
    # ------------------------------------------------------------------ #
    log_dir = os.path.join(args.out_root, out_vfm_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"processing-scannet-{args.split}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Wrapper started – VFM=%s split=%s", args.vfm, args.split)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    for h in root_logger.handlers:
        if isinstance(h, logging.FileHandler):
            h.setLevel(logging.DEBUG)
        elif isinstance(h, logging.StreamHandler):
            h.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # ------------------------------------------------------------------ #
    # Scene discovery                                                    #
    # ------------------------------------------------------------------ #
    splits = ["train", "val"] if args.split == "both" else [args.split]
    scenes = []  # list of (split, scene_id, frames_dir, data_sft)
    for split in splits:
        split_dir = os.path.join(args.scannet_root, split)
        for scene_id in list_scene_dirs(split_dir):
            scene_root = os.path.join(split_dir, scene_id)
            frames_dir = os.path.join(scene_root, "frames")
            data_sft = os.path.join(scene_root, "metadata.sft")
            scenes.append((split, scene_id, frames_dir, data_sft))

    if args.start > 0 or args.end is not None:
        start_idx = args.start
        end_idx = args.end if args.end is not None else len(scenes)
        logging.info(f"Slicing scenes [{start_idx}:{end_idx}] of {len(scenes)}")
        scenes = scenes[start_idx:end_idx]

    total_scenes = len(scenes)
    if total_scenes == 0:
        logging.error("No valid scenes found – exiting.")
        return

    wrapper_start = time.time()
    done = 0

    # ------------------------------------------------------------------ #
    # Process each scene                                                 #
    # ------------------------------------------------------------------ #
    for split, scene_id, frames_dir, data_sft in scenes:
        out_dir = os.path.join(args.out_root, out_vfm_name, split, scene_id)
        os.makedirs(out_dir, exist_ok=True)

        argv = [
            "--scene-dir", frames_dir,
            "--data-sft", data_sft,
            "--out-dir", out_dir,
            "--image-ext", args.image_ext,
            *extractor_args,
        ]
        logging.debug("Extractor argv: %s", " ".join(argv))
        elapsed, ok = call_extractor(extractor_main, argv)

        done += 1
        eta = (time.time() - wrapper_start) / done * (total_scenes - done)
        status = "OK" if ok else "FAILED"
        logging.info(
            "[ %d / %d | %s | ETA %s ] %s/%s … %s",
            done, total_scenes, nice_td(elapsed), nice_td(eta),
            split, scene_id, status,
        )

    logging.info(
        "Finished %d scenes in %s (log saved to %s)",
        total_scenes, nice_td(time.time() - wrapper_start), log_path,
    )


if __name__ == "__main__":
    main()
