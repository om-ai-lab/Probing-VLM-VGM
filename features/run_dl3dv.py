#!/usr/bin/env python3
"""
Quick‑start
-----------
InternVL3-8B (28 layers, with query frame indices for video_probe_dataset_ compatibility):
CUDA_VISIBLE_DEVICES=1 HF_HOME=/tmp/hf_cache python -m features.run_dl3dv \
        --vfm internvl \
        --subset all \
        --vfm-name internvl3-8b \
        --dl3dv-root data/DL3DV/DL3DV-ALL-960P \
        --out-root data/DL3DV/FEAT \
        --model-path ckpt/InternVL3-8B \
        --model-type internvl3 \
        --use-query-frame-indices \
        --context-len 76 \
        --query-idx-divisor 4 \
        --output-layers 15 18 21 24

InternVL3-8B (SenseNova SI version):
CUDA_VISIBLE_DEVICES=3  HF_HOME=/tmp/hf_cache  python -m features.run_dl3dv \
        --vfm internvl \
        --subset 3K \
        --vfm-name internvl3-8b \
        --dl3dv-root data/DL3DV/DL3DV-ALL-960P \
        --out-root data/DL3DV/FEAT_sensenova \
        --model-path ckpt/SenseNova-SI-1.3-InternVL3-8B \
        --model-type sensenova \
        --use-query-frame-indices \
        --context-len 76 \
        --query-idx-divisor 4 \
        --output-layers 15 18 21 24

InternVL3.5-4B (36 layers, Qwen3 LLM hidden size 2560):
CUDA_VISIBLE_DEVICES=1 HF_HOME=/tmp/hf_cache python -m features.run_dl3dv \
        --vfm internvl \
        --vfm-name internvl3.5-4b \
        --subset all \
        --dl3dv-root data/DL3DV/DL3DV-ALL-960P \
        --out-root data/DL3DV/FEAT \
        --model-path ckpt/InternVL3_5-4B \
        --model-type internvl35 \
        --use-query-frame-indices \
        --context-len 76 \
        --query-idx-divisor 4 \
        --output-layers 19 22 25 28 31

InternVL3.5-8B (36 layers, Qwen3 LLM hidden size 4096; requires all 4 shards):
CUDA_VISIBLE_DEVICES=1 HF_HOME=/tmp/hf_cache python -m features.run_dl3dv \
        --vfm internvl \
        --vfm-name internvl3.5-8b \
        --subset all \
        --dl3dv-root data/DL3DV/DL3DV-ALL-960P \
        --out-root data/DL3DV/FEAT \
        --model-path ckpt/InternVL3_5-8B \
        --model-type internvl35 \
        --use-query-frame-indices \
        --context-len 76 \
        --query-idx-divisor 4 \
        --output-layers 19 22 25 28 31

Qwen3-VL-8B (standard, 36 layers, with query frame indices):
CUDA_VISIBLE_DEVICES=4 python -m features.run_dl3dv \
        --vfm qwen3vl \
        --subset all \
        --vfm-name qwen3-vl-8b \
        --dl3dv-root data/DL3DV/DL3DV-ALL-960P \
        --out-root data/DL3DV/FEAT \
        --model-path ckpt/Qwen3-VL-8B-Instruct \
        --model-type qwen3vl \
        --use-query-frame-indices \
        --context-len 76 \
        --query-idx-divisor 4 \
        --output-layers 19 22 25 28 31

Qwen3-VL-4B (same extractor, different ckpt → use --vfm-name to keep outputs side-by-side):
    # Qwen3-VL-4B has 36 layers (same as 8B) but text_config.hidden_size=2560
    # vs the 8B's 4096 — the probe yaml's video_channels must be 2560 for 4B
    # (see configs/experiment/dl3dv/qwen3-vl-4b.yaml).
    # Output lands at <out_root>/qwen3-vl-4b/<subset>/<hash>/ (instead of qwen3-vl-8b/).
CUDA_VISIBLE_DEVICES=3 python -m features.run_dl3dv \
        --vfm qwen3vl \
        --vfm-name qwen3-vl-4b \
        --subset all \
        --dl3dv-root data/DL3DV/DL3DV-ALL-960P \
        --out-root data/DL3DV/FEAT \
        --model-path ckpt/Qwen3-VL-4B-Instruct \
        --model-type qwen3vl \
        --use-query-frame-indices \
        --context-len 76 \
        --query-idx-divisor 4 \
        --output-layers 19 22 25 28 31


Qwen3-VL-8B (SenseNova SI version):
CUDA_VISIBLE_DEVICES=1 python -m features.run_dl3dv \
        --vfm qwen3vl \
        --subset all \
        --vfm-name qwen3-vl-8b \
        --dl3dv-root data/DL3DV/DL3DV-ALL-960P \
        --out-root data/DL3DV/FEAT_sensenova \
        --model-path ckpt/SenseNova-SI-1.1-Qwen3-VL-8B \
        --model-type sensenova \
        --use-query-frame-indices \
        --context-len 76 \
        --query-idx-divisor 4 \
        --output-layers 19 22 25 28 31

Qwen2.5-VL-7B (28 layers, with query frame indices):
CUDA_VISIBLE_DEVICES=2 python -m features.run_dl3dv \
        --vfm qwen25vl \
        --subset all \
        --vfm-name qwen2.5-vl-7b \
        --dl3dv-root data/DL3DV/DL3DV-ALL-960P \
        --out-root data/DL3DV/FEAT \
        --model-path ckpt/Qwen2.5-VL-7B-Instruct \
        --model-type qwen25vl \
        --use-query-frame-indices \
        --context-len 76 \
        --query-idx-divisor 4 \
        --output-layers 12 15 18 21 24

Qwen2.5-VL-3B 36 layers
CUDA_VISIBLE_DEVICES=2 python -m features.run_dl3dv \
        --vfm qwen25vl \
        --vfm-name qwen2.5-vl-3b \
        --subset all \
        --dl3dv-root data/DL3DV/DL3DV-ALL-960P \
        --out-root data/DL3DV/FEAT \
        --model-path ckpt/Qwen2.5-VL-3B-Instruct \
        --model-type qwen25vl \
        --use-query-frame-indices \
        --context-len 76 \
        --query-idx-divisor 4 \
        --output-layers 19 22 25 28 31



LLaVA-NeXT-Video (7B, Vicuna 32 layers, with query frame indices):
CUDA_VISIBLE_DEVICES=4 python -m features.run_dl3dv \
        --vfm llavanextvideo \
        --subset all \
        --dl3dv-root data/DL3DV/DL3DV-ALL-960P \
        --out-root data/DL3DV/FEAT \
        --model-path ckpt/LLaVA-NeXT-Video-7B-hf \
        --use-query-frame-indices \
        --context-len 76 \
        --query-idx-divisor 4 \
        --output-layers 8 16 24 32

Cambrian-S (7B, Qwen2.5-7B LLM 28 layers, SigLIP2-so400m vision tower, fixed 8x8 padded spatial grid):
CUDA_VISIBLE_DEVICES=4 python -m features.run_dl3dv \
        --vfm cambrian_s \
        --subset all \
        --dl3dv-root data/DL3DV/DL3DV-ALL-960P \
        --out-root data/DL3DV/FEAT \
        --model-path ckpt/Cambrian-S-7B \
        --use-query-frame-indices \
        --context-len 76 \
        --query-idx-divisor 4 \
        --output-layers 15 18 21 24

WAN2.1-T2V-1.3B:
CUDA_VISIBLE_DEVICES=4 python -m features.run_dl3dv \
       --vfm wan \
       --subset all \
       --vfm-name wan-t2v-1.3b \
       --model-id ckpt/Wan2.1-T2V-1.3B-Diffusers \
       --dl3dv-root data/DL3DV/DL3DV-ALL-960P \
       --out-root  data/DL3DV/FEAT \
       --prompt "" \
       --output-layers 20 \
       --t 749

DINO:
python -m features.run_dl3dv \
       --vfm dino \
       --subset all \
       --batch-size 64 \
       --dl3dv-root data/DL3DV/DL3DV-ALL-960P \
       --out-root data/DL3DV/FEAT \
       --model-id ckpt/dinov2-large


VJEPA:
CUDA_VISIBLE_DEVICES=4 python -m features.run_dl3dv \
       --vfm vjepa \
       --subset all \
       --dl3dv-root data/DL3DV/DL3DV-ALL-960P \
       --out-root data/DL3DV/FEAT

Opensora:
CUDA_VISIBLE_DEVICES=7 python -m features.run_dl3dv \
       --vfm opensora \
       --subset all \
       --dl3dv-root data/DL3DV/DL3DV-ALL-960P \
       --out-root data/DL3DV/FEAT \
       --t 0.25 \
       --output-layers 20

CogVideoX-I2V-5B:
CUDA_VISIBLE_DEVICES=2 python -m features.run_dl3dv \
       --vfm cogvideox \
       --subset all \
       --vfm-name cogvideox-i2v-5b \
       --model-id ckpt/CogVideoX-5b-I2V \
       --dl3dv-root data/DL3DV/DL3DV-ALL-960P \
       --out-root data/DL3DV/FEAT \
       --t 749 \
       --output-layers 20

Aether:
CUDA_VISIBLE_DEVICES=0 python -m features.run_dl3dv \
       --vfm aether \
       --subset all \
       --dl3dv-root data/DL3DV/DL3DV-ALL-960P \
       --out-root data/DL3DV/FEAT \
       --t 749 \
       --output-layers 20 \
       --task videogen

Fast3R:
python -m features.run_dl3dv \
       --vfm f3r \
       --subset all \
       --model-id jedyang97/Fast3R_ViT_Large_512 \
       --dl3dv-root data/DL3DV/DL3DV-ALL-960P \
       --out-root data/DL3DV/FEAT \
       --output-layers 24
"""

import argparse
import importlib
import logging
import os
import sys
import time
from datetime import timedelta


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def list_hash_dirs(subset_dir):
    """Return the list of immediate sub‑dirs (scene hashes) under subset_dir."""
    return sorted(
        d for d in os.listdir(subset_dir) if os.path.isdir(os.path.join(subset_dir, d))
    )


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")  # Redirect to null device

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def call_extractor(extractor_main, argv):
    """Execute extractor_main(argv) and report (elapsed, success)."""
    start = time.time()
    try:
        with HiddenPrints():
            extractor_main(argv)  # argv is a normal sys.argv[1:]
        return time.time() - start, True
    except SystemExit as exc:  # argparse uses this
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
        description="DL3DV → feature extraction wrapper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset locations (all have reasonable defaults)
    parser.add_argument(
        "--dl3dv-root",
        default="data/DL3DV/DL3DV-10K",
        help="Root that contains DL3DV-10K/1K/…",
    )
    parser.add_argument(
        "--processed-root",
        default="data/DL3DV/DL3DV-processed",
        help="Root where extract_points.py wrote .sft",
    )
    parser.add_argument(
        "--out-root",
        default="data/DL3DV/FEAT",
        help="Root to store extracted features <out_root>/<vfm>/<subset>/<hash>",
    )
    parser.add_argument(
        "--image-ext",
        default="png",
        help="Image file extension to look for in the scene directories",
    )

    # Which part of the dataset
    parser.add_argument(
        "--subset", default="all", help="'1K', '2K', …, or 'all' (default)"
    )

    # Optinally start from specified index and end at specified index
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start processing from this index in the scene list (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End processing at this index in the scene list (default: None, process all)",
    )

    # Which video foundation model
    parser.add_argument(
        "--vfm",
        default="wan",
        choices=["wan", "dino", "vjepa", "opensora", "cogvideox", "aether", "f3r", "internvl", "qwen3vl", "qwen25vl", "llavanextvideo", "cambrian_s", "videollama3", "llavaov15", "mimo"],
        help="Which extractor module to invoke",
    )

    parser.add_argument(
        "--vfm-name",
        default=None,
        help="Override the directory name used under --out-root (defaults to --vfm). "
             "Use this to differentiate model-size variants of the same extractor, "
             "e.g. --vfm qwen3vl --vfm-name qwen3-vl-4b for the 4B checkpoint, so its "
             "features land at <out_root>/qwen3-vl-4b/<subset>/<hash>/ instead of "
             "colliding with the 8B output at <out_root>/qwen3-vl-8b/...",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show DEBUG‑level logs in the terminal (file log is always DEBUG)",
    )

    # Parse known vs unknown flags
    args, extractor_args = parser.parse_known_args()
    extractor_mod = importlib.import_module(f"features.{args.vfm}.extract_features")
    extractor_main = extractor_mod.main  # we’ll call this repeatedly

    out_vfm_name = args.vfm_name or args.vfm

    # ------------------------------------------------------------------ #
    # Logging                                                            #
    # ------------------------------------------------------------------ #
    log_dir = os.path.join(args.out_root, out_vfm_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"processing-{args.subset}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Wrapper started – VFM=%s  subset=%s", args.vfm, args.subset)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            h.setLevel(logging.DEBUG)  # file shows everything
        elif isinstance(h, logging.StreamHandler):
            h.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # ------------------------------------------------------------------ #
    # Gather all scenes first (for progress/ETA)                         #
    # ------------------------------------------------------------------ #
    if args.subset.lower() == "all":
        subsets = sorted(
            d
            for d in os.listdir(args.dl3dv_root)
            if os.path.isdir(os.path.join(args.dl3dv_root, d))
        )
    else:
        subsets = [args.subset]

    scenes = []  # list of (subset, hash_dir, images, data_sft)
    for subset in subsets:
        subset_dir = os.path.join(args.dl3dv_root, subset)
        if not os.path.isdir(subset_dir):
            continue
        for hash_dir in list_hash_dirs(subset_dir):
            # Check both possible image directory locations
            img_dir = os.path.join(subset_dir, hash_dir, "images_4")
            if not os.path.isdir(img_dir):
                img_dir = os.path.join(subset_dir, hash_dir, "colmap", "images_4")
            data_sft = os.path.join(args.processed_root, subset, f"{hash_dir}.sft")
            if os.path.isdir(img_dir) and os.path.isfile(data_sft):
                scenes.append((subset, hash_dir, img_dir, data_sft))

    if args.start > 0 or args.end is not None:
        start_idx = args.start
        end_idx = args.end if args.end is not None else len(scenes)
        logging.info(f"Starting from scene index {start_idx}: {scenes[start_idx]}")
        logging.info(
            f"Ending at scene index {end_idx} (exclusive): {scenes[end_idx] if end_idx <= len(scenes)-1 else 'end of list'}"
        )
        scenes = scenes[start_idx:end_idx]

    total_scenes = len(scenes)
    if total_scenes == 0:
        logging.error("No valid scenes found – exiting.")
        return

    extractor_module = f"features.{args.vfm}.extract_features"
    wrapper_start = time.time()
    done = 0

    # ------------------------------------------------------------------ #
    # Process each scene                                                 #
    # ------------------------------------------------------------------ #
    for subset, hash_dir, scene_images, data_sft in scenes:
        out_dir = os.path.join(args.out_root, out_vfm_name, subset, hash_dir)
        os.makedirs(out_dir, exist_ok=True)

        argv = [
            "--scene-dir",
            scene_images,
            "--data-sft",
            data_sft,
            "--out-dir",
            out_dir,
            "--image-ext",
            args.image_ext,
            *extractor_args,  # whatever extra flags you passed through
        ]
        logging.debug("Extractor argv: %s", " ".join(argv))
        elapsed, ok = call_extractor(extractor_main, argv)

        # elapsed, ok = call_extractor(cmd)
        done += 1
        eta = (time.time() - wrapper_start) / done * (total_scenes - done)

        status = "OK" if ok else "FAILED"
        logging.info(
            "[ %d / %d | %s | ETA %s ] Scene %s … %s",
            done,
            total_scenes,
            nice_td(elapsed),
            nice_td(eta),
            hash_dir[:8],
            status,
        )

    logging.info(
        "Finished %d scenes in %s (log saved to %s)",
        total_scenes,
        nice_td(time.time() - wrapper_start),
        log_path,
    )


if __name__ == "__main__":
    main()
