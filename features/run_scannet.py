#!/usr/bin/env python3
"""ScanNet feature extraction wrapper.

Unlike DL3DV which organizes scenes under subsets (1K, 2K, ...), ScanNet uses
split directories (train, val). We adapt the DL3DV driver to walk
``<scannet_processed_root>/<split>/<scene_id>/`` and dispatch each scene to
the requested VFM extractor, exactly like run_dl3dv.py does for hash dirs.

Output convention:
    <out_root>/<vfm_name>/<split>/<scene_id>/feature_*.sft

Paper-style examples:

WAN2.1-T2V-14B, layer 18, timestep 749:
CUDA_VISIBLE_DEVICES=0 python -m features.run_scannet \
        --vfm wan \
        --vfm-name wan-t2v-14b \
        --split both \
        --scannet-root data/ScanNet/ScanNet-processed \
        --out-root data/ScanNet/FEAT \
        --model-id ckpt/Wan2.1-T2V-14B-Diffusers \
        --prompt "" \
        --output-layers 18 \
        --t 749

Qwen3-VL-8B, layer 22:
CUDA_VISIBLE_DEVICES=0 python -m features.run_scannet \
        --vfm qwen3vl \
        --vfm-name qwen3-vl-8b \
        --split both \
        --scannet-root data/ScanNet/ScanNet-processed \
        --out-root data/ScanNet/FEAT \
        --model-path ckpt/Qwen3-VL-8B-Instruct \
        --model-type qwen3vl \
        --use-query-frame-indices \
        --context-len 76 \
        --query-idx-divisor 4 \
        --output-layers 22
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
            "wan", "opensora", "cogvideox", "aether",
            "internvl", "qwen3vl", "qwen25vl",
        ],
        help="Which extractor module to invoke.",
    )
    parser.add_argument(
        "--vfm-name",
        default=None,
        help="Override the directory name used under --out-root (defaults to --vfm). "
             "Use this to differentiate model-size variants of the same extractor, "
             "e.g. --vfm qwen3vl --vfm-name qwen3-vl-4b for the 4B checkpoint, so its "
             "features land at <out_root>/qwen3-vl-4b/<split>/<scene_id>/ instead of "
             "colliding with the 8B output at <out_root>/qwen3-vl-8b/...",
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
