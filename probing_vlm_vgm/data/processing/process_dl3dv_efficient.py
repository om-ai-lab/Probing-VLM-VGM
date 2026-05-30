#!/usr/bin/env python3
"""
Efficient DL3DV processing script with persistent VGGT model.

Key improvements over process_dl3dv.py:
- VGGT model is loaded only once in a worker process
- Worker process handles multiple scenes without reloading
- Crash isolation: if worker crashes, main process restarts it
- Automatic timeout handling for stuck scenes
"""

import argparse
import glob
import hashlib
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TaskStatus(Enum):
    READY = "ready"
    SUCCESS = "success"
    ERROR = "error"
    SKIP = "skip"


@dataclass
class TaskResult:
    status: TaskStatus
    message: str = ""
    elapsed: float = 0.0


# ================== Worker Process ==================


def worker_main(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    model_path: str,
    num_frames: int,
):
    """
    Worker process: loads the VGGT model once, then processes scenes in a loop.
    Communicates with the main process via queues.
    """
    import numpy as np
    import torch
    from safetensors.torch import save_file

    from probing_vlm_vgm.vggt.models.vggt import VGGT
    from probing_vlm_vgm.vggt.utils.geometry import unproject_depth_map_to_point_map
    from probing_vlm_vgm.vggt.utils.load_fn import load_and_preprocess_images
    from probing_vlm_vgm.vggt.utils.pose_enc import pose_encoding_to_extri_intri

    def get_image_file_list(scene_dir):
        pattern = os.path.join(scene_dir, "**/*.png")
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            raise FileNotFoundError(f"No PNG images found in {scene_dir}")
        return files

    def process_scene(model, scene_dir, output_path, num_frames):
        image_files = get_image_file_list(scene_dir)

        if num_frames > 0 and len(image_files) > num_frames:
            # Deterministic seed based on hash_dir for reproducibility
            hash_dir = os.path.basename(output_path).replace(".sft", "")
            seed = int(hashlib.md5(hash_dir.encode()).hexdigest()[:8], 16)
            np.random.seed(seed)

            start_idx = np.random.randint(0, len(image_files) - num_frames)
            image_files = image_files[start_idx : start_idx + num_frames]
        else:
            # Use all available frames if less than num_frames
            start_idx = 0

        images = load_and_preprocess_images(image_files).to("cuda")

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                predictions = model(images)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)

        depth_map = predictions["depth"]
        world_points = unproject_depth_map_to_point_map(
            depth_map, predictions["extrinsic"], predictions["intrinsic"]
        )
        predictions["world_points_from_depth"] = world_points

        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get(
            "depth_conf", np.ones_like(pred_world_points[..., 0])
        )
        pred_world_points_conf = np.expand_dims(pred_world_points_conf, axis=-1)

        output_dict = {
            "images": torch.from_numpy(predictions["images"] * 255).to(torch.uint8),
            "depthmaps": torch.from_numpy(predictions["depth"]).float(),
            "pointmaps": torch.from_numpy(pred_world_points).float(),
            "confmaps": torch.from_numpy(pred_world_points_conf).float(),
            "intrinsic": torch.from_numpy(predictions["intrinsic"]).float(),
            "extrinsic": torch.from_numpy(predictions["extrinsic"]).float(),
            "start_idx": torch.tensor(start_idx, dtype=torch.int32),
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_file(output_dict, output_path)
        torch.cuda.empty_cache()

    # -------- Load VGGT model (only once) --------
    try:
        model = VGGT.from_pretrained(model_path, local_files_only=True)
        model = model.to("cuda").eval()
        result_queue.put(TaskResult(TaskStatus.READY, "Model loaded successfully"))
    except Exception as e:
        result_queue.put(TaskResult(TaskStatus.ERROR, f"Failed to load model: {e}"))
        return

    # -------- Main loop: process tasks --------
    max_retries = 2  # Number of retries for OOM errors

    while True:
        try:
            task = task_queue.get()
        except KeyboardInterrupt:
            break

        if task is None:  # Shutdown signal
            break

        scene_dir, output_path = task
        start_time = time.time()

        for attempt in range(max_retries + 1):
            try:
                process_scene(model, scene_dir, output_path, num_frames)
                elapsed = time.time() - start_time
                result_queue.put(
                    TaskResult(TaskStatus.SUCCESS, output_path, elapsed)
                )
                break  # Success, exit retry loop

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                error_msg = str(e).lower()
                is_oom = (
                    isinstance(e, torch.cuda.OutOfMemoryError)
                    or "out of memory" in error_msg
                    or "cuda out of memory" in error_msg
                )

                if is_oom and attempt < max_retries:
                    # OOM: clear cache and retry
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    time.sleep(2)  # Brief pause before retry
                    continue
                else:
                    # Max retries reached or non-OOM RuntimeError
                    elapsed = time.time() - start_time
                    result_queue.put(
                        TaskResult(TaskStatus.ERROR, f"{type(e).__name__}: {e}", elapsed)
                    )
                    break

            except Exception as e:
                # Non-recoverable error
                elapsed = time.time() - start_time
                result_queue.put(
                    TaskResult(TaskStatus.ERROR, f"{type(e).__name__}: {e}", elapsed)
                )
                break


# ================== Worker Manager ==================


class VGGTWorkerManager:
    """
    Manages a worker process that holds the VGGT model.
    Handles starting, stopping, and restarting the worker on crashes.
    """

    def __init__(
        self,
        model_path: str,
        num_frames: int = 150,
        task_timeout: int = 600,
        model_load_timeout: int = 300,
        max_consecutive_failures: int = 5,
    ):
        self.model_path = model_path
        self.num_frames = num_frames
        self.task_timeout = task_timeout
        self.model_load_timeout = model_load_timeout
        self.max_consecutive_failures = max_consecutive_failures

        self.task_queue: Optional[mp.Queue] = None
        self.result_queue: Optional[mp.Queue] = None
        self.worker: Optional[mp.Process] = None
        self.consecutive_failures = 0

    def start_worker(self) -> bool:
        """Start the worker process and wait for model to load."""
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

        self.worker = mp.Process(
            target=worker_main,
            args=(
                self.task_queue,
                self.result_queue,
                self.model_path,
                self.num_frames,
            ),
            daemon=True,
        )
        self.worker.start()
        logging.info(f"Worker process started (PID: {self.worker.pid})")

        try:
            result = self.result_queue.get(timeout=self.model_load_timeout)
            if result.status == TaskStatus.READY:
                logging.info("VGGT model loaded successfully in worker.")
                return True
            else:
                logging.error(f"Worker failed to initialize: {result.message}")
                return False
        except mp.queues.Empty:
            logging.error("Timeout waiting for worker to load model.")
            self._terminate_worker()
            return False

    def stop_worker(self):
        """Gracefully stop the worker process."""
        if self.worker and self.worker.is_alive():
            logging.info("Sending shutdown signal to worker...")
            try:
                self.task_queue.put(None)
                self.worker.join(timeout=10)
            except Exception:
                pass

            if self.worker.is_alive():
                logging.warning("Worker did not exit gracefully, terminating...")
                self._terminate_worker()

        logging.info("Worker stopped.")

    def _terminate_worker(self):
        """Force terminate the worker process."""
        if self.worker and self.worker.is_alive():
            self.worker.terminate()
            self.worker.join(timeout=5)
            if self.worker.is_alive():
                os.kill(self.worker.pid, signal.SIGKILL)

    def _restart_worker(self) -> bool:
        """Restart the worker process after a crash."""
        logging.warning("Restarting worker process...")
        self._terminate_worker()
        time.sleep(1)  # Brief pause before restart
        return self.start_worker()

    def process_scene(self, scene_dir: str, output_path: str) -> TaskResult:
        """
        Submit a scene for processing and wait for result.
        Handles timeouts and worker crashes.
        """
        if not self.worker or not self.worker.is_alive():
            logging.error("Worker is not running, attempting restart...")
            if not self._restart_worker():
                return TaskResult(TaskStatus.ERROR, "Failed to restart worker")

        self.task_queue.put((scene_dir, output_path))

        try:
            result = self.result_queue.get(timeout=self.task_timeout)

            if result.status == TaskStatus.SUCCESS:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1

            return result

        except mp.queues.Empty:
            self.consecutive_failures += 1
            logging.error(f"Task timeout ({self.task_timeout}s): {scene_dir}")

            # Worker might be stuck, restart it
            if not self._restart_worker():
                return TaskResult(TaskStatus.ERROR, "Timeout and failed to restart worker")

            return TaskResult(TaskStatus.ERROR, f"Timeout after {self.task_timeout}s")

    def should_abort(self) -> bool:
        """Check if we should abort due to too many consecutive failures."""
        return self.consecutive_failures >= self.max_consecutive_failures


# ================== Main Logic ==================


def collect_scenes(input_base: str, subsets: list) -> list:
    """Collect all (scene_dir, subset, hash_dir) tuples to process."""
    scenes = []

    for subset in subsets:
        subset_input_dir = os.path.join(input_base, subset)
        if not os.path.isdir(subset_input_dir):
            logging.warning(f"Subset directory {subset_input_dir} does not exist. Skipping.")
            continue

        hash_dirs = [
            d
            for d in os.listdir(subset_input_dir)
            if os.path.isdir(os.path.join(subset_input_dir, d))
        ]

        for hash_dir in hash_dirs:
            scene_dir = os.path.join(subset_input_dir, hash_dir, "images_4")
            if not os.path.isdir(scene_dir):
                scene_dir_colmap = os.path.join(
                    subset_input_dir, hash_dir, "colmap", "images_4"
                )
                if not os.path.isdir(scene_dir_colmap):
                    logging.warning(
                        f"Scene directory {scene_dir} or {scene_dir_colmap} does not exist. Skipping."
                    )
                    continue
                scene_dir = scene_dir_colmap

            scenes.append((scene_dir, subset, hash_dir))

    return scenes


def should_skip_scene(output_file: str) -> bool:
    """Check if a scene has already been processed successfully."""
    if not os.path.isfile(output_file):
        return False

    try:
        from safetensors.torch import load_file
        load_file(output_file)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Efficient DL3DV processing with persistent VGGT model."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory for DL3DV (e.g., probing_vlm_vgm/DL3DV)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        help="Subset to process (e.g., '1K', '2K', ..., '11K', or 'all'). Default is all.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="ckpt/VGGT-1B",
        help="Path to VGGT model checkpoint.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=150,
        help="Number of frames to sample per scene.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for processing each scene.",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=5,
        help="Maximum consecutive failures before aborting.",
    )
    args = parser.parse_args()

    # Define input and output directories
    input_base = os.path.join(args.root, "DL3DV-10K-Sample")
    output_base = os.path.join(args.root, "DL3DV-processed")
    os.makedirs(output_base, exist_ok=True)

    # Set up logging
    log_file = os.path.join(output_base, f"processing-{args.subset}-efficient.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="a"),  # Append mode for resume
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("=" * 60)
    logging.info("Starting DL3DV efficient extraction.")
    logging.info(f"Subset: {args.subset}")
    logging.info(f"Input base: {input_base}")
    logging.info(f"Output base: {output_base}")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Num frames: {args.num_frames}")
    logging.info(f"Timeout: {args.timeout}s")
    logging.info("=" * 60)

    # Determine which subsets to process
    if args.subset.lower() == "all":
        subsets = [
            d
            for d in os.listdir(input_base)
            if os.path.isdir(os.path.join(input_base, d))
        ]
    else:
        subsets = [args.subset]

    # Collect all scenes
    scenes = collect_scenes(input_base, subsets)
    logging.info(f"Found {len(scenes)} total scenes to check.")

    # Filter out already processed scenes
    pending_scenes = []
    for scene_dir, subset, hash_dir in scenes:
        subset_output_dir = os.path.join(output_base, subset)
        output_file = os.path.join(subset_output_dir, f"{hash_dir}.sft")
        if should_skip_scene(output_file):
            logging.info(f"Skipping (already exists): {subset}/{hash_dir}")
        else:
            pending_scenes.append((scene_dir, subset, hash_dir, output_file))

    logging.info(f"Pending scenes to process: {len(pending_scenes)}")

    if not pending_scenes:
        logging.info("No scenes to process. Exiting.")
        return

    # Initialize worker manager
    manager = VGGTWorkerManager(
        model_path=args.model_path,
        num_frames=args.num_frames,
        task_timeout=args.timeout,
        max_consecutive_failures=args.max_failures,
    )

    if not manager.start_worker():
        logging.error("Failed to start worker. Exiting.")
        return

    # Process scenes with progress bar
    from tqdm import tqdm

    success_count = 0
    error_count = 0

    pbar = tqdm(
        pending_scenes,
        desc="Processing",
        unit="scene",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    try:
        for scene_dir, subset, hash_dir, output_file in pbar:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            result = manager.process_scene(scene_dir, output_file)

            if result.status == TaskStatus.SUCCESS:
                success_count += 1
                pbar.set_postfix_str(f"✓ {hash_dir} ({result.elapsed:.1f}s)")
            else:
                error_count += 1
                pbar.set_postfix_str(f"✗ {hash_dir}")
                logging.error(f"Failed: {subset}/{hash_dir} - {result.message}")

            if manager.should_abort():
                logging.error(
                    f"Too many consecutive failures ({args.max_failures}). Aborting."
                )
                break

    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
    finally:
        pbar.close()
        manager.stop_worker()

    # Summary
    logging.info("=" * 60)
    logging.info("Processing complete.")
    logging.info(f"Success: {success_count}")
    logging.info(f"Errors: {error_count}")
    logging.info(f"Log saved to: {log_file}")
    logging.info("=" * 60)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
