#!/usr/bin/env python3
"""
Multi-GPU DL3DV processing script with persistent VGGT models.

Key features:
- Multiple Worker processes, each bound to a different GPU
- VGGT model loaded once per GPU, reused for all scenes on that GPU
- Shared task queue for automatic load balancing
- Deterministic random seed based on hash_dir for reproducibility
- Crash isolation and automatic worker restart
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
from typing import List, Optional, Tuple


class TaskStatus(Enum):
    READY = "ready"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class TaskResult:
    status: TaskStatus
    scene_id: str = ""
    gpu_id: int = -1
    message: str = ""
    elapsed: float = 0.0


# ================== Worker Process ==================


def worker_main(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    model_path: str,
    num_frames: int,
):
    """
    Worker process: binds to a specific GPU, loads VGGT model once,
    then processes scenes from the shared task queue.
    """
    # Bind to specific GPU before importing CUDA-related modules
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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

    # -------- Load VGGT model (only once per GPU) --------
    try:
        model = VGGT.from_pretrained(model_path, local_files_only=True)
        model = model.to("cuda").eval()
        result_queue.put(
            TaskResult(TaskStatus.READY, gpu_id=gpu_id, message="Model loaded")
        )
    except Exception as e:
        result_queue.put(
            TaskResult(
                TaskStatus.ERROR, gpu_id=gpu_id, message=f"Failed to load model: {e}"
            )
        )
        return

    # -------- Main loop: process tasks from shared queue --------
    max_retries = 2  # Number of retries for OOM errors

    while True:
        try:
            task = task_queue.get(timeout=5)
        except mp.queues.Empty:
            continue
        except KeyboardInterrupt:
            break

        if task is None:  # Shutdown signal
            break

        scene_dir, output_path, scene_id = task
        start_time = time.time()

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                process_scene(model, scene_dir, output_path, num_frames)
                elapsed = time.time() - start_time
                result_queue.put(
                    TaskResult(
                        TaskStatus.SUCCESS,
                        scene_id=scene_id,
                        gpu_id=gpu_id,
                        elapsed=elapsed,
                    )
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
                    last_error = e
                    continue
                else:
                    # Max retries reached or non-OOM RuntimeError
                    elapsed = time.time() - start_time
                    result_queue.put(
                        TaskResult(
                            TaskStatus.ERROR,
                            scene_id=scene_id,
                            gpu_id=gpu_id,
                            message=f"{type(e).__name__}: {e}",
                            elapsed=elapsed,
                        )
                    )
                    break

            except Exception as e:
                # Non-recoverable error
                elapsed = time.time() - start_time
                result_queue.put(
                    TaskResult(
                        TaskStatus.ERROR,
                        scene_id=scene_id,
                        gpu_id=gpu_id,
                        message=f"{type(e).__name__}: {e}",
                        elapsed=elapsed,
                    )
                )
                break


# ================== Multi-GPU Manager ==================


class MultiGPUManager:
    """
    Manages multiple worker processes across different GPUs.
    Handles task distribution, result collection, and crash recovery.
    """

    def __init__(
        self,
        gpu_ids: List[int],
        model_path: str,
        num_frames: int = 150,
        model_load_timeout: int = 300,
    ):
        self.gpu_ids = gpu_ids
        self.model_path = model_path
        self.num_frames = num_frames
        self.model_load_timeout = model_load_timeout

        self.task_queue: Optional[mp.Queue] = None
        self.result_queue: Optional[mp.Queue] = None
        self.workers: dict = {}  # gpu_id -> Process

    def start_all_workers(self) -> bool:
        """Start worker processes on all GPUs."""
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

        for gpu_id in self.gpu_ids:
            if not self._start_worker(gpu_id):
                return False

        # Wait for all models to load
        ready_count = 0
        deadline = time.time() + self.model_load_timeout

        while ready_count < len(self.gpu_ids) and time.time() < deadline:
            try:
                result = self.result_queue.get(timeout=5)
                if result.status == TaskStatus.READY:
                    ready_count += 1
                    logging.info(f"GPU {result.gpu_id} ready ({ready_count}/{len(self.gpu_ids)})")
                elif result.status == TaskStatus.ERROR:
                    logging.error(f"GPU {result.gpu_id} failed: {result.message}")
                    return False
            except mp.queues.Empty:
                # Check if any worker died
                for gpu_id, worker in list(self.workers.items()):
                    if not worker.is_alive():
                        logging.error(f"Worker on GPU {gpu_id} died during initialization")
                        return False

        if ready_count < len(self.gpu_ids):
            logging.error("Timeout waiting for workers to initialize")
            return False

        logging.info(f"All {len(self.gpu_ids)} workers ready!")
        return True

    def _start_worker(self, gpu_id: int) -> bool:
        """Start a single worker process on the specified GPU."""
        try:
            worker = mp.Process(
                target=worker_main,
                args=(
                    gpu_id,
                    self.task_queue,
                    self.result_queue,
                    self.model_path,
                    self.num_frames,
                ),
                daemon=True,
            )
            worker.start()
            self.workers[gpu_id] = worker
            logging.info(f"Started worker on GPU {gpu_id} (PID: {worker.pid})")
            return True
        except Exception as e:
            logging.error(f"Failed to start worker on GPU {gpu_id}: {e}")
            return False

    def stop_all_workers(self):
        """Gracefully stop all worker processes."""
        logging.info("Stopping all workers...")

        # Send shutdown signals
        for _ in self.gpu_ids:
            try:
                self.task_queue.put(None)
            except Exception:
                pass

        # Wait for workers to exit
        for gpu_id, worker in self.workers.items():
            worker.join(timeout=10)
            if worker.is_alive():
                logging.warning(f"Force terminating worker on GPU {gpu_id}")
                worker.terminate()
                worker.join(timeout=5)

        self.workers.clear()
        logging.info("All workers stopped.")

    def check_workers_health(self) -> List[int]:
        """Check which workers have crashed. Returns list of dead GPU IDs."""
        dead_gpus = []
        for gpu_id, worker in list(self.workers.items()):
            if not worker.is_alive():
                dead_gpus.append(gpu_id)
        return dead_gpus

    def submit_tasks(self, tasks: List[Tuple[str, str, str]]):
        """Submit all tasks to the shared queue."""
        for task in tasks:
            self.task_queue.put(task)
        logging.info(f"Submitted {len(tasks)} tasks to queue.")

    def collect_results(
        self, total_tasks: int, check_interval: float = 1.0
    ) -> Tuple[int, int, List[str]]:
        """
        Collect results from all workers.
        Returns (success_count, error_count, failed_scene_ids).
        """
        from tqdm import tqdm

        success_count = 0
        error_count = 0
        failed_scenes = []

        pbar = tqdm(
            total=total_tasks,
            desc="Processing",
            unit="scene",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        while pbar.n < total_tasks:
            try:
                result = self.result_queue.get(timeout=check_interval)

                if result.status == TaskStatus.READY:
                    continue

                pbar.update(1)

                if result.status == TaskStatus.SUCCESS:
                    success_count += 1
                    pbar.set_postfix_str(
                        f"GPU{result.gpu_id} ✓ {result.scene_id.split('/')[-1]} ({result.elapsed:.1f}s)"
                    )
                else:
                    error_count += 1
                    failed_scenes.append(result.scene_id)
                    pbar.set_postfix_str(
                        f"GPU{result.gpu_id} ✗ {result.scene_id.split('/')[-1]}"
                    )
                    logging.error(
                        f"GPU{result.gpu_id} failed: {result.scene_id} - {result.message}"
                    )

            except mp.queues.Empty:
                dead_gpus = self.check_workers_health()
                if dead_gpus:
                    logging.warning(f"Detected crashed workers on GPUs: {dead_gpus}")

        pbar.close()
        return success_count, error_count, failed_scenes


# ================== Utility Functions ==================


def parse_gpu_ids(gpu_arg: str) -> List[int]:
    """Parse GPU argument into a list of GPU IDs."""
    if gpu_arg.lower() == "all":
        import torch
        return list(range(torch.cuda.device_count()))
    else:
        return [int(x.strip()) for x in gpu_arg.split(",")]


def collect_scenes(input_base: str, subsets: List[str]) -> List[Tuple[str, str, str]]:
    """
    Collect all scenes to process.
    Returns list of (scene_dir, subset, hash_dir) tuples.
    """
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


# ================== Main ==================


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU DL3DV processing with persistent VGGT models."
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
        "--gpus",
        type=str,
        default="all",
        help="GPU IDs to use, comma-separated (e.g., '0,1,2,3') or 'all'. Default is all.",
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
    args = parser.parse_args()

    # Parse GPU IDs
    gpu_ids = parse_gpu_ids(args.gpus)

    # Define input and output directories
    input_base = os.path.join(args.root, "DL3DV-ALL-960P")
    output_base = os.path.join(args.root, "DL3DV-processed")
    os.makedirs(output_base, exist_ok=True)

    # Set up logging
    log_file = os.path.join(output_base, f"processing-{args.subset}-multigpu.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("=" * 70)
    logging.info("Starting Multi-GPU DL3DV Processing")
    logging.info("=" * 70)
    logging.info(f"GPUs: {gpu_ids}")
    logging.info(f"Subset: {args.subset}")
    logging.info(f"Input base: {input_base}")
    logging.info(f"Output base: {output_base}")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Num frames: {args.num_frames}")
    logging.info("=" * 70)

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
    logging.info(f"Found {len(scenes)} total scenes.")

    # Filter out already processed scenes and prepare tasks
    tasks = []
    skipped = 0
    for scene_dir, subset, hash_dir in scenes:
        subset_output_dir = os.path.join(output_base, subset)
        os.makedirs(subset_output_dir, exist_ok=True)
        output_file = os.path.join(subset_output_dir, f"{hash_dir}.sft")

        if should_skip_scene(output_file):
            skipped += 1
        else:
            scene_id = f"{subset}/{hash_dir}"
            tasks.append((scene_dir, output_file, scene_id))

    logging.info(f"Skipped {skipped} already processed scenes.")
    logging.info(f"Pending scenes to process: {len(tasks)}")

    if not tasks:
        logging.info("No scenes to process. Exiting.")
        return

    # Initialize multi-GPU manager
    manager = MultiGPUManager(
        gpu_ids=gpu_ids,
        model_path=args.model_path,
        num_frames=args.num_frames,
    )

    if not manager.start_all_workers():
        logging.error("Failed to start workers. Exiting.")
        return

    # Submit all tasks and collect results
    start_time = time.time()

    try:
        manager.submit_tasks(tasks)
        success_count, error_count, failed_scenes = manager.collect_results(len(tasks))
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        success_count, error_count, failed_scenes = 0, 0, []
    finally:
        manager.stop_all_workers()

    # Summary
    total_time = time.time() - start_time
    logging.info("=" * 70)
    logging.info("Processing Complete")
    logging.info("=" * 70)
    logging.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f}min)")
    logging.info(f"Success: {success_count}")
    logging.info(f"Errors: {error_count}")
    if failed_scenes:
        logging.info(f"Failed scenes: {failed_scenes[:10]}{'...' if len(failed_scenes) > 10 else ''}")
    logging.info(f"Throughput: {success_count / total_time * 60:.2f} scenes/min")
    logging.info(f"Log saved to: {log_file}")
    logging.info("=" * 70)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
