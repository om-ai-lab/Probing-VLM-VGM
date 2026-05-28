import glob
import os
import signal  # noqa: F401
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Cap BLAS / OpenMP thread pools BEFORE numpy / torch / mkl are imported.
# Without this, every training process spawns a thread pool sized to the
# full core count; running several processes at once oversubscribes the
# CPU and they stall fighting for threads. setdefault → an explicit shell
# value (e.g. `OMP_NUM_THREADS=8 python train.py ...`) still wins.
# MUST sit above `import torch` (line below) — these vars are read once at
# library load time, so setting them later has no effect.
for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "4")

# Silence noisy library warnings that flood multi-ckpt eval logs:
#   - pl_bolts checks `hasattr(numpy, "object")` at import time → FutureWarning
#   - pl_bolts' @under_review decorator registers its OWN filterwarnings("once",
#     UnderReviewWarning) per decorated class, which prepends to the filter
#     list and beats any "ignore" filter we register the normal way. Easiest
#     fix is to hook warnings.warn itself and drop UnderReviewWarning before
#     it even reaches the filter chain.
#   - torch warns once per process about non-writable numpy arrays from
#     vfm_pca_images / PIL → harmless.
# Must sit at the very top of train.py so the rules are in place before
# anything (lightning, pl_bolts, hydra) is loaded.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"pl_bolts.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pl_bolts.*")
warnings.filterwarnings("ignore", message=r".*non-writable.*", category=UserWarning)

_orig_warn = warnings.warn
def _quiet_warn(message, category=Warning, stacklevel=1, source=None):  # noqa: E305
    cat_name = getattr(category, "__name__", "")
    msg_str = str(message)
    if cat_name == "UnderReviewWarning" or "under review" in msg_str:
        return
    return _orig_warn(message, category, stacklevel + 1, source=source)
warnings.warn = _quiet_warn

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from probing_vlm_vgm.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def python_eval_resolver(code: str):
    return eval(code)


# Register the resolver with OmegaConf
# usage: ${python_code:1 + 1} in yaml
OmegaConf.register_new_resolver("python_eval", python_eval_resolver)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data.data_module._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data.data_module)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Loggers (e.g. wandb) only run in test mode — pure training skips them
    # to avoid wandb's NFS-bound state files contending with feature reads.
    if cfg.get("test"):
        log.info("Instantiating loggers...")
        logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    else:
        log.info("Skipping loggers (test=False — no wandb in train-only runs).")
        logger = []

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        # ckpt_path = trainer.checkpoint_callback.best_model_path
        # if ckpt_path == "":
        #     log.warning("Best ckpt not found! Using current weights for testing...")
        #     ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        # log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # If no ckpt_path and the autoresume flag is true, see if there's a latest ckpt in the output directory
    user_ckpt = cfg.get("ckpt_path", None)
    if (not user_ckpt or user_ckpt == "null") and cfg.get("autoresume", True):
        auto_ckpt = os.path.join(cfg.paths.output_dir, "checkpoints", "last.ckpt")
        if os.path.exists(auto_ckpt):
            log.info(f"Found and resuming from last checkpoint: {auto_ckpt}")
            # try to load the checkpoint -- if corrupted, also set ckpt_path to None
            try:
                torch.load(auto_ckpt)
            except Exception as e:
                log.warning(f"Checkpoint loading failed: {e}")
                auto_ckpt = None

            if auto_ckpt is not None:
                cfg.ckpt_path = auto_ckpt

                # If wandb is used for logging, fetch the run ID so it can be resumed
                if "wandb" in cfg.logger and cfg.get("train"):
                    wandb_dir = os.path.join(
                        cfg.paths.output_dir, "wandb", "latest-run"
                    )
                    wandb_files = glob.glob(os.path.join(wandb_dir, "run-*.wandb"))
                    if len(wandb_files) > 0:
                        assert len(wandb_files) == 1, "Multiple wandb files found!"
                        run_id = (
                            os.path.basename(wandb_files[0]).split("-")[1].split(".")[0]
                        )
                        cfg.logger.wandb.id = run_id
                        log.info(f"Resuming wandb run with ID: {run_id}")
                    else:
                        log.warning("No wandb run found to resume.")

        else:
            log.info("No checkpoint found; starting from scratch.")

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
    # test:
    # python -m probing_vlm_vgm.train experiment=dl3dv/vjepa task_name=dl3dv-eval job_name=vjepa train=false test=true ckpt_path=/path/to/checkpoints/last.ckpt
