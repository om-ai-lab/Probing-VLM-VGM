"""Shared LightningModule infrastructure for all probe experiments.

Hosts the pieces that every probe module needs identically:
    - configure_optimizers (with LinearWarmupCosineAnnealingLR step-scaling)
    - setup (compile flag + pretrained loading)
    - on_train_epoch_start / on_validation_epoch_start (set_epoch hook for
      our custom dataset/sampler)
    - _load_pretrained_weights (strict=False so old Exp-A ckpts can be
      loaded even after we add new heads — research_plan §3.1 decision)

Subclasses implement the task-specific bits:
    - forward
    - training_step / validation_step / test_step
    - any task-specific metric state

By contract, subclasses must:
    - pass `optimizer` and `scheduler` to __init__ as hydra partials
    - expose the underlying model on either self.probe or self.head
      (override `_probe_module()` if neither name fits)
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from lightning import LightningModule
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from probing_vlm_vgm.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class BaseProbeModule(LightningModule):
    """Shared training loop infrastructure for probe-style LightningModules."""

    def __init__(
        self,
        optimizer: Any,
        scheduler: Any,
        compile: bool = False,
        pretrained: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()
        # Subclasses call self.save_hyperparameters() themselves so they can
        # control which attributes are ignored (nn.Modules etc.). We don't
        # call it here.
        self._compile = bool(compile)
        self.pretrained = pretrained
        self.resume_from_checkpoint = resume_from_checkpoint

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #
    def _probe_module(self) -> nn.Module:
        """Return the underlying model. Default: probe → head → first child."""
        if hasattr(self, "probe"):
            return getattr(self, "probe")
        if hasattr(self, "head"):
            return getattr(self, "head")
        # Final fallback — return first registered nn.Module child.
        for _, m in self.named_children():
            return m
        raise RuntimeError("BaseProbeModule subclass has no nn.Module attribute")

    # ------------------------------------------------------------------ #
    # Optimizer + scheduler (shared, with warmup-step scaling)
    # ------------------------------------------------------------------ #
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}

        scheduler_config = self.hparams.scheduler

        # Special-case the pl_bolts warmup+cosine scheduler so warmup_epochs
        # and max_epochs scale to total *steps* across all training epochs.
        # This is what the original two modules did, just deduplicated.
        if scheduler_config.func is LinearWarmupCosineAnnealingLR:
            kw = dict(scheduler_config.keywords)
            total_steps = self.trainer.estimated_stepping_batches
            if kw["max_epochs"] <= 0:
                raise ValueError(
                    "scheduler.max_epochs must be > 0 for LinearWarmupCosineAnnealingLR. "
                    f"Got {kw['max_epochs']}; check trainer.max_epochs in the experiment config."
                )
            warmup = int(kw["warmup_epochs"] * total_steps / kw["max_epochs"])
            kw.update(warmup_epochs=warmup, max_epochs=total_steps)
            scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, **kw)
            interval = "step"
        else:
            scheduler = scheduler_config(optimizer=optimizer)
            interval = "epoch"

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "name": "train/lr",
                "scheduler": scheduler,
                "interval": interval,
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------ #
    # Compile + pretrained loading
    # ------------------------------------------------------------------ #
    def setup(self, stage: str) -> None:
        # IMPORTANT ordering: load_pretrained BEFORE compile.
        #
        # torch.compile wraps the module with a `_orig_mod.` prefix in the
        # state_dict. If we compile first, then load_state_dict(strict=False)
        # against the wrapped module, all of the ckpt's keys are reported as
        # "missing" (they lack the `_orig_mod.` prefix) and the load silently
        # populates nothing. This is a classic silent footgun. Loading first
        # → compile after means the original keys match and compile wraps the
        # already-loaded weights.
        if self.pretrained and not self.resume_from_checkpoint:
            self._load_pretrained_weights()

        if self._compile and stage == "fit":
            mod = self._probe_module()
            compiled = torch.compile(mod)
            # Re-attach onto whichever attribute the subclass uses
            if hasattr(self, "probe"):
                self.probe = compiled
            elif hasattr(self, "head"):
                self.head = compiled

    def _load_pretrained_weights(self) -> None:
        """Load weights from a previous checkpoint.

        Uses strict=False so that we can load an old Exp-A checkpoint into a
        new unified probe model that has additional heads (instance /
        semantic_tag) — the unseen-by-old-ckpt parameters keep their fresh
        init. This matches the research_plan.md §3.1 commitment to preserve
        existing Exp-A ckpts through the refactor.
        """
        if not self.pretrained:
            return
        if not os.path.isfile(self.pretrained):
            log.warning(f"pretrained path not found: {self.pretrained}")
            return

        log.info(f"Loading pretrained weights from {self.pretrained}")
        ckpt = torch.load(self.pretrained, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)

        mod = self._probe_module()
        # Strip lightning-module prefix ("probe." / "head.") so the inner
        # state_dict matches the underlying nn.Module's keys.
        for prefix in ("probe.", "head."):
            if any(k.startswith(prefix) for k in state.keys()):
                state = {
                    k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
                }
                break

        missing, unexpected = mod.load_state_dict(state, strict=False)
        if missing:
            log.info(
                f"_load_pretrained_weights: {len(missing)} missing keys "
                f"(expected for fresh heads, e.g. {missing[:3]})"
            )
        if unexpected:
            log.warning(
                f"_load_pretrained_weights: {len(unexpected)} unexpected keys "
                f"in ckpt: {unexpected[:3]}"
            )

    # ------------------------------------------------------------------ #
    # set_epoch hooks — same in both old modules
    # ------------------------------------------------------------------ #
    def on_train_epoch_start(self) -> None:
        loader = self.trainer.train_dataloader
        if loader is None:
            return
        if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(self.current_epoch)
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        loaders = self.trainer.val_dataloaders
        if loaders is None:
            return
        if not isinstance(loaders, (list, tuple)):
            loaders = [loaders]
        for loader in loaders:
            if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(0)
            if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(0)
