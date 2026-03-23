from __future__ import annotations

import math
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel

from vlm_det.config import ExperimentRuntimeConfig, config_to_dict
from vlm_det.utils.checkpoint import load_training_checkpoint, save_training_checkpoint
from vlm_det.utils.distributed import is_main_process, reduce_numeric_dict, unwrap_model
from vlm_det.utils.io import ensure_dir
from vlm_det.utils.logging import ExperimentLogger, create_progress_bar


class ArrowTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        processor,
        train_dataloader,
        val_dataloader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        config: ExperimentRuntimeConfig,
        device: torch.device,
        rank: int,
        world_size: int,
        special_tokens: list[str],
        evaluator=None,
        logger: ExperimentLogger | None = None,
    ) -> None:
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.special_tokens = special_tokens
        self.evaluator = evaluator
        self.logger = logger
        self.output_dir = ensure_dir(config.experiment.output_dir)
        self.checkpoint_root = ensure_dir(self.output_dir / "checkpoints")
        model = model.to(device)
        if world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=config.train.find_unused_parameters,
            )
        self.model = model
        self.global_step = 0
        self.best_metric = -math.inf if config.eval.monitor_mode == "max" else math.inf
        self.best_checkpoint_path: str | None = None
        self._accumulated_micro_steps = 0

    def fit(self) -> None:
        self.train()

    def train(self) -> None:
        for epoch in range(self.config.train.epochs):
            if hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)
            metrics = self.evaluate(step=self.global_step, epoch=epoch)
            if metrics:
                self._maybe_update_best(metrics)
                self._log_metrics(metrics, self.global_step)
                self.save_checkpoint(tag="last", is_best=self._is_best(metrics))

    def train_one_epoch(self, epoch: int) -> None:
        self.model.train()
        self._accumulated_micro_steps = 0
        progress = create_progress_bar(
            total=len(self.train_dataloader),
            desc=f"epoch {epoch + 1}/{self.config.train.epochs}",
            ncols=self.config.logging.progress_ncols,
        )
        self.optimizer.zero_grad(set_to_none=True)
        for step_index, batch in enumerate(self.train_dataloader, start=1):
            step_metrics = self.train_one_step(batch)
            if progress is not None:
                progress.set_postfix(
                    {
                        "loss": f"{step_metrics['train/loss']:.4f}",
                        "lr": f"{step_metrics['train/lr']:.2e}",
                    }
                )
                progress.update(1)
            if self.global_step % self.config.train.log_every_steps == 0:
                self._log_metrics(step_metrics, self.global_step)
            if self.global_step % self.config.train.eval_every_steps == 0:
                metrics = self.evaluate(step=self.global_step, epoch=epoch)
                if metrics:
                    self._maybe_update_best(metrics)
                    self._log_metrics(metrics, self.global_step)
                    self.save_checkpoint(tag="last", is_best=self._is_best(metrics))
            elif (
                self.config.train.save_step_checkpoints
                and self.global_step % self.config.train.save_every_steps == 0
                and self.global_step > 0
            ):
                self.save_checkpoint(tag=f"step_{self.global_step}", is_best=False)
        if self._accumulated_micro_steps > 0:
            flush_metrics = self._optimizer_step()
            if flush_metrics:
                if progress is not None:
                    progress.set_postfix(
                        {
                            "loss": "flush",
                            "lr": f"{flush_metrics['train/lr']:.2e}",
                        }
                    )
                if self.global_step % self.config.train.log_every_steps == 0:
                    self._log_metrics(flush_metrics, self.global_step)
        if progress is not None:
            progress.close()

    def train_one_step(self, batch) -> dict[str, float]:
        model_inputs = self._move_batch_to_device(batch)
        autocast_context = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.config.train.bf16 and self.device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            outputs = self.model(**model_inputs)
            loss = outputs.loss / self.config.train.grad_accum_steps
        loss.backward()

        self._accumulated_micro_steps += 1
        self.global_step += 1

        reduced = {
            "train/loss": float(loss.detach().item() * self.config.train.grad_accum_steps),
            "train/grad_norm": 0.0,
            "train/lr": float(self.optimizer.param_groups[0]["lr"]),
        }
        if self._accumulated_micro_steps >= self.config.train.grad_accum_steps:
            step_metrics = self._optimizer_step()
            reduced.update(step_metrics)

        reduced = reduce_numeric_dict(reduced, average=True)
        return reduced

    def _optimizer_step(self) -> dict[str, float]:
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.train.max_grad_norm,
            ).item()
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._accumulated_micro_steps = 0
        return {
            "train/grad_norm": grad_norm,
            "train/lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    def evaluate(self, step: int | None = None, epoch: int | None = None) -> dict[str, float]:
        if self.evaluator is None or self.val_dataloader is None:
            return {}
        metrics = self.evaluator.evaluate_model(self.model, self.val_dataloader)
        if epoch is not None:
            metrics["val/epoch"] = float(epoch)
        if step is not None:
            metrics["val/step"] = float(step)
        return metrics

    def save_checkpoint(self, tag: str | None = None, is_best: bool = False) -> None:
        if not is_main_process():
            return
        tag = tag or f"step_{self.global_step}"
        target_dir = self.checkpoint_root / tag
        trainer_state = {
            "epoch": self._current_epoch_float(),
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_checkpoint_path": self.best_checkpoint_path,
        }
        save_training_checkpoint(
            checkpoint_dir=target_dir,
            model=self.model,
            tokenizer=self.tokenizer,
            processor=self.processor,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            trainer_state=trainer_state,
            config_dict=config_to_dict(self.config),
            special_tokens=self.special_tokens,
        )
        self._refresh_alias("last", target_dir)
        if is_best:
            self._refresh_alias("best", target_dir)
            self.best_checkpoint_path = str(target_dir)
        self._cleanup_old_checkpoints()

    def load_checkpoint(
        self,
        path: str,
        strict: bool = True,
        resume_training_state: bool = True,
    ) -> None:
        trainer_state = load_training_checkpoint(
            checkpoint_dir=path,
            model=self.model,
            tokenizer=self.tokenizer,
            processor=self.processor,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            strict=strict,
            resume_training_state=resume_training_state,
        )
        self.global_step = int(trainer_state.get("global_step", 0))
        self.best_metric = float(trainer_state.get("best_metric", self.best_metric))
        self.best_checkpoint_path = trainer_state.get("best_checkpoint_path")

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        model_inputs = {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "labels": batch["labels"].to(self.device),
            "pixel_values": batch["pixel_values"].to(self.device),
        }
        if batch.get("image_grid_thw") is not None:
            model_inputs["image_grid_thw"] = batch["image_grid_thw"].to(self.device)
        return model_inputs

    def _log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.logger is None:
            return
        display = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        self.logger.info(f"step={step} | {display}")
        self.logger.log_metrics(metrics, step=step)

    def _is_best(self, metrics: dict[str, float]) -> bool:
        monitor = metrics.get(self.config.eval.monitor_metric)
        if monitor is None:
            return False
        if self.config.eval.monitor_mode == "max":
            return monitor >= self.best_metric
        return monitor <= self.best_metric

    def _maybe_update_best(self, metrics: dict[str, float]) -> None:
        monitor = metrics.get(self.config.eval.monitor_metric)
        if monitor is None:
            return
        if self.config.eval.monitor_mode == "max":
            self.best_metric = max(self.best_metric, monitor)
        else:
            self.best_metric = min(self.best_metric, monitor)

    def _refresh_alias(self, alias: str, target_dir: Path) -> None:
        alias_dir = self.checkpoint_root / alias
        if alias_dir.exists() and alias_dir.resolve() == target_dir.resolve():
            return
        if alias_dir.exists():
            shutil.rmtree(alias_dir)
        shutil.copytree(target_dir, alias_dir)

    def _cleanup_old_checkpoints(self) -> None:
        if not is_main_process():
            return
        checkpoints = sorted(
            [
                path
                for path in self.checkpoint_root.glob("step_*")
                if path.is_dir()
            ],
            key=lambda path: int(path.name.split("_")[-1]),
        )
        keep = self.config.train.keep_last_n_checkpoints
        while len(checkpoints) > keep:
            shutil.rmtree(checkpoints.pop(0))

    def _current_epoch_float(self) -> float:
        return float(self.global_step) / max(len(self.train_dataloader), 1)
