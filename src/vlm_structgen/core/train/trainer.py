from __future__ import annotations

import math
import shutil
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel

from vlm_structgen.core.config import ExperimentRuntimeConfig, config_to_dict
from vlm_structgen.core.registry import get_adapter
from vlm_structgen.core.utils.checkpoint import (
    load_initial_model_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from vlm_structgen.core.utils.distributed import (
    is_main_process,
    reduce_numeric_dict,
    reset_model_runtime_state,
    unwrap_model,
)
from vlm_structgen.core.utils.io import ensure_dir
from vlm_structgen.core.utils.logging import ExperimentLogger, create_progress_bar


class Trainer:
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

    def _should_eval_on_step(self) -> bool:
        return self.config.train.eval_strategy == "steps"

    def _should_eval_at_epoch_end(self, epoch: int) -> bool:
        return epoch >= self.config.train.eval_start_epoch

    def fit(self) -> None:
        self.train()

    def train(self) -> None:
        for epoch in range(self.config.train.epochs):
            if hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)
            if not self._should_eval_at_epoch_end(epoch):
                continue
            metrics = self.evaluate(step=self.global_step, epoch=epoch)
            if metrics:
                self._maybe_update_best(metrics)
                self._log_metrics(metrics, self.global_step)
                self.save_checkpoint(tag="last", is_best=self._is_best(metrics))

    def train_one_epoch(self, epoch: int) -> None:
        self.model.train()
        self._accumulated_micro_steps = 0
        route_counter: Counter[tuple[str, str]] = Counter()
        progress = create_progress_bar(
            total=len(self.train_dataloader),
            desc=f"train e{epoch + 1}",
            ncols=self.config.logging.progress_ncols,
            leave=True,
        )
        self.optimizer.zero_grad(set_to_none=True)
        for step_index, batch in enumerate(self.train_dataloader, start=1):
            route_counter.update(self._collect_batch_routes(batch))
            step_metrics = self.train_one_step(batch)
            if progress is not None:
                progress.set_postfix(
                    {
                        "gs": self.global_step,
                        "loss": f"{step_metrics['train/loss']:.4f}",
                        "grad": f"{step_metrics['train/grad_norm']:.2f}",
                        "lr": f"{step_metrics['train/lr']:.1e}",
                    }
                )
                progress.update(1)
            self._log_metrics(step_metrics, self.global_step)
            if (
                self._should_eval_on_step()
                and self.global_step % self.config.train.eval_every_steps == 0
            ):
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
                            "gs": self.global_step,
                            "loss": "flush",
                            "grad": f"{flush_metrics['train/grad_norm']:.2f}",
                            "lr": f"{flush_metrics['train/lr']:.1e}",
                        }
                    )
                self._log_metrics(flush_metrics, self.global_step)
        if progress is not None:
            progress.close()
        self._log_epoch_route_distribution(epoch=epoch, route_counter=route_counter)

    def train_one_step(self, batch) -> dict[str, float]:
        model_inputs = self._move_batch_to_device(batch)
        reset_model_runtime_state(self.model)
        autocast_context = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.config.train.bf16 and self.device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            outputs = self.model(**model_inputs)
            adapter = self._resolve_batch_adapter(batch)
            loss = adapter.compute_loss(outputs, batch, tokenizer=self.tokenizer) / self.config.train.grad_accum_steps
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

    def _collect_batch_routes(self, batch: dict[str, Any]) -> Counter[tuple[str, str]]:
        meta = batch.get("meta", {})
        task_types = list(meta.get("task_type", []))
        domain_types = list(meta.get("domain_type", []))
        routes: Counter[tuple[str, str]] = Counter()
        for task_type, domain_type in zip(task_types, domain_types, strict=False):
            routes[(str(task_type), str(domain_type))] += 1
        return routes

    def _log_epoch_route_distribution(self, *, epoch: int, route_counter: Counter[tuple[str, str]]) -> None:
        if not route_counter:
            return

        reduction_payload = {
            f"__route__::{task_type}::{domain_type}": float(count)
            for (task_type, domain_type), count in route_counter.items()
        }
        reduced_payload = reduce_numeric_dict(reduction_payload, average=False)
        reduced_route_counter: Counter[tuple[str, str]] = Counter()
        for key, value in reduced_payload.items():
            _prefix, task_type, domain_type = key.split("::", 2)
            reduced_route_counter[(task_type, domain_type)] = int(round(value))

        total_samples = sum(reduced_route_counter.values())
        if total_samples <= 0:
            return

        summary_parts = [
            f"{task_type}/{domain_type}={count} ({count / total_samples:.2%})"
            for (task_type, domain_type), count in sorted(
                reduced_route_counter.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
        ]
        message = (
            f"epoch={epoch + 1} observed task/domain distribution "
            f"(total_samples={total_samples}): "
            + ", ".join(summary_parts)
        )
        if self.logger is not None:
            self.logger.info(message)

        metrics = {
            "train/routes/total_samples": float(total_samples),
        }
        for (task_type, domain_type), count in reduced_route_counter.items():
            metric_key = f"train/routes/{task_type}__{domain_type}"
            metrics[metric_key] = float(count)
            metrics[f"{metric_key}_ratio"] = float(count) / float(total_samples)
        self._log_metrics(metrics, self.global_step)

    def _resolve_batch_adapter(self, batch: dict[str, Any]):
        meta = batch.get("meta", {})
        task_types = list(meta.get("task_type", []))
        domain_types = list(meta.get("domain_type", []))
        if not task_types or not domain_types:
            raise ValueError("Batch metadata must include task_type and domain_type.")
        task_type = task_types[0]
        domain_type = domain_types[0]
        if any(value != task_type for value in task_types) or any(value != domain_type for value in domain_types):
            raise ValueError(
                "Mixed task/domain batches are not supported by the current trainer. "
                f"Observed task_types={sorted(set(task_types))}, domain_types={sorted(set(domain_types))}."
            )
        task_options = self._resolve_task_options_for_route(
            task_type=str(task_type),
            domain_type=str(domain_type),
        )
        return get_adapter(
            task_type=task_type,
            domain_type=domain_type,
            num_bins=self.config.tokenizer.num_bins,
            task_options_key=tuple(sorted(task_options.items())),
        )

    def _resolve_task_options_for_route(self, *, task_type: str, domain_type: str) -> dict[str, Any]:
        route_key = f"{task_type}/{domain_type}"
        route_options = self.config.task.route_options.get(route_key)
        if route_options is None:
            known_routes = sorted(self.config.task.route_options.keys())
            raise ValueError(
                "Missing task.route_options for current batch route. "
                f"required={route_key!r}, available={known_routes}."
            )
        return dict(route_options)

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

    def initialize_model_from_checkpoint(
        self,
        path: str,
        strict: bool = True,
    ) -> dict[str, Any]:
        return load_initial_model_checkpoint(
            checkpoint_dir=path,
            model=self.model,
            strict=strict,
        )

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        model_inputs = {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "labels": batch["labels"].to(self.device),
            "pixel_values": batch["pixel_values"].to(self.device),
            "use_cache": False,
        }
        if batch.get("image_grid_thw") is not None:
            model_inputs["image_grid_thw"] = batch["image_grid_thw"].to(self.device)
        if batch.get("mm_token_type_ids") is not None:
            model_inputs["mm_token_type_ids"] = batch["mm_token_type_ids"].to(self.device)
        return model_inputs

    def _log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.logger is None:
            return
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
