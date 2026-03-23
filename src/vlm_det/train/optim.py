from __future__ import annotations

from typing import Any

import torch
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from vlm_det.config import ExperimentRuntimeConfig


def _is_lora_param(name: str) -> bool:
    return "lora_" in name or ".lora_" in name


def build_optimizer(model: torch.nn.Module, config: ExperimentRuntimeConfig) -> torch.optim.Optimizer:
    embed_lr = config.train.embed_learning_rate or config.train.learning_rate
    lm_head_lr = config.train.lm_head_learning_rate or config.train.learning_rate
    lora_lr = config.train.lora_learning_rate or config.train.learning_rate

    groups = {
        "embed_tokens": {"params": [], "lr": embed_lr, "weight_decay": config.train.weight_decay},
        "lm_head": {"params": [], "lr": lm_head_lr, "weight_decay": config.train.weight_decay},
        "lora_params": {"params": [], "lr": lora_lr, "weight_decay": 0.0},
        "other": {"params": [], "lr": config.train.learning_rate, "weight_decay": config.train.weight_decay},
    }

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "embed_tokens" in name:
            groups["embed_tokens"]["params"].append(parameter)
        elif "lm_head" in name:
            groups["lm_head"]["params"].append(parameter)
        elif _is_lora_param(name):
            groups["lora_params"]["params"].append(parameter)
        else:
            groups["other"]["params"].append(parameter)

    param_groups = []
    for group_name, payload in groups.items():
        if not payload["params"]:
            continue
        param_groups.append(
            {
                "name": group_name,
                "params": payload["params"],
                "lr": payload["lr"],
                "weight_decay": payload["weight_decay"],
            }
        )
    return torch.optim.AdamW(param_groups)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ExperimentRuntimeConfig,
    total_training_steps: int,
):
    warmup_steps = int(total_training_steps * config.train.warmup_ratio)
    if config.train.scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )


def current_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        group_name = group.get("name", f"group_{index}")
        result[f"lr/{group_name}"] = float(group["lr"])
    return result
