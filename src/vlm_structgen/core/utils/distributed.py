from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def init_distributed() -> DistributedContext:
    if dist.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return DistributedContext(True, rank, world_size, local_rank, device)
    return DistributedContext(False, 0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def reset_model_runtime_state(model: torch.nn.Module) -> torch.nn.Module:
    raw_model = unwrap_model(model)
    if hasattr(raw_model, "rope_deltas"):
        raw_model.rope_deltas = None
    return raw_model


def seed_everything(seed: int, rank: int = 0) -> None:
    final_seed = seed + rank
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(final_seed)


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    if average:
        reduced /= dist.get_world_size()
    return reduced


def reduce_numeric_dict(values: dict[str, float], average: bool = False) -> dict[str, float]:
    if not values:
        return values
    if not dist.is_available() or not dist.is_initialized():
        return values
    keys = sorted(values.keys())
    tensor = torch.tensor([float(values[key]) for key in keys], device="cuda" if torch.cuda.is_available() else "cpu")
    tensor = reduce_tensor(tensor, average=average)
    return {key: tensor[index].item() for index, key in enumerate(keys)}


def get_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
