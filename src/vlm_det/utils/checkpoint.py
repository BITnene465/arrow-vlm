from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from vlm_det.utils.distributed import get_rng_state, set_rng_state, unwrap_model
from vlm_det.utils.io import ensure_dir, write_json


def save_training_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    tokenizer,
    processor,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    trainer_state: dict[str, Any],
    config_dict: dict[str, Any],
) -> None:
    checkpoint_dir = ensure_dir(checkpoint_dir)
    model_dir = ensure_dir(checkpoint_dir / "model")
    tokenizer_dir = ensure_dir(checkpoint_dir / "tokenizer")
    processor_dir = ensure_dir(checkpoint_dir / "processor")

    unwrapped = unwrap_model(model)
    torch.save(unwrapped.state_dict(), model_dir / "state_dict.pt")
    if hasattr(unwrapped, "config"):
        unwrapped.config.to_json_file(model_dir / "config.json")

    tokenizer.save_pretrained(tokenizer_dir)
    processor.save_pretrained(processor_dir)

    if optimizer is not None:
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

    torch.save(get_rng_state(), checkpoint_dir / "rng_state.pt")
    write_json(checkpoint_dir / "trainer_state.json", trainer_state)
    write_json(
        checkpoint_dir / "meta.json",
        {
            "experiment_name": config_dict["experiment"]["name"],
            "protocol_version": "arrow_v2_json",
            "config": config_dict,
            "trainer_state": trainer_state,
        },
    )


def load_training_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    tokenizer=None,
    processor=None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    strict: bool = True,
    resume_training_state: bool = True,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    state_dict = torch.load(checkpoint_dir / "model" / "state_dict.pt", map_location="cpu")
    unwrap_model(model).load_state_dict(state_dict, strict=strict)

    trainer_state = {}
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if trainer_state_path.exists():
        with trainer_state_path.open("r", encoding="utf-8") as handle:
            trainer_state = json.load(handle)

    if not resume_training_state:
        return trainer_state

    if optimizer is not None and (checkpoint_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu"))
    if scheduler is not None and (checkpoint_dir / "scheduler.pt").exists():
        scheduler.load_state_dict(torch.load(checkpoint_dir / "scheduler.pt", map_location="cpu"))
    if (checkpoint_dir / "rng_state.pt").exists():
        set_rng_state(torch.load(checkpoint_dir / "rng_state.pt", map_location="cpu"))
    return trainer_state
