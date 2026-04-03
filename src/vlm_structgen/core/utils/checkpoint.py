from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from vlm_structgen.core.utils.distributed import get_rng_state, set_rng_state, unwrap_model
from vlm_structgen.core.utils.io import ensure_dir, write_json


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
    allow_partial_by_shape: bool = False,
    resume_training_state: bool = True,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    state_dict = torch.load(checkpoint_dir / "model" / "state_dict.pt", map_location="cpu")
    target_model = unwrap_model(model)
    if allow_partial_by_shape:
        target_state_dict = target_model.state_dict()
        filtered_state_dict: dict[str, torch.Tensor] = {}
        skipped_missing_or_mismatch = 0
        for key, value in state_dict.items():
            target_value = target_state_dict.get(key)
            if target_value is None or target_value.shape != value.shape:
                skipped_missing_or_mismatch += 1
                continue
            filtered_state_dict[key] = value
        target_model.load_state_dict(filtered_state_dict, strict=False)
        print(
            "[checkpoint] partial load enabled: "
            f"loaded={len(filtered_state_dict)}, skipped={skipped_missing_or_mismatch}",
            flush=True,
        )
    else:
        target_model.load_state_dict(state_dict, strict=strict)

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


def load_initial_model_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    state_dict = torch.load(checkpoint_dir / "model" / "state_dict.pt", map_location="cpu")
    meta = load_checkpoint_meta(checkpoint_dir)
    checkpoint_mode = (
        meta.get("config", {})
        .get("finetune", {})
        .get("mode")
    )

    target_model = unwrap_model(model)
    base_model_getter = getattr(target_model, "get_base_model", None)
    candidate_models: list[torch.nn.Module] = []

    if checkpoint_mode == "lora" and not callable(base_model_getter):
        raise ValueError(
            "Cannot initialize a non-LoRA model from a LoRA checkpoint with `init-from`. "
            "Use a matching LoRA config with `init-from`, or use `resume-from` on the original training setup."
        )

    if checkpoint_mode == "full" and callable(base_model_getter):
        # Common stage-2 case: initialize a fresh LoRA-wrapped model from a full-FT
        # checkpoint by loading weights into the underlying base model only.
        candidate_models.append(base_model_getter())
    else:
        candidate_models.append(target_model)
        if callable(base_model_getter):
            base_model = base_model_getter()
            if base_model is not target_model:
                candidate_models.append(base_model)

    load_error: RuntimeError | None = None
    for candidate_model in candidate_models:
        try:
            candidate_state_dict = candidate_model.state_dict()
            remapped_state_dict = _maybe_remap_full_checkpoint_for_lora_base(
                source_state_dict=state_dict,
                target_state_dict=candidate_state_dict,
                checkpoint_mode=checkpoint_mode,
            )
            if remapped_state_dict is not None:
                merged_state_dict = dict(candidate_state_dict)
                merged_state_dict.update(remapped_state_dict)
                candidate_model.load_state_dict(merged_state_dict, strict=True)
            else:
                candidate_model.load_state_dict(state_dict, strict=strict)
            load_error = None
            break
        except RuntimeError as exc:
            load_error = exc

    if load_error is not None:
        raise RuntimeError(
            f"Failed to initialize model weights from checkpoint {checkpoint_dir}."
        ) from load_error
    return meta


def _maybe_remap_full_checkpoint_for_lora_base(
    source_state_dict: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
    checkpoint_mode: str | None,
) -> dict[str, torch.Tensor] | None:
    if checkpoint_mode != "full":
        return None
    if not any(".base_layer." in key for key in target_state_dict):
        return None

    remapped_state_dict: dict[str, torch.Tensor] = {}
    for source_key, value in source_state_dict.items():
        if source_key in target_state_dict:
            remapped_state_dict[source_key] = value
            continue

        base_layer_key = source_key.replace(".weight", ".base_layer.weight")
        base_layer_key = base_layer_key.replace(".bias", ".base_layer.bias")
        if base_layer_key in target_state_dict:
            remapped_state_dict[base_layer_key] = value

    return remapped_state_dict


def load_checkpoint_meta(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    meta_path = checkpoint_dir / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
