from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml
from dotenv import load_dotenv


@dataclass
class ExperimentConfig:
    name: str = "vlm_det_experiment"
    output_dir: str = "outputs/default"
    seed: int = 42


@dataclass
class ModelConfig:
    model_name_or_path: str = "models/Qwen3-VL-2B-Instruct"
    remote_model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    trust_remote_code: bool = True
    attn_implementation: str | None = "flash_attention_2"
    freeze_vision_tower: bool = True
    train_projector: bool = False
    vision_name_substrings: list[str] = field(default_factory=lambda: ["visual"])
    projector_name_substrings: list[str] = field(
        default_factory=lambda: ["merger", "projector", "multi_modal_projector"]
    )
    # Use bounded dynamic resolution instead of raw native resolution.
    # These values follow the Qwen-style pixel-budget approach and act as
    # sane defaults when configs do not override them explicitly.
    min_pixels: int | None = 200704
    max_pixels: int | None = 1003520


@dataclass
class TokenizerConfig:
    num_bins: int = 2048
    add_eos_token: bool = True
    reserve_future_bins: int = 0


@dataclass
class PromptConfig:
    system_prompt: str = (
        "<|arrow_task|>\n"
        "Detect all arrows in the figure and output each arrow using the predefined "
        "structured format. For each arrow, output one bounding box and an ordered "
        "keypoint sequence. The first keypoint is the start point, the last keypoint "
        "is the arrow head, and intermediate keypoints are turning points. Use "
        "<|visible|> for visible keypoints and <|occluded|> for occluded keypoints."
    )


@dataclass
class DataConfig:
    train_path: str = "data/processed/normalized/train.jsonl"
    val_path: str = "data/processed/normalized/val.jsonl"
    shuffle_instances_for_training: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class LoraConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class FineTuneConfig:
    mode: str = "lora"


@dataclass
class TrainConfig:
    epochs: int = 3
    per_device_batch_size: int = 1
    grad_accum_steps: int = 8
    learning_rate: float = 1e-4
    embed_learning_rate: float | None = None
    lm_head_learning_rate: float | None = None
    lora_learning_rate: float | None = None
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    bf16: bool = True
    log_every_steps: int = 10
    eval_every_steps: int = 200
    save_every_steps: int = 200
    save_step_checkpoints: bool = False
    keep_last_n_checkpoints: int = 3
    find_unused_parameters: bool = False


@dataclass
class EvalConfig:
    max_new_tokens: int = 8192
    num_beams: int = 1
    do_sample: bool = False
    use_cache: bool = True
    bbox_iou_threshold: float = 0.5
    strict_point_distance_px: float = 8.0
    monitor_metric: str = "val/end_to_end_score"
    monitor_mode: str = "max"


@dataclass
class LoggingConfig:
    use_wandb: bool = True
    project: str = "vlm_det"
    run_name: str | None = None
    progress_ncols: int = 110
    progress_refresh_rate: int = 1


@dataclass
class CheckpointConfig:
    resume_from: str | None = None


@dataclass
class ExperimentRuntimeConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    data: DataConfig = field(default_factory=DataConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


T = TypeVar("T")
ENV_OVERRIDE_MAP: dict[str, tuple[str, callable]] = {
    "VLM_DET_MODEL_NAME_OR_PATH": ("model.model_name_or_path", str),
    "VLM_DET_REMOTE_MODEL_NAME_OR_PATH": ("model.remote_model_name_or_path", str),
    "VLM_DET_MODEL_MIN_PIXELS": ("model.min_pixels", int),
    "VLM_DET_MODEL_MAX_PIXELS": ("model.max_pixels", int),
    "VLM_DET_OUTPUT_DIR": ("experiment.output_dir", str),
    "VLM_DET_WANDB_PROJECT": ("logging.project", str),
    "VLM_DET_USE_WANDB": ("logging.use_wandb", lambda value: value.lower() in {"1", "true", "yes", "on"}),
}


def _convert_value(value: Any, annotation: Any) -> Any:
    origin = get_origin(annotation)
    if is_dataclass(annotation):
        return _from_dict(annotation, value)
    if origin is list:
        item_type = get_args(annotation)[0]
        return [_convert_value(item, item_type) for item in value]
    if origin is dict:
        key_type, value_type = get_args(annotation)
        return {
            _convert_value(key, key_type): _convert_value(item, value_type)
            for key, item in value.items()
        }
    if origin is tuple:
        item_types = get_args(annotation)
        return tuple(_convert_value(item, t) for item, t in zip(value, item_types))
    if origin is None:
        return value
    if origin is not None:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _convert_value(value, args[0])
    return value


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    type_hints = get_type_hints(cls)
    kwargs = {}
    for field_info in fields(cls):
        if field_info.name not in data:
            continue
        annotation = type_hints.get(field_info.name, field_info.type)
        kwargs[field_info.name] = _convert_value(data[field_info.name], annotation)
    return cls(**kwargs)


def _find_dotenv_path(config_path: Path) -> Path | None:
    search_roots = [config_path.resolve(), Path.cwd().resolve()]
    seen: set[Path] = set()
    for root in search_roots:
        for candidate in [root, *root.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            dotenv_path = candidate / ".env"
            if dotenv_path.exists():
                return dotenv_path
    return None


def _set_nested_value(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    current = payload
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _apply_env_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    for env_name, (target_path, caster) in ENV_OVERRIDE_MAP.items():
        raw_value = os.getenv(env_name)
        if raw_value is None or raw_value == "":
            continue
        _set_nested_value(payload, target_path, caster(raw_value))
    return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> ExperimentRuntimeConfig:
    config_path = Path(path)
    dotenv_path = _find_dotenv_path(config_path)
    if dotenv_path is not None:
        load_dotenv(dotenv_path=dotenv_path, override=False)
    base_payload = _apply_env_overrides({})
    with config_path.open("r", encoding="utf-8") as handle:
        yaml_payload = yaml.safe_load(handle) or {}
    merged_payload = _deep_merge(base_payload, yaml_payload)
    return _from_dict(ExperimentRuntimeConfig, merged_payload)


def config_to_dict(config: ExperimentRuntimeConfig) -> dict[str, Any]:
    return asdict(config)
