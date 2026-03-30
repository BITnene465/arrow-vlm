from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml


@dataclass
class ExperimentConfig:
    name: str = "qwen3vl-exp"
    output_dir: str = "outputs/qwen3vl-exp"
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
    # 1024 x 1024 pixel budget: 1,048,576
    max_pixels: int | None = 1048576


@dataclass
class TokenizerConfig:
    # Relative grounding coordinates are normalized to integer bins in [0, num_bins - 1].
    num_bins: int = 1000
    add_eos_token: bool = True


@dataclass
class PromptConfig:
    system_prompt: str = ""
    system_prompt_template: str | None = None
    user_prompt: str = (
        "Detect all arrows and output only a JSON array, with no markdown and no extra text. "
        "Normalize every coordinate to an integer in [0,999]. "
        'Each item must be either {"label":"single_arrow","bbox_2d":[x1,y1,x2,y2],"keypoints_2d":[[x,y],[x,y]]} '
        'or {"label":"double_arrow","bbox_2d":[x1,y1,x2,y2],"keypoints_2d":[[x,y],[x,y]]}. '
        "For single_arrow, keypoints must be ordered from tail to head. "
        "For double_arrow, keypoints[0] and keypoints[-1] are the two head tips. "
        "Each arrow must contain at least 2 points."
    )
    user_prompt_template: str | None = None


@dataclass
class TaskConfig:
    task_type: str = "joint_structure"
    domain_type: str = "arrow"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    train_path: str = "data/processed/train.jsonl"
    val_path: str = "data/processed/val.jsonl"
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
    lang_target_modules: list[str] = field(
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
    vis_target_modules: list[str] = field(
        default_factory=lambda: [
            "attn.qkv",
            "attn.proj",
            "mlp.linear_fc1",
            "mlp.linear_fc2",
        ]
    )
    proj_target_modules: list[str] = field(default_factory=list)


@dataclass
class FineTuneConfig:
    mode: str = "lora"


@dataclass
class TrainConfig:
    epochs: int = 3
    per_device_batch_size: int = 1
    grad_accum_steps: int = 8
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-4
    embed_learning_rate: float | None = None
    lm_head_learning_rate: float | None = None
    lora_learning_rate: float | None = None
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    bf16: bool = True
    eval_strategy: str = "epoch"
    eval_start_epoch: int = 1
    log_every_steps: int = 10
    eval_every_steps: int = 200
    save_every_steps: int = 200
    save_step_checkpoints: bool = False
    keep_last_n_checkpoints: int = 3
    find_unused_parameters: bool = False


@dataclass
class EvalConfig:
    per_device_batch_size: int = 1
    bucket_by_target_length: bool = True
    max_new_tokens: int = 8192
    num_beams: int = 1
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    use_cache: bool = True
    bbox_iou_threshold: float = 0.5
    strict_point_distance_px: float = 8.0
    monitor_metric: str = "val/end_to_end_score"
    monitor_mode: str = "max"


@dataclass
class LoggingConfig:
    use_wandb: bool = True
    project: str = "vlm_structgen_json"
    run_name: str | None = None
    progress_ncols: int = 88


@dataclass
class CheckpointConfig:
    init_from: str | None = None
    resume_from: str | None = None


@dataclass
class ExperimentRuntimeConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    data: DataConfig = field(default_factory=DataConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

T = TypeVar("T")


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
    with config_path.open("r", encoding="utf-8") as handle:
        yaml_payload = yaml.safe_load(handle) or {}
    config = _from_dict(ExperimentRuntimeConfig, yaml_payload)
    return apply_model_scale_tag(config)


def config_to_dict(config: ExperimentRuntimeConfig) -> dict[str, Any]:
    return asdict(config)


def apply_model_scale_tag(config: ExperimentRuntimeConfig) -> ExperimentRuntimeConfig:
    scale_tag = _extract_model_scale_tag(
        config.model.model_name_or_path,
        fallback=config.model.remote_model_name_or_path,
    )
    if scale_tag is None:
        return config

    if not _contains_standalone_tag(config.experiment.name, scale_tag):
        config.experiment.name = f"{config.experiment.name}-{scale_tag}"

    output_dir = Path(config.experiment.output_dir)
    if output_dir.name != scale_tag:
        config.experiment.output_dir = str(output_dir / scale_tag)

    if config.logging.run_name is not None and not _contains_standalone_tag(config.logging.run_name, scale_tag):
        config.logging.run_name = f"{config.logging.run_name}-{scale_tag}"

    return config


def apply_run_id(
    config: ExperimentRuntimeConfig,
    run_id: str,
    *,
    stage_name: str | None = None,
) -> ExperimentRuntimeConfig:
    normalized_run_id = _normalize_run_component(run_id, field_name="run_id")
    normalized_stage_name = None
    if stage_name is not None:
        normalized_stage_name = _normalize_run_component(stage_name, field_name="stage_name")

    base_experiment_name = config.experiment.name
    base_output_dir = Path(config.experiment.output_dir)
    base_run_name = config.logging.run_name or base_experiment_name

    if normalized_stage_name is None:
        suffix = normalized_run_id
        output_dir = base_output_dir / normalized_run_id
    else:
        suffix = f"{normalized_run_id}-{normalized_stage_name}"
        output_dir = base_output_dir / normalized_run_id / normalized_stage_name

    config.experiment.name = f"{base_experiment_name}-{suffix}"
    config.experiment.output_dir = str(output_dir)
    config.logging.run_name = f"{base_run_name}-{suffix}"
    return config


def _normalize_run_component(value: str, *, field_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    normalized = normalized.strip(".-")
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one alphanumeric character.")
    return normalized


def _extract_model_scale_tag(primary: str | None, *, fallback: str | None = None) -> str | None:
    for candidate in (primary, fallback):
        if not candidate:
            continue
        basename = Path(str(candidate)).name.lower()
        match = re.search(r"([0-9]+(?:\.[0-9]+)?b)", basename)
        if match is not None:
            return match.group(1)
    return None


def _contains_standalone_tag(text: str, tag: str) -> bool:
    return re.search(rf"(^|[-_/]){re.escape(tag)}($|[-_/])", text) is not None
