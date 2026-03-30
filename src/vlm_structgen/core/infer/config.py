from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from vlm_structgen.core.config import ExperimentRuntimeConfig, _from_dict
from vlm_structgen.core.utils.checkpoint import load_checkpoint_meta


@dataclass
class InferModelConfig:
    min_pixels: int | None = None
    max_pixels: int | None = None


@dataclass
class InferPromptConfig:
    system_prompt: str | None = None
    system_prompt_template: str | None = None
    user_prompt: str | None = None
    user_prompt_template: str | None = None


@dataclass
class InferEvalConfig:
    max_new_tokens: int | None = None
    num_beams: int | None = None
    do_sample: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    use_cache: bool | None = None


@dataclass
class InferAppConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False


@dataclass
class OneStageInferenceConfig:
    model: InferModelConfig = field(default_factory=InferModelConfig)
    prompt: InferPromptConfig = field(default_factory=InferPromptConfig)
    eval: InferEvalConfig = field(default_factory=InferEvalConfig)
    app: InferAppConfig = field(default_factory=InferAppConfig)
    output_dir: str | None = None


@dataclass
class TwoStageStageInferenceConfig:
    model: InferModelConfig = field(default_factory=InferModelConfig)
    prompt: InferPromptConfig = field(default_factory=InferPromptConfig)
    eval: InferEvalConfig = field(default_factory=InferEvalConfig)
    batch_size: int = 1
    include_full_image: bool = True
    tile_size_ratios: list[float] = field(default_factory=list)
    min_tile_size: int = 512
    max_tile_size: int = 1280
    tile_stride_ratio: float = 0.75
    proposal_dedup_iou_threshold: float = 0.65


@dataclass
class TwoStageInferenceConfig:
    stage1: TwoStageStageInferenceConfig = field(default_factory=TwoStageStageInferenceConfig)
    stage2: TwoStageStageInferenceConfig = field(default_factory=TwoStageStageInferenceConfig)
    app: InferAppConfig = field(default_factory=InferAppConfig)
    output_dir: str | None = None
    padding_ratio: float = 0.5


@dataclass
class InferenceSettings:
    runtime: ExperimentRuntimeConfig
    checkpoint_path: str
    device: str | None = None
    output_dir: str | None = None
    app: InferAppConfig = field(default_factory=InferAppConfig)


def _find_dotenv_path(explicit_env_file: str | Path | None = None) -> Path | None:
    if explicit_env_file is not None:
        candidate = Path(explicit_env_file).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Env file not found: {candidate}")
        return candidate

    cwd = Path.cwd().resolve()
    for candidate_dir in [cwd, *cwd.parents]:
        dotenv_path = candidate_dir / ".env"
        if dotenv_path.exists():
            return dotenv_path
    return None


def _load_yaml_payload(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_one_stage_inference_config(path: str | Path | None) -> OneStageInferenceConfig:
    return _from_dict(OneStageInferenceConfig, _load_yaml_payload(path))


def load_two_stage_inference_config(path: str | Path | None) -> TwoStageInferenceConfig:
    return _from_dict(TwoStageInferenceConfig, _load_yaml_payload(path))


def _extract_runtime_payload_from_checkpoint_meta(checkpoint_path: str | Path) -> dict[str, Any]:
    meta = load_checkpoint_meta(checkpoint_path)
    checkpoint_config = meta.get("config", {})
    runtime_payload: dict[str, Any] = {}
    for section_name in ("model", "tokenizer", "task", "prompt", "finetune", "lora", "eval", "train"):
        section_value = checkpoint_config.get(section_name)
        if isinstance(section_value, dict):
            runtime_payload[section_name] = section_value
    return runtime_payload


def _build_runtime_from_checkpoint(checkpoint_path: str | Path) -> ExperimentRuntimeConfig:
    runtime = _from_dict(ExperimentRuntimeConfig, _extract_runtime_payload_from_checkpoint_meta(checkpoint_path))
    runtime.train.gradient_checkpointing = False
    return runtime


def _apply_model_overrides(runtime: ExperimentRuntimeConfig, model_cfg: InferModelConfig) -> None:
    if model_cfg.min_pixels is not None:
        runtime.model.min_pixels = model_cfg.min_pixels
    if model_cfg.max_pixels is not None:
        runtime.model.max_pixels = model_cfg.max_pixels


def _apply_prompt_overrides(runtime: ExperimentRuntimeConfig, prompt_cfg: InferPromptConfig) -> None:
    if prompt_cfg.system_prompt is not None:
        runtime.prompt.system_prompt = prompt_cfg.system_prompt
    if prompt_cfg.system_prompt_template is not None:
        runtime.prompt.system_prompt_template = prompt_cfg.system_prompt_template
    if prompt_cfg.user_prompt is not None:
        runtime.prompt.user_prompt = prompt_cfg.user_prompt
    if prompt_cfg.user_prompt_template is not None:
        runtime.prompt.user_prompt_template = prompt_cfg.user_prompt_template


def _apply_eval_overrides(runtime: ExperimentRuntimeConfig, eval_cfg: InferEvalConfig) -> None:
    if eval_cfg.max_new_tokens is not None:
        runtime.eval.max_new_tokens = eval_cfg.max_new_tokens
    if eval_cfg.num_beams is not None:
        runtime.eval.num_beams = eval_cfg.num_beams
    if eval_cfg.do_sample is not None:
        runtime.eval.do_sample = eval_cfg.do_sample
    if eval_cfg.temperature is not None:
        runtime.eval.temperature = eval_cfg.temperature
    if eval_cfg.top_p is not None:
        runtime.eval.top_p = eval_cfg.top_p
    if eval_cfg.top_k is not None:
        runtime.eval.top_k = eval_cfg.top_k
    if eval_cfg.use_cache is not None:
        runtime.eval.use_cache = eval_cfg.use_cache


def build_runtime_from_one_stage_infer_config(
    checkpoint_path: str | Path,
    infer_config: OneStageInferenceConfig,
) -> ExperimentRuntimeConfig:
    runtime = _build_runtime_from_checkpoint(checkpoint_path)
    _apply_model_overrides(runtime, infer_config.model)
    _apply_prompt_overrides(runtime, infer_config.prompt)
    _apply_eval_overrides(runtime, infer_config.eval)
    return runtime


def build_runtime_from_two_stage_infer_config(
    checkpoint_path: str | Path,
    infer_config: TwoStageStageInferenceConfig,
) -> ExperimentRuntimeConfig:
    runtime = _build_runtime_from_checkpoint(checkpoint_path)
    _apply_model_overrides(runtime, infer_config.model)
    _apply_prompt_overrides(runtime, infer_config.prompt)
    _apply_eval_overrides(runtime, infer_config.eval)
    return runtime


def load_inference_settings(
    *,
    checkpoint_path: str | Path | None,
    config_path: str | Path | None = None,
    infer_config: OneStageInferenceConfig | TwoStageStageInferenceConfig | None = None,
    env_file: str | Path | None = None,
) -> InferenceSettings:
    dotenv_path = _find_dotenv_path(env_file)
    if dotenv_path is not None:
        load_dotenv(dotenv_path=dotenv_path, override=False)

    resolved_checkpoint_path = checkpoint_path or os.getenv("CHECKPOINT_PATH")
    if not resolved_checkpoint_path:
        raise ValueError(
            "Inference checkpoint path is required. Pass --checkpoint or set CHECKPOINT_PATH in .env."
        )

    effective_infer_config = infer_config or load_one_stage_inference_config(config_path)
    runtime = build_runtime_from_one_stage_infer_config(resolved_checkpoint_path, effective_infer_config)
    output_dir = getattr(effective_infer_config, "output_dir", None)
    app = getattr(effective_infer_config, "app", InferAppConfig())
    return InferenceSettings(
        runtime=runtime,
        checkpoint_path=str(resolved_checkpoint_path),
        device=None,
        output_dir=output_dir,
        app=app,
    )
