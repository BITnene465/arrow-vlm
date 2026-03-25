from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from vlm_det.config import ExperimentRuntimeConfig, _deep_merge, _from_dict
from vlm_det.utils.checkpoint import load_checkpoint_meta


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got: {value!r}")


def _set_nested_value(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    current = payload
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


INFERENCE_ENV_OVERRIDE_MAP: dict[str, tuple[str | list[str], callable]] = {
    "CHECKPOINT_PATH": ("checkpoint_path", str),
    "MODEL_NAME_OR_PATH": (
        ["runtime.model.model_name_or_path", "runtime.model.remote_model_name_or_path"],
        str,
    ),
    "MODEL_MIN_PIXELS": ("runtime.model.min_pixels", int),
    "MODEL_MAX_PIXELS": ("runtime.model.max_pixels", int),
    "SYSTEM_PROMPT": ("runtime.prompt.system_prompt", str),
    "USER_PROMPT": ("runtime.prompt.user_prompt", str),
    "INFER_MAX_NEW_TOKENS": ("runtime.eval.max_new_tokens", int),
    "INFER_NUM_BEAMS": ("runtime.eval.num_beams", int),
    "INFER_DO_SAMPLE": ("runtime.eval.do_sample", _parse_bool),
    "INFER_USE_CACHE": ("runtime.eval.use_cache", _parse_bool),
    "INFER_TEMPERATURE": ("runtime.eval.temperature", float),
    "INFER_TOP_P": ("runtime.eval.top_p", float),
    "INFER_TOP_K": ("runtime.eval.top_k", int),
    "INFER_DEVICE": ("device", str),
    "INFER_OUTPUT_DIR": ("output_dir", str),
    "APP_HOST": ("app.host", str),
    "APP_PORT": ("app.port", int),
    "APP_SHARE": ("app.share", _parse_bool),
}


@dataclass
class InferenceAppConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False


@dataclass
class InferenceSettings:
    runtime: ExperimentRuntimeConfig = field(default_factory=ExperimentRuntimeConfig)
    checkpoint_path: str = ""
    device: str | None = None
    output_dir: str | None = None
    app: InferenceAppConfig = field(default_factory=InferenceAppConfig)


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


def _apply_env_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    for env_name, (target_paths, caster) in INFERENCE_ENV_OVERRIDE_MAP.items():
        raw_value = os.getenv(env_name)
        if raw_value is None or raw_value == "":
            continue
        resolved_value = caster(raw_value)
        if isinstance(target_paths, str):
            target_paths = [target_paths]
        for target_path in target_paths:
            _set_nested_value(payload, target_path, resolved_value)
    return payload


def _extract_runtime_payload_from_checkpoint_meta(checkpoint_path: str | Path) -> dict[str, Any]:
    meta = load_checkpoint_meta(checkpoint_path)
    checkpoint_config = meta.get("config", {})
    runtime_payload: dict[str, Any] = {}
    for section_name in ("model", "tokenizer", "prompt", "finetune", "lora", "eval", "train"):
        section_value = checkpoint_config.get(section_name)
        if isinstance(section_value, dict):
            runtime_payload[section_name] = section_value
    return runtime_payload


def load_inference_settings(
    *,
    checkpoint_path: str | Path | None = None,
    env_file: str | Path | None = None,
) -> InferenceSettings:
    dotenv_path = _find_dotenv_path(env_file)
    if dotenv_path is not None:
        load_dotenv(dotenv_path=dotenv_path, override=False)

    env_payload = _apply_env_overrides({})
    resolved_checkpoint_path = checkpoint_path or env_payload.get("checkpoint_path")
    if not resolved_checkpoint_path:
        raise ValueError(
            "Inference checkpoint path is required. Set `CHECKPOINT_PATH` in `.env` "
            "or pass `checkpoint_path` explicitly."
        )

    runtime_payload = asdict(ExperimentRuntimeConfig())
    runtime_payload = _deep_merge(
        runtime_payload,
        _extract_runtime_payload_from_checkpoint_meta(resolved_checkpoint_path),
    )
    runtime_payload = _deep_merge(runtime_payload, env_payload.get("runtime", {}))
    runtime = _from_dict(ExperimentRuntimeConfig, runtime_payload)

    # Inference should never inherit training-time activation checkpointing.
    runtime.train.gradient_checkpointing = False

    app_payload = env_payload.get("app", {})
    app = InferenceAppConfig(
        host=app_payload.get("host", InferenceAppConfig.host),
        port=app_payload.get("port", InferenceAppConfig.port),
        share=app_payload.get("share", InferenceAppConfig.share),
    )

    return InferenceSettings(
        runtime=runtime,
        checkpoint_path=str(resolved_checkpoint_path),
        device=env_payload.get("device"),
        output_dir=env_payload.get("output_dir"),
        app=app,
    )
