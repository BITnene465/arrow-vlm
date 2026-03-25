from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor, AutoTokenizer

from vlm_det.config import ExperimentRuntimeConfig

@dataclass
class BuildArtifacts:
    model: torch.nn.Module
    tokenizer: Any
    processor: Any
    trainable_summary: dict[str, int]


def _resolve_model_class():
    transformers_module = importlib.import_module("transformers")
    class_name = "Qwen3VLForConditionalGeneration"
    if not hasattr(transformers_module, class_name):
        raise ImportError(
            "Current transformers installation does not expose Qwen3VLForConditionalGeneration. "
            "Please install the Qwen3-VL compatible transformers version."
        )
    return getattr(transformers_module, class_name)


def _freeze_all_parameters(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def _enable_all_parameters(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def _set_requires_grad_by_name(
    model: torch.nn.Module,
    substrings: list[str],
    requires_grad: bool,
) -> None:
    for name, parameter in model.named_parameters():
        if any(substring in name for substring in substrings):
            parameter.requires_grad = requires_grad


def _trainable_summary(model: torch.nn.Module) -> dict[str, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    return {"trainable_params": trainable, "total_params": total}


def _resolve_model_source(config: ExperimentRuntimeConfig) -> str:
    local_path = Path(config.model.model_name_or_path)
    if local_path.exists():
        return str(local_path)
    return config.model.remote_model_name_or_path


def _is_local_model_source(model_source: str) -> bool:
    return Path(model_source).exists()


def _resolve_attn_implementation(config: ExperimentRuntimeConfig) -> str | None:
    requested = config.model.attn_implementation
    if not requested:
        return None
    if requested != "flash_attention_2":
        return requested
    if not torch.cuda.is_available():
        print("flash_attention_2 requested but CUDA is unavailable; falling back to sdpa.")
        return "sdpa"
    if importlib.util.find_spec("flash_attn") is None:
        print("flash_attention_2 requested but flash_attn is not installed; falling back to sdpa.")
        return "sdpa"
    return requested


def _sanitize_generation_config(model: torch.nn.Module, config: ExperimentRuntimeConfig) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    generation_config.do_sample = config.eval.do_sample
    generation_config.num_beams = config.eval.num_beams
    generation_config.use_cache = config.eval.use_cache
    if not config.eval.do_sample:
        # Greedy / beam search does not consume sampling-only knobs. Clearing them
        # avoids repeated transformers warnings like temperature/top_p/top_k ignored.
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None


def _maybe_enable_gradient_checkpointing(model: torch.nn.Module, config: ExperimentRuntimeConfig) -> None:
    if not config.train.gradient_checkpointing:
        return

    enable_input_require_grads = getattr(model, "enable_input_require_grads", None)
    if callable(enable_input_require_grads):
        enable_input_require_grads()

    gradient_checkpointing_enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(gradient_checkpointing_enable):
        raise ValueError(
            "gradient_checkpointing was requested, but the current model does not expose "
            "`gradient_checkpointing_enable()`."
        )

    try:
        gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        gradient_checkpointing_enable()
    print("[builder] gradient checkpointing enabled.", flush=True)


def build_model_tokenizer_processor(config: ExperimentRuntimeConfig) -> BuildArtifacts:
    if config.finetune.mode not in {"lora", "full"}:
        raise ValueError(
            f"Unsupported finetune.mode={config.finetune.mode!r}. Expected 'lora' or 'full'."
        )
    model_class = _resolve_model_class()
    model_source = _resolve_model_source(config)
    local_files_only = _is_local_model_source(model_source)
    dtype = torch.bfloat16 if config.train.bf16 and torch.cuda.is_available() else None
    model_kwargs = {
        "trust_remote_code": config.model.trust_remote_code,
        "local_files_only": local_files_only,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    attn_implementation = _resolve_attn_implementation(config)
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    print(f"[builder] loading model from: {model_source}", flush=True)
    model = model_class.from_pretrained(model_source, **model_kwargs)
    print("[builder] loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_source,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=local_files_only,
    )
    print("[builder] resolving tokenizer...", flush=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=config.model.trust_remote_code,
            local_files_only=local_files_only,
        )
        processor.tokenizer = tokenizer

    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model, "generation_config") and tokenizer.pad_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    _sanitize_generation_config(model, config)

    if config.finetune.mode == "lora":
        _freeze_all_parameters(model)
        if not config.lora.enabled:
            raise ValueError("finetune.mode='lora' requires lora.enabled=true.")
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,
            target_modules=config.lora.target_modules,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        _enable_all_parameters(model)

    if config.model.freeze_vision_tower:
        _set_requires_grad_by_name(model, config.model.vision_name_substrings, False)
    if config.finetune.mode == "lora" and config.model.train_projector:
        _set_requires_grad_by_name(model, config.model.projector_name_substrings, True)

    # embedding层 和 LM Head 需要训练，否则不会有效果
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is not None:
        for parameter in input_embeddings.parameters():
            parameter.requires_grad = True
    if output_embeddings is not None:
        for parameter in output_embeddings.parameters():
            parameter.requires_grad = True

    _maybe_enable_gradient_checkpointing(model, config)
    summary = _trainable_summary(model)
    return BuildArtifacts(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        trainable_summary=summary,
    )
