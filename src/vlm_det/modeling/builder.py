from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor, AutoTokenizer

from vlm_det.config import ExperimentRuntimeConfig
from vlm_det.protocol.tokens import build_special_tokens

@dataclass
class BuildArtifacts:
    model: torch.nn.Module
    tokenizer: Any
    processor: Any
    special_tokens: list[str]
    num_added_tokens: int
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


def build_model_tokenizer_processor(config: ExperimentRuntimeConfig) -> BuildArtifacts:
    if config.finetune.mode not in {"lora", "full"}:
        raise ValueError(
            f"Unsupported finetune.mode={config.finetune.mode!r}. Expected 'lora' or 'full'."
        )
    model_class = _resolve_model_class()
    model_source = _resolve_model_source(config)
    dtype = torch.bfloat16 if config.train.bf16 and torch.cuda.is_available() else None
    model_kwargs = {
        "trust_remote_code": config.model.trust_remote_code,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if config.model.attn_implementation:
        model_kwargs["attn_implementation"] = config.model.attn_implementation

    model = model_class.from_pretrained(model_source, **model_kwargs)
    processor = AutoProcessor.from_pretrained(
        model_source,
        trust_remote_code=config.model.trust_remote_code,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=config.model.trust_remote_code,
        )
        processor.tokenizer = tokenizer

    special_tokens = build_special_tokens(config.tokenizer.num_bins)
    tokens_to_add = [token for token in special_tokens if token not in tokenizer.get_vocab()]
    
    # 向词表中添加 special token
    num_added_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": tokens_to_add}
    )
    # 调整模型 embedding 层大小，会初始化新 token 的 embedding，原本的 token 的 embedding 会被复制一份
    model.resize_token_embeddings(len(tokenizer))
    
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model, "generation_config") and tokenizer.pad_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

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

    summary = _trainable_summary(model)
    return BuildArtifacts(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        special_tokens=special_tokens,
        num_added_tokens=num_added_tokens,
        trainable_summary=summary,
    )
