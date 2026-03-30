from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vlm_det.config import ExperimentRuntimeConfig
from vlm_det.core.registry import get_adapter
from vlm_det.infer.config import InferenceSettings, OneStageInferenceConfig, load_inference_settings
from vlm_det.modeling.builder import BuildArtifacts, build_model_tokenizer_processor
from vlm_det.prompting import build_chat_prompt, temporary_padding_side
from vlm_det.utils.checkpoint import load_training_checkpoint
from vlm_det.utils.distributed import reset_model_runtime_state, unwrap_model
from vlm_det.utils.generation import (
    build_generate_kwargs,
    find_balanced_json_array_end,
    trim_generated_ids_at_eos,
)


@dataclass
class ArrowInferenceRunner:
    settings: InferenceSettings
    config: ExperimentRuntimeConfig
    artifacts: BuildArtifacts
    adapter: Any
    device: torch.device

    def predict(
        self,
        image: Image.Image,
        *,
        max_new_tokens: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        pil_image = image.convert("RGB")
        width, height = pil_image.size
        model_inputs, prompt_length = self._prepare_inputs(pil_image)
        input_context_length = int(model_inputs["input_ids"].shape[1])
        raw_model = unwrap_model(self.artifacts.model)
        raw_model.eval()
        generate_kwargs = build_generate_kwargs(
            self.artifacts.tokenizer,
            generation_config=getattr(raw_model, "generation_config", None),
            num_bins=self.adapter.num_bins,
            prompt_lengths=[prompt_length],
            max_new_tokens=max_new_tokens or self.config.eval.max_new_tokens,
            num_beams=self.config.eval.num_beams,
            do_sample=self.config.eval.do_sample,
            temperature=self.config.eval.temperature,
            top_p=self.config.eval.top_p,
            top_k=self.config.eval.top_k,
            use_cache=self.config.eval.use_cache,
        )
        requested_max_new_tokens = int(generate_kwargs["max_new_tokens"])
        with torch.inference_mode():
            reset_model_runtime_state(raw_model)
            output_ids = raw_model.generate(
                **model_inputs,
                **generate_kwargs,
            )
        continuation = output_ids[0, input_context_length:]
        continuation_ids = continuation.tolist()
        raw_continuation_text = self.artifacts.tokenizer.decode(continuation_ids, skip_special_tokens=False)
        json_array_end = find_balanced_json_array_end(raw_continuation_text)
        trimmed_ids = trim_generated_ids_at_eos(continuation, generate_kwargs.get("eos_token_id"))
        decoded = self.artifacts.tokenizer.decode(trimmed_ids, skip_special_tokens=False)
        strict_text = self.artifacts.tokenizer.decode(trimmed_ids, skip_special_tokens=True)
        closed_json_array = json_array_end is not None
        hit_max_new_tokens = len(continuation_ids) >= requested_max_new_tokens

        lenient_prediction: dict[str, Any] | None = None
        lenient_error: str | None = None
        lenient_recovered_prefix = False
        strict_error: str | None = None
        try:
            lenient_prediction, lenient_meta = self.adapter.decode_with_meta(
                decoded,
                image_width=width,
                image_height=height,
            )
            lenient_recovered_prefix = bool(lenient_meta.get("recovered_prefix", False))
        except Exception as exc:  # noqa: BLE001
            lenient_error = str(exc)

        if lenient_prediction is not None:
            try:
                self.adapter.decode(
                    strict_text,
                    image_width=width,
                    image_height=height,
                    strict=True,
                )
            except Exception as exc:  # noqa: BLE001
                strict_error = str(exc)
        else:
            strict_error = lenient_error

        return decoded, {
            "generation": {
                "requested_max_new_tokens": requested_max_new_tokens,
                "generated_tokens": len(continuation_ids),
                "returned_tokens": len(trimmed_ids),
                "hit_max_new_tokens": hit_max_new_tokens,
                "closed_json_array": closed_json_array,
                "stop_reason": (
                    "max_new_tokens"
                    if hit_max_new_tokens
                    else "eos_or_unknown"
                ),
            },
            "lenient": {
                "ok": lenient_error is None,
                "prediction": lenient_prediction,
                "error": lenient_error,
                "recovered_prefix": lenient_recovered_prefix,
            },
            "strict": {
                "ok": strict_error is None,
                "prediction": lenient_prediction if strict_error is None else None,
                "error": strict_error,
                "recovered_prefix": False,
            },
        }

    def _prepare_inputs(self, image: Image.Image) -> tuple[dict[str, torch.Tensor], int]:
        prompt = self._build_prompt()
        processor_kwargs: dict[str, Any] = {
            "text": [prompt],
            "images": [image],
            "return_tensors": "pt",
        }
        if self.config.model.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.config.model.min_pixels
        if self.config.model.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.config.model.max_pixels
        with temporary_padding_side(self.artifacts.processor, self.artifacts.tokenizer, padding_side="left"):
            batch = self.artifacts.processor(**processor_kwargs)
        prompt_length = int(batch["attention_mask"].sum(dim=1).item())
        model_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        return model_inputs, prompt_length

    def _build_prompt(self) -> str:
        return build_chat_prompt(
            self.artifacts.processor,
            self.artifacts.tokenizer,
            system_prompt=self.config.prompt.system_prompt,
            user_prompt=self.config.prompt.user_prompt,
        )


def _resolve_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_inference_runner(
    checkpoint_path: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    infer_config: OneStageInferenceConfig | None = None,
    env_file: str | Path | None = None,
    model_name_or_path: str | None = None,
    device_name: str | None = None,
) -> ArrowInferenceRunner:
    if infer_config is None:
        settings = load_inference_settings(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            env_file=env_file,
        )
    else:
        settings = load_inference_settings(
            checkpoint_path=checkpoint_path,
            config_path=None,
            env_file=env_file,
            infer_config=infer_config,
        )
    config = settings.runtime

    if model_name_or_path is not None:
        config.model.model_name_or_path = model_name_or_path
        config.model.remote_model_name_or_path = model_name_or_path

    artifacts = build_model_tokenizer_processor(config)
    device = _resolve_device(device_name or settings.device)
    artifacts.model = artifacts.model.to(device)
    load_training_checkpoint(
        checkpoint_dir=settings.checkpoint_path,
        model=artifacts.model,
        tokenizer=artifacts.tokenizer,
        processor=artifacts.processor,
        strict=True,
        resume_training_state=False,
    )
    unwrap_model(artifacts.model).eval()
    adapter = get_adapter(
        task_type=config.task.task_type or config.task.type,
        domain_type=config.task.domain_type,
        num_bins=config.tokenizer.num_bins,
    )
    return ArrowInferenceRunner(
        settings=settings,
        config=config,
        artifacts=artifacts,
        adapter=adapter,
        device=device,
    )
