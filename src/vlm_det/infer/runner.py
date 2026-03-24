from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vlm_det.config import ExperimentRuntimeConfig, load_config
from vlm_det.modeling.builder import BuildArtifacts, build_model_tokenizer_processor
from vlm_det.protocol.codec import ArrowCodec
from vlm_det.utils.checkpoint import load_training_checkpoint
from vlm_det.utils.distributed import unwrap_model
from vlm_det.utils.generation import build_generate_kwargs, trim_generated_ids_at_eos


@dataclass
class ArrowInferenceRunner:
    config: ExperimentRuntimeConfig
    artifacts: BuildArtifacts
    codec: ArrowCodec
    device: torch.device

    def predict(self, image: Image.Image) -> tuple[str, dict[str, Any]]:
        pil_image = image.convert("RGB")
        width, height = pil_image.size
        model_inputs, prompt_length = self._prepare_inputs(pil_image)
        input_context_length = int(model_inputs["input_ids"].shape[1])
        raw_model = unwrap_model(self.artifacts.model)
        raw_model.eval()
        generate_kwargs = build_generate_kwargs(
            self.artifacts.tokenizer,
            generation_config=getattr(raw_model, "generation_config", None),
            num_bins=self.codec.num_bins,
            prompt_lengths=[prompt_length],
            max_new_tokens=self.config.eval.max_new_tokens,
            num_beams=self.config.eval.num_beams,
            do_sample=self.config.eval.do_sample,
            use_cache=self.config.eval.use_cache,
        )
        with torch.inference_mode():
            output_ids = raw_model.generate(
                **model_inputs,
                **generate_kwargs,
            )
        continuation = output_ids[0, input_context_length:]
        trimmed_ids = trim_generated_ids_at_eos(continuation, generate_kwargs.get("eos_token_id"))
        decoded = self.artifacts.tokenizer.decode(trimmed_ids, skip_special_tokens=False)
        strict_text = self.artifacts.tokenizer.decode(trimmed_ids, skip_special_tokens=True)

        lenient_prediction: dict[str, Any] | None = None
        lenient_error: str | None = None
        strict_error: str | None = None
        try:
            lenient_prediction = self.codec.decode(decoded, image_width=width, image_height=height)
        except Exception as exc:  # noqa: BLE001
            lenient_error = str(exc)

        if lenient_prediction is not None:
            try:
                self.codec.decode(
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
            "lenient": {
                "ok": lenient_error is None,
                "prediction": lenient_prediction,
                "error": lenient_error,
            },
            "strict": {
                "ok": strict_error is None,
                "prediction": lenient_prediction if strict_error is None else None,
                "error": strict_error,
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
        batch = self.artifacts.processor(**processor_kwargs)
        prompt_length = int(batch["attention_mask"].sum(dim=1).item())
        model_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        return model_inputs, prompt_length

    def _build_prompt(self) -> str:
        messages: list[dict[str, Any]] = []
        if self.config.prompt.system_prompt.strip():
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.config.prompt.system_prompt}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.config.prompt.user_prompt},
                ],
            }
        )
        template_owner = (
            self.artifacts.processor
            if hasattr(self.artifacts.processor, "apply_chat_template")
            else self.artifacts.tokenizer
        )
        return template_owner.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def load_inference_runner(config_path: str | Path, checkpoint_path: str | Path) -> ArrowInferenceRunner:
    config = load_config(config_path)
    artifacts = build_model_tokenizer_processor(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts.model = artifacts.model.to(device)
    load_training_checkpoint(
        checkpoint_dir=checkpoint_path,
        model=artifacts.model,
        tokenizer=artifacts.tokenizer,
        processor=artifacts.processor,
        strict=True,
        resume_training_state=False,
    )
    unwrap_model(artifacts.model).eval()
    codec = ArrowCodec(num_bins=config.tokenizer.num_bins)
    return ArrowInferenceRunner(
        config=config,
        artifacts=artifacts,
        codec=codec,
        device=device,
    )
