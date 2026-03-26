from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vlm_det.config import ExperimentRuntimeConfig, load_config
from vlm_det.data.two_stage import (
    build_padded_crop,
    build_stage2_prompt,
    quantize_bbox_2d,
    quantize_keypoints_2d,
    to_crop_local_bbox,
    to_crop_local_keypoints,
)
from vlm_det.infer.runner import ArrowInferenceRunner, _resolve_device
from vlm_det.modeling.builder import BuildArtifacts, build_model_tokenizer_processor
from vlm_det.protocol.keypoint_codec import KeypointSequenceCodec
from vlm_det.utils.checkpoint import load_training_checkpoint
from vlm_det.utils.distributed import reset_model_runtime_state, unwrap_model
from vlm_det.utils.generation import (
    build_generate_kwargs,
    build_json_array_stopping_criteria,
    find_balanced_json_array_end,
    trim_generated_ids_at_eos,
)


@dataclass
class Stage2KeypointInferenceRunner:
    config: ExperimentRuntimeConfig
    artifacts: BuildArtifacts
    codec: KeypointSequenceCodec
    device: torch.device

    def predict(
        self,
        image: Image.Image,
        *,
        label: str,
        bbox_2d: list[int],
        hint_keypoints_2d: list[list[int]],
        max_new_tokens: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        pil_image = image.convert("RGB")
        width, height = pil_image.size
        prompt = build_stage2_prompt(
            label=label,
            bbox_2d=bbox_2d,
            hint_keypoints_2d=hint_keypoints_2d,
        )
        model_inputs, input_context_length = self._prepare_inputs(pil_image, prompt=prompt)
        raw_model = unwrap_model(self.artifacts.model)
        raw_model.eval()
        generate_kwargs = build_generate_kwargs(
            self.artifacts.tokenizer,
            generation_config=getattr(raw_model, "generation_config", None),
            num_bins=self.codec.num_bins,
            prompt_lengths=[input_context_length],
            max_new_tokens=max_new_tokens or self.config.eval.max_new_tokens,
            num_beams=self.config.eval.num_beams,
            do_sample=self.config.eval.do_sample,
            temperature=self.config.eval.temperature,
            top_p=self.config.eval.top_p,
            top_k=self.config.eval.top_k,
            use_cache=self.config.eval.use_cache,
        )
        generate_kwargs["stopping_criteria"] = build_json_array_stopping_criteria(
            self.artifacts.tokenizer,
            prompt_lengths=[input_context_length],
        )
        requested_max_new_tokens = int(generate_kwargs["max_new_tokens"])
        with torch.inference_mode():
            reset_model_runtime_state(raw_model)
            output_ids = raw_model.generate(**model_inputs, **generate_kwargs)
        continuation = output_ids[0, input_context_length:]
        continuation_ids = continuation.tolist()
        raw_continuation_text = self.artifacts.tokenizer.decode(continuation_ids, skip_special_tokens=False)
        json_array_end = find_balanced_json_array_end(raw_continuation_text)
        trimmed_ids = trim_generated_ids_at_eos(continuation, generate_kwargs.get("eos_token_id"))
        decoded = self.artifacts.tokenizer.decode(trimmed_ids, skip_special_tokens=False)
        strict_text = self.artifacts.tokenizer.decode(trimmed_ids, skip_special_tokens=True)
        closed_json_array = json_array_end is not None
        hit_max_new_tokens = len(continuation_ids) >= requested_max_new_tokens
        if closed_json_array:
            decoded = raw_continuation_text[:json_array_end]
            strict_text = decoded

        lenient_prediction: dict[str, Any] | None = None
        lenient_error: str | None = None
        strict_error: str | None = None
        try:
            lenient_prediction = self.codec.decode(decoded, image_width=width, image_height=height)
        except Exception as exc:  # noqa: BLE001
            lenient_error = str(exc)

        if lenient_prediction is not None:
            try:
                self.codec.decode(strict_text, image_width=width, image_height=height, strict=True)
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
                    "json_array_closed"
                    if closed_json_array
                    else "max_new_tokens"
                    if hit_max_new_tokens
                    else "unknown"
                ),
            },
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
            "condition": {
                "label": label,
                "bbox_2d": bbox_2d,
                "keypoints_2d": hint_keypoints_2d,
            },
        }

    def _prepare_inputs(self, image: Image.Image, *, prompt: str) -> tuple[dict[str, torch.Tensor], int]:
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


@dataclass
class TwoStageInferenceRunner:
    stage1_runner: ArrowInferenceRunner
    stage2_runner: Stage2KeypointInferenceRunner
    padding_ratio: float = 0.5

    def predict(
        self,
        image: Image.Image,
        *,
        stage1_max_new_tokens: int | None = None,
        stage2_max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        pil_image = image.convert("RGB")
        stage1_raw_text, stage1_report = self.stage1_runner.predict(
            pil_image,
            max_new_tokens=stage1_max_new_tokens,
        )
        stage1_prediction = stage1_report["strict"]["prediction"] or stage1_report["lenient"]["prediction"]
        if stage1_prediction is None:
            return {
                "stage1_raw_text": stage1_raw_text,
                "stage1_report": stage1_report,
                "stage2_results": [],
                "final_prediction": {"instances": []},
            }

        final_instances: list[dict[str, Any]] = []
        stage2_results: list[dict[str, Any]] = []
        for index, instance in enumerate(stage1_prediction.get("instances", [])):
            crop_image, crop_box = build_padded_crop(
                pil_image,
                bbox=instance["bbox"],
                padding_ratio=self.padding_ratio,
            )
            crop_width, crop_height = crop_image.size
            local_bbox = to_crop_local_bbox(instance["bbox"], crop_box)
            local_hint_keypoints = to_crop_local_keypoints(instance["keypoints"], crop_box)
            local_bbox_2d = quantize_bbox_2d(
                local_bbox,
                crop_width,
                crop_height,
                self.stage2_runner.codec.num_bins,
            )
            local_hint_keypoints_2d = quantize_keypoints_2d(
                local_hint_keypoints,
                crop_width,
                crop_height,
                self.stage2_runner.codec.num_bins,
            )
            stage2_raw_text, stage2_report = self.stage2_runner.predict(
                crop_image,
                label=instance["label"],
                bbox_2d=local_bbox_2d,
                hint_keypoints_2d=local_hint_keypoints_2d,
                max_new_tokens=stage2_max_new_tokens,
            )
            stage2_prediction = stage2_report["strict"]["prediction"] or stage2_report["lenient"]["prediction"]
            if stage2_prediction is None:
                global_keypoints = [list(point) for point in instance["keypoints"]]
            else:
                local_pred_keypoints = stage2_prediction["keypoints"]
                crop_x1, crop_y1, _crop_x2, _crop_y2 = crop_box
                global_keypoints = [
                    [
                        min(max(float(point[0]) + float(crop_x1), 0.0), float(pil_image.width - 1)),
                        min(max(float(point[1]) + float(crop_y1), 0.0), float(pil_image.height - 1)),
                    ]
                    for point in local_pred_keypoints
                ]
            final_instances.append(
                {
                    "label": instance["label"],
                    "bbox": [float(value) for value in instance["bbox"]],
                    "keypoints": global_keypoints,
                }
            )
            stage2_results.append(
                {
                    "index": int(index),
                    "crop_box": crop_box,
                    "raw_text": stage2_raw_text,
                    "report": stage2_report,
                }
            )

        return {
            "stage1_raw_text": stage1_raw_text,
            "stage1_report": stage1_report,
            "stage2_results": stage2_results,
            "final_prediction": {"instances": final_instances},
        }


def _load_stage2_runner(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
    model_name_or_path: str | None = None,
) -> Stage2KeypointInferenceRunner:
    config = load_config(config_path)
    if model_name_or_path is not None:
        config.model.model_name_or_path = model_name_or_path
        config.model.remote_model_name_or_path = model_name_or_path
    artifacts = build_model_tokenizer_processor(config)
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
    return Stage2KeypointInferenceRunner(
        config=config,
        artifacts=artifacts,
        codec=KeypointSequenceCodec(num_bins=config.tokenizer.num_bins),
        device=device,
    )


def load_two_stage_inference_runner(
    *,
    stage1_config_path: str | Path,
    stage1_checkpoint_path: str | Path,
    stage2_config_path: str | Path,
    stage2_checkpoint_path: str | Path,
    device_name: str | None = None,
    stage1_model_name_or_path: str | None = None,
    stage2_model_name_or_path: str | None = None,
    padding_ratio: float = 0.5,
) -> TwoStageInferenceRunner:
    device = _resolve_device(device_name)
    from vlm_det.infer.runner import load_inference_runner

    loaded_stage1_runner = load_inference_runner(
        checkpoint_path=stage1_checkpoint_path,
        config_path=stage1_config_path,
        model_name_or_path=stage1_model_name_or_path,
        device_name=device_name,
    )
    stage2_runner = _load_stage2_runner(
        config_path=stage2_config_path,
        checkpoint_path=stage2_checkpoint_path,
        device=device,
        model_name_or_path=stage2_model_name_or_path,
    )
    return TwoStageInferenceRunner(
        stage1_runner=loaded_stage1_runner,
        stage2_runner=stage2_runner,
        padding_ratio=padding_ratio,
    )
