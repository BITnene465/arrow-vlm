from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vlm_det.config import ExperimentRuntimeConfig
from vlm_det.data.two_stage import (
    build_padded_crop,
    quantize_bbox_2d,
    quantize_keypoints_2d,
    to_crop_local_bbox,
    to_crop_local_keypoints,
)
from vlm_det.infer.config import (
    TwoStageInferenceConfig,
    build_runtime_from_two_stage_infer_config,
    load_two_stage_inference_config,
)
from vlm_det.infer.runner import ArrowInferenceRunner, _resolve_device
from vlm_det.modeling.builder import BuildArtifacts, build_model_tokenizer_processor
from vlm_det.prompting import build_chat_prompt, render_prompt_template
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
    batch_size: int = 1

    def predict_batch(
        self,
        requests: list[Stage2Request],
        *,
        max_new_tokens: int | None = None,
    ) -> list[Stage2PredictionResult]:
        if not requests:
            return []
        sorted_requests = sorted(
            requests,
            key=lambda request: (
                int(request.crop_image.width) * int(request.crop_image.height),
                len(request.label),
                len(request.hint_keypoints_2d),
                int(request.index),
            ),
        )
        raw_model = unwrap_model(self.artifacts.model)
        raw_model.eval()
        results_by_index: dict[int, Stage2PredictionResult] = {}
        effective_batch_size = max(int(self.batch_size), 1)
        for start in range(0, len(sorted_requests), effective_batch_size):
            batch_requests = sorted_requests[start : start + effective_batch_size]
            prompt_texts = [self._build_prompt(request) for request in batch_requests]
            images = [request.crop_image.convert("RGB") for request in batch_requests]
            model_inputs, input_context_length = self._prepare_inputs(images, prompt_texts=prompt_texts)
            generate_kwargs = build_generate_kwargs(
                self.artifacts.tokenizer,
                generation_config=getattr(raw_model, "generation_config", None),
                num_bins=self.codec.num_bins,
                prompt_lengths=[input_context_length] * len(batch_requests),
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
                prompt_lengths=[input_context_length] * len(batch_requests),
            )
            requested_max_new_tokens = int(generate_kwargs["max_new_tokens"])
            with torch.inference_mode():
                reset_model_runtime_state(raw_model)
                output_ids = raw_model.generate(**model_inputs, **generate_kwargs)
            for row_index, request in enumerate(batch_requests):
                width, height = request.crop_image.size
                continuation = output_ids[row_index, input_context_length:]
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
                lenient_recovered_prefix = False
                strict_error: str | None = None
                try:
                    lenient_prediction, lenient_meta = self.codec.decode_with_meta(
                        decoded,
                        image_width=width,
                        image_height=height,
                    )
                    lenient_recovered_prefix = bool(lenient_meta.get("recovered_prefix", False))
                except Exception as exc:  # noqa: BLE001
                    lenient_error = str(exc)

                if lenient_prediction is not None:
                    try:
                        self.codec.decode(strict_text, image_width=width, image_height=height, strict=True)
                    except Exception as exc:  # noqa: BLE001
                        strict_error = str(exc)
                else:
                    strict_error = lenient_error

                results_by_index[request.index] = Stage2PredictionResult(
                    index=int(request.index),
                    crop_box=[int(value) for value in request.crop_box],
                    raw_text=decoded,
                    report={
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
                            "recovered_prefix": lenient_recovered_prefix,
                        },
                        "strict": {
                            "ok": strict_error is None,
                            "prediction": lenient_prediction if strict_error is None else None,
                            "error": strict_error,
                            "recovered_prefix": False,
                        },
                        "condition": {
                            "label": request.label,
                            "bbox_2d": request.bbox_2d,
                            "keypoints_2d": request.hint_keypoints_2d,
                        },
                    },
                )
        return [results_by_index[request.index] for request in requests]

    def _build_prompt(self, request: Stage2Request) -> str:
        prompt = render_prompt_template(
            self.config.prompt.user_prompt_template,
            {
                "label": request.label,
                "bbox_2d": request.bbox_2d,
                "keypoints_2d": request.hint_keypoints_2d,
            },
        )
        return build_chat_prompt(
            self.artifacts.processor,
            self.artifacts.tokenizer,
            system_prompt=self.config.prompt.system_prompt,
            user_prompt=prompt,
        )

    def _prepare_inputs(self, images: list[Image.Image], *, prompt_texts: list[str]) -> tuple[dict[str, torch.Tensor], int]:
        processor_kwargs: dict[str, Any] = {
            "text": prompt_texts,
            "images": images,
            "return_tensors": "pt",
        }
        if self.config.model.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.config.model.min_pixels
        if self.config.model.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.config.model.max_pixels
        batch = self.artifacts.processor(**processor_kwargs)
        prompt_length = int(batch["input_ids"].shape[1])
        model_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        return model_inputs, prompt_length


@dataclass
class Stage2Request:
    index: int
    crop_image: Image.Image
    crop_box: list[int]
    label: str
    bbox_2d: list[int]
    hint_keypoints_2d: list[list[int]]


@dataclass
class Stage2PredictionResult:
    index: int
    crop_box: list[int]
    raw_text: str
    report: dict[str, Any]


@dataclass
class TwoStageInferenceRunner:
    stage1_runner: ArrowInferenceRunner
    stage2_runner: Stage2KeypointInferenceRunner | None
    padding_ratio: float = 0.5

    def predict(
        self,
        image: Image.Image,
        *,
        stage1_max_new_tokens: int | None = None,
        stage2_max_new_tokens: int | None = None,
        stage2_batch_size: int | None = None,
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
        stage2_requests: list[Stage2Request] = []
        stage2_fallback_instances: dict[int, dict[str, Any]] = {}
        for index, instance in enumerate(stage1_prediction.get("instances", [])):
            fallback_instance = {
                "label": instance["label"],
                "bbox": [float(value) for value in instance["bbox"]],
                "keypoints": [[float(x), float(y)] for x, y in instance["keypoints"]],
            }
            if self.stage2_runner is None:
                final_instances.append(fallback_instance)
                continue
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
            stage2_fallback_instances[int(index)] = fallback_instance
            stage2_requests.append(
                Stage2Request(
                    index=int(index),
                    crop_image=crop_image,
                    crop_box=[int(value) for value in crop_box],
                    label=str(instance["label"]),
                    bbox_2d=[int(value) for value in local_bbox_2d],
                    hint_keypoints_2d=[[int(x), int(y)] for x, y in local_hint_keypoints_2d],
                )
            )
        if self.stage2_runner is not None and stage2_requests:
            original_batch_size = self.stage2_runner.batch_size
            try:
                if stage2_batch_size is not None:
                    self.stage2_runner.batch_size = max(int(stage2_batch_size), 1)
                batched_results = self.stage2_runner.predict_batch(
                    stage2_requests,
                    max_new_tokens=stage2_max_new_tokens,
                )
            finally:
                self.stage2_runner.batch_size = original_batch_size
            results_by_index = {int(item.index): item for item in batched_results}
            for index, instance in enumerate(stage1_prediction.get("instances", [])):
                stage2_result = results_by_index.get(int(index))
                if stage2_result is None:
                    final_instances.append(stage2_fallback_instances[int(index)])
                    continue
                stage2_report = stage2_result.report
                stage2_prediction = stage2_report["strict"]["prediction"] or stage2_report["lenient"]["prediction"]
                if stage2_prediction is None:
                    global_keypoints = [list(point) for point in stage2_fallback_instances[int(index)]["keypoints"]]
                else:
                    local_pred_keypoints = stage2_prediction["keypoints"]
                    crop_x1, crop_y1, _crop_x2, _crop_y2 = stage2_result.crop_box
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
                        "crop_box": stage2_result.crop_box,
                        "raw_text": stage2_result.raw_text,
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
    checkpoint_path: str | Path,
    infer_config: Any,
    device: torch.device,
    model_name_or_path: str | None = None,
) -> Stage2KeypointInferenceRunner:
    config = build_runtime_from_two_stage_infer_config(checkpoint_path, infer_config)
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
        batch_size=max(int(getattr(infer_config, "batch_size", 1)), 1),
    )


def load_two_stage_inference_runner(
    *,
    config_path: str | Path,
    stage1_checkpoint_path: str | Path,
    stage2_checkpoint_path: str | Path | None = None,
    device_name: str | None = None,
    stage1_model_name_or_path: str | None = None,
    stage2_model_name_or_path: str | None = None,
) -> TwoStageInferenceRunner:
    device = _resolve_device(device_name)
    from vlm_det.infer.runner import load_inference_runner

    infer_config: TwoStageInferenceConfig = load_two_stage_inference_config(config_path)
    loaded_stage1_runner = load_inference_runner(
        checkpoint_path=stage1_checkpoint_path,
        infer_config=infer_config.stage1,
        model_name_or_path=stage1_model_name_or_path,
        device_name=device_name,
    )
    stage2_runner = None
    if stage2_checkpoint_path is not None:
        stage2_runner = _load_stage2_runner(
            checkpoint_path=stage2_checkpoint_path,
            infer_config=infer_config.stage2,
            device=device,
            model_name_or_path=stage2_model_name_or_path,
        )
    return TwoStageInferenceRunner(
        stage1_runner=loaded_stage1_runner,
        stage2_runner=stage2_runner,
        padding_ratio=infer_config.padding_ratio,
    )
