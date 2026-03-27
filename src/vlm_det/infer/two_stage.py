from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vlm_det.config import ExperimentRuntimeConfig
from vlm_det.data.two_stage import (
    build_padded_crop,
)
from vlm_det.infer.config import (
    TwoStageInferenceConfig,
    build_runtime_from_two_stage_infer_config,
    load_two_stage_inference_config,
)
from vlm_det.infer.runner import ArrowInferenceRunner, _resolve_device
from vlm_det.modeling.builder import BuildArtifacts, build_model_tokenizer_processor
from vlm_det.prompting import build_chat_prompt, render_prompt_template, temporary_padding_side
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
            model_inputs, prompt_lengths = self._prepare_inputs(images, prompt_texts=prompt_texts)
            generate_kwargs = build_generate_kwargs(
                self.artifacts.tokenizer,
                generation_config=getattr(raw_model, "generation_config", None),
                num_bins=self.codec.num_bins,
                prompt_lengths=prompt_lengths,
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
                prompt_lengths=prompt_lengths,
            )
            requested_max_new_tokens = int(generate_kwargs["max_new_tokens"])
            with torch.inference_mode():
                reset_model_runtime_state(raw_model)
                output_ids = raw_model.generate(**model_inputs, **generate_kwargs)
            for row_index, request in enumerate(batch_requests):
                input_context_length = int(prompt_lengths[row_index])
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

    def _prepare_inputs(
        self,
        images: list[Image.Image],
        *,
        prompt_texts: list[str],
    ) -> tuple[dict[str, torch.Tensor], list[int]]:
        processor_kwargs: dict[str, Any] = {
            "text": prompt_texts,
            "images": images,
            "padding": True,
            "return_tensors": "pt",
        }
        if self.config.model.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.config.model.min_pixels
        if self.config.model.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.config.model.max_pixels
        with temporary_padding_side(self.artifacts.processor, self.artifacts.tokenizer, padding_side="left"):
            batch = self.artifacts.processor(**processor_kwargs)
        prompt_lengths = [int(value) for value in batch["attention_mask"].sum(dim=1).tolist()]
        model_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        return model_inputs, prompt_lengths


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
        if self.stage2_runner is None:
            return {
                "stage1_raw_text": stage1_raw_text,
                "stage1_report": stage1_report,
                "stage2_results": [],
                "final_prediction": stage1_prediction,
            }

        stage1_instances = stage1_prediction.get("instances", [])
        if stage2_batch_size is not None:
            self.stage2_runner.batch_size = max(int(stage2_batch_size), 1)

        stage2_requests: list[Stage2Request] = []
        for index, instance in enumerate(stage1_instances):
            bbox = instance.get("bbox", [])
            label = str(instance.get("label", ""))
            if len(bbox) != 4:
                continue
            crop_image, crop_box = build_padded_crop(
                pil_image,
                bbox=[float(value) for value in bbox],
                padding_ratio=self.padding_ratio,
            )
            crop_width, crop_height = crop_image.size
            local_bbox = [
                float(bbox[0]) - float(crop_box[0]),
                float(bbox[1]) - float(crop_box[1]),
                float(bbox[2]) - float(crop_box[0]),
                float(bbox[3]) - float(crop_box[1]),
            ]
            local_bbox_2d = self.stage2_runner.codec.num_bins and [
                self.stage2_runner.codec._quantize(float(local_bbox[0]), crop_width),
                self.stage2_runner.codec._quantize(float(local_bbox[1]), crop_height),
                self.stage2_runner.codec._quantize(float(local_bbox[2]), crop_width),
                self.stage2_runner.codec._quantize(float(local_bbox[3]), crop_height),
            ]
            stage2_requests.append(
                Stage2Request(
                    index=index,
                    crop_image=crop_image,
                    crop_box=[int(value) for value in crop_box],
                    label=label,
                    bbox_2d=local_bbox_2d,
                    hint_keypoints_2d=[],
                )
            )

        batched_results = self.stage2_runner.predict_batch(
            stage2_requests,
            max_new_tokens=stage2_max_new_tokens,
        )

        final_instances: list[dict[str, Any]] = []
        stage2_reports: list[dict[str, Any]] = []
        for request, result in zip(stage2_requests, batched_results):
            lenient_prediction = result.report["lenient"]["prediction"]
            strict_prediction = result.report["strict"]["prediction"]
            local_prediction = strict_prediction or lenient_prediction
            local_keypoints = local_prediction.get("keypoints", []) if local_prediction else []
            crop_width, crop_height = request.crop_image.size
            if local_prediction is None:
                final_instances.append(
                    {
                        "label": request.label,
                        "bbox": [float(value) for value in stage1_instances[request.index]["bbox"]],
                        "keypoints": [],
                    }
                )
                stage2_reports.append(result.report)
                continue

            local_keypoints = local_prediction.get("keypoints", [])
            global_keypoints = [
                [float(point[0]) + float(request.crop_box[0]), float(point[1]) + float(request.crop_box[1])]
                for point in local_keypoints
            ]
            final_instances.append(
                {
                    "label": request.label,
                    "bbox": [float(value) for value in stage1_instances[request.index]["bbox"]],
                    "keypoints": global_keypoints,
                }
            )
            stage2_reports.append(result.report)

        return {
            "stage1_raw_text": stage1_raw_text,
            "stage1_report": stage1_report,
            "stage2_results": stage2_reports,
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
