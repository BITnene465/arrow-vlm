from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vlm_structgen.core.config import ExperimentRuntimeConfig
from vlm_structgen.core.registry import get_adapter
from vlm_structgen.domains.arrow.data.two_stage import (
    _bbox_iou,
    _build_sliding_crop_boxes,
    _resolve_stage1_tile_sizes,
    build_padded_crop,
)
from vlm_structgen.core.infer.config import (
    TwoStageInferenceConfig,
    build_runtime_from_two_stage_infer_config,
    load_two_stage_inference_config,
)
from vlm_structgen.core.infer.runner import InferenceRunner, _resolve_device
from vlm_structgen.core.modeling.builder import BuildArtifacts, build_model_tokenizer_processor
from vlm_structgen.core.prompting import build_chat_prompt, render_prompt_template, temporary_padding_side
from vlm_structgen.core.utils.checkpoint import load_training_checkpoint
from vlm_structgen.core.utils.distributed import reset_model_runtime_state, unwrap_model
from vlm_structgen.core.utils.generation import (
    build_generate_kwargs,
    find_balanced_json_array_end,
    trim_generated_ids_at_eos,
)


@dataclass
class Stage2KeypointInferenceRunner:
    config: ExperimentRuntimeConfig
    artifacts: BuildArtifacts
    adapter: Any
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
                num_bins=self.adapter.num_bins,
                prompt_lengths=[input_context_length] * len(batch_requests),
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
                        self.adapter.decode(strict_text, image_width=width, image_height=height, strict=True)
                    except Exception as exc:  # noqa: BLE001
                        strict_error = str(exc)
                else:
                    strict_error = lenient_error

                results_by_index[request.index] = Stage2PredictionResult(
                    index=int(request.index),
                    crop_box=[int(value) for value in request.crop_box],
                    raw_text=decoded,
                    report={
                        "raw_text": decoded,
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
    ) -> tuple[dict[str, torch.Tensor], int]:
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
        input_context_length = int(batch["input_ids"].shape[1])
        model_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        return model_inputs, input_context_length


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
    stage1_runner: InferenceRunner
    stage2_runner: Stage2KeypointInferenceRunner | None
    infer_config: TwoStageInferenceConfig
    padding_ratio: float = 0.5

    def _extract_stage1_prediction(self, report: dict[str, Any]) -> dict[str, Any] | None:
        return report["strict"]["prediction"] or report["lenient"]["prediction"]

    def _build_stage1_tile_boxes(self, image: Image.Image) -> list[list[int]]:
        infer_cfg = getattr(self, "infer_config", None)
        if infer_cfg is None:
            return []
        stage1_infer = infer_cfg.stage1
        resolved_sizes = _resolve_stage1_tile_sizes(
            image_width=int(image.width),
            image_height=int(image.height),
            tile_size_ratios=list(stage1_infer.tile_size_ratios),
            min_tile_size=int(stage1_infer.min_tile_size),
            max_tile_size=int(stage1_infer.max_tile_size),
        )
        crop_boxes: list[list[int]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for tile_size in resolved_sizes:
            stride = max(int(round(float(tile_size) * float(stage1_infer.tile_stride_ratio))), 1)
            for crop_box in _build_sliding_crop_boxes(
                image_width=int(image.width),
                image_height=int(image.height),
                tile_size=int(tile_size),
                stride=int(stride),
            ):
                key = tuple(int(value) for value in crop_box)
                if key in seen:
                    continue
                seen.add(key)
                crop_boxes.append([int(value) for value in crop_box])
        return crop_boxes

    @staticmethod
    def _map_instances_to_global(
        instances: list[dict[str, Any]],
        *,
        crop_box: list[int] | None,
    ) -> list[dict[str, Any]]:
        if crop_box is None:
            return [
                {
                    "label": str(instance.get("label", "")),
                    "bbox": [float(value) for value in instance.get("bbox", [])],
                    "keypoints": [[float(x), float(y)] for x, y in instance.get("keypoints", [])],
                }
                for instance in instances
            ]
        offset_x = float(crop_box[0])
        offset_y = float(crop_box[1])
        mapped: list[dict[str, Any]] = []
        for instance in instances:
            bbox = instance.get("bbox", [])
            mapped_bbox = [
                float(bbox[0]) + offset_x,
                float(bbox[1]) + offset_y,
                float(bbox[2]) + offset_x,
                float(bbox[3]) + offset_y,
            ] if len(bbox) == 4 else []
            mapped.append(
                {
                    "label": str(instance.get("label", "")),
                    "bbox": mapped_bbox,
                    "keypoints": [
                        [float(x) + offset_x, float(y) + offset_y]
                        for x, y in instance.get("keypoints", [])
                    ],
                }
            )
        return mapped

    def _aggregate_stage1_instances(
        self,
        branch_predictions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        infer_cfg = getattr(self, "infer_config", None)
        dedup_iou_threshold = 0.65
        if infer_cfg is not None:
            dedup_iou_threshold = float(infer_cfg.stage1.proposal_dedup_iou_threshold)

        proposals: list[dict[str, Any]] = []
        for branch in branch_predictions:
            source_type = str(branch["source_type"])
            crop_box = branch.get("crop_box")
            prediction = branch.get("prediction") or {"instances": []}
            instances = self._map_instances_to_global(prediction.get("instances", []), crop_box=crop_box)
            for instance in instances:
                bbox = instance.get("bbox", [])
                if len(bbox) != 4:
                    continue
                proposals.append(
                    {
                        **instance,
                        "_source_type": source_type,
                        "_crop_box": crop_box,
                    }
                )

        proposals.sort(
            key=lambda item: (
                0 if str(item.get("_source_type", "")).startswith("tile_") else 1,
                (float(item["bbox"][2]) - float(item["bbox"][0])) * (float(item["bbox"][3]) - float(item["bbox"][1])),
                float(item["bbox"][1]),
                float(item["bbox"][0]),
            )
        )

        deduped: list[dict[str, Any]] = []
        for proposal in proposals:
            label = str(proposal.get("label", ""))
            bbox = proposal.get("bbox", [])
            if any(
                str(existing.get("label", "")) == label
                and _bbox_iou(existing.get("bbox", []), bbox) >= dedup_iou_threshold
                for existing in deduped
            ):
                continue
            deduped.append(
                {
                    "label": label,
                    "bbox": [float(value) for value in bbox],
                    "keypoints": [],
                }
            )
        return deduped

    def _predict_stage1(self, image: Image.Image, *, max_new_tokens: int | None = None) -> tuple[str, dict[str, Any], dict[str, Any]]:
        return self._predict_stage1_with_options(
            image,
            max_new_tokens=max_new_tokens,
            use_mixed_proposals=None,
        )

    def _predict_stage1_with_options(
        self,
        image: Image.Image,
        *,
        max_new_tokens: int | None = None,
        use_mixed_proposals: bool | None = None,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        infer_cfg = getattr(self, "infer_config", None)
        if infer_cfg is None:
            raw_text, report = self.stage1_runner.predict(image, max_new_tokens=max_new_tokens)
            prediction = self._extract_stage1_prediction(report) or {"instances": []}
            return raw_text, report, prediction

        branch_predictions: list[dict[str, Any]] = []
        raw_texts: list[str] = []
        enable_mixed_proposals = (
            bool(infer_cfg.stage1.tile_size_ratios)
            if use_mixed_proposals is None
            else bool(use_mixed_proposals)
        )

        if bool(infer_cfg.stage1.include_full_image):
            full_raw_text, full_report = self.stage1_runner.predict(image, max_new_tokens=max_new_tokens)
            raw_texts.append(full_raw_text)
            full_prediction = self._extract_stage1_prediction(full_report)
            branch_predictions.append(
                {
                    "source_type": "full_image",
                    "crop_box": None,
                    "raw_text": full_raw_text,
                    "report": full_report,
                    "prediction": full_prediction,
                }
            )

        if enable_mixed_proposals:
            for tile_index, crop_box in enumerate(self._build_stage1_tile_boxes(image)):
                tile_image = image.crop(tuple(crop_box))
                tile_raw_text, tile_report = self.stage1_runner.predict(tile_image, max_new_tokens=max_new_tokens)
                raw_texts.append(tile_raw_text)
                branch_predictions.append(
                    {
                        "source_type": f"tile_{tile_index:04d}",
                        "crop_box": [int(value) for value in crop_box],
                        "raw_text": tile_raw_text,
                        "report": tile_report,
                        "prediction": self._extract_stage1_prediction(tile_report),
                    }
                )

        aggregated_instances = self._aggregate_stage1_instances(branch_predictions)
        aggregated_prediction = {"instances": aggregated_instances}
        stage1_report = {
            "generation": {
                "num_branches": len(branch_predictions),
                "num_full_image_branches": sum(1 for branch in branch_predictions if branch["crop_box"] is None),
                "num_tile_branches": sum(1 for branch in branch_predictions if branch["crop_box"] is not None),
                "mixed_proposals_enabled": enable_mixed_proposals,
            },
            "lenient": {
                "ok": True,
                "prediction": aggregated_prediction,
                "error": None,
                "recovered_prefix": any(
                    bool(branch["report"]["lenient"].get("recovered_prefix", False))
                    for branch in branch_predictions
                ),
            },
            "strict": {
                "ok": True,
                "prediction": aggregated_prediction,
                "error": None,
                "recovered_prefix": False,
            },
            "branches": [
                {
                    "source_type": branch["source_type"],
                    "crop_box": branch["crop_box"],
                    "report": branch["report"],
                }
                for branch in branch_predictions
            ],
        }
        return "\n".join(raw_texts), stage1_report, aggregated_prediction

    def predict(
        self,
        image: Image.Image,
        *,
        stage1_max_new_tokens: int | None = None,
        stage2_max_new_tokens: int | None = None,
        stage2_batch_size: int | None = None,
        stage1_use_mixed_proposals: bool | None = None,
    ) -> dict[str, Any]:
        pil_image = image.convert("RGB")
        stage1_raw_text, stage1_report, stage1_prediction = self._predict_stage1_with_options(
            pil_image,
            max_new_tokens=stage1_max_new_tokens,
            use_mixed_proposals=stage1_use_mixed_proposals,
        )
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
            if local_prediction is None:
                final_instances.append(
                    {
                        "label": request.label,
                        "bbox": [float(value) for value in stage1_instances[request.index]["bbox"]],
                        "keypoints": [],
                        "stage2_status": "failed",
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
                    "stage2_status": "success",
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
        adapter=get_adapter(
            task_type="keypoint_sequence",
            domain_type="arrow",
            num_bins=config.tokenizer.num_bins,
        ),
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
    from vlm_structgen.core.infer.runner import load_inference_runner

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
        infer_config=infer_config,
        padding_ratio=infer_config.padding_ratio,
    )
