from __future__ import annotations

import math
from typing import Any

import torch

from vlm_structgen.core.registry import get_adapter
from vlm_structgen.core.utils.distributed import reduce_numeric_dict, reset_model_runtime_state, unwrap_model
from vlm_structgen.core.utils.generation import (
    build_generate_kwargs,
    trim_generated_ids_at_eos,
)
from vlm_structgen.core.utils.logging import create_progress_bar


class Evaluator:
    def __init__(
        self,
        num_bins: int,
        tokenizer,
        max_new_tokens: int,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        use_cache: bool = True,
        bbox_iou_threshold: float = 0.5,
        strict_point_distance_px: float = 8.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_cache = use_cache
        self.bbox_iou_threshold = bbox_iou_threshold
        self.strict_point_distance_px = strict_point_distance_px
        self.num_bins = int(num_bins)

    def evaluate_model(self, model: torch.nn.Module, dataloader) -> dict[str, float]:
        counts = self._empty_counts()
        raw_model = unwrap_model(model)
        raw_model.eval()
        progress = create_progress_bar(total=len(dataloader), desc="eval", leave=True)
        with torch.no_grad():
            for batch in dataloader:
                batch_counts = self.evaluate_batch(raw_model, batch)
                for key, value in batch_counts.items():
                    counts[key] += value
                if progress is not None:
                    samples = max(counts["samples"], 1.0)
                    parse_rate_lenient = counts["parse_success_lenient"] / samples
                    parse_rate_strict = counts["parse_success_strict"] / samples
                    if counts["stage2_samples"] > 0 and counts["structured_samples"] == 0 and counts["grounding_samples"] == 0:
                        e2e = counts["end_to_end_correct"] / max(counts["gt_instances"], 1.0)
                        l2_mean = counts["point_distance_sum"] / max(counts["point_count"], 1.0)
                        progress.set_postfix(
                            {
                                "parseL": f"{parse_rate_lenient:.2f}",
                                "parseS": f"{parse_rate_strict:.2f}",
                                "e2e": f"{e2e:.2f}",
                                "l2": f"{l2_mean:.1f}",
                            }
                        )
                    else:
                        precision = counts["bbox_tp"] / max(counts["bbox_tp"] + counts["bbox_fp"], 1.0)
                        recall = counts["bbox_tp"] / max(counts["bbox_tp"] + counts["bbox_fn"], 1.0)
                        progress.set_postfix(
                            {
                                "parseL": f"{parse_rate_lenient:.2f}",
                                "parseS": f"{parse_rate_strict:.2f}",
                                "p": f"{precision:.2f}",
                                "r": f"{recall:.2f}",
                            }
                        )
                    progress.update(1)
        if progress is not None:
            progress.close()
        reduced = reduce_numeric_dict(counts, average=False)
        return self.summarize(reduced)

    def evaluate_batch(self, model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, float]:
        generate_inputs = {
            "input_ids": batch["input_ids"].to(next(model.parameters()).device),
            "attention_mask": batch["attention_mask"].to(next(model.parameters()).device),
            "pixel_values": batch["pixel_values"].to(next(model.parameters()).device),
        }
        if batch.get("mm_token_type_ids") is not None:
            generate_inputs["mm_token_type_ids"] = batch["mm_token_type_ids"].to(next(model.parameters()).device)
        generate_inputs.update(
            build_generate_kwargs(
                self.tokenizer,
                generation_config=getattr(model, "generation_config", None),
                num_bins=self.num_bins,
                prompt_lengths=batch["prompt_lengths"].tolist(),
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                use_cache=self.use_cache,
            )
        )
        input_context_length = int(generate_inputs["input_ids"].shape[1])
        if batch.get("image_grid_thw") is not None:
            generate_inputs["image_grid_thw"] = batch["image_grid_thw"].to(next(model.parameters()).device)
        reset_model_runtime_state(model)
        generated = model.generate(**generate_inputs)
        counts = self._empty_counts()
        eos_token_id = generate_inputs.get("eos_token_id")

        for row_index, _prompt_length in enumerate(batch["prompt_lengths"].tolist()):
            counts["samples"] += 1.0
            task_type = batch["meta"]["task_type"][row_index]
            domain_type = batch["meta"]["domain_type"][row_index]
            adapter = get_adapter(
                task_type=task_type,
                domain_type=domain_type,
                num_bins=self.num_bins,
            )
            generated_ids = generated[row_index, input_context_length:]
            trimmed_ids = trim_generated_ids_at_eos(generated_ids, eos_token_id)
            decoded_text = self.tokenizer.decode(trimmed_ids, skip_special_tokens=False)
            strict_text = self.tokenizer.decode(trimmed_ids, skip_special_tokens=True)
            image_width = int(batch["meta"]["image_width"][row_index])
            image_height = int(batch["meta"]["image_height"][row_index])
            gt_struct = batch["meta"]["gt_struct"][row_index]
            parse_error_lenient = None
            parse_error_strict = None
            pred_struct = None
            counts[adapter.task_bucket_key] += 1.0
            try:
                pred_struct = adapter.decode(
                    decoded_text,
                    image_width=image_width,
                    image_height=image_height,
                )
                counts["parse_success_lenient"] += 1.0
            except Exception as exc:  # noqa: BLE001
                pred_struct = adapter.empty_prediction()
                parse_error_lenient = str(exc)
            if parse_error_lenient is None:
                try:
                    adapter.decode(
                        strict_text,
                        image_width=image_width,
                        image_height=image_height,
                        strict=True,
                    )
                    counts["parse_success_strict"] += 1.0
                except Exception as exc:  # noqa: BLE001
                    parse_error_strict = str(exc)
            else:
                parse_error_strict = parse_error_lenient
            local_counts = adapter.score_prediction(
                gt_struct,
                pred_struct,
                bbox_iou_threshold=self.bbox_iou_threshold,
                strict_point_distance_px=self.strict_point_distance_px,
            )
            for key, value in local_counts.items():
                counts[key] += value
        return counts

    def summarize(self, counts: dict[str, float]) -> dict[str, float]:
        if counts["stage2_samples"] > 0 and counts["structured_samples"] == 0 and counts["grounding_samples"] == 0:
            samples = max(counts["samples"], 1.0)
            point_count = max(counts["point_count"], 1.0)
            gt_instances = max(counts["gt_instances"], 1.0)
            matched = max(counts["gt_instances"], 1.0)
            return {
                "val/parse_rate_lenient": counts["parse_success_lenient"] / samples,
                "val/parse_rate_strict": counts["parse_success_strict"] / samples,
                "val/keypoint_l2_mean": counts["point_distance_sum"] / point_count,
                "val/keypoint_count_acc": counts["keypoint_count_exact"] / matched,
                "val/end_to_end_score": counts["end_to_end_correct"] / gt_instances,
            }
        if counts["grounding_samples"] > 0 and counts["structured_samples"] == 0 and counts["stage2_samples"] == 0:
            samples = max(counts["samples"], 1.0)
            tp = counts["bbox_tp"]
            fp = counts["bbox_fp"]
            fn = counts["bbox_fn"]
            matched = max(tp, 1.0)
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(tp + fn, 1.0)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            return {
                "val/parse_rate_lenient": counts["parse_success_lenient"] / samples,
                "val/parse_rate_strict": counts["parse_success_strict"] / samples,
                "val/bbox_precision_at_iou50": precision,
                "val/bbox_f1_at_iou50": f1,
                "val/bbox_recall_at_iou50": recall,
                "val/bbox_iou_mean": counts["bbox_iou_sum"] / matched,
            }
        samples = max(counts["samples"], 1.0)
        tp = counts["bbox_tp"]
        fp = counts["bbox_fp"]
        fn = counts["bbox_fn"]
        matched = max(tp, 1.0)
        point_count = max(counts["point_count"], 1.0)
        gt_instances = max(counts["gt_instances"], 1.0)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return {
            "val/parse_rate_lenient": counts["parse_success_lenient"] / samples,
            "val/parse_rate_strict": counts["parse_success_strict"] / samples,
            "val/bbox_precision_at_iou50": precision,
            "val/bbox_f1_at_iou50": f1,
            "val/bbox_recall_at_iou50": recall,
            "val/bbox_iou_mean": counts["bbox_iou_sum"] / matched,
            "val/keypoint_l2_mean": counts["point_distance_sum"] / point_count,
            "val/keypoint_count_acc": counts["keypoint_count_exact"] / matched,
            "val/end_to_end_score": counts["end_to_end_correct"] / gt_instances,
        }

    def _empty_counts(self) -> dict[str, float]:
        return {
            "samples": 0.0,
            "parse_success_lenient": 0.0,
            "parse_success_strict": 0.0,
            "structured_samples": 0.0,
            "grounding_samples": 0.0,
            "stage2_samples": 0.0,
            "gt_instances": 0.0,
            "pred_instances": 0.0,
            "bbox_tp": 0.0,
            "bbox_fp": 0.0,
            "bbox_fn": 0.0,
            "bbox_iou_sum": 0.0,
            "point_distance_sum": 0.0,
            "point_count": 0.0,
            "keypoint_count_exact": 0.0,
            "end_to_end_correct": 0.0,
        }
