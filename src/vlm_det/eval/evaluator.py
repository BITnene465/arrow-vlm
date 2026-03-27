from __future__ import annotations

import math
from typing import Any

import torch

from vlm_det.protocol.codec import ArrowCodec
from vlm_det.protocol.grounding_codec import GroundingCodec
from vlm_det.protocol.keypoint_codec import KeypointSequenceCodec
from vlm_det.utils.distributed import reduce_numeric_dict, reset_model_runtime_state, unwrap_model
from vlm_det.utils.generation import (
    build_generate_kwargs,
    build_json_array_stopping_criteria,
    trim_generated_ids_at_eos,
)
from vlm_det.utils.logging import create_progress_bar


class ArrowEvaluator:
    def __init__(
        self,
        codec: ArrowCodec,
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
        self.codec = codec
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
        self.keypoint_codec = KeypointSequenceCodec(num_bins=codec.num_bins)
        self.grounding_codec = GroundingCodec(num_bins=codec.num_bins)

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
                num_bins=self.codec.num_bins,
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
        generate_inputs["stopping_criteria"] = build_json_array_stopping_criteria(
            self.tokenizer,
            prompt_lengths=[input_context_length] * int(generate_inputs["input_ids"].shape[0]),
        )
        if batch.get("image_grid_thw") is not None:
            generate_inputs["image_grid_thw"] = batch["image_grid_thw"].to(next(model.parameters()).device)
        reset_model_runtime_state(model)
        generated = model.generate(**generate_inputs)
        counts = self._empty_counts()
        eos_token_id = generate_inputs.get("eos_token_id")

        for row_index, _prompt_length in enumerate(batch["prompt_lengths"].tolist()):
            counts["samples"] += 1.0
            task_type = batch["meta"]["task_type"][row_index]
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
            if task_type == "two_stage_stage2":
                counts["stage2_samples"] += 1.0
                try:
                    pred_struct = self.keypoint_codec.decode(
                        decoded_text,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    counts["parse_success_lenient"] += 1.0
                except Exception as exc:  # noqa: BLE001
                    pred_struct = {"keypoints": []}
                    parse_error_lenient = str(exc)
            elif task_type == "two_stage_stage1_grounding":
                counts["grounding_samples"] += 1.0
                try:
                    pred_struct = self.grounding_codec.decode(
                        decoded_text,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    counts["parse_success_lenient"] += 1.0
                except Exception as exc:  # noqa: BLE001
                    pred_struct = {"instances": []}
                    parse_error_lenient = str(exc)
            else:
                counts["structured_samples"] += 1.0
                try:
                    pred_struct = self.codec.decode(decoded_text, image_width=image_width, image_height=image_height)
                    counts["parse_success_lenient"] += 1.0
                except Exception as exc:  # noqa: BLE001
                    pred_struct = {"instances": []}
                    parse_error_lenient = str(exc)
            if parse_error_lenient is None:
                if task_type == "two_stage_stage2":
                    try:
                        self.keypoint_codec.decode(
                            strict_text,
                            image_width=image_width,
                            image_height=image_height,
                            strict=True,
                        )
                        counts["parse_success_strict"] += 1.0
                    except Exception as exc:  # noqa: BLE001
                        parse_error_strict = str(exc)
                elif task_type == "two_stage_stage1_grounding":
                    try:
                        self.grounding_codec.decode(
                            strict_text,
                            image_width=image_width,
                            image_height=image_height,
                            strict=True,
                        )
                        counts["parse_success_strict"] += 1.0
                    except Exception as exc:  # noqa: BLE001
                        parse_error_strict = str(exc)
                else:
                    try:
                        self.codec.decode(
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
            if task_type == "two_stage_stage2":
                local_counts = self._score_stage2_prediction(gt_struct, pred_struct)
            elif task_type == "two_stage_stage1_grounding":
                local_counts = self._score_grounding_prediction(gt_struct, pred_struct)
            else:
                local_counts = self._score_prediction(gt_struct, pred_struct)
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

    def _score_prediction(self, gt_struct: dict[str, Any], pred_struct: dict[str, Any]) -> dict[str, float]:
        counts = self._empty_counts()
        gt_instances = gt_struct.get("instances", [])
        pred_instances = pred_struct.get("instances", [])
        counts["gt_instances"] = float(len(gt_instances))
        counts["pred_instances"] = float(len(pred_instances))

        matches = self._match_instances(gt_instances, pred_instances)
        matched_gt = set()
        matched_pred = set()
        for gt_index, pred_index, iou_value in matches:
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
            counts["bbox_tp"] += 1.0
            counts["bbox_iou_sum"] += iou_value

            gt_instance = gt_instances[gt_index]
            pred_instance = pred_instances[pred_index]
            gt_points = gt_instance["keypoints"]
            pred_points = pred_instance["keypoints"]
            if len(gt_points) == len(pred_points):
                counts["keypoint_count_exact"] += 1.0
            point_limit = min(len(gt_points), len(pred_points))
            all_points_strict = len(gt_points) == len(pred_points)
            for point_index in range(point_limit):
                gx, gy = gt_points[point_index][:2]
                px, py = pred_points[point_index][:2]
                distance = math.dist((gx, gy), (px, py))
                counts["point_distance_sum"] += distance
                counts["point_count"] += 1.0
                if distance > self.strict_point_distance_px:
                    all_points_strict = False
            if point_limit != len(gt_points) or point_limit != len(pred_points):
                all_points_strict = False
            if all_points_strict and iou_value >= self.bbox_iou_threshold:
                counts["end_to_end_correct"] += 1.0

        counts["bbox_fp"] = float(len(pred_instances) - len(matched_pred))
        counts["bbox_fn"] = float(len(gt_instances) - len(matched_gt))
        return counts

    def _score_stage2_prediction(self, gt_struct: dict[str, Any], pred_struct: dict[str, Any]) -> dict[str, float]:
        counts = self._empty_counts()
        gt_points = gt_struct.get("keypoints", [])
        pred_points = pred_struct.get("keypoints", [])
        counts["gt_instances"] = 1.0
        counts["pred_instances"] = 1.0 if pred_points else 0.0
        if len(gt_points) == len(pred_points):
            counts["keypoint_count_exact"] = 1.0
        point_limit = min(len(gt_points), len(pred_points))
        all_points_strict = len(gt_points) == len(pred_points)
        for point_index in range(point_limit):
            gx, gy = gt_points[point_index][:2]
            px, py = pred_points[point_index][:2]
            distance = math.dist((gx, gy), (px, py))
            counts["point_distance_sum"] += distance
            counts["point_count"] += 1.0
            if distance > self.strict_point_distance_px:
                all_points_strict = False
        if point_limit != len(gt_points) or point_limit != len(pred_points):
            all_points_strict = False
        if all_points_strict:
            counts["end_to_end_correct"] = 1.0
        return counts

    def _score_grounding_prediction(self, gt_struct: dict[str, Any], pred_struct: dict[str, Any]) -> dict[str, float]:
        counts = self._empty_counts()
        gt_instances = gt_struct.get("instances", [])
        pred_instances = pred_struct.get("instances", [])
        counts["gt_instances"] = float(len(gt_instances))
        counts["pred_instances"] = float(len(pred_instances))

        matches = self._match_instances(gt_instances, pred_instances)
        matched_gt = set()
        matched_pred = set()
        for gt_index, pred_index, iou_value in matches:
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
            counts["bbox_tp"] += 1.0
            counts["bbox_iou_sum"] += iou_value

        counts["bbox_fp"] = float(len(pred_instances) - len(matched_pred))
        counts["bbox_fn"] = float(len(gt_instances) - len(matched_gt))
        return counts

    def _match_instances(
        self,
        gt_instances: list[dict[str, Any]],
        pred_instances: list[dict[str, Any]],
    ) -> list[tuple[int, int, float]]:
        adjacency: list[list[int]] = [[] for _ in gt_instances]
        iou_by_pair: dict[tuple[int, int], float] = {}
        for gt_index, gt_instance in enumerate(gt_instances):
            row: list[tuple[int, float]] = []
            for pred_index, pred_instance in enumerate(pred_instances):
                if gt_instance.get("label") != pred_instance.get("label"):
                    continue
                iou_value = self._bbox_iou(gt_instance["bbox"], pred_instance["bbox"])
                if iou_value >= self.bbox_iou_threshold:
                    row.append((pred_index, iou_value))
                    iou_by_pair[(gt_index, pred_index)] = iou_value
            row.sort(key=lambda item: (-item[1], item[0]))
            adjacency[gt_index] = [pred_index for pred_index, _iou_value in row]

        matches = self._maximum_bipartite_matching(adjacency, len(pred_instances))
        return [
            (gt_index, pred_index, iou_by_pair[(gt_index, pred_index)])
            for gt_index, pred_index in matches
        ]

    @staticmethod
    def _maximum_bipartite_matching(
        adjacency: list[list[int]],
        num_right_nodes: int,
    ) -> list[tuple[int, int]]:
        num_left_nodes = len(adjacency)
        if num_left_nodes == 0 or num_right_nodes == 0:
            return []

        pair_left = [-1] * num_left_nodes
        pair_right = [-1] * num_right_nodes
        dist = [0] * num_left_nodes

        def bfs() -> bool:
            queue: list[int] = []
            found_augmenting = False
            for left_index in range(num_left_nodes):
                if pair_left[left_index] == -1:
                    dist[left_index] = 0
                    queue.append(left_index)
                else:
                    dist[left_index] = -1

            queue_index = 0
            while queue_index < len(queue):
                left_index = queue[queue_index]
                queue_index += 1
                for right_index in adjacency[left_index]:
                    matched_left = pair_right[right_index]
                    if matched_left == -1:
                        found_augmenting = True
                    elif dist[matched_left] == -1:
                        dist[matched_left] = dist[left_index] + 1
                        queue.append(matched_left)
            return found_augmenting

        def dfs(left_index: int) -> bool:
            for right_index in adjacency[left_index]:
                matched_left = pair_right[right_index]
                if matched_left == -1 or (
                    dist[matched_left] == dist[left_index] + 1 and dfs(matched_left)
                ):
                    pair_left[left_index] = right_index
                    pair_right[right_index] = left_index
                    return True
            dist[left_index] = -1
            return False

        while bfs():
            for left_index in range(num_left_nodes):
                if pair_left[left_index] == -1:
                    dfs(left_index)

        return [
            (left_index, right_index)
            for left_index, right_index in enumerate(pair_left)
            if right_index != -1
        ]

    @staticmethod
    def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union
