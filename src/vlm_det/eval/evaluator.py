from __future__ import annotations

import math
from typing import Any

import torch

from vlm_det.protocol.codec import ArrowCodec
from vlm_det.utils.distributed import reduce_numeric_dict, unwrap_model
from vlm_det.utils.generation import build_generate_kwargs
from vlm_det.utils.logging import create_progress_bar


class ArrowEvaluator:
    def __init__(
        self,
        codec: ArrowCodec,
        tokenizer,
        max_new_tokens: int,
        num_beams: int = 1,
        do_sample: bool = False,
        use_cache: bool = True,
        preview_samples: int = 0,
        preview_char_limit: int = 600,
        bbox_iou_threshold: float = 0.5,
        strict_point_distance_px: float = 8.0,
    ) -> None:
        self.codec = codec
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.use_cache = use_cache
        self.preview_samples = max(int(preview_samples), 0)
        self.preview_char_limit = max(int(preview_char_limit), 120)
        self.bbox_iou_threshold = bbox_iou_threshold
        self.strict_point_distance_px = strict_point_distance_px
        self.last_previews: list[dict[str, Any]] = []

    def evaluate_model(self, model: torch.nn.Module, dataloader) -> dict[str, float]:
        counts = self._empty_counts()
        previews: list[dict[str, Any]] = []
        raw_model = unwrap_model(model)
        raw_model.eval()
        progress = create_progress_bar(total=len(dataloader), desc="eval", leave=True)
        with torch.no_grad():
            for batch in dataloader:
                batch_counts, batch_previews = self.evaluate_batch(raw_model, batch)
                for key, value in batch_counts.items():
                    counts[key] += value
                if self.preview_samples > 0 and len(previews) < self.preview_samples:
                    remaining = self.preview_samples - len(previews)
                    previews.extend(batch_previews[:remaining])
                if progress is not None:
                    samples = max(counts["samples"], 1.0)
                    parse_rate = counts["parse_success"] / samples
                    precision = counts["bbox_tp"] / max(counts["bbox_tp"] + counts["bbox_fp"], 1.0)
                    recall = counts["bbox_tp"] / max(counts["bbox_tp"] + counts["bbox_fn"], 1.0)
                    progress.set_postfix(
                        {
                            "parse": f"{parse_rate:.2f}",
                            "p": f"{precision:.2f}",
                            "r": f"{recall:.2f}",
                        }
                    )
                    progress.update(1)
        if progress is not None:
            progress.close()
        self.last_previews = previews
        reduced = reduce_numeric_dict(counts, average=False)
        return self.summarize(reduced)

    def evaluate_batch(self, model: torch.nn.Module, batch: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
        generate_inputs = {
            "input_ids": batch["input_ids"].to(next(model.parameters()).device),
            "attention_mask": batch["attention_mask"].to(next(model.parameters()).device),
            "pixel_values": batch["pixel_values"].to(next(model.parameters()).device),
        }
        generate_inputs.update(
            build_generate_kwargs(
                self.tokenizer,
                num_bins=self.codec.num_bins,
                prompt_lengths=batch["prompt_lengths"].tolist(),
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=self.do_sample,
                use_cache=self.use_cache,
            )
        )
        if batch.get("image_grid_thw") is not None:
            generate_inputs["image_grid_thw"] = batch["image_grid_thw"].to(next(model.parameters()).device)
        generated = model.generate(**generate_inputs)
        counts = self._empty_counts()
        previews: list[dict[str, Any]] = []
        input_context_length = int(batch["input_ids"].shape[1])

        for row_index, _prompt_length in enumerate(batch["prompt_lengths"].tolist()):
            counts["samples"] += 1.0
            generated_ids = generated[row_index, input_context_length:]
            decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            image_width = int(batch["meta"]["image_width"][row_index])
            image_height = int(batch["meta"]["image_height"][row_index])
            gt_struct = batch["meta"]["gt_struct"][row_index]
            parse_error = None
            try:
                pred_struct = self.codec.decode(decoded_text, image_width=image_width, image_height=image_height)
                counts["parse_success"] += 1.0
            except Exception as exc:  # noqa: BLE001
                pred_struct = {"instances": []}
                parse_error = str(exc)
            local_counts = self._score_prediction(gt_struct, pred_struct)
            for key, value in local_counts.items():
                counts[key] += value
            if self.preview_samples > 0 and len(previews) < self.preview_samples:
                previews.append(
                    {
                        "sample_id": batch["meta"]["sample_id"][row_index],
                        "parse_ok": parse_error is None,
                        "parse_error": parse_error,
                        "gt_instances": len(gt_struct.get("instances", [])),
                        "pred_instances": len(pred_struct.get("instances", [])),
                        "target_text": self._truncate_preview(batch["meta"]["target_text"][row_index]),
                        "decoded_text": self._truncate_preview(decoded_text),
                    }
                )
        return counts, previews

    def summarize(self, counts: dict[str, float]) -> dict[str, float]:
        samples = max(counts["samples"], 1.0)
        tp = counts["bbox_tp"]
        fp = counts["bbox_fp"]
        fn = counts["bbox_fn"]
        matched = max(tp, 1.0)
        point_count = max(counts["point_count"], 1.0)
        visibility_count = max(counts["visibility_count"], 1.0)
        gt_instances = max(counts["gt_instances"], 1.0)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return {
            "val/parse_rate": counts["parse_success"] / samples,
            "val/bbox_precision_at_iou50": precision,
            "val/bbox_f1_at_iou50": f1,
            "val/bbox_recall_at_iou50": recall,
            "val/bbox_iou_mean": counts["bbox_iou_sum"] / matched,
            "val/keypoint_l2_mean": counts["point_distance_sum"] / point_count,
            "val/keypoint_count_acc": counts["keypoint_count_exact"] / matched,
            "val/visibility_acc": counts["visibility_correct"] / visibility_count,
            "val/end_to_end_score": counts["end_to_end_correct"] / gt_instances,
        }

    def _empty_counts(self) -> dict[str, float]:
        return {
            "samples": 0.0,
            "parse_success": 0.0,
            "gt_instances": 0.0,
            "pred_instances": 0.0,
            "bbox_tp": 0.0,
            "bbox_fp": 0.0,
            "bbox_fn": 0.0,
            "bbox_iou_sum": 0.0,
            "point_distance_sum": 0.0,
            "point_count": 0.0,
            "visibility_correct": 0.0,
            "visibility_count": 0.0,
            "keypoint_count_exact": 0.0,
            "end_to_end_correct": 0.0,
        }

    def format_previews(self) -> list[str]:
        lines: list[str] = []
        for preview in self.last_previews:
            header = (
                f"[eval-preview] sample={preview['sample_id']} "
                f"parse_ok={preview['parse_ok']} "
                f"gt={preview['gt_instances']} pred={preview['pred_instances']}"
            )
            lines.append(header)
            if preview["parse_error"]:
                lines.append(f"[eval-preview] parse_error={preview['parse_error']}")
            lines.append(f"[eval-preview] target={preview['target_text']}")
            lines.append(f"[eval-preview] output={preview['decoded_text']}")
        return lines

    def _truncate_preview(self, text: str) -> str:
        normalized = text.replace("\n", "\\n")
        if len(normalized) <= self.preview_char_limit:
            return normalized
        return normalized[: self.preview_char_limit - 3] + "..."

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
                gx, gy, gv = gt_points[point_index]
                px, py, pv = pred_points[point_index]
                distance = math.dist((gx, gy), (px, py))
                counts["point_distance_sum"] += distance
                counts["point_count"] += 1.0
                counts["visibility_correct"] += float(gv == pv)
                counts["visibility_count"] += 1.0
                if gv != pv or distance > self.strict_point_distance_px:
                    all_points_strict = False
            if point_limit != len(gt_points) or point_limit != len(pred_points):
                all_points_strict = False
            if all_points_strict and iou_value >= self.bbox_iou_threshold:
                counts["end_to_end_correct"] += 1.0

        counts["bbox_fp"] = float(len(pred_instances) - len(matched_pred))
        counts["bbox_fn"] = float(len(gt_instances) - len(matched_gt))
        return counts

    def _match_instances(
        self,
        gt_instances: list[dict[str, Any]],
        pred_instances: list[dict[str, Any]],
    ) -> list[tuple[int, int, float]]:
        candidates: list[tuple[int, int, float]] = []
        for gt_index, gt_instance in enumerate(gt_instances):
            for pred_index, pred_instance in enumerate(pred_instances):
                iou_value = self._bbox_iou(gt_instance["bbox"], pred_instance["bbox"])
                if iou_value >= self.bbox_iou_threshold:
                    candidates.append((gt_index, pred_index, iou_value))
        candidates.sort(key=lambda item: item[2], reverse=True)
        matches: list[tuple[int, int, float]] = []
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        for gt_index, pred_index, iou_value in candidates:
            if gt_index in used_gt or pred_index in used_pred:
                continue
            used_gt.add(gt_index)
            used_pred.add(pred_index)
            matches.append((gt_index, pred_index, iou_value))
        return matches

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
