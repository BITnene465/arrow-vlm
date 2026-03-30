from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from vlm_structgen.domains.arrow.codecs.keypoint_sequence import KeypointSequenceCodec
from vlm_structgen.domains.arrow.task_support import BaseArrowAdapter, empty_counts


@dataclass
class ArrowKeypointSequenceAdapter(BaseArrowAdapter):
    task_type: str = field(init=False, default="keypoint_sequence")
    task_bucket_key: str = field(init=False, default="stage2_samples")

    def build_gt_struct_from_record(self, record: dict[str, Any]) -> dict[str, Any]:
        instances = record.get("instances", [])
        if len(instances) != 1:
            raise ValueError("keypoint_sequence samples must contain exactly one instance.")
        instance = instances[0]
        return {
            "label": instance["label"],
            "keypoints": instance["keypoints"],
        }

    def encode_target_text(self, gt_struct: dict[str, Any], *, image_width: int, image_height: int) -> str:
        return self.codec.encode(gt_struct.get("keypoints", []), image_width=image_width, image_height=image_height)

    def decode_with_meta(
        self,
        text: str,
        *,
        image_width: int,
        image_height: int,
        strict: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.codec.decode_with_meta(text, image_width=image_width, image_height=image_height, strict=strict)

    def decode(self, text: str, *, image_width: int, image_height: int, strict: bool = False) -> dict[str, Any]:
        return self.codec.decode(text, image_width=image_width, image_height=image_height, strict=strict)

    def empty_prediction(self) -> dict[str, Any]:
        return {"keypoints": [], "keypoints_2d": []}

    def score_prediction(
        self,
        gt_struct: dict[str, Any],
        pred_struct: dict[str, Any],
        *,
        bbox_iou_threshold: float,
        strict_point_distance_px: float,
    ) -> dict[str, float]:
        del bbox_iou_threshold
        counts = empty_counts()
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
            if distance > strict_point_distance_px:
                all_points_strict = False
        if point_limit != len(gt_points) or point_limit != len(pred_points):
            all_points_strict = False
        if all_points_strict:
            counts["end_to_end_correct"] = 1.0
        return counts


def build_keypoint_sequence_adapter(*, domain_type: str, num_bins: int, task_options: dict[str, Any] | None = None):
    del task_options
    if domain_type == "arrow":
        return ArrowKeypointSequenceAdapter(codec=KeypointSequenceCodec(num_bins=num_bins))
    raise ValueError(f"Unsupported keypoint_sequence domain_type: {domain_type!r}")
