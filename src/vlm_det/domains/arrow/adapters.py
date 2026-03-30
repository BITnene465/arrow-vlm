from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from vlm_det.protocol.codec import ArrowCodec
from vlm_det.protocol.grounding_codec import GroundingCodec
from vlm_det.protocol.keypoint_codec import KeypointSequenceCodec


def _empty_counts() -> dict[str, float]:
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
    if union <= 0.0:
        return 0.0
    return inter_area / union


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
            if matched_left == -1 or (dist[matched_left] == dist[left_index] + 1 and dfs(matched_left)):
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


def _match_instances(
    gt_instances: list[dict[str, Any]],
    pred_instances: list[dict[str, Any]],
    *,
    bbox_iou_threshold: float,
) -> list[tuple[int, int, float]]:
    adjacency: list[list[int]] = [[] for _ in gt_instances]
    iou_by_pair: dict[tuple[int, int], float] = {}
    for gt_index, gt_instance in enumerate(gt_instances):
        row: list[tuple[int, float]] = []
        for pred_index, pred_instance in enumerate(pred_instances):
            if gt_instance.get("label") != pred_instance.get("label"):
                continue
            iou_value = _bbox_iou(gt_instance["bbox"], pred_instance["bbox"])
            if iou_value >= bbox_iou_threshold:
                row.append((pred_index, iou_value))
                iou_by_pair[(gt_index, pred_index)] = iou_value
        row.sort(key=lambda item: (-item[1], item[0]))
        adjacency[gt_index] = [pred_index for pred_index, _iou_value in row]

    matches = _maximum_bipartite_matching(adjacency, len(pred_instances))
    return [
        (gt_index, pred_index, iou_by_pair[(gt_index, pred_index)])
        for gt_index, pred_index in matches
    ]


@dataclass
class BaseArrowAdapter:
    codec: ArrowCodec | GroundingCodec | KeypointSequenceCodec
    task_type: str = field(init=False)
    task_bucket_key: str = field(init=False)
    domain_type: str = "arrow"

    @property
    def num_bins(self) -> int:
        return int(self.codec.num_bins)

    def empty_prediction(self) -> dict[str, Any]:
        return {"instances": []}


@dataclass
class ArrowJointStructureAdapter(BaseArrowAdapter):
    task_type: str = field(init=False, default="joint_structure")
    task_bucket_key: str = field(init=False, default="structured_samples")

    def build_gt_struct_from_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "instances": [
                {
                    "label": instance["label"],
                    "bbox": instance["bbox"],
                    "keypoints": instance["keypoints"],
                }
                for instance in record.get("instances", [])
            ]
        }

    def encode_target_text(self, gt_struct: dict[str, Any], *, image_width: int, image_height: int) -> str:
        return self.codec.encode(gt_struct, image_width=image_width, image_height=image_height)

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

    def score_prediction(
        self,
        gt_struct: dict[str, Any],
        pred_struct: dict[str, Any],
        *,
        bbox_iou_threshold: float,
        strict_point_distance_px: float,
    ) -> dict[str, float]:
        counts = _empty_counts()
        gt_instances = gt_struct.get("instances", [])
        pred_instances = pred_struct.get("instances", [])
        counts["gt_instances"] = float(len(gt_instances))
        counts["pred_instances"] = float(len(pred_instances))

        matches = _match_instances(gt_instances, pred_instances, bbox_iou_threshold=bbox_iou_threshold)
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
                if distance > strict_point_distance_px:
                    all_points_strict = False
            if point_limit != len(gt_points) or point_limit != len(pred_points):
                all_points_strict = False
            if all_points_strict and iou_value >= bbox_iou_threshold:
                counts["end_to_end_correct"] += 1.0

        counts["bbox_fp"] = float(len(pred_instances) - len(matched_pred))
        counts["bbox_fn"] = float(len(gt_instances) - len(matched_gt))
        return counts


@dataclass
class ArrowGroundingAdapter(BaseArrowAdapter):
    task_type: str = field(init=False, default="grounding")
    task_bucket_key: str = field(init=False, default="grounding_samples")

    def build_gt_struct_from_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "instances": [
                {
                    "label": instance["label"],
                    "bbox": instance["bbox"],
                    "keypoints": [],
                }
                for instance in record.get("instances", [])
            ]
        }

    def encode_target_text(self, gt_struct: dict[str, Any], *, image_width: int, image_height: int) -> str:
        return self.codec.encode(gt_struct, image_width=image_width, image_height=image_height)

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

    def score_prediction(
        self,
        gt_struct: dict[str, Any],
        pred_struct: dict[str, Any],
        *,
        bbox_iou_threshold: float,
        strict_point_distance_px: float,
    ) -> dict[str, float]:
        del strict_point_distance_px
        counts = _empty_counts()
        gt_instances = gt_struct.get("instances", [])
        pred_instances = pred_struct.get("instances", [])
        counts["gt_instances"] = float(len(gt_instances))
        counts["pred_instances"] = float(len(pred_instances))

        matches = _match_instances(gt_instances, pred_instances, bbox_iou_threshold=bbox_iou_threshold)
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
        counts = _empty_counts()
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


def build_arrow_adapter(*, task_type: str, num_bins: int):
    if task_type == "grounding":
        return ArrowGroundingAdapter(codec=GroundingCodec(num_bins=num_bins))
    if task_type == "keypoint_sequence":
        return ArrowKeypointSequenceAdapter(codec=KeypointSequenceCodec(num_bins=num_bins))
    if task_type == "joint_structure":
        return ArrowJointStructureAdapter(codec=ArrowCodec(num_bins=num_bins))
    raise ValueError(f"Unsupported arrow task_type: {task_type!r}")
