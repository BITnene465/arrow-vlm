from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from vlm_structgen.core.utils.logging import get_vlm_logger
from vlm_structgen.domains.arrow.codecs.grounding import GroundingCodec
from vlm_structgen.domains.arrow.codecs.keypoint_sequence import KeypointSequenceCodec
from vlm_structgen.domains.arrow.codecs.structure import ArrowCodec


def empty_counts() -> dict[str, float]:
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


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
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


def maximum_bipartite_matching(
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


def match_instances(
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
            iou_value = bbox_iou(gt_instance["bbox"], pred_instance["bbox"])
            if iou_value >= bbox_iou_threshold:
                row.append((pred_index, iou_value))
                iou_by_pair[(gt_index, pred_index)] = iou_value
        row.sort(key=lambda item: (-item[1], item[0]))
        adjacency[gt_index] = [pred_index for pred_index, _iou_value in row]

    matches = maximum_bipartite_matching(adjacency, len(pred_instances))
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
    _warned_flags: set[str] = field(default_factory=set, init=False, repr=False)

    def _warn_once(self, flag: str, message: str) -> None:
        if flag in self._warned_flags:
            return
        self._warned_flags.add(flag)
        get_vlm_logger().warning(message)

    @property
    def num_bins(self) -> int:
        return int(self.codec.num_bins)

    def empty_prediction(self) -> dict[str, Any]:
        return {"instances": []}

    def build_training_target(
        self,
        gt_struct: dict[str, Any],
        *,
        image_width: int,
        image_height: int,
    ) -> dict[str, Any]:
        return {
            "target_text": self.encode_target_text(
                gt_struct,
                image_width=image_width,
                image_height=image_height,
            ),
            "loss_meta": None,
        }

    def compute_loss(self, model_outputs, batch: dict[str, Any], *, tokenizer=None) -> object:
        del tokenizer
        del batch
        return model_outputs.loss

    def _compute_weighted_token_ce_loss(
        self,
        model_outputs,
        batch: dict[str, Any],
        *,
        tokenizer,
        token_weight_builder,
    ) -> object:
        if tokenizer is None:
            return model_outputs.loss

        logits = model_outputs.logits
        labels = batch["labels"].to(logits.device)
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_weights = self._build_shift_weights(
            batch=batch,
            labels=labels,
            tokenizer=tokenizer,
            device=logits.device,
            token_weight_builder=token_weight_builder,
        )
        if shift_weights is None:
            self._warn_once(
                "weighted_loss_fallback_shift_weights_none",
                "weighted token loss disabled for current adapter: failed to build shift weights; "
                f"fallback to model loss. route={self.task_type}/{self.domain_type}",
            )
            return model_outputs.loss

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        ).view_as(shift_labels)
        valid_mask = (shift_labels != -100).to(token_loss.dtype)
        weighted_loss = token_loss * shift_weights * valid_mask
        denom = (shift_weights * valid_mask).sum().clamp_min(1.0)
        return weighted_loss.sum() / denom

    def _build_shift_weights(
        self,
        *,
        batch: dict[str, Any],
        labels: torch.Tensor,
        tokenizer,
        device: torch.device,
        token_weight_builder,
    ) -> torch.Tensor | None:
        sequence_weights = torch.ones_like(labels, dtype=torch.float32, device=device)
        meta = batch.get("meta", {})
        target_texts = list(meta.get("target_text", []))
        loss_metas = list(meta.get("loss_meta", []))
        if len(target_texts) != labels.shape[0] or len(loss_metas) != labels.shape[0]:
            self._warn_once(
                "weighted_loss_meta_batch_mismatch",
                "weighted token loss disabled for current adapter: batch meta shape mismatch "
                f"(target_text={len(target_texts)}, loss_meta={len(loss_metas)}, batch={labels.shape[0]}). "
                f"route={self.task_type}/{self.domain_type}",
            )
            return None
        for row_index, (target_text, loss_meta) in enumerate(zip(target_texts, loss_metas, strict=True)):
            token_weights = token_weight_builder(
                str(target_text),
                loss_meta=loss_meta,
                tokenizer=tokenizer,
            )
            if token_weights is None:
                return None
            valid_positions = torch.nonzero(labels[row_index] != -100, as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue
            if len(token_weights) != int(valid_positions.numel()):
                if len(token_weights) + 1 == int(valid_positions.numel()):
                    token_weights = token_weights + [1.0]
                else:
                    limit = min(len(token_weights), int(valid_positions.numel()))
                    token_weights = token_weights[:limit]
                    valid_positions = valid_positions[:limit]
            if not token_weights:
                continue
            sequence_weights[row_index, valid_positions] = torch.tensor(
                token_weights,
                dtype=torch.float32,
                device=device,
            )
        return sequence_weights[:, 1:].contiguous()
