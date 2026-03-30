from __future__ import annotations

from functools import lru_cache
from typing import Protocol


class TaskAdapter(Protocol):
    task_type: str
    domain_type: str
    num_bins: int
    task_bucket_key: str

    def build_gt_struct_from_record(self, record: dict) -> dict:
        ...

    def encode_target_text(self, gt_struct: dict, *, image_width: int, image_height: int) -> str:
        ...

    def decode(self, text: str, *, image_width: int, image_height: int, strict: bool = False) -> dict:
        ...

    def decode_with_meta(
        self,
        text: str,
        *,
        image_width: int,
        image_height: int,
        strict: bool = False,
    ) -> tuple[dict, dict]:
        ...

    def empty_prediction(self) -> dict:
        ...

    def score_prediction(
        self,
        gt_struct: dict,
        pred_struct: dict,
        *,
        bbox_iou_threshold: float,
        strict_point_distance_px: float,
    ) -> dict[str, float]:
        ...


def normalize_task_type(task_type: str | None) -> str:
    normalized = (task_type or "").strip().lower()
    if normalized in {"", "one_stage", "arrow_structure", "joint_structure"}:
        return "joint_structure"
    if normalized in {"grounding", "two_stage_stage1_grounding"}:
        return "grounding"
    if normalized in {"keypoint_sequence", "two_stage_stage2"}:
        return "keypoint_sequence"
    return normalized


def normalize_domain_type(domain_type: str | None) -> str:
    normalized = (domain_type or "").strip().lower()
    if not normalized:
        return "arrow"
    return normalized


@lru_cache(maxsize=32)
def get_adapter(*, task_type: str | None, domain_type: str | None, num_bins: int) -> TaskAdapter:
    normalized_task_type = normalize_task_type(task_type)
    normalized_domain_type = normalize_domain_type(domain_type)
    if normalized_domain_type == "arrow":
        from vlm_det.domains.arrow.adapters import build_arrow_adapter

        return build_arrow_adapter(task_type=normalized_task_type, num_bins=num_bins)
    raise ValueError(
        f"Unsupported task/domain combination: task_type={normalized_task_type!r}, "
        f"domain_type={normalized_domain_type!r}"
    )

