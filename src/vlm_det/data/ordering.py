from __future__ import annotations

from typing import Any


def sort_instances_canonical(instances: list[Any]) -> list[Any]:
    return sorted(instances, key=canonical_instance_sort_key)


def canonical_instance_sort_key(instance: Any) -> tuple[float, ...]:
    bbox = _get_instance_value(instance, "bbox")
    keypoints = _get_instance_value(instance, "keypoints")

    x1 = float(bbox[0])
    y1 = float(bbox[1])
    x2 = float(bbox[2])
    y2 = float(bbox[3])

    if keypoints:
        tail_x, tail_y = _point_xy(keypoints[0])
        head_x, head_y = _point_xy(keypoints[-1])
    else:
        tail_x = tail_y = head_x = head_y = float("inf")

    return (
        y1,
        x1,
        y2,
        x2,
        tail_y,
        tail_x,
        head_y,
        head_x,
        float(len(keypoints)),
    )


def _get_instance_value(instance: Any, field_name: str) -> Any:
    if isinstance(instance, dict):
        return instance[field_name]
    return getattr(instance, field_name)


def _point_xy(point: Any) -> tuple[float, float]:
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"])
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x), float(point.y)
    return float(point[0]), float(point[1])
