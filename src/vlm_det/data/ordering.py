from __future__ import annotations

from typing import Any


def sort_instances_canonical(instances: list[Any]) -> list[Any]:
    return sorted(instances, key=canonical_instance_sort_key)


def sort_grounding_instances_canonical(instances: list[Any]) -> list[Any]:
    return sorted(instances, key=grounding_instance_sort_key)


def normalize_instance_keypoint_order(instance: Any) -> Any:
    label = _get_instance_value(instance, "label")
    keypoints = _get_instance_value(instance, "keypoints")
    normalized = normalize_keypoints_for_label(label, keypoints)
    _set_instance_value(instance, "keypoints", normalized)
    return instance


def normalize_keypoints_for_label(label: str, keypoints: list[Any]) -> list[Any]:
    if str(label) != "double_arrow" or len(keypoints) < 2:
        return list(keypoints)
    first_x, first_y = _point_xy(keypoints[0])
    last_x, last_y = _point_xy(keypoints[-1])
    # Double-arrow endpoints are canonicalized so the upper-left head comes
    # first. If x ties, use y as a stable top-to-bottom tie-breaker.
    if (first_x, first_y) <= (last_x, last_y):
        return list(keypoints)
    return list(reversed(keypoints))


def canonical_instance_sort_key(instance: Any) -> tuple[float, ...]:
    bbox = _get_instance_value(instance, "bbox")
    keypoints = _get_instance_value(instance, "keypoints")

    x1 = float(bbox[0])
    y1 = float(bbox[1])
    x2 = float(bbox[2])
    y2 = float(bbox[3])

    tail_x, tail_y = _point_xy(keypoints[0])
    head_x, head_y = _point_xy(keypoints[-1])
    n_points = float(len(keypoints))

    return (
        y1,
        x1,
        y2,
        x2,
        tail_y,
        tail_x,
        head_y,
        head_x,
        n_points,
    )


def grounding_instance_sort_key(instance: Any) -> tuple[float, ...]:
    bbox = _get_instance_value(instance, "bbox")
    label = str(_get_instance_value(instance, "label"))
    x1 = float(bbox[0])
    y1 = float(bbox[1])
    x2 = float(bbox[2])
    y2 = float(bbox[3])
    return (
        y1,
        x1,
        y2,
        x2,
        label,
    )


def _get_instance_value(instance: Any, field_name: str) -> Any:
    if isinstance(instance, dict):
        return instance[field_name]
    return getattr(instance, field_name)


def _set_instance_value(instance: Any, field_name: str, value: Any) -> None:
    if isinstance(instance, dict):
        instance[field_name] = value
        return
    setattr(instance, field_name, value)


def _point_xy(point: Any) -> tuple[float, float]:
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"])
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x), float(point.y)
    return float(point[0]), float(point[1])
