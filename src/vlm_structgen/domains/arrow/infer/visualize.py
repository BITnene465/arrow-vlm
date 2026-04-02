from __future__ import annotations

import math
from typing import Any

from PIL import Image, ImageDraw

from vlm_structgen.core.utils.logging import get_vlm_logger


PALETTE = [
    "#e63946",
    "#1d3557",
    "#2a9d8f",
    "#f4a261",
    "#6a4c93",
    "#d62828",
    "#0077b6",
    "#588157",
]

LOGGER = get_vlm_logger()


def _normalize_bbox(raw_bbox: list[Any]) -> tuple[float, float, float, float] | None:
    if len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in raw_bbox]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def draw_prediction(image: Image.Image, prediction: dict[str, Any]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for index, instance in enumerate(prediction.get("instances", [])):
        color = PALETTE[index % len(PALETTE)]
        label = str(instance.get("label", "unknown"))
        stage2_status = str(instance.get("stage2_status", ""))
        raw_bbox = list(instance.get("bbox", []))
        bbox = _normalize_bbox(raw_bbox)
        outline_color = "#808080" if stage2_status == "failed" else color
        if bbox is not None:
            draw.rectangle(bbox, outline=outline_color, width=3)
        else:
            LOGGER.warning(
                "draw_prediction skip invalid bbox: index=%s label=%s bbox=%s",
                index,
                label,
                raw_bbox,
            )
        keypoints = instance.get("keypoints", [])
        xy_points = []
        invalid_point_count = 0
        for point in keypoints:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                invalid_point_count += 1
                continue
            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError):
                invalid_point_count += 1
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                invalid_point_count += 1
                continue
            xy_points.append((x, y))
        if invalid_point_count > 0:
            LOGGER.warning(
                "draw_prediction skipped invalid keypoints: index=%s label=%s invalid=%s total=%s",
                index,
                label,
                invalid_point_count,
                len(keypoints),
            )
        if len(xy_points) >= 2:
            draw.line(xy_points, fill=color, width=3)
        for point_index, (x, y) in enumerate(xy_points):
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=color, width=2)
            if point_index == 0:
                draw.text((x + 6, y - 12), "K0", fill=color)
            elif point_index == len(xy_points) - 1:
                draw.text((x + 6, y - 12), "K-1", fill=color)
        if bbox is not None:
            suffix = " [S2 fail]" if stage2_status == "failed" else ""
            draw.text((bbox[0] + 4, bbox[1] + 4), f"{label} {index + 1}{suffix}", fill=outline_color)
    return canvas


def format_prediction_summary(prediction: dict[str, Any]) -> str:
    instances = prediction.get("instances", [])
    point_count = sum(len(instance.get("keypoints", [])) for instance in instances)
    single_count = sum(1 for instance in instances if instance.get("label") == "single_arrow")
    double_count = sum(1 for instance in instances if instance.get("label") == "double_arrow")
    stage2_failed_count = sum(1 for instance in instances if instance.get("stage2_status") == "failed")
    return "\n".join(
        [
            f"Detected arrows: {len(instances)}",
            f"Single arrows: {single_count}",
            f"Double arrows: {double_count}",
            f"Total keypoints: {point_count}",
            f"Stage2 failed boxes: {stage2_failed_count}",
        ]
    )
