from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw


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


def draw_prediction(image: Image.Image, prediction: dict[str, Any]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for index, instance in enumerate(prediction.get("instances", [])):
        color = PALETTE[index % len(PALETTE)]
        bbox = [float(value) for value in instance.get("bbox", [])]
        if len(bbox) == 4:
            draw.rectangle(bbox, outline=color, width=3)
        keypoints = instance.get("keypoints", [])
        xy_points = [(float(point[0]), float(point[1])) for point in keypoints]
        if len(xy_points) >= 2:
            draw.line(xy_points, fill=color, width=3)
        for point_index, point in enumerate(keypoints):
            x = float(point[0])
            y = float(point[1])
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=color, width=2)
            if point_index == 0:
                draw.text((x + 6, y - 12), "S", fill=color)
            elif point_index == len(keypoints) - 1:
                draw.text((x + 6, y - 12), "H", fill=color)
        if len(bbox) == 4:
            draw.text((bbox[0] + 4, bbox[1] + 4), f"arrow {index + 1}", fill=color)
    return canvas


def format_prediction_summary(prediction: dict[str, Any]) -> str:
    instances = prediction.get("instances", [])
    point_count = sum(len(instance.get("keypoints", [])) for instance in instances)
    return "\n".join(
        [
            f"Detected arrows: {len(instances)}",
            f"Total keypoints: {point_count}",
        ]
    )
