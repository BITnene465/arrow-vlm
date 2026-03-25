from __future__ import annotations

import math
import random
from io import BytesIO
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

from synthetic_pipeline.scene_sampler import clamp


def weighted_choice(rng: random.Random, mapping: dict[str, float]) -> str:
    keys = list(mapping.keys())
    weights = list(mapping.values())
    return rng.choices(keys, weights=weights, k=1)[0]


def arrow_bbox(points: list[list[float]], line_width: int, head_len: int, head_width: int) -> list[float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    pad = max(line_width, head_len, head_width) * 0.8
    return [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]


def default_arrow_style(width: int, height: int) -> dict[str, Any]:
    base = max(2, int(round(min(width, height) * 0.0045)))
    return {
        "line_width": base,
        "head_len": max(10, int(round(base * 4.5))),
        "head_width": max(8, int(round(base * 3.0))),
        "color": (28, 28, 28),
    }


def scene_style_config(style_cfg: dict[str, Any], scene_mode: str) -> dict[str, Any]:
    if scene_mode != "single_crop":
        return style_cfg
    adjusted = dict(style_cfg)
    adjusted["distractor_lines_range"] = [0, 5]
    adjusted["distractor_boxes_range"] = [0, 2]
    adjusted["text_like_strokes_range"] = [0, 4]
    adjusted["occluder_count_range"] = [0, 2]
    return adjusted


def make_background(width: int, height: int, style_cfg: dict[str, Any], rng: random.Random) -> Image.Image:
    mode = weighted_choice(rng, style_cfg["background_modes"])
    if mode == "plain_white":
        color = (255, 255, 255)
    elif mode == "gray":
        gray = rng.randint(220, 245)
        color = (gray, gray, gray)
    else:
        base = rng.randint(235, 250)
        tint = rng.randint(-8, 8)
        color = (base + tint, base + tint // 2, base - tint // 2)
    image = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(image)
    if mode == "paper":
        for _ in range(max(width * height // 12000, 50)):
            x_value = rng.randint(0, width - 1)
            y_value = rng.randint(0, height - 1)
            delta = rng.randint(-10, 10)
            pixel = tuple(int(clamp(channel + delta, 0, 255)) for channel in color)
            draw.point((x_value, y_value), fill=pixel)
    return image


def sample_occluders(width: int, height: int, style_cfg: dict[str, Any], rng: random.Random) -> list[list[int]]:
    occluders: list[list[int]] = []
    occluder_count = rng.randint(*style_cfg["occluder_count_range"])
    for _ in range(occluder_count):
        occ_w = rng.randint(max(width // 30, 10), max(width // 8, 24))
        occ_h = rng.randint(max(height // 30, 10), max(height // 8, 24))
        x1 = rng.randint(0, max(width - occ_w - 1, 1))
        y1 = rng.randint(0, max(height - occ_h - 1, 1))
        x2 = x1 + occ_w
        y2 = y1 + occ_h
        occluders.append([x1, y1, x2, y2])
    return occluders


def degrade_image(image: Image.Image, style_cfg: dict[str, Any], rng: random.Random, scale: float = 1.0) -> Image.Image:
    output = image
    if rng.random() < float(style_cfg["blur_probability"]) * scale:
        radius = rng.uniform(*style_cfg["blur_radius_range"]) * max(scale, 0.3)
        output = output.filter(ImageFilter.GaussianBlur(radius=radius))
    if rng.random() < float(style_cfg["noise_probability"]) * scale:
        pixels = output.load()
        width, height = output.size
        stddev = rng.uniform(*style_cfg["noise_stddev_range"]) * max(scale, 0.3)
        for y_value in range(height):
            for x_value in range(width):
                red, green, blue = pixels[x_value, y_value]
                jitter = int(rng.gauss(0, stddev))
                pixels[x_value, y_value] = (
                    int(clamp(red + jitter, 0, 255)),
                    int(clamp(green + jitter, 0, 255)),
                    int(clamp(blue + jitter, 0, 255)),
                )
    quality_min, quality_max = style_cfg["jpeg_quality_range"]
    quality = rng.randint(
        int(round(quality_min + (1.0 - scale) * 8)),
        quality_max,
    )
    buffer = BytesIO()
    output.save(buffer, format="JPEG", quality=int(clamp(quality, 40, 99)))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")
