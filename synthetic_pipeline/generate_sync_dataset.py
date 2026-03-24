#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFilter


@dataclass
class ArrowSpec:
    bbox: list[float]
    keypoints: list[list[Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic arrow dataset under data/sync/.")
    parser.add_argument(
        "--config",
        default="synthetic_pipeline/configs/base.yaml",
        help="YAML config path for synthetic dataset generation.",
    )
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _weighted_choice(rng: random.Random, mapping: dict[str, float]) -> str:
    keys = list(mapping.keys())
    weights = list(mapping.values())
    return rng.choices(keys, weights=weights, k=1)[0]


def _weighted_range_choice(rng: random.Random, entries: list[dict[str, Any]]) -> dict[str, Any]:
    return rng.choices(entries, weights=[float(item["weight"]) for item in entries], k=1)[0]


def _sample_resolution(rng: random.Random, buckets: list[dict[str, Any]]) -> tuple[int, int]:
    bucket = _weighted_range_choice(rng, buckets)
    width = rng.randint(int(bucket["min_width"]), int(bucket["max_width"]))
    height = rng.randint(int(bucket["min_height"]), int(bucket["max_height"]))
    return width, height


def _sample_arrow_count(rng: random.Random, bins: list[dict[str, Any]]) -> int:
    bucket = _weighted_range_choice(rng, bins)
    return rng.randint(int(bucket["min"]), int(bucket["max"]))


def _sample_point_count(rng: random.Random, bins: list[dict[str, Any]]) -> int:
    bucket = _weighted_range_choice(rng, bins)
    return int(bucket["count"])


def _sample_geometry_mode(rng: random.Random, entries: list[dict[str, Any]]) -> str:
    return str(_weighted_range_choice(rng, entries)["name"])


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _make_background(width: int, height: int, style_cfg: dict[str, Any], rng: random.Random) -> Image.Image:
    mode = _weighted_choice(rng, style_cfg["background_modes"])
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
            x = rng.randint(0, width - 1)
            y = rng.randint(0, height - 1)
            delta = rng.randint(-10, 10)
            pixel = tuple(int(_clamp(channel + delta, 0, 255)) for channel in color)
            draw.point((x, y), fill=pixel)
    return image


def _draw_distractors(draw: ImageDraw.ImageDraw, width: int, height: int, style_cfg: dict[str, Any], rng: random.Random) -> None:
    distractor_lines = rng.randint(*style_cfg["distractor_lines_range"])
    distractor_boxes = rng.randint(*style_cfg["distractor_boxes_range"])
    text_like_strokes = rng.randint(*style_cfg["text_like_strokes_range"])
    line_width = max(1, int(rng.randint(*style_cfg["distractor_line_width_range"])))

    for _ in range(distractor_lines):
        x1 = rng.randint(0, width - 1)
        y1 = rng.randint(0, height - 1)
        x2 = rng.randint(0, width - 1)
        y2 = rng.randint(0, height - 1)
        color = tuple(rng.randint(80, 180) for _ in range(3))
        draw.line((x1, y1, x2, y2), fill=color, width=line_width)

    for _ in range(distractor_boxes):
        w = rng.randint(max(width // 20, 12), max(width // 6, 24))
        h = rng.randint(max(height // 20, 12), max(height // 6, 24))
        x1 = rng.randint(0, max(width - w - 1, 1))
        y1 = rng.randint(0, max(height - h - 1, 1))
        x2 = x1 + w
        y2 = y1 + h
        color = tuple(rng.randint(90, 170) for _ in range(3))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

    for _ in range(text_like_strokes):
        x = rng.randint(0, width - 60)
        y = rng.randint(0, height - 20)
        color = tuple(rng.randint(60, 140) for _ in range(3))
        for step in range(rng.randint(3, 8)):
            x1 = x + step * rng.randint(8, 14)
            x2 = x1 + rng.randint(4, 10)
            y_jitter = y + rng.randint(-1, 1)
            draw.line((x1, y_jitter, x2, y_jitter), fill=color, width=1)


def _apply_render_style(points: list[tuple[float, float]], render_style: str, rng: random.Random) -> list[tuple[float, float]]:
    if render_style == "clean":
        return points
    jitter_scale = 0.0
    if render_style == "handdrawn":
        jitter_scale = 1.5
    elif render_style == "handdrawn_heavy":
        jitter_scale = 3.0
    elif render_style == "marker_jitter":
        jitter_scale = 1.2
    if jitter_scale <= 0:
        return points
    styled: list[tuple[float, float]] = []
    for idx, (x, y) in enumerate(points):
        if idx == 0 or idx == len(points) - 1:
            styled.append((x, y))
            continue
        styled.append((x + rng.uniform(-jitter_scale, jitter_scale), y + rng.uniform(-jitter_scale, jitter_scale)))
    return styled


def _orthogonal(vec_x: float, vec_y: float) -> tuple[float, float]:
    return -vec_y, vec_x


def _normalize(vec_x: float, vec_y: float) -> tuple[float, float]:
    norm = math.hypot(vec_x, vec_y)
    if norm == 0:
        return 0.0, 0.0
    return vec_x / norm, vec_y / norm


def _sample_arrow_points(
    width: int,
    height: int,
    point_count: int,
    layout_cfg: dict[str, Any],
    geometry_mode: str,
    rng: random.Random,
) -> list[tuple[float, float]]:
    margin = max(int(min(width, height) * float(layout_cfg["edge_margin_ratio"])), 12)
    min_length = min(width, height) * float(layout_cfg["min_arrow_length_ratio"])
    max_length = min(width, height) * float(layout_cfg["max_arrow_length_ratio"])

    start_x = rng.uniform(margin, width - margin)
    start_y = rng.uniform(margin, height - margin)
    angle = rng.uniform(0, math.tau)
    total_length = rng.uniform(min_length, max_length)
    end_x = _clamp(start_x + math.cos(angle) * total_length, margin, width - margin)
    end_y = _clamp(start_y + math.sin(angle) * total_length, margin, height - margin)

    points = [(start_x, start_y)]
    if point_count == 2:
        points.append((end_x, end_y))
        return points

    vx = end_x - start_x
    vy = end_y - start_y
    nx, ny = _normalize(vx, vy)
    ox, oy = _orthogonal(nx, ny)
    jitter_deg = float(layout_cfg["turn_angle_jitter_deg"])
    jitter_scale = total_length * 0.08
    if geometry_mode == "curve_gentle":
        jitter_scale = total_length * 0.14
    elif geometry_mode == "curve_s":
        jitter_scale = total_length * 0.18
    elif geometry_mode == "curve_hook":
        jitter_scale = total_length * 0.22
    for index in range(1, point_count - 1):
        t = index / (point_count - 1)
        base_x = start_x + vx * t
        base_y = start_y + vy * t
        if geometry_mode == "curve_gentle":
            offset = rng.uniform(jitter_scale * 0.35, jitter_scale)
        elif geometry_mode == "curve_s":
            direction = -1.0 if index % 2 == 1 else 1.0
            offset = rng.uniform(jitter_scale * 0.45, jitter_scale) * direction
        elif geometry_mode == "curve_hook":
            direction = 1.0 if index == point_count - 2 else 0.35
            offset = rng.uniform(jitter_scale * 0.4, jitter_scale) * direction
        else:
            offset = rng.uniform(-jitter_scale, jitter_scale)
        angle_jitter = math.radians(rng.uniform(-jitter_deg, jitter_deg))
        mix_x = ox * math.cos(angle_jitter) - nx * math.sin(angle_jitter)
        mix_y = oy * math.cos(angle_jitter) - ny * math.sin(angle_jitter)
        point_x = _clamp(base_x + mix_x * offset, margin, width - margin)
        point_y = _clamp(base_y + mix_y * offset, margin, height - margin)
        points.append((point_x, point_y))
    points.append((end_x, end_y))
    return points


def _catmull_rom_spline(points: list[tuple[float, float]], samples_per_segment: int = 20) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points
    extended = [points[0], *points, points[-1]]
    curve: list[tuple[float, float]] = []
    for index in range(1, len(extended) - 2):
        p0 = extended[index - 1]
        p1 = extended[index]
        p2 = extended[index + 1]
        p3 = extended[index + 2]
        for step in range(samples_per_segment):
            t = step / samples_per_segment
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            curve.append((x, y))
    curve.append(points[-1])
    return curve


def _arrow_bbox(points: list[tuple[float, float]], line_width: int, head_len: int, head_width: int) -> list[float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    pad = max(line_width, head_len, head_width) * 0.8
    return [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    line_width: int,
    head_len: int,
    head_width: int,
    color: tuple[int, int, int],
    head_style: str,
    line_style: str,
    render_style: str,
    rng: random.Random,
    geometry_mode: str,
    double_headed: bool = False,
) -> None:
    points = _apply_render_style(points, render_style, rng=rng)
    line_points = _catmull_rom_spline(points) if geometry_mode != "polyline" else points
    if line_style == "dashed":
        for start, end in zip(line_points[:-1], line_points[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            segment_length = math.hypot(dx, dy)
            if segment_length <= 1e-6:
                continue
            dash_len = max(line_width * 3, 8)
            gap_len = max(line_width * 2, 5)
            ux, uy = dx / segment_length, dy / segment_length
            cursor = 0.0
            while cursor < segment_length:
                dash_end = min(cursor + dash_len, segment_length)
                p1 = (start[0] + ux * cursor, start[1] + uy * cursor)
                p2 = (start[0] + ux * dash_end, start[1] + uy * dash_end)
                draw.line((p1, p2), fill=color, width=line_width)
                cursor += dash_len + gap_len
    elif line_style == "dotted":
        for start, end in zip(line_points[:-1], line_points[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            segment_length = math.hypot(dx, dy)
            if segment_length <= 1e-6:
                continue
            step = max(line_width * 2.2, 6)
            ux, uy = dx / segment_length, dy / segment_length
            cursor = 0.0
            radius = max(1, int(line_width * 0.8))
            while cursor <= segment_length:
                cx = start[0] + ux * cursor
                cy = start[1] + uy * cursor
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
                cursor += step
    elif line_style == "marker":
        draw.line(line_points, fill=color, width=max(line_width, 6), joint="curve")
        overlay_color = tuple(min(255, channel + 18) for channel in color)
        draw.line(line_points, fill=overlay_color, width=max(1, line_width // 2), joint="curve")
    else:
        draw.line(line_points, fill=color, width=line_width, joint="curve")
    if len(points) < 2:
        return
    head_segments = [(-2, -1)]
    if double_headed and len(points) >= 2:
        head_segments.append((1, 0))
    for tail_index, head_index in head_segments:
        tail = points[tail_index]
        head = points[head_index]
        dx = head[0] - tail[0]
        dy = head[1] - tail[1]
        nx, ny = _normalize(dx, dy)
        ox, oy = _orthogonal(nx, ny)
        base_x = head[0] - nx * head_len
        base_y = head[1] - ny * head_len
        left = (base_x + ox * head_width / 2, base_y + oy * head_width / 2)
        right = (base_x - ox * head_width / 2, base_y - oy * head_width / 2)
        if head_style == "open":
            draw.line((left, head), fill=color, width=max(1, line_width))
            draw.line((right, head), fill=color, width=max(1, line_width))
        else:
            draw.polygon([head, left, right], fill=color)


def _sample_arrow_style(style_cfg: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    profile = _weighted_range_choice(rng, style_cfg["arrow_style_profiles"])
    return {
        "line_width": rng.randint(*profile["line_width_range"]),
        "head_len": rng.randint(*profile["head_length_range"]),
        "head_width": rng.randint(*profile["head_width_range"]),
        "head_style": rng.choice(profile["head_styles"]),
        "line_style": rng.choice(profile["line_styles"]),
        "render_style": rng.choice(profile["render_styles"]),
        "color": tuple(rng.choice(profile["palettes"])),
    }


def _sample_arrow(
    width: int,
    height: int,
    cfg: dict[str, Any],
    existing_boxes: list[list[float]],
    rng: random.Random,
) -> tuple[ArrowSpec, dict[str, Any]] | None:
    style_cfg = cfg["style"]
    layout_cfg = cfg["layout"]
    max_iou = float(layout_cfg["max_instance_iou"])
    retries = int(layout_cfg["max_generation_retries_per_arrow"])
    point_count = _sample_point_count(rng, cfg["point_count_bins"])

    for _ in range(retries):
        geometry_mode = _sample_geometry_mode(rng, cfg["geometry_modes"])
        points = _sample_arrow_points(width, height, point_count, layout_cfg, geometry_mode, rng)
        arrow_style = _sample_arrow_style(style_cfg, rng)
        arrow_style["geometry_mode"] = geometry_mode
        line_width = int(arrow_style["line_width"])
        head_len = int(arrow_style["head_len"])
        head_width = int(arrow_style["head_width"])
        bbox = _arrow_bbox(points, line_width, head_len, head_width)
        bbox = [
            _clamp(bbox[0], 0, width - 1),
            _clamp(bbox[1], 0, height - 1),
            _clamp(bbox[2], 0, width - 1),
            _clamp(bbox[3], 0, height - 1),
        ]
        if bbox[2] - bbox[0] < 6 or bbox[3] - bbox[1] < 6:
            continue
        if any(_bbox_iou(bbox, other) > max_iou for other in existing_boxes):
            continue
        keypoints: list[list[Any]] = []
        for point_index, (x, y) in enumerate(points):
            visibility = "visible"
            if 0 < point_index < len(points) - 1 and rng.random() < float(style_cfg["occluded_point_probability"]):
                visibility = "occluded"
            keypoints.append([round(x, 2), round(y, 2), visibility])
        spec = ArrowSpec(
            bbox=[round(value, 2) for value in bbox],
            keypoints=keypoints,
        )
        return spec, arrow_style
    return None


def _draw_occluders(draw: ImageDraw.ImageDraw, width: int, height: int, style_cfg: dict[str, Any], rng: random.Random) -> None:
    occluder_count = rng.randint(*style_cfg["occluder_count_range"])
    for _ in range(occluder_count):
        occ_w = rng.randint(max(width // 30, 10), max(width // 8, 24))
        occ_h = rng.randint(max(height // 30, 10), max(height // 8, 24))
        x1 = rng.randint(0, max(width - occ_w - 1, 1))
        y1 = rng.randint(0, max(height - occ_h - 1, 1))
        x2 = x1 + occ_w
        y2 = y1 + occ_h
        gray = rng.randint(210, 245)
        fill = (gray, gray, gray)
        draw.rectangle((x1, y1, x2, y2), fill=fill, outline=fill)


def _draw_double_head_distractors(draw: ImageDraw.ImageDraw, width: int, height: int, cfg: dict[str, Any], rng: random.Random) -> None:
    style_cfg = cfg["style"]
    count = rng.randint(*style_cfg["double_head_distractor_range"])
    for _ in range(count):
        sampled = _sample_arrow(width, height, cfg, [], rng)
        if sampled is None:
            continue
        spec, arrow_style = sampled
        _draw_arrow(
            draw,
            [(float(point[0]), float(point[1])) for point in spec.keypoints],
            line_width=int(arrow_style["line_width"]),
            head_len=int(arrow_style["head_len"]),
            head_width=int(arrow_style["head_width"]),
            color=tuple(arrow_style["color"]),
            head_style=str(arrow_style["head_style"]),
            line_style=str(arrow_style["line_style"]),
            render_style=str(arrow_style["render_style"]),
            rng=rng,
            geometry_mode="polyline",
            double_headed=True,
        )


def _degrade_image(image: Image.Image, style_cfg: dict[str, Any], rng: random.Random) -> Image.Image:
    output = image
    if rng.random() < float(style_cfg["blur_probability"]):
        radius = rng.uniform(*style_cfg["blur_radius_range"])
        output = output.filter(ImageFilter.GaussianBlur(radius=radius))
    if rng.random() < float(style_cfg["noise_probability"]):
        stddev = rng.uniform(*style_cfg["noise_stddev_range"])
        pixels = output.load()
        width, height = output.size
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                jitter = int(rng.gauss(0, stddev))
                pixels[x, y] = (
                    int(_clamp(r + jitter, 0, 255)),
                    int(_clamp(g + jitter, 0, 255)),
                    int(_clamp(b + jitter, 0, 255)),
                )
    quality = rng.randint(*style_cfg["jpeg_quality_range"])
    buffer = BytesIO()
    output.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _generate_sample(split: str, index: int, cfg: dict[str, Any], rng: random.Random, output_dir: Path) -> dict[str, Any]:
    width, height = _sample_resolution(rng, cfg["resolution_buckets"])
    image = _make_background(width, height, cfg["style"], rng)
    draw = ImageDraw.Draw(image)
    _draw_distractors(draw, width, height, cfg["style"], rng)

    requested_arrows = _sample_arrow_count(rng, cfg["arrow_count_bins"])
    instances: list[dict[str, Any]] = []
    existing_boxes: list[list[float]] = []

    for _ in range(requested_arrows):
        sampled = _sample_arrow(width, height, cfg, existing_boxes, rng)
        if sampled is None:
            continue
        spec, arrow_style = sampled
        _draw_arrow(
            draw,
            [(float(point[0]), float(point[1])) for point in spec.keypoints],
            line_width=int(arrow_style["line_width"]),
            head_len=int(arrow_style["head_len"]),
            head_width=int(arrow_style["head_width"]),
            color=tuple(arrow_style["color"]),
            head_style=str(arrow_style["head_style"]),
            line_style=str(arrow_style["line_style"]),
            render_style=str(arrow_style["render_style"]),
            rng=rng,
            geometry_mode=str(arrow_style.get("geometry_mode", "polyline")),
        )
        instances.append({"bbox": spec.bbox, "keypoints": spec.keypoints})
        existing_boxes.append(spec.bbox)

    if not instances:
        sampled = _sample_arrow(width, height, cfg, existing_boxes, rng)
        if sampled is not None:
            spec, arrow_style = sampled
            _draw_arrow(
                draw,
                [(float(point[0]), float(point[1])) for point in spec.keypoints],
                line_width=int(arrow_style["line_width"]),
                head_len=int(arrow_style["head_len"]),
                head_width=int(arrow_style["head_width"]),
                color=tuple(arrow_style["color"]),
                head_style=str(arrow_style["head_style"]),
                line_style=str(arrow_style["line_style"]),
                render_style=str(arrow_style["render_style"]),
                rng=rng,
                geometry_mode=str(arrow_style.get("geometry_mode", "polyline")),
            )
            instances.append({"bbox": spec.bbox, "keypoints": spec.keypoints})

    _draw_double_head_distractors(draw, width, height, cfg, rng)
    _draw_occluders(draw, width, height, cfg["style"], rng)
    image = _degrade_image(image, cfg["style"], rng)

    sample_id = f"{split}_{index:06d}"
    absolute_image_path = output_dir / "images" / split / f"{sample_id}.jpg"
    absolute_image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(absolute_image_path, quality=95)
    if output_dir.is_absolute():
        image_path_value = str(absolute_image_path)
    else:
        image_path_value = str(output_dir / "images" / split / f"{sample_id}.jpg")

    return {
        "sample_id": sample_id,
        "image_path": image_path_value,
        "image_width": width,
        "image_height": height,
        "instances": instances,
    }


def _generate_sample_from_task(task: tuple[str, int, dict[str, Any], str, int]) -> dict[str, Any]:
    split, index, cfg, output_dir_str, seed = task
    rng = random.Random(seed)
    return _generate_sample(split, index, cfg, rng, Path(output_dir_str))


def _build_tasks(split: str, count: int, cfg: dict[str, Any], output_dir: Path, base_seed: int) -> list[tuple[str, int, dict[str, Any], str, int]]:
    return [
        (split, index, cfg, str(output_dir), base_seed + index)
        for index in range(count)
    ]


def _generate_records(
    split: str,
    count: int,
    cfg: dict[str, Any],
    output_dir: Path,
    base_seed: int,
    workers: int,
) -> list[dict[str, Any]]:
    tasks = _build_tasks(split, count, cfg, output_dir, base_seed)
    if workers <= 1:
        return [_generate_sample_from_task(task) for task in tasks]
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_generate_sample_from_task, tasks, chunksize=8))
    except (PermissionError, OSError):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_generate_sample_from_task, tasks))


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_instances = sum(len(record["instances"]) for record in records)
    total_points = sum(len(instance["keypoints"]) for record in records for instance in record["instances"])
    return {
        "num_samples": len(records),
        "num_instances": total_instances,
        "avg_instances_per_image": round(total_instances / max(len(records), 1), 3),
        "avg_points_per_instance": round(total_points / max(total_instances, 1), 3),
    }


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    if args.train_samples is not None:
        cfg["train_samples"] = args.train_samples
    if args.val_samples is not None:
        cfg["val_samples"] = args.val_samples
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.workers is not None:
        cfg["workers"] = args.workers

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(cfg.get("workers", os.cpu_count() or 1)))
    base_seed = int(cfg["seed"])

    train_records = _generate_records(
        "train",
        int(cfg["train_samples"]),
        cfg,
        output_dir,
        base_seed=base_seed,
        workers=workers,
    )
    val_records = _generate_records(
        "val",
        int(cfg["val_samples"]),
        cfg,
        output_dir,
        base_seed=base_seed + 1_000_000,
        workers=workers,
    )

    _write_jsonl(output_dir / "train.jsonl", train_records)
    _write_jsonl(output_dir / "val.jsonl", val_records)

    summary = {
        "config": cfg,
        "train": _summarize(train_records),
        "val": _summarize(val_records),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
