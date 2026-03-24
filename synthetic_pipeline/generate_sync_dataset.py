#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
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
    line_width = max(1, int(rng.randint(*style_cfg["line_width_range"]) - 1))

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
    for index in range(1, point_count - 1):
        t = index / (point_count - 1)
        base_x = start_x + vx * t
        base_y = start_y + vy * t
        offset = rng.uniform(-jitter_scale, jitter_scale)
        angle_jitter = math.radians(rng.uniform(-jitter_deg, jitter_deg))
        mix_x = ox * math.cos(angle_jitter) - nx * math.sin(angle_jitter)
        mix_y = oy * math.cos(angle_jitter) - ny * math.sin(angle_jitter)
        point_x = _clamp(base_x + mix_x * offset, margin, width - margin)
        point_y = _clamp(base_y + mix_y * offset, margin, height - margin)
        points.append((point_x, point_y))
    points.append((end_x, end_y))
    return points


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
) -> None:
    draw.line(points, fill=color, width=line_width, joint="curve")
    if len(points) < 2:
        return
    tail = points[-2]
    head = points[-1]
    dx = head[0] - tail[0]
    dy = head[1] - tail[1]
    nx, ny = _normalize(dx, dy)
    ox, oy = _orthogonal(nx, ny)
    base_x = head[0] - nx * head_len
    base_y = head[1] - ny * head_len
    left = (base_x + ox * head_width / 2, base_y + oy * head_width / 2)
    right = (base_x - ox * head_width / 2, base_y - oy * head_width / 2)
    draw.polygon([head, left, right], fill=color)


def _sample_arrow(
    width: int,
    height: int,
    cfg: dict[str, Any],
    existing_boxes: list[list[float]],
    rng: random.Random,
) -> tuple[ArrowSpec, int, int] | None:
    style_cfg = cfg["style"]
    layout_cfg = cfg["layout"]
    max_iou = float(layout_cfg["max_instance_iou"])
    retries = int(layout_cfg["max_generation_retries_per_arrow"])
    point_count = _sample_point_count(rng, cfg["point_count_bins"])

    for _ in range(retries):
        points = _sample_arrow_points(width, height, point_count, layout_cfg, rng)
        line_width = rng.randint(*style_cfg["line_width_range"])
        head_len = rng.randint(*style_cfg["arrow_head_length_range"])
        head_width = rng.randint(*style_cfg["arrow_head_width_range"])
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
        return spec, line_width, head_len if head_len > head_width else head_width
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
    colors = [(20, 20, 20), (10, 70, 120), (110, 40, 20), (60, 60, 60)]

    for _ in range(requested_arrows):
        sampled = _sample_arrow(width, height, cfg, existing_boxes, rng)
        if sampled is None:
            continue
        spec, line_width, head_size = sampled
        color = rng.choice(colors)
        _draw_arrow(
            draw,
            [(float(point[0]), float(point[1])) for point in spec.keypoints],
            line_width=line_width,
            head_len=head_size,
            head_width=max(head_size * 0.7, line_width + 2),
            color=color,
        )
        instances.append({"bbox": spec.bbox, "keypoints": spec.keypoints})
        existing_boxes.append(spec.bbox)

    if not instances:
        sampled = _sample_arrow(width, height, cfg, existing_boxes, rng)
        if sampled is not None:
            spec, line_width, head_size = sampled
            color = rng.choice(colors)
            _draw_arrow(
                draw,
                [(float(point[0]), float(point[1])) for point in spec.keypoints],
                line_width=line_width,
                head_len=head_size,
                head_width=max(head_size * 0.7, line_width + 2),
                color=color,
            )
            instances.append({"bbox": spec.bbox, "keypoints": spec.keypoints})

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

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(cfg["seed"]))

    train_records = [
        _generate_sample("train", index, cfg, rng, output_dir)
        for index in range(int(cfg["train_samples"]))
    ]
    val_records = [
        _generate_sample("val", index, cfg, rng, output_dir)
        for index in range(int(cfg["val_samples"]))
    ]

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
