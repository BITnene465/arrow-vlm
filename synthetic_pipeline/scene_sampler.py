from __future__ import annotations

import math
import random
from typing import Any

from synthetic_pipeline.schema import ArrowInstance, SceneSpec, SyntheticSample


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


def _sample_arrow_count_for_scene(rng: random.Random, bins: list[dict[str, Any]], scene_mode: str) -> int:
    if scene_mode in {"single_hero", "single_crop"}:
        return 1
    if scene_mode == "sparse_large":
        return rng.randint(1, 6)
    return _sample_arrow_count(rng, bins)


def _sample_point_count(rng: random.Random, bins: list[dict[str, Any]]) -> int:
    bucket = _weighted_range_choice(rng, bins)
    return int(bucket["count"])


def _sample_geometry_mode(rng: random.Random, entries: list[dict[str, Any]]) -> str:
    return str(_weighted_range_choice(rng, entries)["name"])


def _sample_scene_mode(rng: random.Random, entries: list[dict[str, Any]]) -> str:
    return str(_weighted_range_choice(rng, entries)["name"])


def _sample_arrow_size_mode(rng: random.Random, entries: list[dict[str, Any]]) -> dict[str, Any]:
    return _weighted_range_choice(rng, entries)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
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


def orthogonal(vec_x: float, vec_y: float) -> tuple[float, float]:
    return -vec_y, vec_x


def normalize(vec_x: float, vec_y: float) -> tuple[float, float]:
    norm = math.hypot(vec_x, vec_y)
    if norm == 0:
        return 0.0, 0.0
    return vec_x / norm, vec_y / norm


def sample_arrow_points(
    width: int,
    height: int,
    point_count: int,
    layout_cfg: dict[str, Any],
    geometry_mode: str,
    size_mode: dict[str, Any],
    rng: random.Random,
) -> list[tuple[float, float]]:
    margin = max(int(min(width, height) * float(layout_cfg["edge_margin_ratio"])), 12)
    default_min_length = min(width, height) * float(layout_cfg["min_arrow_length_ratio"])
    min_length = min(width, height) * float(size_mode.get("min_length_ratio", layout_cfg["min_arrow_length_ratio"]))
    max_length = min(width, height) * float(size_mode.get("max_length_ratio", layout_cfg["max_arrow_length_ratio"]))
    min_length = max(min_length, default_min_length * 0.6)
    max_length = min(max(max_length, min_length + 8.0), min(width, height) * 0.95)

    start_x = rng.uniform(margin, width - margin)
    start_y = rng.uniform(margin, height - margin)
    if rng.random() < float(layout_cfg.get("primary_axis_aligned_probability", 0.0)):
        base_angle = rng.choice([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
        jitter = math.radians(
            rng.uniform(
                -float(layout_cfg.get("orthogonal_angle_jitter_deg", 0.0)),
                float(layout_cfg.get("orthogonal_angle_jitter_deg", 0.0)),
            )
        )
        angle = base_angle + jitter
    else:
        angle = rng.uniform(0, math.tau)
    total_length = rng.uniform(min_length, max_length)
    end_x = clamp(start_x + math.cos(angle) * total_length, margin, width - margin)
    end_y = clamp(start_y + math.sin(angle) * total_length, margin, height - margin)

    points = [(start_x, start_y)]
    if point_count == 2:
        points.append((end_x, end_y))
        return points

    vx = end_x - start_x
    vy = end_y - start_y
    nx, ny = normalize(vx, vy)
    ox, oy = orthogonal(nx, ny)
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
        point_x = clamp(base_x + mix_x * offset, margin, width - margin)
        point_y = clamp(base_y + mix_y * offset, margin, height - margin)
        points.append((point_x, point_y))
    points.append((end_x, end_y))
    return points


def arrow_bbox(points: list[list[float]] | list[tuple[float, float]], pad: float) -> list[float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]


class SceneSampler:
    def __init__(self, cfg: dict[str, Any], renderer_name: str) -> None:
        self.cfg = cfg
        self.renderer_name = renderer_name

    def sample(self, split: str, index: int, rng: random.Random) -> SyntheticSample:
        width, height = _sample_resolution(rng, self.cfg["resolution_buckets"])
        scene_mode = _sample_scene_mode(rng, self.cfg["scene_modes"])
        sample_id = f"{split}_{index:06d}"
        requested_arrows = _sample_arrow_count_for_scene(rng, self.cfg["arrow_count_bins"], scene_mode)

        instances: list[ArrowInstance] = []
        existing_boxes: list[list[float]] = []
        for _ in range(requested_arrows):
            instance = self._sample_instance(width, height, scene_mode, existing_boxes, rng)
            if instance is None:
                continue
            instances.append(instance)
            existing_boxes.append(instance.bbox)

        if not instances:
            instance = self._sample_instance(width, height, scene_mode, existing_boxes, rng)
            if instance is not None:
                instances.append(instance)

        single_crop_padding_ratio = None
        if scene_mode == "single_crop" and instances:
            bbox = instances[0].bbox
            bbox_w = max(float(bbox[2]) - float(bbox[0]), 1.0)
            bbox_h = max(float(bbox[3]) - float(bbox[1]), 1.0)
            single_crop_padding_ratio = round(
                max((width - bbox_w) / bbox_w, (height - bbox_h) / bbox_h),
                3,
            )

        scene = SceneSpec(
            split=split,
            sample_id=sample_id,
            image_width=width,
            image_height=height,
            scene_mode=scene_mode,
            renderer_name=self.renderer_name,
            single_crop_padding_ratio=single_crop_padding_ratio,
        )
        return SyntheticSample(
            scene=scene,
            instances=instances,
            render_meta={"requested_arrows": requested_arrows},
        )

    def _sample_instance(
        self,
        width: int,
        height: int,
        scene_mode: str,
        existing_boxes: list[list[float]],
        rng: random.Random,
    ) -> ArrowInstance | None:
        layout_cfg = self.cfg["layout"]
        max_iou = float(layout_cfg["max_instance_iou"])
        retries = int(layout_cfg["max_generation_retries_per_arrow"])
        point_count = _sample_point_count(rng, self.cfg["point_count_bins"])

        for _ in range(retries):
            geometry_mode = _sample_geometry_mode(rng, self.cfg["geometry_modes"])
            size_mode = _sample_arrow_size_mode(rng, self.cfg["arrow_size_modes"])
            if scene_mode == "single_hero":
                size_mode = {"name": "hero", "min_length_ratio": 0.55, "max_length_ratio": 0.92}
            elif scene_mode == "single_crop":
                size_mode = {"name": "single_crop", "min_length_ratio": 0.36, "max_length_ratio": 0.74}
            elif scene_mode == "sparse_large" and rng.random() < 0.7:
                size_mode = {"name": "large", "min_length_ratio": 0.28, "max_length_ratio": 0.62}

            points = sample_arrow_points(
                width=width,
                height=height,
                point_count=point_count,
                layout_cfg=layout_cfg,
                geometry_mode=geometry_mode,
                size_mode=size_mode,
                rng=rng,
            )
            bbox = arrow_bbox(points, pad=max(min(width, height) * 0.02, 12.0))
            bbox = [
                clamp(bbox[0], 0, width - 1),
                clamp(bbox[1], 0, height - 1),
                clamp(bbox[2], 0, width - 1),
                clamp(bbox[3], 0, height - 1),
            ]
            if bbox[2] - bbox[0] < 6 or bbox[3] - bbox[1] < 6:
                continue
            if any(bbox_iou(bbox, other) > max_iou for other in existing_boxes):
                continue
            return ArrowInstance(
                bbox=[round(value, 2) for value in bbox],
                keypoints=[[round(x, 2), round(y, 2)] for x, y in points],
                meta={
                    "geometry_mode": geometry_mode,
                    "size_mode": str(size_mode.get("name", "medium")),
                    "point_count": point_count,
                },
            )
        return None
