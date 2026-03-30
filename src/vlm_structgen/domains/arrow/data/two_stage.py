from __future__ import annotations

import math
import os
import random
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import Any

from PIL import Image

from vlm_structgen.domains.arrow.ordering import (
    grounding_instance_sort_key,
    sort_grounding_instances_canonical,
    sort_instances_canonical,
)
from vlm_structgen.core.utils.io import ensure_dir, load_jsonl, write_json, write_jsonl

Image.MAX_IMAGE_PIXELS = None


def _parse_int_sequence(values: list[int] | tuple[int, ...] | None, *, default: list[int]) -> list[int]:
    resolved = list(values) if values else list(default)
    return [int(value) for value in resolved if int(value) > 0]


def _parse_float_sequence(values: list[float] | tuple[float, ...] | None, *, default: list[float]) -> list[float]:
    resolved = list(values) if values else list(default)
    return [float(value) for value in resolved if float(value) > 0.0]


def _quantize(value: float, size: int, num_bins: int) -> int:
    size = max(int(size), 1)
    if size == 1:
        return 0
    clipped = min(max(float(value), 0.0), float(size - 1))
    return int(round(clipped / float(size - 1) * float(num_bins - 1)))


def _build_stage1_instance(instance: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": instance["label"],
        "bbox": list(instance["bbox"]),
    }


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) * 0.5, (float(bbox[1]) + float(bbox[3])) * 0.5)


def _intersect_bbox(bbox: list[float], crop_box: list[int]) -> list[float] | None:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    x1 = max(float(bbox[0]), float(crop_x1))
    y1 = max(float(bbox[1]), float(crop_y1))
    x2 = min(float(bbox[2]), float(crop_x2))
    y2 = min(float(bbox[3]), float(crop_y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_area(bbox: list[float]) -> float:
    return max(float(bbox[2]) - float(bbox[0]), 0.0) * max(float(bbox[3]) - float(bbox[1]), 0.0)


def _bbox_iou(box_a: list[int] | list[float], box_b: list[int] | list[float]) -> float:
    intersection = _intersect_bbox([float(value) for value in box_a], [int(value) for value in box_b])
    if intersection is None:
        return 0.0
    intersection_area = _bbox_area(intersection)
    if intersection_area <= 0.0:
        return 0.0
    area_a = _bbox_area([float(value) for value in box_a])
    area_b = _bbox_area([float(value) for value in box_b])
    union = area_a + area_b - intersection_area
    if union <= 0.0:
        return 0.0
    return intersection_area / union


def _is_bbox_fully_inside_crop(bbox: list[float], crop_box: list[int]) -> bool:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    return (
        float(crop_x1) <= float(bbox[0])
        and float(crop_y1) <= float(bbox[1])
        and float(bbox[2]) <= float(crop_x2)
        and float(bbox[3]) <= float(crop_y2)
    )


def _crop_image_region(image: Image.Image, crop_box: list[int]) -> Image.Image:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    return image.crop((crop_x1, crop_y1, crop_x2, crop_y2))


def _save_image_atomic(image: Image.Image, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    temp_path = output_path.with_name(f".{output_path.stem}.tmp-{os.getpid()}{output_path.suffix}")
    try:
        image.save(temp_path)
        with Image.open(temp_path) as verify_image:
            verify_image.verify()
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _sliding_window_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts: list[int] = [0]
    current = 0
    while current + tile_size < length:
        current += stride
        current = min(current, length - tile_size)
        if current == starts[-1]:
            break
        starts.append(current)
    return starts


def _build_sliding_crop_boxes(
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    stride: int,
) -> list[list[int]]:
    x_starts = _sliding_window_starts(image_width, tile_size, stride)
    y_starts = _sliding_window_starts(image_height, tile_size, stride)
    crop_boxes: list[list[int]] = []
    for y in y_starts:
        for x in x_starts:
            crop_boxes.append(
                [
                    int(x),
                    int(y),
                    int(min(x + tile_size, image_width)),
                    int(min(y + tile_size, image_height)),
                ]
            )
    return crop_boxes


def _build_density_crop_box(
    *,
    center_x: float,
    center_y: float,
    tile_size: int,
    image_width: int,
    image_height: int,
) -> list[int]:
    width = min(int(tile_size), int(image_width))
    height = min(int(tile_size), int(image_height))
    x1 = int(round(center_x - width * 0.5))
    y1 = int(round(center_y - height * 0.5))
    x1 = min(max(x1, 0), max(image_width - width, 0))
    y1 = min(max(y1, 0), max(image_height - height, 0))
    return [x1, y1, x1 + width, y1 + height]


def _resolve_stage1_tile_sizes(
    *,
    image_width: int,
    image_height: int,
    tile_size_ratios: list[float],
    min_tile_size: int,
    max_tile_size: int,
) -> list[int]:
    short_side = max(min(int(image_width), int(image_height)), 1)
    resolved_sizes: list[int] = []
    seen: set[int] = set()
    clamped_max_tile_size = max(int(min_tile_size), int(max_tile_size))
    for ratio in tile_size_ratios:
        candidate = int(round(short_side * float(ratio)))
        candidate = min(max(candidate, int(min_tile_size)), clamped_max_tile_size)
        if candidate not in seen:
            resolved_sizes.append(candidate)
            seen.add(candidate)
    return resolved_sizes


def _select_instances_for_stage1_crop(
    instances: list[dict[str, Any]],
    *,
    crop_box: list[int],
) -> tuple[list[dict[str, Any]], list[int]]:
    selected_with_index: list[tuple[dict[str, Any], int]] = []
    for instance_index, instance in enumerate(instances):
        intersection = _intersect_bbox(instance["bbox"], crop_box)
        if intersection is None:
            continue
        if not _is_bbox_fully_inside_crop(instance["bbox"], crop_box):
            return [], []
        stage1_instance = _build_stage1_instance(instance)
        # Stage1 tile supervision should stay inside the tile-local frame.
        # Keep only fully enclosed instances so the local bbox stays exact.
        local_bbox = to_crop_local_bbox(instance["bbox"], crop_box)
        stage1_instance["bbox"] = _round_bbox(local_bbox)
        selected_with_index.append((stage1_instance, int(instance_index)))
    selected_with_index.sort(key=lambda item: grounding_instance_sort_key(item[0]))
    return (
        [item[0] for item in selected_with_index],
        [item[1] for item in selected_with_index],
    )


def _write_stage1_crop_image(
    *,
    image: Image.Image,
    output_dir: Path,
    split: str,
    sample_id: str,
    crop_box: list[int],
) -> tuple[Path, int, int]:
    crop_dir = ensure_dir(output_dir / "stage1" / "images" / split)
    crop_path = crop_dir / f"{sample_id}.png"
    crop_image = _crop_image_region(image, crop_box)
    _save_image_atomic(crop_image, crop_path)
    crop_width, crop_height = crop_image.size
    crop_image.close()
    return crop_path, int(crop_width), int(crop_height)


def _build_stage1_crop_record(
    *,
    record: dict[str, Any],
    image: Image.Image,
    split: str,
    output_dir: Path,
    crop_box: list[int],
    sample_suffix: str,
    source_type: str,
) -> dict[str, Any] | None:
    selected_instances, selected_instance_indices = _select_instances_for_stage1_crop(
        record.get("instances", []),
        crop_box=crop_box,
    )
    if not selected_instances:
        return None
    sample_id = f"{record['sample_id']}__{sample_suffix}"
    crop_path, crop_width, crop_height = _write_stage1_crop_image(
        image=image,
        output_dir=output_dir,
        split=split,
        sample_id=sample_id,
        crop_box=crop_box,
    )
    return {
        "task_type": "grounding",
        "domain_type": "arrow",
        "sample_id": sample_id,
        "source_sample_id": record["sample_id"],
        "source_type": source_type,
        "image_path": str(crop_path),
        "image_width": int(crop_width),
        "image_height": int(crop_height),
        "crop_box": [int(value) for value in crop_box],
        "instances": selected_instances,
        "_instance_indices": selected_instance_indices,
    }


def _build_stage1_full_image_record(record: dict[str, Any]) -> dict[str, Any]:
    indexed_instances = [
        (_build_stage1_instance(instance), int(instance_index))
        for instance_index, instance in enumerate(record.get("instances", []))
    ]
    indexed_instances.sort(key=lambda item: grounding_instance_sort_key(item[0]))
    return {
        "task_type": "grounding",
        "domain_type": "arrow",
        "sample_id": record["sample_id"],
        "source_sample_id": record["sample_id"],
        "source_type": "full_image",
        "image_path": record["image_path"],
        "image_width": int(record["image_width"]),
        "image_height": int(record["image_height"]),
        "instances": [item[0] for item in indexed_instances],
        "_instance_indices": [item[1] for item in indexed_instances],
    }


def _stage1_source_priority(source_type: str) -> int:
    if source_type == "full_image":
        return 3
    if source_type.startswith("density_"):
        return 0
    if source_type.startswith("sliding_"):
        return 1
    return 2


def _strip_stage1_internal_fields(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(record)
    cleaned.pop("_instance_indices", None)
    return cleaned


def _deduplicate_stage1_records(
    records: list[dict[str, Any]],
    *,
    dedup_iou_threshold: float,
) -> tuple[list[dict[str, Any]], int]:
    if not records:
        return [], 0

    full_image_records = [record for record in records if record.get("source_type") == "full_image"]
    crop_records = [record for record in records if record.get("source_type") != "full_image"]
    crop_records.sort(
        key=lambda record: (
            _stage1_source_priority(str(record.get("source_type", ""))),
            _bbox_area(record.get("crop_box", [0, 0, 0, 0])),
            record.get("sample_id", ""),
        )
    )

    kept_crop_records: list[dict[str, Any]] = []
    kept_by_signature: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    dropped = 0

    for record in crop_records:
        signature = tuple(int(value) for value in record.get("_instance_indices", []))
        crop_box = record.get("crop_box")
        if signature and crop_box is not None:
            existing_records = kept_by_signature.setdefault(signature, [])
            if any(_bbox_iou(existing["crop_box"], crop_box) >= float(dedup_iou_threshold) for existing in existing_records):
                image_path = record.get("image_path")
                if image_path:
                    Path(image_path).unlink(missing_ok=True)
                dropped += 1
                continue
            existing_records.append(record)
        kept_crop_records.append(record)

    deduped_records = full_image_records + kept_crop_records
    return [_strip_stage1_internal_fields(record) for record in deduped_records], dropped


def _encode_stage2_target(keypoints_2d: list[list[int]]) -> str:
    import json

    return json.dumps({"keypoints_2d": keypoints_2d}, ensure_ascii=False, separators=(",", ":"))


def _round_bbox(bbox: list[float]) -> list[float]:
    return [round(float(value), 4) for value in bbox]


def _round_keypoints(keypoints: list[list[float]]) -> list[list[float]]:
    return [[round(float(x), 4), round(float(y), 4)] for x, y in keypoints]


def _clip_point(point: list[float], image_width: int, image_height: int) -> list[float]:
    return [
        min(max(float(point[0]), 0.0), float(max(image_width - 1, 0))),
        min(max(float(point[1]), 0.0), float(max(image_height - 1, 0))),
    ]


def _clip_bbox(bbox: list[float], image_width: int, image_height: int) -> list[float]:
    x1 = min(max(float(bbox[0]), 0.0), float(max(image_width - 1, 0)))
    y1 = min(max(float(bbox[1]), 0.0), float(max(image_height - 1, 0)))
    x2 = min(max(float(bbox[2]), 0.0), float(max(image_width - 1, 0)))
    y2 = min(max(float(bbox[3]), 0.0), float(max(image_height - 1, 0)))
    if x2 <= x1:
        x2 = min(float(max(image_width - 1, 0)), x1 + 1.0)
        x1 = max(0.0, x2 - 1.0)
    if y2 <= y1:
        y2 = min(float(max(image_height - 1, 0)), y1 + 1.0)
        y1 = max(0.0, y2 - 1.0)
    return [x1, y1, x2, y2]


def _all_points_inside_crop(keypoints: list[list[float]], crop_box: list[int]) -> bool:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    return all(
        crop_x1 <= float(x) <= crop_x2 - 1 and crop_y1 <= float(y) <= crop_y2 - 1
        for x, y in keypoints
    )


def _stable_rng(
    *,
    sample_id: str,
    target_index: int,
    aug_index: int,
    seed: int,
) -> random.Random:
    raw = f"{sample_id}:{target_index}:{aug_index}:{seed}".encode("utf-8")
    digest = sha256(raw).hexdigest()
    return random.Random(int(digest[:16], 16))


def _jitter_bbox(
    bbox: list[float],
    *,
    image_width: int,
    image_height: int,
    center_ratio: float,
    scale_ratio: float,
    rng: random.Random,
) -> tuple[list[float], dict[str, float]]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5

    dx = rng.uniform(-center_ratio * width, center_ratio * width)
    dy = rng.uniform(-center_ratio * height, center_ratio * height)
    dw = rng.uniform(-scale_ratio * width, scale_ratio * width)
    dh = rng.uniform(-scale_ratio * height, scale_ratio * height)

    noisy_width = max(width + dw, 2.0)
    noisy_height = max(height + dh, 2.0)
    noisy_center_x = center_x + dx
    noisy_center_y = center_y + dy

    noisy_bbox = [
        noisy_center_x - noisy_width * 0.5,
        noisy_center_y - noisy_height * 0.5,
        noisy_center_x + noisy_width * 0.5,
        noisy_center_y + noisy_height * 0.5,
    ]
    noisy_bbox = _clip_bbox(noisy_bbox, image_width, image_height)
    return noisy_bbox, {
        "bbox_center_dx_px": round(dx, 4),
        "bbox_center_dy_px": round(dy, 4),
        "bbox_scale_dw_px": round(noisy_bbox[2] - noisy_bbox[0] - width, 4),
        "bbox_scale_dh_px": round(noisy_bbox[3] - noisy_bbox[1] - height, 4),
    }


def build_padded_crop(
    image: Image.Image,
    *,
    bbox: list[float],
    padding_ratio: float,
    max_aspect_ratio: float = 180.0,
) -> tuple[Image.Image, list[int]]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    pad_x = width * float(padding_ratio)
    pad_y = height * float(padding_ratio)

    crop_x1 = math.floor(x1 - pad_x)
    crop_y1 = math.floor(y1 - pad_y)
    crop_x2 = math.ceil(x2 + pad_x)
    crop_y2 = math.ceil(y2 + pad_y)

    crop_w = max(int(crop_x2 - crop_x1), 1)
    crop_h = max(int(crop_y2 - crop_y1), 1)
    if float(max_aspect_ratio) > 1.0:
        if crop_w > crop_h and float(crop_w) / float(crop_h) > float(max_aspect_ratio):
            target_h = int(math.ceil(float(crop_w) / float(max_aspect_ratio)))
            extra = max(target_h - crop_h, 0)
            expand_top = extra // 2
            expand_bottom = extra - expand_top
            crop_y1 -= expand_top
            crop_y2 += expand_bottom
        elif crop_h > crop_w and float(crop_h) / float(crop_w) > float(max_aspect_ratio):
            target_w = int(math.ceil(float(crop_h) / float(max_aspect_ratio)))
            extra = max(target_w - crop_w, 0)
            expand_left = extra // 2
            expand_right = extra - expand_left
            crop_x1 -= expand_left
            crop_x2 += expand_right

    crop_w = max(int(crop_x2 - crop_x1), 1)
    crop_h = max(int(crop_y2 - crop_y1), 1)

    canvas = Image.new("RGB", (crop_w, crop_h), color=(0, 0, 0))
    src_x1 = max(crop_x1, 0)
    src_y1 = max(crop_y1, 0)
    src_x2 = min(crop_x2, image.width)
    src_y2 = min(crop_y2, image.height)
    if src_x2 > src_x1 and src_y2 > src_y1:
        patch = image.crop((src_x1, src_y1, src_x2, src_y2))
        paste_x = int(src_x1 - crop_x1)
        paste_y = int(src_y1 - crop_y1)
        canvas.paste(patch, (paste_x, paste_y))

    return canvas, [int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)]


def to_crop_local_bbox(bbox: list[float], crop_box: list[int]) -> list[float]:
    crop_x1, crop_y1, _crop_x2, _crop_y2 = crop_box
    return [
        float(bbox[0]) - float(crop_x1),
        float(bbox[1]) - float(crop_y1),
        float(bbox[2]) - float(crop_x1),
        float(bbox[3]) - float(crop_y1),
    ]


def to_crop_local_keypoints(keypoints: list[list[float]], crop_box: list[int]) -> list[list[float]]:
    crop_x1, crop_y1, _crop_x2, _crop_y2 = crop_box
    return [
        [float(x) - float(crop_x1), float(y) - float(crop_y1)]
        for x, y in keypoints
    ]


def quantize_bbox_2d(bbox: list[float], image_width: int, image_height: int, num_bins: int) -> list[int]:
    return [
        _quantize(bbox[0], image_width, num_bins),
        _quantize(bbox[1], image_height, num_bins),
        _quantize(bbox[2], image_width, num_bins),
        _quantize(bbox[3], image_height, num_bins),
    ]


def quantize_keypoints_2d(
    keypoints: list[list[float]],
    image_width: int,
    image_height: int,
    num_bins: int,
) -> list[list[int]]:
    return [
        [
            _quantize(point[0], image_width, num_bins),
            _quantize(point[1], image_height, num_bins),
        ]
        for point in keypoints
    ]


def dequantize_keypoints_2d(
    keypoints_2d: list[list[int]],
    image_width: int,
    image_height: int,
    num_bins: int,
) -> list[list[float]]:
    width = max(int(image_width), 1)
    height = max(int(image_height), 1)
    if width == 1:
        x_scale = 0.0
    else:
        x_scale = float(width - 1) / float(num_bins - 1)
    if height == 1:
        y_scale = 0.0
    else:
        y_scale = float(height - 1) / float(num_bins - 1)
    return [
        [float(point[0]) * x_scale, float(point[1]) * y_scale]
        for point in keypoints_2d
    ]


def _build_stage2_record(
    record: dict[str, Any],
    instance: dict[str, Any],
    *,
    image: Image.Image,
    split: str,
    target_index: int,
    sample_suffix: str,
    hint_bbox: list[float],
    hint_keypoints: list[list[float]],
    output_dir: Path,
    padding_ratio: float,
    num_bins: int,
    augmentation: dict[str, Any],
) -> dict[str, Any]:
    crop_image, crop_box = build_padded_crop(
        image,
        bbox=hint_bbox,
        padding_ratio=padding_ratio,
    )
    crop_width, crop_height = crop_image.size
    crop_dir = ensure_dir(output_dir / "stage2" / "images" / split)
    crop_name = f"{record['sample_id']}__inst_{target_index:04d}{sample_suffix}.png"
    crop_path = crop_dir / crop_name
    _save_image_atomic(crop_image, crop_path)
    crop_image.close()

    local_gt_bbox = to_crop_local_bbox(instance["bbox"], crop_box)
    local_gt_keypoints = to_crop_local_keypoints(instance["keypoints"], crop_box)
    local_hint_bbox = to_crop_local_bbox(hint_bbox, crop_box)
    local_hint_keypoints = to_crop_local_keypoints(hint_keypoints, crop_box)
    local_bbox_2d = quantize_bbox_2d(local_hint_bbox, crop_width, crop_height, num_bins)
    local_hint_keypoints_2d = quantize_keypoints_2d(
        local_hint_keypoints,
        crop_width,
        crop_height,
        num_bins,
    )
    local_full_keypoints_2d = quantize_keypoints_2d(
        local_gt_keypoints,
        crop_width,
        crop_height,
        num_bins,
    )

    return {
        "task_type": "keypoint_sequence",
        "domain_type": "arrow",
        "sample_id": f"{record['sample_id']}__inst_{target_index:04d}{sample_suffix}",
        "source_sample_id": record["sample_id"],
        "target_index": int(target_index),
        "image_path": str(crop_path),
        "image_width": int(crop_width),
        "image_height": int(crop_height),
        "system_prompt": "",
        "target_text": _encode_stage2_target(local_full_keypoints_2d),
        "gt_struct": {
            "task_type": "keypoint_sequence",
            "domain_type": "arrow",
            "label": instance["label"],
            "keypoints": _round_keypoints(local_gt_keypoints),
            "keypoints_2d": local_full_keypoints_2d,
        },
        "instances": [
            {
                "label": instance["label"],
                "bbox": _round_bbox(local_gt_bbox),
                "keypoints": _round_keypoints(local_gt_keypoints),
            }
        ],
        "condition": {
            "label": instance["label"],
            "bbox": _round_bbox(local_hint_bbox),
            "bbox_2d": local_bbox_2d,
            "keypoints": _round_keypoints(local_hint_keypoints),
            "keypoints_2d": local_hint_keypoints_2d,
        },
        "crop_box": crop_box,
        "augmentation": augmentation,
    }


def _enumerate_stage1_sliding_records(
    *,
    record: dict[str, Any],
    image: Image.Image,
    split: str,
    output_dir: Path,
    tile_size_ratios: list[float],
    min_tile_size: int,
    max_tile_size: int,
    tile_stride_ratio: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    image_width = int(record["image_width"])
    image_height = int(record["image_height"])
    tile_sizes = _resolve_stage1_tile_sizes(
        image_width=image_width,
        image_height=image_height,
        tile_size_ratios=tile_size_ratios,
        min_tile_size=min_tile_size,
        max_tile_size=max_tile_size,
    )
    for tile_size in tile_sizes:
        stride = max(int(round(tile_size * float(tile_stride_ratio))), 1)
        for index, crop_box in enumerate(
            _build_sliding_crop_boxes(
                image_width=image_width,
                image_height=image_height,
                tile_size=int(tile_size),
                stride=stride,
            )
        ):
            stage1_record = _build_stage1_crop_record(
                record=record,
                image=image,
                split=split,
                output_dir=output_dir,
                crop_box=crop_box,
                sample_suffix=f"slide_t{int(tile_size)}_{index:04d}",
                source_type=f"sliding_{int(tile_size)}",
            )
            if stage1_record is not None:
                records.append(stage1_record)
    return records


def _enumerate_stage1_density_records(
    *,
    record: dict[str, Any],
    image: Image.Image,
    split: str,
    output_dir: Path,
    tile_size_ratios: list[float],
    min_tile_size: int,
    max_tile_size: int,
    min_instances: int,
    max_instances: int,
    max_crops_per_size: int,
) -> list[dict[str, Any]]:
    image_width = int(record["image_width"])
    image_height = int(record["image_height"])
    tile_sizes = _resolve_stage1_tile_sizes(
        image_width=image_width,
        image_height=image_height,
        tile_size_ratios=tile_size_ratios,
        min_tile_size=min_tile_size,
        max_tile_size=max_tile_size,
    )
    instances = record.get("instances", [])
    if not instances:
        return []

    density_records: list[dict[str, Any]] = []
    centers = [(*_bbox_center(instance["bbox"]), idx) for idx, instance in enumerate(instances)]
    desired_mid = (int(min_instances) + int(max_instances)) * 0.5

    for tile_size in tile_sizes:
        candidates: list[tuple[tuple[float, float, float, int], list[int], int]] = []
        seen_boxes: set[tuple[int, int, int, int]] = set()
        for center_x, center_y, center_index in centers:
            crop_box = _build_density_crop_box(
                center_x=center_x,
                center_y=center_y,
                tile_size=int(tile_size),
                image_width=image_width,
                image_height=image_height,
            )
            crop_key = tuple(crop_box)
            if crop_key in seen_boxes:
                continue
            seen_boxes.add(crop_key)
            selected_instances = _select_instances_for_stage1_crop(
                instances,
                crop_box=crop_box,
            )
            instance_count = len(selected_instances)
            if instance_count < int(min_instances) or instance_count > int(max_instances):
                continue
            score = (
                abs(instance_count - desired_mid),
                -instance_count,
                float(crop_box[1]),
                float(crop_box[0]),
                int(center_index),
            )
            candidates.append((score, crop_box, instance_count))

        candidates.sort(key=lambda item: item[0])
        for index, (_score, crop_box, _instance_count) in enumerate(candidates[: max(int(max_crops_per_size), 0)]):
            density_record = _build_stage1_crop_record(
                record=record,
                image=image,
                split=split,
                output_dir=output_dir,
                crop_box=crop_box,
                sample_suffix=f"density_t{int(tile_size)}_{index:04d}",
                source_type=f"density_{int(tile_size)}",
            )
            if density_record is not None:
                density_records.append(density_record)
    return density_records


def _prepare_stage1_record(
    *,
    record: dict[str, Any],
    split: str,
    output_dir: str,
    stage1_include_full_image: bool,
    stage1_tile_size_ratios: list[float],
    stage1_min_tile_size: int,
    stage1_max_tile_size: int,
    stage1_tile_stride_ratio: float,
    stage1_density_min_instances: int,
    stage1_density_max_instances: int,
    stage1_density_max_crops_per_size: int,
) -> list[dict[str, Any]]:
    image_path = Path(record["image_path"])
    image = Image.open(image_path).convert("RGB")
    current_stage1: list[dict[str, Any]] = []
    if stage1_include_full_image:
        current_stage1.append(_build_stage1_full_image_record(record))
    current_stage1.extend(
        _enumerate_stage1_sliding_records(
            record=record,
            image=image,
            split=split,
            output_dir=Path(output_dir),
            tile_size_ratios=stage1_tile_size_ratios,
            min_tile_size=stage1_min_tile_size,
            max_tile_size=stage1_max_tile_size,
            tile_stride_ratio=stage1_tile_stride_ratio,
        )
    )
    current_stage1.extend(
        _enumerate_stage1_density_records(
            record=record,
            image=image,
            split=split,
            output_dir=Path(output_dir),
            tile_size_ratios=stage1_tile_size_ratios,
            min_tile_size=stage1_min_tile_size,
            max_tile_size=stage1_max_tile_size,
            min_instances=stage1_density_min_instances,
            max_instances=stage1_density_max_instances,
            max_crops_per_size=stage1_density_max_crops_per_size,
        )
    )
    image.close()
    return current_stage1


def _prepare_stage2_record_set(
    *,
    record: dict[str, Any],
    split: str,
    output_dir: str,
    padding_ratio: float,
    num_bins: int,
    stage2_aug_ratio: float,
    bbox_center_jitter_ratio: float,
    bbox_scale_jitter_ratio: float,
    augmentation_seed: int,
) -> list[dict[str, Any]]:
    image_path = Path(record["image_path"])
    image = Image.open(image_path).convert("RGB")
    current_stage2: list[dict[str, Any]] = []
    output_dir_path = Path(output_dir)
    for target_index, instance in enumerate(record.get("instances", [])):
        current_stage2.append(
            _build_stage2_record(
                record,
                instance,
                image=image,
                split=split,
                target_index=target_index,
                sample_suffix="",
                hint_bbox=instance["bbox"],
                hint_keypoints=[instance["keypoints"][0], instance["keypoints"][-1]],
                output_dir=output_dir_path,
                padding_ratio=padding_ratio,
                num_bins=num_bins,
                augmentation={
                    "copy_type": "clean",
                    "copy_index": 0,
                    "bbox_center_dx_px": 0.0,
                    "bbox_center_dy_px": 0.0,
                    "bbox_scale_dw_px": 0.0,
                    "bbox_scale_dh_px": 0.0,
                },
            )
        )
        if float(stage2_aug_ratio) <= 0.0:
            continue
        rng = _stable_rng(
            sample_id=record["sample_id"],
            target_index=target_index,
            aug_index=1,
            seed=augmentation_seed,
        )
        if rng.random() >= float(stage2_aug_ratio):
            continue
        noisy_record = None
        for _attempt in range(8):
            noisy_bbox, bbox_meta = _jitter_bbox(
                instance["bbox"],
                image_width=int(record["image_width"]),
                image_height=int(record["image_height"]),
                center_ratio=bbox_center_jitter_ratio,
                scale_ratio=bbox_scale_jitter_ratio,
                rng=rng,
            )
            preview_crop_box = build_padded_crop(
                image,
                bbox=noisy_bbox,
                padding_ratio=padding_ratio,
            )[1]
            if not _all_points_inside_crop(instance["keypoints"], preview_crop_box):
                continue
            noisy_record = _build_stage2_record(
                record,
                instance,
                image=image,
                split=split,
                target_index=target_index,
                sample_suffix="__aug_01",
                hint_bbox=noisy_bbox,
                hint_keypoints=[instance["keypoints"][0], instance["keypoints"][-1]],
                output_dir=output_dir_path,
                padding_ratio=padding_ratio,
                num_bins=num_bins,
                augmentation={
                    "copy_type": "noisy",
                    "copy_index": 1,
                    **bbox_meta,
                },
            )
            break
        if noisy_record is not None:
            current_stage2.append(noisy_record)
    image.close()
    return current_stage2


def _prepare_stage1_split(
    records: list[dict[str, Any]],
    *,
    split: str,
    output_dir: Path,
    num_workers: int,
    stage1_include_full_image: bool,
    stage1_tile_size_ratios: list[float],
    stage1_min_tile_size: int,
    stage1_max_tile_size: int,
    stage1_tile_stride_ratio: float,
    stage1_density_min_instances: int,
    stage1_density_max_instances: int,
    stage1_density_max_crops_per_size: int,
    stage1_dedup_iou_threshold: float,
    stats: Counter,
) -> list[dict[str, Any]]:
    stage1_records: list[dict[str, Any]] = []
    if num_workers <= 1:
        for record in records:
            current_stage1 = _prepare_stage1_record(
                record=record,
                split=split,
                output_dir=str(output_dir),
                stage1_include_full_image=stage1_include_full_image,
                stage1_tile_size_ratios=stage1_tile_size_ratios,
                stage1_min_tile_size=stage1_min_tile_size,
                stage1_max_tile_size=stage1_max_tile_size,
                stage1_tile_stride_ratio=stage1_tile_stride_ratio,
                stage1_density_min_instances=stage1_density_min_instances,
                stage1_density_max_instances=stage1_density_max_instances,
                stage1_density_max_crops_per_size=stage1_density_max_crops_per_size,
            )
            current_stage1, dropped = _deduplicate_stage1_records(
                current_stage1,
                dedup_iou_threshold=stage1_dedup_iou_threshold,
            )
            stage1_records.extend(current_stage1)
            stats[f"{split}_stage1_samples"] += len(current_stage1)
            stats[f"{split}_stage1_dedup_dropped"] += int(dropped)
            stats[f"{split}_stage1_full_image_samples"] += sum(1 for item in current_stage1 if item.get("source_type") == "full_image")
            stats[f"{split}_stage1_sliding_samples"] += sum(1 for item in current_stage1 if str(item.get("source_type", "")).startswith("sliding_"))
            stats[f"{split}_stage1_density_samples"] += sum(1 for item in current_stage1 if str(item.get("source_type", "")).startswith("density_"))
        return stage1_records

    max_workers = min(max(int(num_workers), 1), max(len(records), 1))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _prepare_stage1_record,
                record=record,
                split=split,
                output_dir=str(output_dir),
                stage1_include_full_image=stage1_include_full_image,
                stage1_tile_size_ratios=stage1_tile_size_ratios,
                stage1_min_tile_size=stage1_min_tile_size,
                stage1_max_tile_size=stage1_max_tile_size,
                stage1_tile_stride_ratio=stage1_tile_stride_ratio,
                stage1_density_min_instances=stage1_density_min_instances,
                stage1_density_max_instances=stage1_density_max_instances,
                stage1_density_max_crops_per_size=stage1_density_max_crops_per_size,
            )
            for record in records
        ]
        for future in futures:
            current_stage1 = future.result()
            current_stage1, dropped = _deduplicate_stage1_records(
                current_stage1,
                dedup_iou_threshold=stage1_dedup_iou_threshold,
            )
            stage1_records.extend(current_stage1)
            stats[f"{split}_stage1_samples"] += len(current_stage1)
            stats[f"{split}_stage1_dedup_dropped"] += int(dropped)
            stats[f"{split}_stage1_full_image_samples"] += sum(1 for item in current_stage1 if item.get("source_type") == "full_image")
            stats[f"{split}_stage1_sliding_samples"] += sum(1 for item in current_stage1 if str(item.get("source_type", "")).startswith("sliding_"))
            stats[f"{split}_stage1_density_samples"] += sum(1 for item in current_stage1 if str(item.get("source_type", "")).startswith("density_"))
    return stage1_records


def _prepare_stage2_split(
    records: list[dict[str, Any]],
    *,
    split: str,
    output_dir: Path,
    padding_ratio: float,
    num_bins: int,
    num_workers: int,
    stage2_aug_ratio: float,
    bbox_center_jitter_ratio: float,
    bbox_scale_jitter_ratio: float,
    augmentation_seed: int,
    stats: Counter,
) -> list[dict[str, Any]]:
    stage2_records: list[dict[str, Any]] = []
    if num_workers <= 1:
        for record in records:
            current_stage2 = _prepare_stage2_record_set(
                record=record,
                split=split,
                output_dir=str(output_dir),
                padding_ratio=padding_ratio,
                num_bins=num_bins,
                stage2_aug_ratio=stage2_aug_ratio,
                bbox_center_jitter_ratio=bbox_center_jitter_ratio,
                bbox_scale_jitter_ratio=bbox_scale_jitter_ratio,
                augmentation_seed=augmentation_seed,
            )
            stage2_records.extend(current_stage2)
            stats[f"{split}_stage2_samples"] += len(current_stage2)
        return stage2_records

    max_workers = min(max(int(num_workers), 1), max(len(records), 1))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _prepare_stage2_record_set,
                record=record,
                split=split,
                output_dir=str(output_dir),
                padding_ratio=padding_ratio,
                num_bins=num_bins,
                stage2_aug_ratio=stage2_aug_ratio,
                bbox_center_jitter_ratio=bbox_center_jitter_ratio,
                bbox_scale_jitter_ratio=bbox_scale_jitter_ratio,
                augmentation_seed=augmentation_seed,
            )
            for record in records
        ]
        for future in futures:
            current_stage2 = future.result()
            stage2_records.extend(current_stage2)
            stats[f"{split}_stage2_samples"] += len(current_stage2)
    return stage2_records


def prepare_stage1_data(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    num_workers: int | None = None,
    stage1_include_full_image: bool = True,
    stage1_tile_size_ratios: list[float] | tuple[float, ...] | None = None,
    stage1_min_tile_size: int = 512,
    stage1_max_tile_size: int = 1280,
    stage1_tile_stride_ratio: float = 0.75,
    stage1_density_min_instances: int = 5,
    stage1_density_max_instances: int = 30,
    stage1_density_max_crops_per_size: int = 8,
    stage1_dedup_iou_threshold: float = 0.9,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    resolved_workers = max(int(num_workers or os.cpu_count() or 1), 1)
    resolved_tile_size_ratios = _parse_float_sequence(stage1_tile_size_ratios, default=[0.35, 0.5])
    shutil.rmtree(output_dir / "stage1" / "images", ignore_errors=True)
    train_records = load_jsonl(input_dir / "train.jsonl")
    val_records = load_jsonl(input_dir / "val.jsonl")
    stats: Counter = Counter()

    stage1_train = _prepare_stage1_split(
        train_records,
        split="train",
        output_dir=output_dir,
        num_workers=resolved_workers,
        stage1_include_full_image=stage1_include_full_image,
        stage1_tile_size_ratios=resolved_tile_size_ratios,
        stage1_min_tile_size=int(stage1_min_tile_size),
        stage1_max_tile_size=int(stage1_max_tile_size),
        stage1_tile_stride_ratio=stage1_tile_stride_ratio,
        stage1_density_min_instances=stage1_density_min_instances,
        stage1_density_max_instances=stage1_density_max_instances,
        stage1_density_max_crops_per_size=stage1_density_max_crops_per_size,
        stage1_dedup_iou_threshold=stage1_dedup_iou_threshold,
        stats=stats,
    )
    stage1_val = _prepare_stage1_split(
        val_records,
        split="val",
        output_dir=output_dir,
        num_workers=resolved_workers,
        stage1_include_full_image=stage1_include_full_image,
        stage1_tile_size_ratios=resolved_tile_size_ratios,
        stage1_min_tile_size=int(stage1_min_tile_size),
        stage1_max_tile_size=int(stage1_max_tile_size),
        stage1_tile_stride_ratio=stage1_tile_stride_ratio,
        stage1_density_min_instances=stage1_density_min_instances,
        stage1_density_max_instances=stage1_density_max_instances,
        stage1_density_max_crops_per_size=stage1_density_max_crops_per_size,
        stage1_dedup_iou_threshold=stage1_dedup_iou_threshold,
        stats=stats,
    )

    write_jsonl(output_dir / "stage1" / "train.jsonl", stage1_train)
    write_jsonl(output_dir / "stage1" / "val.jsonl", stage1_val)

    report = {
        "num_workers": int(resolved_workers),
        "stage1_include_full_image": bool(stage1_include_full_image),
        "stage1_tile_size_ratios": resolved_tile_size_ratios,
        "stage1_min_tile_size": int(stage1_min_tile_size),
        "stage1_max_tile_size": int(stage1_max_tile_size),
        "stage1_tile_stride_ratio": float(stage1_tile_stride_ratio),
        "stage1_density_min_instances": int(stage1_density_min_instances),
        "stage1_density_max_instances": int(stage1_density_max_instances),
        "stage1_density_max_crops_per_size": int(stage1_density_max_crops_per_size),
        "stage1_dedup_iou_threshold": float(stage1_dedup_iou_threshold),
        "stage1_train_samples": len(stage1_train),
        "stage1_val_samples": len(stage1_val),
        "counts": dict(stats),
    }
    write_json(output_dir / "reports" / "prepare_stage1_report.json", report)
    return report


def prepare_stage2_data(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    padding_ratio: float = 0.3,
    num_bins: int = 1000,
    num_workers: int | None = None,
    stage2_aug_ratio: float = 0.0,
    bbox_center_jitter_ratio: float = 0.03,
    bbox_scale_jitter_ratio: float = 0.05,
    augmentation_seed: int = 42,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    resolved_workers = max(int(num_workers or os.cpu_count() or 1), 1)
    shutil.rmtree(output_dir / "stage2" / "images", ignore_errors=True)
    train_records = load_jsonl(input_dir / "train.jsonl")
    val_records = load_jsonl(input_dir / "val.jsonl")
    stats: Counter = Counter()

    stage2_train = _prepare_stage2_split(
        train_records,
        split="train",
        output_dir=output_dir,
        padding_ratio=padding_ratio,
        num_bins=num_bins,
        num_workers=resolved_workers,
        stage2_aug_ratio=stage2_aug_ratio,
        bbox_center_jitter_ratio=bbox_center_jitter_ratio,
        bbox_scale_jitter_ratio=bbox_scale_jitter_ratio,
        augmentation_seed=augmentation_seed,
        stats=stats,
    )
    stage2_val = _prepare_stage2_split(
        val_records,
        split="val",
        output_dir=output_dir,
        padding_ratio=padding_ratio,
        num_bins=num_bins,
        num_workers=resolved_workers,
        stage2_aug_ratio=0.0,
        bbox_center_jitter_ratio=bbox_center_jitter_ratio,
        bbox_scale_jitter_ratio=bbox_scale_jitter_ratio,
        augmentation_seed=augmentation_seed,
        stats=stats,
    )

    write_jsonl(output_dir / "stage2" / "train.jsonl", stage2_train)
    write_jsonl(output_dir / "stage2" / "val.jsonl", stage2_val)

    report = {
        "padding_ratio": float(padding_ratio),
        "num_bins": int(num_bins),
        "num_workers": int(resolved_workers),
        "stage2_train_aug_ratio": float(stage2_aug_ratio),
        "stage2_val_aug_ratio": 0.0,
        "bbox_center_jitter_ratio": float(bbox_center_jitter_ratio),
        "bbox_scale_jitter_ratio": float(bbox_scale_jitter_ratio),
        "augmentation_seed": int(augmentation_seed),
        "stage2_train_samples": len(stage2_train),
        "stage2_val_samples": len(stage2_val),
        "counts": dict(stats),
    }
    write_json(output_dir / "reports" / "prepare_stage2_report.json", report)
    return report

