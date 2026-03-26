from __future__ import annotations

import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from vlm_det.data.ordering import sort_instances_canonical
from vlm_det.utils.io import ensure_dir, load_jsonl, write_json, write_jsonl

Image.MAX_IMAGE_PIXELS = None


def _quantize(value: float, size: int, num_bins: int) -> int:
    size = max(int(size), 1)
    if size == 1:
        return 0
    clipped = min(max(float(value), 0.0), float(size - 1))
    return int(round(clipped / float(size - 1) * float(num_bins - 1)))


def _build_stage1_instance(instance: dict[str, Any]) -> dict[str, Any]:
    keypoints = instance["keypoints"]
    return {
        "label": instance["label"],
        "bbox": list(instance["bbox"]),
        "keypoints": [list(keypoints[0]), list(keypoints[-1])],
    }


def build_stage2_prompt(
    *,
    label: str,
    bbox_2d: list[int],
    hint_keypoints_2d: list[list[int]],
) -> str:
    hint_payload = {
        "label": label,
        "bbox_2d": bbox_2d,
        "keypoints_2d": hint_keypoints_2d,
    }
    hint_json = json.dumps(hint_payload, ensure_ascii=False, separators=(",", ":"))
    return (
        "The cropped image may contain multiple arrows.\n"
        f"The target arrow is specified by:\n{hint_json}\n"
        "Output only the complete keypoints_2d of this same arrow as a JSON array of points.\n"
        "Normalize every coordinate to an integer in [0,999].\n"
        "For single_arrow, keypoints must be ordered from tail to head.\n"
        "For double_arrow, keypoints[0] must be the upper-left head and keypoints[-1] the other head.\n"
        "Do not output markdown or extra text."
    )


STAGE2_USER_PROMPT_TEMPLATE = (
    "The cropped image may contain multiple arrows.\n"
    "The target arrow is specified by:\n"
    "{\"label\":\"{{label}}\",\"bbox_2d\":{{bbox_2d}},\"keypoints_2d\":{{keypoints_2d}}}\n"
    "Output only the complete keypoints_2d of this same arrow as a JSON array of points.\n"
    "Normalize every coordinate to an integer in [0,999].\n"
    "For single_arrow, keypoints must be ordered from tail to head.\n"
    "For double_arrow, keypoints[0] must be the upper-left head and keypoints[-1] the other head.\n"
    "Do not output markdown or extra text."
)


def _encode_stage2_target(keypoints_2d: list[list[int]]) -> str:
    return json.dumps(keypoints_2d, ensure_ascii=False, separators=(",", ":"))


def _round_bbox(bbox: list[float]) -> list[float]:
    return [round(float(value), 4) for value in bbox]


def _round_keypoints(keypoints: list[list[float]]) -> list[list[float]]:
    return [[round(float(x), 4), round(float(y), 4)] for x, y in keypoints]


def build_padded_crop(
    image: Image.Image,
    *,
    bbox: list[float],
    padding_ratio: float,
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
    output_dir: Path,
    padding_ratio: float,
    num_bins: int,
) -> dict[str, Any]:
    crop_image, crop_box = build_padded_crop(
        image,
        bbox=instance["bbox"],
        padding_ratio=padding_ratio,
    )
    crop_width, crop_height = crop_image.size
    crop_dir = ensure_dir(output_dir / "stage2" / "images" / split)
    crop_name = f"{record['sample_id']}__inst_{target_index:04d}.png"
    crop_path = crop_dir / crop_name
    crop_image.save(crop_path)

    local_bbox = to_crop_local_bbox(instance["bbox"], crop_box)
    local_keypoints = to_crop_local_keypoints(instance["keypoints"], crop_box)
    local_hint_keypoints = [local_keypoints[0], local_keypoints[-1]]
    local_bbox_2d = quantize_bbox_2d(local_bbox, crop_width, crop_height, num_bins)
    local_hint_keypoints_2d = quantize_keypoints_2d(
        local_hint_keypoints,
        crop_width,
        crop_height,
        num_bins,
    )
    local_full_keypoints_2d = quantize_keypoints_2d(
        local_keypoints,
        crop_width,
        crop_height,
        num_bins,
    )

    return {
        "task_type": "two_stage_stage2",
        "sample_id": f"{record['sample_id']}__inst_{target_index:04d}",
        "source_sample_id": record["sample_id"],
        "target_index": int(target_index),
        "image_path": str(crop_path),
        "image_width": int(crop_width),
        "image_height": int(crop_height),
        "system_prompt": "",
        "target_text": _encode_stage2_target(local_full_keypoints_2d),
        "gt_struct": {
            "task_type": "two_stage_stage2",
            "label": instance["label"],
            "keypoints": _round_keypoints(local_keypoints),
            "keypoints_2d": local_full_keypoints_2d,
        },
        "instances": [
            {
                "label": instance["label"],
                "bbox": _round_bbox(local_bbox),
                "keypoints": _round_keypoints(local_keypoints),
            }
        ],
        "condition": {
            "label": instance["label"],
            "bbox": _round_bbox(local_bbox),
            "bbox_2d": local_bbox_2d,
            "keypoints": _round_keypoints(local_hint_keypoints),
            "keypoints_2d": local_hint_keypoints_2d,
        },
        "crop_box": crop_box,
        "augmentation": {
            "bbox_jitter_px": 0.0,
            "endpoint_jitter_px": 0.0,
        },
    }


def _build_stage1_record(record: dict[str, Any]) -> dict[str, Any]:
    instances = [_build_stage1_instance(instance) for instance in record.get("instances", [])]
    instances = sort_instances_canonical(instances)
    return {
        "task_type": "two_stage_stage1",
        "sample_id": record["sample_id"],
        "image_path": record["image_path"],
        "image_width": int(record["image_width"]),
        "image_height": int(record["image_height"]),
        "instances": instances,
    }


def _prepare_split(
    records: list[dict[str, Any]],
    *,
    split: str,
    output_dir: Path,
    padding_ratio: float,
    num_bins: int,
    num_workers: int,
    stats: Counter,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage1_records: list[dict[str, Any]] = []
    stage2_records: list[dict[str, Any]] = []
    if num_workers <= 1:
        for record in records:
            current_stage1, current_stage2 = _prepare_record(
                record=record,
                split=split,
                output_dir=str(output_dir),
                padding_ratio=padding_ratio,
                num_bins=num_bins,
            )
            stage1_records.append(current_stage1)
            stage2_records.extend(current_stage2)
            stats[f"{split}_stage1_samples"] += 1
            stats[f"{split}_stage2_samples"] += len(current_stage2)
        return stage1_records, stage2_records

    max_workers = min(max(int(num_workers), 1), max(len(records), 1))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _prepare_record,
                record=record,
                split=split,
                output_dir=str(output_dir),
                padding_ratio=padding_ratio,
                num_bins=num_bins,
            )
            for record in records
        ]
        for future in futures:
            current_stage1, current_stage2 = future.result()
            stage1_records.append(current_stage1)
            stage2_records.extend(current_stage2)
            stats[f"{split}_stage1_samples"] += 1
            stats[f"{split}_stage2_samples"] += len(current_stage2)
    return stage1_records, stage2_records


def _prepare_record(
    *,
    record: dict[str, Any],
    split: str,
    output_dir: str,
    padding_ratio: float,
    num_bins: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image_path = Path(record["image_path"])
    image = Image.open(image_path).convert("RGB")
    current_stage1 = _build_stage1_record(record)
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
                output_dir=output_dir_path,
                padding_ratio=padding_ratio,
                num_bins=num_bins,
            )
        )
    image.close()
    return current_stage1, current_stage2


def prepare_two_stage_data(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    padding_ratio: float = 0.5,
    num_bins: int = 1000,
    num_workers: int | None = None,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    resolved_workers = max(int(num_workers or os.cpu_count() or 1), 1)
    train_records = load_jsonl(input_dir / "train.jsonl")
    val_records = load_jsonl(input_dir / "val.jsonl")
    stats: Counter = Counter()

    stage1_train, stage2_train = _prepare_split(
        train_records,
        split="train",
        output_dir=output_dir,
        padding_ratio=padding_ratio,
        num_bins=num_bins,
        num_workers=resolved_workers,
        stats=stats,
    )
    stage1_val, stage2_val = _prepare_split(
        val_records,
        split="val",
        output_dir=output_dir,
        padding_ratio=padding_ratio,
        num_bins=num_bins,
        num_workers=resolved_workers,
        stats=stats,
    )

    write_jsonl(output_dir / "stage1" / "train.jsonl", stage1_train)
    write_jsonl(output_dir / "stage1" / "val.jsonl", stage1_val)
    write_jsonl(output_dir / "stage2" / "train.jsonl", stage2_train)
    write_jsonl(output_dir / "stage2" / "val.jsonl", stage2_val)

    report = {
        "padding_ratio": float(padding_ratio),
        "num_bins": int(num_bins),
        "num_workers": int(resolved_workers),
        "stage1_train_samples": len(stage1_train),
        "stage1_val_samples": len(stage1_val),
        "stage2_train_samples": len(stage2_train),
        "stage2_val_samples": len(stage2_val),
        "counts": dict(stats),
    }
    write_json(output_dir / "reports" / "prepare_two_stage_report.json", report)
    return report
