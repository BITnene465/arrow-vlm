from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from vlm_det.utils.io import write_json, write_jsonl


VISIBILITY_MAP = {
    "p2": "visible",
    "2": "visible",
    "p1": "occluded",
    "1": "occluded",
}

# Some paper figures are extremely large. During data preparation we only need
# image metadata such as width/height, so disable Pillow's decompression bomb
# guard here to avoid aborting on legitimate oversized figures.
Image.MAX_IMAGE_PIXELS = None


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _shape_to_bbox(points: list[list[float]]) -> list[float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _resolve_image_path(image_path_hint: str, image_dir: Path) -> Path:
    stem = Path(image_path_hint).stem
    matches = sorted(image_dir.glob(f"{stem}.*"))
    if not matches:
        raise FileNotFoundError(f"Could not resolve image for stem: {stem}")
    return matches[0]


def _load_labelme(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_sample(json_path: Path, image_dir: Path, stats: Counter) -> dict[str, Any]:
    raw = _load_labelme(json_path)
    image_path = _resolve_image_path(raw["imagePath"], image_dir)
    with Image.open(image_path) as image:
        width, height = image.size

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for shape in raw.get("shapes", []):
        group_id = shape.get("group_id")
        if group_id is None:
            stats["instances_skipped_missing_group_id"] += 1
            continue
        groups[int(group_id)].append(shape)

    instances = []
    for group_id, shapes in groups.items():
        rectangles = [shape for shape in shapes if shape.get("shape_type") == "rectangle"]
        points = [shape for shape in shapes if shape.get("shape_type") == "point"]

        if len(rectangles) == 0:
            stats["instances_dropped_missing_bbox"] += 1
            continue
        if len(rectangles) > 1:
            stats["instances_dropped_multiple_bbox"] += 1
            continue
        if len(points) < 2:
            stats["instances_dropped_too_few_points"] += 1
            continue

        raw_bbox = _shape_to_bbox(rectangles[0]["points"])
        bbox = [
            _clip(raw_bbox[0], 0.0, width - 1),
            _clip(raw_bbox[1], 0.0, height - 1),
            _clip(raw_bbox[2], 0.0, width - 1),
            _clip(raw_bbox[3], 0.0, height - 1),
        ]
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            stats["instances_dropped_invalid_bbox"] += 1
            continue

        raw_keypoints = []
        keypoints = []
        invalid_visibility = False
        for point in points:
            label = str(point.get("label"))
            if label not in VISIBILITY_MAP:
                invalid_visibility = True
                break
            x_value = float(point["points"][0][0])
            y_value = float(point["points"][0][1])
            visibility = VISIBILITY_MAP[label]
            raw_keypoints.append([x_value, y_value, visibility])
            keypoints.append(
                [
                    _clip(x_value, 0.0, width - 1),
                    _clip(y_value, 0.0, height - 1),
                    visibility,
                ]
            )
        if invalid_visibility:
            stats["instances_dropped_unknown_visibility"] += 1
            continue

        instances.append(
            {
                "group_id": group_id,
                "raw_bbox": raw_bbox,
                "bbox": bbox,
                "raw_keypoints": raw_keypoints,
                "keypoints": keypoints,
            }
        )
        stats["instances_kept"] += 1

    return {
        "sample_id": image_path.stem,
        "image_path": str(image_path),
        "image_width": width,
        "image_height": height,
        "instances": instances,
    }


def prepare_normalized_dataset(
    raw_json_dir: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> dict[str, Any]:
    raw_json_dir = Path(raw_json_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    stats: Counter = Counter()

    samples = []
    for json_path in sorted(raw_json_dir.glob("*.json")):
        stats["json_files"] += 1
        sample = _normalize_sample(json_path, image_dir, stats)
        samples.append(sample)

    random.Random(seed).shuffle(samples)
    split_index = int(len(samples) * train_ratio)
    train_samples = samples[:split_index]
    val_samples = samples[split_index:]

    # output_dir is the final normalized-data directory itself, e.g.
    # data/processed/normalized. Do not append another "normalized" level.
    normalized_dir = output_dir
    reports_dir = output_dir.parent / "reports"
    write_jsonl(normalized_dir / "train.jsonl", train_samples)
    write_jsonl(normalized_dir / "val.jsonl", val_samples)

    report = {
        "counts": dict(stats),
        "num_samples": len(samples),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "train_ratio": train_ratio,
        "seed": seed,
    }
    split_manifest = {
        "seed": seed,
        "train_ids": [sample["sample_id"] for sample in train_samples],
        "val_ids": [sample["sample_id"] for sample in val_samples],
    }
    write_json(reports_dir / "data_cleaning_report.json", report)
    write_json(reports_dir / "split_manifest.json", split_manifest)
    return report
