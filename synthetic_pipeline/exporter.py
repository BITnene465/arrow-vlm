from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from vlm_det.utils.io import write_json, write_jsonl

from synthetic_pipeline.schema import SyntheticSample


class DatasetExporter:
    def __init__(self, output_dir: str | Path, debug_cfg: dict[str, Any]) -> None:
        self.output_dir = Path(output_dir)
        self.debug_cfg = debug_cfg
        self._saved_debug_counts: dict[str, int] = {}

    def save_sample(self, sample: SyntheticSample, image: Image.Image) -> dict[str, Any]:
        image_path = self.output_dir / "images" / sample.scene.split / f"{sample.scene.sample_id}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path, quality=95)
        record = sample.to_record(image_path=self._image_path_value(image_path))
        self._save_debug_visualization(sample, image)
        return record

    def finalize(self, train_records: list[dict[str, Any]], val_records: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
        write_jsonl(self.output_dir / "train.jsonl", train_records)
        write_jsonl(self.output_dir / "val.jsonl", val_records)
        summary = {
            "config": cfg,
            "train": summarize_records(train_records),
            "val": summarize_records(val_records),
        }
        write_json(self.output_dir / "manifest.json", summary)
        return summary

    def _save_debug_visualization(self, sample: SyntheticSample, image: Image.Image) -> None:
        if not bool(self.debug_cfg.get("save_visualizations", False)):
            return
        split = sample.scene.split
        count = self._saved_debug_counts.get(split, 0)
        if count >= int(self.debug_cfg.get("max_saved_samples_per_split", 20)):
            return
        debug_dir = self.output_dir / "debug" / split
        debug_dir.mkdir(parents=True, exist_ok=True)
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for instance in sample.instances:
            bbox = instance.bbox
            draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), outline=(255, 0, 0), width=2)
            draw.text((bbox[0] + 4, bbox[1] + 4), instance.label, fill=(255, 0, 0))
            for point in instance.keypoints:
                x_value, y_value = point
                draw.ellipse((x_value - 3, y_value - 3, x_value + 3, y_value + 3), fill=(0, 180, 255))
        overlay.save(debug_dir / f"{sample.scene.sample_id}.jpg", quality=95)
        self._saved_debug_counts[split] = count + 1

    def _image_path_value(self, image_path: Path) -> str:
        if self.output_dir.is_absolute():
            return str(image_path)
        return str(self.output_dir / "images" / image_path.parent.name / image_path.name)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_instances = sum(len(record["instances"]) for record in records)
    total_points = sum(len(instance["keypoints"]) for record in records for instance in record["instances"])
    renderer_counts: dict[str, int] = {}
    asset_backgrounds = 0
    for record in records:
        renderer_name = str(record.get("renderer_name", "unknown"))
        renderer_counts[renderer_name] = renderer_counts.get(renderer_name, 0) + 1
        render_meta = record.get("render_meta", {})
        if render_meta.get("asset_backed_background"):
            asset_backgrounds += 1
    return {
        "num_samples": len(records),
        "num_instances": total_instances,
        "avg_instances_per_image": round(total_instances / max(len(records), 1), 3),
        "avg_points_per_instance": round(total_points / max(total_instances, 1), 3),
        "renderer_counts": renderer_counts,
        "asset_backed_backgrounds": asset_backgrounds,
    }


def dumps_summary(summary: dict[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2)
