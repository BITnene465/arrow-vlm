from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from PIL import Image

from vlm_det.utils.io import load_jsonl

from synthetic_pipeline.scene_sampler import clamp

Image.MAX_IMAGE_PIXELS = None


class AssetBank:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.records = self._load_records(cfg)
        self.record_by_sample_id = {
            str(record["sample_id"]): record
            for record in self.records
            if "sample_id" in record
        }
        self.instance_assets = self._build_instance_index(self.records)

    def is_available(self) -> bool:
        return bool(self.records) and bool(self.instance_assets)

    def sample_background(self, width: int, height: int, rng: random.Random) -> tuple[Image.Image, dict[str, Any]] | None:
        if not self.records:
            return None
        retries = int(self.cfg["background_retry"])
        for _ in range(retries):
            record = rng.choice(self.records)
            image = self._open_image(record["image_path"])
            crop = self._sample_clean_aspect_crop(
                image=image,
                instances=record.get("instances", []),
                target_width=width,
                target_height=height,
                rng=rng,
                min_crop_ratio=float(self.cfg["background_min_crop_ratio"]),
            )
            if crop is None:
                continue
            patch, crop_bbox = crop
            return patch.resize((width, height), Image.Resampling.BICUBIC), {
                "background_source_sample_id": record["sample_id"],
                "background_source_crop_bbox": crop_bbox,
            }
        return None

    def sample_context_patch(
        self,
        rng: random.Random,
    ) -> tuple[Image.Image, dict[str, Any]] | None:
        if not self.records:
            return None
        patch_size_range = self.cfg["context_patch_size_range"]
        retries = int(self.cfg["negative_crop_retry"])
        for _ in range(retries):
            record = rng.choice(self.records)
            image = self._open_image(record["image_path"])
            crop_w = rng.randint(int(patch_size_range[0]), int(patch_size_range[1]))
            crop_h = rng.randint(int(patch_size_range[0]), int(patch_size_range[1]))
            crop = self._sample_negative_crop(
                image=image,
                instances=record.get("instances", []),
                crop_w=min(crop_w, image.width),
                crop_h=min(crop_h, image.height),
                rng=rng,
            )
            if crop is None:
                continue
            patch, crop_bbox = crop
            return patch, {
                "source_sample_id": record["sample_id"],
                "source_crop_bbox": crop_bbox,
            }
        return None

    def sample_arrow_patch(
        self,
        rng: random.Random,
    ) -> tuple[Image.Image, dict[str, Any]] | None:
        if not self.instance_assets:
            return None
        retries = int(self.cfg["arrow_patch_retry"])
        min_context_px = int(self.cfg["arrow_patch_min_context_px"])
        min_pad_ratio, max_pad_ratio = self.cfg["arrow_patch_pad_ratio_range"]
        max_instances = int(self.cfg["arrow_patch_max_instances_per_crop"])
        for _ in range(retries):
            anchor = rng.choice(self.instance_assets)
            record = self.record_by_sample_id.get(str(anchor["sample_id"]))
            if record is None:
                continue
            image = self._open_image(record["image_path"])
            bbox = [float(value) for value in anchor["bbox"]]
            bbox_w = max(bbox[2] - bbox[0], 1.0)
            bbox_h = max(bbox[3] - bbox[1], 1.0)
            pad_ratio = rng.uniform(float(min_pad_ratio), float(max_pad_ratio))
            pad = max(max(bbox_w, bbox_h) * pad_ratio, float(min_context_px))
            crop_bbox = [
                clamp(bbox[0] - pad, 0, image.width - 1),
                clamp(bbox[1] - pad, 0, image.height - 1),
                clamp(bbox[2] + pad, 1, image.width),
                clamp(bbox[3] + pad, 1, image.height),
            ]
            crop_x1 = int(crop_bbox[0])
            crop_y1 = int(crop_bbox[1])
            crop_x2 = int(max(crop_bbox[2], crop_x1 + 1))
            crop_y2 = int(max(crop_bbox[3], crop_y1 + 1))
            transformed_instances = self._instances_fully_inside_crop(
                record.get("instances", []),
                [float(crop_x1), float(crop_y1), float(crop_x2), float(crop_y2)],
                sample_id=str(record["sample_id"]),
            )
            if not transformed_instances:
                continue
            if len(transformed_instances) > max_instances:
                continue
            patch = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            union_bbox = self._union_bbox([instance["bbox"] for instance in transformed_instances])
            return patch, {
                "source_sample_id": record["sample_id"],
                "source_crop_bbox": [crop_x1, crop_y1, crop_x2, crop_y2],
                "instances": transformed_instances,
                "instance_count": len(transformed_instances),
                "union_bbox": union_bbox,
            }
        return None

    def _load_records(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for raw_path in cfg["processed_jsonl_paths"]:
            path = Path(raw_path)
            if not path.exists():
                continue
            records.extend(load_jsonl(path))
        max_records = cfg.get("max_records")
        if max_records is not None:
            records = records[: int(max_records)]
        return records

    @staticmethod
    def _build_instance_index(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        assets: list[dict[str, Any]] = []
        for record in records:
            for index, instance in enumerate(record.get("instances", [])):
                assets.append(
                    {
                        "sample_id": record["sample_id"],
                        "image_path": record["image_path"],
                        "instance_id": f"{record['sample_id']}:{index}",
                        "label": str(instance.get("label", "single_arrow")),
                        "bbox": [float(value) for value in instance["bbox"]],
                        "keypoints": [[float(point[0]), float(point[1])] for point in instance["keypoints"]],
                    }
                )
        return assets

    @staticmethod
    def _open_image(path: str) -> Image.Image:
        return Image.open(Path(path)).convert("RGB")

    def _sample_negative_crop(
        self,
        image: Image.Image,
        instances: list[dict[str, Any]],
        crop_w: int,
        crop_h: int,
        rng: random.Random,
    ) -> tuple[Image.Image, list[int]] | None:
        if crop_w <= 1 or crop_h <= 1:
            return None
        image_w, image_h = image.size
        if crop_w > image_w or crop_h > image_h:
            return None
        for _ in range(int(self.cfg["negative_crop_retry"])):
            x1 = rng.randint(0, image_w - crop_w)
            y1 = rng.randint(0, image_h - crop_h)
            crop_bbox = [float(x1), float(y1), float(x1 + crop_w), float(y1 + crop_h)]
            if any(self._crop_contains_instance(crop_bbox, instance) for instance in instances):
                continue
            return image.crop((x1, y1, x1 + crop_w, y1 + crop_h)), [x1, y1, x1 + crop_w, y1 + crop_h]
        return None

    def _sample_clean_aspect_crop(
        self,
        image: Image.Image,
        instances: list[dict[str, Any]],
        target_width: int,
        target_height: int,
        rng: random.Random,
        min_crop_ratio: float,
    ) -> tuple[Image.Image, list[int]] | None:
        image_w, image_h = image.size
        if image_w < 2 or image_h < 2:
            return None
        target_aspect = float(target_width) / float(target_height)
        image_aspect = float(image_w) / float(image_h)
        for _ in range(int(self.cfg["background_retry"])):
            scale = rng.uniform(min_crop_ratio, 1.0)
            if image_aspect >= target_aspect:
                crop_h = max(int(image_h * scale), 2)
                crop_w = min(int(round(crop_h * target_aspect)), image_w)
                crop_h = min(int(round(crop_w / target_aspect)), image_h)
            else:
                crop_w = max(int(image_w * scale), 2)
                crop_h = min(int(round(crop_w / target_aspect)), image_h)
                crop_w = min(int(round(crop_h * target_aspect)), image_w)
            if crop_w < 2 or crop_h < 2:
                continue
            x1 = rng.randint(0, max(image_w - crop_w, 0))
            y1 = rng.randint(0, max(image_h - crop_h, 0))
            crop_bbox = [float(x1), float(y1), float(x1 + crop_w), float(y1 + crop_h)]
            if any(self._crop_contains_instance(crop_bbox, instance) for instance in instances):
                continue
            return image.crop((x1, y1, x1 + crop_w, y1 + crop_h)), [x1, y1, x1 + crop_w, y1 + crop_h]
        return None

    @staticmethod
    def _instances_fully_inside_crop(
        instances: list[dict[str, Any]],
        crop_bbox: list[float],
        *,
        sample_id: str,
    ) -> list[dict[str, Any]]:
        included: list[dict[str, Any]] = []
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
        for index, instance in enumerate(instances):
            bbox = [float(value) for value in instance["bbox"]]
            if bbox[0] < crop_x1 or bbox[1] < crop_y1 or bbox[2] > crop_x2 or bbox[3] > crop_y2:
                continue
            keypoints = []
            for point in instance.get("keypoints", []):
                keypoints.append(
                    [
                        round(float(point[0]) - crop_x1, 2),
                        round(float(point[1]) - crop_y1, 2),
                    ]
                )
            included.append(
                {
                    "label": str(instance.get("label", "single_arrow")),
                    "bbox": [
                        round(bbox[0] - crop_x1, 2),
                        round(bbox[1] - crop_y1, 2),
                        round(bbox[2] - crop_x1, 2),
                        round(bbox[3] - crop_y1, 2),
                    ],
                    "keypoints": keypoints,
                    "source_asset_sample_id": sample_id,
                    "source_asset_instance_id": f"{sample_id}:{index}",
                }
            )
        return included

    @staticmethod
    def _crop_to_aspect(
        image: Image.Image,
        target_width: int,
        target_height: int,
        rng: random.Random,
        min_crop_ratio: float,
    ) -> Image.Image | None:
        image_w, image_h = image.size
        if image_w < 2 or image_h < 2:
            return None
        target_aspect = float(target_width) / float(target_height)
        image_aspect = float(image_w) / float(image_h)
        scale = rng.uniform(min_crop_ratio, 1.0)
        if image_aspect >= target_aspect:
            crop_h = max(int(image_h * scale), 2)
            crop_w = min(int(round(crop_h * target_aspect)), image_w)
            crop_h = min(int(round(crop_w / target_aspect)), image_h)
        else:
            crop_w = max(int(image_w * scale), 2)
            crop_h = min(int(round(crop_w / target_aspect)), image_h)
            crop_w = min(int(round(crop_h * target_aspect)), image_w)
        if crop_w < 2 or crop_h < 2:
            return None
        x1 = rng.randint(0, max(image_w - crop_w, 0))
        y1 = rng.randint(0, max(image_h - crop_h, 0))
        return image.crop((x1, y1, x1 + crop_w, y1 + crop_h))

    def _crop_contains_instance(self, crop_bbox: list[float], instance: dict[str, Any]) -> bool:
        margin_ratio = float(self.cfg["instance_exclusion_margin_ratio"])
        bbox = [float(value) for value in instance["bbox"]]
        bbox_w = max(bbox[2] - bbox[0], 1.0)
        bbox_h = max(bbox[3] - bbox[1], 1.0)
        margin = max(max(bbox_w, bbox_h) * margin_ratio, 4.0)
        expanded_bbox = [
            bbox[0] - margin,
            bbox[1] - margin,
            bbox[2] + margin,
            bbox[3] + margin,
        ]
        if self._boxes_intersect(crop_bbox, expanded_bbox):
            return True
        for point in instance.get("keypoints", []):
            x_value = float(point[0])
            y_value = float(point[1])
            if crop_bbox[0] <= x_value <= crop_bbox[2] and crop_bbox[1] <= y_value <= crop_bbox[3]:
                return True
        return False

    @staticmethod
    def _boxes_intersect(box_a: list[float], box_b: list[float]) -> bool:
        return not (
            box_a[2] <= box_b[0]
            or box_a[0] >= box_b[2]
            or box_a[3] <= box_b[1]
            or box_a[1] >= box_b[3]
        )

    @staticmethod
    def _union_bbox(boxes: list[list[float]]) -> list[float]:
        x_values = [float(box[0]) for box in boxes] + [float(box[2]) for box in boxes]
        y_values = [float(box[1]) for box in boxes] + [float(box[3]) for box in boxes]
        return [
            round(min(x_values), 2),
            round(min(y_values), 2),
            round(max(x_values), 2),
            round(max(y_values), 2),
        ]
