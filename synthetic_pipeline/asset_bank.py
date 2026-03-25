from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vlm_det.utils.io import load_jsonl

from synthetic_pipeline.scene_sampler import bbox_iou, clamp

Image.MAX_IMAGE_PIXELS = None


class AssetBank:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.records = self._load_records(cfg)
        self.instance_assets = self._build_instance_index(self.records)

    def is_available(self) -> bool:
        return bool(self.records) and bool(self.instance_assets)

    def sample_background(self, width: int, height: int, rng: random.Random) -> tuple[Image.Image, dict[str, Any]] | None:
        if not self.records:
            return None
        record = rng.choice(self.records)
        image = self._open_image(record["image_path"])
        crop = self._crop_to_aspect(
            image=image,
            target_width=width,
            target_height=height,
            rng=rng,
            min_crop_ratio=float(self.cfg["background_min_crop_ratio"]),
        )
        if crop is None:
            return None
        return crop.resize((width, height), Image.Resampling.BICUBIC), {
            "background_source_sample_id": record["sample_id"],
        }

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

    def sample_texture_patch(
        self,
        rng: random.Random,
    ) -> tuple[Image.Image, dict[str, Any]] | None:
        if not self.records:
            return None
        texture_range = self.cfg["texture_patch_size_range"]
        retries = int(self.cfg["negative_crop_retry"])
        for _ in range(retries):
            record = rng.choice(self.records)
            image = self._open_image(record["image_path"])
            patch_size = rng.randint(int(texture_range[0]), int(texture_range[1]))
            crop = self._sample_negative_crop(
                image=image,
                instances=record.get("instances", []),
                crop_w=min(patch_size, image.width),
                crop_h=min(patch_size, image.height),
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

    def sample_style_hint(self, rng: random.Random) -> dict[str, Any] | None:
        if not self.instance_assets:
            return None
        asset = rng.choice(self.instance_assets)
        image = self._open_image(asset["image_path"])
        bbox = asset["bbox"]
        pad = max((bbox[2] - bbox[0]) * 0.15, (bbox[3] - bbox[1]) * 0.15, 8.0)
        x1 = int(clamp(bbox[0] - pad, 0, image.width - 1))
        y1 = int(clamp(bbox[1] - pad, 0, image.height - 1))
        x2 = int(clamp(bbox[2] + pad, x1 + 1, image.width))
        y2 = int(clamp(bbox[3] + pad, y1 + 1, image.height))
        crop = image.crop((x1, y1, x2, y2))
        color = self._estimate_stroke_color(crop)
        bbox_w = max(float(bbox[2]) - float(bbox[0]), 1.0)
        bbox_h = max(float(bbox[3]) - float(bbox[1]), 1.0)
        minor_dim = max(min(bbox_w, bbox_h), 4.0)
        line_width = int(round(clamp(minor_dim * 0.14, 1.0, 10.0)))
        head_len = int(round(clamp(minor_dim * 0.42, 10.0, 34.0)))
        head_width = int(round(clamp(minor_dim * 0.32, 6.0, 26.0)))
        orientation = "vertical" if bbox_h > bbox_w * 1.2 else "horizontal" if bbox_w > bbox_h * 1.2 else "mixed"
        line_style = rng.choices(["solid", "dashed", "dotted"], weights=[0.72, 0.2, 0.08], k=1)[0]
        return {
            "line_width": line_width,
            "head_len": head_len,
            "head_width": head_width,
            "head_style": rng.choices(["filled", "open"], weights=[0.76, 0.24], k=1)[0],
            "line_style": line_style,
            "render_style": rng.choices(
                ["clean", "marker_jitter", "handdrawn"],
                weights=[0.66, 0.18, 0.16],
                k=1,
            )[0],
            "color": color,
            "source_asset_sample_id": asset["sample_id"],
            "source_asset_instance_id": asset["instance_id"],
            "orientation": orientation,
        }

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
            if any(bbox_iou(crop_bbox, [float(value) for value in instance["bbox"]]) > 0.02 for instance in instances):
                continue
            return image.crop((x1, y1, x1 + crop_w, y1 + crop_h)), [x1, y1, x1 + crop_w, y1 + crop_h]
        return None

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

    @staticmethod
    def _estimate_stroke_color(crop: Image.Image) -> tuple[int, int, int]:
        array = np.asarray(crop.convert("RGB"), dtype=np.uint8).reshape(-1, 3)
        if array.size == 0:
            return (40, 40, 40)
        intensities = array.sum(axis=1)
        threshold = np.quantile(intensities, 0.18)
        darkest = array[intensities <= threshold]
        if darkest.size == 0:
            darkest = array
        color = darkest.mean(axis=0)
        return tuple(int(clamp(channel, 20, 220)) for channel in color.tolist())
