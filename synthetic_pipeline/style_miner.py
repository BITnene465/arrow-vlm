from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthetic_pipeline.scene_sampler import clamp
from synthetic_pipeline.style_schema import ArrowStyleFamily, ArrowStyleFeature, ArrowStyleMiningReport
from vlm_structgen.core.utils.io import load_jsonl, write_json

Image.MAX_IMAGE_PIXELS = None

FAMILY_DEFAULTS: dict[str, dict[str, Any]] = {
    "scientific_thin": {"head_styles": ["filled", "open"], "line_styles": ["solid"], "render_styles": ["clean"]},
    "scientific_bold": {"head_styles": ["filled", "open"], "line_styles": ["solid"], "render_styles": ["clean"]},
    "mechanism_curve": {"head_styles": ["filled", "open"], "line_styles": ["solid"], "render_styles": ["clean"]},
    "flowchart_elbow": {"head_styles": ["filled", "open"], "line_styles": ["solid", "dashed"], "render_styles": ["clean"]},
    "ppt_colored": {"head_styles": ["filled", "open"], "line_styles": ["solid", "dashed"], "render_styles": ["clean"]},
    "drawio_clean": {"head_styles": ["filled", "open"], "line_styles": ["solid", "dashed"], "render_styles": ["clean"]},
    "callout_annotation": {"head_styles": ["filled", "open"], "line_styles": ["solid"], "render_styles": ["clean"]},
    "marker_rough": {"head_styles": ["filled"], "line_styles": ["solid", "marker"], "render_styles": ["marker", "marker_jitter"]},
    "handdrawn_sketch": {"head_styles": ["filled", "open"], "line_styles": ["solid", "dashed"], "render_styles": ["handdrawn", "handdrawn_heavy"]},
    "cropped_partial": {"head_styles": ["filled", "open"], "line_styles": ["solid"], "render_styles": ["clean"]},
    "dashed_process": {"head_styles": ["filled", "open"], "line_styles": ["dashed", "dotted"], "render_styles": ["clean"]},
    "hooked_pointer": {"head_styles": ["filled", "open"], "line_styles": ["solid"], "render_styles": ["clean"]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine arrow style families from processed JSONL and raw figures.")
    parser.add_argument(
        "--processed-jsonl",
        action="append",
        dest="processed_jsonl_paths",
        required=True,
        help="Processed JSONL path. Pass multiple times for train/val.",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--max-example-features", type=int, default=64)
    return parser.parse_args()


class StyleMiner:
    def __init__(
        self,
        *,
        processed_jsonl_paths: list[str],
        max_records: int | None = None,
        max_instances: int | None = None,
        max_example_features: int = 64,
    ) -> None:
        self.processed_jsonl_paths = processed_jsonl_paths
        self.max_records = max_records
        self.max_instances = max_instances
        self.max_example_features = max_example_features
        self._image_cache: dict[str, Image.Image] = {}

    def mine(self) -> ArrowStyleMiningReport:
        records = self._load_records()
        features: list[ArrowStyleFeature] = []
        family_counts: Counter[str] = Counter()
        for record in records:
            image = self._open_image(record["image_path"])
            for index, instance in enumerate(record.get("instances", [])):
                feature = self._extract_feature(record, index, instance, image)
                features.append(feature)
                family_counts[feature.family_name] += 1
                if self.max_instances is not None and len(features) >= self.max_instances:
                    break
            if self.max_instances is not None and len(features) >= self.max_instances:
                break

        families = self._aggregate_families(features, family_counts)
        feature_summary = self._summarize_features(features)
        return ArrowStyleMiningReport(
            source_jsonl_paths=list(self.processed_jsonl_paths),
            num_records=len(records),
            num_instances=len(features),
            families=families,
            family_counts=dict(family_counts),
            feature_summary=feature_summary,
            example_features=features[: self.max_example_features],
        )

    def _load_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path_str in self.processed_jsonl_paths:
            path = Path(path_str)
            if not path.exists():
                continue
            records.extend(load_jsonl(path))
        if self.max_records is not None:
            records = records[: self.max_records]
        return records

    def _open_image(self, path: str) -> Image.Image:
        cached = self._image_cache.get(path)
        if cached is not None:
            return cached
        image = Image.open(Path(path)).convert("RGB")
        self._image_cache[path] = image
        return image

    def _extract_feature(
        self,
        record: dict[str, Any],
        index: int,
        instance: dict[str, Any],
        image: Image.Image,
    ) -> ArrowStyleFeature:
        bbox = [float(value) for value in instance["bbox"]]
        keypoints = [[float(point[0]), float(point[1])] for point in instance["keypoints"]]
        patch, patch_origin = self._extract_patch(image, bbox)
        path_length = self._path_length(keypoints)
        dark_mask = self._build_dark_mask(patch)
        color = self._estimate_stroke_color(patch)
        saturation = self._estimate_saturation(color)
        continuity = self._estimate_continuity(dark_mask, keypoints, patch_origin)
        roughness = self._estimate_roughness(patch)
        curvature, hook_score, elbow_score = self._estimate_turn_signals(keypoints)
        bbox_width = max(bbox[2] - bbox[0], 1.0)
        bbox_height = max(bbox[3] - bbox[1], 1.0)
        line_width_est = self._estimate_line_width(dark_mask, keypoints, patch_origin)
        minor_dim = max(min(bbox_width, bbox_height), 4.0)
        major_dim = max(bbox_width, bbox_height)
        head_length_est = clamp(
            max(line_width_est * 2.8, min(major_dim * 0.24, minor_dim * 0.62)),
            8.0,
            36.0,
        )
        head_width_est = clamp(
            max(line_width_est * 1.8, min(minor_dim * 0.34, major_dim * 0.2)),
            6.0,
            28.0,
        )
        touches_border = (
            bbox[0] <= 2.0
            or bbox[1] <= 2.0
            or bbox[2] >= image.width - 3.0
            or bbox[3] >= image.height - 3.0
        )
        orientation = self._orientation_label(keypoints)
        family_name = self._classify_family(
            saturation=saturation,
            line_width_est=line_width_est,
            continuity=continuity,
            roughness=roughness,
            curvature=curvature,
            hook_score=hook_score,
            elbow_score=elbow_score,
            point_count=len(keypoints),
            touches_border=touches_border,
            orientation=orientation,
        )
        return ArrowStyleFeature(
            sample_id=str(record["sample_id"]),
            instance_id=f"{record['sample_id']}:{index}",
            family_name=family_name,
            color=list(color),
            line_width_est=float(line_width_est),
            head_length_est=float(head_length_est),
            head_width_est=float(head_width_est),
            saturation=float(saturation),
            continuity=float(continuity),
            roughness=float(roughness),
            curvature=float(curvature),
            hook_score=float(hook_score),
            point_count=len(keypoints),
            bbox_width=float(bbox_width),
            bbox_height=float(bbox_height),
            touches_border=touches_border,
            orientation=orientation,
        )

    @staticmethod
    def _extract_patch(image: Image.Image, bbox: list[float]) -> tuple[Image.Image, tuple[int, int]]:
        pad = max((bbox[2] - bbox[0]) * 0.18, (bbox[3] - bbox[1]) * 0.18, 8.0)
        x1 = int(clamp(math.floor(bbox[0] - pad), 0, image.width - 1))
        y1 = int(clamp(math.floor(bbox[1] - pad), 0, image.height - 1))
        x2 = int(clamp(math.ceil(bbox[2] + pad), x1 + 1, image.width))
        y2 = int(clamp(math.ceil(bbox[3] + pad), y1 + 1, image.height))
        return image.crop((x1, y1, x2, y2)), (x1, y1)

    @staticmethod
    def _path_length(keypoints: list[list[float]]) -> float:
        total = 0.0
        for start, end in zip(keypoints[:-1], keypoints[1:]):
            total += math.hypot(end[0] - start[0], end[1] - start[1])
        return max(total, 1.0)

    @staticmethod
    def _estimate_stroke_color(crop: Image.Image) -> tuple[int, int, int]:
        array = np.asarray(crop, dtype=np.uint8).reshape(-1, 3)
        if array.size == 0:
            return (40, 40, 40)
        intensities = array.sum(axis=1)
        threshold = np.quantile(intensities, 0.18)
        darkest = array[intensities <= threshold]
        if darkest.size == 0:
            darkest = array
        color = darkest.mean(axis=0)
        return tuple(int(clamp(channel, 20, 230)) for channel in color.tolist())

    @staticmethod
    def _estimate_saturation(color: tuple[int, int, int]) -> float:
        red, green, blue = [float(channel) / 255.0 for channel in color]
        maximum = max(red, green, blue)
        minimum = min(red, green, blue)
        if maximum <= 1e-6:
            return 0.0
        return (maximum - minimum) / maximum

    @staticmethod
    def _build_dark_mask(crop: Image.Image) -> np.ndarray:
        gray = np.asarray(crop.convert("L"), dtype=np.float32)
        threshold = np.quantile(gray, 0.22)
        return gray <= threshold

    @staticmethod
    def _estimate_roughness(crop: Image.Image) -> float:
        gray = np.asarray(crop.convert("L"), dtype=np.float32)
        if gray.size == 0:
            return 0.0
        grad_x = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
        grad_y = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
        normalized = (grad_x + grad_y) / 255.0
        return float(clamp(normalized, 0.0, 1.0))

    @staticmethod
    def _estimate_continuity(
        dark_mask: np.ndarray,
        keypoints: list[list[float]],
        patch_origin: tuple[int, int],
    ) -> float:
        x_offset, y_offset = patch_origin
        samples = StyleMiner._densify_path(keypoints, samples_per_segment=24)
        hits = 0
        total = 0
        for x_value, y_value in samples:
            cx = int(round(x_value - x_offset))
            cy = int(round(y_value - y_offset))
            found = False
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    xx = cx + dx
                    yy = cy + dy
                    if 0 <= yy < dark_mask.shape[0] and 0 <= xx < dark_mask.shape[1] and dark_mask[yy, xx]:
                        found = True
                        break
                if found:
                    break
            hits += int(found)
            total += 1
        if total == 0:
            return 1.0
        return float(hits / total)

    @staticmethod
    def _estimate_line_width(
        dark_mask: np.ndarray,
        keypoints: list[list[float]],
        patch_origin: tuple[int, int],
    ) -> float:
        x_offset, y_offset = patch_origin
        samples = StyleMiner._densify_path(keypoints, samples_per_segment=18)
        local_widths: list[float] = []
        for x_value, y_value in samples:
            cx = int(round(x_value - x_offset))
            cy = int(round(y_value - y_offset))
            dark_count = 0
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    xx = cx + dx
                    yy = cy + dy
                    if 0 <= yy < dark_mask.shape[0] and 0 <= xx < dark_mask.shape[1] and dark_mask[yy, xx]:
                        dark_count += 1
            if dark_count <= 0:
                continue
            local_widths.append(math.sqrt(float(dark_count)))
        if not local_widths:
            return 2.0
        estimate = float(np.quantile(local_widths, 0.6))
        return float(clamp(estimate, 1.0, 10.0))

    @staticmethod
    def _densify_path(keypoints: list[list[float]], samples_per_segment: int) -> list[tuple[float, float]]:
        if len(keypoints) < 2:
            return []
        dense: list[tuple[float, float]] = []
        for start, end in zip(keypoints[:-1], keypoints[1:]):
            for step in range(samples_per_segment):
                t_value = step / samples_per_segment
                dense.append(
                    (
                        start[0] + (end[0] - start[0]) * t_value,
                        start[1] + (end[1] - start[1]) * t_value,
                    )
                )
        dense.append((keypoints[-1][0], keypoints[-1][1]))
        return dense

    @staticmethod
    def _estimate_turn_signals(keypoints: list[list[float]]) -> tuple[float, float, float]:
        if len(keypoints) < 3:
            return 0.0, 0.0, 0.0
        turns: list[float] = []
        right_angle_turns = 0
        for p0, p1, p2 in zip(keypoints[:-2], keypoints[1:-1], keypoints[2:]):
            v1 = (p1[0] - p0[0], p1[1] - p0[1])
            v2 = (p2[0] - p1[0], p2[1] - p1[1])
            norm1 = math.hypot(*v1)
            norm2 = math.hypot(*v2)
            if norm1 <= 1e-6 or norm2 <= 1e-6:
                continue
            dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (norm1 * norm2)
            dot = max(-1.0, min(1.0, dot))
            angle = math.degrees(math.acos(dot))
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            signed_angle = angle if cross >= 0 else -angle
            turns.append(signed_angle)
            if 55.0 <= angle <= 125.0:
                right_angle_turns += 1
        if not turns:
            return 0.0, 0.0, 0.0
        abs_turns = [abs(turn) for turn in turns]
        curvature = float(clamp(np.mean(abs_turns) / 90.0, 0.0, 1.0))
        hook_score = float(clamp(abs(turns[-1]) / 120.0, 0.0, 1.0))
        elbow_score = float(right_angle_turns / len(turns))
        return curvature, hook_score, elbow_score

    @staticmethod
    def _orientation_label(keypoints: list[list[float]]) -> str:
        start = keypoints[0]
        end = keypoints[-1]
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        if dx > dy * 1.2:
            return "horizontal"
        if dy > dx * 1.2:
            return "vertical"
        return "mixed"

    @staticmethod
    def _classify_family(
        *,
        saturation: float,
        line_width_est: float,
        continuity: float,
        roughness: float,
        curvature: float,
        hook_score: float,
        elbow_score: float,
        point_count: int,
        touches_border: bool,
        orientation: str,
    ) -> str:
        if touches_border and (orientation != "mixed" or point_count <= 2):
            return "cropped_partial"
        if point_count >= 4 and hook_score >= 0.55:
            return "hooked_pointer"
        if point_count >= 3 and elbow_score >= 0.55 and orientation in {"horizontal", "vertical"}:
            return "flowchart_elbow"
        if point_count >= 3 and curvature >= 0.38:
            return "mechanism_curve"
        if saturation >= 0.32 and line_width_est >= 4.5:
            return "ppt_colored"
        if saturation >= 0.18:
            return "drawio_clean"
        if roughness >= 0.34 and line_width_est >= 4.0:
            return "marker_rough"
        if roughness >= 0.27:
            return "handdrawn_sketch"
        if continuity <= 0.78:
            return "dashed_process"
        if line_width_est <= 2.2:
            return "scientific_thin"
        if line_width_est >= 4.2:
            return "scientific_bold"
        return "callout_annotation"

    def _aggregate_families(
        self,
        features: list[ArrowStyleFeature],
        family_counts: Counter[str],
    ) -> list[ArrowStyleFamily]:
        total = max(sum(family_counts.values()), 1)
        grouped: dict[str, list[ArrowStyleFeature]] = defaultdict(list)
        for feature in features:
            grouped[feature.family_name].append(feature)

        families: list[ArrowStyleFamily] = []
        for family_name, family_features in sorted(grouped.items()):
            defaults = FAMILY_DEFAULTS[family_name]
            line_widths = [feature.line_width_est for feature in family_features]
            head_lengths = [feature.head_length_est for feature in family_features]
            head_widths = [feature.head_width_est for feature in family_features]
            palettes = self._top_palettes([feature.color for feature in family_features], limit=4)
            families.append(
                ArrowStyleFamily(
                    name=family_name,
                    weight=family_counts[family_name] / total,
                    line_width_range=self._percentile_range(line_widths, minimum=1, maximum=12),
                    head_length_range=self._percentile_range(head_lengths, minimum=8, maximum=36),
                    head_width_range=self._percentile_range(head_widths, minimum=6, maximum=28),
                    head_styles=list(defaults["head_styles"]),
                    line_styles=list(defaults["line_styles"]),
                    render_styles=list(defaults["render_styles"]),
                    palettes=palettes,
                    meta={
                        "num_instances": len(family_features),
                        "avg_saturation": round(float(np.mean([feature.saturation for feature in family_features])), 4),
                        "avg_continuity": round(float(np.mean([feature.continuity for feature in family_features])), 4),
                        "avg_roughness": round(float(np.mean([feature.roughness for feature in family_features])), 4),
                        "avg_curvature": round(float(np.mean([feature.curvature for feature in family_features])), 4),
                    },
                )
            )
        return families

    @staticmethod
    def _percentile_range(values: list[float], minimum: int, maximum: int) -> list[int]:
        if not values:
            return [minimum, maximum]
        low = int(round(clamp(np.quantile(values, 0.15), minimum, maximum)))
        high = int(round(clamp(np.quantile(values, 0.85), low, maximum)))
        return [low, high]

    @staticmethod
    def _top_palettes(colors: list[list[int]], limit: int) -> list[list[int]]:
        counter: Counter[tuple[int, int, int]] = Counter()
        for color in colors:
            bucket = tuple(int(round(channel / 8.0) * 8) for channel in color)
            counter[bucket] += 1
        palettes: list[list[int]] = []
        for bucket, _count in counter.most_common(limit):
            palettes.append([int(clamp(value, 0, 255)) for value in bucket])
        if not palettes:
            palettes = [[40, 40, 40]]
        return palettes

    @staticmethod
    def _summarize_features(features: list[ArrowStyleFeature]) -> dict[str, Any]:
        if not features:
            return {
                "avg_line_width_est": 0.0,
                "avg_head_length_est": 0.0,
                "avg_head_width_est": 0.0,
                "avg_saturation": 0.0,
                "avg_continuity": 0.0,
                "avg_roughness": 0.0,
                "avg_curvature": 0.0,
            }
        return {
            "avg_line_width_est": round(float(np.mean([feature.line_width_est for feature in features])), 4),
            "avg_head_length_est": round(float(np.mean([feature.head_length_est for feature in features])), 4),
            "avg_head_width_est": round(float(np.mean([feature.head_width_est for feature in features])), 4),
            "avg_saturation": round(float(np.mean([feature.saturation for feature in features])), 4),
            "avg_continuity": round(float(np.mean([feature.continuity for feature in features])), 4),
            "avg_roughness": round(float(np.mean([feature.roughness for feature in features])), 4),
            "avg_curvature": round(float(np.mean([feature.curvature for feature in features])), 4),
        }


def main() -> None:
    args = parse_args()
    miner = StyleMiner(
        processed_jsonl_paths=args.processed_jsonl_paths,
        max_records=args.max_records,
        max_instances=args.max_instances,
        max_example_features=args.max_example_features,
    )
    report = miner.mine()
    write_json(args.output_path, report.to_dict())
    print(f"Wrote style mining report to {args.output_path}")


if __name__ == "__main__":
    main()
