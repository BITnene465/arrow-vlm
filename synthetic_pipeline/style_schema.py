from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArrowStyleFeature:
    sample_id: str
    instance_id: str
    family_name: str
    color: list[int]
    line_width_est: float
    head_length_est: float
    head_width_est: float
    saturation: float
    continuity: float
    roughness: float
    curvature: float
    hook_score: float
    point_count: int
    bbox_width: float
    bbox_height: float
    touches_border: bool
    orientation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "instance_id": self.instance_id,
            "family_name": self.family_name,
            "color": list(self.color),
            "line_width_est": round(float(self.line_width_est), 4),
            "head_length_est": round(float(self.head_length_est), 4),
            "head_width_est": round(float(self.head_width_est), 4),
            "saturation": round(float(self.saturation), 4),
            "continuity": round(float(self.continuity), 4),
            "roughness": round(float(self.roughness), 4),
            "curvature": round(float(self.curvature), 4),
            "hook_score": round(float(self.hook_score), 4),
            "point_count": int(self.point_count),
            "bbox_width": round(float(self.bbox_width), 4),
            "bbox_height": round(float(self.bbox_height), 4),
            "touches_border": bool(self.touches_border),
            "orientation": self.orientation,
        }


@dataclass
class ArrowStyleFamily:
    name: str
    weight: float
    line_width_range: list[int]
    head_length_range: list[int]
    head_width_range: list[int]
    head_styles: list[str]
    line_styles: list[str]
    render_styles: list[str]
    palettes: list[list[int]]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_profile_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weight": round(float(self.weight), 6),
            "line_width_range": [int(self.line_width_range[0]), int(self.line_width_range[1])],
            "head_length_range": [int(self.head_length_range[0]), int(self.head_length_range[1])],
            "head_width_range": [int(self.head_width_range[0]), int(self.head_width_range[1])],
            "head_styles": list(self.head_styles),
            "line_styles": list(self.line_styles),
            "render_styles": list(self.render_styles),
            "palettes": [list(color) for color in self.palettes],
            "meta": dict(self.meta),
        }


@dataclass
class ArrowStyleMiningReport:
    source_jsonl_paths: list[str]
    num_records: int
    num_instances: int
    families: list[ArrowStyleFamily]
    family_counts: dict[str, int]
    feature_summary: dict[str, Any]
    example_features: list[ArrowStyleFeature] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_jsonl_paths": list(self.source_jsonl_paths),
            "num_records": int(self.num_records),
            "num_instances": int(self.num_instances),
            "family_counts": {str(key): int(value) for key, value in self.family_counts.items()},
            "feature_summary": dict(self.feature_summary),
            "families": [family.to_profile_dict() for family in self.families],
            "example_features": [feature.to_dict() for feature in self.example_features],
        }
