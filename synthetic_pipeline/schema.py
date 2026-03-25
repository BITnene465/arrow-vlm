from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArrowInstance:
    bbox: list[float]
    keypoints: list[list[float]]
    render_bbox: list[float] | None = None
    source_asset_sample_id: str | None = None
    source_asset_instance_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        payload = {
            "bbox": [round(float(value), 2) for value in self.bbox],
            "keypoints": [
                [round(float(point[0]), 2), round(float(point[1]), 2)]
                for point in self.keypoints
            ],
        }
        if self.render_bbox is not None:
            payload["render_bbox"] = [round(float(value), 2) for value in self.render_bbox]
        if self.source_asset_sample_id is not None:
            payload["source_asset_sample_id"] = self.source_asset_sample_id
        if self.source_asset_instance_id is not None:
            payload["source_asset_instance_id"] = self.source_asset_instance_id
        if self.meta:
            payload["meta"] = dict(self.meta)
        return payload


@dataclass
class ContextPatch:
    bbox: list[float]
    kind: str
    source_sample_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        payload = {
            "bbox": [round(float(value), 2) for value in self.bbox],
            "kind": self.kind,
        }
        if self.source_sample_id is not None:
            payload["source_sample_id"] = self.source_sample_id
        if self.meta:
            payload["meta"] = dict(self.meta)
        return payload


@dataclass
class SceneSpec:
    split: str
    sample_id: str
    image_width: int
    image_height: int
    scene_mode: str
    renderer_name: str
    single_crop_padding_ratio: float | None = None
    domain_tag: str = "arrow_sync_v1"


@dataclass
class SyntheticSample:
    scene: SceneSpec
    instances: list[ArrowInstance] = field(default_factory=list)
    context_patches: list[ContextPatch] = field(default_factory=list)
    render_meta: dict[str, Any] = field(default_factory=dict)

    def to_record(self, image_path: str) -> dict[str, Any]:
        return {
            "sample_id": self.scene.sample_id,
            "image_path": image_path,
            "image_width": self.scene.image_width,
            "image_height": self.scene.image_height,
            "scene_mode": self.scene.scene_mode,
            "renderer_name": self.scene.renderer_name,
            "domain_tag": self.scene.domain_tag,
            "single_crop_padding_ratio": self.scene.single_crop_padding_ratio,
            "render_meta": dict(self.render_meta),
            "context_patches": [patch.to_record() for patch in self.context_patches],
            "instances": [instance.to_record() for instance in self.instances],
        }
