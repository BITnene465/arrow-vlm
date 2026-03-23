from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Visibility = str


@dataclass
class ArrowPoint:
    x: float
    y: float
    visibility: Visibility


@dataclass
class ArrowInstance:
    bbox: list[float]
    keypoints: list[ArrowPoint]
    group_id: int | None = None
    raw_bbox: list[float] | None = None
    raw_keypoints: list[list[Any]] | None = None


@dataclass
class ArrowAnnotation:
    instances: list[ArrowInstance] = field(default_factory=list)


def annotation_from_dict(payload: dict[str, Any]) -> ArrowAnnotation:
    instances = []
    for item in payload.get("instances", []):
        keypoints = [
            ArrowPoint(float(point[0]), float(point[1]), str(point[2]))
            for point in item.get("keypoints", [])
        ]
        instances.append(
            ArrowInstance(
                bbox=[float(value) for value in item.get("bbox", [])],
                keypoints=keypoints,
                group_id=item.get("group_id"),
                raw_bbox=item.get("raw_bbox"),
                raw_keypoints=item.get("raw_keypoints"),
            )
        )
    return ArrowAnnotation(instances=instances)


def annotation_to_dict(annotation: ArrowAnnotation) -> dict[str, Any]:
    return {
        "instances": [
            {
                "bbox": list(instance.bbox),
                "keypoints": [
                    [point.x, point.y, point.visibility] for point in instance.keypoints
                ],
                **({"group_id": instance.group_id} if instance.group_id is not None else {}),
                **({"raw_bbox": instance.raw_bbox} if instance.raw_bbox is not None else {}),
                **(
                    {"raw_keypoints": instance.raw_keypoints}
                    if instance.raw_keypoints is not None
                    else {}
                ),
            }
            for instance in annotation.instances
        ]
    }
