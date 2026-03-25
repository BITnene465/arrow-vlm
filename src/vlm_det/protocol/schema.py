from __future__ import annotations

from dataclasses import dataclass, field
@dataclass
class ArrowPoint:
    x: float
    y: float


@dataclass
class ArrowInstance:
    bbox: list[float]
    keypoints: list[ArrowPoint]


@dataclass
class ArrowAnnotation:
    instances: list[ArrowInstance] = field(default_factory=list)


def annotation_from_dict(payload: dict[str, Any]) -> ArrowAnnotation:
    instances = []
    for item in payload.get("instances", []):
        keypoints = []
        for point in item.get("keypoints", []):
            keypoints.append(ArrowPoint(float(point[0]), float(point[1])))
        instances.append(
            ArrowInstance(
                bbox=[float(value) for value in item.get("bbox", [])],
                keypoints=keypoints,
            )
        )
    return ArrowAnnotation(instances=instances)


def annotation_to_dict(annotation: ArrowAnnotation) -> dict[str, Any]:
    return {
        "instances": [
            {
                "bbox": list(instance.bbox),
                "keypoints": [
                    [point.x, point.y] for point in instance.keypoints
                ],
            }
            for instance in annotation.instances
        ]
    }
