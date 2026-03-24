from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from vlm_det.protocol.schema import ArrowAnnotation, ArrowInstance, ArrowPoint, annotation_from_dict, annotation_to_dict

JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class ValidationReport:
    valid: bool
    errors: list[str]


class ArrowCodec:
    def __init__(self, num_bins: int = 1000) -> None:
        self.num_bins = num_bins

    def encode(self, gt_struct: dict[str, Any] | ArrowAnnotation, image_width: int, image_height: int) -> str:
        annotation = gt_struct if isinstance(gt_struct, ArrowAnnotation) else annotation_from_dict(gt_struct)
        report = self.validate_struct(annotation)
        if not report.valid:
            raise ValueError(f"Invalid annotation for encoding: {report.errors}")

        payload: list[dict[str, Any]] = []
        for instance in annotation.instances:
            bbox = [
                self._quantize(instance.bbox[0], image_width),
                self._quantize(instance.bbox[1], image_height),
                self._quantize(instance.bbox[2], image_width),
                self._quantize(instance.bbox[3], image_height),
            ]
            points = [
                [
                    self._quantize(point.x, image_width),
                    self._quantize(point.y, image_height),
                ]
                for point in instance.keypoints
            ]
            payload.append(
                {
                    "label": "arrow",
                    "bbox_2d": bbox,
                    "keypoints_2d": points,
                }
            )
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def decode(self, text: str, image_width: int, image_height: int) -> dict[str, Any]:
        payload = self._parse_json_payload(text)
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            raise ValueError("Decoded payload must be a JSON array or object.")

        annotation = ArrowAnnotation()
        for item_index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"Item at index {item_index} must be a JSON object.")
            label = item.get("label", "arrow")
            if label != "arrow":
                raise ValueError(f"Item at index {item_index} must have label='arrow'.")
            bbox_values = item.get("bbox_2d")
            if not isinstance(bbox_values, list) or len(bbox_values) != 4:
                raise ValueError(f"Item at index {item_index} must contain bbox_2d with 4 values.")
            bbox = [
                self._dequantize(self._parse_coord(bbox_values[0], "x"), image_width),
                self._dequantize(self._parse_coord(bbox_values[1], "y"), image_height),
                self._dequantize(self._parse_coord(bbox_values[2], "x"), image_width),
                self._dequantize(self._parse_coord(bbox_values[3], "y"), image_height),
            ]

            raw_points = item.get("keypoints_2d")
            if not isinstance(raw_points, list):
                raise ValueError(f"Item at index {item_index} must contain keypoints_2d as a list.")
            keypoints: list[ArrowPoint] = []
            for point_index, raw_point in enumerate(raw_points):
                x_value, y_value = self._parse_point(raw_point, item_index, point_index)
                keypoints.append(
                    ArrowPoint(
                        x=self._dequantize(x_value, image_width),
                        y=self._dequantize(y_value, image_height),
                    )
                )
            annotation.instances.append(ArrowInstance(bbox=bbox, keypoints=keypoints))

        report = self.validate_struct(annotation)
        if not report.valid:
            raise ValueError("; ".join(report.errors))
        return annotation_to_dict(annotation)

    def validate_struct(self, gt_struct: dict[str, Any] | ArrowAnnotation) -> ValidationReport:
        annotation = gt_struct if isinstance(gt_struct, ArrowAnnotation) else annotation_from_dict(gt_struct)
        errors: list[str] = []
        for index, instance in enumerate(annotation.instances):
            if len(instance.bbox) != 4:
                errors.append(f"instance[{index}] bbox length must be 4")
            if len(instance.keypoints) < 2:
                errors.append(f"instance[{index}] must contain at least 2 keypoints")
        return ValidationReport(valid=not errors, errors=errors)

    def validate_text(self, text: str, image_width: int, image_height: int) -> ValidationReport:
        try:
            self.decode(text, image_width=image_width, image_height=image_height)
        except Exception as exc:  # noqa: BLE001
            return ValidationReport(valid=False, errors=[str(exc)])
        return ValidationReport(valid=True, errors=[])

    def _quantize(self, value: float, size: int) -> int:
        size = max(int(size), 1)
        if size == 1:
            return 0
        clipped = min(max(float(value), 0.0), float(size - 1))
        return int(round(clipped / float(size - 1) * float(self.num_bins - 1)))

    def _dequantize(self, value: int, size: int) -> float:
        size = max(int(size), 1)
        if size == 1:
            return 0.0
        return float(value) / float(self.num_bins - 1) * float(size - 1)

    def _parse_coord(self, value: Any, axis: str) -> int:
        try:
            parsed = int(round(float(value)))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Expected numeric {axis} coordinate, got {value!r}.") from exc
        if parsed < 0 or parsed >= self.num_bins:
            raise ValueError(f"{axis} coordinate {parsed} out of range [0, {self.num_bins - 1}].")
        return parsed

    def _parse_point(self, raw_point: Any, item_index: int, point_index: int) -> tuple[int, int]:
        if isinstance(raw_point, dict):
            point_values = raw_point.get("point_2d") or raw_point.get("point") or raw_point.get("xy")
        else:
            point_values = raw_point

        if not isinstance(point_values, list) or len(point_values) != 2:
            raise ValueError(
                f"Item {item_index} point {point_index} must be [x, y]."
            )

        x_value = self._parse_coord(point_values[0], "x")
        y_value = self._parse_coord(point_values[1], "y")
        return x_value, y_value

    def _parse_json_payload(self, text: str) -> Any:
        stripped = text.strip()
        if not stripped:
            raise ValueError("Decoded text is empty.")
        fenced = JSON_FENCE_PATTERN.search(stripped)
        if fenced is not None:
            stripped = fenced.group(1).strip()
        payload_text = self._extract_balanced_json(stripped)
        if payload_text is None:
            raise ValueError("No JSON payload found in decoded text.")
        try:
            return json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON payload: {exc.msg}.") from exc

    @staticmethod
    def _extract_balanced_json(text: str) -> str | None:
        for opener, closer in (("[", "]"), ("{", "}")):
            start = text.find(opener)
            if start < 0:
                continue
            depth = 0
            in_string = False
            escape = False
            for index in range(start, len(text)):
                char = text[index]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                    continue
                if char == opener:
                    depth += 1
                elif char == closer:
                    depth -= 1
                    if depth == 0:
                        return text[start : index + 1]
        return None
