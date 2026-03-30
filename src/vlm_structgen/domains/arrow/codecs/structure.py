from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from vlm_structgen.domains.arrow.schema import (
    ARROW_LABELS,
    ArrowAnnotation,
    ArrowInstance,
    ArrowPoint,
    annotation_from_dict,
    annotation_to_dict,
)

JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class ValidationReport:
    valid: bool
    errors: list[str]


def extract_balanced_json(text: str) -> str | None:
    for opener, closer in (("[", "]"), ("{", "}")):
        payload = extract_balanced_json_with_delimiters(text, opener, closer)
        if payload is not None:
            return payload
    return None


def extract_balanced_json_with_delimiters(text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    if start < 0:
        return None
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


def recover_truncated_json_array(text: str) -> str | None:
    start = text.find("[")
    if start < 0:
        return None

    items: list[Any] = []
    in_string = False
    escape = False
    depth = 1
    item_start = start + 1

    for index in range(start + 1, len(text)):
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
        if char in "[{":
            depth += 1
            continue
        if char in "]}":
            depth -= 1
            if depth == 0:
                item_text = text[item_start:index].strip()
                if item_text:
                    try:
                        items.append(json.loads(item_text))
                    except json.JSONDecodeError:
                        pass
                return json.dumps(items, ensure_ascii=False, separators=(",", ":"))
            continue
        if char == "," and depth == 1:
            item_text = text[item_start:index].strip()
            if item_text:
                try:
                    items.append(json.loads(item_text))
                except json.JSONDecodeError:
                    pass
            item_start = index + 1

    tail_text = text[item_start:].strip()
    if tail_text:
        try:
            items.append(json.loads(tail_text))
        except json.JSONDecodeError:
            pass
    if not items:
        return None
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


class ArrowCodec:
    def __init__(self, num_bins: int = 1000) -> None:
        self.num_bins = num_bins

    def encode(self, gt_struct: dict[str, Any] | ArrowAnnotation, image_width: int, image_height: int) -> str:
        annotation = gt_struct if isinstance(gt_struct, ArrowAnnotation) else annotation_from_dict(gt_struct)
        report = self.validate_struct(annotation, strict=True)
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
                    "label": instance.label,
                    "bbox_2d": bbox,
                    "keypoints_2d": points,
                }
            )
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def decode(
        self,
        text: str,
        image_width: int,
        image_height: int,
        *,
        strict: bool = False,
    ) -> dict[str, Any]:
        parsed, _parse_meta = self.decode_with_meta(
            text,
            image_width=image_width,
            image_height=image_height,
            strict=strict,
        )
        return parsed

    def decode_with_meta(
        self,
        text: str,
        image_width: int,
        image_height: int,
        *,
        strict: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload, recovered_prefix = self._parse_json_payload(text, strict=strict)
        if isinstance(payload, dict):
            if strict:
                raise ValueError("Strict decoded payload must be a JSON array.")
            payload = [payload]
        if not isinstance(payload, list):
            raise ValueError("Decoded payload must be a JSON array or object.")

        annotation = ArrowAnnotation()
        for item_index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"Item at index {item_index} must be a JSON object.")
            if "label" not in item:
                raise ValueError(
                    f"Item at index {item_index} must explicitly contain label in {sorted(ARROW_LABELS)}."
                )
            label = item["label"]
            if label not in ARROW_LABELS:
                raise ValueError(
                    f"Item at index {item_index} must have label in {sorted(ARROW_LABELS)}."
                )
            bbox_values = item.get("bbox_2d")
            if not isinstance(bbox_values, list) or len(bbox_values) != 4:
                raise ValueError(f"Item at index {item_index} must contain bbox_2d with 4 values.")
            bbox = [
                self._dequantize(self._parse_coord(bbox_values[0], "x", strict=strict), image_width),
                self._dequantize(self._parse_coord(bbox_values[1], "y", strict=strict), image_height),
                self._dequantize(self._parse_coord(bbox_values[2], "x", strict=strict), image_width),
                self._dequantize(self._parse_coord(bbox_values[3], "y", strict=strict), image_height),
            ]

            raw_points = item.get("keypoints_2d")
            if not isinstance(raw_points, list):
                raise ValueError(f"Item at index {item_index} must contain keypoints_2d as a list.")
            keypoints: list[ArrowPoint] = []
            for point_index, raw_point in enumerate(raw_points):
                x_value, y_value = self._parse_point(
                    raw_point,
                    item_index,
                    point_index,
                    strict=strict,
                )
                keypoints.append(
                    ArrowPoint(
                        x=self._dequantize(x_value, image_width),
                        y=self._dequantize(y_value, image_height),
                    )
                )
            annotation.instances.append(ArrowInstance(label=label, bbox=bbox, keypoints=keypoints))

        report = self.validate_struct(annotation, strict=strict)
        if not report.valid:
            raise ValueError("; ".join(report.errors))
        return annotation_to_dict(annotation), {"recovered_prefix": recovered_prefix}

    def validate_struct(
        self,
        gt_struct: dict[str, Any] | ArrowAnnotation,
        *,
        strict: bool = False,
    ) -> ValidationReport:
        annotation = gt_struct if isinstance(gt_struct, ArrowAnnotation) else annotation_from_dict(gt_struct)
        errors: list[str] = []
        for index, instance in enumerate(annotation.instances):
            if instance.label not in ARROW_LABELS:
                errors.append(f"instance[{index}] label must be one of {sorted(ARROW_LABELS)}")
            if len(instance.bbox) != 4:
                errors.append(f"instance[{index}] bbox length must be 4")
            elif strict:
                x1, y1, x2, y2 = instance.bbox
                if x1 >= x2 or y1 >= y2:
                    errors.append(f"instance[{index}] bbox must satisfy x1 < x2 and y1 < y2")
            if len(instance.keypoints) < 2:
                errors.append(f"instance[{index}] must contain at least 2 keypoints")
        return ValidationReport(valid=not errors, errors=errors)

    def validate_text(
        self,
        text: str,
        image_width: int,
        image_height: int,
        *,
        strict: bool = False,
    ) -> ValidationReport:
        try:
            self.decode(text, image_width=image_width, image_height=image_height, strict=strict)
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

    def _parse_coord(self, value: Any, axis: str, *, strict: bool = False) -> int:
        if strict:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Expected integer {axis} coordinate, got {value!r}.")
            parsed = int(value)
        else:
            try:
                parsed = int(round(float(value)))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Expected numeric {axis} coordinate, got {value!r}.") from exc
        if parsed < 0 or parsed >= self.num_bins:
            raise ValueError(f"{axis} coordinate {parsed} out of range [0, {self.num_bins - 1}].")
        return parsed

    def _parse_point(
        self,
        raw_point: Any,
        item_index: int,
        point_index: int,
        *,
        strict: bool = False,
    ) -> tuple[int, int]:
        if strict:
            point_values = raw_point
        elif isinstance(raw_point, dict):
            point_values = raw_point.get("point_2d") or raw_point.get("point") or raw_point.get("xy")
        else:
            point_values = raw_point

        if not isinstance(point_values, list) or len(point_values) != 2:
            raise ValueError(
                f"Item {item_index} point {point_index} must be [x, y]."
            )

        x_value = self._parse_coord(point_values[0], "x", strict=strict)
        y_value = self._parse_coord(point_values[1], "y", strict=strict)
        return x_value, y_value

    def _parse_json_payload(self, text: str, *, strict: bool = False) -> tuple[Any, bool]:
        stripped = text.strip()
        if not stripped:
            raise ValueError("Decoded text is empty.")
        if strict:
            try:
                return json.loads(stripped), False
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Strict JSON payload must occupy the entire decoded text: {exc.msg}."
                ) from exc
        fenced = JSON_FENCE_PATTERN.search(stripped)
        if fenced is not None:
            stripped = fenced.group(1).strip()
        payload_text = None
        recovered_prefix = False
        if "[" in stripped:
            payload_text = extract_balanced_json_with_delimiters(stripped, "[", "]")
            if payload_text is None:
                payload_text = recover_truncated_json_array(stripped)
                recovered_prefix = payload_text is not None
        if payload_text is None and "{" in stripped:
            payload_text = extract_balanced_json_with_delimiters(stripped, "{", "}")
        if payload_text is None:
            payload_text = extract_balanced_json(stripped)
        if payload_text is None:
            raise ValueError("No JSON payload found in decoded text.")
        try:
            return json.loads(payload_text), recovered_prefix
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON payload: {exc.msg}.") from exc
