from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from vlm_det.domains.arrow.codecs.structure import (
    JSON_FENCE_PATTERN,
    extract_balanced_json,
    recover_truncated_json_array,
)


@dataclass
class KeypointValidationReport:
    valid: bool
    errors: list[str]


class KeypointSequenceCodec:
    def __init__(self, num_bins: int = 1000) -> None:
        self.num_bins = int(num_bins)

    def encode(self, keypoints: list[list[float]], image_width: int, image_height: int) -> str:
        keypoints_2d = [
            [
                self._quantize(point[0], image_width),
                self._quantize(point[1], image_height),
            ]
            for point in keypoints
        ]
        report = self.validate_points(keypoints_2d)
        if not report.valid:
            raise ValueError("; ".join(report.errors))
        return json.dumps({"keypoints_2d": keypoints_2d}, ensure_ascii=False, separators=(",", ":"))

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
            payload = payload.get("keypoints_2d")
        if not isinstance(payload, list):
            raise ValueError("Decoded payload must be an object with keypoints_2d or a JSON array of points.")

        keypoints_2d: list[list[int]] = []
        keypoints: list[list[float]] = []
        for point_index, raw_point in enumerate(payload):
            if not isinstance(raw_point, list) or len(raw_point) != 2:
                raise ValueError(f"Point at index {point_index} must be [x, y].")
            x_value = self._parse_coord(raw_point[0], "x", strict=strict)
            y_value = self._parse_coord(raw_point[1], "y", strict=strict)
            keypoints_2d.append([x_value, y_value])
            keypoints.append(
                [
                    self._dequantize(x_value, image_width),
                    self._dequantize(y_value, image_height),
                ]
            )

        report = self.validate_points(keypoints_2d)
        if not report.valid:
            raise ValueError("; ".join(report.errors))
        return (
            {
                "keypoints": keypoints,
                "keypoints_2d": keypoints_2d,
            },
            {"recovered_prefix": recovered_prefix},
        )

    def validate_points(self, keypoints_2d: list[list[int]]) -> KeypointValidationReport:
        errors: list[str] = []
        if len(keypoints_2d) < 2:
            errors.append("Point list must contain at least 2 points.")
        for point_index, point in enumerate(keypoints_2d):
            if not isinstance(point, list) or len(point) != 2:
                errors.append(f"Point at index {point_index} must be [x, y].")
                continue
            for axis_name, value in (("x", point[0]), ("y", point[1])):
                if not isinstance(value, int):
                    errors.append(f"Point {point_index} {axis_name} must be an integer.")
                    continue
                if value < 0 or value >= self.num_bins:
                    errors.append(
                        f"Point {point_index} {axis_name}={value} out of range [0, {self.num_bins - 1}]."
                    )
        return KeypointValidationReport(valid=not errors, errors=errors)

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
        payload_text = extract_balanced_json(stripped)
        recovered_prefix = False
        if payload_text is None:
            payload_text = recover_truncated_json_array(stripped)
            recovered_prefix = payload_text is not None
        if payload_text is None:
            raise ValueError("No JSON payload found in decoded text.")
        try:
            return json.loads(payload_text), recovered_prefix
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON payload: {exc.msg}.") from exc
