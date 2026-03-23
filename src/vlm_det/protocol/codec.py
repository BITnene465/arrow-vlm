from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from vlm_det.protocol.schema import ArrowAnnotation, ArrowInstance, ArrowPoint, annotation_from_dict, annotation_to_dict
from vlm_det.protocol.tokens import (
    ARROW_BEGIN_TOKEN,
    ARROW_END_TOKEN,
    ARROWS_BEGIN_TOKEN,
    ARROWS_END_TOKEN,
    BOX_BEGIN_TOKEN,
    BOX_END_TOKEN,
    OCCLUDED_TOKEN,
    POINT_BEGIN_TOKEN,
    POINT_END_TOKEN,
    POINTS_BEGIN_TOKEN,
    POINTS_END_TOKEN,
    VISIBLE_TOKEN,
    x_token,
    y_token,
)


TOKEN_PATTERN = re.compile(r"<\|[^|]+\|>")


@dataclass
class ValidationReport:
    valid: bool
    errors: list[str]


class ArrowCodec:
    def __init__(self, num_bins: int = 2048) -> None:
        self.num_bins = num_bins

    def encode(self, gt_struct: dict[str, Any] | ArrowAnnotation, image_width: int, image_height: int) -> str:
        annotation = gt_struct if isinstance(gt_struct, ArrowAnnotation) else annotation_from_dict(gt_struct)
        report = self.validate_struct(annotation)
        if not report.valid:
            raise ValueError(f"Invalid annotation for encoding: {report.errors}")

        lines: list[str] = [ARROWS_BEGIN_TOKEN]
        for instance in annotation.instances:
            bbox_tokens = [
                self._x_token(self._quantize(instance.bbox[0], image_width)),
                self._y_token(self._quantize(instance.bbox[1], image_height)),
                self._x_token(self._quantize(instance.bbox[2], image_width)),
                self._y_token(self._quantize(instance.bbox[3], image_height)),
            ]
            lines.append(ARROW_BEGIN_TOKEN)
            lines.append(" ".join([BOX_BEGIN_TOKEN, *bbox_tokens, BOX_END_TOKEN]))
            lines.append(POINTS_BEGIN_TOKEN)
            for point in instance.keypoints:
                visibility_token = self._visibility_to_token(point.visibility)
                lines.append(
                    " ".join(
                        [
                            POINT_BEGIN_TOKEN,
                            self._x_token(self._quantize(point.x, image_width)),
                            self._y_token(self._quantize(point.y, image_height)),
                            visibility_token,
                            POINT_END_TOKEN,
                        ]
                    )
                )
            lines.append(POINTS_END_TOKEN)
            lines.append(ARROW_END_TOKEN)
        lines.append(ARROWS_END_TOKEN)
        return "\n".join(lines)

    def decode(self, text: str, image_width: int, image_height: int) -> dict[str, Any]:
        tokens = TOKEN_PATTERN.findall(text)
        if not tokens:
            raise ValueError("No protocol tokens found in text.")
        cursor = 0

        def consume(expected: str) -> None:
            nonlocal cursor
            if cursor >= len(tokens) or tokens[cursor] != expected:
                actual = tokens[cursor] if cursor < len(tokens) else "<eos>"
                raise ValueError(f"Expected token {expected}, got {actual}.")
            cursor += 1

        def parse_bbox() -> list[float]:
            consume(BOX_BEGIN_TOKEN)
            parsed = [
                self._dequantize(self._parse_x(tokens[cursor]), image_width),
                self._dequantize(self._parse_y(tokens[cursor + 1]), image_height),
                self._dequantize(self._parse_x(tokens[cursor + 2]), image_width),
                self._dequantize(self._parse_y(tokens[cursor + 3]), image_height),
            ]
            advance(4)
            consume(BOX_END_TOKEN)
            return parsed

        def parse_point() -> ArrowPoint:
            consume(POINT_BEGIN_TOKEN)
            x_value = self._dequantize(self._parse_x(tokens[cursor]), image_width)
            y_value = self._dequantize(self._parse_y(tokens[cursor + 1]), image_height)
            visibility = self._token_to_visibility(tokens[cursor + 2])
            advance(3)
            consume(POINT_END_TOKEN)
            return ArrowPoint(x_value, y_value, visibility)

        def advance(steps: int) -> None:
            nonlocal cursor
            cursor += steps

        annotation = ArrowAnnotation()
        consume(ARROWS_BEGIN_TOKEN)
        while cursor < len(tokens) and tokens[cursor] != ARROWS_END_TOKEN:
            consume(ARROW_BEGIN_TOKEN)
            bbox = parse_bbox()
            consume(POINTS_BEGIN_TOKEN)
            keypoints: list[ArrowPoint] = []
            while cursor < len(tokens) and tokens[cursor] != POINTS_END_TOKEN:
                keypoints.append(parse_point())
            consume(POINTS_END_TOKEN)
            consume(ARROW_END_TOKEN)
            annotation.instances.append(ArrowInstance(bbox=bbox, keypoints=keypoints))
        consume(ARROWS_END_TOKEN)
        if cursor != len(tokens):
            raise ValueError("Trailing tokens remain after parsing.")
        return annotation_to_dict(annotation)

    def validate_struct(self, gt_struct: dict[str, Any] | ArrowAnnotation) -> ValidationReport:
        annotation = gt_struct if isinstance(gt_struct, ArrowAnnotation) else annotation_from_dict(gt_struct)
        errors: list[str] = []
        for index, instance in enumerate(annotation.instances):
            if len(instance.bbox) != 4:
                errors.append(f"instance[{index}] bbox length must be 4")
            if len(instance.keypoints) < 2:
                errors.append(f"instance[{index}] must contain at least 2 keypoints")
            for point_index, point in enumerate(instance.keypoints):
                if point.visibility not in {"visible", "occluded"}:
                    errors.append(
                        f"instance[{index}].keypoints[{point_index}] visibility must be visible/occluded"
                    )
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

    def _x_token(self, index: int) -> str:
        return x_token(index)

    def _y_token(self, index: int) -> str:
        return y_token(index)

    def _parse_x(self, token: str) -> int:
        prefix = "<|x_"
        if not token.startswith(prefix) or not token.endswith("|>"):
            raise ValueError(f"Expected x token, got {token}")
        value = int(token[len(prefix) : -2])
        if value < 0 or value >= self.num_bins:
            raise ValueError(f"x token {token} out of range")
        return value

    def _parse_y(self, token: str) -> int:
        prefix = "<|y_"
        if not token.startswith(prefix) or not token.endswith("|>"):
            raise ValueError(f"Expected y token, got {token}")
        value = int(token[len(prefix) : -2])
        if value < 0 or value >= self.num_bins:
            raise ValueError(f"y token {token} out of range")
        return value

    @staticmethod
    def _visibility_to_token(visibility: str) -> str:
        if visibility == "visible":
            return VISIBLE_TOKEN
        if visibility == "occluded":
            return OCCLUDED_TOKEN
        raise ValueError(f"Unsupported visibility: {visibility}")

    @staticmethod
    def _token_to_visibility(token: str) -> str:
        if token == VISIBLE_TOKEN:
            return "visible"
        if token == OCCLUDED_TOKEN:
            return "occluded"
        raise ValueError(f"Unsupported visibility token: {token}")
