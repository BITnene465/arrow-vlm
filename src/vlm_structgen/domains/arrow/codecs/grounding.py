from __future__ import annotations

import json
from typing import Any

from vlm_structgen.domains.arrow.codecs.structure import ArrowCodec, ValidationReport
from vlm_structgen.domains.arrow.schema import ARROW_LABELS


class GroundingCodec(ArrowCodec):
    def encode(self, gt_struct: dict[str, Any], image_width: int, image_height: int) -> str:
        payload, _loss_meta = self.encode_with_loss_meta(
            gt_struct,
            image_width=image_width,
            image_height=image_height,
        )
        return payload

    def encode_with_loss_meta(
        self,
        gt_struct: dict[str, Any],
        image_width: int,
        image_height: int,
    ) -> tuple[str, dict[str, Any]]:
        instances = gt_struct.get("instances", [])
        parts: list[str] = []
        cursor = 0
        label_spans: list[list[int]] = []
        bbox_spans: list[list[int]] = []

        def append(text: str) -> None:
            nonlocal cursor
            parts.append(text)
            cursor += len(text)

        append("[")
        for instance in instances:
            bbox = instance.get("bbox", [])
            if len(bbox) != 4:
                raise ValueError("Grounding instances must contain bbox with 4 values.")
            if cursor > 1:
                append(",")
            append('{"label":')
            label_text = json.dumps(str(instance.get("label", "")), ensure_ascii=False, separators=(",", ":"))
            label_start = cursor
            append(label_text)
            label_spans.append([label_start, cursor])
            append(',"bbox_2d":')
            bbox_values = [
                self._quantize(float(bbox[0]), image_width),
                self._quantize(float(bbox[1]), image_height),
                self._quantize(float(bbox[2]), image_width),
                self._quantize(float(bbox[3]), image_height),
            ]
            bbox_text = json.dumps(bbox_values, ensure_ascii=False, separators=(",", ":"))
            bbox_start = cursor
            append(bbox_text)
            bbox_spans.append([bbox_start, cursor])
            append("}")
        append("]")
        return "".join(parts), {
            "field_char_spans": {
                "label": label_spans,
                "bbox_2d": bbox_spans,
            }
        }

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

        instances: list[dict[str, Any]] = []
        for item_index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"Item at index {item_index} must be a JSON object.")
            label = item.get("label")
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
            instances.append(
                {
                    "label": str(label),
                    "bbox": bbox,
                    "keypoints": [],
                }
            )

        parsed = {"instances": instances}
        report = self.validate_struct(parsed, strict=strict)
        if not report.valid:
            raise ValueError("; ".join(report.errors))
        return parsed, {"recovered_prefix": recovered_prefix}

    def validate_struct(
        self,
        gt_struct: dict[str, Any],
        *,
        strict: bool = False,
    ) -> ValidationReport:
        errors: list[str] = []
        for index, instance in enumerate(gt_struct.get("instances", [])):
            label = str(instance.get("label", ""))
            if label not in ARROW_LABELS:
                errors.append(f"instance[{index}] label must be one of {sorted(ARROW_LABELS)}")
            bbox = instance.get("bbox", [])
            if len(bbox) != 4:
                errors.append(f"instance[{index}] bbox length must be 4")
                continue
            if strict:
                x1, y1, x2, y2 = bbox
                if float(x1) >= float(x2) or float(y1) >= float(y2):
                    errors.append(f"instance[{index}] bbox must satisfy x1 < x2 and y1 < y2")
        return ValidationReport(valid=not errors, errors=errors)

    @staticmethod
    def _dump_json(payload: list[dict[str, Any]]) -> str:
        import json

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
