from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from vlm_det.protocol.codec import ArrowCodec
from vlm_det.utils.io import load_jsonl

# Training can encounter extremely large figure images. We only decode images
# that are already part of the trusted dataset, so disable Pillow's
# decompression bomb guard here as well.
Image.MAX_IMAGE_PIXELS = None
TEMPLATE_PATTERN = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")


class ArrowSFTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        codec: ArrowCodec,
        system_prompt: str,
        user_prompt: str,
        system_prompt_template: str | None = None,
        user_prompt_template: str | None = None,
    ) -> None:
        self.records = load_jsonl(jsonl_path)
        self.codec = codec
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = Path(record["image_path"])
        image = Image.open(image_path).convert("RGB")
        record_gt_struct = record.get("gt_struct")
        if record_gt_struct is not None:
            gt_struct = record_gt_struct
        else:
            instances = record.get("instances", [])
            gt_struct = {
                "instances": [
                    {
                        "label": instance["label"],
                        "bbox": instance["bbox"],
                        "keypoints": instance["keypoints"],
                    }
                    for instance in instances
                ]
            }
        target_text = record.get("target_text")
        if target_text is None:
            target_text = self.codec.encode(
                gt_struct,
                image_width=record["image_width"],
                image_height=record["image_height"],
            )
        condition = record.get("condition", {})
        system_prompt = record.get("system_prompt")
        if system_prompt is None:
            template = record.get("system_prompt_template", self.system_prompt_template)
            if template:
                system_prompt = self._render_prompt_template(template, condition)
            else:
                system_prompt = self.system_prompt
        user_prompt = record.get("user_prompt")
        if user_prompt is None:
            template = record.get("user_prompt_template", self.user_prompt_template)
            if template:
                user_prompt = self._render_prompt_template(template, condition)
            else:
                user_prompt = self.user_prompt
        return {
            "task_type": record.get("task_type", "one_stage"),
            "sample_id": record.get("sample_id", image_path.stem),
            "image_path": str(image_path),
            "image": image,
            "image_width": int(record["image_width"]),
            "image_height": int(record["image_height"]),
            "system_prompt": str(system_prompt),
            "user_prompt": str(user_prompt),
            "target_text": str(target_text),
            "gt_struct": gt_struct,
        }

    def _render_prompt_template(self, template: str, context: dict[str, Any]) -> str:
        def _replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in context:
                raise KeyError(f"Prompt template variable {key!r} is missing from condition payload.")
            value = context[key]
            if isinstance(value, (list, dict)):
                return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            return str(value)

        return TEMPLATE_PATTERN.sub(_replace, template)
