from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from vlm_det.prompting import render_prompt_template
from vlm_det.protocol.codec import ArrowCodec
from vlm_det.protocol.grounding_codec import GroundingCodec
from vlm_det.utils.io import load_jsonl

# Training can encounter extremely large figure images. We only decode images
# that are already part of the trusted dataset, so disable Pillow's
# decompression bomb guard here as well.
Image.MAX_IMAGE_PIXELS = None


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
        self.grounding_codec = GroundingCodec(num_bins=codec.num_bins)
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self._target_token_lengths_cache: dict[int, list[int]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        task_type = str(record.get("task_type", "one_stage"))
        image_path = Path(record["image_path"])
        image = Image.open(image_path).convert("RGB")
        record_gt_struct = record.get("gt_struct")
        if record_gt_struct is not None:
            gt_struct = record_gt_struct
        else:
            instances = record.get("instances", [])
            gt_struct = {
                "instances": [
                    self._build_gt_instance(instance, task_type=task_type)
                    for instance in instances
                ]
            }
        target_text = record.get("target_text")
        if target_text is None:
            target_text = self._encode_target_text(
                gt_struct,
                task_type=task_type,
                image_width=record["image_width"],
                image_height=record["image_height"],
            )
        condition = record.get("condition", {})
        system_prompt = record.get("system_prompt")
        if system_prompt is None:
            template = record.get("system_prompt_template", self.system_prompt_template)
            if template:
                system_prompt = render_prompt_template(template, condition)
            else:
                system_prompt = self.system_prompt
        user_prompt = record.get("user_prompt")
        if user_prompt is None:
            template = record.get("user_prompt_template", self.user_prompt_template)
            if template:
                user_prompt = render_prompt_template(template, condition)
            else:
                user_prompt = self.user_prompt
        return {
            "task_type": task_type,
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

    def get_target_token_lengths(self, tokenizer) -> list[int]:
        cache_key = id(tokenizer)
        cached = self._target_token_lengths_cache.get(cache_key)
        if cached is not None:
            return cached
        lengths: list[int] = []
        for record in self.records:
            task_type = str(record.get("task_type", "one_stage"))
            record_gt_struct = record.get("gt_struct")
            if record_gt_struct is not None:
                gt_struct = record_gt_struct
            else:
                instances = record.get("instances", [])
                gt_struct = {
                    "instances": [
                        self._build_gt_instance(instance, task_type=task_type)
                        for instance in instances
                    ]
                }
            target_text = record.get("target_text")
            if target_text is None:
                target_text = self._encode_target_text(
                    gt_struct,
                    task_type=task_type,
                    image_width=record["image_width"],
                    image_height=record["image_height"],
                )
            tokenized = tokenizer(
                str(target_text),
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            lengths.append(len(tokenized["input_ids"]))
        self._target_token_lengths_cache[cache_key] = lengths
        return lengths

    @staticmethod
    def _build_gt_instance(instance: dict[str, Any], *, task_type: str) -> dict[str, Any]:
        gt_instance = {
            "label": instance["label"],
            "bbox": instance["bbox"],
        }
        if task_type != "two_stage_stage1_grounding":
            gt_instance["keypoints"] = instance["keypoints"]
        return gt_instance

    def _encode_target_text(
        self,
        gt_struct: dict[str, Any],
        *,
        task_type: str,
        image_width: int,
        image_height: int,
    ) -> str:
        if task_type == "two_stage_stage1_grounding":
            return self.grounding_codec.encode(
                gt_struct,
                image_width=image_width,
                image_height=image_height,
            )
        return self.codec.encode(
            gt_struct,
            image_width=image_width,
            image_height=image_height,
        )
