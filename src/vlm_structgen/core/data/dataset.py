from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from vlm_structgen.core.registry import get_adapter
from vlm_structgen.core.prompting import render_prompt_template
from vlm_structgen.core.utils.io import load_jsonl

# Training can encounter extremely large figure images. We only decode images
# that are already part of the trusted dataset, so disable Pillow's
# decompression bomb guard here as well.
Image.MAX_IMAGE_PIXELS = None


class SFTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        num_bins: int,
        system_prompt: str,
        user_prompt: str,
        system_prompt_template: str | None = None,
        user_prompt_template: str | None = None,
    ) -> None:
        self.records = load_jsonl(jsonl_path)
        self.num_bins = int(num_bins)
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self._target_token_lengths_cache: dict[int, list[int]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        adapter = self._get_adapter(record)
        image_path = Path(record["image_path"])
        image = Image.open(image_path).convert("RGB")
        record_gt_struct = record.get("gt_struct")
        if record_gt_struct is not None:
            gt_struct = record_gt_struct
        else:
            gt_struct = adapter.build_gt_struct_from_record(record)
        target_text = record.get("target_text")
        loss_meta = record.get("loss_meta")
        if target_text is None or loss_meta is None:
            training_target = adapter.build_training_target(
                gt_struct,
                image_width=record["image_width"],
                image_height=record["image_height"],
            )
            target_text = training_target["target_text"]
            loss_meta = training_target.get("loss_meta")
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
            "task_type": adapter.task_type,
            "domain_type": adapter.domain_type,
            "sample_id": record.get("sample_id", image_path.stem),
            "image_path": str(image_path),
            "image": image,
            "image_width": int(record["image_width"]),
            "image_height": int(record["image_height"]),
            "system_prompt": str(system_prompt),
            "user_prompt": str(user_prompt),
            "target_text": str(target_text),
            "loss_meta": loss_meta,
            "gt_struct": gt_struct,
        }

    def get_target_token_lengths(self, tokenizer) -> list[int]:
        cache_key = id(tokenizer)
        cached = self._target_token_lengths_cache.get(cache_key)
        if cached is not None:
            return cached
        lengths: list[int] = []
        for record in self.records:
            adapter = self._get_adapter(record)
            record_gt_struct = record.get("gt_struct")
            if record_gt_struct is not None:
                gt_struct = record_gt_struct
            else:
                gt_struct = adapter.build_gt_struct_from_record(record)
            target_text = record.get("target_text")
            if target_text is None:
                target_text = adapter.build_training_target(
                    gt_struct,
                    image_width=record["image_width"],
                    image_height=record["image_height"],
                )["target_text"]
            tokenized = tokenizer(
                str(target_text),
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            lengths.append(len(tokenized["input_ids"]))
        self._target_token_lengths_cache[cache_key] = lengths
        return lengths

    def _get_adapter(self, record: dict[str, Any]):
        try:
            return get_adapter(
                task_type=record.get("task_type"),
                domain_type=record.get("domain_type"),
                num_bins=self.num_bins,
            )
        except Exception as exc:  # noqa: BLE001
            sample_id = record.get("sample_id") or Path(str(record.get("image_path", ""))).stem or "<unknown>"
            raise ValueError(
                "Dataset record is missing a valid task/domain route. "
                f"sample_id={sample_id!r}, task_type={record.get('task_type')!r}, "
                f"domain_type={record.get('domain_type')!r}. "
                "Regenerate the dataset with the current preparation scripts."
            ) from exc
