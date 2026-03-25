from __future__ import annotations

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


class ArrowSFTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        codec: ArrowCodec,
        system_prompt: str,
        user_prompt: str,
    ) -> None:
        self.records = load_jsonl(jsonl_path)
        self.codec = codec
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = Path(record["image_path"])
        image = Image.open(image_path).convert("RGB")
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
        target_text = self.codec.encode(
            gt_struct,
            image_width=record["image_width"],
            image_height=record["image_height"],
        )
        return {
            "sample_id": record.get("sample_id", image_path.stem),
            "image_path": str(image_path),
            "image": image,
            "image_width": int(record["image_width"]),
            "image_height": int(record["image_height"]),
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "target_text": target_text,
            "gt_struct": gt_struct,
        }
