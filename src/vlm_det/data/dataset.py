from __future__ import annotations

import copy
import random
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
        shuffle_instances: bool = False,
    ) -> None:
        self.records = load_jsonl(jsonl_path)
        self.codec = codec
        self.system_prompt = system_prompt
        self.shuffle_instances = shuffle_instances

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = copy.deepcopy(self.records[index])
        image_path = Path(record["image_path"])
        image = Image.open(image_path).convert("RGB")
        instances = record.get("instances", [])
        if self.shuffle_instances and len(instances) > 1:
            random.shuffle(instances)
        gt_struct = {
            "instances": [
                {
                    "bbox": instance["bbox"],
                    "keypoints": instance["keypoints"],
                    **({"group_id": instance["group_id"]} if "group_id" in instance else {}),
                    **({"raw_bbox": instance["raw_bbox"]} if "raw_bbox" in instance else {}),
                    **(
                        {"raw_keypoints": instance["raw_keypoints"]}
                        if "raw_keypoints" in instance
                        else {}
                    ),
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
            "target_text": target_text,
            "gt_struct": gt_struct,
        }
