from __future__ import annotations

from typing import Any

import torch


class ArrowSFTCollator:
    def __init__(
        self,
        processor,
        tokenizer,
        add_eos_token: bool = True,
        ignore_index: int = -100,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        include_targets_in_inputs: bool = True,
    ) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.add_eos_token = add_eos_token
        self.ignore_index = ignore_index
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.include_targets_in_inputs = include_targets_in_inputs

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        messages = [self._build_messages(item["system_prompt"]) for item in batch]
        prefix_texts = [self._apply_chat_template(message) for message in messages]
        images = [item["image"] for item in batch]
        processor_kwargs = {
            "text": prefix_texts,
            "images": images,
            "padding": True,
            "return_tensors": "pt",
        }
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        prefix_batch = self.processor(**processor_kwargs)

        target_batch = self.tokenizer(
            [item["target_text"] for item in batch],
            add_special_tokens=False,
            return_attention_mask=False,
        )
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id

        prompt_lengths = prefix_batch["attention_mask"].sum(dim=1).tolist()
        final_input_ids: list[torch.Tensor] = []
        final_labels: list[torch.Tensor] = []
        final_attention_masks: list[torch.Tensor] = []
        prompt_length_tensor: list[int] = []

        for row_index, prompt_length in enumerate(prompt_lengths):
            prefix_mask = prefix_batch["attention_mask"][row_index].bool()
            prefix_ids = prefix_batch["input_ids"][row_index][prefix_mask]
            if self.include_targets_in_inputs:
                target_ids = list(target_batch["input_ids"][row_index])
                if self.add_eos_token and eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
                    target_ids.append(eos_id)
                target_tensor = torch.tensor(target_ids, dtype=torch.long)
                input_ids = torch.cat([prefix_ids, target_tensor], dim=0)
                labels = torch.cat(
                    [
                        torch.full((prefix_ids.shape[0],), self.ignore_index, dtype=torch.long),
                        target_tensor.clone(),
                    ],
                    dim=0,
                )
            else:
                input_ids = prefix_ids
                labels = torch.full((prefix_ids.shape[0],), self.ignore_index, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            final_input_ids.append(input_ids)
            final_labels.append(labels)
            final_attention_masks.append(attention_mask)
            prompt_length_tensor.append(prefix_ids.shape[0])

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            final_input_ids,
            batch_first=True,
            padding_value=pad_id,
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            final_labels,
            batch_first=True,
            padding_value=self.ignore_index,
        )
        padded_attention_masks = torch.nn.utils.rnn.pad_sequence(
            final_attention_masks,
            batch_first=True,
            padding_value=0,
        )

        output = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_masks,
            "labels": padded_labels,
            "pixel_values": prefix_batch["pixel_values"],
            "image_grid_thw": prefix_batch.get("image_grid_thw"),
            "prompt_lengths": torch.tensor(prompt_length_tensor, dtype=torch.long),
            "meta": {
                "sample_id": [item["sample_id"] for item in batch],
                "image_path": [item["image_path"] for item in batch],
                "image_width": [item["image_width"] for item in batch],
                "image_height": [item["image_height"] for item in batch],
                "system_prompt": [item["system_prompt"] for item in batch],
                "gt_struct": [item["gt_struct"] for item in batch],
                "target_text": [item["target_text"] for item in batch],
            },
        }
        return output

    def _build_messages(self, system_prompt: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "image"}],
            },
        ]

    def _apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        template_owner = self.processor if hasattr(self.processor, "apply_chat_template") else self.tokenizer
        return template_owner.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
