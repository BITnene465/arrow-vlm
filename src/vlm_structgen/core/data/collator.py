from __future__ import annotations

from typing import Any

import torch


class SFTCollator:
    def __init__(
        self,
        processor,
        tokenizer,
        add_eos_token: bool = True,
        ignore_index: int = -100,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        include_targets_in_inputs: bool = True,
        padding_side: str = "right",
    ) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.add_eos_token = add_eos_token
        self.ignore_index = ignore_index
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.include_targets_in_inputs = include_targets_in_inputs
        if padding_side not in {"left", "right"}:
            raise ValueError(f"Unsupported padding_side={padding_side!r}. Expected 'left' or 'right'.")
        self.padding_side = padding_side

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        messages = [self._build_messages(item["system_prompt"], item["user_prompt"]) for item in batch]
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
        final_mm_token_type_ids: list[torch.Tensor] = []
        prompt_length_tensor: list[int] = []
        prefix_mm_token_type_ids = prefix_batch.get("mm_token_type_ids")

        for row_index, prompt_length in enumerate(prompt_lengths):
            prefix_mask = prefix_batch["attention_mask"][row_index].bool()
            prefix_ids = prefix_batch["input_ids"][row_index][prefix_mask]
            prefix_mm_ids = None
            if prefix_mm_token_type_ids is not None:
                prefix_mm_ids = prefix_mm_token_type_ids[row_index][prefix_mask]
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
                if prefix_mm_ids is not None:
                    target_mm_ids = torch.zeros_like(target_tensor)
                    mm_token_type_ids = torch.cat([prefix_mm_ids, target_mm_ids], dim=0)
            else:
                input_ids = prefix_ids
                labels = torch.full((prefix_ids.shape[0],), self.ignore_index, dtype=torch.long)
                if prefix_mm_ids is not None:
                    mm_token_type_ids = prefix_mm_ids
            attention_mask = torch.ones_like(input_ids)
            final_input_ids.append(input_ids)
            final_labels.append(labels)
            final_attention_masks.append(attention_mask)
            if prefix_mm_ids is not None:
                final_mm_token_type_ids.append(mm_token_type_ids)
            prompt_length_tensor.append(prefix_ids.shape[0])

        padded_input_ids = self._pad_sequences(
            final_input_ids,
            padding_value=pad_id,
        )
        padded_labels = self._pad_sequences(
            final_labels,
            padding_value=self.ignore_index,
        )
        padded_attention_masks = self._pad_sequences(
            final_attention_masks,
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
                "task_type": [item["task_type"] for item in batch],
                "domain_type": [item["domain_type"] for item in batch],
                "sample_id": [item["sample_id"] for item in batch],
                "image_path": [item["image_path"] for item in batch],
                "image_width": [item["image_width"] for item in batch],
                "image_height": [item["image_height"] for item in batch],
                "system_prompt": [item["system_prompt"] for item in batch],
                "user_prompt": [item["user_prompt"] for item in batch],
                "gt_struct": [item["gt_struct"] for item in batch],
                "target_text": [item["target_text"] for item in batch],
            },
        }
        if prefix_mm_token_type_ids is not None:
            output["mm_token_type_ids"] = self._pad_sequences(
                final_mm_token_type_ids,
                padding_value=0,
            )
        return output

    def _build_messages(self, system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        )
        return messages

    def _apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        template_owner = self.processor if hasattr(self.processor, "apply_chat_template") else self.tokenizer
        return template_owner.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _pad_sequences(self, sequences: list[torch.Tensor], padding_value: int) -> torch.Tensor:
        max_length = max(sequence.shape[0] for sequence in sequences)
        padded: list[torch.Tensor] = []
        for sequence in sequences:
            if sequence.shape[0] == max_length:
                padded.append(sequence)
                continue
            pad = torch.full(
                (max_length - sequence.shape[0],),
                padding_value,
                dtype=sequence.dtype,
            )
            if self.padding_side == "left":
                padded.append(torch.cat([pad, sequence], dim=0))
            else:
                padded.append(torch.cat([sequence, pad], dim=0))
        return torch.stack(padded, dim=0)
