from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from vlm_structgen.domains.arrow.codecs.grounding import GroundingCodec
from vlm_structgen.domains.arrow.task_support import BaseArrowAdapter, empty_counts, match_instances


@dataclass
class ArrowGroundingAdapter(BaseArrowAdapter):
    task_type: str = field(init=False, default="grounding")
    task_bucket_key: str = field(init=False, default="grounding_samples")
    bbox_token_loss_weight: float = 1.0
    label_token_loss_weight: float = 1.0

    def build_gt_struct_from_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "instances": [
                {
                    "label": instance["label"],
                    "bbox": instance["bbox"],
                    "keypoints": [],
                }
                for instance in record.get("instances", [])
            ]
        }

    def build_training_target(
        self,
        gt_struct: dict[str, Any],
        *,
        image_width: int,
        image_height: int,
    ) -> dict[str, Any]:
        target_text, loss_meta = self.codec.encode_with_loss_meta(
            gt_struct,
            image_width=image_width,
            image_height=image_height,
        )
        return {
            "target_text": target_text,
            "loss_meta": loss_meta,
        }

    def encode_target_text(self, gt_struct: dict[str, Any], *, image_width: int, image_height: int) -> str:
        return self.codec.encode(gt_struct, image_width=image_width, image_height=image_height)

    def decode_with_meta(
        self,
        text: str,
        *,
        image_width: int,
        image_height: int,
        strict: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.codec.decode_with_meta(text, image_width=image_width, image_height=image_height, strict=strict)

    def decode(self, text: str, *, image_width: int, image_height: int, strict: bool = False) -> dict[str, Any]:
        return self.codec.decode(text, image_width=image_width, image_height=image_height, strict=strict)

    def score_prediction(
        self,
        gt_struct: dict[str, Any],
        pred_struct: dict[str, Any],
        *,
        bbox_iou_threshold: float,
        strict_point_distance_px: float,
    ) -> dict[str, float]:
        del strict_point_distance_px
        counts = empty_counts()
        gt_instances = gt_struct.get("instances", [])
        pred_instances = pred_struct.get("instances", [])
        counts["gt_instances"] = float(len(gt_instances))
        counts["pred_instances"] = float(len(pred_instances))

        matches = match_instances(gt_instances, pred_instances, bbox_iou_threshold=bbox_iou_threshold)
        matched_gt = set()
        matched_pred = set()
        for gt_index, pred_index, iou_value in matches:
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
            counts["bbox_tp"] += 1.0
            counts["bbox_iou_sum"] += iou_value

        counts["bbox_fp"] = float(len(pred_instances) - len(matched_pred))
        counts["bbox_fn"] = float(len(gt_instances) - len(matched_gt))
        return counts

    def compute_loss(self, model_outputs, batch: dict[str, Any], *, tokenizer=None) -> object:
        if tokenizer is None:
            return model_outputs.loss
        if float(self.bbox_token_loss_weight) <= 1.0 and float(self.label_token_loss_weight) <= 1.0:
            return model_outputs.loss

        logits = model_outputs.logits
        labels = batch["labels"].to(logits.device)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_weights = self._build_shift_weights(
            batch=batch,
            labels=labels,
            tokenizer=tokenizer,
            device=logits.device,
        )
        if shift_weights is None:
            return model_outputs.loss

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        ).view_as(shift_labels)
        valid_mask = (shift_labels != -100).to(token_loss.dtype)
        weighted_loss = token_loss * shift_weights * valid_mask
        denom = (shift_weights * valid_mask).sum().clamp_min(1.0)
        return weighted_loss.sum() / denom

    def _build_shift_weights(
        self,
        *,
        batch: dict[str, Any],
        labels: torch.Tensor,
        tokenizer,
        device: torch.device,
    ) -> torch.Tensor | None:
        sequence_weights = torch.ones_like(labels, dtype=torch.float32, device=device)
        meta = batch.get("meta", {})
        target_texts = list(meta.get("target_text", []))
        loss_metas = list(meta.get("loss_meta", []))
        if len(target_texts) != labels.shape[0] or len(loss_metas) != labels.shape[0]:
            return None
        for row_index, (target_text, loss_meta) in enumerate(zip(target_texts, loss_metas, strict=True)):
            token_weights = self._target_token_weights(
                str(target_text),
                loss_meta=loss_meta,
                tokenizer=tokenizer,
            )
            if token_weights is None:
                return None
            valid_positions = torch.nonzero(labels[row_index] != -100, as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue
            if len(token_weights) != int(valid_positions.numel()):
                if len(token_weights) + 1 == int(valid_positions.numel()):
                    token_weights = token_weights + [1.0]
                else:
                    limit = min(len(token_weights), int(valid_positions.numel()))
                    token_weights = token_weights[:limit]
                    valid_positions = valid_positions[:limit]
            if not token_weights:
                continue
            sequence_weights[row_index, valid_positions] = torch.tensor(
                token_weights,
                dtype=torch.float32,
                device=device,
            )
        return sequence_weights[:, 1:].contiguous()

    def _target_token_weights(
        self,
        target_text: str,
        *,
        loss_meta: dict[str, Any] | None,
        tokenizer,
    ) -> list[float] | None:
        encoded = tokenizer(
            target_text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping")
        input_ids = encoded.get("input_ids")
        if offsets is None or input_ids is None:
            return None

        field_spans = dict((loss_meta or {}).get("field_char_spans", {}))
        weighted_spans: list[tuple[int, int, float]] = []
        for start, end in field_spans.get("label", []):
            weighted_spans.append((int(start), int(end), float(self.label_token_loss_weight)))
        for start, end in field_spans.get("bbox_2d", []):
            weighted_spans.append((int(start), int(end), float(self.bbox_token_loss_weight)))

        weights: list[float] = []
        for start, end in offsets:
            token_weight = 1.0
            if end > start:
                for span_start, span_end, span_weight in weighted_spans:
                    if max(int(start), span_start) < min(int(end), span_end):
                        token_weight = max(token_weight, span_weight)
            weights.append(float(token_weight))
        if tokenizer.eos_token_id is not None:
            weights.append(1.0)
        return weights


def build_grounding_adapter(*, domain_type: str, num_bins: int, task_options: dict[str, Any] | None = None):
    task_options = dict(task_options or {})
    if domain_type == "arrow":
        return ArrowGroundingAdapter(
            codec=GroundingCodec(num_bins=num_bins),
            bbox_token_loss_weight=float(task_options.get("bbox_token_loss_weight", 1.0)),
            label_token_loss_weight=float(task_options.get("label_token_loss_weight", 1.0)),
        )
    raise ValueError(f"Unsupported grounding domain_type: {domain_type!r}")
