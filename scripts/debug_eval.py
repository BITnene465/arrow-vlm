#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vlm_det.config import load_config
from vlm_det.data.collator import ArrowSFTCollator
from vlm_det.data.dataset import ArrowSFTDataset
from vlm_det.modeling.builder import build_model_tokenizer_processor
from vlm_det.protocol.codec import ArrowCodec
from vlm_det.utils.checkpoint import load_training_checkpoint
from vlm_det.utils.distributed import unwrap_model
from vlm_det.utils.generation import build_generate_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose eval-time generation behavior on a few samples.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--show-text", action="store_true")
    return parser.parse_args()


def _build_dataset(config, codec: ArrowCodec, split: str) -> ArrowSFTDataset:
    jsonl_path = config.data.val_path if split == "val" else config.data.train_path
    return ArrowSFTDataset(
        jsonl_path=jsonl_path,
        codec=codec,
        system_prompt=config.prompt.system_prompt,
        user_prompt=config.prompt.user_prompt,
        shuffle_instances=False,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.max_new_tokens is not None:
        config.eval.max_new_tokens = args.max_new_tokens

    artifacts = build_model_tokenizer_processor(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts.model = artifacts.model.to(device)
    load_training_checkpoint(
        checkpoint_dir=args.checkpoint,
        model=artifacts.model,
        tokenizer=artifacts.tokenizer,
        processor=artifacts.processor,
        strict=True,
        resume_training_state=False,
    )
    model = unwrap_model(artifacts.model)
    model.eval()

    codec = ArrowCodec(num_bins=config.tokenizer.num_bins)
    dataset = _build_dataset(config, codec, args.split)
    collator = ArrowSFTCollator(
        processor=artifacts.processor,
        tokenizer=artifacts.tokenizer,
        add_eos_token=config.tokenizer.add_eos_token,
        min_pixels=config.model.min_pixels,
        max_pixels=config.model.max_pixels,
        include_targets_in_inputs=False,
        padding_side="left",
    )

    end_index = min(args.start_index + args.num_samples, len(dataset))
    if args.start_index >= end_index:
        raise ValueError("Requested sample range is empty.")

    print(
        json.dumps(
            {
                "split": args.split,
                "range": [args.start_index, end_index],
                "max_new_tokens": config.eval.max_new_tokens,
                "model": config.model.model_name_or_path,
                "checkpoint": str(Path(args.checkpoint)),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for index in range(args.start_index, end_index):
        sample = dataset[index]
        batch = collator([sample])
        prompt_length = int(batch["prompt_lengths"][0].item())
        input_context_length = int(batch["input_ids"].shape[1])
        model_inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "pixel_values": batch["pixel_values"].to(device),
        }
        if batch.get("image_grid_thw") is not None:
            model_inputs["image_grid_thw"] = batch["image_grid_thw"].to(device)
        generate_kwargs = build_generate_kwargs(
            artifacts.tokenizer,
            num_bins=codec.num_bins,
            prompt_lengths=[prompt_length],
            max_new_tokens=config.eval.max_new_tokens,
            num_beams=config.eval.num_beams,
            do_sample=config.eval.do_sample,
            use_cache=config.eval.use_cache,
        )

        start_time = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(**model_inputs, **generate_kwargs)
        elapsed = time.perf_counter() - start_time

        continuation = output_ids[0, input_context_length:]
        continuation_ids = continuation.tolist()
        generated_tokens = len(continuation_ids)
        text = artifacts.tokenizer.decode(continuation, skip_special_tokens=False)
        try:
            prediction = codec.decode(
                text,
                image_width=sample["image_width"],
                image_height=sample["image_height"],
            )
            parse_ok = True
            predicted_instances = len(prediction.get("instances", []))
        except Exception as exc:  # noqa: BLE001
            parse_ok = False
            predicted_instances = 0
            prediction = {"error": str(exc)}

        summary = {
            "index": index,
            "sample_id": sample["sample_id"],
            "image_size": [sample["image_width"], sample["image_height"]],
            "prompt_tokens": prompt_length,
            "generated_tokens": generated_tokens,
            "truncated": generated_tokens >= config.eval.max_new_tokens,
            "parse_ok": parse_ok,
            "pred_instances": predicted_instances,
            "gt_instances": len(sample["gt_struct"]["instances"]),
            "elapsed_sec": round(elapsed, 3),
            "tokens_per_sec": round(generated_tokens / max(elapsed, 1e-6), 3),
        }
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if args.show_text:
            print(text, flush=True)
        if not parse_ok:
            print(json.dumps(prediction, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
