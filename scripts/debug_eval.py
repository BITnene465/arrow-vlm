#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vlm_structgen.core import load_config
from vlm_structgen.core.data import SFTCollator, SFTDataset
from vlm_structgen.core.modeling.builder import build_model_tokenizer_processor
from vlm_structgen.core.utils.checkpoint import load_training_checkpoint
from vlm_structgen.core.utils.distributed import reset_model_runtime_state, unwrap_model
from vlm_structgen.core.utils.generation import build_generate_kwargs, trim_generated_ids_at_eos


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


def _build_dataset(config, split: str) -> SFTDataset:
    jsonl_path = config.data.val_path if split == "val" else config.data.train_path
    return SFTDataset(
        jsonl_path=jsonl_path,
        num_bins=config.tokenizer.num_bins,
        system_prompt=config.prompt.system_prompt,
        user_prompt=config.prompt.user_prompt,
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

    dataset = _build_dataset(config, args.split)
    collator = SFTCollator(
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
        if batch.get("mm_token_type_ids") is not None:
            model_inputs["mm_token_type_ids"] = batch["mm_token_type_ids"].to(device)
        if batch.get("image_grid_thw") is not None:
            model_inputs["image_grid_thw"] = batch["image_grid_thw"].to(device)
        generate_kwargs = build_generate_kwargs(
            artifacts.tokenizer,
            generation_config=getattr(model, "generation_config", None),
            num_bins=config.tokenizer.num_bins,
            prompt_lengths=[prompt_length],
            max_new_tokens=config.eval.max_new_tokens,
            num_beams=config.eval.num_beams,
            do_sample=config.eval.do_sample,
            temperature=config.eval.temperature,
            top_p=config.eval.top_p,
            top_k=config.eval.top_k,
            use_cache=config.eval.use_cache,
        )

        start_time = time.perf_counter()
        with torch.inference_mode():
            reset_model_runtime_state(model)
            output_ids = model.generate(**model_inputs, **generate_kwargs)
        elapsed = time.perf_counter() - start_time

        continuation = output_ids[0, input_context_length:]
        continuation_ids = trim_generated_ids_at_eos(continuation, generate_kwargs.get("eos_token_id"))
        generated_tokens = len(continuation_ids)
        text = artifacts.tokenizer.decode(continuation_ids, skip_special_tokens=False)
        strict_text = artifacts.tokenizer.decode(continuation_ids, skip_special_tokens=True)
        try:
            prediction = codec.decode(
                text,
                image_width=sample["image_width"],
                image_height=sample["image_height"],
            )
            parse_ok_lenient = True
            predicted_instances = len(prediction.get("instances", []))
        except Exception as exc:  # noqa: BLE001
            parse_ok_lenient = False
            predicted_instances = 0
            prediction = {"lenient_error": str(exc)}

        strict_error = None
        if parse_ok_lenient:
            try:
                codec.decode(
                    strict_text,
                    image_width=sample["image_width"],
                    image_height=sample["image_height"],
                    strict=True,
                )
                parse_ok_strict = True
            except Exception as exc:  # noqa: BLE001
                parse_ok_strict = False
                strict_error = str(exc)
        else:
            parse_ok_strict = False
            strict_error = prediction["lenient_error"]

        summary = {
            "index": index,
            "sample_id": sample["sample_id"],
            "image_size": [sample["image_width"], sample["image_height"]],
            "prompt_tokens": prompt_length,
            "generated_tokens": generated_tokens,
            "truncated": generated_tokens >= config.eval.max_new_tokens,
            "parse_ok_lenient": parse_ok_lenient,
            "parse_ok_strict": parse_ok_strict,
            "pred_instances": predicted_instances,
            "gt_instances": len(sample["gt_struct"]["instances"]),
            "elapsed_sec": round(elapsed, 3),
            "tokens_per_sec": round(generated_tokens / max(elapsed, 1e-6), 3),
        }
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if args.show_text:
            print(text, flush=True)
        if not parse_ok_lenient or strict_error is not None:
            print(
                json.dumps(
                    {
                        **prediction,
                        **({"strict_error": strict_error} if strict_error is not None else {}),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
