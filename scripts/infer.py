#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from vlm_det.config import load_config
from vlm_det.modeling.builder import build_model_tokenizer_processor
from vlm_det.protocol.codec import ArrowCodec
from vlm_det.utils.checkpoint import load_training_checkpoint
from vlm_det.utils.distributed import unwrap_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run greedy inference on one image.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
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
    image = Image.open(Path(args.image)).convert("RGB")
    width, height = image.size
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": config.prompt.system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "image"}],
        },
    ]
    template_owner = (
        artifacts.processor
        if hasattr(artifacts.processor, "apply_chat_template")
        else artifacts.tokenizer
    )
    prompt = template_owner.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processor_kwargs = {
        "text": [prompt],
        "images": [image],
        "return_tensors": "pt",
    }
    if config.model.min_pixels is not None:
        processor_kwargs["min_pixels"] = config.model.min_pixels
    if config.model.max_pixels is not None:
        processor_kwargs["max_pixels"] = config.model.max_pixels
    batch = artifacts.processor(**processor_kwargs)
    model_inputs = {key: value.to(device) for key, value in batch.items()}
    prompt_length = int(batch["attention_mask"].sum(dim=1).item())
    raw_model = unwrap_model(artifacts.model)
    raw_model.eval()
    with torch.no_grad():
        output_ids = raw_model.generate(
            **model_inputs,
            max_new_tokens=config.eval.max_new_tokens,
            num_beams=config.eval.num_beams,
            do_sample=config.eval.do_sample,
            use_cache=config.eval.use_cache,
            pad_token_id=artifacts.tokenizer.pad_token_id,
        )
    continuation = output_ids[0, prompt_length:]
    decoded = artifacts.tokenizer.decode(continuation, skip_special_tokens=False)
    codec = ArrowCodec(num_bins=config.tokenizer.num_bins)
    prediction = codec.decode(decoded, image_width=width, image_height=height)
    print(json.dumps(prediction, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
