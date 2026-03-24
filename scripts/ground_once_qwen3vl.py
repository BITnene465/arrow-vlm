#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DEFAULT_PROMPT = (
    'Locate every instance that belongs to the following categories: '
    '"head, hand, man, woman, glasses". Report bbox coordinates in JSON format.'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-off grounding with Qwen3-VL-2B.")
    parser.add_argument("--model-path", default="models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--image", default="data/raw/figure/00001.jpg")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--attn-implementation", default=None)
    return parser.parse_args()


def _default_output_path(image_path: Path) -> Path:
    return Path("outputs/qwen3vl_grounding_once") / f"{image_path.stem}.json"


def _default_visualization_path(output_path: Path) -> Path:
    return output_path.with_suffix(".vis.jpg")


def _extract_json_candidate(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped

    code_block_match = re.search(r"```(?:json)?\s*(.+?)\s*```", text, flags=re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    object_match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if object_match:
        return object_match.group(1).strip()
    return None


def _safe_parse_json(text: str) -> Any | None:
    candidate = _extract_json_candidate(text)
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _recover_grounding_predictions(text: str) -> list[dict[str, Any]]:
    parsed = _safe_parse_json(text)
    if isinstance(parsed, list):
        return [
            item
            for item in parsed
            if isinstance(item, dict) and "bbox_2d" in item and "label" in item
        ]

    pattern = re.compile(
        r'\{\s*"bbox_2d"\s*:\s*\[\s*'
        r'(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*'
        r'\]\s*,\s*"label"\s*:\s*"([^"]+)"\s*\}'
    )
    predictions: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        x1, y1, x2, y2, label = match.groups()
        predictions.append(
            {
                "bbox_2d": [int(x1), int(y1), int(x2), int(y2)],
                "label": label,
            }
        )
    return predictions


def _draw_predictions(image: Image.Image, predictions: list[dict[str, Any]]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(predictions, start=1):
        bbox = item.get("bbox_2d")
        label = str(item.get("label", "object"))
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(value) for value in bbox]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1 + 2, max(y1 - 14, 0)), f"{index}:{label}", fill="red")
    return canvas


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    output_path = Path(args.output) if args.output else _default_output_path(image_path)

    model_kwargs: dict[str, Any] = {
        "torch_dtype": "auto",
        "device_map": "auto",
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_path)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    parsed_json = _safe_parse_json(response_text)
    recovered_predictions = _recover_grounding_predictions(response_text)
    visualization_path = _default_visualization_path(output_path)
    visualization = _draw_predictions(image, recovered_predictions)

    result = {
        "model_path": str(args.model_path),
        "image": str(image_path),
        "prompt": args.prompt,
        "response_text": response_text,
        "parsed_json": parsed_json,
        "recovered_predictions": recovered_predictions,
        "num_recovered_predictions": len(recovered_predictions),
        "visualization": str(visualization_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visualization.save(visualization_path, quality=95)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved json to: {output_path}")
    print(f"Saved visualization to: {visualization_path}")


if __name__ == "__main__":
    main()
