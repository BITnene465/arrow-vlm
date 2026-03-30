#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from vlm_structgen.domains.arrow import draw_prediction, format_prediction_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a saved prediction JSON on top of the source image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prediction", required=True, help="Path to *.prediction.json saved by scripts/infer.py")
    parser.add_argument("--output", default=None, help="Optional output image path. Defaults next to prediction JSON.")
    return parser.parse_args()


def _default_output_path(prediction_path: Path) -> Path:
    stem = prediction_path.name
    if stem.endswith(".prediction.json"):
        stem = stem[: -len(".prediction.json")]
    else:
        stem = prediction_path.stem
    return prediction_path.with_name(f"{stem}.vis.jpg")


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    prediction_path = Path(args.prediction)
    output_path = Path(args.output) if args.output is not None else _default_output_path(prediction_path)

    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    image = Image.open(image_path).convert("RGB")
    overlay = draw_prediction(image, prediction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path, quality=95)

    print(json.dumps(prediction, ensure_ascii=False, indent=2))
    print(format_prediction_summary(prediction))
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
