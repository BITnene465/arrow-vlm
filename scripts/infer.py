#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from vlm_det.infer.runner import load_inference_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run greedy inference on one image.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = load_inference_runner(args.config, args.checkpoint)
    image = Image.open(Path(args.image)).convert("RGB")
    _, prediction = runner.predict(image)
    print(json.dumps(prediction, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
