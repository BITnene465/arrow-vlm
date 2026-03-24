#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from vlm_det.infer.visualize import format_prediction_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run greedy inference on one image.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default=None, help="Optional directory to save parsed prediction files.")
    return parser.parse_args()


def _save_outputs(
    output_dir: Path,
    image_path: Path,
    raw_text: str,
    parse_report: dict[str, object],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    prediction_path = output_dir / f"{stem}.prediction.json"
    raw_text_path = output_dir / f"{stem}.raw.txt"
    prediction_path.write_text(json.dumps(parse_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw_text_path.write_text(raw_text + "\n", encoding="utf-8")
    return prediction_path, raw_text_path


def main() -> None:
    args = parse_args()
    from vlm_det.infer.runner import load_inference_runner

    runner = load_inference_runner(args.config, args.checkpoint)
    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    raw_text, parse_report = runner.predict(image)
    print(json.dumps(parse_report, ensure_ascii=False, indent=2))

    lenient_prediction = parse_report["lenient"]["prediction"]
    if lenient_prediction is not None:
        print(format_prediction_summary(lenient_prediction))
    else:
        print("Detected arrows: parse failed")

    print(
        "\n".join(
            [
                f"Lenient parse ok: {parse_report['lenient']['ok']}",
                f"Strict parse ok: {parse_report['strict']['ok']}",
            ]
        )
    )
    if args.output_dir is not None:
        prediction_path, raw_text_path = _save_outputs(Path(args.output_dir), image_path, raw_text, parse_report)
        print(f"Saved parsed prediction to: {prediction_path}")
        print(f"Saved raw output to: {raw_text_path}")


if __name__ == "__main__":
    main()
