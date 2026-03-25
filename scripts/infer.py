#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from vlm_det.infer.visualize import format_prediction_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run greedy inference on one image.")
    parser.add_argument("--config", default=None, help="Legacy training config path. Prefer environment-driven inference settings.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory. Falls back to CHECKPOINT_PATH in .env.")
    parser.add_argument("--env-file", default=None, help="Optional path to a .env file for inference/app settings.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override inference max_new_tokens for this run.")
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

    runner = load_inference_runner(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        env_file=args.env_file,
    )
    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    raw_text, parse_report = runner.predict(image, max_new_tokens=args.max_new_tokens)
    print(json.dumps(parse_report, ensure_ascii=False, indent=2))
    print("\n[raw-output]")
    print(raw_text)
    generation = parse_report["generation"]
    print(
        "\n".join(
            [
                "[generation]",
                f"requested_max_new_tokens={generation['requested_max_new_tokens']}",
                f"generated_tokens={generation['generated_tokens']}",
                f"returned_tokens={generation['returned_tokens']}",
                f"stop_reason={generation['stop_reason']}",
                f"saw_eos={generation['saw_eos']}",
                f"closed_json_array={generation['closed_json_array']}",
                f"hit_max_new_tokens={generation['hit_max_new_tokens']}",
            ]
        )
    )

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
    output_dir = args.output_dir or runner.settings.output_dir
    if output_dir is not None:
        prediction_path, raw_text_path = _save_outputs(Path(output_dir), image_path, raw_text, parse_report)
        print(f"Saved parsed prediction to: {prediction_path}")
        print(f"Saved raw output to: {raw_text_path}")


if __name__ == "__main__":
    main()
