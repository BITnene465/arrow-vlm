#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from vlm_det.infer.visualize import draw_prediction, format_prediction_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-stage arrow inference on one image.")
    parser.add_argument("--config", default="configs/infer/infer_two_stage.yaml", help="Two-stage inference config path.")
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage2-checkpoint", default=None)
    parser.add_argument("--stage1-model", default=None)
    parser.add_argument("--stage2-model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image", required=True)
    parser.add_argument("--stage1-max-new-tokens", type=int, default=None)
    parser.add_argument("--stage2-max-new-tokens", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _save_outputs(output_dir: Path, image_path: Path, report: dict[str, object], overlay_image) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    report_path = output_dir / f"{stem}.two_stage.json"
    overlay_path = output_dir / f"{stem}.two_stage.png"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    overlay_image.save(overlay_path)
    return report_path, overlay_path


def main() -> None:
    args = parse_args()
    from vlm_det.infer.config import load_two_stage_inference_config
    from vlm_det.infer.two_stage import load_two_stage_inference_runner

    infer_config = load_two_stage_inference_config(args.config)
    runner = load_two_stage_inference_runner(
        config_path=args.config,
        stage1_checkpoint_path=args.stage1_checkpoint,
        stage2_checkpoint_path=args.stage2_checkpoint,
        device_name=args.device,
        stage1_model_name_or_path=args.stage1_model,
        stage2_model_name_or_path=args.stage2_model,
    )
    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    report = runner.predict(
        image,
        stage1_max_new_tokens=args.stage1_max_new_tokens,
        stage2_max_new_tokens=args.stage2_max_new_tokens,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    final_prediction = report["final_prediction"]
    print("\n[summary]")
    print(format_prediction_summary(final_prediction))
    output_dir = args.output_dir or infer_config.output_dir
    if output_dir is not None:
        overlay = draw_prediction(image, final_prediction)
        report_path, overlay_path = _save_outputs(Path(output_dir), image_path, report, overlay)
        print(f"Saved report to: {report_path}")
        print(f"Saved overlay to: {overlay_path}")


if __name__ == "__main__":
    main()
