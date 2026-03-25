#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from PIL import Image

from vlm_det.infer import draw_prediction, format_prediction_summary
from vlm_det.infer.runner import load_inference_runner

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio demo for ArrowVLM.")
    parser.add_argument("--config", default=None, help="Legacy training config path. Prefer environment-driven inference settings.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory. Falls back to CHECKPOINT_PATH in .env.")
    parser.add_argument("--env-file", default=None, help="Optional path to a .env file for inference/app settings.")
    return parser.parse_args()

def build_demo(runner):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import gradio for the inference app. "
            "If this environment uses a SOCKS proxy, install `httpx[socks]`/`socksio` "
            "or unset the proxy variables before launching `app/demo.py`."
        ) from exc

    def predict(image: Image.Image | None) -> tuple[Image.Image | None, str, str, str]:
        if image is None:
            return None, "No image provided.", "", ""

        raw_text, parse_report = runner.predict(image)
        strict_prediction = parse_report["strict"]["prediction"]
        lenient_prediction = parse_report["lenient"]["prediction"]
        display_prediction = strict_prediction or lenient_prediction

        if display_prediction is not None:
            rendered = draw_prediction(image, display_prediction)
            summary = format_prediction_summary(display_prediction)
        else:
            rendered = image
            summary = "Detected arrows: parse failed"

        summary = "\n".join(
            [
                summary,
                f"Lenient parse ok: {parse_report['lenient']['ok']}",
                f"Strict parse ok: {parse_report['strict']['ok']}",
            ]
        )
        return rendered, summary, raw_text, json.dumps(parse_report, ensure_ascii=False, indent=2)

    with gr.Blocks(title="Arrow VLM Inference") as demo:
        gr.Markdown("## Arrow VLM Inference")
        with gr.Row():
            image_input = gr.Image(type="pil", label="Input Image")
            image_output = gr.Image(type="pil", label="Prediction Overlay")
        summary_output = gr.Textbox(label="Summary", lines=4)
        raw_output = gr.Textbox(label="Raw Model Output", lines=12)
        parse_output = gr.Code(label="Parse Report", language="json")
        run_button = gr.Button("Run Inference")
        run_button.click(
            fn=predict,
            inputs=image_input,
            outputs=[image_output, summary_output, raw_output, parse_output],
        )
    return demo


def main() -> None:
    args = parse_args()
    runner = load_inference_runner(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        env_file=args.env_file,
    )
    demo = build_demo(runner)
    demo.launch(
        server_name=runner.settings.app.host,
        server_port=runner.settings.app.port,
        share=runner.settings.app.share,
    )


if __name__ == "__main__":
    main()
