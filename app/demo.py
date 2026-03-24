#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gradio as gr
from PIL import Image

from vlm_det.infer.runner import ArrowInferenceRunner, load_inference_runner
from vlm_det.infer.visualize import draw_prediction, format_prediction_summary

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio demo for ArrowVLM.")
    parser.add_argument("--config", default="configs/train_lora.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()

def build_demo(runner: ArrowInferenceRunner) -> gr.Blocks:
    title = "ArrowVLM Demo"
    description = (
        "Upload a figure image and run structured arrow detection with the current checkpoint. "
        "The demo returns an overlay image, decoded JSON, and the raw protocol sequence."
    )

    def predict(image: Image.Image) -> tuple[Image.Image | None, str, str, str]:
        if image is None:
            return None, "Please upload an image.", "{}", ""
        pil_image = image.convert("RGB")
        try:
            raw_text, prediction = runner.predict(pil_image)
            overlay = draw_prediction(pil_image, prediction)
            summary = format_prediction_summary(prediction)
            formatted_json = json.dumps(prediction, ensure_ascii=False, indent=2)
            return overlay, summary, formatted_json, raw_text
        except Exception as exc:  # noqa: BLE001
            return pil_image, f"Inference failed: {exc}", "{}", ""

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)
        gr.Markdown(
            f"**Config:** `{runner.config.experiment.name}`  \n"
            f"**Model source:** `{runner.config.model.model_name_or_path}`"
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Input figure")
                run_button = gr.Button("Run Inference", variant="primary")
            with gr.Column(scale=1):
                image_output = gr.Image(type="pil", label="Visualization")
                summary_output = gr.Textbox(label="Summary", lines=3)
        with gr.Row():
            json_output = gr.Code(label="Decoded JSON", language="json")
            raw_output = gr.Textbox(label="Raw protocol output", lines=20)
        run_button.click(
            fn=predict,
            inputs=[image_input],
            outputs=[image_output, summary_output, json_output, raw_output],
        )
    return demo


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    runner = load_inference_runner(args.config, args.checkpoint)
    demo = build_demo(runner)
    demo.queue(api_open=False).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
