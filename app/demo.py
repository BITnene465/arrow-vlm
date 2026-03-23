#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from PIL import Image, ImageDraw

from vlm_det.config import ExperimentRuntimeConfig, load_config
from vlm_det.modeling.builder import BuildArtifacts, build_model_tokenizer_processor
from vlm_det.protocol.codec import ArrowCodec
from vlm_det.utils.checkpoint import load_training_checkpoint
from vlm_det.utils.distributed import unwrap_model


PALETTE = [
    "#e63946",
    "#1d3557",
    "#2a9d8f",
    "#f4a261",
    "#6a4c93",
    "#d62828",
    "#0077b6",
    "#588157",
]


@dataclass
class DemoArtifacts:
    config: ExperimentRuntimeConfig
    model_artifacts: BuildArtifacts
    codec: ArrowCodec
    device: torch.device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio demo for ArrowVLM.")
    parser.add_argument("--config", default="configs/train_lora.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def _load_demo_artifacts(config_path: str, checkpoint_path: str) -> DemoArtifacts:
    config = load_config(config_path)
    model_artifacts = build_model_tokenizer_processor(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_artifacts.model = model_artifacts.model.to(device)
    load_training_checkpoint(
        checkpoint_dir=checkpoint_path,
        model=model_artifacts.model,
        tokenizer=model_artifacts.tokenizer,
        processor=model_artifacts.processor,
        strict=True,
        resume_training_state=False,
    )
    unwrap_model(model_artifacts.model).eval()
    codec = ArrowCodec(num_bins=config.tokenizer.num_bins)
    return DemoArtifacts(
        config=config,
        model_artifacts=model_artifacts,
        codec=codec,
        device=device,
    )


def _build_prompt(artifacts: DemoArtifacts) -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": artifacts.config.prompt.system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "image"}],
        },
    ]
    template_owner = (
        artifacts.model_artifacts.processor
        if hasattr(artifacts.model_artifacts.processor, "apply_chat_template")
        else artifacts.model_artifacts.tokenizer
    )
    return template_owner.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _prepare_inputs(artifacts: DemoArtifacts, image: Image.Image) -> tuple[dict[str, torch.Tensor], int]:
    prompt = _build_prompt(artifacts)
    processor_kwargs: dict[str, Any] = {
        "text": [prompt],
        "images": [image],
        "return_tensors": "pt",
    }
    if artifacts.config.model.min_pixels is not None:
        processor_kwargs["min_pixels"] = artifacts.config.model.min_pixels
    if artifacts.config.model.max_pixels is not None:
        processor_kwargs["max_pixels"] = artifacts.config.model.max_pixels
    batch = artifacts.model_artifacts.processor(**processor_kwargs)
    prompt_length = int(batch["attention_mask"].sum(dim=1).item())
    model_inputs = {
        key: value.to(artifacts.device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }
    return model_inputs, prompt_length


def _run_generation(artifacts: DemoArtifacts, image: Image.Image) -> tuple[str, dict[str, Any]]:
    width, height = image.size
    model_inputs, prompt_length = _prepare_inputs(artifacts, image)
    raw_model = unwrap_model(artifacts.model_artifacts.model)
    with torch.inference_mode():
        output_ids = raw_model.generate(
            **model_inputs,
            max_new_tokens=artifacts.config.eval.max_new_tokens,
            num_beams=artifacts.config.eval.num_beams,
            do_sample=artifacts.config.eval.do_sample,
            use_cache=artifacts.config.eval.use_cache,
            pad_token_id=artifacts.model_artifacts.tokenizer.pad_token_id,
        )
    continuation = output_ids[0, prompt_length:]
    decoded = artifacts.model_artifacts.tokenizer.decode(continuation, skip_special_tokens=False)
    prediction = artifacts.codec.decode(decoded, image_width=width, image_height=height)
    return decoded, prediction


def _draw_prediction(image: Image.Image, prediction: dict[str, Any]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for index, instance in enumerate(prediction.get("instances", [])):
        color = PALETTE[index % len(PALETTE)]
        bbox = [float(value) for value in instance.get("bbox", [])]
        if len(bbox) == 4:
            draw.rectangle(bbox, outline=color, width=3)
        keypoints = instance.get("keypoints", [])
        xy_points = [(float(point[0]), float(point[1])) for point in keypoints]
        if len(xy_points) >= 2:
            draw.line(xy_points, fill=color, width=3)
        for point_index, point in enumerate(keypoints):
            x = float(point[0])
            y = float(point[1])
            visibility = str(point[2])
            radius = 5 if visibility == "visible" else 4
            fill = color if visibility == "visible" else "#ffffff"
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=color, width=2)
            if point_index == 0:
                draw.text((x + 6, y - 12), "S", fill=color)
            elif point_index == len(keypoints) - 1:
                draw.text((x + 6, y - 12), "H", fill=color)
        if len(bbox) == 4:
            draw.text((bbox[0] + 4, bbox[1] + 4), f"arrow {index + 1}", fill=color)
    return canvas


def _format_summary(prediction: dict[str, Any]) -> str:
    instances = prediction.get("instances", [])
    point_count = sum(len(instance.get("keypoints", [])) for instance in instances)
    return "\n".join(
        [
            f"Detected arrows: {len(instances)}",
            f"Total keypoints: {point_count}",
        ]
    )


def build_demo(artifacts: DemoArtifacts) -> gr.Blocks:
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
            raw_text, prediction = _run_generation(artifacts, pil_image)
            overlay = _draw_prediction(pil_image, prediction)
            summary = _format_summary(prediction)
            formatted_json = json.dumps(prediction, ensure_ascii=False, indent=2)
            return overlay, summary, formatted_json, raw_text
        except Exception as exc:  # noqa: BLE001
            return pil_image, f"Inference failed: {exc}", "{}", ""

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)
        gr.Markdown(
            f"**Config:** `{artifacts.config.experiment.name}`  \n"
            f"**Model source:** `{artifacts.config.model.model_name_or_path}`"
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
    artifacts = _load_demo_artifacts(args.config, args.checkpoint)
    demo = build_demo(artifacts)
    demo.queue(api_open=False).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
