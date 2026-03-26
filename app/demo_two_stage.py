#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from PIL import Image

from vlm_det.infer.visualize import draw_prediction, format_prediction_summary
from vlm_det.infer.two_stage import load_two_stage_inference_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio demo for two-stage ArrowVLM inference.")
    parser.add_argument("--stage1-config", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage2-config", required=True)
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--stage1-model", default=None)
    parser.add_argument("--stage2-model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--padding-ratio", type=float, default=0.5)
    parser.add_argument("--stage1-max-new-tokens", type=int, default=None)
    parser.add_argument("--stage2-max-new-tokens", type=int, default=None)
    return parser.parse_args()


def _discover_model_choices(current_model_name_or_path: str | None) -> list[str]:
    discovered: set[str] = set()
    if current_model_name_or_path:
        discovered.add(current_model_name_or_path)
    models_dir = Path("models")
    if models_dir.exists():
        for child in sorted(models_dir.iterdir()):
            if child.is_dir():
                discovered.add(str(child))
    return sorted(discovered)


def _discover_checkpoint_choices(current_checkpoint_path: str) -> list[str]:
    discovered: set[str] = {current_checkpoint_path}
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        for child in sorted(outputs_dir.glob("**/checkpoints/best")):
            if child.is_dir():
                discovered.add(str(child))
        for child in sorted(outputs_dir.glob("**/checkpoints/last")):
            if child.is_dir():
                discovered.add(str(child))
    return sorted(discovered)


def build_demo(args: argparse.Namespace):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import gradio for the two-stage inference app. "
            "Install gradio and retry."
        ) from exc

    initial_runner = load_two_stage_inference_runner(
        stage1_config_path=args.stage1_config,
        stage1_checkpoint_path=args.stage1_checkpoint,
        stage2_config_path=args.stage2_config,
        stage2_checkpoint_path=args.stage2_checkpoint,
        device_name=args.device,
        stage1_model_name_or_path=args.stage1_model,
        stage2_model_name_or_path=args.stage2_model,
        padding_ratio=args.padding_ratio,
    )
    runner_holder = {
        "runner": initial_runner,
        "stage1_model": args.stage1_model or initial_runner.stage1_runner.config.model.model_name_or_path,
        "stage2_model": args.stage2_model or initial_runner.stage2_runner.config.model.model_name_or_path,
        "stage1_checkpoint": args.stage1_checkpoint,
        "stage2_checkpoint": args.stage2_checkpoint,
        "padding_ratio": args.padding_ratio,
    }

    def _gallery_items(image: Image.Image | None) -> list[Image.Image]:
        return [image] if image is not None else []

    def _reload_runner(
        stage1_model: str,
        stage1_checkpoint: str,
        stage2_model: str,
        stage2_checkpoint: str,
        padding_ratio: float,
    ):
        stage1_model = stage1_model.strip()
        stage1_checkpoint = stage1_checkpoint.strip()
        stage2_model = stage2_model.strip()
        stage2_checkpoint = stage2_checkpoint.strip()
        if not stage1_checkpoint or not stage2_checkpoint:
            raise ValueError("Checkpoint path cannot be empty.")
        current = (
            runner_holder["stage1_model"],
            runner_holder["stage1_checkpoint"],
            runner_holder["stage2_model"],
            runner_holder["stage2_checkpoint"],
            runner_holder["padding_ratio"],
        )
        requested = (
            stage1_model,
            stage1_checkpoint,
            stage2_model,
            stage2_checkpoint,
            float(padding_ratio),
        )
        if current == requested:
            return runner_holder["runner"]
        new_runner = load_two_stage_inference_runner(
            stage1_config_path=args.stage1_config,
            stage1_checkpoint_path=stage1_checkpoint,
            stage2_config_path=args.stage2_config,
            stage2_checkpoint_path=stage2_checkpoint,
            device_name=args.device,
            stage1_model_name_or_path=stage1_model,
            stage2_model_name_or_path=stage2_model,
            padding_ratio=float(padding_ratio),
        )
        old_runner = runner_holder["runner"]
        runner_holder["runner"] = new_runner
        runner_holder["stage1_model"] = stage1_model
        runner_holder["stage1_checkpoint"] = stage1_checkpoint
        runner_holder["stage2_model"] = stage2_model
        runner_holder["stage2_checkpoint"] = stage2_checkpoint
        runner_holder["padding_ratio"] = float(padding_ratio)
        try:
            del old_runner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return new_runner

    def _render_status(report: dict[str, object]) -> str:
        final_prediction = report["final_prediction"]
        instances = final_prediction.get("instances", [])
        stage1_prediction = report["stage1_report"]["strict"]["prediction"] or report["stage1_report"]["lenient"]["prediction"]
        stage1_count = len(stage1_prediction.get("instances", [])) if stage1_prediction else 0
        stage1_recovered = bool(report["stage1_report"]["lenient"].get("recovered_prefix", False))
        stage2_recovered = sum(
            1 for item in report["stage2_results"] if item["report"]["lenient"].get("recovered_prefix", False)
        )
        return (
            f"Stage1 detected {stage1_count} arrows. "
            f"Final output contains {len(instances)} arrows. "
            f"Stage2 refined {len(report['stage2_results'])} crops. "
            f"Recovered prefixes: stage1={stage1_recovered}, stage2={stage2_recovered}."
        )

    def run_inference(
        image: Image.Image | None,
        stage1_model: str,
        stage1_checkpoint: str,
        stage2_model: str,
        stage2_checkpoint: str,
        padding_ratio: float,
        stage1_max_new_tokens: int,
        stage2_max_new_tokens: int,
    ):
        if image is None:
            raise gr.Error("Please upload an image.")
        runner = _reload_runner(
            stage1_model,
            stage1_checkpoint,
            stage2_model,
            stage2_checkpoint,
            padding_ratio,
        )
        pil_image = image.convert("RGB")
        report = runner.predict(
            pil_image,
            stage1_max_new_tokens=stage1_max_new_tokens,
            stage2_max_new_tokens=stage2_max_new_tokens,
        )
        final_prediction = report["final_prediction"]
        overlay = draw_prediction(pil_image, final_prediction)
        return (
            _gallery_items(pil_image),
            _gallery_items(overlay),
            _render_status(report),
            format_prediction_summary(final_prediction),
            json.dumps(final_prediction, ensure_ascii=False, indent=2),
            json.dumps(report["stage1_report"], ensure_ascii=False, indent=2),
            json.dumps(report["stage2_results"], ensure_ascii=False, indent=2),
        )

    with gr.Blocks(title="ArrowVLM Two-Stage Demo") as demo:
        gr.Markdown("## ArrowVLM Two-Stage Demo\n上传原图，Stage 1 先检测箭头和两端点，Stage 2 再对每个 crop 补完整点列。")
        with gr.Row():
            with gr.Column(scale=1):
                stage1_model = gr.Dropdown(
                    choices=_discover_model_choices(runner_holder["stage1_model"]),
                    value=runner_holder["stage1_model"],
                    label="Stage1 Base Model",
                    allow_custom_value=True,
                )
                stage1_checkpoint = gr.Dropdown(
                    choices=_discover_checkpoint_choices(runner_holder["stage1_checkpoint"]),
                    value=runner_holder["stage1_checkpoint"],
                    label="Stage1 Checkpoint",
                    allow_custom_value=True,
                )
                stage2_model = gr.Dropdown(
                    choices=_discover_model_choices(runner_holder["stage2_model"]),
                    value=runner_holder["stage2_model"],
                    label="Stage2 Base Model",
                    allow_custom_value=True,
                )
                stage2_checkpoint = gr.Dropdown(
                    choices=_discover_checkpoint_choices(runner_holder["stage2_checkpoint"]),
                    value=runner_holder["stage2_checkpoint"],
                    label="Stage2 Checkpoint",
                    allow_custom_value=True,
                )
                padding_ratio = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=args.padding_ratio, label="Crop Padding Ratio")
                stage1_max_new_tokens = gr.Number(
                    value=args.stage1_max_new_tokens or initial_runner.stage1_runner.config.eval.max_new_tokens,
                    precision=0,
                    label="Stage1 Max New Tokens",
                )
                stage2_max_new_tokens = gr.Number(
                    value=args.stage2_max_new_tokens or initial_runner.stage2_runner.config.eval.max_new_tokens,
                    precision=0,
                    label="Stage2 Max New Tokens",
                )
                image_input = gr.Image(type="pil", label="Input Image")
                run_button = gr.Button("Run Two-Stage Inference", variant="primary")
            with gr.Column(scale=1):
                input_gallery = gr.Gallery(label="Input Preview", columns=1, height=320, object_fit="contain")
                output_gallery = gr.Gallery(label="Final Overlay", columns=1, height=320, object_fit="contain")
                status_text = gr.Textbox(label="Status", lines=2)
                summary_text = gr.Textbox(label="Summary", lines=4)
        with gr.Tabs():
            with gr.Tab("Final Prediction"):
                final_json = gr.Code(language="json")
            with gr.Tab("Stage1 Report"):
                stage1_json = gr.Code(language="json")
            with gr.Tab("Stage2 Reports"):
                stage2_json = gr.Code(language="json")

        run_button.click(
            fn=run_inference,
            inputs=[
                image_input,
                stage1_model,
                stage1_checkpoint,
                stage2_model,
                stage2_checkpoint,
                padding_ratio,
                stage1_max_new_tokens,
                stage2_max_new_tokens,
            ],
            outputs=[
                input_gallery,
                output_gallery,
                status_text,
                summary_text,
                final_json,
                stage1_json,
                stage2_json,
            ],
        )
    return demo


def main() -> None:
    args = parse_args()
    demo = build_demo(args)
    demo.launch()


if __name__ == "__main__":
    main()
