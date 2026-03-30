#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from PIL import Image

from vlm_structgen.core.infer.config import load_two_stage_inference_config
from vlm_structgen.domains.arrow import (
    draw_prediction,
    format_prediction_summary,
    load_two_stage_inference_runner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio demo for two-stage arrow inference.")
    parser.add_argument("--config", default="configs/infer/infer_two_stage.yaml", help="Two-stage inference config path.")
    parser.add_argument("--stage1-checkpoint", default=None)
    parser.add_argument("--stage2-checkpoint", default=None)
    parser.add_argument("--stage1-model", default=None)
    parser.add_argument("--stage2-model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stage1-max-new-tokens", type=int, default=None)
    parser.add_argument("--stage2-max-new-tokens", type=int, default=None)
    parser.add_argument("--stage2-batch-size", type=int, default=None)
    parser.add_argument(
        "--stage1-mixed-proposals",
        dest="stage1_mixed_proposals",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
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


def _discover_checkpoint_choices(current_checkpoint_path: str | None) -> list[str]:
    discovered: set[str] = set()
    if current_checkpoint_path:
        discovered.add(current_checkpoint_path)
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        for child in sorted(outputs_dir.glob("**/checkpoints/best")):
            if child.is_dir():
                discovered.add(str(child))
        for child in sorted(outputs_dir.glob("**/checkpoints/last")):
            if child.is_dir():
                discovered.add(str(child))
    return sorted(discovered)


def _disable_socks_proxy_env_for_gradio() -> list[str]:
    removed: list[str] = []
    for key in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(key)
        if value and value.lower().startswith("socks"):
            os.environ.pop(key, None)
            removed.append(key)
    return removed


def build_demo(args: argparse.Namespace):
    removed_proxy_keys = _disable_socks_proxy_env_for_gradio()
    try:
        import gradio as gr
    except ImportError as exc:
        proxy_hint = ""
        if removed_proxy_keys:
            proxy_hint = f" Removed SOCKS proxy env: {', '.join(removed_proxy_keys)}."
        raise RuntimeError(
            "Failed to import gradio for the two-stage inference app. "
            "If this environment uses a SOCKS proxy, install `httpx[socks]`/`socksio` "
            "or unset the proxy variables before launching `app/demo_two_stage.py`."
            f"{proxy_hint}"
        ) from exc

    infer_config = load_two_stage_inference_config(args.config)
    initial_runner = None
    if args.stage1_checkpoint:
        initial_runner = load_two_stage_inference_runner(
            config_path=args.config,
            stage1_checkpoint_path=args.stage1_checkpoint,
            stage2_checkpoint_path=args.stage2_checkpoint,
            device_name=args.device,
            stage1_model_name_or_path=args.stage1_model,
            stage2_model_name_or_path=args.stage2_model,
        )
    runner_holder = {
        "runner": initial_runner,
        "stage1_model": args.stage1_model or "",
        "stage1_checkpoint": args.stage1_checkpoint or "",
        "stage2_model": args.stage2_model or "",
        "stage2_checkpoint": args.stage2_checkpoint or "",
    }

    def _gallery_items(image: Image.Image | None) -> list[Image.Image]:
        return [image] if image is not None else []

    def _reload_runner(
        stage1_model: str,
        stage1_checkpoint: str,
        stage2_model: str,
        stage2_checkpoint: str,
    ):
        stage1_model = stage1_model.strip()
        stage1_checkpoint = stage1_checkpoint.strip()
        stage2_model = stage2_model.strip()
        stage2_checkpoint = stage2_checkpoint.strip()
        if not stage1_checkpoint:
            raise ValueError("Stage1 checkpoint path cannot be empty.")
        current = (
            runner_holder["stage1_model"],
            runner_holder["stage1_checkpoint"],
            runner_holder["stage2_model"],
            runner_holder["stage2_checkpoint"],
        )
        requested = (
            stage1_model,
            stage1_checkpoint,
            stage2_model,
            stage2_checkpoint,
        )
        if current == requested:
            return runner_holder["runner"]
        new_runner = load_two_stage_inference_runner(
            config_path=args.config,
            stage1_checkpoint_path=stage1_checkpoint,
            stage2_checkpoint_path=stage2_checkpoint or None,
            device_name=args.device,
            stage1_model_name_or_path=stage1_model,
            stage2_model_name_or_path=stage2_model or None,
        )
        old_runner = runner_holder["runner"]
        runner_holder["runner"] = new_runner
        runner_holder["stage1_model"] = stage1_model
        runner_holder["stage1_checkpoint"] = stage1_checkpoint
        runner_holder["stage2_model"] = stage2_model
        runner_holder["stage2_checkpoint"] = stage2_checkpoint
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
        stage1_mixed = bool(report["stage1_report"]["generation"].get("mixed_proposals_enabled", False))
        stage2_loaded = runner_holder["stage2_checkpoint"].strip() != ""
        stage2_refined = len(report.get("stage2_results", []))
        return (
            f"{'当前运行两阶段推理。' if stage2_loaded else '当前仅运行 Stage1 grounding。'}"
            f" Stage1 mixed proposals: {stage1_mixed}."
            f" Stage1 detected {stage1_count} arrows."
            f" Final output contains {len(instances)} arrows."
            f" Stage2 refined: {stage2_refined}."
            f" Recovered prefixes: stage1={stage1_recovered}."
        )

    def run_inference(
        image: Image.Image | None,
        stage1_model: str,
        stage1_checkpoint: str,
        stage2_model: str,
        stage2_checkpoint: str,
        stage1_max_new_tokens: int,
        stage2_max_new_tokens: int,
        stage2_batch_size: int,
        stage1_mixed_proposals: bool,
    ):
        if image is None:
            raise gr.Error("Please upload an image.")
        runner = _reload_runner(
            stage1_model,
            stage1_checkpoint,
            stage2_model,
            stage2_checkpoint,
        )
        pil_image = image.convert("RGB")
        report = runner.predict(
            pil_image,
            stage1_max_new_tokens=stage1_max_new_tokens,
            stage2_max_new_tokens=stage2_max_new_tokens,
            stage2_batch_size=stage2_batch_size,
            stage1_use_mixed_proposals=stage1_mixed_proposals,
        )
        stage1_prediction = report["stage1_report"]["strict"]["prediction"] or report["stage1_report"]["lenient"]["prediction"]
        final_prediction = report["final_prediction"]
        stage1_overlay = draw_prediction(pil_image, stage1_prediction) if stage1_prediction is not None else pil_image
        final_overlay = draw_prediction(pil_image, final_prediction)
        return (
            _gallery_items(pil_image),
            _gallery_items(stage1_overlay),
            _gallery_items(final_overlay),
            _render_status(report),
            format_prediction_summary(final_prediction),
            json.dumps(final_prediction, ensure_ascii=False, indent=2),
            json.dumps(report["stage1_report"], ensure_ascii=False, indent=2),
            json.dumps(report.get("stage2_results", []), ensure_ascii=False, indent=2),
        )

    stage1_default_max_new_tokens = args.stage1_max_new_tokens or infer_config.stage1.eval.max_new_tokens or 2048
    stage2_default_max_new_tokens = args.stage2_max_new_tokens or infer_config.stage2.eval.max_new_tokens or 256
    stage2_default_batch_size = args.stage2_batch_size or infer_config.stage2.batch_size or 1
    stage1_default_mixed_proposals = (
        bool(infer_config.stage1.tile_size_ratios)
        if args.stage1_mixed_proposals is None
        else bool(args.stage1_mixed_proposals)
    )

    with gr.Blocks(title="ArrowVLM Two-Stage Demo") as demo:
        gr.Markdown("## ArrowVLM Two-Stage Demo\n可单独检查 Stage1 grounding，也可加载 Stage2 做两阶段推理。")
        with gr.Row():
            with gr.Column(scale=1):
                stage1_model = gr.Dropdown(
                    choices=_discover_model_choices(runner_holder["stage1_model"]),
                    value=runner_holder["stage1_model"] or None,
                    label="Stage1 Base Model",
                    allow_custom_value=True,
                )
                stage1_checkpoint = gr.Dropdown(
                    choices=_discover_checkpoint_choices(runner_holder["stage1_checkpoint"]),
                    value=runner_holder["stage1_checkpoint"] or None,
                    label="Stage1 Checkpoint",
                    allow_custom_value=True,
                )
                stage2_model = gr.Dropdown(
                    choices=_discover_model_choices(runner_holder["stage2_model"]),
                    value=runner_holder["stage2_model"] or None,
                    label="Stage2 Base Model",
                    allow_custom_value=True,
                )
                stage2_checkpoint = gr.Dropdown(
                    choices=_discover_checkpoint_choices(runner_holder["stage2_checkpoint"]),
                    value=runner_holder["stage2_checkpoint"] or None,
                    label="Stage2 Checkpoint",
                    allow_custom_value=True,
                )
                stage1_max_new_tokens = gr.Number(
                    value=stage1_default_max_new_tokens,
                    precision=0,
                    label="Stage1 Max New Tokens",
                )
                stage2_max_new_tokens = gr.Number(
                    value=stage2_default_max_new_tokens,
                    precision=0,
                    label="Stage2 Max New Tokens",
                )
                stage2_batch_size = gr.Number(
                    value=stage2_default_batch_size,
                    precision=0,
                    label="Stage2 Batch Size",
                )
                stage1_mixed_proposals = gr.Checkbox(
                    value=stage1_default_mixed_proposals,
                    label="Enable Stage1 Mixed Proposals",
                )
                image_input = gr.Image(type="pil", label="Input Image")
                run_button = gr.Button("Run Two-Stage Inference", variant="primary")
            with gr.Column(scale=2):
                with gr.Row():
                    input_gallery = gr.Gallery(
                        label="Input",
                        columns=1,
                        height=300,
                        object_fit="contain",
                        allow_preview=True,
                        preview=True,
                        buttons=["fullscreen", "download"],
                    )
                    stage1_gallery = gr.Gallery(
                        label="Stage1 Overlay",
                        columns=1,
                        height=300,
                        object_fit="contain",
                        allow_preview=True,
                        preview=True,
                        buttons=["fullscreen", "download"],
                    )
                    stage2_gallery = gr.Gallery(
                        label="Final Overlay",
                        columns=1,
                        height=300,
                        object_fit="contain",
                        allow_preview=True,
                        preview=True,
                        buttons=["fullscreen", "download"],
                    )
                status_text = gr.Textbox(label="Status", lines=3)
                summary_text = gr.Textbox(label="Summary", lines=4)
        with gr.Tabs():
            with gr.Tab("Final Prediction"):
                final_json = gr.Code(language="json")
            with gr.Tab("Stage1 Report"):
                stage1_json = gr.Code(language="json")
            with gr.Tab("Stage2 Report"):
                stage2_json = gr.Code(language="json")

        image_input.change(
            fn=_gallery_items,
            inputs=image_input,
            outputs=input_gallery,
        )
        run_button.click(
            fn=run_inference,
            inputs=[
                image_input,
                stage1_model,
                stage1_checkpoint,
                stage2_model,
                stage2_checkpoint,
                stage1_max_new_tokens,
                stage2_max_new_tokens,
                stage2_batch_size,
                stage1_mixed_proposals,
            ],
            outputs=[
                input_gallery,
                stage1_gallery,
                stage2_gallery,
                status_text,
                summary_text,
                final_json,
                stage1_json,
                stage2_json,
            ],
        )
    return demo, infer_config


def main() -> None:
    args = parse_args()
    demo, infer_config = build_demo(args)
    demo.launch(
        server_name=infer_config.app.host,
        server_port=infer_config.app.port,
        share=infer_config.app.share,
    )


if __name__ == "__main__":
    main()
