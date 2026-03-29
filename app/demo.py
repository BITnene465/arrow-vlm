#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from PIL import Image

from vlm_det.infer import draw_prediction
from vlm_det.infer.config import load_one_stage_inference_config
from vlm_det.infer.runner import load_inference_runner


APP_CSS = """
.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(56, 189, 248, 0.08), transparent 28%),
    radial-gradient(circle at top right, rgba(250, 204, 21, 0.08), transparent 24%),
    linear-gradient(180deg, #fcfcfd 0%, #f5f7fb 100%);
}

.app-shell {
  max-width: 1360px;
  margin: 0 auto;
  padding: 24px 16px 36px;
}

.hero-panel {
  background:
    linear-gradient(135deg, rgba(15, 118, 110, 0.96), rgba(17, 24, 39, 0.9)),
    linear-gradient(135deg, rgba(217, 119, 6, 0.16), transparent);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 28px;
  padding: 30px 32px 24px;
  color: #f7faf8;
  box-shadow: 0 24px 60px rgba(15, 23, 42, 0.16);
  overflow: hidden;
  margin-bottom: 18px;
}

.hero-title {
  margin: 0;
  font-family: "Baskerville", "Palatino Linotype", serif;
  font-size: 2.5rem;
  line-height: 1;
  letter-spacing: -0.03em;
}

.hero-copy {
  margin: 12px 0 0;
  max-width: 780px;
  color: rgba(247, 250, 248, 0.84);
  font-size: 1rem;
  line-height: 1.55;
}

.meta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 20px;
}

.meta-chip {
  display: block;
  padding: 12px 14px;
  min-width: 220px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.10);
  border: 1px solid rgba(255, 255, 255, 0.14);
}

.meta-label {
  display: block;
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: rgba(226, 232, 240, 0.74);
}

.meta-value {
  display: block;
  margin-top: 6px;
  font-size: 0.95rem;
  color: #f8fafc;
  word-break: break-all;
}

.surface-panel {
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 22px;
  background: rgba(255, 255, 255, 0.92);
  box-shadow: 0 16px 38px rgba(15, 23, 42, 0.08);
}

.surface-pad {
  padding: 18px 18px 10px;
}

.section-title {
  margin: 0 0 8px;
  font-family: "Baskerville", "Palatino Linotype", serif;
  font-size: 1.2rem;
  color: #0f172a;
}

.section-copy {
  margin: 0;
  color: #475569;
  font-size: 0.94rem;
  line-height: 1.45;
}

.status-board {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.status-card {
  padding: 14px 16px;
  border-radius: 18px;
  background: #ffffff;
  border: 1px solid rgba(15, 23, 42, 0.10);
}

.status-label {
  display: block;
  color: #64748b;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.status-value {
  display: block;
  margin-top: 6px;
  font-size: 1.25rem;
  font-weight: 700;
  color: #0f172a;
}

.badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 78px;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 700;
}

.badge.ok {
  color: #166534;
  background: #dcfce7;
}

.badge.fail {
  color: #991b1b;
  background: #fee2e2;
}

.error-strip {
  margin-top: 14px;
  padding: 12px 14px;
  border-radius: 16px;
  background: #fff1f2;
  border: 1px solid #fecdd3;
  color: #9f1239;
  font-size: 0.92rem;
  line-height: 1.45;
}

.compact-note {
  color: #475569;
  font-size: 0.88rem;
}

#run-button button {
  min-height: 54px;
  font-size: 1rem;
  font-weight: 700;
  border-radius: 16px;
  background: linear-gradient(135deg, #0f766e, #115e59);
  color: #ffffff;
  border: none;
}

#clear-button button {
  min-height: 54px;
  border-radius: 16px;
  background: #ffffff;
  color: #334155;
  border: 1px solid rgba(15, 23, 42, 0.08);
}

#input-image, #output-image, #raw-output, #parse-output, #generation-panel, #status-panel {
  border-radius: 24px;
}

#status-panel {
  min-height: 210px;
}

.gradio-container .tabs {
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 22px;
  background: rgba(255, 255, 255, 0.92);
  box-shadow: 0 16px 38px rgba(15, 23, 42, 0.06);
}

@media (max-width: 1100px) {
  .status-board {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .hero-panel {
    padding: 24px 20px 20px;
  }

  .hero-title {
    font-size: 1.95rem;
  }

  .status-board {
    grid-template-columns: 1fr;
  }
}
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio demo for ArrowVLM.")
    parser.add_argument("--config", default="configs/infer/infer_one_stage.yaml", help="Inference config path.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory. Falls back to CHECKPOINT_PATH in .env.")
    parser.add_argument("--env-file", default=None, help="Optional path to a .env file when checkpoint falls back to CHECKPOINT_PATH.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override inference max_new_tokens for this app session.")
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


def _render_status_panel(
    parse_report: dict[str, object],
    *,
    max_new_tokens: int,
) -> str:
    generation = parse_report["generation"]
    strict = parse_report["strict"]
    lenient = parse_report["lenient"]
    display_prediction = strict["prediction"] or lenient["prediction"]
    instances = display_prediction.get("instances", []) if display_prediction is not None else []
    point_count = sum(len(instance.get("keypoints", [])) for instance in instances)

    lenient_badge = '<span class="badge ok">PASS</span>' if lenient["ok"] else '<span class="badge fail">FAIL</span>'
    strict_badge = '<span class="badge ok">PASS</span>' if strict["ok"] else '<span class="badge fail">FAIL</span>'

    error_message = strict["error"] or lenient["error"]
    error_html = ""
    if error_message:
        error_html = f'<div class="error-strip"><strong>Parse issue:</strong> {error_message}</div>'

    return f"""
    <div class="status-board">
      <div class="status-card">
        <span class="status-label">Detected Arrows</span>
        <span class="status-value">{len(instances)}</span>
      </div>
      <div class="status-card">
        <span class="status-label">Total Keypoints</span>
        <span class="status-value">{point_count}</span>
      </div>
      <div class="status-card">
        <span class="status-label">Lenient Parse</span>
        <span class="status-value">{lenient_badge}</span>
      </div>
      <div class="status-card">
        <span class="status-label">Strict Parse</span>
        <span class="status-value">{strict_badge}</span>
      </div>
      <div class="status-card">
        <span class="status-label">Recovered Prefix</span>
        <span class="status-value">{'<span class="badge ok">YES</span>' if lenient["recovered_prefix"] else '<span class="badge fail">NO</span>'}</span>
      </div>
      <div class="status-card">
        <span class="status-label">Generated Tokens</span>
        <span class="status-value">{generation["generated_tokens"]}</span>
      </div>
      <div class="status-card">
        <span class="status-label">Stop Reason</span>
        <span class="status-value">{generation["stop_reason"]}</span>
      </div>
    </div>
    <div class="compact-note" style="margin-top: 14px;">
      Current generation budget: <strong>{max_new_tokens}</strong> new tokens
      &nbsp;|&nbsp; hit max: <strong>{generation["hit_max_new_tokens"]}</strong>
    </div>
    {error_html}
"""


def _disable_socks_proxy_env_for_gradio() -> list[str]:
    removed: list[str] = []
    for key in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(key)
        if value and value.lower().startswith("socks"):
            os.environ.pop(key, None)
            removed.append(key)
    return removed

def build_demo(
    runner,
    *,
    infer_config,
    default_max_new_tokens: int | None = None,
    runner_factory=None,
):
    removed_proxy_keys = _disable_socks_proxy_env_for_gradio()
    try:
        import gradio as gr
    except ImportError as exc:
        proxy_hint = ""
        if removed_proxy_keys:
            proxy_hint = f" Removed SOCKS proxy env: {', '.join(removed_proxy_keys)}."
        raise RuntimeError(
            "Failed to import gradio for the inference app. "
            "If this environment uses a SOCKS proxy, install `httpx[socks]`/`socksio` "
            "or unset the proxy variables before launching `app/demo.py`."
            f"{proxy_hint}"
        ) from exc

    effective_default_max_new_tokens = default_max_new_tokens or infer_config.eval.max_new_tokens or 2048
    model_choices = _discover_model_choices(runner.config.model.model_name_or_path if runner is not None else None)
    checkpoint_choices = _discover_checkpoint_choices(runner.settings.checkpoint_path if runner is not None else None)
    runner_holder = {
        "runner": runner,
        "model_name_or_path": runner.config.model.model_name_or_path if runner is not None else "",
        "checkpoint_path": runner.settings.checkpoint_path if runner is not None else "",
    }

    def _gallery_items(image: Image.Image | None) -> list[Image.Image]:
        return [image] if image is not None else []

    def _get_runner(model_name_or_path: str, checkpoint_path: str):
        selected_model = model_name_or_path.strip()
        selected_checkpoint = checkpoint_path.strip()
        if not selected_model:
            raise ValueError("Model path cannot be empty.")
        if not selected_checkpoint:
            raise ValueError("Checkpoint path cannot be empty.")
        current_model = runner_holder["model_name_or_path"]
        current_checkpoint = runner_holder["checkpoint_path"]
        if runner_holder["runner"] is not None and selected_model == current_model and selected_checkpoint == current_checkpoint:
            return runner_holder["runner"]
        if runner_factory is None:
            raise RuntimeError("Runner factory is unavailable for model switching.")

        previous_runner = runner_holder["runner"]
        runner_holder["runner"] = None
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            next_runner = runner_factory(selected_model, selected_checkpoint)
        except Exception:
            runner_holder["runner"] = previous_runner
            raise

        runner_holder["runner"] = next_runner
        runner_holder["model_name_or_path"] = selected_model
        runner_holder["checkpoint_path"] = selected_checkpoint
        del previous_runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return next_runner

    def predict(
        image: Image.Image | None,
        max_new_tokens: int,
        model_name_or_path: str,
        checkpoint_path: str,
    ) -> tuple[list[Image.Image], str, str, str]:
        if image is None:
            return [], "<div class='error-strip'>No image provided.</div>", "", ""

        try:
            active_runner = _get_runner(model_name_or_path, checkpoint_path)
            raw_text, parse_report = active_runner.predict(image, max_new_tokens=max_new_tokens)
        except Exception as exc:  # noqa: BLE001
            return [], f"<div class='error-strip'><strong>Inference failed:</strong> {exc}</div>", "", ""
        strict_prediction = parse_report["strict"]["prediction"]
        lenient_prediction = parse_report["lenient"]["prediction"]
        display_prediction = strict_prediction or lenient_prediction

        if display_prediction is not None:
            rendered = draw_prediction(image, display_prediction)
        else:
            rendered = image

        status_html = _render_status_panel(parse_report, max_new_tokens=max_new_tokens)
        return [rendered], status_html, raw_text, json.dumps(parse_report, ensure_ascii=False, indent=2)

    meta_html = f"""
    <div class="hero-panel">
      <h1 class="hero-title">Arrow VLM Workbench</h1>
      <p class="hero-copy">
        Upload a figure, inspect the overlay, compare strict versus lenient parsing,
        and tune generation behavior without leaving the page.
      </p>
    </div>
    """

    with gr.Blocks(title="Arrow VLM Inference", css=APP_CSS) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(meta_html)
            with gr.Row(equal_height=True):
                with gr.Column(scale=4, elem_id="generation-panel", elem_classes=["surface-panel"]):
                    gr.HTML(
                        """
                        <div class="surface-pad">
                          <h3 class="section-title">Input & Controls</h3>
                          <p class="section-copy">
                            Tune generation budget here, then run one-shot structured inference.
                          </p>
                        </div>
                        """
                    )
                    image_input = gr.Image(
                        type="pil",
                        label="Upload Figure",
                        elem_id="input-upload",
                        sources=["upload", "clipboard"],
                    )
                    input_preview = gr.Gallery(
                        label="Input Preview",
                        elem_id="input-image",
                        columns=1,
                        rows=1,
                        object_fit="contain",
                        allow_preview=True,
                        preview=True,
                        buttons=["fullscreen", "download"],
                        height=360,
                    )
                    max_new_tokens_input = gr.Slider(
                        label="Max New Tokens",
                        minimum=256,
                        maximum=16384,
                        step=256,
                        value=max(256, int(effective_default_max_new_tokens)),
                    )
                    model_name_or_path_input = gr.Dropdown(
                        label="Base Model",
                        choices=model_choices,
                        value=runner.config.model.model_name_or_path if runner is not None else None,
                        allow_custom_value=True,
                        info="Checkpoint and base model size must match.",
                    )
                    checkpoint_path_input = gr.Dropdown(
                        label="Checkpoint",
                        choices=checkpoint_choices,
                        value=runner.settings.checkpoint_path if runner is not None else None,
                        allow_custom_value=True,
                        info="Choose a checkpoint directory such as checkpoints/best.",
                    )
                    with gr.Row():
                        run_button = gr.Button("Run Inference", elem_id="run-button")
                with gr.Column(scale=6):
                    image_output = gr.Gallery(
                        label="Prediction Overlay",
                        elem_id="output-image",
                        columns=1,
                        rows=1,
                        object_fit="contain",
                        allow_preview=True,
                        preview=True,
                        buttons=["fullscreen", "download"],
                        height=520,
                    )
                    with gr.Group(elem_id="status-panel", elem_classes=["surface-panel"]):
                        gr.HTML(
                            """
                            <div class="surface-pad">
                              <h3 class="section-title">Run Status</h3>
                              <p class="section-copy">
                                Compact summary of parsed results and generation budget.
                              </p>
                            </div>
                            """
                        )
                        status_output = gr.HTML(
                            "<div class='compact-note' style='padding: 0 18px 18px;'>Run an image to populate results.</div>"
                        )
            with gr.Tabs():
                with gr.Tab("Raw Output"):
                    raw_output = gr.Textbox(label="Raw Model Output", lines=14, elem_id="raw-output")
                with gr.Tab("Parse Report"):
                    parse_output = gr.Code(label="Parse Report", language="json", elem_id="parse-output")
            clear_button = gr.ClearButton(
                [
                    image_input,
                    input_preview,
                    image_output,
                    status_output,
                    raw_output,
                    parse_output,
                    model_name_or_path_input,
                    checkpoint_path_input,
                ],
                value="Clear",
                elem_id="clear-button",
            )

        image_input.change(
            fn=_gallery_items,
            inputs=image_input,
            outputs=input_preview,
        )
        run_button.click(
            fn=predict,
            inputs=[image_input, max_new_tokens_input, model_name_or_path_input, checkpoint_path_input],
            outputs=[image_output, status_output, raw_output, parse_output],
        )
    return demo


def main() -> None:
    args = parse_args()
    infer_config = load_one_stage_inference_config(args.config)
    def _runner_factory(
        model_name_or_path: str | None = None,
        checkpoint_path: str | None = None,
    ):
        return load_inference_runner(
            checkpoint_path=checkpoint_path or args.checkpoint,
            config_path=args.config,
            env_file=args.env_file,
            model_name_or_path=model_name_or_path,
        )

    runner = _runner_factory() if args.checkpoint else None
    demo = build_demo(
        runner,
        infer_config=infer_config,
        default_max_new_tokens=args.max_new_tokens,
        runner_factory=_runner_factory,
    )
    demo.launch(
        server_name=infer_config.app.host,
        server_port=infer_config.app.port,
        share=infer_config.app.share,
    )


if __name__ == "__main__":
    main()
