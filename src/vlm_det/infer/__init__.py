from __future__ import annotations

from typing import Any

__all__ = [
    "ArrowInferenceRunner",
    "load_inference_runner",
    "draw_prediction",
    "format_prediction_summary",
]


def __getattr__(name: str) -> Any:
    if name in {"ArrowInferenceRunner", "load_inference_runner"}:
        from vlm_det.infer.runner import ArrowInferenceRunner, load_inference_runner

        return {
            "ArrowInferenceRunner": ArrowInferenceRunner,
            "load_inference_runner": load_inference_runner,
        }[name]
    if name in {"draw_prediction", "format_prediction_summary"}:
        from vlm_det.infer.visualize import draw_prediction, format_prediction_summary

        return {
            "draw_prediction": draw_prediction,
            "format_prediction_summary": format_prediction_summary,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
