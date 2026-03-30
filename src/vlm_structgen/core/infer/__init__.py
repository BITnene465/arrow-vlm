from __future__ import annotations

from typing import Any

__all__ = [
    "InferenceRunner",
    "load_inference_runner",
]


def __getattr__(name: str) -> Any:
    if name in {"InferenceRunner", "load_inference_runner"}:
        from vlm_structgen.core.infer.runner import InferenceRunner, load_inference_runner

        return {
            "InferenceRunner": InferenceRunner,
            "load_inference_runner": load_inference_runner,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
