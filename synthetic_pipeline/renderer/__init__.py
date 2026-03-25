from __future__ import annotations

from synthetic_pipeline.renderer.hybrid import HybridRenderer
from synthetic_pipeline.renderer.procedural import ProceduralRenderer


def build_renderer(cfg: dict):
    renderer_name = cfg["renderer"]["name"]
    if renderer_name == "procedural":
        return ProceduralRenderer(cfg)
    if renderer_name == "hybrid":
        return HybridRenderer(cfg)
    raise ValueError(f"Unsupported renderer: {renderer_name!r}")
