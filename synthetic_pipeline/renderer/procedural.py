from __future__ import annotations

import random
from typing import Any

from PIL import Image

from synthetic_pipeline.renderer.common import (
    arrow_bbox,
    default_arrow_style,
    degrade_image,
    make_background,
    sample_occluders,
    scene_style_config,
)
from synthetic_pipeline.renderer.svg import SvgCanvas, draw_distractors_svg, draw_occluders_svg
from synthetic_pipeline.scene_sampler import clamp
from synthetic_pipeline.schema import SyntheticSample


class ProceduralRenderer:
    name = "procedural"

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def render(self, sample: SyntheticSample, rng: random.Random, asset_bank=None) -> tuple[Image.Image, dict[str, Any]]:
        scene = sample.scene
        style_cfg = scene_style_config(self.cfg["style"], scene.scene_mode)
        image = make_background(scene.image_width, scene.image_height, style_cfg, rng)
        canvas = SvgCanvas(scene.image_width, scene.image_height)
        draw_distractors_svg(canvas, scene.image_width, scene.image_height, style_cfg, rng)
        arrow_style = default_arrow_style(scene.image_width, scene.image_height)

        for instance in sample.instances:
            self._render_instance(canvas, scene.image_width, scene.image_height, instance, arrow_style)

        occluders = sample_occluders(scene.image_width, scene.image_height, style_cfg, rng)
        draw_occluders_svg(canvas, occluders, rng)
        image = self._composite_overlay(image, canvas.rasterize())
        image = degrade_image(image, style_cfg, rng)
        return image, {
            "renderer": self.name,
            "num_occluders": len(occluders),
            "asset_backed_background": False,
            "vector_backend": "svg",
        }

    @staticmethod
    def _composite_overlay(base_image: Image.Image, overlay: Image.Image) -> Image.Image:
        composed = base_image.convert("RGBA")
        composed.alpha_composite(overlay)
        return composed.convert("RGB")

    @staticmethod
    def _render_instance(
        canvas: SvgCanvas,
        width: int,
        height: int,
        instance,
        style: dict[str, Any],
    ) -> None:
        points = [(float(point[0]), float(point[1])) for point in instance.keypoints]
        canvas.add_arrow(
            points=points,
            line_width=int(style["line_width"]),
            head_len=int(style["head_len"]),
            head_width=int(style["head_width"]),
            color=tuple(style["color"]),
            double_headed=instance.label == "double_arrow",
        )
        render_box = arrow_bbox(instance.keypoints, int(style["line_width"]), int(style["head_len"]), int(style["head_width"]))
        instance.render_bbox = [
            round(clamp(render_box[0], 0, width - 1), 2),
            round(clamp(render_box[1], 0, height - 1), 2),
            round(clamp(render_box[2], 0, width - 1), 2),
            round(clamp(render_box[3], 0, height - 1), 2),
        ]
