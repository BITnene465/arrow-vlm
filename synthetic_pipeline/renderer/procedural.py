from __future__ import annotations

import random
from typing import Any

from PIL import Image, ImageDraw

from synthetic_pipeline.renderer.common import (
    arrow_bbox,
    degrade_image,
    draw_arrow,
    draw_distractors,
    draw_occluders,
    make_background,
    sample_arrow_style,
    sample_occluders,
    scene_style_config,
)
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
        draw = ImageDraw.Draw(image)
        draw_distractors(draw, scene.image_width, scene.image_height, style_cfg, rng)

        style_profiles: dict[str, int] = {}
        for instance in sample.instances:
            style = sample_arrow_style(style_cfg, rng)
            style["geometry_mode"] = instance.meta.get("geometry_mode", "polyline")
            self._render_instance(draw, scene.image_width, scene.image_height, instance, style, rng)
            style_profiles[style["style_profile"]] = style_profiles.get(style["style_profile"], 0) + 1

        occluders = sample_occluders(scene.image_width, scene.image_height, style_cfg, rng)
        draw_occluders(draw, occluders, rng)
        image = degrade_image(image, style_cfg, rng)
        return image, {
            "renderer": self.name,
            "num_occluders": len(occluders),
            "style_profiles": style_profiles,
            "asset_backed_background": False,
        }

    @staticmethod
    def _render_instance(
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        instance,
        style: dict[str, Any],
        rng: random.Random,
    ) -> None:
        points = [(float(point[0]), float(point[1])) for point in instance.keypoints]
        draw_arrow(
            draw=draw,
            points=points,
            line_width=int(style["line_width"]),
            head_len=int(style["head_len"]),
            head_width=int(style["head_width"]),
            color=tuple(style["color"]),
            head_style=str(style["head_style"]),
            line_style=str(style["line_style"]),
            render_style=str(style["render_style"]),
            rng=rng,
            geometry_mode=str(style.get("geometry_mode", "polyline")),
        )
        render_box = arrow_bbox(instance.keypoints, int(style["line_width"]), int(style["head_len"]), int(style["head_width"]))
        instance.render_bbox = [
            round(clamp(render_box[0], 0, width - 1), 2),
            round(clamp(render_box[1], 0, height - 1), 2),
            round(clamp(render_box[2], 0, width - 1), 2),
            round(clamp(render_box[3], 0, height - 1), 2),
        ]
        instance.meta.update(
            {
                "style_profile": style["style_profile"],
                "line_width": int(style["line_width"]),
                "head_len": int(style["head_len"]),
                "head_width": int(style["head_width"]),
                "line_style": str(style["line_style"]),
                "head_style": str(style["head_style"]),
                "render_style": str(style["render_style"]),
                "color": list(style["color"]),
            }
        )
