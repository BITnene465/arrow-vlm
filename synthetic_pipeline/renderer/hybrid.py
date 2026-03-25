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
    render_textured_mask,
    sample_arrow_style,
    sample_occluders,
    scene_style_config,
)
from synthetic_pipeline.scene_sampler import clamp
from synthetic_pipeline.schema import ContextPatch, SyntheticSample


class HybridRenderer:
    name = "hybrid"

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.hybrid_cfg = cfg["renderer"]["hybrid"]

    def render(self, sample: SyntheticSample, rng: random.Random, asset_bank=None) -> tuple[Image.Image, dict[str, Any]]:
        scene = sample.scene
        style_cfg = scene_style_config(self.cfg["style"], scene.scene_mode)
        background_meta: dict[str, Any] = {"asset_backed_background": False}

        image = None
        if asset_bank is not None and asset_bank.is_available() and rng.random() < float(self.hybrid_cfg["use_real_background_probability"]):
            background_sample = asset_bank.sample_background(scene.image_width, scene.image_height, rng)
            if background_sample is not None:
                image, asset_meta = background_sample
                background_meta.update(asset_meta)
                background_meta["asset_backed_background"] = True
        if image is None:
            image = make_background(scene.image_width, scene.image_height, style_cfg, rng)

        draw = ImageDraw.Draw(image)
        context_patch_count = 0
        if asset_bank is not None and asset_bank.is_available() and rng.random() < float(self.hybrid_cfg["context_patch_probability"]):
            patch_range = self.hybrid_cfg["context_patches_range"]
            target_patches = rng.randint(int(patch_range[0]), int(patch_range[1]))
            for _ in range(target_patches):
                patch_sample = asset_bank.sample_context_patch(rng)
                if patch_sample is None:
                    continue
                patch_image, patch_meta = patch_sample
                x1 = rng.randint(0, max(scene.image_width - 1, 0))
                y1 = rng.randint(0, max(scene.image_height - 1, 0))
                max_patch_w = max(scene.image_width // 2, 1)
                max_patch_h = max(scene.image_height // 2, 1)
                patch_w = int(clamp(patch_image.width, 24, max_patch_w))
                patch_h = int(clamp(patch_image.height, 24, max_patch_h))
                patch = patch_image.resize((patch_w, patch_h), Image.Resampling.BICUBIC)
                paste_x = int(clamp(x1, 0, max(scene.image_width - patch_w, 0)))
                paste_y = int(clamp(y1, 0, max(scene.image_height - patch_h, 0)))
                image.paste(patch, (paste_x, paste_y))
                draw.rectangle(
                    (paste_x, paste_y, paste_x + patch_w, paste_y + patch_h),
                    outline=(230, 230, 230),
                    width=1,
                )
                sample.context_patches.append(
                    ContextPatch(
                        bbox=[paste_x, paste_y, paste_x + patch_w, paste_y + patch_h],
                        kind="asset_negative_patch",
                        source_sample_id=patch_meta["source_sample_id"],
                        meta={"source_crop_bbox": patch_meta["source_crop_bbox"]},
                    )
                )
                context_patch_count += 1

        if rng.random() < float(self.hybrid_cfg["draw_procedural_distractors_probability"]):
            draw_distractors(
                draw,
                scene.image_width,
                scene.image_height,
                style_cfg,
                rng,
                scale=float(self.hybrid_cfg["procedural_distractor_scale"]),
            )

        asset_style_count = 0
        textured_arrow_count = 0
        for instance in sample.instances:
            style = None
            if asset_bank is not None and asset_bank.is_available():
                style = asset_bank.sample_style_hint(rng)
            if style is None:
                style = sample_arrow_style(style_cfg, rng)
                style["source_asset_sample_id"] = None
                style["source_asset_instance_id"] = None
            else:
                style["style_profile"] = "asset_guided"
                asset_style_count += 1
            style["geometry_mode"] = instance.meta.get("geometry_mode", "polyline")
            if asset_bank is not None and asset_bank.is_available() and rng.random() < float(self.hybrid_cfg["texture_patch_probability"]):
                texture_sample = asset_bank.sample_texture_patch(rng)
            else:
                texture_sample = None
            self._render_instance(image, draw, scene.image_width, scene.image_height, instance, style, rng, texture_sample)
            if texture_sample is not None:
                textured_arrow_count += 1

        occluders = sample_occluders(scene.image_width, scene.image_height, style_cfg, rng)
        draw_occluders(draw, occluders, rng)
        image = degrade_image(
            image,
            style_cfg,
            rng,
            scale=float(self.hybrid_cfg["degradation_scale"]),
        )
        return image, {
            "renderer": self.name,
            "num_context_patches": context_patch_count,
            "num_occluders": len(occluders),
            "asset_guided_instances": asset_style_count,
            "textured_instances": textured_arrow_count,
            **background_meta,
        }

    def _render_instance(
        self,
        image: Image.Image,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        instance,
        style: dict[str, Any],
        rng: random.Random,
        texture_sample,
    ) -> None:
        points = [(float(point[0]), float(point[1])) for point in instance.keypoints]
        if texture_sample is None:
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
        else:
            texture_image, texture_meta = texture_sample
            mask = Image.new("L", (width, height), 0)
            mask_draw = ImageDraw.Draw(mask)
            draw_arrow(
                draw=mask_draw,
                points=points,
                line_width=int(style["line_width"]),
                head_len=int(style["head_len"]),
                head_width=int(style["head_width"]),
                color=255,
                head_style=str(style["head_style"]),
                line_style=str(style["line_style"]),
                render_style=str(style["render_style"]),
                rng=rng,
                geometry_mode=str(style.get("geometry_mode", "polyline")),
            )
            textured = render_textured_mask(
                base_image=image,
                mask=mask,
                texture=texture_image,
                color=tuple(style["color"]),
                opacity=rng.uniform(*self.hybrid_cfg["texture_opacity_range"]),
            )
            image.paste(textured)
            style["texture_source_sample_id"] = texture_meta["source_sample_id"]
            style["texture_source_crop_bbox"] = texture_meta["source_crop_bbox"]

        render_box = arrow_bbox(instance.keypoints, int(style["line_width"]), int(style["head_len"]), int(style["head_width"]))
        instance.render_bbox = [
            round(clamp(render_box[0], 0, width - 1), 2),
            round(clamp(render_box[1], 0, height - 1), 2),
            round(clamp(render_box[2], 0, width - 1), 2),
            round(clamp(render_box[3], 0, height - 1), 2),
        ]
        instance.source_asset_sample_id = style.get("source_asset_sample_id")
        instance.source_asset_instance_id = style.get("source_asset_instance_id")
        instance.meta.update(
            {
                "style_profile": style.get("style_profile", "asset_guided"),
                "line_width": int(style["line_width"]),
                "head_len": int(style["head_len"]),
                "head_width": int(style["head_width"]),
                "line_style": str(style["line_style"]),
                "head_style": str(style["head_style"]),
                "render_style": str(style["render_style"]),
                "color": list(style["color"]),
            }
        )
        if style.get("texture_source_sample_id") is not None:
            instance.meta["texture_source_sample_id"] = style["texture_source_sample_id"]
            instance.meta["texture_source_crop_bbox"] = style["texture_source_crop_bbox"]
