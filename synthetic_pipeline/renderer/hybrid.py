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
from synthetic_pipeline.renderer.svg import (
    SvgCanvas,
    draw_distractors_svg,
    draw_occluders_svg,
)
from synthetic_pipeline.scene_sampler import bbox_iou, clamp, resolution_instance_cap
from synthetic_pipeline.schema import ArrowInstance, ContextPatch, SyntheticSample


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
                sample.context_patches.append(
                    ContextPatch(
                        bbox=[paste_x, paste_y, paste_x + patch_w, paste_y + patch_h],
                        kind="asset_negative_patch",
                        source_sample_id=patch_meta["source_sample_id"],
                        meta={"source_crop_bbox": patch_meta["source_crop_bbox"]},
                    )
                )
                context_patch_count += 1

        arrow_patch_count, arrow_patch_instances = self._paste_arrow_patches(sample, image, asset_bank, rng)

        canvas = SvgCanvas(scene.image_width, scene.image_height)
        if rng.random() < float(self.hybrid_cfg["draw_procedural_distractors_probability"]):
            draw_distractors_svg(
                canvas,
                scene.image_width,
                scene.image_height,
                style_cfg,
                rng,
                scale=float(self.hybrid_cfg["procedural_distractor_scale"]),
            )

        arrow_style = default_arrow_style(scene.image_width, scene.image_height)
        for instance in sample.instances:
            if instance.meta.get("render_origin") == "asset_patch":
                continue
            self._render_instance(canvas, scene.image_width, scene.image_height, instance, arrow_style)

        occluders = sample_occluders(scene.image_width, scene.image_height, style_cfg, rng)
        draw_occluders_svg(canvas, occluders, rng)
        image = self._composite_overlay(image, canvas.rasterize())
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
            "num_arrow_patches": arrow_patch_count,
            "asset_patch_instances": arrow_patch_instances,
            **background_meta,
            "vector_backend": "svg",
        }

    @staticmethod
    def _composite_overlay(base_image: Image.Image, overlay: Image.Image) -> Image.Image:
        composed = base_image.convert("RGBA")
        composed.alpha_composite(overlay)
        return composed.convert("RGB")

    def _render_instance(
        self,
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

    def _paste_arrow_patches(
        self,
        sample: SyntheticSample,
        image: Image.Image,
        asset_bank,
        rng: random.Random,
    ) -> tuple[int, int]:
        if asset_bank is None or not asset_bank.is_available():
            return 0, 0
        if rng.random() >= float(self.hybrid_cfg["arrow_patch_probability"]):
            return 0, 0
        patch_range = self.hybrid_cfg["arrow_patches_range"]
        target_patches = rng.randint(int(patch_range[0]), int(patch_range[1]))
        if target_patches <= 0:
            return 0, 0
        existing_boxes = [list(instance.bbox) for instance in sample.instances]
        max_instances = resolution_instance_cap(sample.scene.image_width, sample.scene.image_height)
        pasted_patches = 0
        added_instances = 0
        for _ in range(target_patches):
            if len(existing_boxes) >= max_instances:
                break
            patch_sample = asset_bank.sample_arrow_patch(rng)
            if patch_sample is None:
                continue
            patch_image, patch_meta = patch_sample
            placement = self._place_arrow_patch(
                patch_image=patch_image,
                patch_meta=patch_meta,
                canvas_width=sample.scene.image_width,
                canvas_height=sample.scene.image_height,
                existing_boxes=existing_boxes,
                remaining_slots=max_instances - len(existing_boxes),
                rng=rng,
            )
            if placement is None:
                continue
            patch, paste_x, paste_y, placed_instances = placement
            image.paste(patch, (paste_x, paste_y))
            sample.context_patches.append(
                ContextPatch(
                    bbox=[paste_x, paste_y, paste_x + patch.width, paste_y + patch.height],
                    kind="asset_arrow_patch",
                    source_sample_id=patch_meta["source_sample_id"],
                    meta={
                        "source_crop_bbox": patch_meta["source_crop_bbox"],
                        "instance_count": len(placed_instances),
                    },
                )
            )
            for placed in placed_instances:
                sample.instances.append(placed)
                existing_boxes.append(list(placed.bbox))
            pasted_patches += 1
            added_instances += len(placed_instances)
        return pasted_patches, added_instances

    def _place_arrow_patch(
        self,
        *,
        patch_image: Image.Image,
        patch_meta: dict[str, Any],
        canvas_width: int,
        canvas_height: int,
        existing_boxes: list[list[float]],
        remaining_slots: int,
        rng: random.Random,
    ) -> tuple[Image.Image, int, int, list[ArrowInstance]] | None:
        if remaining_slots <= 0:
            return None
        scale = rng.uniform(*self.hybrid_cfg["arrow_patch_scale_range"])
        patch_w = max(24, int(round(patch_image.width * scale)))
        patch_h = max(24, int(round(patch_image.height * scale)))
        if patch_w >= canvas_width or patch_h >= canvas_height:
            fit_scale = min((canvas_width - 4) / max(patch_image.width, 1), (canvas_height - 4) / max(patch_image.height, 1))
            if fit_scale <= 0.15:
                return None
            scale = min(scale, fit_scale)
            patch_w = max(24, int(round(patch_image.width * scale)))
            patch_h = max(24, int(round(patch_image.height * scale)))
        transformed_defs = patch_meta["instances"][:remaining_slots]
        resized_patch = patch_image.resize((patch_w, patch_h), Image.Resampling.BICUBIC)
        retries = int(self.hybrid_cfg["arrow_patch_placement_retry"])
        max_iou = float(self.hybrid_cfg["arrow_patch_max_iou"])
        for _ in range(retries):
            paste_x = rng.randint(0, max(canvas_width - patch_w, 0))
            paste_y = rng.randint(0, max(canvas_height - patch_h, 0))
            placed_instances = self._project_patch_instances(
                transformed_defs=transformed_defs,
                paste_x=paste_x,
                paste_y=paste_y,
                scale=scale,
                source_crop_bbox=patch_meta["source_crop_bbox"],
            )
            if not placed_instances:
                continue
            if any(
                bbox_iou(placed.bbox, existing_box) > max_iou
                for placed in placed_instances
                for existing_box in existing_boxes
            ):
                continue
            return resized_patch, paste_x, paste_y, placed_instances
        return None

    @staticmethod
    def _project_patch_instances(
        *,
        transformed_defs: list[dict[str, Any]],
        paste_x: int,
        paste_y: int,
        scale: float,
        source_crop_bbox: list[int],
    ) -> list[ArrowInstance]:
        placed_instances: list[ArrowInstance] = []
        for instance_def in transformed_defs:
            bbox = [float(value) for value in instance_def["bbox"]]
            keypoints = [[float(point[0]), float(point[1])] for point in instance_def["keypoints"]]
            scaled_bbox = [
                round(paste_x + bbox[0] * scale, 2),
                round(paste_y + bbox[1] * scale, 2),
                round(paste_x + bbox[2] * scale, 2),
                round(paste_y + bbox[3] * scale, 2),
            ]
            scaled_keypoints = [
                [
                    round(paste_x + point[0] * scale, 2),
                    round(paste_y + point[1] * scale, 2),
                ]
                for point in keypoints
            ]
            placed_instances.append(
                ArrowInstance(
                    label=str(instance_def.get("label", "single_arrow")),
                    bbox=scaled_bbox,
                    keypoints=scaled_keypoints,
                    render_bbox=list(scaled_bbox),
                    source_asset_sample_id=instance_def.get("source_asset_sample_id"),
                    source_asset_instance_id=instance_def.get("source_asset_instance_id"),
                    meta={
                        "render_origin": "asset_patch",
                        "source_crop_bbox": list(source_crop_bbox),
                        "scale_from_asset_patch": round(scale, 3),
                    },
                )
            )
        return placed_instances
