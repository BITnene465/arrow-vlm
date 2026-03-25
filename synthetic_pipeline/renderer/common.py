from __future__ import annotations

import math
import random
from io import BytesIO
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from synthetic_pipeline.scene_sampler import clamp, normalize, orthogonal


def weighted_choice(rng: random.Random, mapping: dict[str, float]) -> str:
    keys = list(mapping.keys())
    weights = list(mapping.values())
    return rng.choices(keys, weights=weights, k=1)[0]


def apply_render_style(points: list[tuple[float, float]], render_style: str, rng: random.Random) -> list[tuple[float, float]]:
    if render_style == "clean":
        return points
    jitter_scale = 0.0
    if render_style == "handdrawn":
        jitter_scale = 1.5
    elif render_style == "handdrawn_heavy":
        jitter_scale = 3.0
    elif render_style == "marker_jitter":
        jitter_scale = 1.2
    if jitter_scale <= 0:
        return points
    styled: list[tuple[float, float]] = []
    for index, (x_value, y_value) in enumerate(points):
        if index == 0 or index == len(points) - 1:
            styled.append((x_value, y_value))
            continue
        styled.append(
            (
                x_value + rng.uniform(-jitter_scale, jitter_scale),
                y_value + rng.uniform(-jitter_scale, jitter_scale),
            )
        )
    return styled


def catmull_rom_spline(points: list[tuple[float, float]], samples_per_segment: int = 20) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points
    extended = [points[0], *points, points[-1]]
    curve: list[tuple[float, float]] = []
    for index in range(1, len(extended) - 2):
        p0 = extended[index - 1]
        p1 = extended[index]
        p2 = extended[index + 1]
        p3 = extended[index + 2]
        for step in range(samples_per_segment):
            t_value = step / samples_per_segment
            t_squared = t_value * t_value
            t_cubed = t_squared * t_value
            x_value = 0.5 * (
                (2 * p1[0])
                + (-p0[0] + p2[0]) * t_value
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t_squared
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t_cubed
            )
            y_value = 0.5 * (
                (2 * p1[1])
                + (-p0[1] + p2[1]) * t_value
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t_squared
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t_cubed
            )
            curve.append((x_value, y_value))
    curve.append(points[-1])
    return curve


def arrow_bbox(points: list[list[float]], line_width: int, head_len: int, head_width: int) -> list[float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    pad = max(line_width, head_len, head_width) * 0.8
    return [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    line_width: int,
    head_len: int,
    head_width: int,
    color: tuple[int, int, int] | int,
    head_style: str,
    line_style: str,
    render_style: str,
    rng: random.Random,
    geometry_mode: str,
    double_headed: bool = False,
) -> None:
    points = apply_render_style(points, render_style, rng=rng)
    line_points = catmull_rom_spline(points) if geometry_mode != "polyline" else points
    if line_style == "dashed":
        for start, end in zip(line_points[:-1], line_points[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            segment_length = math.hypot(dx, dy)
            if segment_length <= 1e-6:
                continue
            dash_len = max(line_width * 3, 8)
            gap_len = max(line_width * 2, 5)
            ux, uy = dx / segment_length, dy / segment_length
            cursor = 0.0
            while cursor < segment_length:
                dash_end = min(cursor + dash_len, segment_length)
                p1 = (start[0] + ux * cursor, start[1] + uy * cursor)
                p2 = (start[0] + ux * dash_end, start[1] + uy * dash_end)
                draw.line((p1, p2), fill=color, width=line_width)
                cursor += dash_len + gap_len
    elif line_style == "dotted":
        for start, end in zip(line_points[:-1], line_points[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            segment_length = math.hypot(dx, dy)
            if segment_length <= 1e-6:
                continue
            step = max(line_width * 2.2, 6)
            ux, uy = dx / segment_length, dy / segment_length
            cursor = 0.0
            radius = max(1, int(line_width * 0.8))
            while cursor <= segment_length:
                cx = start[0] + ux * cursor
                cy = start[1] + uy * cursor
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
                cursor += step
    elif line_style == "marker":
        draw.line(line_points, fill=color, width=max(line_width, 6), joint="curve")
        if not isinstance(color, int):
            overlay_color = tuple(min(255, channel + 18) for channel in color)
            draw.line(line_points, fill=overlay_color, width=max(1, line_width // 2), joint="curve")
    else:
        draw.line(line_points, fill=color, width=line_width, joint="curve")

    if len(points) < 2:
        return
    head_segments = [(-2, -1)]
    if double_headed and len(points) >= 2:
        head_segments.append((1, 0))
    for tail_index, head_index in head_segments:
        tail = points[tail_index]
        head = points[head_index]
        dx = head[0] - tail[0]
        dy = head[1] - tail[1]
        nx, ny = normalize(dx, dy)
        ox, oy = orthogonal(nx, ny)
        base_x = head[0] - nx * head_len
        base_y = head[1] - ny * head_len
        left = (base_x + ox * head_width / 2, base_y + oy * head_width / 2)
        right = (base_x - ox * head_width / 2, base_y - oy * head_width / 2)
        if head_style == "open":
            draw.line((left, head), fill=color, width=max(1, line_width))
            draw.line((right, head), fill=color, width=max(1, line_width))
        else:
            draw.polygon([head, left, right], fill=color)


def sample_arrow_style(style_cfg: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    profile = rng.choices(
        style_cfg["arrow_style_profiles"],
        weights=[float(item["weight"]) for item in style_cfg["arrow_style_profiles"]],
        k=1,
    )[0]
    return {
        "style_profile": profile["name"],
        "line_width": rng.randint(*profile["line_width_range"]),
        "head_len": rng.randint(*profile["head_length_range"]),
        "head_width": rng.randint(*profile["head_width_range"]),
        "head_style": rng.choice(profile["head_styles"]),
        "line_style": rng.choice(profile["line_styles"]),
        "render_style": rng.choice(profile["render_styles"]),
        "color": tuple(rng.choice(profile["palettes"])),
    }


def scene_style_config(style_cfg: dict[str, Any], scene_mode: str) -> dict[str, Any]:
    if scene_mode != "single_crop":
        return style_cfg
    adjusted = dict(style_cfg)
    adjusted["distractor_lines_range"] = [0, 5]
    adjusted["distractor_boxes_range"] = [0, 2]
    adjusted["text_like_strokes_range"] = [0, 4]
    adjusted["double_head_distractor_range"] = [0, 2]
    adjusted["occluder_count_range"] = [0, 2]
    return adjusted


def make_background(width: int, height: int, style_cfg: dict[str, Any], rng: random.Random) -> Image.Image:
    mode = weighted_choice(rng, style_cfg["background_modes"])
    if mode == "plain_white":
        color = (255, 255, 255)
    elif mode == "gray":
        gray = rng.randint(220, 245)
        color = (gray, gray, gray)
    else:
        base = rng.randint(235, 250)
        tint = rng.randint(-8, 8)
        color = (base + tint, base + tint // 2, base - tint // 2)
    image = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(image)
    if mode == "paper":
        for _ in range(max(width * height // 12000, 50)):
            x_value = rng.randint(0, width - 1)
            y_value = rng.randint(0, height - 1)
            delta = rng.randint(-10, 10)
            pixel = tuple(int(clamp(channel + delta, 0, 255)) for channel in color)
            draw.point((x_value, y_value), fill=pixel)
    return image


def draw_distractors(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    style_cfg: dict[str, Any],
    rng: random.Random,
    scale: float = 1.0,
) -> None:
    def scaled_range(values: list[int]) -> tuple[int, int]:
        lower, upper = values
        return max(0, int(round(lower * scale))), max(0, int(round(upper * scale)))

    distractor_lines = rng.randint(*scaled_range(style_cfg["distractor_lines_range"]))
    distractor_boxes = rng.randint(*scaled_range(style_cfg["distractor_boxes_range"]))
    text_like_strokes = rng.randint(*scaled_range(style_cfg["text_like_strokes_range"]))
    line_width = max(1, int(rng.randint(*style_cfg["distractor_line_width_range"])))

    for _ in range(distractor_lines):
        x1 = rng.randint(0, width - 1)
        y1 = rng.randint(0, height - 1)
        x2 = rng.randint(0, width - 1)
        y2 = rng.randint(0, height - 1)
        color = tuple(rng.randint(80, 180) for _ in range(3))
        draw.line((x1, y1, x2, y2), fill=color, width=line_width)

    for _ in range(distractor_boxes):
        box_w = rng.randint(max(width // 20, 12), max(width // 6, 24))
        box_h = rng.randint(max(height // 20, 12), max(height // 6, 24))
        x1 = rng.randint(0, max(width - box_w - 1, 1))
        y1 = rng.randint(0, max(height - box_h - 1, 1))
        x2 = x1 + box_w
        y2 = y1 + box_h
        color = tuple(rng.randint(90, 170) for _ in range(3))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

    for _ in range(text_like_strokes):
        x_value = rng.randint(0, max(width - 60, 1))
        y_value = rng.randint(0, max(height - 20, 1))
        color = tuple(rng.randint(60, 140) for _ in range(3))
        for step in range(rng.randint(3, 8)):
            x1 = x_value + step * rng.randint(8, 14)
            x2 = x1 + rng.randint(4, 10)
            y_jitter = y_value + rng.randint(-1, 1)
            draw.line((x1, y_jitter, x2, y_jitter), fill=color, width=1)


def sample_occluders(width: int, height: int, style_cfg: dict[str, Any], rng: random.Random) -> list[list[int]]:
    occluders: list[list[int]] = []
    occluder_count = rng.randint(*style_cfg["occluder_count_range"])
    for _ in range(occluder_count):
        occ_w = rng.randint(max(width // 30, 10), max(width // 8, 24))
        occ_h = rng.randint(max(height // 30, 10), max(height // 8, 24))
        x1 = rng.randint(0, max(width - occ_w - 1, 1))
        y1 = rng.randint(0, max(height - occ_h - 1, 1))
        x2 = x1 + occ_w
        y2 = y1 + occ_h
        occluders.append([x1, y1, x2, y2])
    return occluders


def draw_occluders(draw: ImageDraw.ImageDraw, occluders: list[list[int]], rng: random.Random) -> None:
    for x1, y1, x2, y2 in occluders:
        gray = rng.randint(210, 245)
        fill = (gray, gray, gray)
        draw.rectangle((x1, y1, x2, y2), fill=fill, outline=fill)


def degrade_image(image: Image.Image, style_cfg: dict[str, Any], rng: random.Random, scale: float = 1.0) -> Image.Image:
    output = image
    if rng.random() < float(style_cfg["blur_probability"]) * scale:
        radius = rng.uniform(*style_cfg["blur_radius_range"]) * max(scale, 0.3)
        output = output.filter(ImageFilter.GaussianBlur(radius=radius))
    if rng.random() < float(style_cfg["noise_probability"]) * scale:
        pixels = output.load()
        width, height = output.size
        stddev = rng.uniform(*style_cfg["noise_stddev_range"]) * max(scale, 0.3)
        for y_value in range(height):
            for x_value in range(width):
                red, green, blue = pixels[x_value, y_value]
                jitter = int(rng.gauss(0, stddev))
                pixels[x_value, y_value] = (
                    int(clamp(red + jitter, 0, 255)),
                    int(clamp(green + jitter, 0, 255)),
                    int(clamp(blue + jitter, 0, 255)),
                )
    quality_min, quality_max = style_cfg["jpeg_quality_range"]
    quality = rng.randint(
        int(round(quality_min + (1.0 - scale) * 8)),
        quality_max,
    )
    buffer = BytesIO()
    output.save(buffer, format="JPEG", quality=int(clamp(quality, 40, 99)))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def render_textured_mask(
    base_image: Image.Image,
    mask: Image.Image,
    texture: Image.Image,
    color: tuple[int, int, int],
    opacity: float,
) -> Image.Image:
    texture_canvas = texture.resize(base_image.size, Image.Resampling.BICUBIC).convert("RGB")
    grayscale = ImageOps.grayscale(texture_canvas)
    colored = ImageOps.colorize(grayscale, black=tuple(max(channel - 30, 0) for channel in color), white=color)
    if opacity < 0.999:
        alpha = mask.point(lambda value: int(value * opacity))
    else:
        alpha = mask
    composed = base_image.copy()
    composed.paste(colored, (0, 0), alpha)
    return composed
