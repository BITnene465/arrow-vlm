from __future__ import annotations

from io import BytesIO
from typing import Any

from cairosvg import svg2png
from PIL import Image

from synthetic_pipeline.scene_sampler import normalize, orthogonal


def _fmt(value: float) -> str:
    return f"{float(value):.2f}"


def _rgb(color: tuple[int, int, int] | int) -> str:
    if isinstance(color, int):
        channel = max(0, min(255, int(color)))
        return f"rgb({channel},{channel},{channel})"
    red, green, blue = [max(0, min(255, int(channel))) for channel in color]
    return f"rgb({red},{green},{blue})"


def _path_d(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    parts = [f"M {_fmt(points[0][0])} {_fmt(points[0][1])}"]
    for x_value, y_value in points[1:]:
        parts.append(f"L {_fmt(x_value)} {_fmt(y_value)}")
    return " ".join(parts)


def _head_polygon_points(
    tail: tuple[float, float],
    head: tuple[float, float],
    head_len: float,
    head_width: float,
) -> list[tuple[float, float]]:
    dx = head[0] - tail[0]
    dy = head[1] - tail[1]
    nx, ny = normalize(dx, dy)
    ox, oy = orthogonal(nx, ny)
    base_x = head[0] - nx * head_len
    base_y = head[1] - ny * head_len
    left = (base_x + ox * head_width / 2.0, base_y + oy * head_width / 2.0)
    right = (base_x - ox * head_width / 2.0, base_y - oy * head_width / 2.0)
    return [head, left, right]


def _head_svg(
    tail: tuple[float, float],
    head: tuple[float, float],
    *,
    color: tuple[int, int, int] | int,
    line_width: int,
    head_len: int,
    head_width: int,
    opacity: float,
) -> str:
    polygon_points = _head_polygon_points(
        tail=tail,
        head=head,
        head_len=float(head_len),
        head_width=float(head_width),
    )
    point_string = " ".join(f"{_fmt(x_value)},{_fmt(y_value)}" for x_value, y_value in polygon_points)
    stroke_color = _rgb(color)
    return (
        f'<polygon points="{point_string}" fill="{stroke_color}" stroke="{stroke_color}" '
        f'stroke-width="{_fmt(max(line_width * 0.12, 0.6))}" opacity="{opacity:.3f}"/>'
    )


def build_arrow_elements(
    *,
    points: list[tuple[float, float]],
    line_width: int,
    head_len: int,
    head_width: int,
    color: tuple[int, int, int] | int,
    double_headed: bool = False,
    opacity: float = 1.0,
) -> list[str]:
    path_d = _path_d(points)
    if not path_d:
        return []
    stroke_color = _rgb(color)
    elements = [
        (
            f'<path d="{path_d}" fill="none" stroke="{stroke_color}" '
            f'stroke-width="{_fmt(line_width)}" stroke-linecap="round" '
            f'stroke-linejoin="round" opacity="{opacity:.3f}"/>'
        )
    ]
    if len(points) < 2:
        return elements
    elements.append(
        _head_svg(
            tail=points[-2],
            head=points[-1],
            color=color,
            line_width=line_width,
            head_len=head_len,
            head_width=head_width,
            opacity=opacity,
        )
    )
    if double_headed:
        elements.append(
            _head_svg(
                tail=points[1],
                head=points[0],
                color=color,
                line_width=line_width,
                head_len=head_len,
                head_width=head_width,
                opacity=opacity,
            )
        )
    return elements


class SvgCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.elements: list[str] = []

    def add_arrow(self, **kwargs: Any) -> None:
        self.elements.extend(build_arrow_elements(**kwargs))

    def add_line(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: tuple[int, int, int],
        width: int,
        opacity: float = 1.0,
    ) -> None:
        self.elements.append(
            f'<line x1="{_fmt(start[0])}" y1="{_fmt(start[1])}" '
            f'x2="{_fmt(end[0])}" y2="{_fmt(end[1])}" '
            f'stroke="{_rgb(color)}" stroke-width="{_fmt(width)}" '
            f'stroke-linecap="round" opacity="{opacity:.3f}"/>'
        )

    def add_rect(
        self,
        bbox: list[float],
        *,
        stroke: tuple[int, int, int] | None = None,
        fill: tuple[int, int, int] | None = None,
        width: int = 1,
        opacity: float = 1.0,
        radius: float = 0.0,
    ) -> None:
        x1, y1, x2, y2 = bbox
        fill_value = _rgb(fill) if fill is not None else "none"
        stroke_value = _rgb(stroke) if stroke is not None else "none"
        self.elements.append(
            f'<rect x="{_fmt(x1)}" y="{_fmt(y1)}" '
            f'width="{_fmt(max(x2 - x1, 1.0))}" height="{_fmt(max(y2 - y1, 1.0))}" '
            f'rx="{_fmt(radius)}" ry="{_fmt(radius)}" fill="{fill_value}" '
            f'stroke="{stroke_value}" stroke-width="{_fmt(width)}" opacity="{opacity:.3f}"/>'
        )

    def rasterize(self) -> Image.Image:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">'
            + "".join(self.elements)
            + "</svg>"
        )
        png_bytes = svg2png(bytestring=svg.encode("utf-8"), output_width=self.width, output_height=self.height)
        return Image.open(BytesIO(png_bytes)).convert("RGBA")


def rasterize_arrow_mask(
    width: int,
    height: int,
    *,
    points: list[tuple[float, float]],
    line_width: int,
    head_len: int,
    head_width: int,
) -> Image.Image:
    canvas = SvgCanvas(width, height)
    canvas.add_arrow(
        points=points,
        line_width=line_width,
        head_len=head_len,
        head_width=head_width,
        color=255,
        opacity=1.0,
    )
    return canvas.rasterize()


def draw_distractors_svg(
    canvas: SvgCanvas,
    width: int,
    height: int,
    style_cfg: dict[str, Any],
    rng,
    *,
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
        color = tuple(rng.randint(84, 176) for _ in range(3))
        canvas.add_line((x1, y1), (x2, y2), color=color, width=line_width, opacity=0.88)

    for _ in range(distractor_boxes):
        box_w = rng.randint(max(width // 20, 18), max(width // 6, 28))
        box_h = rng.randint(max(height // 20, 18), max(height // 6, 28))
        x1 = rng.randint(0, max(width - box_w - 1, 1))
        y1 = rng.randint(0, max(height - box_h - 1, 1))
        x2 = x1 + box_w
        y2 = y1 + box_h
        color = tuple(rng.randint(90, 170) for _ in range(3))
        canvas.add_rect([x1, y1, x2, y2], stroke=color, width=line_width, opacity=0.84, radius=max(line_width, 2))

    for _ in range(text_like_strokes):
        x_value = rng.randint(0, max(width - 72, 1))
        y_value = rng.randint(0, max(height - 20, 1))
        color = tuple(rng.randint(65, 145) for _ in range(3))
        cursor = x_value
        for _ in range(rng.randint(3, 8)):
            word_width = rng.randint(6, 16)
            y_jitter = y_value + rng.randint(-1, 1)
            canvas.add_line((cursor, y_jitter), (cursor + word_width, y_jitter), color=color, width=1, opacity=0.82)
            cursor += word_width + rng.randint(4, 10)


def draw_occluders_svg(canvas: SvgCanvas, occluders: list[list[int]], rng) -> None:
    for x1, y1, x2, y2 in occluders:
        gray = rng.randint(212, 245)
        fill = (gray, gray, gray)
        canvas.add_rect([x1, y1, x2, y2], fill=fill, stroke=fill, width=1, opacity=1.0, radius=2.0)
