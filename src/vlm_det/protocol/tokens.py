from __future__ import annotations

ARROW_TASK_TOKEN = "<|arrow_task|>"
ARROWS_BEGIN_TOKEN = "<|arrows_begin|>"
ARROWS_END_TOKEN = "<|arrows_end|>"
ARROW_BEGIN_TOKEN = "<|arrow_begin|>"
ARROW_END_TOKEN = "<|arrow_end|>"
BOX_BEGIN_TOKEN = "<|box_begin|>"
BOX_END_TOKEN = "<|box_end|>"
POINTS_BEGIN_TOKEN = "<|points_begin|>"
POINTS_END_TOKEN = "<|points_end|>"
POINT_BEGIN_TOKEN = "<|point_begin|>"
POINT_END_TOKEN = "<|point_end|>"
VISIBLE_TOKEN = "<|visible|>"
OCCLUDED_TOKEN = "<|occluded|>"


STRUCTURE_TOKENS = [
    ARROW_TASK_TOKEN,
    ARROWS_BEGIN_TOKEN,
    ARROWS_END_TOKEN,
    ARROW_BEGIN_TOKEN,
    ARROW_END_TOKEN,
    BOX_BEGIN_TOKEN,
    BOX_END_TOKEN,
    POINTS_BEGIN_TOKEN,
    POINTS_END_TOKEN,
    POINT_BEGIN_TOKEN,
    POINT_END_TOKEN,
    VISIBLE_TOKEN,
    OCCLUDED_TOKEN,
]


def x_token(index: int) -> str:
    return f"<|x_{index:04d}|>"


def y_token(index: int) -> str:
    return f"<|y_{index:04d}|>"


def build_special_tokens(num_bins: int) -> list[str]:
    return STRUCTURE_TOKENS + [x_token(index) for index in range(num_bins)] + [
        y_token(index) for index in range(num_bins)
    ]
