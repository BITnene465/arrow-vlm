from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ASSET_CONFIG = {
    "processed_jsonl_paths": [
        "data/processed/train.jsonl",
        "data/processed/val.jsonl",
    ],
    "max_records": None,
    "background_min_crop_ratio": 0.35,
    "background_retry": 24,
    "negative_crop_retry": 20,
    "instance_exclusion_margin_ratio": 0.12,
    "context_patch_size_range": [96, 320],
    "arrow_patch_retry": 24,
    "arrow_patch_pad_ratio_range": [0.12, 0.34],
    "arrow_patch_min_context_px": 12,
    "arrow_patch_max_instances_per_crop": 4,
}

DEFAULT_RENDERER_CONFIG = {
    "name": "procedural",
    "hybrid": {
        "use_real_background_probability": 1.0,
        "context_patch_probability": 0.8,
        "context_patches_range": [1, 4],
        "arrow_patch_probability": 0.42,
        "arrow_patches_range": [0, 2],
        "arrow_patch_scale_range": [0.7, 1.15],
        "arrow_patch_placement_retry": 28,
        "arrow_patch_max_iou": 0.08,
        "draw_procedural_distractors_probability": 0.25,
        "procedural_distractor_scale": 0.45,
        "degradation_scale": 0.6,
    },
}

DEFAULT_DEBUG_CONFIG = {
    "save_visualizations": False,
    "max_saved_samples_per_split": 20,
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return normalize_config(cfg)


def normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(cfg)
    renderer_cfg = normalized.setdefault("renderer", {})
    for key, value in DEFAULT_RENDERER_CONFIG.items():
        if key not in renderer_cfg:
            renderer_cfg[key] = deepcopy(value)
        elif isinstance(value, dict):
            for nested_key, nested_value in value.items():
                renderer_cfg[key].setdefault(nested_key, deepcopy(nested_value))

    assets_cfg = normalized.setdefault("assets", {})
    for key, value in DEFAULT_ASSET_CONFIG.items():
        assets_cfg.setdefault(key, deepcopy(value))

    debug_cfg = normalized.setdefault("debug", {})
    for key, value in DEFAULT_DEBUG_CONFIG.items():
        debug_cfg.setdefault(key, deepcopy(value))

    return normalized
