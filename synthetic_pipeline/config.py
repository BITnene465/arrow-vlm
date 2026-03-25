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
    "negative_crop_retry": 20,
    "texture_patch_size_range": [48, 160],
    "context_patch_size_range": [96, 320],
}

DEFAULT_RENDERER_CONFIG = {
    "name": "procedural",
    "hybrid": {
        "use_real_background_probability": 1.0,
        "context_patch_probability": 0.8,
        "context_patches_range": [1, 4],
        "texture_patch_probability": 0.95,
        "texture_opacity_range": [0.75, 0.95],
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
