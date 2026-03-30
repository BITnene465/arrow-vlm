#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from vlm_structgen.domains.arrow.data.two_stage import prepare_stage1_data


def _parse_float_sequence(raw: str) -> list[float]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    return [float(item) for item in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare stage1 mixed full-image/tile dataset from processed arrow annotations.")
    parser.add_argument("--input-dir", required=True, help="Directory containing processed train.jsonl and val.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory to write stage1 JSONL and tile images.")
    parser.add_argument(
        "--stage1-include-full-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to keep original full-image stage1 samples.",
    )
    parser.add_argument(
        "--stage1-tile-size-ratios",
        type=_parse_float_sequence,
        default=[0.35, 0.5],
        help="Comma-separated stage1 crop size ratios relative to the image short side, e.g. 0.35,0.5.",
    )
    parser.add_argument(
        "--stage1-min-tile-size",
        type=int,
        default=512,
        help="Minimum stage1 crop size in pixels after ratio resolution.",
    )
    parser.add_argument(
        "--stage1-max-tile-size",
        type=int,
        default=1280,
        help="Maximum stage1 crop size in pixels after ratio resolution.",
    )
    parser.add_argument(
        "--stage1-tile-stride-ratio",
        type=float,
        default=0.75,
        help="Stride ratio for stage1 sliding-window crops.",
    )
    parser.add_argument(
        "--stage1-density-min-instances",
        type=int,
        default=5,
        help="Minimum arrow count for a stage1 density crop to be kept.",
    )
    parser.add_argument(
        "--stage1-density-max-instances",
        type=int,
        default=30,
        help="Maximum arrow count for a stage1 density crop to be kept.",
    )
    parser.add_argument(
        "--stage1-density-max-crops-per-size",
        type=int,
        default=8,
        help="Maximum number of density-driven crops to keep per tile size and image.",
    )
    parser.add_argument(
        "--stage1-dedup-iou-threshold",
        type=float,
        default=0.9,
        help="Drop near-duplicate stage1 crops when they contain the same instance set and crop IoU exceeds this threshold.",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="Number of worker processes for per-image crop export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_stage1_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        stage1_include_full_image=args.stage1_include_full_image,
        stage1_tile_size_ratios=args.stage1_tile_size_ratios,
        stage1_min_tile_size=args.stage1_min_tile_size,
        stage1_max_tile_size=args.stage1_max_tile_size,
        stage1_tile_stride_ratio=args.stage1_tile_stride_ratio,
        stage1_density_min_instances=args.stage1_density_min_instances,
        stage1_density_max_instances=args.stage1_density_max_instances,
        stage1_density_max_crops_per_size=args.stage1_density_max_crops_per_size,
        stage1_dedup_iou_threshold=args.stage1_dedup_iou_threshold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
