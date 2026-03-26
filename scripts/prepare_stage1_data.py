#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from vlm_det.data.two_stage import prepare_stage1_data


def _parse_tile_sizes(raw: str) -> list[int]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    return [int(item) for item in values]


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
        "--stage1-tile-sizes",
        type=_parse_tile_sizes,
        default=[768, 1024],
        help="Comma-separated stage1 sliding/density crop sizes, e.g. 768,1024.",
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
        "--stage1-min-visible-area-ratio",
        type=float,
        default=0.5,
        help="Minimum visible bbox area ratio for keeping an instance in a stage1 crop.",
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
        stage1_tile_sizes=args.stage1_tile_sizes,
        stage1_tile_stride_ratio=args.stage1_tile_stride_ratio,
        stage1_density_min_instances=args.stage1_density_min_instances,
        stage1_density_max_instances=args.stage1_density_max_instances,
        stage1_density_max_crops_per_size=args.stage1_density_max_crops_per_size,
        stage1_min_visible_area_ratio=args.stage1_min_visible_area_ratio,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
