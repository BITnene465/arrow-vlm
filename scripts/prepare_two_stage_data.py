#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from vlm_det.data.two_stage import prepare_two_stage_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare stage1/stage2 datasets from processed arrow annotations.")
    parser.add_argument("--input-dir", required=True, help="Directory containing processed train.jsonl and val.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory to write stage1/stage2 datasets and crop images.")
    parser.add_argument("--padding-ratio", type=float, default=0.5, help="Padding ratio around bbox for stage2 crop generation.")
    parser.add_argument("--num-bins", type=int, default=1000, help="Coordinate quantization bins for prompt/target serialization.")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of worker processes for per-image crop export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_two_stage_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        padding_ratio=args.padding_ratio,
        num_bins=args.num_bins,
        num_workers=args.num_workers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
