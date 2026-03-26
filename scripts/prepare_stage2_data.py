#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from vlm_det.data.two_stage import prepare_stage2_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare stage2 crop dataset from processed arrow annotations.")
    parser.add_argument("--input-dir", required=True, help="Directory containing processed train.jsonl and val.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory to write stage2 JSONL and crop images.")
    parser.add_argument("--padding-ratio", type=float, default=0.5, help="Padding ratio around bbox for stage2 crop generation.")
    parser.add_argument("--num-bins", type=int, default=1000, help="Coordinate quantization bins for prompt/target serialization.")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of worker processes for per-image crop export.")
    parser.add_argument("--stage2-aug-copies", type=int, default=2, help="Additional noisy stage2 copies to generate per train instance.")
    parser.add_argument("--bbox-center-jitter-ratio", type=float, default=0.05, help="Relative bbox center jitter range, applied in original pixel space.")
    parser.add_argument("--bbox-scale-jitter-ratio", type=float, default=0.08, help="Relative bbox width/height jitter range, applied in original pixel space.")
    parser.add_argument("--endpoint-jitter-ratio", type=float, default=0.03, help="Relative endpoint jitter range, scaled by bbox diagonal length.")
    parser.add_argument("--augmentation-seed", type=int, default=42, help="Seed used for deterministic stage2 hint augmentation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_stage2_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        padding_ratio=args.padding_ratio,
        num_bins=args.num_bins,
        num_workers=args.num_workers,
        stage2_aug_copies=args.stage2_aug_copies,
        bbox_center_jitter_ratio=args.bbox_center_jitter_ratio,
        bbox_scale_jitter_ratio=args.bbox_scale_jitter_ratio,
        endpoint_jitter_ratio=args.endpoint_jitter_ratio,
        augmentation_seed=args.augmentation_seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
