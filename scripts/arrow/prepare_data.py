#!/usr/bin/env python
from __future__ import annotations

import argparse

from vlm_structgen.domains.arrow.data.prepare import prepare_normalized_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare arrow annotations into train/val JSONL files.")
    parser.add_argument("--raw-json-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_normalized_dataset(
        raw_json_dir=args.raw_json_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print(report)


if __name__ == "__main__":
    main()
