#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthetic_pipeline.asset_bank import AssetBank
from synthetic_pipeline.config import load_config, normalize_config
from synthetic_pipeline.exporter import DatasetExporter, dumps_summary
from synthetic_pipeline.renderer import build_renderer
from synthetic_pipeline.scene_sampler import SceneSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic arrow dataset under data/sync/.")
    parser.add_argument(
        "--config",
        default="synthetic_pipeline/configs/base.yaml",
        help="YAML config path for synthetic dataset generation.",
    )
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--renderer",
        choices=["procedural", "hybrid"],
        default=None,
        help="Override renderer backend for this run.",
    )
    return parser.parse_args()


def _generate_split(
    *,
    split: str,
    count: int,
    base_seed: int,
    sampler: SceneSampler,
    renderer,
    exporter: DatasetExporter,
    workers: int,
    asset_bank: AssetBank | None,
) -> list[dict]:
    def generate_one(index: int) -> dict:
        rng = random.Random(base_seed + index)
        sample = sampler.sample(split=split, index=index, rng=rng)
        image, render_meta = renderer.render(sample=sample, rng=rng, asset_bank=asset_bank)
        sample.render_meta.update(render_meta)
        return exporter.save_sample(sample, image)

    indices = list(range(count))
    if workers <= 1:
        return [generate_one(index) for index in indices]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(generate_one, indices))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.train_samples is not None:
        cfg["train_samples"] = args.train_samples
    if args.val_samples is not None:
        cfg["val_samples"] = args.val_samples
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.workers is not None:
        cfg["workers"] = args.workers
    if args.renderer is not None:
        cfg["renderer"]["name"] = args.renderer
    cfg = normalize_config(cfg)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = build_renderer(cfg)
    sampler = SceneSampler(cfg=cfg, renderer_name=renderer.name)
    asset_bank = AssetBank(cfg["assets"])
    exporter = DatasetExporter(output_dir=output_dir, debug_cfg=cfg["debug"])
    workers = max(1, int(cfg.get("workers", 1)))
    base_seed = int(cfg["seed"])

    train_records = _generate_split(
        split="train",
        count=int(cfg["train_samples"]),
        base_seed=base_seed,
        sampler=sampler,
        renderer=renderer,
        exporter=exporter,
        workers=workers,
        asset_bank=asset_bank if renderer.name == "hybrid" else None,
    )
    val_records = _generate_split(
        split="val",
        count=int(cfg["val_samples"]),
        base_seed=base_seed + 1_000_000,
        sampler=sampler,
        renderer=renderer,
        exporter=exporter,
        workers=workers,
        asset_bank=asset_bank if renderer.name == "hybrid" else None,
    )

    summary = exporter.finalize(train_records=train_records, val_records=val_records, cfg=cfg)
    print(dumps_summary(summary))


if __name__ == "__main__":
    main()
