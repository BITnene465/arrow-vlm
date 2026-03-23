#!/usr/bin/env python
from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader, DistributedSampler

from vlm_det.config import load_config, config_to_dict
from vlm_det.data.collator import ArrowSFTCollator
from vlm_det.data.dataset import ArrowSFTDataset
from vlm_det.eval.evaluator import ArrowEvaluator
from vlm_det.modeling.builder import build_model_tokenizer_processor
from vlm_det.protocol.codec import ArrowCodec
from vlm_det.train.optim import build_optimizer, build_scheduler
from vlm_det.train.trainer import ArrowTrainer
from vlm_det.utils.distributed import barrier, cleanup_distributed, init_distributed, seed_everything
from vlm_det.utils.logging import ExperimentLogger, format_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3-VL on arrow detection protocol.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None)
    return parser.parse_args()


def _build_dataloader(dataset, collator, batch_size, num_workers, pin_memory, persistent_workers, distributed, shuffle):
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collator,
    )


def main() -> None:
    args = parse_args()
    print("[startup] loading config...", flush=True)
    config = load_config(args.config)
    dist_ctx = init_distributed()
    seed_everything(config.experiment.seed, rank=dist_ctx.rank)

    print("[startup] building model, tokenizer, and processor...", flush=True)
    build_artifacts = build_model_tokenizer_processor(config)
    print("[startup] building codec and collator...", flush=True)
    codec = ArrowCodec(num_bins=config.tokenizer.num_bins)
    train_collator = ArrowSFTCollator(
        processor=build_artifacts.processor,
        tokenizer=build_artifacts.tokenizer,
        add_eos_token=config.tokenizer.add_eos_token,
        min_pixels=config.model.min_pixels,
        max_pixels=config.model.max_pixels,
        include_targets_in_inputs=True,
    )
    val_collator = ArrowSFTCollator(
        processor=build_artifacts.processor,
        tokenizer=build_artifacts.tokenizer,
        add_eos_token=config.tokenizer.add_eos_token,
        min_pixels=config.model.min_pixels,
        max_pixels=config.model.max_pixels,
        include_targets_in_inputs=False,
    )
    print("[startup] loading datasets...", flush=True)
    train_dataset = ArrowSFTDataset(
        jsonl_path=config.data.train_path,
        codec=codec,
        system_prompt=config.prompt.system_prompt,
        shuffle_instances=config.data.shuffle_instances_for_training,
    )
    val_dataset = ArrowSFTDataset(
        jsonl_path=config.data.val_path,
        codec=codec,
        system_prompt=config.prompt.system_prompt,
        shuffle_instances=False,
    )
    print("[startup] building dataloaders...", flush=True)
    train_loader = _build_dataloader(
        train_dataset,
        train_collator,
        config.train.per_device_batch_size,
        config.data.num_workers,
        config.data.pin_memory,
        config.data.persistent_workers,
        dist_ctx.distributed,
        shuffle=True,
    )
    val_loader = _build_dataloader(
        val_dataset,
        val_collator,
        config.train.per_device_batch_size,
        config.data.num_workers,
        config.data.pin_memory,
        config.data.persistent_workers,
        dist_ctx.distributed,
        shuffle=False,
    )

    total_steps_per_epoch = math.ceil(
        len(train_loader) / max(config.train.grad_accum_steps, 1)
    )
    total_steps = max(total_steps_per_epoch * config.train.epochs, 1)
    print("[startup] building optimizer and scheduler...", flush=True)
    optimizer = build_optimizer(build_artifacts.model, config)
    scheduler = build_scheduler(optimizer, config, total_steps)
    print("[startup] initializing logger...", flush=True)
    logger = ExperimentLogger(
        output_dir=config.experiment.output_dir,
        use_wandb=config.logging.use_wandb,
        project=config.logging.project,
        run_name=config.logging.run_name or config.experiment.name,
        config=config_to_dict(config),
    )
    trainable_params = build_artifacts.trainable_summary["trainable_params"]
    total_params = build_artifacts.trainable_summary["total_params"]
    trainable_ratio = 100.0 * trainable_params / max(total_params, 1)
    logger.info(
        "Loaded model with "
        f"{build_artifacts.num_added_tokens} added tokens; "
        f"trainable={format_count(trainable_params)} / {format_count(total_params)} "
        f"({trainable_ratio:.2f}%)"
    )
    evaluator = ArrowEvaluator(
        codec=codec,
        tokenizer=build_artifacts.tokenizer,
        max_new_tokens=config.eval.max_new_tokens,
        num_beams=config.eval.num_beams,
        do_sample=config.eval.do_sample,
        use_cache=config.eval.use_cache,
        bbox_iou_threshold=config.eval.bbox_iou_threshold,
        strict_point_distance_px=config.eval.strict_point_distance_px,
    )
    print("[startup] building trainer...", flush=True)
    trainer = ArrowTrainer(
        model=build_artifacts.model,
        tokenizer=build_artifacts.tokenizer,
        processor=build_artifacts.processor,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=dist_ctx.device,
        rank=dist_ctx.rank,
        world_size=dist_ctx.world_size,
        special_tokens=build_artifacts.special_tokens,
        evaluator=evaluator,
        logger=logger,
    )
    resume_path = args.resume_from or config.checkpoint.resume_from
    if resume_path:
        print(f"[startup] resuming from checkpoint: {resume_path}", flush=True)
        trainer.load_checkpoint(resume_path, strict=True, resume_training_state=True)
    print("[startup] start training.", flush=True)
    trainer.fit()
    barrier()
    logger.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
