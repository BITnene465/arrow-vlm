#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from vlm_det.config import apply_run_id, load_config, config_to_dict
from vlm_det.data.collator import ArrowSFTCollator
from vlm_det.data.dataset import ArrowSFTDataset
from vlm_det.eval.evaluator import ArrowEvaluator
from vlm_det.modeling.builder import build_model_tokenizer_processor
from vlm_det.protocol.codec import ArrowCodec
from vlm_det.train.optim import build_optimizer, build_scheduler
from vlm_det.train.trainer import ArrowTrainer
from vlm_det.utils.distributed import barrier, cleanup_distributed, init_distributed, seed_everything
from vlm_det.utils.logging import ExperimentLogger, format_count


def _parse_bool_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3-VL on arrow grounding and keypoint prediction.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stage-name", default=None)
    parser.add_argument("--freeze-vision-tower", type=_parse_bool_flag, default=None)
    parser.add_argument("--gradient-checkpointing", type=_parse_bool_flag, default=None)
    parser.add_argument("--init-from", default=None)
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


class _SortedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        indices: list[int],
        batch_size: int,
        *,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        self.batch_size = max(int(batch_size), 1)
        self.indices = list(indices)[int(rank) :: max(int(world_size), 1)]

    def __iter__(self) -> Iterator[list[int]]:
        for start in range(0, len(self.indices), self.batch_size):
            yield self.indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)


def _build_val_dataloader(
    dataset,
    collator,
    batch_size,
    num_workers,
    pin_memory,
    persistent_workers,
    distributed,
    world_size,
    rank,
    tokenizer,
    bucket_by_target_length: bool,
):
    if not bucket_by_target_length:
        return _build_dataloader(
            dataset,
            collator,
            batch_size,
            num_workers,
            pin_memory,
            persistent_workers,
            distributed,
            shuffle=False,
        )
    target_lengths = dataset.get_target_token_lengths(tokenizer)
    sorted_indices = [index for index, _length in sorted(enumerate(target_lengths), key=lambda item: item[1])]
    batch_sampler = _SortedBatchSampler(
        sorted_indices,
        batch_size,
        world_size=world_size if distributed else 1,
        rank=rank if distributed else 0,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collator,
    )


def main() -> None:
    args = parse_args()
    print("[startup] loading config...", flush=True)
    config = load_config(args.config)
    if args.run_id:
        config = apply_run_id(config, args.run_id, stage_name=args.stage_name)
        print(
            f"[startup] applied run_id={args.run_id!r} "
            f"stage_name={args.stage_name!r} "
            f"output_dir={config.experiment.output_dir}",
            flush=True,
        )
    if args.freeze_vision_tower is not None:
        config.model.freeze_vision_tower = args.freeze_vision_tower
        print(
            f"[startup] override freeze_vision_tower={config.model.freeze_vision_tower}",
            flush=True,
        )
    if args.gradient_checkpointing is not None:
        config.train.gradient_checkpointing = args.gradient_checkpointing
        print(
            f"[startup] override gradient_checkpointing={config.train.gradient_checkpointing}",
            flush=True,
        )
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
        padding_side="left",
    )
    print("[startup] loading datasets...", flush=True)
    train_dataset = ArrowSFTDataset(
        jsonl_path=config.data.train_path,
        codec=codec,
        system_prompt=config.prompt.system_prompt,
        user_prompt=config.prompt.user_prompt,
        system_prompt_template=config.prompt.system_prompt_template,
        user_prompt_template=config.prompt.user_prompt_template,
    )
    val_dataset = ArrowSFTDataset(
        jsonl_path=config.data.val_path,
        codec=codec,
        system_prompt=config.prompt.system_prompt,
        user_prompt=config.prompt.user_prompt,
        system_prompt_template=config.prompt.system_prompt_template,
        user_prompt_template=config.prompt.user_prompt_template,
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
    val_loader = _build_val_dataloader(
        val_dataset,
        val_collator,
        config.eval.per_device_batch_size,
        config.data.num_workers,
        config.data.pin_memory,
        config.data.persistent_workers,
        dist_ctx.distributed,
        dist_ctx.world_size,
        dist_ctx.rank,
        build_artifacts.tokenizer,
        config.eval.bucket_by_target_length,
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
        "Loaded model; "
        f"trainable={format_count(trainable_params)} / {format_count(total_params)} "
        f"({trainable_ratio:.2f}%)"
    )
    evaluator = ArrowEvaluator(
        codec=codec,
        tokenizer=build_artifacts.tokenizer,
        max_new_tokens=config.eval.max_new_tokens,
        num_beams=config.eval.num_beams,
        do_sample=config.eval.do_sample,
        temperature=config.eval.temperature,
        top_p=config.eval.top_p,
        top_k=config.eval.top_k,
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
        evaluator=evaluator,
        logger=logger,
    )
    init_path = args.init_from or config.checkpoint.init_from
    resume_path = args.resume_from or config.checkpoint.resume_from
    if init_path and resume_path:
        raise ValueError("`init-from` and `resume-from` are mutually exclusive. Choose only one.")
    if init_path:
        print(f"[startup] initializing model weights from checkpoint: {init_path}", flush=True)
        init_meta = trainer.initialize_model_from_checkpoint(init_path, strict=True)
        init_mode = init_meta.get("config", {}).get("finetune", {}).get("mode")
        if init_mode:
            print(f"[startup] loaded initialization weights from finetune.mode={init_mode}", flush=True)
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
