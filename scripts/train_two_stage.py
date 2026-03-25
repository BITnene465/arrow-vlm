#!/usr/bin/env python
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from vlm_det.config import apply_run_id, load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ENTRYPOINT = REPO_ROOT / "scripts" / "train.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run two-stage training: stage 1 synthetic post-train, then stage 2 real-data SFT "
            "initialized from the stage-1 checkpoint."
        )
    )
    parser.add_argument("--stage1-config", default="configs/train_sync_posttrain.yaml")
    parser.add_argument("--stage2-config", default="configs/train_full_ft.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stage1-checkpoint-tag", choices=["best", "last"], default="best")
    parser.add_argument(
        "--stage1-checkpoint-dir",
        default=None,
        help="Optional explicit stage-1 checkpoint directory. If omitted, infer from stage1 config output_dir.",
    )
    parser.add_argument(
        "--runner",
        default=None,
        help=(
            "Optional runner prefix, for example: 'torchrun --nproc_per_node=2'. "
            "By default uses the current Python interpreter."
        ),
    )
    parser.add_argument(
        "--train-entrypoint",
        default=str(DEFAULT_TRAIN_ENTRYPOINT),
        help="Training entrypoint to execute for each stage.",
    )
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage1-init-from", default=None)
    parser.add_argument("--stage1-resume-from", default=None)
    parser.add_argument("--stage2-resume-from", default=None)
    return parser.parse_args()


def _resolve_runner_prefix(runner: str | None) -> list[str]:
    if runner is None or not runner.strip():
        return [sys.executable]
    return shlex.split(runner)


def _resolve_checkpoint_dir(
    *,
    stage1_config_path: str,
    stage1_checkpoint_dir: str | None,
    stage1_checkpoint_tag: str,
    run_id: str | None,
    stage_name: str | None,
) -> Path:
    if stage1_checkpoint_dir is not None:
        return Path(stage1_checkpoint_dir)
    config = load_config(stage1_config_path)
    if run_id:
        config = apply_run_id(config, run_id, stage_name=stage_name)
    return Path(config.experiment.output_dir) / "checkpoints" / stage1_checkpoint_tag


def _build_stage_command(
    *,
    runner_prefix: list[str],
    train_entrypoint: str,
    config_path: str,
    run_id: str | None = None,
    stage_name: str | None = None,
    init_from: str | None = None,
    resume_from: str | None = None,
) -> list[str]:
    command = [*runner_prefix, train_entrypoint, "--config", config_path]
    if run_id:
        command.extend(["--run-id", run_id])
    if stage_name:
        command.extend(["--stage-name", stage_name])
    if init_from:
        command.extend(["--init-from", init_from])
    if resume_from:
        command.extend(["--resume-from", resume_from])
    return command


def _print_command(label: str, command: list[str]) -> None:
    rendered = shlex.join(command)
    print(f"[{label}] {rendered}", flush=True)


def _run_or_print(label: str, command: list[str], *, dry_run: bool) -> None:
    _print_command(label, command)
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    runner_prefix = _resolve_runner_prefix(args.runner)
    stage1_name = "stage1_sync_posttrain"
    stage2_name = "stage2_real_sft"
    stage1_checkpoint_dir = _resolve_checkpoint_dir(
        stage1_config_path=args.stage1_config,
        stage1_checkpoint_dir=args.stage1_checkpoint_dir,
        stage1_checkpoint_tag=args.stage1_checkpoint_tag,
        run_id=args.run_id,
        stage_name=stage1_name,
    )

    if not args.skip_stage1:
        stage1_command = _build_stage_command(
            runner_prefix=runner_prefix,
            train_entrypoint=args.train_entrypoint,
            config_path=args.stage1_config,
            run_id=args.run_id,
            stage_name=stage1_name if args.run_id else None,
            init_from=args.stage1_init_from,
            resume_from=args.stage1_resume_from,
        )
        _run_or_print("stage1", stage1_command, dry_run=args.dry_run)

    if not args.dry_run and not stage1_checkpoint_dir.exists():
        raise FileNotFoundError(
            "Stage-1 checkpoint not found after stage 1 finished: "
            f"{stage1_checkpoint_dir}"
        )

    stage2_command = _build_stage_command(
        runner_prefix=runner_prefix,
        train_entrypoint=args.train_entrypoint,
        config_path=args.stage2_config,
        run_id=args.run_id,
        stage_name=stage2_name if args.run_id else None,
        init_from=None if args.stage2_resume_from else str(stage1_checkpoint_dir),
        resume_from=args.stage2_resume_from,
    )
    _run_or_print("stage2", stage2_command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
