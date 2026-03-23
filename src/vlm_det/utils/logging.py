from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm
import wandb

from vlm_det.utils.distributed import is_main_process
from vlm_det.utils.io import ensure_dir


class ExperimentLogger:
    def __init__(
        self,
        output_dir: str | Path,
        use_wandb: bool,
        project: str,
        run_name: str | None,
        config: dict[str, Any],
    ) -> None:
        self.output_dir = ensure_dir(output_dir)
        self._logger = logging.getLogger("vlm_det")
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        if is_main_process():
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self._logger.addHandler(stream_handler)
        file_handler = logging.FileHandler(self.output_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)
        self.run = None
        if use_wandb and is_main_process():
            try:
                settings = wandb.Settings(init_timeout=30)
                mode = os.getenv("WANDB_MODE")
                init_kwargs = {
                    "project": project,
                    "name": run_name,
                    "config": config,
                    "settings": settings,
                }
                if mode:
                    init_kwargs["mode"] = mode
                self.run = wandb.init(**init_kwargs)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(f"wandb init failed, continuing without wandb: {exc}")

    def info(self, message: str) -> None:
        self._logger.info(message)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if not metrics:
            return
        if is_main_process() and self.run is not None:
            self.run.log(metrics, step=step)

    def watch_artifact(self, payload: dict[str, Any], step: int) -> None:
        if is_main_process() and self.run is not None:
            self.run.log(payload, step=step)

    def close(self) -> None:
        if self.run is not None:
            self.run.finish()


def create_progress_bar(
    total: int,
    desc: str,
    ncols: int = 110,
    leave: bool = False,
) -> tqdm | None:
    if not is_main_process():
        return None
    return tqdm(
        total=total,
        desc=desc,
        leave=leave,
        dynamic_ncols=False,
        ncols=ncols,
        ascii=" ▏▎▍▌▋▊▉█",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
