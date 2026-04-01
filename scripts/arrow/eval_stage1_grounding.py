from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

from vlm_structgen.core.infer import load_inference_runner
from vlm_structgen.core.registry import get_adapter
from vlm_structgen.core.utils.io import ensure_dir, load_jsonl, write_json, write_jsonl
from vlm_structgen.core.utils.logging import create_progress_bar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stage1 grounding on a JSONL split.")
    parser.add_argument("--config", default="configs/infer/infer_stage1_grounding.yaml", help="Stage1 grounding inference config path.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory. Falls back to CHECKPOINT_PATH in .env.")
    parser.add_argument("--env-file", default=None, help="Optional path to a .env file for checkpoint fallback.")
    parser.add_argument("--model", default=None, help="Optional model path/name override.")
    parser.add_argument("--device", default=None, help="Optional torch device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--jsonl", default="data/two_stage/stage1/val_full.jsonl", help="Grounding dataset JSONL path.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max_new_tokens for evaluation run.")
    parser.add_argument("--bbox-iou-threshold", type=float, default=0.5, help="IoU threshold for TP/FP/FN matching.")
    parser.add_argument("--max-samples", type=int, default=None, help="Evaluate at most N samples from JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory to write summary and per-sample outputs.")
    parser.add_argument(
        "--save-per-sample",
        dest="save_per_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save per-sample JSONL records.",
    )
    parser.add_argument("--save-badcases-topk", type=int, default=200, help="How many worst metric badcases to save.")
    return parser.parse_args()


def _resolve_image_path(record_image_path: str, *, jsonl_path: Path) -> Path:
    candidate = Path(record_image_path)
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"Image path does not exist: {candidate}")
        return candidate

    candidates = [
        Path.cwd() / candidate,
        jsonl_path.parent / candidate,
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        "Image file not found for JSONL record. "
        f"image_path={record_image_path!r}, tried={[str(path) for path in candidates]}"
    )


def _empty_counts() -> dict[str, float]:
    return {
        "samples": 0.0,
        "parse_success_lenient": 0.0,
        "parse_success_strict": 0.0,
        "bbox_tp": 0.0,
        "bbox_fp": 0.0,
        "bbox_fn": 0.0,
        "bbox_iou_sum": 0.0,
        "generated_tokens": 0.0,
        "returned_tokens": 0.0,
        "hit_max_new_tokens": 0.0,
        "closed_json_array": 0.0,
    }


def _summarize(counts: dict[str, float], *, iou_threshold: float) -> dict[str, float]:
    samples = max(counts["samples"], 1.0)
    tp = counts["bbox_tp"]
    fp = counts["bbox_fp"]
    fn = counts["bbox_fn"]
    matched = max(tp, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "samples": counts["samples"],
        "parse_rate_lenient": counts["parse_success_lenient"] / samples,
        "parse_rate_strict": counts["parse_success_strict"] / samples,
        f"bbox_precision_at_iou{int(round(iou_threshold * 100))}": precision,
        f"bbox_recall_at_iou{int(round(iou_threshold * 100))}": recall,
        f"bbox_f1_at_iou{int(round(iou_threshold * 100))}": f1,
        "bbox_iou_mean": counts["bbox_iou_sum"] / matched,
        "generation/generated_tokens_mean": counts["generated_tokens"] / samples,
        "generation/returned_tokens_mean": counts["returned_tokens"] / samples,
        "generation/hit_max_new_tokens_rate": counts["hit_max_new_tokens"] / samples,
        "generation/closed_json_array_rate": counts["closed_json_array"] / samples,
    }


def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    records = load_jsonl(jsonl_path)
    if args.max_samples is not None:
        records = records[: max(int(args.max_samples), 0)]
    if not records:
        raise ValueError(f"No records to evaluate: {jsonl_path}")

    runner = load_inference_runner(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        env_file=args.env_file,
        model_name_or_path=args.model,
        device_name=args.device,
    )
    adapter = get_adapter(
        task_type="grounding",
        domain_type="arrow",
        num_bins=runner.config.tokenizer.num_bins,
    )

    output_dir = ensure_dir(args.output_dir)
    per_sample_rows: list[dict[str, Any]] = []
    parse_badcases: list[dict[str, Any]] = []
    metric_badcases: list[dict[str, Any]] = []
    counts = _empty_counts()

    progress = create_progress_bar(total=len(records), desc="eval s1", leave=True)
    for record in records:
        task_type = str(record.get("task_type", ""))
        domain_type = str(record.get("domain_type", ""))
        if task_type != "grounding" or domain_type != "arrow":
            raise ValueError(
                "eval_stage1_grounding only accepts grounding/arrow records. "
                f"sample_id={record.get('sample_id')!r}, task_type={task_type!r}, domain_type={domain_type!r}"
            )

        image_path = _resolve_image_path(str(record.get("image_path", "")), jsonl_path=jsonl_path)
        image = Image.open(image_path).convert("RGB")
        raw_text, parse_report = runner.predict(image, max_new_tokens=args.max_new_tokens)

        strict_ok = bool(parse_report["strict"]["ok"])
        lenient_ok = bool(parse_report["lenient"]["ok"])
        pred_struct = parse_report["strict"]["prediction"] if strict_ok else parse_report["lenient"]["prediction"]
        if pred_struct is None:
            pred_struct = adapter.empty_prediction()

        gt_struct = adapter.build_gt_struct_from_record(record)
        local_counts = adapter.score_prediction(
            gt_struct,
            pred_struct,
            bbox_iou_threshold=float(args.bbox_iou_threshold),
            strict_point_distance_px=8.0,
        )

        counts["samples"] += 1.0
        if lenient_ok:
            counts["parse_success_lenient"] += 1.0
        if strict_ok:
            counts["parse_success_strict"] += 1.0
        counts["bbox_tp"] += float(local_counts.get("bbox_tp", 0.0))
        counts["bbox_fp"] += float(local_counts.get("bbox_fp", 0.0))
        counts["bbox_fn"] += float(local_counts.get("bbox_fn", 0.0))
        counts["bbox_iou_sum"] += float(local_counts.get("bbox_iou_sum", 0.0))

        generation = dict(parse_report.get("generation", {}))
        counts["generated_tokens"] += float(generation.get("generated_tokens", 0.0))
        counts["returned_tokens"] += float(generation.get("returned_tokens", 0.0))
        counts["hit_max_new_tokens"] += 1.0 if bool(generation.get("hit_max_new_tokens", False)) else 0.0
        counts["closed_json_array"] += 1.0 if bool(generation.get("closed_json_array", False)) else 0.0

        row = {
            "sample_id": record.get("sample_id"),
            "source_sample_id": record.get("source_sample_id"),
            "source_type": record.get("source_type"),
            "image_path": str(image_path),
            "task_type": task_type,
            "domain_type": domain_type,
            "parse_lenient_ok": lenient_ok,
            "parse_strict_ok": strict_ok,
            "parse_lenient_error": parse_report["lenient"].get("error"),
            "parse_strict_error": parse_report["strict"].get("error"),
            "generation": generation,
            "gt_instances": int(len(gt_struct.get("instances", []))),
            "pred_instances": int(len(pred_struct.get("instances", []))),
            "bbox_tp": float(local_counts.get("bbox_tp", 0.0)),
            "bbox_fp": float(local_counts.get("bbox_fp", 0.0)),
            "bbox_fn": float(local_counts.get("bbox_fn", 0.0)),
            "bbox_iou_sum": float(local_counts.get("bbox_iou_sum", 0.0)),
            "raw_text": raw_text,
            "prediction": pred_struct,
        }
        if args.save_per_sample:
            per_sample_rows.append(row)

        if not lenient_ok:
            parse_badcases.append(row)

        error_score = float(local_counts.get("bbox_fp", 0.0) + local_counts.get("bbox_fn", 0.0))
        metric_badcases.append({**row, "metric_error_score": error_score})

        if progress is not None:
            samples = max(counts["samples"], 1.0)
            progress.set_postfix(
                {
                    "parseS": f"{counts['parse_success_strict'] / samples:.2f}",
                    "p": f"{counts['bbox_tp'] / max(counts['bbox_tp'] + counts['bbox_fp'], 1.0):.2f}",
                    "r": f"{counts['bbox_tp'] / max(counts['bbox_tp'] + counts['bbox_fn'], 1.0):.2f}",
                }
            )
            progress.update(1)

    if progress is not None:
        progress.close()

    summary = _summarize(counts, iou_threshold=float(args.bbox_iou_threshold))
    summary_payload = {
        "jsonl": str(jsonl_path),
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "config": str(args.config),
        "iou_threshold": float(args.bbox_iou_threshold),
        "metrics": summary,
    }
    write_json(output_dir / "summary.json", summary_payload)

    if args.save_per_sample:
        write_jsonl(output_dir / "per_sample.jsonl", per_sample_rows)

    write_jsonl(output_dir / "badcases_parse.jsonl", parse_badcases)
    metric_badcases.sort(
        key=lambda row: (
            -float(row.get("metric_error_score", 0.0)),
            -float(row.get("bbox_fn", 0.0)),
            -float(row.get("bbox_fp", 0.0)),
            str(row.get("sample_id", "")),
        )
    )
    topk = max(int(args.save_badcases_topk), 0)
    write_jsonl(output_dir / "badcases_metric.jsonl", metric_badcases[:topk] if topk > 0 else [])

    print("[summary]")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print(f"Saved summary to: {output_dir / 'summary.json'}")
    if args.save_per_sample:
        print(f"Saved per-sample records to: {output_dir / 'per_sample.jsonl'}")
    print(f"Saved parse badcases to: {output_dir / 'badcases_parse.jsonl'}")
    print(f"Saved metric badcases to: {output_dir / 'badcases_metric.jsonl'}")


if __name__ == "__main__":
    main()
