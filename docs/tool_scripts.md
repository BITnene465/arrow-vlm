# Tool Scripts

本文件统一维护非主线辅助脚本的使用说明：

- 推理（infer）
- Demo
- 离线可复盘评估（eval）

主线训练入口仍然是 `scripts/train.py`，训练相关说明请看主 README。

## 1. One-Stage 推理

脚本：

- `scripts/arrow/infer.py`

单图推理示例：

```bash
python scripts/arrow/infer.py \
  --config configs/infer/infer_one_stage.yaml \
  --checkpoint outputs/your_experiment/checkpoints/best \
  --image /path/to/figure.jpg
```

目录批量推理示例：

```bash
python scripts/arrow/infer.py \
  --config configs/infer/infer_one_stage.yaml \
  --checkpoint outputs/your_experiment/checkpoints/best \
  --image-dir /path/to/images \
  --recursive \
  --output-dir outputs/infer_one_stage_batch
```

常用覆盖参数：

- `--max-new-tokens`
- `--model`
- `--device`
- `--env-file`

输出产物（`--output-dir`）：

- `reports/*.one_stage.json`
- `raw/*.raw.txt`
- `manifest.json`（目录模式）

## 2. Two-Stage 推理

脚本：

- `scripts/arrow/infer_two_stage.py`

完整两阶段推理示例：

```bash
python scripts/arrow/infer_two_stage.py \
  --config configs/infer/infer_two_stage.yaml \
  --stage1-checkpoint outputs/qwen3vl-s1-lora/4b/checkpoints/best \
  --stage2-checkpoint outputs/qwen3vl-s2-lora/4b/checkpoints/best \
  --image /path/to/example.png \
  --output-dir outputs/two_stage_demo
```

仅 Stage1 检查模式（不传 Stage2 checkpoint）：

```bash
python scripts/arrow/infer_two_stage.py \
  --config configs/infer/infer_two_stage.yaml \
  --stage1-checkpoint outputs/qwen3vl-s1-lora/4b/checkpoints/best \
  --image /path/to/example.png \
  --output-dir outputs/two_stage_stage1_only
```

常用覆盖参数：

- `--stage1-max-new-tokens`
- `--stage2-max-new-tokens`
- `--stage2-batch-size`
- `--stage1-mixed-proposals / --no-stage1-mixed-proposals`
- `--stage1-model`
- `--stage2-model`
- `--device`
- `--env-file`

输出产物（`--output-dir`）：

- `reports/*.two_stage.json`
- `stage1_overlay/*.png`
- `final_overlay/*.png`
- `manifest.json`（目录模式）

## 3. Demo

One-stage Demo：

- `app/demo.py`

```bash
python app/demo.py \
  --config configs/infer/infer_one_stage.yaml \
  --checkpoint outputs/your_experiment/checkpoints/best
```

Two-stage Demo：

- `app/demo_two_stage.py`

```bash
python app/demo_two_stage.py \
  --config configs/infer/infer_two_stage.yaml \
  --stage1-checkpoint outputs/qwen3vl-s1-lora/4b/checkpoints/best \
  --stage2-checkpoint outputs/qwen3vl-s2-lora/4b/checkpoints/best
```

## 4. Stage1 Grounding 离线可复盘评估

脚本：

- `scripts/arrow/eval_stage1_grounding.py`

用途：

- 在指定 JSONL 上离线评估 Stage1 grounding checkpoint
- 产出可复盘的逐样本结果和 badcase 文件

示例：

```bash
python scripts/arrow/eval_stage1_grounding.py \
  --config configs/infer/infer_one_stage.yaml \
  --checkpoint outputs/qwen3vl-s1-lora/4b/checkpoints/best \
  --jsonl data/two_stage/stage1/val.jsonl \
  --output-dir outputs/eval/stage1_grounding/run_001
```

只给 JSONL 时，图片定位规则：

- `image_path` 为绝对路径：直接使用
- `image_path` 为相对路径：先按工作目录解析，再按 JSONL 所在目录解析

关键参数：

- `--bbox-iou-threshold`
- `--max-samples`
- `--max-new-tokens`
- `--save-per-sample / --no-save-per-sample`
- `--save-badcases-topk`

输出产物：

- `summary.json`
- `per_sample.jsonl`
- `badcases_parse.jsonl`
- `badcases_metric.jsonl`

## 5. Stage2 Keypoints 离线可复盘评估

脚本：

- `scripts/arrow/eval_stage2_keypoints.py`

用途：

- 在指定 Stage2 JSONL 上离线评估 `keypoint_sequence` checkpoint
- 产出可复盘的逐样本结果和 badcase 文件

示例：

```bash
python scripts/arrow/eval_stage2_keypoints.py \
  --config configs/infer/infer_one_stage.yaml \
  --checkpoint outputs/qwen3vl-s2-lora/4b/checkpoints/best \
  --jsonl data/two_stage/stage2/val.jsonl \
  --output-dir outputs/eval/stage2_keypoints/run_001
```

只给 JSONL 时，图片定位规则：

- `image_path` 为绝对路径：直接使用
- `image_path` 为相对路径：先按工作目录解析，再按 JSONL 所在目录解析

关键参数：

- `--strict-point-distance-px`
- `--max-samples`
- `--max-new-tokens`
- `--save-per-sample / --no-save-per-sample`
- `--save-badcases-topk`

输出产物：

- `summary.json`
- `per_sample.jsonl`
- `badcases_parse.jsonl`
- `badcases_metric.jsonl`

## 6. 配置说明

推理配置与训练 YAML 分离：

- `configs/infer/infer_one_stage.yaml`
- `configs/infer/infer_two_stage.yaml`

checkpoint 路径和模型覆盖参数都通过 CLI 传入。
