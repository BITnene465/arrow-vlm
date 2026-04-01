# Refactor Relaunch

这份文档对应当前 `refactor-task-domain-routing` 分支上的新结构：

- `core`
- `tasks`
- `domains`

以及新的显式路由字段：

- `task_type`
- `domain_type`

当前代码已经**不兼容旧数据和旧 checkpoint**。如果要继续训练或推理，必须按新结构重新准备数据，再重新训练。

---

## 1. 环境同步

```bash
uv sync
```

如果你使用的是已有 `.venv`，也建议重新激活一次环境。

---

## 2. 重新准备 one-stage 基础数据

这一步会生成新的 `data/processed/train.jsonl` 和 `data/processed/val.jsonl`，并写入：

- `task_type`
- `domain_type`

命令：

```bash
python scripts/arrow/prepare_data.py \
  --raw-json-dir data/raw/json \
  --image-dir data/raw/figure \
  --output-dir data/processed
```

生成后的 one-stage 数据语义：

- `task_type = joint_structure`
- `domain_type = arrow`

---

## 3. 重新准备 two-stage 数据

### 3.1 Stage1 grounding 数据

命令：

```bash
python scripts/arrow/prepare_stage1_data.py \
  --input-dir data/processed \
  --output-dir data/two_stage \
  --num-workers 16 \
  --stage1-include-full-image \
  --stage1-tile-size-ratios 0.1,0.2,0.35,0.5,0.7 \
  --stage1-min-tile-size 256 \
  --stage1-max-tile-size 1024 \
  --stage1-density-min-instances 2 \
  --stage1-density-max-instances 30 \
  --stage1-dedup-iou-threshold 0.65
```

生成后的 Stage1 数据语义：

- `task_type = grounding`
- `domain_type = arrow`

### 3.2 Stage2 keypoint_sequence 数据

命令：

```bash
python scripts/arrow/prepare_stage2_data.py \
  --input-dir data/processed \
  --output-dir data/two_stage \
  --padding-ratio 0.3 \
  --num-workers 8 \
  --stage2-aug-ratio 0.0
```

生成后的 Stage2 数据语义：

- `task_type = keypoint_sequence`
- `domain_type = arrow`

---

## 4. 训练命令

### 4.1 one-stage joint_structure

4B:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
  --config configs/train/train_lora_4b.yaml \
  --run-id relaunch-joint-4b
```

2B:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
  --config configs/train/train_lora.yaml \
  --run-id relaunch-joint-2b
```

### 4.2 Stage1 grounding

4B:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
  --config configs/train/train_stage1_lora_4b.yaml \
  --run-id relaunch-stage1-4b
```

2B:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
  --config configs/train/train_stage1_lora.yaml \
  --run-id relaunch-stage1-2b
```

### 4.3 Stage2 keypoint_sequence

4B:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
  --config configs/train/train_stage2_lora_4b.yaml \
  --run-id relaunch-stage2-4b
```

2B:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
  --config configs/train/train_stage2_lora.yaml \
  --run-id relaunch-stage2-2b
```

---

## 5. 推理 smoke

### 5.1 one-stage

```bash
python scripts/arrow/infer.py \
  --config configs/infer/infer_one_stage.yaml \
  --checkpoint outputs/qwen3vl-lora/4b/relaunch-joint-4b/checkpoints/best \
  --image data/val_images/00008.jpg
```

### 5.2 two-stage

```bash
python scripts/arrow/infer_two_stage.py \
  --config configs/infer/infer_two_stage.yaml \
  --stage1-checkpoint outputs/qwen3vl-s1-lora/4b/relaunch-stage1-4b/checkpoints/best \
  --stage2-checkpoint outputs/qwen3vl-s2-lora/4b/relaunch-stage2-4b/checkpoints/best \
  --stage1-model models/Qwen3-VL-4B-Instruct \
  --stage2-model models/Qwen3-VL-4B-Instruct \
  --image data/val_images/00008.jpg
```

如果只检查 Stage1：

```bash
python scripts/arrow/infer_two_stage.py \
  --config configs/infer/infer_two_stage.yaml \
  --stage1-checkpoint outputs/qwen3vl-s1-lora/4b/relaunch-stage1-4b/checkpoints/best \
  --stage1-model models/Qwen3-VL-4B-Instruct \
  --image data/val_images/00008.jpg
```

---

## 6. Demo 启动

one-stage:

```bash
python app/demo.py --config configs/infer/infer_one_stage.yaml
```

two-stage:

```bash
python app/demo_two_stage.py --config configs/infer/infer_two_stage.yaml
```

当前 demo 启动前会自动移除 SOCKS 代理环境变量，避免 `gradio/httpx` 因缺少 `socksio` 导入失败。

---

## 7. 重新起跑前的硬性判断

出现下面任一情况，都说明你还在混用旧产物：

1. 数据 record 里没有：
   - `task_type`
   - `domain_type`
2. Stage1 数据里仍出现：
   - `task_type = two_stage_stage1_grounding`
3. Stage2 数据里仍出现：
   - `task_type = two_stage_stage2`
4. 训练脚本报：
   - `Dataset record is missing a valid task/domain route.`

这时不要调代码，直接重做对应数据。

---

## 8. 当前推荐的最小验证顺序

1. `uv sync`
2. `prepare_data.py`
3. `prepare_stage1_data.py`
4. `prepare_stage2_data.py`
5. 先训 `Stage1 grounding`
6. 再训 `Stage2 keypoint_sequence`
7. 最后跑 `infer_two_stage.py` 做两阶段验证

这样最容易定位问题，不会把 one-stage、stage1、stage2 三条线混在一起。
