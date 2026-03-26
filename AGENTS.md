# AGENTS

## 项目现状

- 当前主分支是 `main`。
- 项目目标是训练一个基于 `Qwen3-VL` 的箭头理解模型，输入图像，输出每个箭头的：
  - `label`
  - `bbox_2d`
  - `keypoints_2d`
- 当前协议已经稳定为 JSON 数字协议，不再使用旧 DSL / special tokens 路线。

## 当前协议

输出格式：

```json
[
  {
    "label": "single_arrow | double_arrow",
    "bbox_2d": [x1, y1, x2, y2],
    "keypoints_2d": [[x0, y0], [x1, y1], ...]
  }
]
```

关键约束：

- 只输出 JSON array，不输出 markdown，不输出额外自然语言
- 所有坐标量化到 `[0, 999]`
- `label` 只能是 `single_arrow` 或 `double_arrow`
- `single_arrow` 的点顺序是 `tail -> ... -> head`
- `double_arrow` 的点顺序是“左上的 head 为起点，另一个 head 为终点”
- 图内 instance 顺序必须固定，当前 canonical sort key 为：
  - `(y1, x1, y2, x2, y_tail, x_tail, y_head, x_head, n_points)`

## 数据管线

- 真实数据准备入口：`scripts/prepare_data.py`
- 标准命令：

```bash
python scripts/prepare_data.py \
  --raw-json-dir data/raw/json \
  --image-dir data/raw/figure \
  --output-dir data/processed
```

- `data/processed/train.jsonl` 和 `data/processed/val.jsonl` 中保存的是像素坐标
- 坐标归一化到 `[0,999]` 发生在 `ArrowCodec.encode()`
- canonical order 必须在数据准备 / synthetic 导出阶段确定，`dataset.py` 只忠实读取 JSONL
- LabelMe 框类别映射：
  - `c0~c3 -> single_arrow`
  - `c4~c7 -> double_arrow`

## 训练入口

- 单阶段训练：`scripts/train.py`
- 两阶段训练：`scripts/train_two_stage.py`
- 当前主要配置：
  - `configs/train_full_ft.yaml`
  - `configs/train_full_ft_4b.yaml`
  - `configs/train_lora.yaml`
  - `configs/train_lora_4b.yaml`
  - `configs/train_sync_posttrain.yaml`

默认命名规则：

- `qwen3vl-ft-*`
- `qwen3vl-lora-*`
- `qwen3vl-post-*`

并自动带模型规模标签，例如：

- `qwen3vl-ft-2b`
- `qwen3vl-ft-4b`

## 推理入口

- CLI：`scripts/infer.py`
- 两阶段 CLI：`scripts/infer_two_stage.py`
- Demo：`app/demo.py`
- 两阶段 Demo：`app/demo_two_stage.py`
- 推理配置走 `.env` / 环境变量，不走训练 YAML
- demo 现在支持切换：
  - base model
  - checkpoint

## 两阶段实验入口

- 两阶段数据准备：`scripts/prepare_two_stage_data.py`
- Stage 1 训练配置：
  - `configs/train_stage1_lora.yaml`
  - `configs/train_stage1_lora_4b.yaml`
- Stage 2 训练配置：
  - `configs/train_stage2_lora.yaml`
  - `configs/train_stage2_lora_4b.yaml`

当前两阶段约定：

- Stage 1：整图输出 `label + bbox + 2-point keypoints`
- Stage 2：输入单目标 crop 和 crop-local 文本 hint，输出该目标箭头完整点列
- Stage 2 的 target/prompt 坐标都必须是 crop-local `[0,999]`
- Stage 2 crop 默认 `padding_ratio = 0.5`
- crop 超出原图边界时，黑边补齐

## 当前经验结论

- 在当前箭头任务上，`Qwen3-VL 2B` 和 `4B` 的训练效果都表现为解冻视觉塔优于冻结视觉塔
- `2B` 和 `4B` 都存在长输出末尾不闭合并进入重复生成的问题
- `full FT + 解冻视觉塔` 已经优于冻结视觉塔，但没有彻底解决长结构化生成失稳问题
- 这类有序 JSON 监督任务，必须规定 canonical order

更细的积累笔记放在：

- `lab_notes/canonical-order.md`
- `lab_notes/qwen3vl-runtime-state.md`
- `lab_notes/findings.md`

## 对后续 agent 的要求

- 不要把这个项目当作普通 OCR / 检测项目处理，它的核心难点是“视觉 grounding + 结构化 JSON 生成”
- 优先检查训练目标、数据顺序、生成停止行为、parse 指标口径，不要先陷入无效超参搜索
- 如果修改协议、数据顺序或推理停止逻辑，必须同步更新：
  - 数据准备
  - synthetic 导出
  - codec
  - evaluator
  - infer/demo
  - 文档

## 用户使用 AI 的风格

- 用户偏好直接、务实、少废话的工程沟通
- 用户通常希望先改代码、再汇报，不喜欢空谈方案
- 用户对一致性要求很高：
  - 命名要统一
  - 文档要同步
  - 不要遗留半旧半新的逻辑
- 用户希望沉淀真正可复用的“炼丹经验”，不希望保留只属于一次失误的噪声记录
- 用户不希望 agent 擅自启动长训练；涉及训练时，应先说明命令或在明确要求下再执行
