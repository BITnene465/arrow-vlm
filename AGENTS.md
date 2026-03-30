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
  - `configs/train/train_full_ft.yaml`
  - `configs/train/train_full_ft_4b.yaml`
  - `configs/train/train_lora.yaml`
  - `configs/train/train_lora_4b.yaml`
  - `configs/train/train_sync_posttrain.yaml`

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
- 推理配置走独立 infer config，不复用训练 YAML
- one-stage infer config：`configs/infer/infer_one_stage.yaml`
- two-stage infer config：`configs/infer/infer_two_stage.yaml`
- demo 现在支持切换：
  - base model
  - checkpoint

## 两阶段实验入口

- Stage1 数据准备：`scripts/prepare_stage1_data.py`
- Stage2 数据准备：`scripts/prepare_stage2_data.py`
- Stage 1 训练配置：
  - `configs/train/train_stage1_lora.yaml`
  - `configs/train/train_stage1_lora_4b.yaml`
- Stage 2 训练配置：
  - `configs/train/train_stage2_lora.yaml`
  - `configs/train/train_stage2_lora_4b.yaml`

当前两阶段约定：

- Stage 1：整图输出 `label + bbox`
- Stage 1 grounding prompt 采用 Qwen3-VL 官方 grounding 风格：短自然语言指令 + 相对坐标 `[0,999]`
- Stage 2：输入单目标 crop，并通过 crop-local `label + bbox_2d` 指定 main arrow，输出其 `keypoints_2d` 骨架
- 推理阶段的 Stage 1 现在默认采用 mixed proposals：
  - full image
  - 按比例生成的多尺度 tile
  - 之后按 `label + IoU` 做 proposal dedup
- CLI / demo 都提供 Stage1 mixed proposal 开关，默认开启
- Stage 2 的 target/prompt 坐标都必须是 crop-local `[0,999]`
- Stage 2 prompt 采用配置模板渲染；当前 prompt 会显式注入 crop-local `label + bbox_2d`，数据记录仍保留 `condition` 以兼容现有链路
- `demo_two_stage` 可在不提供 Stage 2 checkpoint 的情况下直接做 Stage 1 可视化检查
- Stage 2 crop 默认 `padding_ratio = 0.3`
- crop 超出原图边界时，黑边补齐
- Stage1 现在支持三条线并行：
  - 原始整图样本
  - 按图像短边比例生成的多尺度滑窗样本
  - 按箭头数量区间筛选的 density crop 样本
- Stage1 的 crop 尺寸现在按图像短边比例计算，并用 `stage1_min_tile_size` / `stage1_max_tile_size` 做像素上下限兜底
- Stage1 tile 中的实例必须满足 `bbox` 被 tile **完整包含**；如果某个 tile 与任意 bbox 只是部分相交，该 tile 直接丢弃
- Stage1 数据准备会自动去掉“实例集合相同且 crop 高度重叠”的近重复样本，默认用 `stage1_dedup_iou_threshold=0.9`

## 当前 LoRA 语义

- `lang_target_modules`：语言塔 LoRA 挂载点
- `vis_target_modules`：视觉塔 LoRA 挂载点
- `proj_target_modules`：projector LoRA 挂载点

当前约定：

- `freeze_vision_tower: false` 且 `finetune.mode=lora` 时，不是视觉塔全量训练，而是给视觉塔匹配到的线性层挂 LoRA
- 视觉塔当前默认 LoRA 挂点是：
  - `attn.qkv`
  - `attn.proj`
  - `mlp.linear_fc1`
  - `mlp.linear_fc2`
- `train_projector: true` 且 `finetune.mode=lora` 时，不是 projector 全量训练，而是给 projector 匹配到的线性层挂 LoRA
- `proj_target_modules: []` 表示 projector 下所有匹配到的线性层都可作为 LoRA 挂点

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
- 用户对一致性要求很高：
  - 命名要统一
  - 文档要同步
  - 不要遗留半旧半新的逻辑
- 用户不希望 agent 擅自启动长训练；涉及训练时，应先说明命令或在明确要求下再执行
