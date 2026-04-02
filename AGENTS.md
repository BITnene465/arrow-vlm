# AGENTS

## 项目现状

- 当前开发分支是 `refactor-task-domain-routing`。
- 当前 Python 包名是 `vlm_structgen`，不再使用旧的 `vlm_det`。
- 当前仓库已经完成 `core / tasks / domains` 三层重构：
  - `core`
    - 通用训练 / 推理 / 评估 / 数据读取框架
  - `tasks`
    - `grounding`
    - `keypoint_sequence`
    - `joint_structure`
  - `domains`
    - 当前正式 domain 是 `arrow`
- 当前项目目标仍然是训练基于 `Qwen3-VL` 的箭头理解模型。

当前主任务输出仍然围绕：

- `label`
- `bbox_2d`
- `keypoints_2d`

但仓库的主抽象已经不是“单一箭头任务仓库”，而是：

- 多模态生成式结构预测框架
- 在 `arrow` domain 下实现三个正式 task

## 当前路由约定

所有标准训练 / 验证 JSONL 记录都必须显式包含：

- `task_type`
- `domain_type`

当前支持的 task：

- `joint_structure`
- `grounding`
- `keypoint_sequence`

当前正式 domain：

- `arrow`

不要让 dataset 或 infer 去猜 task/domain。

## 当前正式数据格式

标准格式文档在：

- `docs/standard_data_format.md`

关键原则：

- 仓库内部只长期维护**标准 JSONL 格式**
- 各种外部原始格式和一次性中间产物，应由二次开发人员在仓库外先转换，再交给当前源码

当前三类正式数据：

### one-stage joint_structure

- 路径：
  - `data/processed/train.jsonl`
  - `data/processed/val.jsonl`
- 语义：
  - 整图直接输出 `label + bbox + keypoints`

### Stage1 grounding

- 路径：
  - `data/two_stage/stage1/train.jsonl`
  - `data/two_stage/stage1/val.jsonl`
- 语义：
  - 整图 / tile 输出 `label + bbox`

### Stage2 keypoint_sequence

- 路径：
  - `data/two_stage/stage2/train.jsonl`
  - `data/two_stage/stage2/val.jsonl`
- 语义：
  - 单目标 crop + crop-local `label + bbox_2d`
  - 输出 `{"keypoints_2d":[...]}` 

## 当前协议

### one-stage / joint_structure 输出格式

```json
[
  {
    "label": "single_arrow | double_arrow",
    "bbox_2d": [x1, y1, x2, y2],
    "keypoints_2d": [[x0, y0], [x1, y1], ...]
  }
]
```

### Stage1 / grounding 输出格式

```json
[
  {
    "label": "single_arrow | double_arrow",
    "bbox_2d": [x1, y1, x2, y2]
  }
]
```

### Stage2 / keypoint_sequence 输出格式

```json
{
  "keypoints_2d": [[x0, y0], [x1, y1], ...]
}
```

通用约束：

- 只输出 JSON，不输出 markdown，不输出额外自然语言
- 所有量化坐标都在 `[0, 999]`
- `label` 只能是：
  - `single_arrow`
  - `double_arrow`
- `single_arrow` 的点顺序固定为：
  - `tail -> ... -> head`
- `double_arrow` 的点顺序固定为：
  - 左上 head 在前
  - 另一个 head 在后

整图 canonical sort key 仍然是：

- `(y1, x1, y2, x2, y_tail, x_tail, y_head, x_head, n_points)`

实例顺序必须在数据准备阶段固化，dataset 只忠实读取。

## 数据准备入口

### 真实数据

- 入口：
  - `scripts/arrow/prepare_data.py`

标准命令：

```bash
python scripts/arrow/prepare_data.py \
  --raw-json-dir data/raw/json \
  --image-dir data/raw/figure \
  --output-dir data/processed
```

当前 LabelMe 映射：

- `c0~c3 -> single_arrow`
- `c4~c7 -> double_arrow`

### 两阶段数据

- Stage1：
  - `scripts/arrow/prepare_stage1_data.py`
- Stage2：
  - `scripts/arrow/prepare_stage2_data.py`

Stage1 当前约定：

- 原始整图样本
- 按图像短边比例生成的多尺度滑窗样本
- density crop 样本
- tile 中实例必须满足 `bbox` 被 tile **完整包含**
- 与任意 bbox 部分相交的 tile 直接丢弃
- 去重阈值：
  - `stage1_dedup_iou_threshold`

Stage2 当前约定：

- 输入单目标 crop
- prompt 显式注入 crop-local：
  - `label`
  - `bbox_2d`
- 默认 `padding_ratio = 0.3`
- crop 超界黑边补齐

注意：

- Stage1 的整图样本也会复制到：
  - `data/two_stage/stage1/images/<split>/`
- 当前训练不会再回原始目录找图

## 当前训练入口

### 主训练入口

- `scripts/train.py`

### 当前主要配置

- one-stage：
  - `configs/train/train_lora.yaml`
  - `configs/train/train_lora_4b.yaml`
  - `configs/train/train_full_ft.yaml`
  - `configs/train/train_full_ft_4b.yaml`
- Stage1：
  - `configs/train/train_stage1_lora.yaml`
  - `configs/train/train_stage1_lora_4b.yaml`
- Stage2：
  - `configs/train/train_stage2_lora.yaml`
  - `configs/train/train_stage2_lora_4b.yaml`

## 辅助脚本文档

infer / demo / eval 属于辅助脚本，不在本文件维护使用说明。

- 统一文档：
  - `docs/tool_scripts.md`

## 当前两阶段语义

### Stage1

- task：
  - `grounding`
- 输出：
  - `label + bbox`
- prompt 采用 Qwen3-VL grounding 风格
- 推理默认启用 mixed proposals：
  - full image
  - ratio-based multi-scale tiles
  - 然后做 proposal dedup

### Stage2

- task：
  - `keypoint_sequence`
- 输入：
  - 单目标 crop
  - crop-local `label + bbox_2d`
- 输出：
  - `{"keypoints_2d":[...]}`

当前 prompt 语义是：

- target arrow 由 crop-local `label + bbox_2d` 指定
- 即使 crop 内还有其他箭头，也只输出该 target arrow 的骨架

## 当前 LoRA 语义

- `lang_target_modules`
  - 语言塔 LoRA 挂载点
- `vis_target_modules`
  - 视觉塔 LoRA 挂载点
- `proj_target_modules`
  - projector LoRA 挂载点

当前约定：

- `freeze_vision_tower: false` 且 `finetune.mode=lora`
  - 不是视觉塔全量训练
  - 是给视觉塔匹配到的线性层挂 LoRA
- `train_projector: true` 且 `finetune.mode=lora`
  - 不是 projector 全量训练
  - 是给 projector 匹配到的线性层挂 LoRA

当前默认视觉塔 LoRA 挂点：

- `attn.qkv`
- `attn.proj`
- `mlp.linear_fc1`
- `mlp.linear_fc2`

当前 projector LoRA 规则：

- `proj_target_modules: []`
  - 表示 projector 下所有匹配到的线性层都可挂 LoRA

## 当前 loss 约定

当前已经正式支持 **结构化 weighted token loss**，但必须通过 codec 提供结构化 `loss_meta`，不能在 task 层从 `target_text` 反解析字段。

### Stage1 grounding

当前默认启用：

- `bbox_token_loss_weight: 2.0`
- `label_token_loss_weight: 1.5`

实现原则：

- `GroundingCodec` 在序列化阶段返回：
  - `target_text`
  - `loss_meta.field_char_spans`
- task adapter 只消费 span 做 weighted CE
- trainer 不理解 `label / bbox_2d`

### Stage2 keypoint_sequence

当前默认启用：

- `coordinate_token_loss_weight: 1.5`

实现原则：

- `KeypointSequenceCodec` 在序列化阶段返回坐标字符跨度
- task adapter 只对坐标 token 加权
- 不从 JSON 文本 regex/反解析坐标

如果以后新增 task-specific loss：

- 优先扩展 task adapter / codec 接口
- 不要把解析逻辑塞进 trainer

## 当前学习率约定

当前 LoRA 配置已经收成保守版：

- `learning_rate = 5e-5`
- `embed_learning_rate = 5e-5`
- `lm_head_learning_rate = 5e-5`
- `lora_learning_rate = 1e-4`

当前 Stage1 4B 配置：

- `per_device_batch_size = 4`
- `grad_accum_steps = 2`

当前 Stage2 4B 配置：

- `per_device_batch_size = 24`
- `grad_accum_steps = 1`

## 当前经验结论

- `Qwen3-VL 2B` 和 `4B` 在当前箭头任务上，都表现出：
  - 解冻视觉塔 LoRA 优于冻结视觉塔
- one-stage `joint_structure` 仍然更容易受长结构化生成失稳影响
- 两阶段当前更稳：
  - Stage1 做 grounding
  - Stage2 做 keypoint_sequence
- 这类有序 JSON 监督任务，canonical order 是硬约束
- weighted token loss 应通过结构化 `loss_meta` 实现，不要写文本级 hack

## 当前训练限制

- 当前版本训练链路按 batch 路由单一 `task_type/domain_type`，不支持多任务混训。
- 如果将多种 task/domain 样本混在同一次训练中，训练会在 batch 路由检查处失败。
- 现阶段请保持：
  - 一次训练只使用单一 task/domain 数据集
  - 配置中的 `task.route_options` 只配置当前训练 route
- 混合任务训练（multi-task mixed training）属于后续能力，不在当前正式支持范围内。

更细的实验笔记仍在：

- `lab_notes/canonical-order.md`
- `lab_notes/qwen3vl-runtime-state.md`
- `lab_notes/findings.md`

## 对后续 agent 的要求

- 不要把这个项目当作普通 OCR / 检测项目处理
- 当前核心难点仍然是：
  - 视觉 grounding
  - 结构化 JSON 生成
- 优先检查：
  - 训练目标
  - 数据顺序
  - task/domain 路由
  - 生成停止行为
  - parse 指标口径
- 如果修改以下任一内容，必须同步更新：
  - 数据准备
  - codec
  - evaluator
  - infer/demo
  - configs
  - docs

## 用户使用 AI 的风格

- 用户偏好直接、务实、少废话的工程沟通
- 用户通常希望先改代码，再汇报
- 用户对一致性要求很高：
  - 命名要统一
  - 文档要同步
  - 不要遗留半旧半新的逻辑
- 用户不希望 agent 擅自启动长训练
  - 涉及训练时，应先给命令，或在明确要求下再执行
