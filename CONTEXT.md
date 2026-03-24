# 项目上下文

最后更新：2026-03-21

## 任务目标

- 在论文图中检测 `arrow`。
- 对每个 `arrow` 输出一个检测框和一组有序关键点。
- 当前阶段不使用 `arrow_type`（原始标注中的 `c0~c7`）。
- 计划使用 `Qwen3-VL-2B-Instruct` 作为基础 VLM 检查点进行微调。

## 数据位置

- 原始图像目录：`data/raw/raw/figure`
- 原始标注目录：`data/raw/raw/json`

## 原始标注语义

- 标注为 LabelMe 风格 JSON。
- 一个实例通过 `group_id` 关联：
  - 1 个矩形框
  - 1 组关键点
- 框和关键点属于同一个 `group_id` 时，表示同一个 `arrow` 实例。

## Arrow 关键点定义

- 关键点是有序序列，顺序有语义。
- 第 1 个点是起点。
- 最后 1 个点是终点，即箭头尖端（arrow head）。
- 中间所有点都是拐点。
- 关键点数量是变长的，不固定。

## 可见性定义

- `P2`：关键点可见。
- `P1`：关键点被其他元素遮挡。
- 遮挡点的坐标是人工估计的大致位置，不是严格可观测位置。
- 因此关键点最终应建模为 `[x, y, v]`，其中 `v` 至少包含：
  - `2`：可见
  - `1`：被遮挡

## 箭头类别补充

- `c0~c3`：单向箭头
- `c4~c7`：双向箭头
- 但当前阶段这些子类信息不进入训练目标。

## 最终输入输出协议

### 输入

- 使用任务 special token：`<|arrow_task|>`
- 输入为单张图像 + 固定任务指令。
- 模型目标是输出图中所有 `arrow` 的结构化结果。

### 输出总体原则

- 不使用 JSON 作为训练主协议。
- 不直接输出裸数值，避免数值漂移。
- 所有坐标均通过离散位置 token 表达。
- 负样本不使用 `<|no_arrow|>`，空结果直接表示为空集合。

### 坐标量化

- 坐标基于原图尺寸归一化。
- 使用 `B = 2048` 个离散 bin。
- 量化公式：

```text
qx = round(x / (W - 1) * 2047)
qy = round(y / (H - 1) * 2047)
```

- 反量化公式：

```text
x = qx / 2047 * (W - 1)
y = qy / 2047 * (H - 1)
```

### Special Token 设计

结构 token：

- `<|arrow_task|>`
- `<|arrows_begin|>`
- `<|arrows_end|>`
- `<|arrow_begin|>`
- `<|arrow_end|>`
- `<|box_begin|>`
- `<|box_end|>`
- `<|points_begin|>`
- `<|points_end|>`
- `<|point_begin|>`
- `<|point_end|>`
- `<|visible|>`
- `<|occluded|>`

位置 token：

- `<|x_0000|>` 到 `<|x_2047|>`
- `<|y_0000|>` 到 `<|y_2047|>`

### 最终输出格式

有箭头时：

```text
<|arrows_begin|>
<|arrow_begin|>
<|box_begin|> <|x_0120|> <|y_0344|> <|x_0276|> <|y_0601|> <|box_end|>
<|points_begin|>
<|point_begin|> <|x_0130|> <|y_0350|> <|visible|> <|point_end|>
<|point_begin|> <|x_0188|> <|y_0471|> <|occluded|> <|point_end|>
<|point_begin|> <|x_0270|> <|y_0590|> <|visible|> <|point_end|>
<|points_end|>
<|arrow_end|>
<|arrows_end|>
```

无箭头时：

```text
<|arrows_begin|>
<|arrows_end|>
```

### 语义约束

- 一个 `arrow` 实例包含：
  - 一个 `bbox`
  - 一个变长关键点序列
- `bbox` 固定表示为：
  - `x1 y1 x2 y2`
- 每个关键点固定表示为：
  - `x y visibility`
- 关键点顺序语义固定：
  - 第 1 个点是起点
  - 最后 1 个点是终点，即 arrow head
  - 中间点都是拐点
- 不额外输出 `start/end/turn` 类型 token。
- 不额外输出关键点数量 token。

### 可见性语义

- `<|visible|>`：关键点可见
- `<|occluded|>`：关键点被遮挡

### 顺序约束

- 不要求多个 `arrow` 在语义上存在固定顺序。
- 训练时建议随机打乱 `arrow` 实例顺序，参考 Pix2Seq 的集合建模方式。
- 单个 `arrow` 内部关键点顺序必须严格保持标注顺序。

### 训练损失方向

- 第一阶段主损失采用标准自回归 token-level CE。
- 重点先保证协议稳定、可解析和坐标 token 学习稳定。
- 后续可考虑对坐标 token 和遮挡 token 做权重增强。

## Tokenizer 与模型改造提示

### 词表扩展

- 当前协议依赖大量新增 special token。
- 除结构 token 外，还需要扩展位置 token：
  - `<|x_0000|>` 到 `<|x_2047|>`
  - `<|y_0000|>` 到 `<|y_2047|>`
- 因此 tokenizer 词表需要显式扩展。

### 模型输出层影响

- 扩展 tokenizer 后，模型 embedding 必须 resize。
- LM head 输出维度也必须随词表扩展而调整。
- 即使底层实现采用 tied embeddings，也仍应按扩词表流程显式处理 `resize_token_embeddings(...)`。
- 新增 token 对应参数属于冷启动参数，需要训练。

### 训练参数冻结策略提示

- 不能采用“纯 LoRA，embedding 和 lm_head 全冻结”的方案。
- 至少应保证以下部分可训练：
  - 新增 token 的 embedding
  - LM head 中与新增 token 对应的参数
  - LoRA 挂载模块
- 当前阶段更合理的训练思路是：
  - 主干大部分冻结
  - 视觉塔冻结
  - LoRA 可训练
  - `embed_tokens` / `lm_head` 至少部分可训练

### 资源判断补充

- 新增 4k+ 位置 token 本身不会成为主要显存瓶颈。
- 资源压力主要仍来自：
  - 图像 token 数
  - 序列长度
  - activation
- 但扩词表会带来明显的 cold-start 学习问题，因此训练框架设计上需要为 embedding 和 lm_head 预留独立参数组与学习率策略。

## 数据检查结论

- 当前数据集包含 816 个 JSON 标注文件。
- 原始图像文件实际均位于 `figure` 目录中，文件扩展名均为 `.jpg`。
- 部分 JSON 中的 `imagePath` 写成了 `.png`，但按文件 stem 可一一对应到实际 `.jpg` 图像。
- `text` 字段可忽略，当前为空或缺失。

## 数据转换与清洗定稿

### 总体数据流

- 采用三层数据流：
  - `raw -> normalized annotations -> sft targets`
- 不直接从原始 LabelMe 标注生成最终训练 target。
- 必须保留规范化中间层，便于排查原始标注问题、清洗逻辑问题和 token 编码问题。

### 训练粒度

- 采用整图训练。
- 一张图对应一个训练样本。
- 模型输出该图中全部 `arrow` 实例。

## 临时问题记录

- `src/vlm_det/eval/evaluator.py` 在 DDP 多卡评估时，进度条上的 `parse/p/r` 目前只反映本地进程统计，不是全局 reduce 后的真实值。当前单卡训练可忽略，后续若恢复多卡训练再修。
- 后续可以在此基础上扩展混合训练和数据增强，但当前协议以整图训练为基线。

### 数据切分

- 采用图像级随机切分。
- 当前仅划分：
  - `train`
  - `val`
- 不单独设 `test`。
- 切分应固定随机种子，确保可复现。
- 不能按实例切分，避免同图信息泄漏。
- 当前建议默认比例：
  - `train = 90%`
  - `val = 10%`

### 原始数据解析

- 图像目录：`data/raw/raw/figure`
- 标注目录：`data/raw/raw/json`
- 原始标注为 LabelMe 风格。
- 以 `group_id` 作为实例主键聚合 shapes。

### 图像路径修复

- 不信任 JSON 中 `imagePath` 的扩展名。
- 应按 `imagePath.stem` 到真实图像目录中匹配文件。
- 当前已确认 `.png/.jpg` 不一致问题可以通过 stem 完全修复。

### 实例构建规则

- 每个实例必须满足：
  - 恰好 `1` 个 bbox
  - 至少 `2` 个 keypoint
- 同组 keypoint 顺序完全信任 JSON 原始顺序。
- 不做任何自动重排。

### 点序语义

- 第 `1` 个点：起点
- 最后 `1` 个点：终点，即 arrow head
- 中间点：拐点

### 可见性清洗与映射

- `p2` / `2` -> `<|visible|>`
- `p1` / `1` -> `<|occluded|>`

### 裁剪与合法性检查

- 点坐标允许裁剪到图像边界。
- bbox 也允许裁剪到图像边界。
- 裁剪后需再次检查 bbox 是否仍然合法。
- 如果裁剪后 bbox 退化，则丢弃该实例。

### 直接丢弃实例的情况

- `group_id` 下没有 bbox
- `group_id` 下有多个 bbox
- `group_id` 下少于 `2` 个 keypoint
- bbox 非法：
  - `x1 >= x2`
  - `y1 >= y2`
- 裁剪后实例仍不可用

### 明确不做的清洗

- 不自动扩框去包住所有 keypoint
- 不根据几何关系重排点序
- 不将 `arrow_type` 纳入当前训练目标

### 规范化中间层建议保留字段

- `image_path`
- `image_width`
- `image_height`
- `instances`
- `group_id`
- `raw_bbox`
- `bbox`
- `raw_keypoints`
- `keypoints`

### 建议的处理产物

- `data/processed/normalized/train.jsonl`
- `data/processed/normalized/val.jsonl`
- `data/processed/sft/train.jsonl`
- `data/processed/sft/val.jsonl`
- `data/processed/reports/data_cleaning_report.json`
- `data/processed/reports/split_manifest.json`

## 建模方向

- 该任务不是固定拓扑的人体 pose。
- 更接近“单类目标检测 + 变长、有序、带可见性的 arrow skeleton 预测”。
- 如果采用 VLM 微调，优先考虑“图像 -> 结构化结果生成”的方式，而不是固定关键点数的 YOLO-pose 头。
- 当前已经确定使用离散 special token 协议，而不是裸数字或 JSON 主协议。

## 评估偏好

- 用户接受直接沿用相关任务常见的 loss 和 metric 设计思路。
- 评估应覆盖：
  - 检测框质量
  - 关键点位置质量
  - 关键点可见性信息

## 基础模型备注

- 用户指定使用 `Qwen3-VL-2B` 的最新可用版本。
- 截至 2026-03-21，已核实可用官方模型页为：
  - `Qwen/Qwen3-VL-2B-Instruct`
- 参考来源：
  - https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
