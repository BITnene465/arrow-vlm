# AGENT protocol-v2 Handover

本文档写给接手 `protocol-v2-explore` 分支的后续 agent / 开发者。

## 1. 项目目标

本项目使用 `Qwen3-VL-2B-Instruct` 训练一个箭头理解模型，输入图像，输出每个箭头的：

- `bbox_2d`
- `keypoints_2d`

当前这条分支的核心目标是：

- 放弃旧的重 DSL / special-token 协议
- 改成更贴近 Qwen3-VL 官方 grounding 风格的 **JSON 数字协议**
- 坐标统一归一化到整数区间 `[0, 999]`

## 2. 当前协议

当前训练 / eval / infer 使用的目标格式是：

```json
[
  {
    "label": "single_arrow | double_arrow",
    "bbox_2d": [x1, y1, x2, y2],
    "keypoints_2d": [[x0, y0], [xk1, yk1], ..., [xN, yN]]
  }
]
```

约束：

- 只输出 JSON array
- 不输出 markdown
- 不输出额外自然语言
- 所有坐标都是 `[0,999]` 的整数
- `label` 只能是 `single_arrow` 或 `double_arrow`
- 对 `single_arrow`：`keypoints_2d` 顺序固定为 **tail -> head**
- 对 `single_arrow`：`keypoints_2d[0]` = 箭头尾部中心线点
- 对 `single_arrow`：`keypoints_2d[-1]` = 箭头头部尖点
- 对 `double_arrow`：`keypoints_2d[0]` 与 `keypoints_2d[-1]` 是两端头部尖点
- 对 `double_arrow`：顺序固定为左侧 head 在前，右侧 head 在后；若 `x` 相同，则更靠上的点在前
- 中间 keypoints = 路径控制点 / 折点，不是箭头头部两侧角点
- 每个箭头至少 2 个 keypoints

## 3. 为什么有这条分支

旧主线的问题是：

- 使用大量新增 special tokens
- 使用重 DSL（`arrows_begin / arrow_begin / point_begin ...`）
- 训练结果里外层结构可能学会一部分，但坐标内容容易塌缩
- `teacher forcing` 下 loss 看起来正常，但自回归输出结构和内容都不稳

因此本分支的策略是：

- 不再让模型学习一门新的协议语言
- 尽量利用它原生对 JSON / grounding / 数字坐标的先验

## 4. 当前主要代码链路

### 4.1 数据准备

- CLI 入口：`scripts/prepare_data.py`
- 实现：`src/vlm_det/data/prepare.py`
- 当前标准命令：

```bash
python scripts/prepare_data.py \
  --raw-json-dir data/raw/json \
  --image-dir data/raw/figure \
  --output-dir data/processed
```

当前输出 JSONL 的 `instances` 内部格式是：

```json
{
  "label": "single_arrow | double_arrow",
  "bbox": [x1, y1, x2, y2],
  "keypoints": [[x, y], ...]
}
```

注意：

- 这里保存的是 **原图像素坐标**
- 还没有归一化到 `[0,999]`
- 归一化发生在 `ArrowCodec.encode()`
- `keypoints` 语义和最终协议一致
- `c0~c3` -> `single_arrow`
- `c4~c7` -> `double_arrow`
- `double_arrow` 会在落盘前统一重排成“左 head 在前，右 head 在后”
- 最终写入 JSONL 前，会先对图内 `instances` 做 canonical 排序：
  `(y1, x1, y2, x2, y_first, x_first, y_last, x_last, n_points)`
- 这条排序规则在真实数据 prepare 和 synthetic export 两侧都要执行，
  dataset 读取阶段不再负责重排

### 4.2 Dataset

文件：`src/vlm_det/data/dataset.py`

行为：

- 读入 JSONL
- 取出每个 instance 的 `bbox` / `keypoints`
- 调 `ArrowCodec.encode(...)`
- 动态生成训练 target JSON 文本

### 4.3 Collator

文件：`src/vlm_det/data/collator.py`

行为：

- 默认 `system_prompt=""`
- 当前主说明放在 `user_prompt`
- user message 中包含：
  - image
  - text

训练时：

- `input_ids = prefix + target + eos`
- `labels = -100 for prefix + target + eos`

eval / infer 时：

- 只喂 prefix
- 不拼 target

### 4.4 Codec

文件：`src/vlm_det/protocol/codec.py`

这是 protocol-v2 的核心：

- `encode()`：像素坐标 -> `[0,999]` -> JSON
- `decode()`：文本中提取 JSON -> 反归一化回像素坐标

### 4.5 训练

- 入口：`scripts/train.py`
- 训练器：`src/vlm_det/train/trainer.py`

训练目标目前仍然是：

- 标准 next-token CE

还没有做：

- JSON schema constrained decoding
- 结构 token / 结束 token 的加权 loss
- sequence-level reward

### 4.6 评估

文件：`src/vlm_det/eval/evaluator.py`

当前评估指标包括：

- `val/parse_rate`
- `val/bbox_precision_at_iou50`
- `val/bbox_recall_at_iou50`
- `val/bbox_f1_at_iou50`
- `val/bbox_iou_mean`
- `val/keypoint_l2_mean`
- `val/keypoint_count_acc`
- `val/end_to_end_score`

### 4.7 推理

- 主推理：`scripts/infer.py`
- 推理实现：`src/vlm_det/infer/runner.py`
- 可视化：`src/vlm_det/infer/visualize.py`

## 5. prompt 现状

默认 `system_prompt`：

```text
""
```

默认 `user_prompt`：

```text
Detect all arrows and output only a JSON array, with no markdown and no extra text.
Normalize every coordinate to an integer in [0,999].
Each item must be either {"label":"single_arrow","bbox_2d":[x1,y1,x2,y2],"keypoints_2d":[[x,y],[x,y]]}
or {"label":"double_arrow","bbox_2d":[x1,y1,x2,y2],"keypoints_2d":[[x,y],[x,y]]}.
For single_arrow, keypoints are ordered from tail to head.
For double_arrow, keypoints[0] and keypoints[-1] are the two head tips.
Each arrow must contain at least 2 points.
```

## 6. 当前命名

为了和旧路线区分，模板和默认配置已经改成：

- `qwen3_vl_json_*`
- `outputs/qwen3_vl_json_*`
- wandb `project = vlm_det_json`

## 7. 已经清掉的旧路线残留

本分支已经删除或停用了这些旧逻辑：

- old DSL protocol
- special token 注入
- tokenizer resize embedding
- protocol state machine constrained decoding
- visibility 在协议中的表达
- visibility 相关评估指标

## 8. 当前明确存在的坑

### 8.1 `parse_rate` 口径偏宽

`ArrowCodec.decode()` 目前会：

- 从任意文本里提取第一个平衡 JSON
- 允许单 object 而不是 array
- 要求显式 `label`，但 lenient 仍允许从非纯净文本中抽出 JSON
- 坐标做 `float -> int(round)`

这意味着：

- `parse_rate` 并不是严格的“完整输出完全符合 prompt”
- 会偏乐观

如果后面要把 `parse_rate` 当作真正的格式指标，建议：

- 增加严格模式
- 要求完整输出就是 JSON array
- 不允许前后额外文本
- 不允许单 object

### 8.2 `infer.py` 对 decode 失败不容错

`src/vlm_det/infer/runner.py` 中：

- `codec.decode()` 失败会直接抛异常

这不影响训练，但第一次验模时很容易崩。
建议后续：

- 保存 raw output
- 解析失败时也落盘

### 8.3 真实数据准备按原始点顺序落盘

`prepare.py` 当前不会使用 point label 决定 keypoint 顺序。

- 点顺序直接沿用原始标注顺序
- `p1/p2/1/2` 不再参与排序或过滤

因此真实数据的 keypoint 语义，完全取决于原始标注文件本身是否已经按正确顺序存储。

### 8.4 synthetic pipeline 还有一个过时配置项

`synthetic_pipeline/configs/base.yaml` 里还保留：

- `occluded_point_probability`

协议已经不使用 visibility，这个字段目前无效，后续应清理。

### 8.5 `scripts/ground_once_qwen3vl.py` 不是主链脚本

它还是一个 bbox-only grounding 实验脚本：

- 不走当前 `bbox_2d + keypoints_2d` 主协议
- 不适合作为主链验收依据

## 9. 当前没有实现的内容

以下内容目前**还没有实现**，如果后续效果不够好，优先考虑这些方向：

- 更严格的 JSON 解析 / 校验模式
- 推理阶段 JSON repair 或 partial parse
- 基于 JSON schema 的 constrained decoding
- 结构正确率的更严格指标
- 用真实数据统计去约束 synthetic 分布
- 更系统的 hard negatives
- 面向 `single_crop` 的单独评估 bucket
- LLaMA-Factory 基线对照

## 10. synthetic pipeline 当前定位

目录：`synthetic_pipeline/`

作用：

- 生成 `data/sync/`
- 输出当前训练框架可直接读取的 JSONL
- 现在已经支持：
  - 多风格箭头
  - 曲线箭头
  - 横平竖直偏置
  - single crop / single hero / sparse large 等 scene mode
  - 多进程并行生成

但要注意：

- synthetic 分布还没有和真实业务数据做严格统计对齐
- 这仍然是后续最重要的改进方向之一

## 11. 训练前建议检查

如果你接手后要立刻开始训练，建议先确认：

1. 训练数据 JSONL 是否存在
2. `prepare_data.py` 跑出的数据里，`instances` 是否大量被丢弃
3. `train_sync_posttrain.yaml` 是否真的指向 `data/sync/*.jsonl`
4. eval preview 是否能打印出可解析的 JSON
5. 首轮 infer 是否有 raw output 保存机制

## 12. 当前仓库状态说明

本分支当前只剩一个常见未跟踪文件：

- `tmp.txt`

提交时不要把它带进去。

## 13. 我对这个项目当前阶段的判断

当前最重要的事情不是继续微调旧 DSL，而是：

- 把 protocol-v2 这条 JSON 数字路线训通
- 确认模型至少能稳定输出 JSON 结构
- 再判断几何能力是否开始提升

也就是说，当前阶段最重要的验证问题是：

**“新协议是否明显比旧 DSL 更好训、更稳、更符合 Qwen3-VL 的原生分布。”**

如果答案是肯定的，再继续在这条分支上做：

- 数据分布对齐
- 指标收紧
- 推理健壮性增强
