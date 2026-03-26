# 两阶段箭头理解重构方案

## 目标

当前单阶段端到端输出整图 `bbox + 变长 keypoints` 的难度过高，`Qwen3-VL 2B/4B` 都会出现长输出不闭合、重复生成、末尾崩坏的问题。

新的主线方案改为两阶段推理：

1. **Stage 1：整图固定长度输出**
2. **Stage 2：单目标 crop 条件下输出完整变长点列**

训练上采用：

- 一个共享 base model
- 两个独立 LoRA adapter
  - LoRA A：Stage 1
  - LoRA B：Stage 2

---

## Stage 1 定义

### 任务

输入整图，输出每个箭头的：

- `label`
- `bbox_2d`
- `2-point keypoints_2d`

这是一个**固定长度**目标，用来替代当前“整图直接输出完整变长 polyline”。

### 输出协议

```json
[
  {
    "label": "single_arrow | double_arrow",
    "bbox_2d": [x1, y1, x2, y2],
    "keypoints_2d": [[x0, y0], [x1, y1]]
  }
]
```

### 坐标系

- 坐标统一使用整图 `[0,999]` 归一化网格

### label 约束

- `label` 只能是：
  - `single_arrow`
  - `double_arrow`

### keypoints 顺序

- `single_arrow`
  - `keypoints_2d[0]` = tail
  - `keypoints_2d[1]` = head
- `double_arrow`
  - `keypoints_2d[0]` = 左上的 head
  - `keypoints_2d[1]` = 另一个 head
  - 这里的“左上”规则是：
    - 先比 `x`
    - 若 `x` 相同，再比 `y`

### instance canonical order

整图内 instance 顺序继续固定，避免同一张图对应多种监督文本。

当前 canonical sort key 保持为：

`(y1, x1, y2, x2, y_tail, x_tail, y_head, x_head, n_points)`

注意：

- Stage 1 只输出 2 个点，但图内 instance 顺序仍然沿用这套 canonical order
- 数据准备和 synthetic 导出阶段就要写成最终顺序
- dataset 不允许再二次改写图内 instance 顺序

### Stage 1 prompt 建议

```text
Detect all arrows and output only a JSON array, with no markdown and no extra text.
Normalize every coordinate to an integer in [0,999].
Each item must be either {"label":"single_arrow","bbox_2d":[x1,y1,x2,y2],"keypoints_2d":[[x,y],[x,y]]}
or {"label":"double_arrow","bbox_2d":[x1,y1,x2,y2],"keypoints_2d":[[x,y],[x,y]]}.
For single_arrow, keypoints must be ordered from tail to head.
For double_arrow, keypoints[0] must be the upper-left head and keypoints[1] the other head.
Output exactly two keypoints for each arrow.
```

---

## Stage 2 定义

### 任务

输入一个 crop 图和该目标箭头的文本条件提示，输出这**一个目标箭头**的完整变长 `keypoints_2d`。

Stage 2 不是“bbox crop -> 自动猜哪条箭头”，而是：

- `target-conditioned crop -> 该目标箭头完整点列`

这点非常关键，因为一个 bbox/crop 内可能仍然有多个重叠箭头。

### Stage 2 输入

输入由两部分组成：

1. crop 图
2. 文本提示中的目标箭头 hint

文本 hint 建议显式给：

- `label`
- `bbox_2d`
- `keypoints_2d`（只给 2 个锚点）

这些提示坐标必须是 **crop-local [0,999]**，不能再用原图坐标。

### Stage 2 输出协议

建议只输出该箭头完整点列：

```json
[
  [x0, y0],
  [x1, y1],
  ...
]
```

也可以保守一点输出：

```json
{
  "keypoints_2d": [[x0, y0], [x1, y1], ...]
}
```

但为了最小化输出复杂度，推荐只输出点列本身。

### 坐标系

Stage 2 必须使用 **crop-local 坐标系**。

也就是说：

1. 从原图裁出 crop
2. 对目标箭头的：
   - bbox
   - 2-point hint
   - 完整 keypoints
3. 全部减去 crop 左上角偏移
4. 再重新量化到 crop 自己的 `[0,999]`

推理时如果要回到原图坐标：

1. 从 Stage 2 输出反量化到 crop 像素
2. 加回 crop 偏移

### keypoints 顺序

Stage 2 输出完整点列时，仍然沿用全项目统一语义：

- `single_arrow`
  - `tail -> ... -> head`
- `double_arrow`
  - `左上的 head -> ... -> 另一个 head`

### Stage 2 prompt 建议

```text
The cropped image may contain multiple arrows.
The target arrow is specified by:
{"label":"single_arrow","bbox_2d":[x1,y1,x2,y2],"keypoints_2d":[[x0,y0],[x1,y1]]}
Output only the complete keypoints_2d of this same arrow as a JSON array of points.
Normalize every coordinate to an integer in [0,999].
Do not output markdown or extra text.
```

对于 `double_arrow`，同样沿用相同格式，只是 2 个 hint 点表示两个 head。

### 为什么 Stage 2 先选文本坐标条件

对于 `Qwen3-VL`，文本里的 `[0,999]` grounding 坐标是它原生能理解的条件信号，因此第二阶段优先采用：

- crop 图提供上下文
- 文本坐标提示指定目标箭头

是否进一步把目标框/点画到图像上，可以作为增强版实验，而不是第一优先实现。

---

## 数据构造要求

### Stage 1 数据

- 直接从当前 processed / synthetic 数据中构造
- 每个 instance 只保留：
  - `label`
  - `bbox`
  - 2-point keypoints

### Stage 2 数据

每个 GT instance 派生出一个 Stage 2 样本：

1. 从原图中裁出目标 crop
2. 生成 crop-local 的：
   - `label`
   - `bbox`
   - 2-point hint
   - 完整 keypoints
3. prompt 使用 crop-local hint
4. target 输出 crop-local 完整点列

### crop 设计要求

- crop 不能只紧贴 bbox，要留一定上下文 padding
- 但也不能过大，否则重叠歧义太强
- 后续需要把 padding 做成可配置项

### 训练增强建议

Stage 2 训练时应对 hint 做轻微扰动，模拟 Stage 1 预测误差：

- bbox 小幅抖动
- 2 endpoints 小幅抖动

这样第二阶段不会只适应完美 GT 条件。

---

## 重构边界

### 需要新增

- Stage 1 协议 / dataset / config
- Stage 2 协议 / dataset / config
- Stage 2 crop 数据生成逻辑
- 两套独立 LoRA 训练入口或配置
- 两阶段串联推理逻辑

### 需要保持一致

以下内容必须在两个阶段中保持统一：

- arrow label 定义
- keypoint 顺序语义
- canonical order 规则（Stage 1）
- `[0,999]` 量化体系
- strict / lenient parser 的风格约束

---

## 当前结论

- Stage 1 负责整图固定长度输出：`label + bbox + 2 endpoints`
- Stage 2 负责单目标变长点列恢复
- Stage 2 必须是 **带目标条件的 crop 任务**
- 目标条件优先通过 **文本坐标提示** 提供
- Stage 2 所有提示和 GT 都必须转换到 **crop-local [0,999]** 坐标系
