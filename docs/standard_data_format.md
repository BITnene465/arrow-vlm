# Standard Data Format

这份文档定义当前源码**直接支持**的标准 JSONL 数据格式。

适用范围：

- one-stage `joint_structure`
- two-stage Stage1 `grounding`
- two-stage Stage2 `keypoint_sequence`

不适用范围：

- 任意外部原始格式
- 一次性中间产物
- 第三方标注平台导出的非标准记录

这些外部格式应由二次开发人员在仓库外先转换成这里定义的标准格式，再交给当前源码。

---

## 1. 通用约束

当前训练数据采用 JSONL。

- 一个文件一行一个 sample
- 每条记录都必须显式带：
  - `task_type`
  - `domain_type`
- 当前箭头任务统一：
  - `domain_type = "arrow"`

所有图像路径都由记录显式给出：

- `image_path`

训练时不会去猜路径，也不会回退到别的目录查找图片。

---

## 2. 通用字段

所有标准记录都支持以下通用字段：

```json
{
  "task_type": "...",
  "domain_type": "arrow",
  "sample_id": "sample_0001",
  "image_path": "data/.../image.png",
  "image_width": 1024,
  "image_height": 768
}
```

字段说明：

- `task_type`
  - 当前支持：
    - `joint_structure`
    - `grounding`
    - `keypoint_sequence`
- `domain_type`
  - 当前箭头任务固定为：
    - `arrow`
- `sample_id`
  - 样本唯一标识
- `image_path`
  - 当前样本使用的图像路径
- `image_width`
  - 当前图像宽度，像素坐标系
- `image_height`
  - 当前图像高度，像素坐标系

---

## 3. Arrow Domain 的实例字段

箭头实例的标准结构是：

```json
{
  "label": "single_arrow",
  "bbox": [123.0, 45.0, 456.0, 300.0],
  "keypoints": [
    [130.0, 50.0],
    [200.0, 120.0],
    [420.0, 290.0]
  ]
}
```

字段说明：

- `label`
  - 只能是：
    - `single_arrow`
    - `double_arrow`
- `bbox`
  - 像素坐标
  - 格式：
    - `[x1, y1, x2, y2]`
- `keypoints`
  - 像素坐标
  - 格式：
    - `[[x0, y0], [x1, y1], ...]`

顺序约束：

- `single_arrow`
  - `keypoints` 必须按：
    - `tail -> ... -> head`
- `double_arrow`
  - `keypoints[0]` 和 `keypoints[-1]` 是两个箭头 head
  - 顺序固定为：
    - 左上 head 在前
    - 另一个 head 在后

整图多实例顺序约束：

- 必须在数据准备阶段固化 canonical order
- 当前 canonical sort key 是：
  - `(y1, x1, y2, x2, y_tail, x_tail, y_head, x_head, n_points)`

当前源码不会在 dataset 读取阶段替你重排实例顺序。

---

## 4. one-stage: joint_structure

这是 `data/processed/*.jsonl` 的标准格式。

### 必需字段

```json
{
  "task_type": "joint_structure",
  "domain_type": "arrow",
  "sample_id": "00772",
  "image_path": "data/raw/figure/00772.jpg",
  "image_width": 2816,
  "image_height": 1536,
  "instances": [
    {
      "label": "single_arrow",
      "bbox": [2184.21, 399.12, 2281.58, 545.61],
      "keypoints": [[2270.61, 406.14], [2186.84, 541.23]]
    }
  ]
}
```

### 说明

- `instances` 是监督主字段
- 这是当前 one-stage 训练的标准输入
- `target_text` 可以不预先写入
  - dataset 会按 `task_type + domain_type` 自动编码

---

## 5. Stage1: grounding

这是 `data/two_stage/stage1/*.jsonl` 的标准格式。

### 必需字段

```json
{
  "task_type": "grounding",
  "domain_type": "arrow",
  "sample_id": "00772__tile_0003",
  "image_path": "data/two_stage/stage1/images/train/00772__tile_0003.png",
  "image_width": 768,
  "image_height": 768,
  "instances": [
    {
      "label": "single_arrow",
      "bbox": [120.0, 88.0, 211.0, 240.0]
    }
  ]
}
```

### 说明

- Stage1 只需要：
  - `label`
  - `bbox`
- 不需要 `keypoints`
- `bbox` 是当前图像坐标系下的像素坐标
  - 对整图样本，是整图像素坐标
  - 对 crop / tile 样本，是 crop-local 像素坐标

可选字段：

- `source_type`
- `crop_box`
- `parent_sample_id`

这些可用于分析或调试，但不是训练必需。

---

## 6. Stage2: keypoint_sequence

这是 `data/two_stage/stage2/*.jsonl` 的标准格式。

### 必需字段

```json
{
  "task_type": "keypoint_sequence",
  "domain_type": "arrow",
  "sample_id": "00772__inst_0000",
  "image_path": "data/two_stage/stage2/images/train/00772__inst_0000.png",
  "image_width": 138,
  "image_height": 206,
  "instances": [
    {
      "label": "single_arrow",
      "bbox": [20.0, 18.0, 118.0, 180.0],
      "keypoints": [[106.61, 37.14], [22.84, 172.23]]
    }
  ],
  "condition": {
    "label": "single_arrow",
    "bbox": [20.0, 18.0, 118.0, 180.0],
    "bbox_2d": [147, 147, 857, 861],
    "keypoints": [[106.61, 37.14], [22.84, 172.23]],
    "keypoints_2d": [[777, 181], [167, 839]]
  }
}
```

### 说明

- 当前 Stage2 是单目标 crop 任务
- `instances` 当前只包含这个 target arrow
- `condition` 用于 prompt 模板渲染
- 当前 prompt 会显式使用：
  - `condition.label`
  - `condition.bbox_2d`

坐标语义：

- `instances[*].bbox`
  - crop-local 像素坐标
- `instances[*].keypoints`
  - crop-local 像素坐标
- `condition.bbox_2d`
  - crop-local `[0,999]` 量化坐标
- `condition.keypoints_2d`
  - crop-local `[0,999]` 量化坐标

---

## 7. target_text 和 prompt 字段

标准记录**可以不提前写** `target_text`、`system_prompt`、`user_prompt`。

当前 dataset 支持两种模式：

### 模式 A：完全离线写好

记录里直接写：

- `target_text`
- `system_prompt`
- `user_prompt`

### 模式 B：只写结构化字段

记录里只写：

- `task_type`
- `domain_type`
- `instances`
- `condition`

然后由当前源码根据：

- task adapter
- domain codec
- prompt template

在线生成：

- `target_text`
- prompt

当前仓库主线更推荐模式 B。

---

## 8. 不建议放进标准记录的内容

以下内容不应该混进标准训练数据格式：

- 外部标注平台原始字段
  - 例如：
    - `shapes`
    - `group_id`
    - `arrow_id`
    - `arrow_type`
    - `branch_group_id`
- 一次性调试字段
- 仅服务单次清洗脚本的中间字段

这些内容应在仓库外先整理，再导成这里定义的标准格式。

---

## 9. 二次开发边界

当前仓库内部只长期维护：

- 标准格式读取
- task/domain 路由
- 当前 arrow domain 的正式数据准备逻辑

仓库**不负责**长期兼容各种外部原始格式。

如果你接入新来源数据，推荐流程是：

1. 在仓库外写一次性转换脚本
2. 导出成这里定义的标准 JSONL
3. 再用当前训练/推理入口

相关文档：

- [data_prepare.md](/home/tanjingyuan/code/arrow-vlm/docs/data_prepare.md)
- [developer_task_extension.md](/home/tanjingyuan/code/arrow-vlm/docs/developer_task_extension.md)
- [task_domain_routing.md](/home/tanjingyuan/code/arrow-vlm/docs/task_domain_routing.md)
