# 2026-03-25 Canonical Instance Ordering And Faithful Dataset Loading

## 背景

当前任务的监督目标是一个有序 JSON array：

- 字段顺序固定：`bbox_2d` 在前，`keypoints_2d` 在后
- instance 内 keypoints 顺序固定：`tail -> ... -> head`
- 图内 instance 顺序也需要固定，否则同一张图会对应多个等价但不同文本目标

此前仓库里同时存在三类会干扰监督稳定性的行为：

1. `prepare.py` 会基于 `p1/p2/1/2` 标签处理真实数据点顺序
2. `dataset.py` 会在训练时随机打乱单张图内部的 `instances`
3. processed JSONL 里还保留 `group_id/raw_bbox/raw_keypoints` 这类训练无关字段，dataset 还会继续把它们带入 `gt_struct`

这些逻辑混在一起后，职责边界不清晰：

- 数据准备层不只是在“整理数据”，还在隐式改写点顺序
- dataset 不再是忠实读取 JSONL，而是在运行时改写 instance 顺序
- 同一份 JSONL 不是稳定监督来源

## 问题判断

这里真正应该固定的是“落盘结果”，不是“训练时临时打乱再编码”。

对于当前项目，合理的职责拆分应当是：

- `prepare.py` 负责把真实数据整理成最终规范顺序
- synthetic 导出负责把合成数据整理成最终规范顺序
- `dataset.py` 只忠实读取 JSONL 并编码成 target text
- train dataloader 只负责样本级 shuffle，不碰单张图内部 instance 顺序

另外，`p1/p2` 在当前业务定义里表示可见/不可见，不表示顺序，因此不能再作为 keypoint 顺序依据。

## 这次修改

### 1. 新增 canonical instance 排序 helper

新增文件：

- `src/vlm_det/data/ordering.py`

统一使用以下排序键：

`(y1, x1, y2, x2, y_tail, x_tail, y_head, x_head, n_points)`

即：

- 先按 bbox 扫描顺序：上到下、左到右
- 再用 tail/head 坐标和点数做 tie-break

这样做的目的：

- 方向无关
- 对端点轻微抖动更稳
- bbox-only / full-task 可共用同一 instance 顺序

### 2. 真实数据准备不再依赖 `p1/p2`

修改：

- `src/vlm_det/data/prepare.py`

当前行为：

- 直接沿用原始标注中的 point 顺序
- 不再接受/拒绝 `p1/p2/1/2` 这类标签约束
- 不再用 point label 排序
- 仍保留 bbox 和点数的基础结构校验
- 落盘前统一执行 canonical instance 排序

这意味着：

- keypoint 顺序语义完全由原始标注文件保证
- `prepare.py` 不再擅自推断点顺序

### 3. synthetic 数据落盘前统一 canonical 排序

修改：

- `synthetic_pipeline/schema.py`

`SyntheticSample.to_record()` 现在会在写出 `instances` 前调用 canonical 排序。

这样真实数据和 synthetic 数据共用同一排序规则，不会分叉。

### 4. dataset 只忠实读取 JSONL

修改：

- `src/vlm_det/data/dataset.py`
- `src/vlm_det/config.py`
- `scripts/train.py`
- `scripts/debug_eval.py`
- `configs/train_full_ft.yaml`
- `configs/train_lora.yaml`
- `configs/train_sync_posttrain.yaml`

清理内容：

- 移除 `shuffle_instances_for_training`
- 移除 dataset 内部 `random.shuffle(instances)`
- 移除 dataset 不必要的 `deepcopy`
- processed JSONL 不再保留 `group_id/raw_bbox/raw_keypoints`
- dataset 不再把这类训练无关字段混入训练 `gt_struct`

保留下来的只有：

- 样本级 shuffle：仍由 train dataloader 负责
- JSONL 内部 instance 顺序：完全按落盘顺序保留

## 当前主链语义

现在主链的顺序语义是明确的：

1. 原始真实标注：
   point 顺序由标注文件自己负责
2. 数据准备 / synthetic 导出：
   统一生成 canonical instance 顺序
3. dataset：
   忠实读取 JSONL，不做 instance 重排
4. train dataloader：
   只打乱样本顺序，不改图内 instance 顺序
5. codec：
   直接按 JSONL 中的稳定顺序编码文本监督

## 为什么这次改动必要

如果不这样做，会持续有三个风险：

1. 同一张图在不同 epoch 可能产生不同 target text
2. 真实数据和 synthetic 数据的 instance 顺序口径不一致
3. `p1/p2` 这类业务标签被误用成几何顺序规则

这些问题不会立刻造成 crash，但会让训练目标不稳定，并污染后续对模型行为的判断。

## 结论

这次修改的核心不是“换一个排序习惯”，而是把顺序职责固定下来：

- 顺序在数据生成/准备阶段确定
- dataset 不再改写监督目标
- `p1/p2` 不再被错误解释为顺序标签

这使得训练监督变成单一、稳定、可追溯的文本目标。
