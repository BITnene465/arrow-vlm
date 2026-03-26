# Canonical Order 笔记

这类输出有序 JSON 的任务，必须规定图内 instance 的 canonical order，否则同一张图会对应多种等价但不同的监督文本，训练会变得不稳定。

当前项目的做法是：

- 在数据准备和 synthetic 导出阶段就把 instance 顺序固定下来
- dataset 只忠实读取 JSONL，不再在运行时改写图内 instance 顺序
- train dataloader 只负责样本级 shuffle，不碰单张图内部顺序

当前采用的 instance 排序键是：

`(y1, x1, y2, x2, y_tail, x_tail, y_head, x_head, n_points)`

实现位置：

- `src/vlm_det/data/ordering.py`
- `src/vlm_det/data/prepare.py`
- `synthetic_pipeline/schema.py`
