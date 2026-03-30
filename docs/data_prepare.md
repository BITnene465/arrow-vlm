# Data Prepare

## 真实数据基础清洗

先把原始 LabelMe 标注转成标准 `processed` 数据：

```bash
python scripts/prepare_data.py \
  --raw-json-dir data/raw/json \
  --image-dir data/raw/figure \
  --output-dir data/processed
```

产物：

```text
data/processed/train.jsonl
data/processed/val.jsonl
data/processed/reports/data_cleaning_report.json
data/processed/reports/split_manifest.json
```

## Stage1 数据准备

Stage1 数据由三条线组成：

- 原始整图样本
- 多尺度滑窗样本
- 按箭头数量分布裁剪的 density crop 样本

命令：

```bash
python scripts/prepare_stage1_data.py \
  --input-dir data/processed \
  --output-dir data/two_stage \
  --num-workers 8 \
  --stage1-include-full-image \
  --stage1-tile-size-ratios 0.35,0.5 \
  --stage1-min-tile-size 512 \
  --stage1-max-tile-size 1280 \
  --stage1-density-min-instances 5 \
  --stage1-density-max-instances 30 \
  --stage1-dedup-iou-threshold 0.9
```

关键参数：

- `--stage1-include-full-image`
  是否保留原始整图样本。
- `--stage1-tile-size-ratios`
  Stage1 滑窗与 density crop 共用的裁剪尺寸比例列表，相对图像短边计算。
- `--stage1-min-tile-size`
  按比例换算后的最小 crop 像素尺寸。
- `--stage1-max-tile-size`
  按比例换算后的最大 crop 像素尺寸。
- `--stage1-tile-stride-ratio`
  滑窗步长比例，实际步长 = `tile_size * stride_ratio`。
- `--stage1-density-min-instances`
  density crop 至少保留多少个箭头。
- `--stage1-density-max-instances`
  density crop 最多保留多少个箭头。
- `--stage1-density-max-crops-per-size`
  每张图、每种 tile size 最多保留多少个 density crop。
- `--stage1-dedup-iou-threshold`
  当两个 crop 包含相同实例集合，且 crop IoU 超过该阈值时，去掉近重复样本。

产物：

```text
data/two_stage/stage1/train.jsonl
data/two_stage/stage1/val.jsonl
data/two_stage/stage1/images/train/
data/two_stage/stage1/images/val/
data/two_stage/reports/prepare_stage1_report.json
```

说明：

- train 和 val 都按同一套规则生成。
- Stage1 整图样本也会复制到 `data/two_stage/stage1/images/<split>/`，训练时不会再回 `data/processed` 或原始图片目录找图。
- Stage1 整图样本保留原图像素坐标。
- Stage1 crop / tile 样本会转换成各自局部图像的像素坐标。
- Stage1 只保留每个实例的：
  - `label`
  - `bbox`
- Stage1 实例是否进入某个 crop，规则只有一条：**bbox 必须完整落在 crop 内**。
- 如果某个 crop 与任意 bbox 只是部分相交，那么这个 crop 会被直接丢弃，不会进入最终数据集。
- Stage1 会自动去掉“实例集合相同且 crop 高度重叠”的近重复 crop，避免滑窗与 density crop 产出过多几乎相同的样本。

## Stage2 数据准备

Stage2 数据是单目标 crop 数据集，训练时输出由 crop-local `label + bbox_2d` 指定的 target arrow 的 `keypoints_2d` 骨架。
当前 target 格式是：

```json
{"keypoints_2d":[[x0,y0],[x1,y1],...]}
```

当前 prompt 会显式注入 crop-local 的 `label + bbox_2d`。

命令：

```bash
python scripts/prepare_stage2_data.py \
  --input-dir data/processed \
  --output-dir data/two_stage \
  --padding-ratio 0.3 \
  --num-workers 8 \
  --stage2-aug-ratio 0.0
```

关键参数：

- `--padding-ratio`
  目标 bbox 的 crop padding 比例。
- `--stage2-aug-ratio`
  训练实例中有多少比例额外生成 1 条 noisy hint 副本。比如 `0.3` 表示约 30% 的训练实例会多 1 条 noisy 样本。
- `--bbox-center-jitter-ratio`
  在原图坐标系下，对 hint bbox 中心的相对扰动范围，默认 `0.03`。
- `--bbox-scale-jitter-ratio`
  在原图坐标系下，对 hint bbox 宽高的相对扰动范围，默认 `0.05`。

产物：

```text
data/two_stage/stage2/train.jsonl
data/two_stage/stage2/val.jsonl
data/two_stage/stage2/images/train/
data/two_stage/stage2/images/val/
data/two_stage/reports/prepare_stage2_report.json
```

说明：

- Stage2 默认不生成 noisy hint 样本；只有显式设置 `--stage2-aug-ratio > 0` 才会生成。
- Stage2 的 noisy augmentation 只扰动 bbox crop 条件，不再扰动 endpoint。
- Stage2 的 `val` 不做 augmentation，只保留 clean 样本。
- Stage2 的 `target` 坐标已经转换成 crop-local `[0,999]`。
- 当前 Stage2 prompt 会注入 crop-local `label + bbox_2d`，再要求输出该 main arrow 的骨架；数据中的 `condition` 仍保留，用于兼容现有数据链路和后续扩展。
