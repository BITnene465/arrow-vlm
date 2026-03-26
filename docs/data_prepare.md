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
  --stage1-tile-sizes 768,1024 \
  --stage1-density-min-instances 5 \
  --stage1-density-max-instances 30
```

关键参数：

- `--stage1-include-full-image`
  是否保留原始整图样本。
- `--stage1-tile-sizes`
  Stage1 滑窗与 density crop 共用的裁剪尺寸列表。
- `--stage1-tile-stride-ratio`
  滑窗步长比例，实际步长 = `tile_size * stride_ratio`。
- `--stage1-density-min-instances`
  density crop 至少保留多少个箭头。
- `--stage1-density-max-instances`
  density crop 最多保留多少个箭头。
- `--stage1-density-max-crops-per-size`
  每张图、每种 tile size 最多保留多少个 density crop。
- `--stage1-min-visible-area-ratio`
  一个实例要进入 stage1 crop，要求其 bbox 在 crop 中的可见面积比例至少达到该阈值。

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
- Stage1 样本里的坐标都会转换成 tile-local 像素坐标。
- Stage1 只保留每个实例的：
  - `label`
  - `bbox`
  - 两个关键点

## Stage2 数据准备

Stage2 数据是目标条件 crop 数据集，训练时只输出该目标箭头的完整点列。

命令：

```bash
python scripts/prepare_stage2_data.py \
  --input-dir data/processed \
  --output-dir data/two_stage \
  --padding-ratio 0.5 \
  --num-workers 8 \
  --stage2-aug-copies 2
```

关键参数：

- `--padding-ratio`
  目标 bbox 的 crop padding 比例。
- `--stage2-aug-copies`
  每个训练实例额外生成多少条 noisy hint 副本。
- `--bbox-center-jitter-ratio`
  在原图坐标系下，对 hint bbox 中心的相对扰动范围，默认 `0.05`。
- `--bbox-scale-jitter-ratio`
  在原图坐标系下，对 hint bbox 宽高的相对扰动范围，默认 `0.08`。
- `--endpoint-jitter-ratio`
  在原图坐标系下，对两个 hint 点的相对扰动范围，默认 `0.03`。

产物：

```text
data/two_stage/stage2/train.jsonl
data/two_stage/stage2/val.jsonl
data/two_stage/stage2/images/train/
data/two_stage/stage2/images/val/
data/two_stage/reports/prepare_stage2_report.json
```

说明：

- Stage2 的 `train` 会生成 noisy hint 样本。
- Stage2 的 `val` 不做 augmentation，只保留 clean 样本。
- Stage2 的 `condition` 和 `target` 坐标都已经转换成 crop-local `[0,999]`。
