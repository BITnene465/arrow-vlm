# Synthetic Data Pipeline

这个目录提供第一版最小可用的箭头合成数据 pipeline，用来批量生成 `data/sync/` 数据，并直接接入当前训练框架。

## 文件

- `generate_sync_dataset.py`
  - 生成图像与 JSONL 标注
- `configs/base.yaml`
  - 默认合成参数

## 输出目录

默认输出到：

```text
data/sync/
  images/
    train/*.jpg
    val/*.jpg
  train.jsonl
  val.jsonl
  manifest.json
```

其中：

- `train.jsonl` / `val.jsonl`
  - 与当前 `ArrowSFTDataset` 直接兼容
- `manifest.json`
  - 记录本次生成所用配置与简单统计

## 快速开始

直接生成默认规模：

```bash
python synthetic_pipeline/generate_sync_dataset.py
```

默认参数：

- `train_samples = 10000`
- `val_samples = 200`
- `workers = 8`

## 常用覆盖参数

覆盖样本数：

```bash
python synthetic_pipeline/generate_sync_dataset.py \
  --train-samples 10000 \
  --val-samples 200
```

改输出目录：

```bash
python synthetic_pipeline/generate_sync_dataset.py \
  --output-dir /tmp/arrow_sync_debug
```

改随机种子：

```bash
python synthetic_pipeline/generate_sync_dataset.py \
  --seed 123
```

改并行进程数：

```bash
python synthetic_pipeline/generate_sync_dataset.py \
  --workers 8
```

换配置文件：

```bash
python synthetic_pipeline/generate_sync_dataset.py \
  --config synthetic_pipeline/configs/base.yaml
```

## 当前第一版生成策略

第一版重点是先生成足量、可控、能直接训练的数据，不追求复杂模块化。

当前会覆盖：

- 多档分辨率，从低分辨率到 `1024`
- 不同数量的 arrows
- 不同点数的 polyline arrows
- 多样式曲线箭头，关键点顺序固定为：
  - 第一个点 = 箭头尾部
  - 最后一个点 = 箭头头部
  - 中间点 = 拐点 / 曲线控制拐点
- 多种箭头样式：
  - 细线 / 中粗 / 粗线 / marker 风格
  - 实心箭头头 / 空心箭头头
  - 实线 / 虚线
- 基础干扰元素：
  - 直线
  - 矩形
  - 类文本短横线
- 基础成像退化：
  - 轻模糊
  - JPEG 压缩
  - 噪声
- 适度遮挡

同时会限制实例之间的大面积重叠，避免分布过度偏向极端拥挤场景。

此外当前还加入了：

- 双头箭头
  - 先作为 hard negative / 干扰元素绘制
  - 不进入正样本标注
- 多主题风格化箭头
  - scientific
  - drawio
  - ppt
  - grant_figure
  - marker_like
  - handdrawn

## 训练接入

生成完后，直接用这份配置训练：

```bash
python scripts/train.py --config configs/train_sync_posttrain.yaml
```

这份配置默认使用：

- `data/sync/train.jsonl`
- `data/sync/val.jsonl`
- `finetune.mode: full`
- `freeze_vision_tower: true`

## 备注

如果只是先做一次小规模冒烟测试，可以用：

```bash
python synthetic_pipeline/generate_sync_dataset.py \
  --train-samples 100 \
  --val-samples 20 \
  --output-dir /tmp/vlm_det_sync_smoke
```

## 当前进度记录

- 已支持直接导出 `data/sync/`，与当前训练框架兼容
- 已加入 `train_sync_posttrain.yaml` 作为 synthetic post-training 配置
- 坐标离散统一为 `1000 x 1000` 网格
- 已加入多主题箭头风格、不同粗细、不同线型
- 已加入曲线箭头，关键点顺序固定为尾部到头部
- 已加入双头箭头作为干扰元素
