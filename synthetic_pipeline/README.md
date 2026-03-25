# Synthetic Data Pipeline

这个目录提供第一版最小可用的箭头合成数据 pipeline，用来批量生成 `data/sync/` 数据，并直接接入当前训练框架。

当前分支已经切到更接近 `Qwen3-VL` 官方 grounding 风格的协议：

- 输出直接使用 JSON
- 坐标统一归一化到 `0~999`
- 不再依赖 task-specific special tokens

## 文件

- `generate_sync_dataset.py`
  - CLI 入口
- `scene_sampler.py`
  - 生成几何骨架与干净标注
- `asset_bank.py`
  - 从 `data/processed/*.jsonl` 读取真实资产索引
- `renderer/procedural.py`
  - 纯程序渲染基线
- `renderer/hybrid.py`
  - 真实背景 / patch / 纹理混合渲染
- `exporter.py`
  - 输出 JSONL / manifest / debug 可视化
- `configs/base.yaml`
  - 默认生成参数

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

训练时，这些 JSONL 记录会被编码成如下监督目标风格：

```json
[
  {
    "label": "arrow",
    "bbox_2d": [123, 456, 789, 900],
    "keypoints_2d": [[130, 470], [188, 471]]
  }
]
```

## 快速开始

直接生成默认规模：

```bash
python synthetic_pipeline/generate_sync_dataset.py
```

默认参数：

- `train_samples = 20000`
- `val_samples = 200`
- `workers = 20`
- `renderer = procedural`

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

切换到第一版 hybrid renderer：

```bash
python synthetic_pipeline/generate_sync_dataset.py \
  --renderer hybrid
```

换配置文件：

```bash
python synthetic_pipeline/generate_sync_dataset.py \
  --config synthetic_pipeline/configs/base.yaml
```

## 当前第一版生成策略

第一版重点是先生成足量、可控、能直接训练的数据，并把引擎拆成“scene sampler + renderer + asset bank”结构，方便后续继续研究真实感。

当前会覆盖：

- 多档分辨率，从低分辨率到 `1024`
- 不同数量的 arrows
- 横平竖直主方向箭头占多数，贴近真实图中的主流分布
- 更宽的尺寸分布：
  - 小箭头
  - 中等箭头
  - 大箭头
  - `single_hero` 单大箭头场景
  - `single_crop` 单箭头 crop 场景
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

`single_crop` 场景会混入最终数据集，而不是单独导出一套子集。它的特点是：

- 每图只保留一个主箭头
- crop 相对 bbox 留一圈上下文边距
- 干扰元素和遮挡明显减少
- 仍然保留一定背景上下文，贴近检测器先裁一块再做精细解析的使用方式

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

## Hybrid Renderer

第一版 hybrid renderer 的原则是：

- 箭头几何和标注仍由程序生成
- 真实感来自 `data/processed/*.jsonl` 对应的业务图片资产
- 不直接从最终图像反推标注，避免标签变脏

当前 hybrid 具体会做：

- 从真实业务图裁切背景，作为整张图的 base canvas
- 采样无箭头负样本 patch，作为上下文块粘贴回画面
- 从真实箭头实例中估计颜色 / 粗细 / 头部尺度，指导程序箭头渲染
- 用真实纹理 patch 给程序箭头填充纹理，再做轻量退化

这意味着第一版还不是“直接贴真实箭头 crop”，但已经具备：

- 统一 schema
- procedural / hybrid 双后端
- 基于 `data/processed` 的资产索引
- 对训练栈兼容的 JSONL 导出

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
- 坐标统一归一化到 `0~999`
- 已加入多主题箭头风格、不同粗细、不同线型
- 已加入曲线箭头，关键点顺序固定为尾部到头部
- 已加入双头箭头作为干扰元素
- 已加入 `single_crop` 单箭头 crop 场景，并混入最终生成分布
- 已切换到 JSON 数字协议，更贴近 `Qwen3-VL` 官方 grounding 用法
- 已拆出模块化引擎，并支持 `procedural` / `hybrid` 双 renderer
