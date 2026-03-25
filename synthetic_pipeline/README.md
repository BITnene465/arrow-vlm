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
  - 真实背景 / patch 混合渲染
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
    "label": "single_arrow",
    "bbox_2d": [123, 456, 789, 900],
    "keypoints_2d": [[130, 470], [188, 471]]
  }
]
```

其中 `label` 现在有两类：

- `single_arrow`
- `double_arrow`

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
- `single_arrow` / `double_arrow` 两类实例
- 低分辨率图更稀疏，高分辨率图允许更多 arrows
- 更宽的尺寸分布：
  - 小箭头
  - 中等箭头
  - 大箭头
  - `single_hero` 单大箭头场景
  - `single_crop` 单箭头 crop 场景
- 直线箭头，关键点顺序固定为：
  - `single_arrow`：第一个点 = 箭头尾部，最后一个点 = 箭头头部尖点
  - `double_arrow`：第一个点和最后一个点 = 两端头部尖点
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

关键点语义需要特别固定下来，避免后续出现理解错位：

- `single_arrow`：`keypoints[0]` = 尾部中心线点，`keypoints[-1]` = 头部尖点
- `double_arrow`：`keypoints[0]` 与 `keypoints[-1]` = 两端头部尖点
- `double_arrow`：写入 JSONL 前统一规范成左侧 head 在前，右侧 head 在后；若 `x` 相同，则更靠上的点在前
- 中间 keypoints 只表示路径形状，不表示箭头头部两侧轮廓
- 训练、评估、可视化都应沿用这套定义

synthetic 默认也会按 `arrow_label_weights` 采样类别，目前配置里：

- `single_arrow: 0.82`
- `double_arrow: 0.18`

除了 keypoint 语义之外，图内多个 instance 的顺序也必须固定，避免
同一张图在不同导出路径下产生不同 JSON 序列。当前 synthetic 导出在
最终写 JSONL 前统一做 canonical 排序，sort key 为：

- `(y1, x1, y2, x2, y_first, x_first, y_last, x_last, n_points)`
- 其中 `bbox = [x1, y1, x2, y2]`
- `y_first, x_first` 来自 `keypoints[0]`
- `y_last, x_last` 来自 `keypoints[-1]`

这条规则和真实数据 `prepare_data.py` 的落盘规则保持一致，不允许再由
dataset 读取阶段做二次重排。

`single_crop` 场景会混入最终数据集，而不是单独导出一套子集。它的特点是：

- 每图只保留一个主箭头
- crop 相对 bbox 留一圈上下文边距
- 干扰元素和遮挡明显减少
- 仍然保留一定背景上下文，贴近检测器先裁一块再做精细解析的使用方式

此外当前还加入了：

- 固定样式的黑色直箭头

## Hybrid Renderer

第一版 hybrid renderer 的原则是：

- 箭头几何和标注仍由程序生成
- 真实感来自 `data/processed/*.jsonl` 对应的业务图片资产
- 不直接从最终图像反推标注，避免标签变脏

当前 hybrid 具体会做：

- 从真实业务图裁切背景，作为整张图的 base canvas
- 采样无箭头负样本 patch，作为上下文块粘贴回画面
- 采样带正样本箭头的 crop patch，并把 crop 内完整实例同步映射成新的 GT
- 程序箭头统一使用固定 SVG 直箭头渲染

当前箭头的主绘制链已经切到 `SVG -> RGBA raster -> PIL compositing`：

- 箭头、程序干扰物、occluder 不再走 `PIL.ImageDraw`
- `PIL` 只保留背景生成、真实 patch 粘贴、退化和导出
- 程序箭头当前固定为单一直箭头

这意味着当前版本已经具备：

- 统一 schema
- procedural / hybrid 双后端
- 基于 `data/processed` 的资产索引
- 真实背景 patch / 负样本 context patch / 正样本 arrow patch 三类资产
- 按分辨率动态调整箭头密度，低分辨率更稀疏，高分辨率允许更多实例
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
- 已收敛为固定样式的 SVG 直箭头，关键点顺序固定为尾部到头部
- 已加入 `single_crop` 单箭头 crop 场景，并混入最终生成分布
- 已切换到 JSON 数字协议，更贴近 `Qwen3-VL` 官方 grounding 用法
- 已拆出模块化引擎，并支持 `procedural` / `hybrid` 双 renderer
- 已把箭头主渲染链切到 SVG 光栅化，PIL 不再负责箭头主体绘制
- 已加入 arrow patch 资产复用，并同步做 crop 内实例坐标变换
- 已按分辨率对箭头数量做动态缩放和上限约束
