# Synthetic Data Pipeline

这个子目录用于生产大规模箭头合成数据，并直接导出到 `data/sync/`，供当前训练框架使用。

第一版只保留最小结构：

- `configs/base.yaml`: 合成参数
- `generate_sync_dataset.py`: 直接生成 `data/sync/`

运行方式：

```bash
python synthetic_pipeline/generate_sync_dataset.py
```

默认会生成：

- `data/sync/images/train/*.jpg`
- `data/sync/images/val/*.jpg`
- `data/sync/train.jsonl`
- `data/sync/val.jsonl`
- `data/sync/manifest.json`

也可以覆盖样本数：

```bash
python synthetic_pipeline/generate_sync_dataset.py \
  --train-samples 10000 \
  --val-samples 200
```

对应的 post-training 配置文件：

- `configs/train_sync_posttrain.yaml`
