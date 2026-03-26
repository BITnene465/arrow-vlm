# Qwen3-VL 训练状态残留笔记

Qwen3-VL 这类多模态模型在 train 和 eval 共用同一个模型实例时，要特别注意模型内部状态残留问题。

这次问题的结论是：

- `processor` 产出的 `mm_token_type_ids` 必须从 collator 一路传到 train / eval / infer
- 每次独立 batch 前向或生成前，都要清理模型运行时状态，避免上一个 batch 的多模态位置状态污染当前 batch

实现位置：

- `src/vlm_det/data/collator.py`
- `src/vlm_det/train/trainer.py`
- `src/vlm_det/eval/evaluator.py`
- `src/vlm_det/infer/runner.py`
- `src/vlm_det/utils/distributed.py`
