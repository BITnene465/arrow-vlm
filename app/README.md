# App Usage

## One-Stage Demo

```bash
python app/demo.py \
  --config configs/infer/infer_one_stage.yaml \
  --checkpoint outputs/qwen3vl-ft/4b/your-run/checkpoints/best
```

## Two-Stage Demo

```bash
python app/demo_two_stage.py \
  --config configs/infer/infer_two_stage.yaml \
  --stage1-checkpoint outputs/qwen3vl-s1-lora/4b/your-stage1-run/checkpoints/best
```

只看 Stage 1：

```bash
python app/demo_two_stage.py \
  --config configs/infer/infer_two_stage.yaml \
  --stage1-checkpoint outputs/qwen3vl-s1-lora/4b/your-stage1-run/checkpoints/best \
  --stage1-model models/Qwen3-VL-4B-Instruct
```

页面会固定显示三张图：

- 输入图
- Stage1 可视化
- Stage2 / 最终可视化

当前 `demo_two_stage` 同时支持：

- Stage1-only grounding 可视化检查
- 完整 two-stage 推理可视化
