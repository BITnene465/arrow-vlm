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
  --stage1-checkpoint outputs/qwen3vl-s1-lora/4b/your-stage1-run/checkpoints/best \
  --stage2-checkpoint outputs/qwen3vl-s2-lora/4b/your-stage2-run/checkpoints/best
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

如果没有加载 Stage 2 checkpoint：

- 页面会提示当前处于 Stage1-only 模式
- 只显示 Stage1 可视化
