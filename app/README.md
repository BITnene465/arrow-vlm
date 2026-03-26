# App Usage

## One-Stage Demo

```bash
python app/demo.py
```

如果需要显式指定 checkpoint：

```bash
python app/demo.py --checkpoint outputs/qwen3vl-ft/4b/your-run/checkpoints/best
```

## Two-Stage Demo

```bash
python app/demo_two_stage.py \
  --stage1-config configs/train_stage1_lora_4b.yaml \
  --stage1-checkpoint outputs/qwen3vl-s1-lora/4b/your-stage1-run/checkpoints/best \
  --stage2-config configs/train_stage2_lora_4b.yaml \
  --stage2-checkpoint outputs/qwen3vl-s2-lora/4b/your-stage2-run/checkpoints/best
```

如果需要显式指定 base model：

```bash
python app/demo_two_stage.py \
  --stage1-config configs/train_stage1_lora_4b.yaml \
  --stage1-checkpoint outputs/qwen3vl-s1-lora/4b/your-stage1-run/checkpoints/best \
  --stage2-config configs/train_stage2_lora_4b.yaml \
  --stage2-checkpoint outputs/qwen3vl-s2-lora/4b/your-stage2-run/checkpoints/best \
  --stage1-model models/Qwen3-VL-4B-Instruct \
  --stage2-model models/Qwen3-VL-4B-Instruct
```
