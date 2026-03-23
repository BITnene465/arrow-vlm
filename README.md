# ArrowVLM

Train `Qwen3-VL` to read scientific figures, detect arrows, and generate structured arrow skeletons.

`ArrowVLM` is a focused multimodal fine-tuning stack for a narrow but practical task: understanding arrows in paper figures. Instead of treating the problem as generic captioning or forcing it into a fixed-size pose head, this project trains a vision-language model to emit a strict, parseable protocol containing:

- one bounding box per arrow
- an ordered, variable-length keypoint chain
- visibility labels for each keypoint

The result is a training and inference pipeline that stays close to modern VLM practice while remaining usable for structured geometry tasks.

## Why This Project

Arrows in technical figures are small, thin, irregular, and often partially occluded. They are also not a good fit for classic fixed-topology pose formulations. This repo takes a different route:

- use `Qwen3-VL-2B-Instruct` as the base model
- formulate arrow detection as structured sequence generation
- represent coordinates with discrete special tokens instead of raw numbers
- keep decoding fully machine-readable through a custom `ArrowCodec`

## Core Ideas

- **Whole-figure training**: the model sees the entire paper figure, not cropped arrow instances.
- **Structured protocol output**: predictions are emitted through a strict special-token format.
- **Variable-length skeletons**: each arrow can have a different number of turning points.
- **Visibility-aware keypoints**: keypoints are labeled as `visible` or `occluded`.
- **Local-first model loading**: by default, models are loaded from `models/` before falling back to remote sources.

## Output Representation

Each arrow is represented as:

- one `bbox`
- one ordered keypoint sequence
- first point = start point
- last point = arrow head
- middle points = turning points

Coordinates are normalized on the original image and discretized into `2048` bins using dedicated `x/y` tokens.

A decoded prediction looks like:

```json
{
  "instances": [
    {
      "bbox": [184, 92, 311, 245],
      "keypoints": [
        [190, 97, "visible"],
        [245, 141, "occluded"],
        [307, 240, "visible"]
      ]
    }
  ]
}
```

The model does not train against this JSON directly. Internally, it learns a stricter special-token protocol defined by `ArrowCodec`.

## What Is Implemented

- `Qwen3-VL` fine-tuning with explicit `LoRA` and `full` modes
- tokenizer extension and embedding resize for protocol tokens
- raw-to-normalized conversion from LabelMe-style annotations
- strict protocol encode/decode with `ArrowCodec`
- native `torch.distributed` DDP training
- `wandb` logging and fixed-width `tqdm` progress bars
- checkpoint save/resume for `last`, `best`, and step-based snapshots
- local-first inference and single-image decode pipeline

## Repository Layout

```text
configs/        training configs
scripts/        preparation, training, inference, and utility entrypoints
src/vlm_det/    core package
data/           raw and processed datasets
models/         local model weights
```

## Quick Start

### 1. Create the environment

This repository is designed around `uv` and Python `3.11`.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

If you need an explicit CUDA-enabled PyTorch wheel:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install -e . --no-deps
```

If you also want the optional FlashAttention dependency:

```bash
uv pip install -e ".[gpu]"
```

### 2. Configure local paths

```bash
cp .env.example .env
```

Use `.env` for machine-level defaults such as:

- local model path
- remote fallback model path
- output directory
- cache directory

Use `configs/*.yaml` for experiment-specific overrides.

Image inputs are not fed at native resolution. The processor uses bounded
dynamic resizing via `min_pixels` / `max_pixels`, which keeps aspect ratio
while preventing oversized figure images from blowing up visual token counts.

### 3. Prepare the dataset

```bash
python scripts/prepare_data.py \
  --raw-json-dir data/raw/json \
  --image-dir data/raw/figure \
  --output-dir data/processed/normalized
```

This converts the raw LabelMe-style arrow annotations into normalized `train/val` JSONL files.

### 4. Train

Single-GPU LoRA:

```bash
python scripts/train.py --config configs/train_lora.yaml
```

Single-GPU full fine-tuning:

```bash
python scripts/train.py --config configs/train_full_ft.yaml
```

Multi-GPU LoRA:

```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/train_lora.yaml
```

Multi-GPU full fine-tuning:

```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/train_full_ft.yaml
```

### 5. Run inference

```bash
python scripts/infer.py \
  --config configs/train_lora.yaml \
  --checkpoint outputs/your_experiment/checkpoints/best \
  --image /path/to/figure.jpg
```

### 6. Launch the demo

```bash
python app/demo.py \
  --config configs/train_lora.yaml \
  --checkpoint outputs/your_experiment/checkpoints/best
```

## Training Notes

- prompt and image-prefix positions are masked during training
- only the structured assistant target contributes to the loss
- protocol tokens are added to the tokenizer and require embedding resize
- in LoRA mode, `embed_tokens` and `lm_head` remain trainable
- long outputs are expected; evaluation generation length is configured accordingly

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
