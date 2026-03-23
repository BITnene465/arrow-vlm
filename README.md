# Figure Arrow VLM

Structured VLM fine-tuning stack for detecting arrows in paper figures and predicting ordered arrow skeleton keypoints.

Recommended repository name: `figure-arrow-vlm`

## Overview

This project targets a specialized document-figure understanding task built on top of `Qwen3-VL-2B-Instruct`.

Given a full figure image, the model is trained to:

- detect every `arrow`
- output one bounding box per arrow
- output an ordered, variable-length keypoint chain
- predict keypoint visibility with `visible` / `occluded`
- generate results with a strict special-token protocol instead of free-form JSON

The training stack is built with:

- `transformers`
- `peft`
- native `torch.distributed` DDP
- a custom `ArrowCodec`
- local-first model resolution from `models/`

## Output Format

Each arrow instance is represented as:

- one `bbox`
- one ordered keypoint sequence
- first point = start point
- last point = arrow head
- intermediate points = turning points

Coordinates are normalized against the original image and discretized into `2048` bins via dedicated `x/y` special tokens.

## Features

- Qwen3-VL fine-tuning with explicit `LoRA` and `full` modes
- strict protocol encode/decode through `ArrowCodec`
- raw-to-normalized dataset conversion for LabelMe-style annotations
- native DDP training with `torchrun`
- local-first model loading from `models/`
- checkpointing, `wandb`, and fixed-width `tqdm` logging

## Repository Layout

```text
configs/        training configs for LoRA and full fine-tuning
scripts/        data preparation, training, inference, and utility scripts
src/vlm_det/    core package
data/           raw and processed data
models/         local model weights
```

## Quick Start

### Environment

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

### Configuration

Create a local environment file:

```bash
cp .env.example .env
```

Machine-level defaults live in `.env`. Experiment-level overrides live in `configs/*.yaml`.

### Data Preparation

```bash
python scripts/prepare_data.py
```

This converts the raw LabelMe-style annotations into normalized `train/val` JSONL files.

### Training

LoRA:

```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/train_lora.yaml
```

Full fine-tuning:

```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/train_full_ft.yaml
```

### Inference

```bash
python scripts/infer.py --config configs/train_lora.yaml --image /path/to/figure.jpg
```

## Notes

- Prompt and image-prefix positions are masked during training.
- Only the structured assistant target contributes to the training loss.
- Added protocol tokens require tokenizer extension plus embedding resize.
- In LoRA mode, `embed_tokens` and `lm_head` remain trainable.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
