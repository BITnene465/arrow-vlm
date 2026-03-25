# ArrowVLM

Train `Qwen3-VL` to detect arrows in scientific figures and output structured grounding results with:

- one `bbox_2d` per arrow
- one ordered `keypoints_2d` chain per arrow

This branch is the `protocol-v2` refactor. It moves the project closer to the official `Qwen3-VL` grounding usage pattern:

- use standard chat-style `user` prompts
- keep coordinates as normalized integers in `[0, 999]`
- output plain JSON instead of a custom special-token DSL
- do not extend the tokenizer with task-specific protocol tokens

## Task Format

Each prediction is a JSON array. Every item must be:

```json
{
  "label": "arrow",
  "bbox_2d": [123, 456, 789, 900],
  "keypoints_2d": [
    [130, 470],
    [188, 471],
    [270, 590]
  ]
}
```

Rules:

- all coordinates are normalized integers in `[0, 999]`
- `bbox_2d` uses `[x1, y1, x2, y2]`
- `keypoints_2d` are ordered from **tail to head**
- each point is `[x, y]`
- each arrow must contain at least `2` points

At training and evaluation time, coordinates are mapped between:

- normalized integer grid `[0, 999]`
- original image pixel coordinates

The model trains against the JSON text directly.

## Why This Version

Earlier versions of this project used a custom protocol with many task-specific special tokens such as:

- begin/end markers
- `x/y` coordinate tokens
- explicit point-list delimiters

That formulation was harder to train and drifted away from the official `Qwen3-VL` grounding style. This branch reduces that mismatch by:

- keeping the output machine-readable
- retaining keypoints
- avoiding new protocol token embeddings
- staying closer to the model’s native text-generation distribution

## Repository Layout

```text
configs/              training configs
scripts/              preparation, training, inference, and utility entrypoints
src/vlm_det/          core package
synthetic_pipeline/   synthetic data generation
data/                 raw and processed datasets
models/               local model weights
```

## Environment

This repository is designed around:

- Python `3.11`
- `uv`
- CUDA-enabled PyTorch when training on GPU

Create the environment:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

If you want explicit CUDA wheels:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install -e . --no-deps
```

If you also want the optional FlashAttention dependency:

```bash
uv pip install -e ".[gpu]"
```

## Data Preparation

Raw LabelMe-style annotations can be converted into normalized JSONL files with:

```bash
python scripts/prepare_data.py \
  --raw-json-dir data/raw/json \
  --image-dir data/raw/figure \
  --output-dir data/processed/normalized
```

This produces:

```text
data/processed/normalized/train.jsonl
data/processed/normalized/val.jsonl
```

Each record stores:

- image path
- image width / height
- per-arrow bbox
- per-arrow ordered keypoints

The JSONL records are then encoded into the normalized JSON grounding target during dataset loading.

## Synthetic Post-Training Data

Synthetic data generation lives under:

- [synthetic_pipeline/README.md](/Users/nene/codespace/PGstudy/mess_work/vlm_det/synthetic_pipeline/README.md)

The synthetic pipeline writes directly to:

```text
data/sync/
  images/
    train/
    val/
  train.jsonl
  val.jsonl
  manifest.json
```

Those files are directly consumable by the existing training stack.

## Training

### Single-GPU LoRA

```bash
python scripts/train.py --config configs/train_lora.yaml
```

### Single-GPU Full Fine-Tuning

```bash
python scripts/train.py --config configs/train_full_ft.yaml
```

### Single-GPU Synthetic Post-Training

```bash
python scripts/train.py --config configs/train_sync_posttrain.yaml
```

### Multi-GPU

LoRA:

```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/train_lora.yaml
```

Full FT:

```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/train_full_ft.yaml
```

### Stage-2 SFT From An Earlier Checkpoint

Use `--init-from` when you want to start a fresh training stage from earlier
weights without restoring optimizer, scheduler, RNG, or global step:

```bash
python scripts/train.py \
  --config configs/train_full_ft.yaml \
  --init-from outputs/qwen3_vl_json_sync_posttrain_ft/checkpoints/best
```

Use `--resume-from` only when you want to continue the same interrupted run.

### Two-Stage Training Launcher

Run synthetic post-train first, then launch real-data SFT from the stage-1 checkpoint:

```bash
python scripts/train_two_stage.py \
  --stage1-config configs/train_sync_posttrain.yaml \
  --stage2-config configs/train_full_ft.yaml
```

Use LoRA for stage 2 if needed:

```bash
python scripts/train_two_stage.py \
  --stage1-config configs/train_sync_posttrain.yaml \
  --stage2-config configs/train_lora.yaml
```

Preview commands without starting training:

```bash
python scripts/train_two_stage.py --dry-run
```

Use multi-GPU launchers for both stages:

```bash
python scripts/train_two_stage.py \
  --runner "torchrun --nproc_per_node=2"
```

## Prompting Style

This branch keeps the `system_prompt` interface, but the default configuration follows the official `Qwen3-VL` style:

- no custom system prompt by default
- the task instruction lives in the `user` message together with the image

The default prompt asks the model to:

- output only a JSON array
- avoid markdown and extra text
- use normalized integer coordinates in `[0, 999]`
- keep keypoints ordered from tail to head

## Inference

Run single-image inference:

```bash
python scripts/infer.py \
  --config configs/train_lora.yaml \
  --checkpoint outputs/your_experiment/checkpoints/best \
  --image /path/to/figure.jpg
```

Optionally save parsed output:

```bash
python scripts/infer.py \
  --config configs/train_lora.yaml \
  --checkpoint outputs/your_experiment/checkpoints/best \
  --image /path/to/figure.jpg \
  --output-dir outputs/infer_results
```

This saves:

- `*.prediction.json`
- `*.raw.txt`

## Evaluation and Debugging

To inspect a few decoded validation samples:

```bash
python scripts/debug_eval.py \
  --config configs/train_sync_posttrain.yaml \
  --checkpoint outputs/your_experiment/checkpoints/last \
  --split val \
  --num-samples 1 \
  --show-text
```

This prints:

- prompt length
- generated token count
- parse success / failure
- decoded raw text

During training evaluation, preview samples are also logged automatically.

## Important Notes

- images are not fed at raw native resolution without control
- the processor uses bounded dynamic resizing with `min_pixels` / `max_pixels`
- decoder-only batch generation must use left padding during evaluation
- coordinates are normalized to `[0, 999]`, but scoring is still performed in original image pixels after de-normalization

## Current Branch Status

This `protocol-v2` branch intentionally diverges from the older DSL-based protocol. In particular:

- no task-specific protocol token expansion
- no tokenizer resize for arrow protocol markers
- no special-token state-machine decoding
- JSON-based supervision and decoding instead

Old checkpoints trained with the previous DSL protocol should be treated as incompatible with this branch’s task format.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
