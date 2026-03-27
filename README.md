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
  "label": "single_arrow",
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
- `label` must be `single_arrow` or `double_arrow`
- `bbox_2d` uses `[x1, y1, x2, y2]`
- for `single_arrow`, `keypoints_2d` are ordered from **tail to head**
- for `single_arrow`, `keypoints_2d[0]` is the arrow tail point on the centerline
- for `single_arrow`, `keypoints_2d[-1]` is the arrow head tip on the centerline
- for `double_arrow`, `keypoints_2d[0]` and `keypoints_2d[-1]` are the two head tips
- for `double_arrow`, the stored order starts from the upper-left head and ends at the other head; if `x` ties, smaller `y` comes first
- for polyline / curve arrows, intermediate keypoints are path control points rather than arrow-head corner points
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
  --output-dir data/processed
```

This produces:

```text
data/processed/train.jsonl
data/processed/val.jsonl
data/processed/reports/data_cleaning_report.json
data/processed/reports/split_manifest.json
```

Each record stores:

- image path
- image width / height
- per-arrow label
- per-arrow bbox
- per-arrow ordered keypoints

Before writing each sample to JSONL, the pipeline canonicalizes the in-image
instance order to keep training targets deterministic. The sort key is:

- `(y1, x1, y2, x2, y_tail, x_tail, y_head, x_head, n_points)`
- where `bbox = [x1, y1, x2, y2]`
- in practice this means instances are written top-to-bottom first, then
  left-to-right, with endpoint coordinates and point count used only as
  tie-breakers

Keypoint semantics are fixed across real and synthetic data:

- `single_arrow`: first keypoint = tail point, last keypoint = head tip
- `double_arrow`: first keypoint and last keypoint = the two head tips
- `double_arrow`: point order is normalized so the upper-left head comes first before writing JSONL
- intermediate keypoints = path control points

For real LabelMe data, rectangle classes are mapped as:

- `c0~c3` -> `single_arrow`
- `c4~c7` -> `double_arrow`

The JSONL records are then encoded into the normalized JSON grounding target during dataset loading.

## Two-Stage Data Preparation

Two-stage experiments derive two datasets from the processed annotations:

- `stage1`: mixed full-image + tile supervision
  - original full-image samples
  - ratio-based multi-scale sliding-window samples
  - density-driven crop samples
  - grounding target: `label + bbox`
  - prompt style aligned with the Qwen3-VL grounding cookbook: short instruction + relative coordinates
- `stage2`: target-conditioned crop supervision
  - crop image
  - main-arrow keypoint skeleton target

Prepare Stage1:

```bash
python scripts/prepare_stage1_data.py \
  --input-dir data/processed \
  --output-dir data/two_stage \
  --num-workers 8 \
  --stage1-include-full-image \
  --stage1-tile-size-ratios 0.35,0.5 \
  --stage1-min-tile-size 512 \
  --stage1-max-tile-size 1280 \
  --stage1-density-min-instances 5 \
  --stage1-density-max-instances 30 \
  --stage1-dedup-iou-threshold 0.9
```

Prepare Stage2:

```bash
python scripts/prepare_stage2_data.py \
  --input-dir data/processed \
  --output-dir data/two_stage \
  --padding-ratio 0.2 \
  --num-workers 8 \
  --stage2-aug-copies 0
```

This writes:

```text
data/two_stage/
  stage1/
    train.jsonl
    val.jsonl
  stage2/
    train.jsonl
    val.jsonl
    images/
      train/
      val/
  reports/
    prepare_stage1_report.json
    prepare_stage2_report.json
```

Stage1 tile sizing is now ratio-driven:

- crop sizes are resolved from image short-side ratios
- `stage1_min_tile_size` / `stage1_max_tile_size` clamp the resolved pixel size
- an instance is kept in a tile only when its bbox is fully enclosed by that tile
- any tile that partially intersects a bbox is discarded instead of becoming a training sample
- near-duplicate crops with the same instance set are removed using `stage1_dedup_iou_threshold`

Stage 2 uses crop-local coordinates. For every target instance:

- the crop is centered on the target bbox
- default padding ratio is `0.2`
- out-of-bound crop area is padded with black pixels
- training targets are reprojected into the crop-local `[0,999]` coordinate system
- stage2 JSONL still stores structured `condition` fields for compatibility, while the current prompt focuses on the main arrow in the crop
- stage2 does not add noisy hint copies by default; keep `--stage2-aug-copies 0` unless you explicitly want augmentation

More detailed usage is documented in:

- [docs/data_prepare.md](/home/tanjingyuan/code/arrow-vlm/docs/data_prepare.md)

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

Synthetic records are also canonicalized before export with the same
instance-order rule:

- `(y1, x1, y2, x2, y_tail, x_tail, y_head, x_head, n_points)`

## Training

### Single-GPU LoRA

```bash
python scripts/train.py --config configs/train/train_lora.yaml
```

Add a run id to make the output directory and W&B run traceable:

```bash
python scripts/train.py \
  --config configs/train/train_lora.yaml \
  --run-id 20260325-exp01
```

The training config automatically appends the model scale tag inferred from
`model_name_or_path` / `remote_model_name_or_path`. For example, a 2B model
will use names and output paths ending in `2b`, while 4B / 8B configs will be
separated automatically.

Override the vision-tower freezing behavior for one run:

```bash
python scripts/train.py \
  --config configs/train/train_full_ft.yaml \
  --run-id 20260325-exp01 \
  --freeze-vision-tower false
```

Gradient checkpointing is enabled by default. Override it explicitly if you want to turn it off:

```bash
python scripts/train.py \
  --config configs/train/train_full_ft.yaml \
  --run-id 20260325-exp01 \
  --gradient-checkpointing false
```

### Single-GPU Full Fine-Tuning

```bash
python scripts/train.py --config configs/train/train_full_ft.yaml
```

### Single-GPU Synthetic Post-Training

```bash
python scripts/train.py --config configs/train/train_sync_posttrain.yaml
```

### Multi-GPU

LoRA:

```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/train/train_lora.yaml
```

Full FT:

```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/train/train_full_ft.yaml
```

### Stage-2 SFT From An Earlier Checkpoint

Use `--init-from` when you want to start a fresh training stage from earlier
weights without restoring optimizer, scheduler, RNG, or global step:

```bash
python scripts/train.py \
  --config configs/train/train_full_ft.yaml \
  --init-from outputs/qwen3vl-post/2b/checkpoints/best
```

Use `--resume-from` only when you want to continue the same interrupted run.

### Two-Stage LoRA Training

Stage 1, 2B:

```bash
python scripts/train.py --config configs/train/train_stage1_lora.yaml
```

Stage 1, 4B:

```bash
python scripts/train.py --config configs/train/train_stage1_lora_4b.yaml
```

Stage 2, 2B:

```bash
python scripts/train.py --config configs/train/train_stage2_lora.yaml
```

Stage 2, 4B:

```bash
python scripts/train.py --config configs/train/train_stage2_lora_4b.yaml
```

Stage 2 records carry structured `condition` fields plus `target_text`. The
final user prompt is rendered at training time from
`prompt.user_prompt_template`, so the normal `scripts/train.py` entrypoint can
still be reused without baking full prompt strings into every sample.

### LoRA Target Groups

LoRA configs now split target modules into three groups:

- `lang_target_modules`: language tower LoRA targets such as
  `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`
- `vis_target_modules`: visual tower LoRA targets such as
  `attn.qkv/attn.proj/mlp.linear_fc1/mlp.linear_fc2`
- `proj_target_modules`: projector LoRA targets; an empty list means "all
  linear layers matched by `projector_name_substrings`"

Behavior:

- `freeze_vision_tower: true`: no visual-tower LoRA is attached
- `freeze_vision_tower: false`: visual-tower LoRA is attached on
  `vis_target_modules`
- `train_projector: true` in LoRA mode: projector LoRA is attached on
  `proj_target_modules`, rather than fully unfreezing projector weights

## Two-Stage Inference

Run the current Stage 1 grounding inspection flow with:

```bash
python scripts/infer_two_stage.py \
  --config configs/infer/infer_two_stage.yaml \
  --stage1-checkpoint outputs/qwen3vl-s1-lora/4b/checkpoints/best \
  --image path/to/example.png \
  --output-dir outputs/two_stage_demo
```

Current two-stage flow:

1. Stage 1 grounding predicts `label + bbox` on the full image.
2. `demo_two_stage` and `infer_two_stage.py` currently serve as Stage 1 inspection tools.
3. The previous Stage 2 refinement path was tied to the retired Stage 1 `bbox + 2-point hint` formulation.
4. Stage 2 will be reintroduced after its task definition is redesigned around the new grounding-only Stage 1.

## Two-Stage Demo

Launch the two-stage Gradio app with:

```bash
python app/demo_two_stage.py \
  --config configs/infer/infer_two_stage.yaml \
  --stage1-checkpoint outputs/qwen3vl-s1-lora/4b/checkpoints/best
```

`demo_two_stage.py` 固定展示三张图：

- 输入图
- Stage1 Overlay
- Stage2 / Final Overlay

如果不提供 `--stage2-checkpoint`，页面会进入 Stage1-only 模式：

- 状态区会明确提示未加载 Stage2
- 第二张图显示 Stage1 可视化
- 第三张图留空

### Two-Stage Training Launcher

Run synthetic post-train first, then launch real-data SFT from the stage-1 checkpoint:

```bash
python scripts/train_two_stage.py \
  --run-id 20260325-exp01 \
  --stage1-freeze-vision-tower false \
  --stage2-freeze-vision-tower true \
  --stage2-gradient-checkpointing true \
  --stage1-config configs/train/train_sync_posttrain.yaml \
  --stage2-config configs/train/train_full_ft.yaml
```

Use LoRA for stage 2 if needed:

```bash
python scripts/train_two_stage.py \
  --run-id 20260325-exp01 \
  --stage1-config configs/train/train_sync_posttrain.yaml \
  --stage2-config configs/train/train_lora.yaml
```

Preview commands without starting training:

```bash
python scripts/train_two_stage.py --dry-run
```

With `--run-id`, the two stages are separated automatically, for example:

```text
outputs/qwen3vl-post/2b/20260325-exp01/stage1_sync_posttrain
outputs/qwen3vl-ft/2b/20260325-exp01/stage2_real_sft
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
- emit `single_arrow` / `double_arrow` labels explicitly
- keep `single_arrow` keypoints ordered from tail to head

## Inference

One-stage inference uses an infer config instead of a training YAML.

Create a local `.env` only if you want `CHECKPOINT_PATH` fallback:

```bash
cp .env.example .env
```

Run inference with:

```bash
python scripts/infer.py \
  --config configs/infer/infer_one_stage.yaml \
  --checkpoint outputs/your_experiment/checkpoints/best \
  --max-new-tokens 4096 \
  --image /path/to/figure.jpg
```

This saves:

- `*.prediction.json`
- `*.raw.txt`

or let the checkpoint fall back to `CHECKPOINT_PATH` in `.env`:

```bash
python scripts/infer.py \
  --config configs/infer/infer_one_stage.yaml \
  --env-file /path/to/.env \
  --max-new-tokens 4096 \
  --image /path/to/figure.jpg \
  --output-dir outputs/infer_results
```

The CLI also prints the raw decoded model text to stdout so you can inspect strict-vs-lenient parse failures directly.

## App

Launch the one-stage Gradio app with the infer config:

```bash
python app/demo.py \
  --config configs/infer/infer_one_stage.yaml \
  --checkpoint outputs/your_experiment/checkpoints/best
```

or override `max_new_tokens` for the whole app session:

```bash
python app/demo.py \
  --config configs/infer/infer_one_stage.yaml \
  --checkpoint outputs/your_experiment/checkpoints/best \
  --max-new-tokens 4096
```

Inference configs only store prompt/template and inference-time settings.
They do not store checkpoint paths or base-model paths.

- one-stage infer config: `configs/infer/infer_one_stage.yaml`
- two-stage infer config: `configs/infer/infer_two_stage.yaml`

Training configuration is separate: `scripts/train.py` still only reads training YAML files.

## Evaluation and Debugging

To inspect a few decoded validation samples:

```bash
python scripts/debug_eval.py \
  --config configs/train/train_sync_posttrain.yaml \
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
