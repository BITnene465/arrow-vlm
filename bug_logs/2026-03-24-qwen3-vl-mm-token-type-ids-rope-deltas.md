# Qwen3-VL Training Crash After Eval

Date: 2026-03-25

## Summary

一次 synthetic 全量 post-train 在第 1 轮训练和第 1 次 eval 都正常结束后，回到第 2 轮训练的第一个 step 立即崩溃。

报错核心是：

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (434x0 and 2048x2048)
```

这个问题的主因不是数据脏样本，也不是训练发散，而是当前仓库在 Qwen3-VL 输入链路上漏传了 `mm_token_type_ids`，导致模型在 `eval.generate()` 后残留的 `rope_deltas` 被错误复用到下一轮训练 batch，最终把 attention 内部的张量最后一维压成了 `0`。

## Observed Symptoms

训练日志的关键片段：

```text
train e1  ####| 20000/20000 [59:20<00:00] , gs=2e+4, loss=0.5769, grad=10.25, lr=5.9e-06
eval      ##############| 50/50 [24:19<00:00] , parseL=1.00, parseS=0.92, p=0.88, r=0.67
...
train e2 | 0/20000 [00:00<?] Traceback ...
RuntimeError: mat1 and mat2 shapes cannot be multiplied (434x0 and 2048x2048)
```

可见模式非常固定：

1. epoch 1 训练正常
2. eval 正常
3. 切回 epoch 2 训练时，在第一个 batch 立刻崩溃

这种模式更像“eval 改坏了模型内部状态”，而不是“某个随机训练样本坏了”。

## Why This Was Suspicious

如果是数据样本坏掉，通常会表现为：

- 在某个随机 step 崩溃
- 再次运行时崩溃位置不稳定
- 不依赖于是否插入 eval

这次不是。它高度依赖“eval 之后的第一个 training step”，所以第一判断应当是：

- 训练和 eval 共享了同一个模型实例
- eval 路径写入了某种缓存状态
- 训练路径没有正确重建或清理这部分状态

## Relevant Log Excerpt

崩溃栈的尾部落在 Qwen3-VL self-attention 的输出投影：

```text
File ".../modeling_qwen3_vl.py", line 500, in forward
    attn_output = self.o_proj(attn_output)
...
RuntimeError: mat1 and mat2 shapes cannot be multiplied (434x0 and 2048x2048)
```

`o_proj` 的输入最后一维理论上应当是 `2048`，但实际变成了 `0`，说明更早的 attention shape 已经被破坏。

## Repo-Level Root Cause

### 1. Processor 原始输出里本来就有 `mm_token_type_ids`

本地检查 `processor(...)` 的返回值，实际包含：

```text
['attention_mask', 'image_grid_thw', 'input_ids', 'mm_token_type_ids', 'pixel_values']
```

说明 Qwen3-VL 的 processor 已经正确产出多模态 token type 信息。

### 2. 我们的 collator 把它丢了

在当前仓库里，[src/vlm_det/data/collator.py](/home/tanjingyuan/code/arrow-vlm/src/vlm_det/data/collator.py#L45) 先拿到：

```python
prefix_batch = self.processor(**processor_kwargs)
```

但最后输出给训练/评估的 batch 只保留了：

- `input_ids`
- `attention_mask`
- `labels`
- `pixel_values`
- `image_grid_thw`

见 [src/vlm_det/data/collator.py](/home/tanjingyuan/code/arrow-vlm/src/vlm_det/data/collator.py#L99)。

也就是说，`mm_token_type_ids` 在 collator 层被静默丢弃了。

### 3. 训练和 eval 后续都没把 `mm_token_type_ids` 传给模型

训练输入构造见 [src/vlm_det/train/trainer.py](/home/tanjingyuan/code/arrow-vlm/src/vlm_det/train/trainer.py#L143)。

评估生成输入构造见 [src/vlm_det/eval/evaluator.py](/home/tanjingyuan/code/arrow-vlm/src/vlm_det/eval/evaluator.py#L75)。

这两条路径都没有处理 `mm_token_type_ids`。

## Qwen3-VL Internal Mechanism Behind The Crash

### 1. Qwen3-VL 会在模型对象上缓存 `rope_deltas`

本地 transformers 实现里，Qwen3-VL 模型对象在初始化时就挂了这个状态：

- [modeling_qwen3_vl.py](/home/tanjingyuan/code/arrow-vlm/.venv/lib/python3.11/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L970)

```python
self.rope_deltas = None
```

### 2. 当 `mm_token_type_ids` 可用时，模型会重新计算多模态 rope 信息

在 forward 里，只有在下面条件成立时，模型才会根据视觉 token 正确计算新的 3D rope：

- `input_ids is not None`
- `mm_token_type_ids is not None`
- `image_grid_thw is not None` 或 `video_grid_thw is not None`

见 [modeling_qwen3_vl.py](/home/tanjingyuan/code/arrow-vlm/.venv/lib/python3.11/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L1241)。

关键代码：

```python
if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
    position_ids, rope_deltas = self.get_rope_index(...)
    self.rope_deltas = rope_deltas
```

### 3. 如果 `mm_token_type_ids` 缺失，但 `rope_deltas` 还留着，模型会错误复用旧值

同一段逻辑里，下一分支是：

```python
elif self.rope_deltas is not None:
    ...
    delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
```

见 [modeling_qwen3_vl.py](/home/tanjingyuan/code/arrow-vlm/.venv/lib/python3.11/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L1251)。

这就是本次事故的核心。

### 4. `generate()` 会更新 `rope_deltas`

Qwen3-VL 的 generation 准备阶段，如果 `mm_token_type_ids` 存在，会显式计算视觉位置并写回：

- [modeling_qwen3_vl.py](/home/tanjingyuan/code/arrow-vlm/.venv/lib/python3.11/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L1616)

```python
vision_positions, rope_deltas = self.model.get_rope_index(inputs_tensor, **model_kwargs)
self.model.rope_deltas = rope_deltas
```

所以 eval 之后，模型对象上已经带着上一个 eval batch 的 `rope_deltas` 了。

## Exact Failure Chain In This Run

结合这次日志和当前配置，失败链路如下：

1. epoch 1 训练时，虽然没有传 `mm_token_type_ids`，但模型初始 `rope_deltas` 为空，所以训练仍能跑。
2. eval 时，`generate()` 走了多模态位置计算，模型内部写入了新的 `rope_deltas`。
3. eval 的 batch size 是 `8`，所以此时 `self.rope_deltas.shape[0]` 很可能是 `8`。
4. 回到 epoch 2 第一个 train step 时，训练 batch size 是 `2`。
5. 由于 collator 和 trainer 都没传 `mm_token_type_ids`，模型无法重新计算当前 batch 的多模态 rope。
6. 模型错误进入“复用旧 `rope_deltas`”路径。
7. 执行：

```python
batch_size // self.rope_deltas.shape[0]
```

即：

```text
2 // 8 = 0
```

8. `repeat_interleave(0, ...)` 使 delta 扩成空张量。
9. 后续位置编码和 attention shape 被破坏。
10. 最后在 `o_proj` 处出现：

```text
mat1 and mat2 shapes cannot be multiplied (434x0 and 2048x2048)
```

## Why The Crash Happened Only After Eval

因为 eval 和 train 共用了同一个模型实例，而 Qwen3-VL 会在模型对象上缓存 `rope_deltas`。

如果没有跑 eval：

- `rope_deltas` 可能一直保持 `None`
- 训练路径暂时不会踩到这个状态残留问题

一旦跑了 eval.generate()：

- `rope_deltas` 被写入
- 下一次训练前向又没传 `mm_token_type_ids`
- 就会错误复用上一批 eval 的多模态位置状态

所以这个 bug 的触发条件就是：

```text
共享同一模型实例 + eval.generate() 写缓存 + train 路径漏传 mm_token_type_ids
```

## Supporting Evidence Collected

### Evidence A: processor 确实产出 `mm_token_type_ids`

本地检查结果：

```text
['attention_mask', 'image_grid_thw', 'input_ids', 'mm_token_type_ids', 'pixel_values']
```

### Evidence B: collator 输出里没有 `mm_token_type_ids`

本地检查当前 collator 结果：

```text
['attention_mask', 'image_grid_thw', 'input_ids', 'labels', 'meta', 'pixel_values', 'prompt_lengths']
has_mm_token_type_ids= False
```

### Evidence C: 栈信息正好落在 attention 内部形状损坏点

报错最后落在 `self.o_proj(attn_output)`，说明不是 loss、label 或 tokenizer 的问题，而是更前面的 attention shape 已经坏掉。

## What This Bug Is Not

这次事故不是下面这些问题：

- 不是 parser strict/lenient 逻辑导致
- 不是 eval matching 逻辑导致
- 不是 JSON decode 导致
- 不是单个 GT 样本 bbox/keypoint 标注坏掉
- 不是学习率导致训练发散
- 不是 `init-from` 逻辑导致

这些模块都不会把 attention 输出的最后一维变成 `0`。

## Fix Plan

### Required Fix 1

把 `mm_token_type_ids` 从 processor 一路传下去：

- collator 输出增加 `mm_token_type_ids`
- trainer `_move_batch_to_device(...)` 传给模型
- evaluator `generate_inputs` 传给 `model.generate(...)`
- infer 路径同样传递

这是主修复。

### Required Fix 2

在 train/eval/infer 的每个独立 batch 前向前，显式清理模型对象上的 `rope_deltas`。

例如在调用前：

```python
raw_model = unwrap_model(model)
if hasattr(raw_model, "rope_deltas"):
    raw_model.rope_deltas = None
```

这不是替代主修复，而是防御性修复，避免以后再被类似的跨阶段状态残留击中。

### Recommended Fix 3

训练前向显式传 `use_cache=False`。

原因：

- 训练不需要 KV cache
- 明确关闭 cache 可减少状态残留和实现歧义
- 对 decoder-only/VL 模型来说，训练阶段默认关 cache 更稳

## Suggested Regression Tests

建议至少补这三类回归：

1. collator 回归
   - processor 产出 `mm_token_type_ids` 时，collator 输出必须保留它

2. eval -> train 状态回归
   - 同一模型实例先做一次 `generate()`，再做一次 train forward
   - 不应因 batch size 变化而崩溃

3. 多 batch size 回归
   - eval batch size 和 train batch size 不同
   - 尤其覆盖 `8 -> 2`、`4 -> 1` 这类会触发整数除法问题的组合

## Practical Lessons

这次问题有几个很值得记住的工程点：

1. 多模态 processor 给出的字段不能想当然删
   - 对 Qwen3-VL 来说，`mm_token_type_ids` 不是可有可无的附属信息
   - 它直接参与多模态位置编码构建

2. `generate()` 不一定是纯函数
   - 很多模型会把 cache / rope / position 之类状态写回模型对象
   - 训练和 eval 共用同一实例时，必须警惕状态泄漏

3. “只在 eval 后崩”的问题，优先查状态残留
   - 比起先怀疑数据和优化器，更应该先查 model object 上是否有缓存成员

4. 训练 batch 和 eval batch 不同大小时，更容易把这类 bug 放大出来
   - 这次就是 `8 -> 2` 让 `repeat_interleave(0)` 暴露得非常明显

## Status

截至本报告更新时：

- 根因已定位
- 复现链路已讲清
- 修复已落地

修复内容包括：

- collator 保留并扩展 `mm_token_type_ids`
- train/eval/infer/debug 路径全部正确传递 `mm_token_type_ids`
- train forward 显式 `use_cache=False`
- 每次独立的 train/generate 调用前都清理模型对象上的 `rope_deltas`

本地已做一个最小回归：

- 先用 batch size `8` 走一次 eval-style `generate`
- 再立刻用 batch size `2` 走一次 train-style forward
- 原始崩溃未再出现
