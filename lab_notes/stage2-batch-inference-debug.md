# Stage2 Batch Inference Debug

## 现象

在 `ufv-exp2` 的两阶段批量推理里，很多 `Stage2` 结果显示为 `S2 fail`。

报告中的典型失败形式：

```json
{
  "lenient": {
    "ok": false,
    "error": "Point at index 0 must be [x, y]."
  },
  "strict": {
    "ok": false,
    "error": "Point at index 0 must be [x, y]."
  }
}
```

进一步把 `raw_text` 写入 report 之后，看到的原始输出并不是目标格式：

标准目标格式应该是：

```json
[[777,181],[167,839]]
```

而实际失败样本里经常出现：

```text
bbox_2d":[191,219,813,832]
```

或者：

```text
196,219,816,832]}\nOutput only the keypoints_2d skeleton of this arrow as a JSON array of points in [0,999]
```

更糟的样本还会出现：

```text
<|image_pad|><|image_pad|><|image_pad|>...
```

或者：

```text
<|endoftext|><|im_start|>user ...
```

这说明失败并不只是“模型没学会输出点列”，而是推理链本身把 prompt/chat template 残片混进了 continuation。

## 根因

根因出在 `Stage2` 的 **batched inference + left padding** 组合上。

在 `src/vlm_det/infer/two_stage.py` 里，旧实现用每条样本自己的：

- `attention_mask.sum()`

作为 continuation 的起点，并且 stopping criteria 也使用这个长度。

这在 **left padding** 场景下是错的。

对于 batched decoder-only generation，真正的生成起点应该是整批统一的：

- `input_ids.shape[1]`

也就是 padding 后的统一输入长度。

旧实现带来的两个直接后果：

1. continuation 切片太早  
   会把 prompt 尾部的一部分当成模型生成结果。

2. JSON array stopping criteria 被 prompt 污染  
   prompt 里本来就含有：

   ```json
   {"label":"single_arrow","bbox_2d":[191,219,813,832]}
   ```

   这里的 `bbox_2d` 自身就是一个合法的 `[...]` 数组。
   错误的 continuation 起点会让 stopping criteria 以为模型已经生成出了闭合 JSON array，于是提前停止。

这就能解释为什么很多样本同时满足：

- `closed_json_array = true`
- `stop_reason = "json_array_closed"`
- `raw_text` 却只是 `bbox_2d` 片段或 prompt 文本

## 修复

修复点在：

- `src/vlm_det/infer/two_stage.py`

具体改动：

1. `_prepare_inputs()` 不再返回逐样本 `prompt_lengths`
   而是返回统一的：

   - `input_context_length = batch["input_ids"].shape[1]`

2. `generate()` 之后的 continuation slicing 改成：

   - `output_ids[row_index, input_context_length:]`

3. `build_json_array_stopping_criteria(...)` 也统一使用：

   - `[input_context_length] * batch_size`

4. 额外把 `Stage2` 的 `raw_text` 写入 report，方便后续定位失败模式。

## 结论

这个问题不是普通的训练效果波动，而是一个明确的推理实现 bug。

在修复前：

- 不能用 `S2 fail` 比例来判断 `Stage2` 的真实能力
- 也不能用旧 report 判断 prompt/formulation 是否合理

在修复后，才可以重新跑批量推理并重新评估：

- `raw_text` 是否仍然在复述 prompt
- `Stage2` 真正的 fail 比例是多少
- 剩余问题究竟来自训练不足还是任务定义本身
