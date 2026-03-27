在 decoder-only 的生成场景里，batch 推理必须使用 left padding，否则会触发错误警告，并且可能影响生成结果。

这次问题出现在推理链路而不是训练链路：训练侧可以继续用右侧 padding 做 teacher forcing，但 one-stage / two-stage 的 batched inference 在送入 `generate()` 前，必须把 tokenizer 的 `padding_side` 临时切到 `left`。

当前修复方式是只在推理侧的 processor 调用周围临时设置 `padding_side="left"`，避免污染训练配置和数据整理逻辑。

简单例子：

- right padding
  - `[a b c PAD PAD]`
  - `[x y z w q]`
- left padding
  - `[PAD PAD a b c]`
  - `[x y z w q]`

对 decoder-only 生成来说，模型是从“序列真实结尾”继续往后写。left padding 会让 batch 内所有样本的真实结尾都对齐在最右侧，更适合 `generate()`。right padding 虽然在训练时通常没问题，但在 batched generation 中更容易出现警告，严重时会影响结果稳定性。

训练和推理的差别：

- 训练：teacher forcing，整条目标序列已经给出，只要 `attention_mask` 和 `labels` 正确，right padding 通常可以正常工作。
- 推理：自回归生成，模型需要一步一步从当前真实结尾继续写，因此 decoder-only 架构对 padding 方向更敏感，batch 推理应使用 left padding。
