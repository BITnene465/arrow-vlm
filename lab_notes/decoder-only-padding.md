在 decoder-only 的生成场景里，batch 推理必须使用 left padding，否则会触发错误警告，并且可能影响生成结果。

这次问题出现在推理链路而不是训练链路：训练侧可以继续用右侧 padding 做 teacher forcing，但 one-stage / two-stage 的 batched inference 在送入 `generate()` 前，必须把 tokenizer 的 `padding_side` 临时切到 `left`。

当前修复方式是只在推理侧的 processor 调用周围临时设置 `padding_side="left"`，避免污染训练配置和数据整理逻辑。
