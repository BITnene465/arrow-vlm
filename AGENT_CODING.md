# AGENT_CODING

最后更新：2026-03-21

## 目标

本文件定义当前项目的工程实现规范，重点约束：

- 训练与推理主链路
- 技术栈选型
- 数据集接口
- collator 接口
- protocol codec 接口
- 各模块职责与边界

本文件面向后续实现，不再讨论任务定义本身。任务语义与标注约束见 `CONTEXT.md`。

## 技术栈

当前训练主框架固定为：

- `PyTorch`
- `transformers`
- `peft`
- `torch.distributed` + `DDP`

当前明确不作为主线采用：

- `trl`
- `accelerate`
- 传统检测框架（如 YOLO / MMDetection / Detectron2）

## 训练范式

- 模型：`Qwen3-VL-2B-Instruct`
- 训练方式：VLM 结构化生成式微调
- 微调分支：
  - `finetune.mode = lora`
  - `finetune.mode = full`
- 输入粒度：整图
- 输出目标：整图中全部 `arrow` 的结构化协议序列
- 训练监督方式：teacher forcing
- 验证/推理生成方式：greedy decoding

## 训练与推理总流程

### 训练

1. `Dataset` 返回单样本语义对象
2. `Collator` 使用 `processor` 构造图像 + 对话模板前缀
3. `Collator` 使用 `ArrowCodec.encode(...)` 获取 `target_text`
4. `Collator` 手工拼接 `prefix_ids + target_ids`
5. `Collator` 手工构造 `labels`
6. DDP 训练循环执行 teacher forcing，优化 token-level CE

### 推理

1. `Dataset` 或推理入口提供图像与 prompt
2. `Collator` 或推理构造器构建 prefix
3. `model.generate(...)` 执行 greedy decode
4. 使用 `ArrowCodec.decode(...)` 将输出协议还原为结构化实例
5. 使用原图尺寸完成反量化与评估/可视化

## Prompt 与监督规则

### System Prompt

当前保留完整 system prompt，包含：

- task token：`<|arrow_task|>`
- 一句自然语言任务定义

该 prompt 属于条件输入，不属于直接监督目标。

### Label Mask

固定规则：

- 图像相关 token：`-100`
- system prompt token：`-100`
- 若未来加入 user prompt，其 token 也设为 `-100`
- 只有 assistant target protocol 部分参与 loss
- padding token：`-100`

说明：

- `task token` 虽然不直接参与 label loss，但其 embedding 会通过后续 target token 的 loss 反传获得训练信号。

## Tokenizer 与模型改造

### 扩词表

需要一次性注册全部协议 token：

- 结构 token
- `<|x_0000|> ~ <|x_2047|>`
- `<|y_0000|> ~ <|y_2047|>`

约束：

- `x/y` token 固定 4 位零填充
- token 统一追加到词表尾部
- 可以预留未来命名空间，但当前只注册当前确定需要的 token

### 模型改造

必须执行：

- tokenizer 扩词表
- `resize_token_embeddings(...)`
- LM head 输出维度同步扩展

新增 token 参数为冷启动参数，必须训练。

### 参数训练策略提示

当前训练设计不能采用“纯 LoRA + 全冻结 embedding/lm_head”。

至少应保证以下部分可训练：

- 新增 token 的 embedding
- LM head 中与新增 token 相关的参数
- LoRA 挂载模块

当前建议：

- 主干大部分冻结
- 视觉塔冻结
- LoRA 可训练
- `embed_tokens` / `lm_head` 至少部分可训练

### 微调模式约定

- `finetune.mode = lora`
  - 训练 `LoRA + embed_tokens + lm_head`
  - 若显式开启 `train_projector`，projector 也参与训练
- `finetune.mode = full`
  - 训练全模型
  - 若 `freeze_vision_tower = true`，则视觉塔仍冻结
  - 该模式不依赖 LoRA

## 坐标协议

- 坐标基于原图归一化
- bin 数固定：`B = 2048`
- 使用离散位置 token，而不是裸数字

量化：

```text
qx = round(x / (W - 1) * 2047)
qy = round(y / (H - 1) * 2047)
```

反量化：

```text
x = qx / 2047 * (W - 1)
y = qy / 2047 * (H - 1)
```

## Protocol 模块

### 命名

核心协议模块固定命名为：

- 模块：`src/protocol/codec.py`
- 核心类：`ArrowCodec`

不再使用泛化命名如 `parser`、`formatter`、`serializer` 作为核心抽象名。

### 职责

`ArrowCodec` 负责协议级编解码与校验，不负责原始数据清洗，不负责 batch 构造，不负责训练优化。

应承担：

- 结构化 GT -> 协议文本/序列
- 协议文本/序列 -> 结构化实例
- 协议合法性校验
- 可选 round-trip 检查

### 不承担

- LabelMe 原始解析
- 图像读取
- processor 调用
- labels mask 逻辑
- 评估指标计算
- 分布式训练逻辑

### 形式接口

建议至少提供：

- `encode(gt_struct, image_width, image_height) -> str`
- `decode(text, image_width, image_height) -> decoded_struct`
- `validate_struct(gt_struct) -> bool | report`
- `validate_text(text) -> bool | report`

### 协议语义

`ArrowCodec` 输入输出所使用的结构化对象统一为：

```python
{
    "instances": [
        {
            "bbox": [x1, y1, x2, y2],
            "keypoints": [
                [x0, y0, "visible"],
                [x1, y1, "occluded"],
                [x2, y2, "visible"]
            ]
        }
    ]
}
```

说明：

- `bbox` 与 `keypoints` 均以原图像素坐标表示
- `ArrowCodec` 内部负责量化与反量化
- 空结果应编码为：

```text
<|arrows_begin|>
<|arrows_end|>
```

## Dataset 模块

### 命名

- 模块建议：`src/data/dataset.py`
- 核心类建议：`ArrowSFTDataset`

### 职责

`ArrowSFTDataset` 只负责提供单样本语义数据，不负责 batch 化，不负责 tokenizer/processor，不负责 labels 构造。

### 单样本输出契约

每个样本至少返回：

- `sample_id`
- `image_path`
- `image`
- `image_width`
- `image_height`
- `system_prompt`
- `target_text`
- `gt_struct`

字段说明：

- `sample_id`
  - 建议使用图像 stem 或稳定唯一 id
- `image_path`
  - 原图路径，供 debug / 可视化使用
- `image`
  - `PIL.Image` 或等价图像对象
- `image_width` / `image_height`
  - 原图尺寸
- `system_prompt`
  - 固定 system prompt，含 `<|arrow_task|>`
- `target_text`
  - `ArrowCodec.encode(gt_struct, image_width, image_height)` 的结果
- `gt_struct`
  - 规范化后的结构化 GT

### Dataset 不承担

- tokenization
- processor 调用
- chat template 拼接
- labels 生成
- 分布式采样逻辑

## Collator 模块

### 命名

- 模块建议：`src/data/collator.py`
- 核心类建议：`ArrowSFTCollator`

### 职责

`ArrowSFTCollator` 负责将一批语义样本转换为模型训练/验证所需张量 batch。

### 核心规则

- 使用 `processor` 处理图像与对话模板
- 只让 `processor` 负责 prefix
- target token 由项目代码自己控制
- labels 构造必须由项目代码自己控制

### 处理流程

对一个 batch：

1. 为每个样本构造消息：
   - `system`: `system_prompt`
   - `user`: 图像
   - 不在 chat template 内放 assistant target
2. 使用 processor/chat template 构造 prefix
3. 单独 tokenize `target_text`
4. 拼接：
   - `input_ids = prefix_ids + target_ids`
5. 构造 labels：
   - prefix 对应位置全为 `-100`
   - target 对应位置为真实 token id
6. 对 batch 执行 padding

### Batch 输出契约

至少输出：

- `input_ids`
- `attention_mask`
- `labels`
- `pixel_values`
- `image_grid_thw`
- `meta`

### `meta` 字段

建议至少保留：

- `sample_id`
- `image_path`
- `image_width`
- `image_height`
- `gt_struct`
- `target_text`

### Collator 不承担

- 训练循环
- loss 计算
- 指标计算
- 预测解析

## 模型构建模块

### 职责

模型构建层负责将 pretrained checkpoint 变成 ready-to-train model。

### 应承担

- 加载模型
- 加载 tokenizer / processor
- 注册 special token
- resize embeddings
- 挂载 LoRA
- 设置 trainable 参数
- 输出 trainable parameter summary

### 不承担

- 数据读取
- batch 构造
- 训练 step
- 评估逻辑

## 训练模块

### 职责

训练模块负责优化过程本身，不负责业务协议实现。

### 应承担

- DDP 初始化与清理
- sampler / epoch 管理
- optimizer / scheduler 构建
- mixed precision 控制
- train step
- grad clip
- checkpoint 保存与恢复
- 日志记录

### 不承担

- LabelMe 清洗
- 协议编解码
- 指标定义本身

### 训练器命名与接口

建议：

- 模块：`src/train/trainer.py`
- 核心类：`ArrowTrainer`

`ArrowTrainer` 是训练主控对象，负责训练、验证、保存、恢复与日志调度。

建议至少提供以下接口：

- `fit()`
- `train()`
- `train_one_epoch(epoch: int)`
- `train_one_step(batch) -> dict`
- `evaluate(step: int | None = None, epoch: int | None = None) -> dict`
- `save_checkpoint(tag: str | None = None, is_best: bool = False)`
- `load_checkpoint(path: str, strict: bool = True, resume_training_state: bool = True)`

接口约束：

- `train_one_step(...)` 返回扁平日志字典
- `evaluate(...)` 返回可直接写入终端日志与 wandb 的指标字典
- `fit()` 作为统一入口，内部调度训练、验证与保存

### 训练器依赖输入

`ArrowTrainer` 构造时应至少接收：

- `model`
- `tokenizer`
- `processor`
- `train_dataloader`
- `val_dataloader`
- `optimizer`
- `scheduler`
- `config`
- `device`
- `rank`
- `world_size`

可选依赖：

- `evaluator`
- `logger`
- `wandb_run`

### 参数组要求

optimizer 构建层必须支持独立参数组，至少区分：

- `embed_tokens`
- `lm_head`
- `lora_params`
- 其他可训练参数

要求：

- 参数组命名稳定
- 便于日志打印与 checkpoint 恢复检查
- 允许后续为不同参数组设置不同学习率与 weight decay

### 训练循环要求

训练循环必须支持：

- DDP
- bf16 或 autocast 混合精度
- gradient accumulation
- grad clip
- step 级日志
- 周期性验证
- 周期性保存 checkpoint

当前不强制支持但应预留扩展点：

- EMA
- curriculum learning
- 多数据源混采
- 自定义 loss weighting

## 日志与监控

### 终端日志

必须提供简单、稳定、可读的终端日志。

建议至少输出：

- `epoch`
- `global_step`
- `loss`
- `lr`
- `grad_norm`
- `tokens_per_sec` 或 `samples_per_sec`
- `gpu_mem`（若易于获取）

约束：

- 仅 `rank0` 打印主要日志
- 日志结构保持扁平
- 训练与验证日志命名风格统一，例如：
  - `train/loss`
  - `train/lr`
  - `val/parse_rate`
  - `val/bbox_ap50`

### tqdm 进度条

必须提供 `tqdm` 进度条。

要求：

- 仅 `rank0` 显示
- 固定宽度，不要过长
- 可以适当美化，但不能影响稳定性与可读性

建议：

- `ncols` 固定在约 `100~120`
- 使用简洁 `bar_format`
- 优先展示：
  - `epoch`
  - `step/total`
  - `loss`
  - `lr`

应避免：

- 刷新频率过高导致终端抖动
- 过长进度条遮挡日志

### wandb

必须接入 `wandb`。

固定要求：

- `project` 名称固定为：`vlm_det`
- 仅 `rank0` 初始化与写入主要日志
- 配置、训练指标、验证指标、checkpoint 元信息应同步记录

至少记录：

- 配置摘要
- 训练曲线
- 验证指标
- 当前 best checkpoint 对应指标
- 后续如实现可视化，记录样本图或样本路径

要求：

- 允许通过配置关闭 wandb 联网写入，但接口始终保留
- trainer 不应将 wandb 调用散落在多个低层模块中
- 建议统一由 logger 或 trainer 边界集中上报

## Checkpoint 规范

### 总体原则

checkpoint 必须保证：

- 可恢复训练
- 可离线评估
- 可追溯 token 协议版本
- 可区分模型权重与训练状态

### 建议目录结构

建议在实验目录下组织为：

- `checkpoints/last/`
- `checkpoints/best/`
- `checkpoints/step_{global_step}/`

其中：

- `last/`：最近一次可恢复训练状态
- `best/`：当前最佳验证指标对应状态
- `step_xxx/`：按策略保留的历史快照

### 每个 checkpoint 应包含

至少包含以下内容：

- 模型权重
- tokenizer 文件
- processor 配置
- special token 配置快照
- LoRA 配置
- 训练配置快照
- optimizer state
- scheduler state
- scaler/autocast state（如使用）
- RNG state
- 当前：
  - `epoch`
  - `global_step`
  - `best_metric`

### 推荐文件内容

建议语义上拆成：

- `model/`
- `tokenizer/`
- `processor/`
- `trainer_state.json`
- `optimizer.pt`
- `scheduler.pt`
- `rng_state.pt`
- `meta.json`

`meta.json` 建议至少记录：

- 实验名
- 创建时间
- git commit（若可获取）
- special token 总数
- 协议版本
- 数据版本或 manifest 路径
- 保存时的关键验证指标

### 保存策略

必须支持：

- 按 step 保存
- 按 epoch 保存
- 保存 `last`
- 保存 `best`

建议：

- `last` 总是覆盖
- `best` 仅在主监控指标提升时覆盖
- 历史 step checkpoint 按保留策略清理

### 恢复策略

必须支持两类恢复：

1. 仅加载模型用于推理/评估
2. 完整恢复训练状态继续训练

要求：

- 恢复前校验 tokenizer 与 special token 配置一致
- 恢复前校验协议相关配置与 checkpoint 元信息兼容
- 若仅加载模型评估，可跳过 optimizer/scheduler/RNG 恢复

### 恢复严格性

恢复接口应支持：

- `strict=True`
- `strict=False`

推荐语义：

- `strict=True`
  - tokenizer / 协议 / 关键模型形状不一致时报错
- `strict=False`
  - 允许在受控情况下跳过非关键训练状态

### 最佳指标约定

trainer 必须配置一个主监控指标，用于：

- 判定 `best checkpoint`
- 在 wandb 中标记最佳 step

当前允许该指标在配置中指定，默认应选一个验证主指标，例如：

- `val/parse_rate`
- 或 `val/end_to_end_score`

## 评估模块

### 职责

评估模块负责将模型输出恢复为结构化预测并计算指标。

### 应承担

- greedy generate
- 调用 `ArrowCodec.decode(...)`
- 反量化
- bbox matching
- keypoint / visibility 指标
- parse success 统计
- 可选可视化抽样

### 说明

- 当前不单独定义“预测 parser”抽象。
- 评估统一直接使用 `ArrowCodec.decode(...)`。

### 评估器接口

建议：

- 模块：`src/eval/evaluator.py`
- 核心类：`ArrowEvaluator`

至少提供：

- `evaluate_model(model, dataloader) -> dict`
- `evaluate_batch(predictions, batch_meta) -> dict | intermediate`
- `summarize(accumulator) -> dict`

`ArrowEvaluator` 负责：

- greedy generate
- 调用 `ArrowCodec.decode(...)`
- 协议解码失败统计
- 指标聚合

`ArrowEvaluator` 不负责：

- checkpoint 保存
- DDP 初始化
- 模型结构改造

## 原始数据处理

### 约束

- 原始 LabelMe 解析与清洗通过独立脚本完成
- 不作为长期在线模块持久化为核心抽象
- 输出为规范化中间格式，再交由 `Dataset` 使用

### 原始数据处理脚本应完成

- 图像路径修复
- `group_id` 聚合
- bbox / keypoint 校验
- `visible/occluded` 映射
- 越界裁剪
- train/val 随机切分
- 生成 normalized jsonl

## 目录建议

建议职责划分如下：

- `configs/`
- `src/modeling/`
- `src/protocol/`
- `src/data/`
- `src/train/`
- `src/eval/`
- `src/utils/`
- `scripts/`

要求：

- `scripts/` 只做薄入口
- 核心逻辑放在 `src/`

## 关键边界总结

### `ArrowCodec`

负责：

- 协议编解码
- 协议校验

不负责：

- raw 清洗
- batch 构造
- 训练逻辑

### `ArrowSFTDataset`

负责：

- 单样本语义数据产出

不负责：

- tokenization
- labels
- batch

### `ArrowSFTCollator`

负责：

- prefix 构造
- target tokenization
- labels 构造
- batch padding

不负责：

- 优化
- 指标
- 业务清洗

### 训练器

负责：

- DDP
- 优化
- 保存恢复

不负责：

- 协议定义
- 原始数据清洗

### 评估器

负责：

- generate
- decode
- metric

不负责：

- 模型结构改造
- 数据清洗

## 当前实现原则

- 使用现成框架解决通用问题
- 使用项目代码掌控任务专属问题
- 不依赖黑盒 trainer 自动推断 labels
- 不依赖传统检测头框架
- 先做薄而稳的实现，再考虑混合训练、增强和更复杂策略
