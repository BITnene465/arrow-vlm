# Developer Task Extension Guide

这份文档描述当前重构后的开发接口，目标读者是：

- 需要新增 `task_type`
- 需要新增 `domain_type`
- 需要把自己的 SFT JSON/图像任务接到当前训练与推理框架里

当前代码结构：

```text
src/vlm_structgen/
  core/
  tasks/
  domains/
```

职责划分：

- `core`
  - 通用训练 / 推理 / 评估 / 数据读取 / 路由
- `tasks`
  - 定义“输出什么、怎么 decode、怎么算指标、怎么接 loss”
- `domains`
  - 定义“对象语义是什么、label 集是什么、schema 长什么样、数据怎么准备”

---

## 1. 两个路由字段

每条样本都必须显式带：

```json
{
  "task_type": "...",
  "domain_type": "..."
}
```

这两个字段的职责必须解耦：

- `task_type`
  - 定义任务形式
  - 例如：
    - `grounding`
    - `keypoint_sequence`
    - `joint_structure`
- `domain_type`
  - 定义对象语义
  - 当前是：
    - `arrow`

不要把领域名当成 task 名，也不要把 task 名写进 domain。

---

## 2. 当前 public API

### 2.1 Core 入口

- [vlm_structgen.core](/home/tanjingyuan/code/arrow-vlm/src/vlm_structgen/core/__init__.py)
  - `load_config`
  - `apply_run_id`
  - `config_to_dict`
  - `get_adapter`
  - `TaskAdapter`
  - `SUPPORTED_TASK_TYPES`
  - `SUPPORTED_DOMAIN_TYPES`

### 2.2 Core 子模块

- [core.data](/home/tanjingyuan/code/arrow-vlm/src/vlm_structgen/core/data/__init__.py)
  - `SFTDataset`
  - `SFTCollator`
- [core.eval](/home/tanjingyuan/code/arrow-vlm/src/vlm_structgen/core/eval/__init__.py)
  - `Evaluator`
- [core.infer](/home/tanjingyuan/code/arrow-vlm/src/vlm_structgen/core/infer/__init__.py)
  - `InferenceRunner`
  - `load_inference_runner`
- [core.modeling](/home/tanjingyuan/code/arrow-vlm/src/vlm_structgen/core/modeling/__init__.py)
  - `BuildArtifacts`
  - `build_model_tokenizer_processor`
- [core.train](/home/tanjingyuan/code/arrow-vlm/src/vlm_structgen/core/train/__init__.py)
  - `Trainer`

### 2.3 Arrow domain 入口

- [vlm_structgen.domains.arrow](/home/tanjingyuan/code/arrow-vlm/src/vlm_structgen/domains/arrow/__init__.py)
  - `load_two_stage_inference_runner`
  - `TwoStageInferenceRunner`
  - `Stage2KeypointInferenceRunner`
  - `draw_prediction`
  - `format_prediction_summary`

---

## 3. 如果你要新增一个 task_type

例如你想新增：

- `counting`
- `mask_sequence`
- `bbox_and_attributes`

最小步骤如下。

### 3.1 新建 task adapter

目录建议：

```text
src/vlm_structgen/tasks/<your_task>/
  __init__.py
  adapter.py
```

这个 adapter 要实现 [TaskAdapter](/home/tanjingyuan/code/arrow-vlm/src/vlm_structgen/core/registry.py) 协议：

- `build_gt_struct_from_record(...)`
- `encode_target_text(...)`
- `decode(...)`
- `decode_with_meta(...)`
- `empty_prediction(...)`
- `score_prediction(...)`
- `compute_loss(...)`

说明：

- `compute_loss(...)` 当前默认可以直接返回 `model_outputs.loss`
- 如果你以后要加额外 loss 项，就在这里实现，不要去改 `Trainer` 主循环

### 3.2 在 registry 注册 task

文件：
- [core/registry.py](/home/tanjingyuan/code/arrow-vlm/src/vlm_structgen/core/registry.py)

你需要更新：

- `SUPPORTED_TASK_TYPES`
- `get_adapter(...)`

让新的 `task_type` 能正确路由到你的 adapter。

### 3.3 数据准备写入 task_type

在数据准备脚本里，把样本写成：

```json
{
  "task_type": "<your_task>",
  "domain_type": "<your_domain>"
}
```

不要把这个字段留空，也不要让 dataset 去猜。

---

## 4. 如果你要新增一个 domain_type

例如：

- `table`
- `chart`
- `ui_element`

最小步骤如下。

### 4.1 新建 domain 目录

建议：

```text
src/vlm_structgen/domains/<your_domain>/
  __init__.py
  schema.py
  data/
  infer/
```

是否需要 `ordering.py`、`codecs/`、`visualize.py`，取决于你的任务。

### 4.2 实现领域语义

domain 层应该承接这些东西：

- label vocabulary
- schema
- canonical ordering
- 领域特定的数据准备
- 领域特定的可视化

不要把这些逻辑塞到 `core`。

### 4.3 为现有 task 写 domain-specific adapter

例如你要支持：

- `task_type = grounding`
- `domain_type = table`

那就在：

- `tasks/grounding/adapter.py`

里加：

```python
if domain_type == "table":
    return TableGroundingAdapter(...)
```

也就是说：

- task 决定接口形状
- domain 决定具体语义实现

---

## 5. 如果你只想接入自己的 SFT 数据

这是最常见情况。

### 5.1 数据格式最低要求

每条 JSONL 至少需要：

```json
{
  "task_type": "...",
  "domain_type": "...",
  "image_path": "...",
  "image_width": 1234,
  "image_height": 567,
  "instances": [...],
  "condition": {...}
}
```

其中：

- `instances`
  - 由 task adapter 的 `build_gt_struct_from_record(...)` 消费
- `condition`
  - 用于 prompt 模板渲染
- `target_text`
  - 可选
  - 如果你已经离线生成好了，可以直接写进数据
  - 如果没写，dataset 会通过 adapter 在线编码

### 5.2 prompt 的使用方式

当前 [SFTDataset](/home/tanjingyuan/code/arrow-vlm/src/vlm_structgen/core/data/dataset.py) 支持两种方式：

1. 记录里直接给：
   - `system_prompt`
   - `user_prompt`

2. 或者给：
   - `system_prompt_template`
   - `user_prompt_template`
   - `condition`

第二种更适合 task/domain 统一管理。

### 5.3 什么时候写 `target_text`

建议：

- 如果你的输出协议已经稳定，且想避免运行时重复编码
  - 直接在数据准备阶段写 `target_text`
- 如果你还在频繁改 codec
  - 可以先让 adapter 在线编码

---

## 6. 训练链怎么路由

当前训练链的关键节点是：

### Dataset

- 读取 record
- 根据 `task_type + domain_type` 调 `get_adapter(...)`
- 生成：
  - `gt_struct`
  - `target_text`
  - prompt

文件：
- [core/data/dataset.py](/home/tanjingyuan/code/arrow-vlm/src/vlm_structgen/core/data/dataset.py)

### Collator

- 只负责图像 + prompt + target 的 batch 装配
- 不关心具体 task/domain 语义

文件：
- [core/data/collator.py](/home/tanjingyuan/code/arrow-vlm/src/vlm_structgen/core/data/collator.py)

### Trainer

- 默认走通用训练循环
- loss 通过 `adapter.compute_loss(...)` 留扩展口

文件：
- [core/train/trainer.py](/home/tanjingyuan/code/arrow-vlm/src/vlm_structgen/core/train/trainer.py)

### Evaluator

- 生成后按 `task_type + domain_type` 路由
- 调 adapter 的 `decode / score_prediction`

文件：
- [core/eval/evaluator.py](/home/tanjingyuan/code/arrow-vlm/src/vlm_structgen/core/eval/evaluator.py)

### InferenceRunner

- one-stage 通用 runner
- 生成后按 `task_type + domain_type` decode

文件：
- [core/infer/runner.py](/home/tanjingyuan/code/arrow-vlm/src/vlm_structgen/core/infer/runner.py)

---

## 7. 多任务混训的约束

当前 `Trainer` 已经留了多任务 loss 扩展口，但还没有做真正的 heterogeneous batch 调度。

现在的约束是：

- 一个 batch 内必须只有一种：
  - `task_type`
  - `domain_type`

否则 `Trainer` 会直接报错。

这是故意保守的设计，因为：

- 不同 task 的 target 协议不同
- 不同 task 的 loss 也可能不同

如果你以后要支持真正的多任务混训，建议按这条线扩展：

1. sampler 先做 task-aware batching
2. collator 保持不变
3. trainer 继续通过 `adapter.compute_loss(...)` 路由

不要先去改 collator。

---

## 8. 新任务落地时的推荐顺序

1. 先定义：
   - `task_type`
   - `domain_type`
2. 先写 adapter
3. 再写 codec / metrics
4. 再写数据准备
5. 再接训练
6. 最后接 demo / visualize

不要反过来。

---

## 9. 当前三类任务的对应关系

### one-stage 直接输出全部

- `task_type = joint_structure`
- `domain_type = arrow`

### Stage1

- `task_type = grounding`
- `domain_type = arrow`

### Stage2

- `task_type = keypoint_sequence`
- `domain_type = arrow`

这三条线都已经是新结构下的标准示例。
