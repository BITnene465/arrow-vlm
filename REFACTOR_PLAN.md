# Refactor Plan

这份文档是当前仓库从“箭头任务实现”走向“生成式结构预测框架 + 任务域插件”的详细重构计划。

目标不是一次性把所有代码抽成完全通用框架，而是：

1. 先把 `core` 和 `task/domain` 的边界定义清楚
2. 在不打断现有训练/推理链的前提下分阶段迁移
3. 给后续多任务、多 domain、多 objective 留出扩展接口

---

## 1. 当前问题

当前仓库已经具备一层通用框架能力：

- 多模态 SFT 训练
- 生成式推理
- prompt 模板渲染
- JSON 文本协议
- 训练/评估/推理/可视化链路

但任务特化逻辑仍然散在：

- `dataset`
- `runner`
- `evaluator`
- `codec`
- `two_stage` 数据准备与推理

这会带来几个直接问题：

### 1.1 task 和 domain 混在一起

当前很多地方实际上把：

- `grounding`
- `keypoint_sequence`
- `joint_structure`

这些任务类型，和：

- `arrow`

这个具体对象域混在一起了。

结果是：

- 同一个 domain 下多个任务不干净
- 同一个任务换 domain 也不干净

### 1.2 数据路由不统一

当前很多地方隐式依赖：

- `task_type`
- prompt 内容
- 某个 codec 的存在

但没有一层正式的 registry / adapter 来做路由。

结果是：

- `dataset` 写死 target 生成逻辑
- `runner` 写死 decode 分支
- `evaluator` 写死指标分支

### 1.3 loss 扩展口不存在

当前 trainer 基本默认是标准 SFT / CE loss。

如果以后要做：

- bbox auxiliary loss
- point count loss
- matching / ranking loss
- consistency loss

现在没有一个正式接口去承接这些 objective。

### 1.4 two-stage 逻辑过深地嵌入框架层

`Stage1` / `Stage2` 的数据准备、crop 规则、proposal merge、visualize、评估逻辑都和当前箭头任务绑定很深。

这部分应该被明确视为 task/domain 层，而不是框架层。

---

## 2. 重构目标

重构后，仓库应拆成三层：

### 2.1 Core 层

负责通用能力：

- config
- dataset/collator 框架
- trainer
- infer runner 框架
- evaluator 框架
- builder
- utils

### 2.2 Task 层

负责“这次要做什么”：

- `grounding`
- `keypoint_sequence`
- `joint_structure`

每个 task 负责：

- prompt contract
- target_text 协议
- decode
- metric
- objective/loss

### 2.3 Domain 层

负责“这次在处理什么对象语义”：

- `arrow`

每个 domain 负责：

- label vocabulary
- schema 语义
- ordering
- prepare 规则
- visualize 语义

---

## 3. 关键抽象

### 3.1 task_type

定义任务目标和输出形式。

建议保留三类：

#### `grounding`

输出：

```json
[
  {"label":"single_arrow","bbox_2d":[x1,y1,x2,y2]}
]
```

当前对应：

- Stage1

#### `keypoint_sequence`

输出：

```json
{"keypoints_2d":[[x0,y0],[x1,y1],...]}
```

当前对应：

- Stage2

#### `joint_structure`

输出：

```json
[
  {
    "label":"single_arrow",
    "bbox_2d":[x1,y1,x2,y2],
    "keypoints_2d":[[x0,y0],[x1,y1],...]
  }
]
```

当前对应：

- one-stage 直接输出全部

### 3.2 domain_type

定义对象和语义规则。

当前只有：

- `arrow`

`arrow domain` 的职责是：

- 定义 `single_arrow` / `double_arrow`
- 定义 bbox + keypoints 语义
- 定义 canonical ordering
- 定义评估与可视化中的箭头解释方式

### 3.3 registry

需要新增一层 registry，根据：

- `task_type`
- `domain_type`

找到对应 adapter。

示意：

```python
adapter = registry.get(
    task_type=record["task_type"],
    domain_type=record["domain_type"],
)
```

---

## 4. 新目录草案

建议目标目录如下：

```text
src/vlm_det/
  config.py
  prompting.py

  core/
    registry.py

    data/
      collator.py
      dataset.py

    infer/
      config.py
      runner.py

    eval/
      evaluator.py

    train/
      trainer.py
      optim.py

    modeling/
      builder.py

    utils/
      checkpoint.py
      distributed.py
      generation.py
      io.py
      logging.py

  tasks/
    grounding/
      adapter.py
    keypoint_sequence/
      adapter.py
    joint_structure/
      adapter.py

  domains/
    arrow/
      schema.py
      ordering.py
      visualize.py
      prepare.py
      two_stage.py
      codecs/
        grounding.py
        keypoints.py
        structure.py
      eval/
        metrics.py
      infer/
        two_stage.py
```

说明：

- `tasks/` 负责“输出协议和训练目标”
- `domains/` 负责“对象语义和规则”
- `core/` 不应再直接理解箭头语义

---

## 5. Adapter 接口设计

每个 `task_type + domain_type` 组合，对应一个 adapter。

建议接口：

```python
class TaskAdapter(Protocol):
    def build_prompts(self, record: dict) -> tuple[str, str]:
        ...

    def build_target_text(self, record: dict) -> str:
        ...

    def decode_prediction(
        self,
        text: str,
        *,
        image_width: int,
        image_height: int,
        strict: bool = False,
    ) -> dict:
        ...

    def evaluate_prediction(
        self,
        prediction: dict,
        gt_struct: dict,
    ) -> dict[str, float]:
        ...

    def compute_loss(
        self,
        model_outputs,
        batch: dict,
    ):
        ...
```

其中：

- `build_prompts`
  - 根据 record 和模板生成 prompt
- `build_target_text`
  - 负责编码 target JSON/text
- `decode_prediction`
  - strict/lenient parse
- `evaluate_prediction`
  - 单样本指标
- `compute_loss`
  - 未来承接额外 objective

默认情况下：

- `compute_loss` 可以只是标准 CE loss 包装

---

## 6. 各子流程如何改

### 6.1 Dataset

#### 当前问题

当前 `dataset.py` 里混了：

- `GroundingCodec`
- `ArrowCodec`
- `two_stage_stage1_grounding`
- `label/bbox/keypoints` 的字段构造

#### 目标

`dataset` 只负责：

- 读 JSONL
- 打开图片
- 读 `task_type + domain_type`
- 调 adapter 构造：
  - `system_prompt`
  - `user_prompt`
  - `target_text`

#### 重构后流程

1. 读取一条 record
2. 如果 record 中已有：
   - `system_prompt`
   - `user_prompt`
   - `target_text`
   直接使用
3. 否则：
   - 根据 `task_type + domain_type` 找 adapter
   - 由 adapter 生成 prompt 和 target_text

#### dataset 最终不应再做的事

- 不直接 import 某个具体 codec
- 不直接判断：
  - `if task_type == ...`
- 不直接写死：
  - `label/bbox/keypoints`

---

### 6.2 Collator

#### 当前状态

`collator.py` 已经基本通用。

#### 目标

保持它尽可能不懂任务语义，只处理：

- image
- prompt
- target_text

#### 不应引入的职责

- 任何 bbox/keypoint 语义
- 任何 domain-specific 字段解释
- 任何 task-specific decode 逻辑

---

### 6.3 Runner

#### 当前问题

当前 `infer/runner.py` 里直接分支：

- `ArrowCodec`
- `GroundingCodec`

#### 目标

`runner` 只做：

- prepare inputs
- generate
- collect raw text
- 调 adapter decode

#### 重构后流程

1. build prompt
2. processor
3. generate
4. 根据 `task_type + domain_type` 找 adapter
5. adapter decode strict/lenient

#### runner 不应再做的事

- import task-specific codec
- 写死 strict/lenient parser 分支
- 直接知道箭头语义

---

### 6.4 Evaluator

#### 当前问题

当前 `evaluator.py` 把：

- one-stage structure
- stage1 grounding
- stage2 skeleton

的指标都写进了同一个类。

#### 目标

`core evaluator` 只负责：

- 批量 generate
- 聚合数值
- 分布式 reduce

每条样本怎么打分，交给 adapter。

#### 重构后流程

1. generate prediction text
2. adapter decode
3. adapter evaluate single sample
4. core evaluator 聚合 metrics

#### 需要支持的能力

- 多任务混训时按 task_type 路由
- 同一个 batch 内可能混有不同 task
- 保留当前进度条 summary 机制，但从 adapter 提供可聚合字段

---

### 6.5 Trainer

#### 当前问题

当前 trainer 基本默认标准 SFT loss。

#### 目标

在不打乱现有训练循环的前提下，给 future objectives 留出接口。

#### 重构后流程

1. batch 进入 trainer
2. 根据 `task_type + domain_type` 找 adapter
3. `model forward`
4. `adapter.compute_loss(...)`
5. backward / step

#### 兼容策略

第一阶段不需要立刻把所有 loss 拆出来。

可以先提供：

- `compute_loss(...)`

默认实现：

- 用现有 `labels` 做 CE loss

后续新任务再覆盖：

- bbox auxiliary
- point-count
- matching
- consistency

---

## 7. 配置层重构

### 当前问题

配置里现在混着：

- core 训练参数
- task prompt
- task schema 约束

### 目标

配置拆成两层语义：

#### core config

- model
- tokenizer
- train
- eval
- logging

#### task config

- `task_type`
- `domain_type`
- prompt 模板
- task-specific metric 参数
- task-specific infer 参数

建议最终配置至少显式包含：

```yaml
task:
  task_type: grounding
  domain_type: arrow
```

或者：

```yaml
task:
  task_type: keypoint_sequence
  domain_type: arrow
```

---

## 8. 数据层重构

### 强制新增字段

之后所有训练/验证样本都建议写入：

```json
{
  "task_type": "grounding",
  "domain_type": "arrow"
}
```

### 当前三类样本如何标记

#### one-stage 直接输出全部

```json
{
  "task_type": "joint_structure",
  "domain_type": "arrow"
}
```

#### stage1 grounding

```json
{
  "task_type": "grounding",
  "domain_type": "arrow"
}
```

#### stage2 skeleton

```json
{
  "task_type": "keypoint_sequence",
  "domain_type": "arrow"
}
```

### 为什么必须在数据准备阶段写入

因为这样：

- dataset 可以直接路由
- runner/evaluator/trainer 也能一致路由
- 不需要靠 prompt 文本或脚本路径去猜任务类型

---

## 9. 当前模块迁移建议

### 9.1 直接迁到 core

- `src/vlm_det/data/collator.py`
- `src/vlm_det/modeling/builder.py`
- `src/vlm_det/train/optim.py`
- `src/vlm_det/train/trainer.py`
- `src/vlm_det/utils/*`
- `src/vlm_det/infer/config.py`

### 9.2 改造成 core 后保留

- `src/vlm_det/data/dataset.py`
- `src/vlm_det/infer/runner.py`
- `src/vlm_det/eval/evaluator.py`

### 9.3 迁到 domains/arrow

- `src/vlm_det/protocol/schema.py`
- `src/vlm_det/data/ordering.py`
- `src/vlm_det/data/prepare.py`
- `src/vlm_det/data/two_stage.py`
- `src/vlm_det/infer/two_stage.py`
- `src/vlm_det/infer/visualize.py`
- `src/vlm_det/protocol/codec.py`
- `src/vlm_det/protocol/grounding_codec.py`
- `src/vlm_det/protocol/keypoint_codec.py`

### 9.4 task adapters 需要新建

- `src/vlm_det/tasks/grounding/adapter.py`
- `src/vlm_det/tasks/keypoint_sequence/adapter.py`
- `src/vlm_det/tasks/joint_structure/adapter.py`

---

## 10. 分阶段执行计划

### Phase 0: 文档和字段约定

目标：

- 只约定，不大改实现

任务：

1. 所有数据样本统一补：
   - `task_type`
   - `domain_type`
2. 文档同步

退出条件：

- 三类数据都已经能显式带上 `task_type + domain_type`

### Phase 1: 引入 registry 和 adapter 接口

目标：

- 不大规模迁移目录
- 先把路由接口建出来

任务：

1. 新建 `core/registry.py`
2. 新建三个 task adapter
3. 先让旧代码通过 registry 调用

退出条件：

- dataset/runner/evaluator 至少有一处已能通过 registry 获取 adapter

### Phase 2: 改 dataset

目标：

- dataset 去任务化

任务：

1. 让 `dataset` 不再直接 import 具体 codec
2. 让 `target_text` 生成交给 adapter
3. 保持训练链可运行

退出条件：

- `dataset.py` 中不再出现 arrow-specific 结构构造

### Phase 3: 改 runner

目标：

- runner 去 codec 分支

任务：

1. strict/lenient decode 交给 adapter
2. 保留现有报告结构

退出条件：

- `infer/runner.py` 不再直接判断 grounding / arrow codec

### Phase 4: 改 evaluator

目标：

- evaluator 只做壳子

任务：

1. 将具体 metrics 下沉到 adapter/domain
2. 保留当前汇总结果字段

退出条件：

- `evaluator.py` 不再直接写死当前三种任务的打分逻辑

### Phase 5: 引入 objective 接口

目标：

- 为 future loss 扩展留接口

任务：

1. trainer 支持 `adapter.compute_loss`
2. 默认实现继续走现有 CE loss

退出条件：

- 新任务可以不改 trainer 主循环就接入额外 loss

### Phase 6: 目录迁移

目标：

- 完成目录意义上的 core/task/domain 分层

任务：

1. 新建 `core/ tasks/ domains/`
2. 迁移模块
3. 旧路径保留 re-export 过渡层

退出条件：

- 目录结构与抽象边界一致

---

## 11. 风险与注意事项

### 11.1 不要一次性大搬家

目录迁移如果和逻辑迁移一起做，最容易把当前可运行链路打断。

建议：

- 先建 adapter
- 再换调用关系
- 最后才真正移动文件

### 11.2 保留兼容层

迁移初期，旧 import 路径建议保留薄包装：

- 旧模块 re-export 新模块

这样：

- 训练脚本不会立刻全炸
- demo/infer 也能渐进切换

### 11.3 不要先改 trainer

trainer 是框架主循环，最稳定。

真正的耦合点在：

- dataset
- runner
- evaluator

这些应该先改。

### 11.4 文档和配置必须同步

只要改：

- `task_type`
- `domain_type`
- prompt 协议
- target 协议

就必须同步：

- 数据准备
- codec
- evaluator
- infer/demo
- README/docs

---

## 12. 最终判断

当前仓库最合理的未来方向不是：

- 把一切都继续叫 `arrow`
- 或者把一切都强行泛化成无任务框架

而是：

- 先承认现在已经有一层可复用的生成式框架
- 再把箭头任务完整下沉成：
  - `task_type`
  - `domain_type`
 组合下的一组实现

最终目标是：

- `core` 稳定
- `tasks` 清晰
- `domains` 明确
- 新任务和新损失都能按接口接入

这是当前仓库最稳、也最可持续的重构方向。
