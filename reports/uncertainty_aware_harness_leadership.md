# Uncertainty-Aware Scientific Judge Harness

## 0. 当前进展

### 0.1 核心结论

- 本项目研究的不是普通的答案评分，而是：**科学智能体在信息不充分、模型可能犯错且实验预算有限时，如何选择下一步值得执行的科学行动。**
- 我们已经实现一套 **Uncertainty-Aware Scientific Judge Harness**，将 LLM 或本地模型给出的判断信号，与科学效用、模型不确定性、验证价值、执行成本和失败风险统一起来。
- Harness 的最终输出不是单一分数，而是可执行决策：`execute`、`verify`、`stop` 或保守 fallback。
- 我们 follow **Recursive Harness Self-Improvement（RHI，arXiv:2607.15524）**，使系统能够根据失败轨迹修改 Harness 的判断协议、使用信号、角色分工、决策路由和接受规则。
- 当前实验支持的主要结论是：**Scientific VoI Harness 明显优于 reliability-only Judge/Harness；单独依赖 verbal confidence 虽然可能获得较高 raw utility，但科学风险过高。**
- 当前实验尚不能证明所有 RHI mutation 都会自动提升性能，也尚未完成统一的 `GPT-5.5 + internal/external uncertainty + VoI + LLM-proposer RHI` 全量实验。

### 0.2 已完成工作

| 工作模块 | 当前状态 | 已完成内容 |
| --- | --- | --- |
| Scientific Judge | 已完成 | 将答案评价改造成候选科学行动评价 |
| Scientific VoI | 已完成 | 联合建模 reliability、utility、uncertainty、cost、risk 和 verification value |
| 决策路由 | 已完成 | 支持 execute、verify、stop 和 fallback |
| RHI Pipeline | 已完成 | 支持 Harness mutation、schema validation、acceptance、rollback 和 checkpoint evaluation |
| 历史 frozen 主实验 | 已完成 | 15,717 条历史代理行动记录，5 seeds，共 105 个 held-out-regime folds |
| 新版材料 benchmark | 数据已完成 | 20,987 条材料/化学行动记录，45 个 regimes，已完成泄漏和标签审计 |
| 新版 benchmark 全量重跑 | 未完成 | 当前仅完成数据审计和 Matbench pairwise smoke test |
| GPT-5.5 direct judge + static VoI | 子集实验完成 | 在 500-record subset 上改善综合分数和校准误差 |
| 开放模型 logits uncertainty | 接口与 pilot 已完成 | 尚未与 full benchmark 和 RHI 完整统一 |
| LLM-driven Harness proposer | 接口已完成 | 尚未完成全量 LLM-proposer RHI 实验 |

### 0.3 当前主结果

| Method | Net utility ↑ | Risk-adjusted utility ↑ | Selective risk ↓ | Hit rate ↑ |
| --- | ---: | ---: | ---: | ---: |
| Verbal Confidence | 0.7577 | 0.5605 | 0.7889 | 0.3767 |
| LLM Judge | 0.5990 | 0.4709 | 0.5123 | 0.2429 |
| LLM Judge + VoI | 0.6372 | 0.5410 | 0.3845 | 0.2398 |
| **LLM Judge + VoI + RHI** | **0.6443** | **0.6043** | **0.1600** | 0.2452 |

为保证汇报命名清晰，主文统一使用以上四个方法名。技术上，15,717 条 frozen benchmark 中的 `LLM Judge` 对应可离线执行的 reliability-oriented Judge proxy；`LLM Judge + VoI` 对应静态 Scientific VoI policy；`LLM Judge + VoI + RHI` 对应表现最好的递归 Harness policy。真正使用 GPT-5.5 API 的 direct judge 与 hybrid 结果在后文单独列出，避免把离线 proxy 实验误写成四组 GPT-5.5 调用。

### 0.4 对结果的准确解释

- `Verbal Confidence` 的 raw utility 最高，但 selective risk 达到 0.7889，说明它会放行大量不可靠行动，不能作为安全科学决策策略。
- `LLM Judge` 只判断行动是否可靠，没有显式考虑行动价值、验证收益和执行成本，因此 risk-adjusted utility 只有 0.4709。
- `LLM Judge + VoI` 将 scientific utility、uncertainty、cost 和 verification value 加入决策后，net utility 从 0.5990 提升到 0.6372，selective risk 从 0.5123 降至 0.3845。
- `LLM Judge + VoI + RHI` 的 risk-adjusted utility 最高，为 0.6043；selective risk 最低，为 0.1600，是当前完整主实验中表现最好的安全策略。
- 相比 `LLM Judge + VoI`，加入 RHI 后 raw utility 增益较小，但风险明显下降。因此，当前 RHI 的主要价值表现为优化决策协议和风险控制，而不是单纯提高 raw utility。

---

## 1. 研究背景

### 1.1 现有问题

普通 LLM-as-a-Judge 主要处理以下问题：

- 答案是否正确；
- 两个回答中哪个更好；
- 输出是否符合指令；
- 模型对当前回答有多大 confidence。

科学智能体面对的问题不同：

- 是否应该执行一个实验；
- 是否应该运行一次昂贵模拟；
- 是否应该继续检索文献或调用工具；
- 是否应该对一个高价值但高风险的候选进行验证；
- 是否应该停止当前方向，避免继续消耗资源。

因此，科学行动选择同时涉及以下因素：

| 因素 | 需要回答的问题 |
| --- | --- |
| Reliability | 这个行动大概率正确或可行吗？ |
| Scientific utility | 如果成功，它能产生多大科学价值？ |
| Uncertainty | 当前判断有多不确定？不确定性来自哪里？ |
| Verification value | 额外验证是否可能改变最终决策？ |
| Cost | 执行或验证需要多少实验、计算和时间成本？ |
| Risk | 错误执行会造成多大损失？ |
| Budget | 在只能选择少量行动时，应优先选择哪些？ |

### 1.2 Research Question

> 在未知科学 regime、有限行动预算和不完备证据条件下，如何构建一个可执行、可校准、可递归改进的 Scientific Judge Harness，使其能够综合模型判断、科学效用、不确定性、验证价值、成本与风险，选择应该执行、验证或停止的下一步科学行动？

### 1.3 为什么 Direct LLM Judge 不够

- Direct LLM Judge 擅长利用语义信息和科学背景，但自报 confidence 不一定经过校准。
- 高 confidence 不等价于高 scientific utility。
- 高 reliability 不等价于值得占用实验预算。
- Direct Judge 通常只能输出 accept/reject 或一个分数，缺少“先验证”的中间决策。
- 模型可能在证据冲突、分布外样本或工具结果不一致时仍保持高 confidence。
- 单独依赖 confidence 容易产生 confidently-wrong actions。

实验中的 `verbal_confidence` 基线验证了这一问题：

| 指标 | Verbal Confidence | LLM Judge + VoI + RHI |
| --- | ---: | ---: |
| Net utility | **0.7577** | 0.6443 |
| Risk-adjusted utility | 0.5605 | **0.6043** |
| Selective risk | 0.7889 | **0.1600** |

该结果说明：

- confidence 具有一定排序信息；
- 但 confidence 不能直接等价为安全决策；
- 科学行动系统必须同时评价效用和风险；
- raw utility 不能作为唯一结论，必须结合 risk-adjusted utility 和 selective risk。

### 1.4 为什么需要 VoI

Value of Information 解决的是：**当系统不确定时，额外获得一次信息是否值得。**

| 情况 | Reliability-only 策略 | VoI 策略 |
| --- | --- | --- |
| 高价值、低风险 | 执行 | 执行 |
| 低价值、高可靠 | 可能执行 | 可能停止，避免浪费预算 |
| 高价值、高不确定 | 可能直接拒绝 | 先验证，再决定是否执行 |
| 低价值、高不确定 | 拒绝 | 停止或 fallback |
| 证据冲突但验证成本低 | 难以处理 | 路由到检索、模拟或专家复核 |

VoI 的核心作用是把 uncertainty 从一个被动惩罚项，转化为主动验证决策的依据。

### 1.5 为什么需要 RHI

- 固定 prompt 无法保证在不同任务、模型和 scientific regime 上都有效。
- 不同任务可能需要不同 uncertainty signals、成本模型和验证路径。
- Harness 会出现系统性失败，例如过度信任模型 confidence、忽略证据冲突、错误估计成本或缺少验证路径。
- 如果 Harness 固定，这些失败只能依靠人工不断修改 prompt 和规则。
- RHI 将 Harness 本身作为可优化对象，根据执行轨迹和失败模式自动提出下一版本。

---

## 2. 方法

### 2.1 Follow 的论文与我们的扩展

本项目 follow **Recursive Harness Self-Improvement（RHI，arXiv:2607.15524）** 的 Harness-level recursive improvement 范式。

| 维度 | 原始 RHI | 本项目 Scientific VoI-RHI |
| --- | --- | --- |
| 优化对象 | Agent Harness | Scientific Judge Harness |
| 基础模型 | 不修改模型权重 | 不修改模型权重 |
| 反馈来源 | 输出偏好或任务评价 | 行动效用、风险、错误类型和验证结果 |
| Harness 内容 | Prompt、角色、上下文和交互协议 | Judge schema、信号、角色、gates、cost、risk 和 routing |
| 主要目标 | 产生更优输出 | 选择更有价值且更安全的科学行动 |
| 输出 | 新一版 Harness | 新一版可执行 Scientific VoI Harness |

我们的新增内容包括：

- 将一般输出评价改造成 scientific action selection；
- 引入 continuous scientific utility，而不只使用 binary correctness；
- 引入 epistemic/model uncertainty；
- 引入 action cost 和 failure risk；
- 引入 Value-of-Information verification routing；
- 将输出定义为 execute、verify、stop 或 fallback；
- 在完全未参与 Harness evolution 的 scientific regimes 上评价泛化能力。

### 2.2 整体框架

```text
Scientific state + candidate action + evidence
                       ↓
          LLM Judge / local predictors
                       ↓
 reliability + utility + uncertainty + cost + risk
                       ↓
            Scientific VoI Harness
                       ↓
        execute / verify / stop / fallback
                       ↓
       execution feedback and failure traces
                       ↓
                 RHI proposer
                       ↓
         candidate Harness H_(t+1)
                       ↓
       validation + acceptance / rollback
```

### 2.3 输入

| 输入类别 | 具体内容 |
| --- | --- |
| Scientific context | 当前研究目标、任务背景和状态 |
| Candidate action | 拟执行的实验、模拟、检索、筛选或工具调用 |
| Evidence | 支持证据、反对证据、证据数量和证据冲突 |
| Model signals | LLM confidence、开放模型 logits、sample disagreement |
| External signals | tool agreement、source reliability、perturbation stability |
| Distribution signals | OOD score、与训练 regime 的差异 |
| Operational signals | cost、reversibility、action complexity |

隐藏的 benchmark label、真实结果和 oracle utility 不会作为 Judge 输入，只用于训练监督和最终评价。

### 2.4 输出

| 输出 | 含义 |
| --- | --- |
| Reliability | 行动成功、正确或值得执行的概率估计 |
| Expected utility | 行动成功后预期产生的科学价值 |
| Epistemic uncertainty | 当前模型判断由于知识或分布不足产生的不确定性 |
| Verification value | 增加一次检索、模拟或复核可能带来的决策改进 |
| Estimated cost | 执行行动或获得额外证据的成本 |
| Risk | 错误执行的预期损失 |
| Route | execute、verify、stop 或 fallback |

### 2.5 Scientific VoI 决策层

| 决策 | 触发条件 | 系统行为 |
| --- | --- | --- |
| Execute | 预期效用高、风险可接受、验证增益有限 | 直接执行候选科学行动 |
| Verify | 潜在效用高，但 uncertainty 或 evidence conflict 较高 | 先检索、模拟、工具复核或专家验证 |
| Stop | 预期净效用低，或风险与成本不可接受 | 停止继续投入 |
| Fallback | 当前信号不足以稳定判断 | 转入更保守的预定义路径 |

Scientific VoI 与 reliability-only gate 的区别是：

- reliability-only 只判断“是否可能正确”；
- Scientific VoI 判断“是否值得执行、是否值得验证、是否应该停止”；
- 一个高可靠但低价值行动可能被停止；
- 一个高价值但高不确定行动可能被路由到验证，而不是直接拒绝。

### 2.6 Uncertainty 信号

| 信号类型 | 开放模型 | 闭源模型/API | 当前作用 |
| --- | --- | --- | --- |
| Token logits / margin | 可获得 | 通常不可完整获得 | 估计候选判断的内部概率间隔 |
| Entropy | 可由 logits 计算 | 取决于 API | 估计输出分布不确定性 |
| Verbal confidence | 可获得 | 可获得 | 作为可观测 self-report signal |
| Sampling disagreement | 可获得 | 可通过多次调用获得 | 估计判断稳定性 |
| Model disagreement | 可获得 | 可获得 | 比较多个模型或多个 Judge |
| Evidence conflict | 可获得 | 可获得 | 衡量外部证据冲突 |
| Tool agreement | 可获得 | 可获得 | 判断模型与工具结果是否一致 |
| Utility ensemble disagreement | 可获得 | 与 LLM 接口无关 | 本地 epistemic uncertainty proxy |

当前结论边界：

- 已经实现开放模型 logits provider；
- 已经实现闭源模型 self-reported confidence provider；
- 已经实现 uncertainty signal 的统一接口；
- 完整主实验主要验证 Harness-level uncertainty routing；
- 尚不能声称获得 GPT-5.5 hidden states 意义上的内部 uncertainty；
- 尚未完成所有信号在新版 full benchmark 上的统一消融。

### 2.7 Harness 如何进化

每轮 RHI 包含以下步骤：

1. 使用当前 Harness `H_t` 对 source regimes 中的行动进行判断。
2. 收集 false proceed、false stop、confident error、evidence conflict 和 high-cost failure 等失败轨迹。
3. Harness proposer 根据失败模式生成候选版本 `H_(t+1)`。
4. 对候选 Harness 进行 schema validation 和 executable-field validation。
5. 在独立 acceptance partition 上比较新旧 Harness。
6. 根据 acceptance policy 接受候选版本或回滚到旧版本。
7. Harness 确定后，才在 untouched held-out regime 上做最终测试。

RHI 可以优化的内容包括：

| Harness 组成 | 可优化内容 |
| --- | --- |
| Prompt/Instruction | 判断目标、证据使用方式和风险要求 |
| Output schema | 是否输出 utility、uncertainty、cost 和 verification value |
| Features | 新增、删除或重组可用信号 |
| Roles | Evidence auditor、uncertainty assessor、utility estimator 等角色 |
| Gates | reliability gate、uncertainty gate、risk gate |
| Routing | execute、verify、stop 和 fallback 的条件 |
| Information flow | 哪个角色可以读取哪些中间结果 |
| Acceptance policy | 新 Harness 在什么条件下被接受或回滚 |

### 2.8 Acceptance Policy

| Policy | 定义 | 研究作用 |
| --- | --- | --- |
| Never accept | 始终保留 H0 | 判断 mutation 是否真正带来收益 |
| Always accept | 接受所有 schema-valid Harness mutation | 接近原始 RHI 的无回滚更新方式 |
| Mean guarded | 平均 source-regime 指标改善才接受 | 防止明显无效 mutation |
| Robust guarded | 同时约束均值、最差 regime、风险和 loss rate | 降低局部过拟合和风险退化 |

`always_accept` 表示接受 Harness 修改，不表示系统会执行所有候选科学行动。

---

## 3. 实验

### 3.1 Benchmark 构造原则

- Benchmark 不是普通问答数据，而是 action-level scientific decision records。
- 每条记录表示：在给定科学上下文和证据下，一个候选行动是否值得执行，以及它能产生多少连续科学效用。
- 数据来自公开发表的优化、材料发现和分子发现任务。
- 原始任务记录被转换为统一的 `ActionRecord`。
- 真实 benchmark outcome、label 和 utility 作为隐藏 oracle 信息。
- 可见 context、evidence 和 features 中删除所有目标值、reward 和 hit indicator，防止标签泄漏。
- 以 `group_id` 定义 scientific regime，用于检验跨 regime 泛化。

### 3.2 ActionRecord 数据结构

| 字段 | 含义 | 决策时是否可见 |
| --- | --- | --- |
| `record_id` | 稳定的行动编号 | 是 |
| `benchmark` | 任务来源 | 是 |
| `visible_context` | 当前科学上下文 | 是 |
| `candidate_action` | 候选行动 | 是 |
| `evidence` | 清洗后的可见证据 | 是 |
| `features` | uncertainty、cost、stability 等特征 | 是 |
| `label` | 行动是否值得执行 | 否，隐藏监督信号 |
| `utility` | 连续下游科学价值 | 否，隐藏监督信号 |
| `group_id` | scientific regime | 用于数据划分 |

### 3.3 新版材料主 Benchmark

新版 benchmark 是本项目最终面向材料科学主任务的统一 action-level benchmark。它不是普通问答数据，也不是人工编写的 prompt-answer 对，而是将公开材料与化学发现任务转换为“在当前科学状态下，是否值得执行某个候选行动”的决策记录。

| Task | 公开来源 | Records | Regimes | 任务定义 |
| --- | --- | ---: | ---: | --- |
| `matbench_pairwise` | Matbench `log10(K_VRH)` / Materials Project | 8,000 | 28 | 比较两个真实材料，选择更高体积模量候选 |
| `discover_unique` | DiSCoVeR + Matbench | 10,987 | 7 | 选择高性能且化学上独特的材料 |
| `extreme_properties` | RL-CC | 2,000 | 10 | 选择达到极端目标性质的分子候选 |
| **Total** | 三个公开任务来源 | **20,987** | **45** | 材料与化学行动选择 |

#### 任务一：Matbench Pairwise Preference

- 数据来源：Matbench `log10(K_VRH)`，底层属性来自 Materials Project elasticity data。
- 科学问题：给定两个真实材料，判断哪一个更值得进入后续高体积模量研究或实验筛选。
- Action 形式：`choose(A, B)`，系统需要选择候选 A 或候选 B。
- Hidden outcome：两个材料的真实 `log10(K_VRH)` 属性差异。
- Action label：选择的候选是否优于另一候选。
- Continuous utility：依据真实属性差异构造的归一化 preference utility。
- Scientific regimes：28 个 crystal-system-pair regimes。
- 数据规模：8,000 条 action records。

#### 任务二：Unique-Material Discovery

- 数据来源：DiSCoVeR-style material discovery workflow 与 Matbench 材料属性数据。
- 科学问题：选择同时具有较高目标性能和化学独特性的材料候选。
- Action 形式：`choose_candidate(material)`，系统决定是否将候选送入下一步筛选或实验。
- Hidden outcome：候选的目标性能和独特性结果。
- Action label：候选是否进入定义的高性能/独特性目标区域。
- Continuous utility：综合性能与独特性得到的 discovery utility。
- Scientific regimes：7 个 crystal-system regimes。
- 数据规模：10,987 条 action records。

#### 任务三：Extreme-Property Discovery

- 数据来源：RL-CC extreme-property molecular generation benchmark。
- 科学问题：选择最可能达到极端目标性质的分子候选。
- Action 形式：`choose_candidate(molecule)`，系统决定是否执行后续计算、验证或实验。
- Hidden outcome：候选在多个目标属性上的真实结果及是否达到目标阈值。
- Action label：候选是否满足定义的 extreme-property target。
- Continuous utility：候选相对于目标性质的连续 discovery reward。
- Scientific regimes：10 个 target-property regimes。
- 数据规模：2,000 条 action records。

#### 统一 ActionRecord 结构

| 字段 | 含义 | 决策时是否可见 |
| --- | --- | --- |
| `record_id` | 稳定的行动编号 | 是 |
| `benchmark` | 任务来源 | 是 |
| `visible_context` | 当前科学上下文 | 是 |
| `candidate_action` | 候选实验、筛选或工具行动 | 是 |
| `evidence` | 清洗后的可见证据 | 是 |
| `features` | uncertainty、cost、stability 等特征 | 是 |
| `group_id` | scientific regime 标识 | 用于划分，不能泄漏 outcome |
| `label` | 行动是否值得执行 | 否，隐藏监督信号 |
| `utility` | 连续下游科学价值 | 否，隐藏监督信号 |

#### Benchmark 构造与泄漏控制

- 先从公开任务中取得真实材料、分子或候选比较记录。
- 再将每条原始记录转换为统一的 action-level decision record。
- 将真实属性、reward、hit indicator 和最终 outcome 放入隐藏 oracle fields。
- 从 `visible_context`、`evidence` 和可见 `features` 中删除目标值及其直接代理，避免模型通过文本或特征直接读出标签。
- 使用 `group_id` 定义 scientific regime，使评价可以检验跨 regime 泛化。
- 所有 label 和 utility 都能够由隐藏 benchmark outcome 重新计算。
- audit 结果显示：20,987 条记录的 label consistency 和 utility consistency 均通过，visible leakage 为 0。

#### 新版 benchmark 的当前状态

| 内容 | 状态 |
| --- | --- |
| 20,987 条记录转换为 action records | 已完成 |
| 三个任务来源统一 schema | 已完成 |
| 45 个 scientific regimes | 已完成 |
| Label consistency audit | 已完成 |
| Utility consistency audit | 已完成 |
| Visible leakage audit | 已完成 |
| Matbench pairwise smoke test | 已完成 |
| 全部 baseline full rerun | 尚未完成 |
| 全部 45 regimes 多 seed 主实验 | 尚未完成 |
| GPT Judge + VoI + RHI 统一实验 | 尚未完成 |
| Uncertainty source 完整消融 | 尚未完成 |

本报告后面的完整数值表来自已经完成的历史 frozen 主实验；新版 20,987-record benchmark 的详细组成如上，待全量重跑后用于最终材料科学主结果。这样可以区分“benchmark 已经构建并审计”和“所有方法已经在新版 benchmark 上完成比较”这两个不同状态。

### 3.5 Baselines

| 汇报名称 | 方法含义 | 目的 |
| --- | --- | --- |
| Verbal Confidence | 直接使用模型自报的 confidence 进行行动排序 | 检验直接信任模型置信度是否足够 |
| LLM Judge | Judge 只判断候选行动是否可靠 | 检验 reliability-only Judge |
| LLM Judge + VoI | Judge 同时估计 utility、uncertainty、cost 和 verification value | 检验 Scientific VoI 决策层 |
| LLM Judge + VoI + RHI | 在 VoI Judge 基础上根据失败轨迹递归修改 Harness | 检验 Harness-level recursive improvement |

完整实验中还保留了 evidence heuristic、static utility、static full reliability、不同 RHI acceptance policy 等诊断控制，但这些不作为汇报主方法名称，以避免把工程变体和核心方法混在一起。

### 3.6 Metrics

| Metric | 含义 | 方向 |
| --- | --- | --- |
| Net scientific utility | 固定预算下所选行动获得的 oracle-normalized 连续价值 | 越高越好 |
| Risk-adjusted utility | 对失败风险进行惩罚后的科学效用 | 越高越好 |
| Selective risk | 被放行行动中不值得执行的比例 | 越低越好 |
| Hit rate | 预算内选中正样本的比例 | 越高越好 |
| Utility efficiency | 相对于 oracle 最优选择保留的效用比例 | 越高越好 |
| Simple regret | 与 oracle 最优行动之间的价值差距 | 越低越好 |
| AURC | 不同 coverage 下的风险曲线面积 | 越低越好 |
| Worst-regime utility | 最困难 scientific regime 上的效用 | 越高越好 |
| Route coverage | execute、verify、stop 的使用比例 | 检查路由是否退化 |
| Confidently-wrong proceed rate | 高置信但错误执行的比例 | 越低越好 |

统计分析采用：

- matched held-out regimes 上的 paired difference；
- bootstrap 95% confidence interval；
- per-regime win rate；
- exact sign test；
- scientific regime 作为统计推断单位，而不是 individual record。

### 3.7 主实验结果

| 汇报方法 | Net utility ↑ | Risk-adjusted ↑ | Selective risk ↓ | Hit rate ↑ | Utility efficiency ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Verbal Confidence | 0.7577 | 0.5605 | 0.7889 | 0.3767 | 0.7577 |
| LLM Judge | 0.5990 | 0.4709 | 0.5123 | 0.2429 | 0.5990 |
| LLM Judge + VoI | 0.6372 | 0.5410 | 0.3845 | 0.2398 | 0.6372 |
| **LLM Judge + VoI + RHI** | **0.6443** | **0.6043** | **0.1600** | 0.2452 | **0.6443** |

完整诊断结果如下。它们用于分析每个组件的作用，但不作为对外汇报时的独立方法名称：

| Diagnostic control | Net utility | Risk-adjusted | Selective risk |
| --- | ---: | ---: | ---: |
| Evidence heuristic | 0.5725 | 0.4247 | 0.5914 |
| Static full reliability | 0.5635 | 0.4307 | 0.5312 |
| Utility-only component | 0.6099 | 0.5017 | 0.4325 |
| Feature mutation component | 0.6068 | 0.4735 | 0.5329 |
| Uncertainty-only component | 0.5890 | 0.4610 | 0.5123 |
| Mean-guarded RHI | 0.6384 | 0.5980 | 0.1616 |
| Robust RHI | 0.6137 | 0.5547 | 0.2361 |
| Original reliability RHI | 0.5635 | 0.4307 | 0.5314 |

### 3.8 Paired Comparisons

| Comparison | Utility difference | 95% CI | Win rate | Sign-test p |
| --- | ---: | --- | ---: | ---: |
| LLM Judge + VoI vs LLM Judge | **+0.0382** | [0.0009, 0.0734] | 0.676 | 0.0004 |
| LLM Judge + VoI + RHI vs LLM Judge + VoI | +0.0072 | [0.0028, 0.0128] | 0.419 | 0.2543 |
| LLM Judge + VoI + RHI vs reliability-only RHI control | **+0.0808** | [0.0549, 0.1059] | 0.829 | < 0.0001 |
| LLM Judge + VoI + RHI vs static reliability control | **+0.0809** | [0.0584, 0.1025] | 0.829 | < 0.0001 |

### 3.9 分任务结果

下表保留三个任务上的完整核心结果。每个任务仍然使用同一套四级方法命名。

| Task | Method | Net utility ↑ | Selective risk ↓ |
| --- | --- | ---: | ---: |
| `discover_unique` | Verbal Confidence | 0.7044 | 0.8958 |
| `discover_unique` | LLM Judge | 0.7383 | 0.7837 |
| `discover_unique` | LLM Judge + VoI | 0.7649 | 0.7837 |
| `discover_unique` | **LLM Judge + VoI + RHI** | **0.7662** | **0.0000** |
| `extreme_properties` | Verbal Confidence | 1.0000 | 0.8435 |
| `extreme_properties` | LLM Judge | 0.6678 | 0.3732 |
| `extreme_properties` | LLM Judge + VoI | 0.6835 | 0.1532 |
| `extreme_properties` | **LLM Judge + VoI + RHI** | **0.6895** | 0.1800 |
| `preferential_bo` | Verbal Confidence | 0.2451 | 0.4652 |
| `preferential_bo` | LLM Judge | 0.1831 | 0.3850 |
| `preferential_bo` | LLM Judge + VoI | 0.2976 | **0.2641** |
| `preferential_bo` | **LLM Judge + VoI + RHI** | **0.3181** | 0.3902 |

分任务结果说明：

- `discover_unique` 上，LLM Judge + VoI + RHI 同时获得最高 utility 和最低 risk。
- `extreme_properties` 上，VoI 使 risk 从 0.3732 降至 0.1532；RHI 版本保持较高 utility，但 risk 略高于静态 VoI。
- `preferential_bo` 上，RHI 版本 utility 最高；静态 VoI 的 risk 最低，说明不同任务中 utility 与 risk 的最优点并不完全一致。
- 因此，主结论应当表述为“VoI/RHI 在跨任务平均意义上改善风险调整后的行动选择”，而不是声称每个任务的每个指标都由同一版本最优。

### 3.10 结果回答了什么问题

| Research question | 实验结论 |
| --- | --- |
| Utility 是否比 reliability 更适合科学行动选择？ | 是。LLM Judge + VoI 明显优于只判断可靠性的 LLM Judge。 |
| Confidence 能否直接作为最终策略？ | 不能。Raw utility 高，但 selective risk 极高。 |
| VoI 是否有价值？ | 有。它在保持 utility 的同时支持风险控制和 verification routing。 |
| RHI 是否带来额外收益？ | 在跨 regime 平均结果上进一步提高 risk-adjusted utility，但相对静态 VoI 的 raw utility 增益较小。 |
| 当前 RHI 是否已经稳定自我提升？ | 否。Pipeline 已运行，但独立 self-evolution ablation 未显示单调提升。 |
| 当前是否已获得 GPT-5.5 内部 hidden uncertainty？ | 否。闭源模型使用 self-report 和外部 observable signals。 |
| 当前是否已完成统一 GPT-5.5 + VoI + RHI？ | 否。仅完成 500-record static hybrid subset。 |

### 3.11 RHI 自进化消融

该实验的 primary score 越低越好。

| Checkpoint | Acceptance policy | Primary score ↓ | Oracle-normalized utility ↑ |
| --- | --- | ---: | ---: |
| H0 | Initial Harness | **0.4390** | 0.5396 |
| H1 | Guarded | 0.4436 | 0.5537 |
| H2 | Guarded | 0.4487 | 0.5706 |
| H3 | Guarded | 0.4465 | 0.5792 |
| H1/H2/H3 | Always accept | 0.4455 | 0.5923 |

消融结论：

- Harness checkpoints、mutation 和 acceptance 均真实运行。
- H0 到 H3 的综合 primary score 没有稳定改善。
- Extreme-property discovery 在 mutation 后出现明显退化。
- 当前结果不支持“递归次数越多，Harness 必然越好”。
- 主实验收益不能简单归因于递归循环本身。
- 当前更准确的机制解释是：Scientific VoI contract 提供了有效 inductive bias；RHI 提供搜索和更新机制，但 proposer 与 acceptance objective 仍需改进。

### 3.12 GPT-5.5 Hybrid 子集实验

| Method | Diagnostic score ↓ | ECE ↓ | Risk@10% | Hit rate | LLM call coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5.5 direct judge | 0.3602 | 0.1548 | 0.3462 | 0.6800 | 100% |
| **GPT-5.5 + static VoI hybrid** | **0.3315** | **0.0974** | 0.3462 | 0.6800 | 100% |

该子集实验支持：

- Harness fusion 能改善 GPT-5.5 direct judge 的综合分数；
- Harness fusion 能改善 confidence calibration；
- 当前没有改善 Risk@10% 和 hit rate；
- 当前最优配置仍对每条记录调用 LLM，没有实现调用成本下降；
- 该实验是 static fusion，不是完整 LLM-driven RHI。

---

## 4. 结论与下一阶段

### 4.1 当前可以成立的结论

- 科学行动选择不能被简化为 LLM confidence ranking。
- Reliability 不是 scientific utility 的充分替代指标。
- Utility-aware 和 VoI-aware Harness 明显优于 reliability-only Harness。
- Scientific VoI recursive policy 能显著降低错误放行风险。
- 当前最佳安全策略为 `LLM Judge + VoI + RHI`：
  - net utility：0.6443；
  - risk-adjusted utility：0.6043；
  - selective risk：0.1600。
- 相比只使用 reliability 的 RHI control，`LLM Judge + VoI + RHI` 的 normalized utility 提升 0.0808，selective risk 从 0.5314 降至 0.1600。
- GPT-5.5 + static VoI 在 500-record subset 上改善了综合分数和 ECE，证明 Harness 可以补充 Direct LLM Judge。

### 4.2 当前不能过度声称的结论

- 不能声称 verbal confidence 就是可靠的模型内部 uncertainty。
- 不能声称已经获得 GPT-5.5 hidden-state-level uncertainty。
- 不能声称所有 RHI mutation 都会改善性能。
- 不能声称 RHI checkpoint 会随迭代轮数单调变好。
- 不能声称已经完成 full-scale GPT-5.5 + VoI + LLM-proposer RHI。
- 不能把 offline proxy benchmark 直接等同于真实实验室在线收益。
- 不能把 15,717-record frozen 结果与尚未全量重跑的 20,987-record 新 benchmark 混为同一组主实验。

### 4.3 下一阶段优先级

| 优先级 | 工作 | 目标 |
| ---: | --- | --- |
| P0 | 在 20,987-record、45-regime 新 benchmark 上全量重跑 | 形成真正材料科学主结果 |
| P0 | 完成统一 GPT Judge + VoI + RHI pipeline | 验证最终目标方法，而不是分散组件 |
| P0 | 使用 LLM 根据 failure traces 提出 Harness mutation | 从 deterministic proposer 升级为 LLM-driven RHI |
| P1 | 开放模型 logits 与闭源 self-report uncertainty 消融 | 明确内部/外部信号的独立贡献 |
| P1 | 改进 acceptance objective | 直接优化 risk-adjusted utility 和 worst-regime performance |
| P1 | 比较 always-accept、mean-guarded、robust-guarded | 解释为什么保守接受可能错过有益 mutation |
| P2 | 增加专家标注或在线 MatBot trajectory | 缩小 offline proxy 与真实科学行动之间的差距 |
| P2 | 测试不同模型与任务的即插即用性 | 验证 Harness provider-agnostic 设计 |

### 4.4 最终方法定位

> 本项目提出一种面向科学行动选择的 Uncertainty-Aware Scientific Judge Harness。该方法以 LLM Judge 或本地模型为语义判断基础，将 reliability、scientific utility、epistemic uncertainty、verification value、cost 与 risk 纳入统一的 Value-of-Information 决策层，并输出 execute、verify、stop 或 fallback 等可执行路由。在此基础上，系统采用 Recursive Harness Self-Improvement，根据执行轨迹和失败模式递归修改 Judge 的输入信号、输出协议、角色分工、决策 gates、信息流和接受策略，从而实现 Harness-level 的自适应优化。

## 参考方法与数据来源

- Recursive Harness Self-Improvement, arXiv:2607.15524.
- González et al., Preferential Bayesian Optimization, ICML 2017.
- Matbench `log10(K_VRH)`, derived from Materials Project elasticity data.
- DiSCoVeR, materials screening for high-performing and unique candidates.
- RL-CC, reinforcement-learning-based molecular generation for extreme target properties.
