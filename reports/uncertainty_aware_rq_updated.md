# Uncertainty-Aware Scientific Judge Harness

## 0. Executive Summary

### Research problem

普通 LLM-as-a-Judge 主要评价已经生成的答案、计划或候选输出，但科学智能体需要解决的是一个更具体的决策问题：

> 在有限实验预算和不完备证据条件下，下一步应该执行哪个科学行动，是否应该先验证，或者是否应该停止当前方向？

科学行动选择不能只依赖 reliability 或 verbal confidence，还需要同时考虑：

- 候选行动是否可靠；
- 行动成功后能产生多大 scientific utility；
- 当前判断有多大 uncertainty；
- 额外验证是否可能改变最终决策；
- 执行和验证需要多少成本；
- 错误执行会带来多大风险。

### Proposed method

我们构建了一个 uncertainty-aware scientific action-selection harness。系统以 LLM Judge 或本地预测模型为判断基础，将 reliability、utility、uncertainty、verification value、cost 和 risk 统一输入 Scientific Value-of-Information（VoI）决策层，并输出：

- `execute`：直接执行候选科学行动；
- `verify`：先检索、模拟、工具复核或专家验证；
- `stop`：停止当前行动；
- `fallback`：进入更保守的预定义路径。

在此基础上，系统 follow **Recursive Harness Self-Improvement（RHI，arXiv:2607.15524）**。RHI 根据执行轨迹、错误类型、calibration failure 和 route failure 修改 Harness 的判断协议、输入信号、输出 schema、决策 gate、路由和接受规则。

### Main conclusion from completed experiments

当前完整的 15,717-record frozen protocol 结果表明：

- 直接使用 Verbal Confidence 的 raw utility 最高，但 risk 极高；
- 只判断可靠性的 LLM Judge 不足以表达科学行动价值；
- 加入 utility、uncertainty、cost 和 verification value 后，LLM Judge + VoI 的 utility 和风险控制均改善；
- LLM Judge + VoI + RHI 取得最高的 risk-adjusted utility 和最低的 selective risk；
- 当前结果支持 Scientific VoI contract 的有效性，但尚不能声称 RHI 递归修改本身已经在所有任务上稳定提升性能。

当前主结果如下：

| Method | Net utility ↑ | Risk-adjusted utility ↑ | Selective risk ↓ | Hit rate ↑ |
| --- | ---: | ---: | ---: | ---: |
| Verbal Confidence | 0.7577 | 0.5605 | 0.7889 | 0.3767 |
| LLM Judge | 0.5990 | 0.4709 | 0.5123 | 0.2429 |
| LLM Judge + VoI | 0.6372 | 0.5410 | 0.3845 | 0.2398 |
| **LLM Judge + VoI + RHI** | **0.6443** | **0.6043** | **0.1600** | 0.2452 |

需要特别区分：上表的完整数值来自 15,717 条历史离线 proxy records；新版 20,987 条材料/化学 benchmark 已经构建和审计，但尚未完成所有方法的 full rerun。

---

## 1. Four Main Methods

为避免把工程变体和核心方法混在一起，主报告只使用四个方法名称。

| Method | 方法含义 | 主要作用 |
| --- | --- | --- |
| **Verbal Confidence** | 直接使用模型自报的 confidence 对候选行动排序 | 测试模型 confidence 能否直接作为行动策略 |
| **LLM Judge** | Judge 只判断候选行动是否可靠 | 提供 reliability-only baseline |
| **LLM Judge + VoI** | 在 Judge 输出基础上加入 utility、uncertainty、cost 和 verification value | 将答案评价转化为科学行动决策 |
| **LLM Judge + VoI + RHI** | 在 VoI Judge 基础上，根据失败轨迹递归修改 Harness | 测试 Harness-level recursive improvement |

底层实现还包含 reliability-only gate、static utility、static VoI、不同 uncertainty component 和 acceptance policy 等诊断控制。这些控制用于拆解组件贡献，但不作为主报告方法名。

技术映射关系如下：

| 汇报方法 | Frozen protocol 中的代表性实现 |
| --- | --- |
| Verbal Confidence | `verbal_confidence` |
| LLM Judge | `h0_reliability` |
| LLM Judge + VoI | `static_voi` |
| LLM Judge + VoI + RHI | `scivoi_policy_always_accept`，并辅以 guarded RHI 对照 |

这里的 “LLM Judge” 在 15,717-record frozen protocol 中指可离线执行的 structured Judge proxy；真正使用 GPT-5.5 API 的 direct judge 与 hybrid 实验是独立的 500-record subset，后文单独报告。

---

## 2. Research Questions

### RQ1：Scientific VoI 是否优于 reliability-only Judge？

在固定行动预算下，加入 scientific utility、uncertainty、cost 和 verification-aware routing，是否能比只判断 reliability 的 LLM Judge 选择更有价值且更安全的科学行动？

### RQ2：VoI 是否能够改善 Direct LLM Judge？

在匹配的 GPT-5.5 subset 上，将 Direct Judge 的 action score 与静态 VoI Harness 融合，是否能够改善 action-worthiness calibration 和诊断分数？

### RQ3：RHI 是否带来超出静态 VoI 的额外收益？

在 Scientific VoI contract 已固定的基础上，根据失败轨迹递归修改 Harness，是否能够在 unseen scientific regimes 上进一步提高 risk-adjusted utility，同时不引入显著的 risk regression？

### RQ4：uncertainty signal 是否具有跨模型可迁移性？

同一个 Harness 是否可以接入不同来源的 uncertainty signal，包括：

- 开放模型 logits；
- 闭源模型 self-reported confidence；
- 多次采样 disagreement；
- 多模型 disagreement；
- evidence conflict；
- tool agreement；
- utility ensemble disagreement。

RQ4 是当前下一阶段的扩展问题，当前已实现 provider interface 和部分 pilot，但尚未在新版 full benchmark 上完成完整信号消融。

---

## 3. Method

### 3.1 System interface

系统输入是一条候选科学行动记录：

| Input | 含义 |
| --- | --- |
| Scientific context | 当前科学目标、状态和已有结果 |
| Candidate action | 实验、模拟、检索、筛选、工具调用或验证行动 |
| Evidence | 当前可见的支持证据、反对证据和证据冲突 |
| Model signals | confidence、logit margin、sampling disagreement 等 |
| External signals | tool agreement、source reliability、OOD、perturbation stability 等 |
| Operational signals | cost、reversibility、action complexity |
| Current Harness | 当前版本的判断协议和路由规则 |

隐藏的真实 label、utility、最终材料性质、reward 和 hit indicator 不会作为决策时输入。

系统输出包括：

| Output | 含义 |
| --- | --- |
| Reliability | 行动正确、成功或值得执行的概率估计 |
| Scientific utility | 行动成功后预期产生的连续科学价值 |
| Uncertainty | 当前模型判断的不确定性 |
| Verification value | 额外验证可能带来的决策改进 |
| Cost | 执行或验证的预期成本 |
| Risk | 错误执行的预期损失 |
| Route | `execute`、`verify`、`stop` 或 `fallback` |

### 3.2 LLM Judge

LLM Judge 的作用是从当前科学上下文和候选行动中抽取结构化判断。它不是最终的行动策略，而是提供决策信号。

对于候选行动 $a$，Judge 可以输出：

- reliability；
- expected scientific utility；
- verbal confidence；
- uncertainty-related evidence；
- estimated cost；
- verification value；
- failure risk；
- predicted failure modes。

在最简单的 LLM Judge 方法中，系统只根据 reliability 判断行动是否值得执行。该方法没有显式建模“这个行动是否值得占用预算”以及“是否应该先验证”。

### 3.3 Value-of-Information decision layer

Scientific VoI 层将 Judge 输出转化为行动路由。一般形式为：

\[
V(a)=\mathbb{E}[U(a)\mid s,E]-\lambda_c C(a)-\lambda_r R(a),
\]

其中：

- $U(a)$ 是候选行动的 scientific utility；
- $C(a)$ 是执行或验证成本；
- $R(a)$ 是错误执行或失败带来的风险；
- $s$ 是当前 scientific state；
- $E$ 是当前 evidence。

当前实现是基于 utility、uncertainty、verification value、cost 和 risk signals 的 action-level VoI approximation，并不是完整环境模型下的严格 Bayesian posterior VoI 计算。

决策逻辑可以概括为：

| 条件 | Route |
| --- | --- |
| utility 高、risk 可接受、额外验证价值有限 | `execute` |
| utility 潜力高，但 uncertainty 或 evidence conflict 高，验证成本可接受 | `verify` |
| utility 低，或 cost/risk 不可接受 | `stop` |
| 当前证据不足以稳定判断 | `fallback` |

VoI 的核心作用是把 uncertainty 转化为“是否值得先验证”的决策信号，而不是单纯把 uncertainty 当成惩罚项。

### 3.4 Uncertainty signals

| Signal source | 可获得的信息 | 当前定位 |
| --- | --- | --- |
| Open-model logits | token probability、entropy、margin、candidate-label distribution | 较接近模型内部可观测信号 |
| Closed-model self-report | 模型显式输出的 confidence 和 rationale | noisy observable self-report |
| Sampling disagreement | 多次采样结果之间的差异 | 输出稳定性 proxy |
| Model disagreement | 不同模型或不同 Judge 的预测差异 | epistemic disagreement proxy |
| External evidence | evidence conflict、tool agreement、source reliability | 外部 observable signal |
| Utility ensemble | 多个 utility models 的预测差异 | local epistemic proxy |

当前不能把闭源模型 self-reported confidence 称为 hidden-state intrinsic uncertainty。当前更准确的表述是：

> 对开放模型，可以提取 logits-based uncertainty；对闭源模型，可以通过 self-report 和外部可观测信号构造 uncertainty proxy；两者通过统一 provider interface 接入 Harness。

### 3.5 Recursive Harness Improvement

RHI follow Recursive Harness Self-Improvement（arXiv:2607.15524）的基本范式：

\[
H_t \rightarrow \text{execution traces} \rightarrow \text{failure feedback}
\rightarrow \text{candidate } H_{t+1}
\rightarrow \text{validation and acceptance}.
\]

RHI 修改的不是基础模型权重，而是 Harness 的执行协议，包括：

| Harness component | 可修改内容 |
| --- | --- |
| Prompt and instruction | 判断目标、证据使用方式和安全要求 |
| Output schema | 是否输出 utility、uncertainty、cost 和 verification value |
| Signal provider | 新增、删除或重组 uncertainty/evidence signals |
| Roles | evidence auditor、uncertainty assessor、utility estimator 等 |
| Decision gates | reliability gate、uncertainty gate、risk gate |
| Routing | execute、verify、stop 和 fallback 条件 |
| Information flow | 不同角色之间传递哪些中间结果 |
| Acceptance policy | 何时接受、拒绝或回滚新 Harness |

RHI 的作用可以概括为：

- VoI 决定当前应该执行什么行动；
- RHI 决定下一轮应该如何改进做决策的 Harness。

当前完整 frozen experiment 中，mutation proposer 主要是 deterministic、trajectory-conditioned proposer。LLM proposer interface 已实现，但统一的 full-scale LLM-proposer RHI 结果尚未完成。因此，当前报告不把已有主结果称为“GPT-5.5 + VoI + LLM-proposer RHI”。

---

## 4. Benchmark

### 4.1 Benchmark definition

新版 benchmark 是基于公开材料科学和化学发现任务构造的 offline action-level proxy benchmark。它使用真实材料和分子任务结果，但不是：

- 在线 MatBot trajectory logging；
- 专家直接标注的“下一步是否值得做”数据；
- 新测量的实验数据；
- 普通问答数据集。

每条记录表示：在给定 scientific context 和 visible evidence 下，一个候选行动是否值得执行，以及该行动能够产生多大连续 scientific utility。

### 4.2 Benchmark composition

| Task | Public source | Records | Regimes | Scientific question |
| --- | --- | ---: | ---: | --- |
| `matbench_pairwise` | Matbench `log10(K_VRH)` / Materials Project | 8,000 | 28 | 哪一个真实材料更值得后续研究？ |
| `discover_unique` | DiSCoVeR + Matbench | 10,987 | 7 | 哪个候选同时具有高性能和高独特性？ |
| `extreme_properties` | RL-CC | 2,000 | 10 | 哪个分子最值得进行极端性质验证？ |
| **Total** | 三个公开任务来源 | **20,987** | **45** | 材料与化学科学行动选择 |

### 4.3 Task details

#### `matbench_pairwise`

- 使用 Matbench `log10(K_VRH)`，底层属性来自 Materials Project elasticity data。
- 每条记录是一个真实材料 A/B preference action。
- 候选行动是选择 A 或 B 进入后续高体积模量研究。
- Hidden outcome 是两个材料的真实 `log10(K_VRH)` 及其差异。
- Label 表示选择的材料是否具有更高真实属性。
- Utility 根据两个候选真实属性差异构造为连续 preference utility。
- 共 8,000 条 records，覆盖 28 个 crystal-system-pair regimes。

#### `discover_unique`

- 使用 DiSCoVeR-style material discovery workflow 和 Matbench material property data。
- 每条记录是一个候选材料是否值得进入下一轮筛选的 action。
- Hidden outcome 包含候选性能和材料独特性结果。
- Label 表示候选是否达到预设的高性能/高独特性区域。
- Utility 综合性能和独特性，形成 continuous discovery utility。
- 共 10,987 条 records，覆盖 7 个 crystal-system regimes。

#### `extreme_properties`

- 使用 RL-CC extreme-property molecular generation benchmark。
- 每条记录是一个分子候选是否值得进入后续计算或实验验证的 action。
- Hidden outcome 包含多个目标属性及其是否达到目标阈值。
- Label 表示候选是否达到定义的 extreme-property target。
- Utility 根据候选距离目标性质的程度构造连续 discovery reward。
- 共 2,000 条 records，覆盖 10 个 target-property regimes。

### 4.4 Unified ActionRecord

| Field | Meaning | Visible at decision time? |
| --- | --- | --- |
| `record_id` | 稳定行动编号 | Yes |
| `benchmark` | 任务来源 | Yes |
| `visible_context` | 当前科学上下文 | Yes |
| `candidate_action` | 候选行动 | Yes |
| `action_type` | 行动类型 | Yes |
| `evidence` | 清洗后的可见证据 | Yes |
| `features` | uncertainty、cost、stability 等特征 | Yes |
| `group_id` | scientific regime | 用于划分 |
| `label` | 行动是否值得执行 | No，hidden supervision |
| `utility` | 连续科学价值 | No，hidden supervision |
| `metadata` | 来源和审计字段 | 部分可见 |

### 4.5 Leakage and consistency audit

新版 benchmark 已完成：

- 20,987 条原始记录全部转换为 action records；
- label consistency audit 通过；
- utility consistency audit 通过；
- visible leakage audit 通过；
- hidden oracle fields 明确记录并排除；
- Matbench pairwise 单任务 smoke test 完成。

审计结果：

| Audit item | Result |
| --- | --- |
| Raw records | 20,987 |
| Converted action records | 20,987 |
| Label consistency | Passed |
| Utility consistency | Passed |
| Visible leakage | 0 |
| Scientific regimes | 45 |

### 4.6 Benchmark status

需要区分 benchmark 构建状态和实验重跑状态：

| Item | Status |
| --- | --- |
| 新版 benchmark 构建 | 已完成 |
| 数据 schema 统一 | 已完成 |
| Label/utility/leakage audit | 已完成 |
| Matbench pairwise smoke test | 已完成 |
| 所有四类方法 full rerun | 尚未完成 |
| 全部 45 regimes 多 seed 主实验 | 尚未完成 |
| GPT Judge + VoI + RHI 统一实验 | 尚未完成 |
| uncertainty source 完整消融 | 尚未完成 |

---

## 5. Evaluation Protocol

当前已经完成完整数值评价的是历史 frozen protocol，而不是新版 20,987-record benchmark。

| Protocol item | Setting |
| --- | --- |
| Records | 15,717 sanitized historical proxy records |
| Scientific regimes | 21 |
| Outer evaluation | Leave-one-regime-out |
| Random seeds | 5 |
| Matched folds | 105 |
| Action budget | Top 10% actions |
| Inference unit | Scientific regime |

held-out regime 不参与：

- model fitting；
- confidence/uncertainty calibration；
- Harness mutation；
- candidate acceptance；
- stopping；
- 最终 Harness 选择。

### Primary metric

主指标是固定 top-10% action budget 下的 oracle-normalized continuous net scientific utility，并在 outer scientific regimes 上取 macro average。

### Guardrail metrics

| Metric | Meaning | Direction |
| --- | --- | --- |
| Net scientific utility | 所选行动保留的连续科学价值 | Higher is better |
| Risk-adjusted utility | 对失败风险进行惩罚后的效用 | Higher is better |
| Selective risk | 被放行行动中不值得执行的比例 | Lower is better |
| Hit rate | 预算内选中正样本的比例 | Higher is better |
| Utility efficiency | 相对于 oracle 最优选择保留的效用比例 | Higher is better |
| Simple regret | 与 oracle 最优行动之间的差距 | Lower is better |
| AURC | coverage-risk 曲线面积 | Lower is better |
| Worst-regime utility | 最差 regime 的效用 | Higher is better |
| Confidently-wrong proceed rate | 高置信但错误执行的比例 | Lower is better |

统计推断：

- 使用同一 held-out regime 上的 paired differences；
- 报告 bootstrap 95% confidence intervals；
- 报告 per-regime win rate；
- 报告 exact sign test；
- 统计推断单位是 scientific regime，而不是 individual record。

---

## 6. Completed Results

### 6.1 Aggregate results

| Method | Net utility ↑ | Risk-adjusted utility ↑ | Selective risk ↓ | Hit rate ↑ | Utility efficiency ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Verbal Confidence | 0.7577 | 0.5605 | 0.7889 | 0.3767 | 0.7577 |
| LLM Judge | 0.5990 | 0.4709 | 0.5123 | 0.2429 | 0.5990 |
| LLM Judge + VoI | 0.6372 | 0.5410 | 0.3845 | 0.2398 | 0.6372 |
| **LLM Judge + VoI + RHI** | **0.6443** | **0.6043** | **0.1600** | 0.2452 | **0.6443** |

### 6.2 Full diagnostic controls

以下控制用于拆解组件贡献，不作为主方法名称：

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

### 6.3 Paired comparisons

| Comparison | Utility difference | 95% CI | Win rate | Sign-test p |
| --- | ---: | --- | ---: | ---: |
| LLM Judge + VoI vs LLM Judge | **+0.0382** | [0.0009, 0.0734] | 0.676 | 0.0004 |
| LLM Judge + VoI + RHI vs LLM Judge + VoI | +0.0072 | [0.0028, 0.0128] | 0.419 | 0.2543 |
| LLM Judge + VoI + RHI vs reliability-only RHI control | **+0.0808** | [0.0549, 0.1059] | 0.829 | < 0.0001 |
| LLM Judge + VoI + RHI vs static reliability control | **+0.0809** | [0.0584, 0.1025] | 0.829 | < 0.0001 |

### 6.4 Task-level results

这些分任务结果来自已完成的 15,717-record historical frozen protocol。

#### `discover_unique`

| Method | Net utility ↑ | Selective risk ↓ |
| --- | ---: | ---: |
| Verbal Confidence | 0.7044 | 0.8958 |
| LLM Judge | 0.7383 | 0.7837 |
| LLM Judge + VoI | 0.7649 | 0.7837 |
| **LLM Judge + VoI + RHI** | **0.7662** | **0.0000** |

结论：RHI 版本同时取得最高 utility 和最低 risk，是最支持完整方法故事的 task。

#### `extreme_properties`

| Method | Net utility ↑ | Selective risk ↓ |
| --- | ---: | ---: |
| Verbal Confidence | **1.0000** | 0.8435 |
| LLM Judge | 0.6678 | 0.3732 |
| LLM Judge + VoI | 0.6835 | **0.1532** |
| **LLM Judge + VoI + RHI** | 0.6895 | 0.1800 |

结论：Verbal Confidence raw utility 最高但风险极高；VoI 的主要贡献是风险控制；RHI 恢复部分 utility，但 risk 略高于静态 VoI。

#### `preferential_bo`

| Method | Net utility ↑ | Selective risk ↓ |
| --- | ---: | ---: |
| Verbal Confidence | 0.2451 | 0.4652 |
| LLM Judge | 0.1831 | 0.3850 |
| LLM Judge + VoI | 0.2976 | **0.2641** |
| **LLM Judge + VoI + RHI** | **0.3181** | 0.3902 |

结论：RHI 版本 utility 最高，但静态 VoI risk 最低，说明 RHI 会改变 utility-risk trade-off，acceptance policy 对最终风险非常重要。

### 6.5 RHI self-evolution ablation

该实验 primary score 越低越好。

| Checkpoint | Policy | Primary score ↓ | Oracle-normalized utility ↑ |
| --- | --- | ---: | ---: |
| H0 | Initial Harness | **0.4390** | 0.5396 |
| H1 | Guarded | 0.4436 | 0.5537 |
| H2 | Guarded | 0.4487 | 0.5706 |
| H3 | Guarded | 0.4465 | 0.5792 |
| H1/H2/H3 | Always accept | 0.4455 | 0.5923 |

该消融表明：

- mutation、checkpoint evaluation 和 acceptance 均真实运行；
- H0 到 H3 的综合 primary score 没有稳定改善；
- extreme-property discovery 在 mutation 后出现退化；
- 不能声称当前 RHI 会随迭代次数单调提升；
- 主实验中的正向结果应主要归因于 Scientific VoI contract 和 risk-aware routing，而不能简单归因于 RHI recursion 本身。

### 6.6 GPT-5.5 direct judge versus static VoI subset

这是独立的 500-record GPT-5.5 subset，不是 15,717-record frozen main result，也不是完整的 LLM-proposer RHI。

| Method | Diagnostic score ↓ | ECE ↓ | Risk@10% | Hit rate | LLM call coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5.5 direct judge | 0.3602 | 0.1548 | 0.3462 | 0.6800 | 100% |
| **GPT-5.5 + static VoI hybrid** | **0.3315** | **0.0974** | 0.3462 | 0.6800 | 100% |

该结果支持：

- static VoI fusion 可以改善 GPT-5.5 direct judge 的诊断分数；
- static VoI fusion 可以改善 calibration；
- 当前没有改善 Risk@10% 和 hit rate；
- 最优配置仍然对每条记录调用 LLM；
- 该结果不能被表述为 full-scale GPT-5.5 + VoI + RHI。

---

## 7. Interpretation

### 7.1 Verbal Confidence

Verbal Confidence 在 aggregate net utility 上最高，为 0.7577，但 selective risk 为 0.7889。它说明模型 confidence 可能有一定排序信息，但不能直接视为经过校准的行动成功概率，更不能直接作为安全科学行动策略。

### 7.2 LLM Judge

LLM Judge 的 aggregate net utility 为 0.5990，risk-adjusted utility 为 0.4709。只判断 reliability 会忽略一个关键问题：即使一个行动可靠，也不代表它值得占用当前实验或计算预算。

### 7.3 LLM Judge + VoI

加入 VoI 后，net utility 提升到 0.6372，selective risk 降到 0.3845。它使系统能够区分：

- 应该立即执行的行动；
- 潜在价值高但需要先验证的行动；
- 不值得继续投入的行动。

### 7.4 LLM Judge + VoI + RHI

LLM Judge + VoI + RHI 的 risk-adjusted utility 为 0.6043，selective risk 为 0.1600，均为当前完整 frozen experiment 中最佳。与 reliability-oriented RHI control 相比，utility difference 为 +0.0808，95% CI 为 [0.0549, 0.1059]，在 82.9% 的 paired folds 上获胜。

不过，与静态 VoI 相比，utility difference 为 +0.0072，sign-test p 为 0.2543。因此当前最稳健的表述是：

> Scientific VoI structure provides the main improvement over reliability-only action selection. RHI further changes the utility-risk trade-off and obtains the best risk-adjusted aggregate result, but its independent raw-utility gain over static VoI is not yet statistically convincing.

---

## 8. Claim Boundary

### Claims supported by current evidence

- 科学行动选择不能被简化为答案质量评价或 confidence ranking。
- Reliability-only Judge 不是 scientific utility 的充分替代指标。
- LLM Judge + VoI 在固定预算下优于 reliability-only Judge。
- LLM Judge + VoI + RHI 在当前 frozen protocol 上获得最高 risk-adjusted utility 和最低 selective risk。
- Verbal Confidence raw utility 高，但由于 risk 极高，不适合作为安全执行策略。
- RHI mutation、evaluation、acceptance 和 rollback pipeline 已经实现并运行。
- GPT-5.5 + static VoI 在 500-record subset 上改善 diagnostic score 和 ECE。

### Claims not yet supported

- 尚不能声称已经获得 GPT-5.5 hidden-state-level intrinsic uncertainty。
- 尚不能声称 self-reported confidence 等价于 intrinsic uncertainty 或 logits uncertainty。
- 尚不能声称 full-scale GPT-5.5 + VoI + LLM-proposer RHI 已完成。
- 尚不能声称 LLM proposer 已经优于 deterministic proposer。
- 尚不能声称 RHI checkpoints 会随着迭代单调提升。
- 尚不能把新版 20,987-record benchmark 的构建审计写成全量方法实验已经完成。
- 尚不能把 offline action-level proxy benchmark 直接等同于在线实验室科学收益。

---

## 9. Next Experiments

| Priority | Experiment | Purpose |
| ---: | --- | --- |
| P0 | 在 20,987-record、45-regime 新 benchmark 上完成四类方法 full rerun | 形成真正材料科学主结果 |
| P0 | 完成统一 LLM Judge + VoI + RHI pipeline | 验证目标方法而不是分散组件 |
| P0 | 使用 LLM 根据 failure traces 提出 Harness mutation | 验证 LLM-driven RHI |
| P1 | 比较 logits、self-report、disagreement 和 external signals | 分析 uncertainty source 的独立贡献 |
| P1 | 改进 acceptance objective | 直接优化 risk-adjusted utility 和 worst-regime performance |
| P1 | 完成 guarded、always-accept 和 robust acceptance 对照 | 解释 mutation 接受策略的作用 |
| P2 | 增加专家标注或在线 trajectory | 缩小 offline proxy 与真实科学行动的差距 |
| P2 | 测试不同模型和任务的 provider portability | 验证即插即用性 |

## Final positioning

本项目提出一种面向科学行动选择的 Uncertainty-Aware Scientific Judge Harness。该方法以 LLM Judge 或本地预测模型为语义判断基础，将 reliability、scientific utility、epistemic uncertainty、verification value、cost 和 risk 纳入统一的 Value-of-Information 决策层，并输出 execute、verify、stop 或 fallback 等可执行路由。在此基础上，系统采用 Recursive Harness Self-Improvement，根据执行轨迹和失败模式递归修改 Judge 的输入信号、输出协议、角色分工、决策 gates、信息流和 acceptance policy。

当前最强证据支持的核心故事是：**从 reliability-only judging 走向 uncertainty-aware, utility-aware scientific action selection，可以显著改善风险调整后的科学行动价值；RHI 为 Harness 自适应提供了机制，但其独立的普遍自进化收益仍需要在新版材料 benchmark 和真正 LLM proposer 实验中进一步验证。**
