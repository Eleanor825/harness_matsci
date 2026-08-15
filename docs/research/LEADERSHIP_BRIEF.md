# Sci-VoI-RHI 领导汇报材料

## 一句话总结

我们给材料科学 Agent 加了一层“行动价值判断”harness：不改底座模型，而是在每一步行动前判断这一步是应该直接执行、先补证据/验证，还是停止，从而让科学 Agent 知道“这一步值不值得做”。

## 我们做了什么

- 做了一个面向 MatSci Agent 的 action-level uncertainty / value-of-information harness。
- 把三类科学发现问题整理成可以量化验证的 action-worthiness benchmark：
  - Matbench material pairwise preference：用真实 Materials Project / Matbench `log10(K_VRH)` 属性表判断候选 A 是否优于候选 B；
  - unique-material discovery：判断候选材料是否值得筛选、是否有独特科学价值；
  - extreme-property discovery：判断候选分子/材料是否值得推进到极端性质目标。
- 新增了 8,000 条真实材料 A/B pairwise action records；当前主材料数据共 20,987 条，覆盖 45 个 scientific regimes，并通过 label/utility/leakage audit。
- 保留了旧版 15,717 条、105 folds 的 Sci-VoI-RHI 完整结果作为 legacy 对照；下一步需要在新的 20,987 条主材料数据上完整 rerun。
- 实现了类似 RHI 的 recursive harness improvement，但优化目标从“通用偏好/输出质量”改成“科学行动是否值得做”。
- 补了真实 direct LLM-as-judge baseline：`gpt-5.5` 在 500 条 balanced subset 上完成评分。

## 为什么要这样做

材料科学 Agent 的关键风险不是最后一句话答错，而是在过程中做错下一步：比如推进一个证据很弱的候选、浪费昂贵实验预算、忽略 OOD 风险、或者在工具分歧时过度自信。

传统 confidence 只回答“模型觉得自己对不对”。我们的核心问题更接近科学流程：

> 这一步行动在当前证据、成本、风险和潜在科学收益下，值不值得现在做？

所以我们不只估计 correctness，而是估计 action 的科学价值、风险和是否需要先验证。

## 方法怎么做

每条 action record 包含当前可见上下文、候选 action、证据和不确定性特征。Harness 会计算：

```text
visible context + evidence + candidate action
        -> reliability head: p_success
        -> utility head: expected scientific utility
        -> uncertainty head: epistemic/OOD/source risk
        -> cost and reversibility adjustment
        -> route: execute / verify / stop
```

递归优化流程是：

```text
H_i harness
  -> replay historical/scientific action trajectories
  -> summarize failure modes
  -> mutate harness contract/features/roles/gates/hops
  -> compare H_i vs H_{i+1} on source-regime acceptance data
  -> accept or rollback
  -> evaluate only on held-out scientific regimes
```

和原始 RHI 的区别是：我们优化的不是普通 prompt，也不是纯 reliability gate，而是一个可执行的 scientific action-value contract。Harness 的变化会真实改变 feature set、utility score、uncertainty penalty、threshold 和 route decision。

## 优化目标是什么

我们的目标不是单纯分类准确率，而是科学行动价值最大化：

- Reliability：学习 calibrated `p_success`，让放行 action 的风险可控。
- Utility：学习连续 scientific utility，区分“正确但低价值”和“高价值但需要验证”的行动。
- VoI routing：根据 expected value of information 决定 `execute / verify / stop`。
- Fixed-budget discovery：在 top-10% action budget 下最大化 oracle-normalized net scientific utility。
- Risk control：同时降低 selective risk，避免为了高 coverage 放行大量错误/低价值 action。
- Recursive acceptance：只有当新 harness 在 source-regime acceptance set 上带来 utility/risk 改善时才接受，held-out test regime 不参与调参。

这个目标更符合科学发现：实验预算有限、行动成本不对称、直接 abstain 不一定最优，有些行动应该先补证据或模拟验证。

## 当前实验结果

### 全量离线科学 benchmark

| Method | Net utility ↑ | Risk-adjusted ↑ | Selective risk ↓ | Folds |
| --- | ---: | ---: | ---: | ---: |
| Sci-VoI-RHI, direct RHI-style | 0.6443 ± 0.2052 | 0.6043 | 0.1600 | 105 |
| Sci-VoI-RHI, robust-guarded | 0.6137 ± 0.2201 | 0.5547 | 0.2361 | 105 |
| Original reliability-only RHI | 0.5635 ± 0.1916 | 0.4307 | 0.5314 | 105 |
| Static full reliability gate | 0.5635 ± 0.2114 | 0.4307 | 0.5312 | 105 |
| Static VoI head | 0.6372 ± 0.2105 | 0.5410 | 0.3845 | 105 |
| Verbal confidence heuristic | 0.7577 ± 0.2888 | 0.5605 | 0.7889 | 105 |

关键结论：

- 相比原始 reliability-only RHI，Sci-VoI-RHI 的 utility 提升 +0.0808，selective risk 降低 0.3713。
- 相比 frozen harness，recursive update 带来 +0.0454 utility，并降低 0.3522 risk。
- Static utility / static VoI 的 raw utility 接近，但风险更高；Sci-VoI-RHI 的 risk-adjusted profile 更好。
- Verbal confidence raw utility 高，但 selective risk 达到 0.7889，不适合作为安全科学行动策略。

### Direct LLM-as-judge baseline

| Method | Protocol | Score ↓ | Risk@10% ↓ | Hit rate ↑ | ECE ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| `gpt-5.5` direct judge | 500-record subset | 0.3602 | 0.3462 | 0.6800 | 0.1548 |
| verbal confidence | same subset | 0.4757 | 0.4231 | 0.6000 | 0.3017 |
| evidence heuristic | same subset | 0.8131 | 0.8846 | 0.1200 | 0.5879 |

解释：

- `gpt-5.5` direct judge 是很强的 one-shot baseline，明显强于简单 heuristic。
- 在这个小的 direct-judge subset 上，`gpt-5.5` 当前比我们的 quick same-subset RHI check 更会做 top-k ranking。
- 但 direct judge 只是 500 条 subset；旧完整结果是在 15,717 条 legacy 数据上完成，新主材料数据已经补齐到 20,987 条，完整 rerun 还需要继续跑。
- `gpt-5.6-luna` 已尝试 100 条 subset，但 provider 返回 HTTP 429，所以目前不能算有效结果。

## 当前可以怎么讲贡献

- 从 answer-level uncertainty 转到 action-level value-of-information。
- 让 scientific agent 不只是“知道自己不确定”，而是知道下一步应该执行、验证还是停止。
- 把 RHI 从通用 harness preference 改成材料科学行动价值优化。
- 设计了可执行 harness：roles/contracts/hops 的修改会落到实际 scoring 和 routing 上。
- 用三类科学发现任务做了可量化评估，并保留 direct LLM judge baseline 作为强对照。

## 当前边界和风险

- 当前 labels 和 utility 来自历史论文/benchmark proxy，不是在线 MatBot 真实轨迹。
- `gpt-5.5` direct judge 在小子集上很强，说明论文里不能简单写“全面超过强 LLM judge”。
- 我们方法在一些 subset 检查里偏保守，后续需要继续调 coverage-risk trade-off。
- 最终投稿前需要补在线或 semi-online MatBot trajectory validation。

## 下一步计划

- 先跑 `gpt-5.4` 和 `gpt-5.4-mini` direct judge baseline，等 `gpt-5.6-luna` 限流解除再补强模型。
- 调整 Sci-VoI threshold / acceptance policy，提高 coverage，同时保留低风险和好校准。
- 接入 MatBot runtime logging，把在线 action 转成同样的 `ActionRecord`。
- 做在线或半在线验证：在真实 discovery workflow 中比较 execute / verify / stop 对最终科学收益的影响。
