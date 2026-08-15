# Data Provenance and Action Construction

This document is the reproducibility map for the current offline MatSci uncertainty-harness experiments. It answers exactly where the data came from, what was visible to the agent, what was hidden as oracle outcome, and how `label` and `utility` were computed.

## Core Boundary

- The main benchmark is **not** live MatBot trajectory logging and is **not** 15,717 manually extracted PDF actions.
- The main benchmark contains **paper-artifact / benchmark-derived action proxies** from three public scientific task families.
- Each row is converted into the same action-level `ActionRecord` interface that MatBot should emit online: context, candidate action, evidence, uncertainty features, hidden outcome label, and hidden utility.
- The 500 electrolyte-paper dataset is a separate pilot for paper-derived weak labels; it is not part of the 15,717-record main benchmark.

## Dataset Inventory

| Task family | Scientific question | Public source | Raw file expected by this repo | Raw rows | Converted action records | Converted positives | Groups / regimes | Label source |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `preferential_bo` | Is candidate A better than candidate B? | González et al., Preferential Bayesian Optimization, ICML 2017, `https://proceedings.mlr.press/v70/gonzalez17a.html` | `pairwise_optimization_actions.jsonl` | 2,730 | 2,730 pairwise duels | 1,385 | 4 objectives | Noisy visible duel choice checked against hidden latent objective utility |
| `discover_unique` | Which material is both high-performing and chemically unique? | DiSCoVeR, DOI `10.1039/d1dd00028d`; code `https://github.com/sparks-baird/mat_discover`; open Matbench artifact `https://raw.githubusercontent.com/materialsproject/matbench/main/scripts/artifacts/matbench_log_kvrh.json.bz2` | `unique_materials_actions.jsonl` | 10,987 | 10,987 screening actions | 1,099 | 7 crystal systems | Top-decile combined performance + uniqueness proxy |
| `extreme_properties` | Which generated molecule hits extreme target properties? | RL-CC extreme-property discovery, DOI `10.1039/d3sc05281h`; code `https://github.com/Haeyeon-Choi/RL-CC` | `extreme_properties_actions.jsonl` | 2,000 | 2,000 molecule-advance actions | 313 | 10 targets | All seven generated properties within Table 1 RMSE bounds |

The local raw-data directory used for the current runs is `/Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks`. It contains:

- `pairwise_optimization_actions.jsonl`: 2,730 raw candidate rows.
- `unique_materials_actions.jsonl`: 10,987 raw material rows.
- `extreme_properties_actions.jsonl`: 2,000 raw molecule rows.
- `all_three_tasks_actions.jsonl`: concatenated raw rows for inspection only.
- `summary.json`: source manifest and raw-record summary.

Important count distinction: `summary.json` reports raw candidate positives for the PBO candidate-builder stage (`273` top-decile candidates). The actual harness experiment converts those candidates into pairwise duels in `src/harness_matsci/historical.py`; after conversion, PBO has `1,385` positive duel actions and `1,345` negative duel actions.

## `ActionRecord` Contract

Every experiment consumes `ActionRecord` rows from `src/harness_matsci/schema.py`.

| Field | Meaning | Visible at decision time? |
| --- | --- | --- |
| `record_id` | Stable action identifier | Yes |
| `benchmark` | Task family, such as `discover_unique` | Yes |
| `split` | Train / feedback / acceptance / test assignment | No for runtime policy; yes for evaluation bookkeeping |
| `visible_context` | What the scientific agent or harness is allowed to read | Yes |
| `candidate_action` | The proposed next step | Yes |
| `action_type` | Action category, such as `choose_candidate` or `choose` | Yes |
| `evidence` | Pre-execution evidence snippets, after leakage sanitization | Yes |
| `features` | Numeric uncertainty / cost / source / stability signals | Yes |
| `label` | `1` if action was worth executing, else `0` | Hidden oracle label |
| `utility` | Normalized downstream scientific gain | Hidden oracle utility |
| `metadata` | Source, group, raw IDs, and audit fields | Mixed; oracle subfields are not model inputs |

The harness is trained to estimate whether the **current action** is worth doing under the visible information, not to solve the whole scientific discovery task end-to-end.

## Task 1: Preferential BO Pairwise Optimization

### Public Artifact

- Paper: González et al., “Preferential Bayesian Optimization,” ICML 2017.
- Paper URL: `https://proceedings.mlr.press/v70/gonzalez17a.html`.
- Reproduced benchmark functions: Forrester, Six-Hump Camel, Goldstein-Price, and Levy.
- Raw rows are generated from the published latent objective functions, not downloaded from a separate dataset.

### Raw Candidate Construction

- For each objective, a deterministic grid is evaluated.
- `grid_size=30` gives 30 Forrester points and `30 × 30 = 900` points for each 2D objective.
- Objective values are minimization objectives: lower objective value is scientifically better.
- Raw normalized utility is `1 - normalize(objective_value)`, so larger utility is better.
- Raw candidate success in the upstream summary is top-decile latent utility; this raw label is only a builder-stage diagnostic.

### Action Conversion Used by the Harness

The loader converts raw candidates to pairwise actions:

1. Group raw candidates by objective, e.g. `gonzalez17a::forrester`.
2. For each candidate, choose a deterministic partner using a stable hash of the record id.
3. Build a visible noisy surrogate score for each arm:
   `0.5 + 0.15 * tanh(smooth_prior) + 0.1 * (hash_prior - 0.5) + Gaussian(0, noise_scale)`.
4. Set `noise_scale = 0.06 + 0.18 * average_domain_edge_score`.
5. The candidate action is `Choose candidate A` or `Choose candidate B`, depending on the larger visible surrogate score.
6. Hidden oracle label is `1` if the chosen arm has true normalized utility greater than or equal to the other arm, else `0`.
7. Hidden oracle utility is the true utility margin if the choice is correct, else `0`.

### Visible vs Hidden Fields

- Visible context names the objective family and the two candidate coordinates.
- Visible evidence contains only `surrogate_margin`, estimated noise, and domain-edge score.
- Hidden fields include true objective values, true utilities, chosen true utility, and true margin.
- The visible text explicitly says the latent objective is not observed.

### Example Converted Record

```json
{
  "record_id": "pairwise-duel::forrester::00000",
  "benchmark": "preferential_bo",
  "visible_context": "Preferential optimization duel on forrester. Candidate A: x0=0; candidate B: x0=0.896552. The agent observes only a noisy surrogate ordering, not the latent objective.",
  "candidate_action": "Choose candidate B for the next preferential comparison.",
  "evidence": ["surrogate_margin=0.7296", "estimated_noise=0.2214", "domain_edge_score=0.8966"],
  "label": 0,
  "utility": 0.0
}
```

Interpretation: the visible surrogate preferred B, but hidden normalized utility shows A was better, so the action was not worth executing.

## Task 2: DiSCoVeR-Style Unique Materials

### Public Artifact

- Paper: “DiSCoVeR: a materials discovery screening tool for high performance, unique chemical compositions.”
- DOI: `10.1039/d1dd00028d`.
- Paper code: `https://github.com/sparks-baird/mat_discover`.
- Open reproduction dataset: Matbench `matbench_log_kvrh`, `https://raw.githubusercontent.com/materialsproject/matbench/main/scripts/artifacts/matbench_log_kvrh.json.bz2`.
- Scientific alignment: high bulk modulus plus uniqueness / non-redundancy, matching the DiSCoVeR discovery-screening motivation.

### Raw Material Fields

Each Matbench row contributes:

- `mbid`: material id.
- `composition`: formula string.
- `log10(K_VRH)`: bulk-modulus target value.
- `spg_num`: space group number.
- `crys_sys`: crystal system.

### Label and Utility Construction

The current benchmark uses a lightweight uniqueness proxy instead of fully reproducing DiSCoVeR’s DensMAP density pipeline:

1. `performance_score = normalize(log10(K_VRH))`.
2. Element rarity is computed from inverse square-root element frequency: `mean(1 / sqrt(count(element)))`, then normalized.
3. Space-group rarity is `1 / sqrt(count(space_group))`, then normalized.
4. Crystal-system rarity is `1 / sqrt(count(crystal_system))`, then normalized.
5. `uniqueness_score = 0.6 * element_rarity + 0.3 * space_group_rarity + 0.1 * crystal_system_rarity`.
6. `discovery_score = 0.5 * performance_score + 0.5 * uniqueness_score`.
7. `label = 1` if the material is in the top decile of `discovery_score`, else `0`.
8. `utility = discovery_score`.

### Visible vs Hidden Fields

- Visible context contains composition, crystal system, and space group.
- Exact `log10(K_VRH)` is removed from visible context/evidence during conversion.
- Evidence text replaces exact property values with `property estimate`.
- Hidden oracle features include `performance_score`, `uniqueness_score`, and `discovery_score`.

### Example Converted Record

```json
{
  "record_id": "unique-material::mb-log-kvrh-00001",
  "benchmark": "discover_unique",
  "visible_context": "DiSCoVeR-style task: screen a Materials Project elasticity candidate for high bulk modulus and chemical uniqueness. Composition: Ca1 Ag2 Ge2; crystal system: tetragonal; space group: 139.",
  "candidate_action": "Screen Ca1 Ag2 Ge2 as a high-performance, chemically unique material candidate.",
  "evidence": ["Ca1 Ag2 Ge2", "property estimate", "space_group=139", "crystal_system=tetragonal"],
  "label": 0,
  "utility": 0.355283
}
```

Interpretation: the material has a nonzero discovery utility, but it is not in the top decile of the current performance-plus-uniqueness proxy.

## Task 3: RL-CC Extreme Properties

### Public Artifact

- Paper: “Materials discovery with extreme properties via reinforcement learning-guided combinatorial chemistry.”
- DOI: `10.1039/d3sc05281h`.
- Paper code: `https://github.com/Haeyeon-Choi/RL-CC`.
- Targets: `https://raw.githubusercontent.com/Haeyeon-Choi/RL-CC/main/Target_C1_to_C10.csv`.
- Generated result files: `https://raw.githubusercontent.com/Haeyeon-Choi/RL-CC/main/result/extrapolation/evaluate_Point{point}.csv`.

### Raw Molecule Fields

Each generated row contributes:

- `SMILES`: generated molecule.
- Target and generated values for `MW`, `logP`, `TPSA`, `QED`, `HBA`, `HBD`, and `DRD2`.
- `reward`: score from the paper’s generated result file.

### Target-Hit Bounds

The current label uses the paper’s Table 1 average RMSE bounds:

| Property | Bound |
| --- | ---: |
| `logP` | 0.373 |
| `TPSA` | 5.292 |
| `QED` | 0.078 |
| `HBA` | 1.068 |
| `HBD` | 0.235 |
| `MW` | 7.113 |
| `DRD2` | 0.105 |

### Label and Utility Construction

1. For each property, compute `error = abs(generated_value - target_value)`.
2. A property is a hit if `error <= bound`.
3. `hit_fraction = number_of_hit_properties / 7`.
4. `all_hit = true` only if all seven properties hit.
5. `reward_score = normalize(reward)` within the target file.
6. `target_hit_score = 0.75 * hit_fraction + 0.25 * reward_score`.
7. `label = 1` if `all_hit`, else `0`.
8. `utility = target_hit_score`.

The stored 2,000-record benchmark uses `200` selected generated molecules per target. For each target, positives and negatives are both represented: positives receive up to one third of the per-target budget when available, and selection mixes high-scoring rows with seeded random rows.

### Visible vs Hidden Fields

- Visible context names the target and lists the seven properties to hit.
- Candidate action contains the generated molecule SMILES.
- During harness conversion, evidence values such as `hit_fraction`, `reward`, and `all_hit` are replaced with `outcome withheld`.
- Hidden oracle fields include property errors, hit booleans, reward, hit fraction, and target-hit score.

### Example Converted Record

```json
{
  "record_id": "extreme-rlcc::C1::b450726656c6",
  "benchmark": "extreme_properties",
  "visible_context": "RL-CC extrapolation target C1 from PubChem SARS-CoV-2 clinical-trial molecules. The candidate should hit MW, logP, TPSA, QED, HBA, HBD, and DRD2 within the paper's Table 1 RMSE bounds.",
  "candidate_action": "Advance generated molecule for target C1: CC#CCc1cc2c(...)",
  "evidence": ["target=C1", "outcome withheld", "outcome withheld", "outcome withheld"],
  "label": 1,
  "utility": 1.0
}
```

Interpretation: the hidden generated-property row hits all seven target bounds, but that success signal is not visible to the local harness.

## Feature Construction After Loading

The raw JSONL rows contain many useful fields, but the harness loader intentionally removes oracle-like signals before training. The conversion code is in `src/harness_matsci/historical.py`.

### Oracle Features Excluded

The loader excludes raw uncertainty keys that directly encode outcomes, including:

- `preference_strength`, `latent_utility`, `metric_value`.
- `performance_score`, `uniqueness_score`, `discovery_score`.
- `target_hit_score`, `hit_fraction`, `reward`, `reward_score`, `mean_clipped_error`, `five_hit`.
- Raw `evidence_support`, `evidence_conflict`, `verbal_confidence`, `tool_agreement`, `consensus_spread`, and `ood_score` when those were computed from oracle outcomes upstream.

The loader also excludes oracle context fields such as `performance_score`, `uniqueness_score`, `hit_fraction`, `reward`, `reward_score`, `mean_clipped_error`, and `five_hit`.

### Features Recomputed From Visible Information

After filtering, the loader adds or recomputes visible features:

- Text-derived `verbal_confidence`, `evidence_support`, and `evidence_conflict` from `visible_context + candidate_action`.
- `cost` from `cost_level` and `reversibility` from `reversibility`.
- `action_complexity` from action length and `evidence_count` from number of evidence snippets.
- `source_reliability = 1 - source_risk` when `source_risk` is visible and non-oracle.
- Non-oracle context features, normalized by simple domain rules, such as `space_group / 230`.
- Candidate-derived structure/composition features, such as formula length, number of unique elements, crystal-system one-hot features, SMILES atom count, ring density, and branch density.

For auditability, each converted record stores `metadata.excluded_oracle_features`, `metadata.raw_uncertainty_keys`, and `metadata.raw_context_keys`.

## Splits and Evaluation Protocol

- The scientific regime boundary is `metadata.group_id`.
- Current full-benchmark experiments evaluate 21 regimes: 4 PBO objectives, 7 crystal systems, and 10 RL-CC targets.
- For each seed, a complete `group_id` is held out as the final test regime when possible.
- Remaining records are split into train, feedback, and acceptance partitions.
- RHI-style harness mutation uses feedback trajectories, then accepts or rejects candidate harnesses on the held-out acceptance set.
- Final metrics are computed only on the untouched test partition.
- Main full-benchmark protocol uses five seeds: `1,7,13,21,42`.
- The fixed discovery budget is top `10%` of actions by policy score.
- The target selective-risk threshold is `alpha=0.10`.

## Main Method Inputs and Outputs

### Local Sci-VoI-RHI

- Input: one action record’s visible context, candidate action, visible evidence, and visible numeric features.
- Trained outputs: reliability, expected utility, epistemic uncertainty, verification value, and action-cost-adjusted VoI score.
- Runtime output: route the action to `proceed`, `retrieve_more`, `simulate`, `ask_expert`, `experiment`, or `abstain`.
- Optimization objective: maximize oracle-normalized net scientific utility under a fixed top-10% action budget while keeping selective risk low.

### Direct LLM Judge Baseline

- Input: the same visible context, candidate action, and evidence; no hidden label or utility.
- Output: one scalar `p_success` from an LLM judge.
- Calibration: validation/feedback scores choose thresholds; test scores are final held-out evaluation.
- Current completed direct judge subset uses `gpt-5.5` on 500 balanced records.

### Hybrid LLM + VoI Judge

- Input: local Sci-VoI score plus cached direct LLM judge score for the same action.
- Output: blended or routed action-worthiness score.
- Selection: blend weights and LLM-call budgets are chosen on local calibration records only.
- Current result is a 500-record subset experiment, not the full held-out-regime benchmark.

## Reproduction Commands

Run the leakage and label audit:

```bash
PYTHONPATH=src python3 -m harness_matsci label-audit \
  --data-dir /Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks \
  --out runs/label_audit_v1/summary.json \
  --markdown-out runs/label_audit_v1/README.md
```

Run the full non-LLM Sci-VoI-RHI suite:

```bash
PYTHONPATH=src python3 -m harness_matsci voi-experiment-suite \
  --data-dir /Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks \
  --seeds 1,7,13,21,42 \
  --epochs 60 \
  --iterations 3 \
  --out runs/scivoi_rhi_v1/summary.json \
  --markdown-out runs/scivoi_rhi_v1/README.md
```

Run the mechanism ablation:

```bash
PYTHONPATH=src python3 -m harness_matsci mechanism-ablation \
  --data-dir /Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks \
  --seeds 1,7,13,21,42 \
  --epochs 60 \
  --out runs/mechanism_ablation_v2/summary.json \
  --markdown-out runs/mechanism_ablation_v2/README.md
```

Run the cached hybrid LLM + VoI subset experiment:

```bash
PYTHONPATH=src python3 -m harness_matsci.hybrid_judge \
  --records runs/direct_judge_subset500_v1/records.jsonl \
  --judge-cache runs/direct_judge_cache/subset500_gpt55_scores.json \
  --local-train-fraction 0.5 \
  --out-dir runs/hybrid_llm_judge_subset_v1
```

## Existing Validation Artifacts

| Artifact | What it verifies |
| --- | --- |
| `runs/label_audit_v1/README.md` | Labels and utilities are internally consistent with the proxy definitions; visible text is checked for leakage patterns. |
| `runs/scivoi_rhi_v1/README.md` | Full held-out-regime Sci-VoI-RHI result over 15,717 converted actions. |
| `runs/mechanism_ablation_v2/summary.json` | Utility head, uncertainty, routing, cost, and recursive acceptance ablations. |
| `runs/related_work_baselines_v1/README.md` | Non-LLM related-work baseline sweep. |
| `runs/direct_judge_subset500_v1/README.md` | Real `gpt-5.5` one-shot direct judge result on a balanced 500-record subset. |
| `runs/hybrid_llm_judge_subset_v1/README.md` | Cached `gpt-5.5` plus local VoI hybrid subset result. |

## 500 Electrolyte-Paper Pilot

The 500-row pilot lives at `/Users/huan/matsci_uncertainty/data/raw/nature_electrolyte/training_actions_500.jsonl` and is used by `src/harness_matsci/paper_bootstrap.py`.

- Domain: battery electrolyte papers.
- Record type: paper segment converted into an action such as `retrieve_more`, `recommend_experiment`, `execute_tool`, or `commit_decision`.
- Label: weak `outcome_success` assigned by the paper-data builder according to whether the route is appropriate for the extracted evidence state.
- Utility: raw `metric_value`, converted directly to `ActionRecord.utility`.
- Purpose: pilot evidence that historical papers can bootstrap action-level gate training.
- Limitation: this pilot is smaller, noisier, and less benchmark-grounded than the three main public artifact families.

## Limitations to State in Papers

- These are offline proxy action labels, not expert annotations of live MatBot decisions.
- `discover_unique` uses a transparent rarity proxy instead of a full DensMAP reproduction of DiSCoVeR.
- `preferential_bo` is an optimization benchmark and not itself a materials database.
- `extreme_properties` depends on published RL-CC generated result files and property-hit tolerances.
- The local gate has limited materials semantic grounding unless paired with stronger tools, retrieval, or an LLM judge.
- The next required validation is online MatBot trajectory logging with expert or downstream experiment labels.
