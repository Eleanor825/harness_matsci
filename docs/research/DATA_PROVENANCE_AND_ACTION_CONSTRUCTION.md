# Data Provenance and Action Construction

This is the reproducibility map for the offline MatSci uncertainty-harness data. It states exactly which public artifacts are used, how each material record becomes an action-level decision, what the harness can see, and how hidden `label` / `utility` are computed.

## Core Boundary

- The main benchmark is **not** live MatBot trajectory logging and is **not** manual PDF extraction.
- The main materials benchmark now uses **20,987 action records** from three materials / chemical discovery task families.
- `preferential_bo` is retained only as an **auxiliary controlled optimization sanity check**. It is not counted as a main materials benchmark.
- The 500 electrolyte-paper dataset is a separate historical-paper pilot; it is not part of the 20,987-record main materials benchmark.

## Main Materials Benchmark Inventory

| Task family | Scientific question | Public source | Raw file | Records | Positives | Groups / regimes | Label source |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `matbench_pairwise` | Which real material in A/B is better for high-bulk-modulus follow-up? | Matbench `log10(K_VRH)`, derived from Materials Project elasticity data, `https://raw.githubusercontent.com/materialsproject/matbench/main/scripts/artifacts/matbench_log_kvrh.json.bz2` | `matbench_pairwise_actions.jsonl` | 8,000 | 4,427 | 28 crystal-system pairs | Hidden normalized `log10(K_VRH)` comparison |
| `discover_unique` | Which material is high-performing and chemically unique? | DiSCoVeR, DOI `10.1039/d1dd00028d`; code `https://github.com/sparks-baird/mat_discover`; Matbench `log10(K_VRH)` | `unique_materials_actions.jsonl` | 10,987 | 1,099 | 7 crystal systems | Top-decile performance + uniqueness proxy |
| `extreme_properties` | Which generated molecule hits extreme target properties? | RL-CC, DOI `10.1039/d3sc05281h`; code `https://github.com/Haeyeon-Choi/RL-CC` | `extreme_properties_actions.jsonl` | 2,000 | 313 | 10 targets | Seven-property target hit within Table 1 RMSE bounds |

Main materials total: `20,987` action records across `45` scientific regimes.

## Auxiliary Sanity Check

| Task family | Purpose | Source | Records | Why auxiliary only |
| --- | --- | --- | ---: | --- |
| `preferential_bo` | Controlled A/B preference-optimization sanity check | González et al., Preferential Bayesian Optimization, ICML 2017, `https://proceedings.mlr.press/v70/gonzalez17a.html` | 2,730 converted duels | It has a clean latent objective but no material composition, structure, property table, or materials database distribution. |

PBO can still test whether the harness handles noisy pairwise decisions, but it should not support the paper’s materials-discovery claim.

## Local Raw Data Directory

Current local data live at `/Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks`.

- `matbench_pairwise_actions.jsonl`: 8,000 real-material A/B preference actions.
- `unique_materials_actions.jsonl`: 10,987 DiSCoVeR-style screening actions.
- `extreme_properties_actions.jsonl`: 2,000 RL-CC molecule actions.
- `pairwise_optimization_actions.jsonl`: 2,730 auxiliary PBO candidate rows, converted to duels by the loader.
- `summary.json`: local manifest. If all four files are counted, total records are `23,717`; main materials only are `20,987`.

## `ActionRecord` Contract

Every experiment consumes `ActionRecord` rows from `src/harness_matsci/schema.py`.

| Field | Meaning | Visible at decision time? |
| --- | --- | --- |
| `record_id` | Stable action id | Yes |
| `benchmark` | Task family, e.g. `matbench_pairwise` | Yes |
| `visible_context` | What the agent/harness can read | Yes |
| `candidate_action` | Proposed next step | Yes |
| `action_type` | Action category, e.g. `choose` or `choose_candidate` | Yes |
| `evidence` | Pre-execution evidence after leakage sanitation | Yes |
| `features` | Numeric uncertainty, source, cost, stability, and visible-candidate signals | Yes |
| `label` | Whether the action was worth executing | Hidden oracle label |
| `utility` | Normalized downstream scientific gain | Hidden oracle utility |
| `metadata` | Source, group id, raw ids, and audit fields | Mixed; oracle fields are not model inputs |

The harness learns whether the **current proposed action** is worth executing under visible information; it does not directly train a property predictor.

## Task 1: `matbench_pairwise`

### Why This Replaces PBO as the Main Pairwise Task

The previous PBO task had the right A/B preference form but was not materials-related. `matbench_pairwise` keeps the A/B action structure while grounding every comparison in real Materials Project / Matbench materials and a real hidden material property.

### Public Artifact

- Dataset: Matbench `log10(K_VRH)`.
- URL: `https://raw.githubusercontent.com/materialsproject/matbench/main/scripts/artifacts/matbench_log_kvrh.json.bz2`.
- Source domain: Materials Project elasticity data.
- Scientific objective: prefer the material with higher hidden bulk-modulus target.

### Construction Pipeline

1. Load all `10,987` Matbench rows with `composition`, `log10(K_VRH)`, `spg_num`, and `crys_sys`.
2. Normalize true `log10(K_VRH)` to `[0,1]`; this normalized value is hidden.
3. Sample 8,000 material pairs, mostly close-property pairs to make decisions nontrivial.
4. Group pairs by unordered crystal-system pair, giving 28 regimes.
5. Create a noisy surrogate score for each material from the hidden normalized property plus noise.
6. The candidate action chooses the material with the higher noisy surrogate score.
7. `label = 1` if the chosen material has hidden normalized `log10(K_VRH)` at least as high as the other material.
8. `utility = abs(normalized_property_A - normalized_property_B)` if the choice is correct, else `0`.
9. Pairs with true normalized gap below `0.003` are filtered so positive actions have nonzero utility.

### Visible vs Hidden

Visible fields include:

- Composition of material A and B.
- Crystal system and space group.
- Coarse surrogate bulk-modulus tier: `low`, `medium`, or `high`.
- Surrogate margin and estimated noise.

Hidden fields include:

- Exact `log10(K_VRH)` for both materials.
- Normalized `log10(K_VRH)` for both materials.
- Which material is truly better.
- True property gap and utility.

Exact `log10(K_VRH)` is never placed in `visible_context` or `evidence`.

### Example Converted Record

```json
{
  "benchmark": "matbench_pairwise",
  "visible_context": "Matbench elasticity pairwise follow-up. Candidate A: Composition: Sm1 Te1; crystal system: cubic; space group: 221; surrogate bulk-modulus tier: high. Candidate B: Composition: Na4 Mn4 F12; crystal system: orthorhombic; space group: 62; surrogate bulk-modulus tier: high. The true Materials Project elasticity target is withheld until evaluation.",
  "candidate_action": "Choose candidate A for high-bulk-modulus materials follow-up.",
  "evidence": ["A_surrogate_tier=high", "B_surrogate_tier=high", "surrogate_margin=0.0343", "estimated_noise=0.4468"],
  "label": 1,
  "utility": 0.003007
}
```

## Task 2: `discover_unique`

### Public Artifact

- Paper: “DiSCoVeR: a materials discovery screening tool for high performance, unique chemical compositions.”
- DOI: `10.1039/d1dd00028d`.
- Paper code: `https://github.com/sparks-baird/mat_discover`.
- Open reproduction dataset: Matbench `log10(K_VRH)`.

### Label and Utility

1. `performance_score = normalize(log10(K_VRH))`.
2. Element rarity uses inverse square-root element frequency.
3. Space-group rarity uses inverse square-root space-group frequency.
4. Crystal-system rarity uses inverse square-root crystal-system frequency.
5. `uniqueness_score = 0.6 * element_rarity + 0.3 * space_group_rarity + 0.1 * crystal_system_rarity`.
6. `discovery_score = 0.5 * performance_score + 0.5 * uniqueness_score`.
7. `label = 1` if the material is in the top decile of `discovery_score`.
8. `utility = discovery_score`.

Visible fields contain composition, crystal system, and space group. Exact property values are replaced by `property estimate` before training/evaluation.

## Task 3: `extreme_properties`

### Public Artifact

- Paper: “Materials discovery with extreme properties via reinforcement learning-guided combinatorial chemistry.”
- DOI: `10.1039/d3sc05281h`.
- Code: `https://github.com/Haeyeon-Choi/RL-CC`.
- Targets: `https://raw.githubusercontent.com/Haeyeon-Choi/RL-CC/main/Target_C1_to_C10.csv`.
- Results: `https://raw.githubusercontent.com/Haeyeon-Choi/RL-CC/main/result/extrapolation/evaluate_Point{point}.csv`.

### Label and Utility

The target-hit bounds are the paper’s Table 1 average RMSE values: `logP=0.373`, `TPSA=5.292`, `QED=0.078`, `HBA=1.068`, `HBD=0.235`, `MW=7.113`, and `DRD2=0.105`.

1. Compute absolute error for each generated property against the target.
2. A property is a hit if `error <= bound`.
3. `hit_fraction = number_of_hit_properties / 7`.
4. `label = 1` only if all seven properties hit.
5. `reward_score = normalize(reward)` within the target file.
6. `utility = target_hit_score = 0.75 * hit_fraction + 0.25 * reward_score`.

Visible evidence values such as `hit_fraction`, `reward`, and `all_hit` are replaced by `outcome withheld` before training/evaluation.

## Auxiliary: `preferential_bo`

PBO is still useful for a clean controlled check, but it is not a materials benchmark.

- It uses Forrester, Six-Hump Camel, Goldstein-Price, and Levy latent objectives.
- The loader turns raw objective candidates into noisy pairwise duels.
- `label = 1` if the surrogate-chosen arm has higher hidden normalized utility.
- `utility = true_margin` if correct, else `0`.
- It should be reported as appendix / sanity check only.

## Leakage Controls

The conversion code is in `src/harness_matsci/historical.py`.

- `log10(K_VRH)` and `log10_k_vrh` patterns are removed from visible text/evidence for Matbench-derived tasks.
- RL-CC outcome fields such as `hit_fraction`, `reward`, and `all_hit` are replaced by `outcome withheld`.
- Raw oracle signals such as `latent_utility`, `performance_score`, `uniqueness_score`, `discovery_score`, `target_hit_score`, `reward`, and `mean_clipped_error` are excluded from `features`.
- Converted records store `metadata.excluded_oracle_features`, `metadata.raw_uncertainty_keys`, and `metadata.raw_context_keys` for audit.

## Evaluation Protocol

- Main material regimes: 28 Matbench crystal-system pairs, 7 DiSCoVeR crystal systems, and 10 RL-CC targets.
- Main material total: 45 held-out regimes.
- Splits are action-level train / feedback / acceptance plus complete-regime held-out test where the suite supports it.
- Fixed action budget: top 10% actions by policy score.
- Target selective risk: `alpha=0.10`.
- Main objective: maximize oracle-normalized net scientific utility under the action budget while keeping selective risk low.

## Reproduction Commands

Generate the Matbench pairwise data from the local Matbench cache:

```bash
PYTHONPATH=src python3 -m harness_matsci.matbench_pairwise \
  --source-path /Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks/cache/matbench_log_kvrh.json.bz2 \
  --out /Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks/matbench_pairwise_actions.jsonl \
  --n-pairs 8000 \
  --seed 1729 \
  --min-true-gap 0.003 \
  --update-summary
```

Audit the new pairwise task:

```bash
PYTHONPATH=src python3 -m harness_matsci label-audit \
  --data-dir /Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks \
  --tasks matbench_pairwise \
  --out runs/label_audit_matbench_pairwise_v1/summary.json \
  --markdown-out runs/label_audit_matbench_pairwise_v1/README.md
```

Audit the three main material tasks:

```bash
PYTHONPATH=src python3 -m harness_matsci label-audit \
  --data-dir /Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks \
  --tasks matbench_pairwise,discover_unique,extreme_properties \
  --out runs/label_audit_main_materials_v1/summary.json \
  --markdown-out runs/label_audit_main_materials_v1/README.md
```

Run a Matbench pairwise VoI smoke check:

```bash
PYTHONPATH=src python3 -m harness_matsci voi-experiment-suite \
  --data-dir /Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks \
  --tasks matbench_pairwise \
  --methods verbal_confidence,evidence_heuristic,h0_reliability,static_voi,scivoi_rhi \
  --components '' \
  --acceptance-policies '' \
  --seeds 1 \
  --epochs 10 \
  --iterations 1 \
  --out runs/matbench_pairwise_smoke_v1/summary.json \
  --markdown-out runs/matbench_pairwise_smoke_v1/README.md
```

## Current Validation Artifacts

| Artifact | What it verifies |
| --- | --- |
| `runs/label_audit_matbench_pairwise_v1/README.md` | New Matbench pairwise labels/utilities are exactly recomputable from hidden property values; visible text has no property leakage. |
| `runs/label_audit_main_materials_v1/README.md` | The three main material tasks have consistent labels/utilities and no visible leakage. |
| `runs/matbench_pairwise_smoke_v1/README.md` | One-seed smoke check that the VoI suite runs on the new real-material pairwise task. |
| `runs/scivoi_rhi_v1/README.md` | Preserved legacy full run over the earlier 15,717-record setting that included PBO. |

## 500 Electrolyte-Paper Pilot

The pilot file is `/Users/huan/matsci_uncertainty/data/raw/nature_electrolyte/training_actions_500.jsonl` and is consumed by `src/harness_matsci/paper_bootstrap.py`.

- Domain: battery electrolyte papers.
- Record type: paper segment converted into actions such as `retrieve_more`, `recommend_experiment`, `execute_tool`, or `commit_decision`.
- Purpose: pilot evidence that historical papers can bootstrap action-level gate training.
- Limitation: smaller and noisier than the main public material-artifact tasks.

## Current Limitations

- The main data are still offline action reconstructions, not live MatBot trajectories.
- `matbench_pairwise` is real materials data, but candidate actions are generated from a noisy surrogate rather than emitted by MatBot.
- `discover_unique` uses a transparent rarity proxy instead of a full DensMAP reproduction.
- `extreme_properties` is molecular / chemical discovery, not inorganic crystal screening.
- Full paper results should be rerun on `matbench_pairwise,discover_unique,extreme_properties`; preserved full results currently correspond to the earlier PBO-including setup.
