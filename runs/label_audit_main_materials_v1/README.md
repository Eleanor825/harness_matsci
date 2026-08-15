# Label and Utility Audit

This audit checks whether the offline proxy labels/utilities are internally consistent with their stored benchmark outcomes and whether hidden outcome fields are withheld from visible decision-time text.

## Aggregate Verdict

- Raw records: `20987`.
- Converted action records: `20987`.
- Label consistency passed: `True`.
- Utility consistency passed: `True`.
- Visible leakage free: `True`.
- Raw oracle keys recorded as excluded: `True`.

## Task Summary

| Task | Records | Groups | Positive rate | Utility mean | Label mismatches | Utility mismatches | Visible leaks | Raw oracle keys excluded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `matbench_pairwise` | 8000 | 28 | 0.553 | 0.0443 | 0 | 0 | 0 | n/a |
| `discover_unique` | 10987 | 7 | 0.100 | 0.3998 | 0 | 0 | 0 | 109870/109870 |
| `extreme_properties` | 2000 | 10 | 0.157 | 0.6878 | 0 | 0 | 0 | 18000/18000 |

## What This Establishes

- `label` is reproducible from stored benchmark outcome metadata for all converted action records.
- `utility` is reproducible from raw benchmark utility fields for non-pairwise tasks and from hidden true margins for pairwise tasks such as Matbench A/B preference and auxiliary PBO duels.
- Hidden outcome strings such as exact objective values, `log10(K_VRH)`, `hit_fraction`, `reward`, and `all_hit` are absent from visible context/evidence after conversion.
- Raw oracle-valued uncertainty fields are recorded in `excluded_oracle_features` so reviewers can audit what was withheld.

## Claim Boundary

This is a proxy-label consistency audit, not proof that live MatBot action labels are correct. A final paper should still add a small expert spot-check or online trajectory audit.
