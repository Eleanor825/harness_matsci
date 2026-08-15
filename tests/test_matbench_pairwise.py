from __future__ import annotations

import bz2
import json
import tempfile
import unittest
from pathlib import Path

from harness_matsci.audit_experiments import LabelAuditConfig, run_label_utility_audit
from harness_matsci.historical import load_historical_task_records
from harness_matsci.matbench_pairwise import MatbenchPairwiseConfig, write_matbench_pairwise_dataset


class MatbenchPairwiseTests(unittest.TestCase):
    def test_matbench_table_builds_pairwise_material_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source_path = directory / "matbench_log_kvrh.json.bz2"
            out_path = directory / "matbench_pairwise_actions.jsonl"
            _write_fake_matbench_log_kvrh(source_path)

            report = write_matbench_pairwise_dataset(
                MatbenchPairwiseConfig(
                    out=str(out_path),
                    source_path=str(source_path),
                    n_pairs=60,
                    seed=13,
                    update_summary=True,
                )
            )
            self.assertEqual(report["summary"]["records"], 60)
            self.assertTrue((directory / "summary.json").exists())

            records = load_historical_task_records(directory, "matbench_pairwise")
            self.assertEqual(len(records), 60)
            self.assertTrue({record.label for record in records} <= {0, 1})
            self.assertGreater(len({record.metadata["group_id"] for record in records}), 1)
            for record in records:
                visible_text = " ".join([record.visible_context, record.candidate_action, *record.evidence]).lower()
                self.assertNotIn("log10(k_vrh)=", visible_text)
                self.assertNotIn("log10_k_vrh=", visible_text)

            audit = run_label_utility_audit(
                LabelAuditConfig(
                    data_dir=str(directory),
                    tasks=("matbench_pairwise",),
                    sample_per_task=4,
                )
            )
            self.assertTrue(audit["aggregate"]["all_label_consistency_passed"])
            self.assertTrue(audit["aggregate"]["all_utility_consistency_passed"])
            self.assertTrue(audit["aggregate"]["all_visible_leakage_free"])


def _write_fake_matbench_log_kvrh(path: Path) -> None:
    rows = [
        ("mb-0001", "Al2 O3", 2.10, 167, "trigonal"),
        ("mb-0002", "Ca1 Ag2 Ge2", 1.70, 139, "tetragonal"),
        ("mb-0003", "Si1 O2", 1.95, 154, "trigonal"),
        ("mb-0004", "Mg1 O1", 1.55, 225, "cubic"),
        ("mb-0005", "Ti1 O2", 1.82, 136, "tetragonal"),
        ("mb-0006", "Fe1 S2", 1.05, 205, "cubic"),
        ("mb-0007", "Zn1 O1", 1.38, 186, "hexagonal"),
        ("mb-0008", "B1 N1", 2.35, 216, "cubic"),
        ("mb-0009", "Ga1 N1", 1.48, 186, "hexagonal"),
        ("mb-0010", "Li1 F1", 1.25, 225, "cubic"),
        ("mb-0011", "Y1 Al1 O3", 1.62, 62, "orthorhombic"),
        ("mb-0012", "Zr1 O2", 1.88, 14, "monoclinic"),
    ]
    payload = {
        "mbid": {},
        "composition": {},
        "log10(K_VRH)": {},
        "spg_num": {},
        "crys_sys": {},
    }
    for index, row in enumerate(rows):
        key = str(index)
        material_id, composition, property_value, space_group, crystal_system = row
        payload["mbid"][key] = material_id
        payload["composition"][key] = composition
        payload["log10(K_VRH)"][key] = property_value
        payload["spg_num"][key] = space_group
        payload["crys_sys"][key] = crystal_system
    path.write_bytes(bz2.compress(json.dumps(payload).encode("utf-8")))


if __name__ == "__main__":
    unittest.main()
