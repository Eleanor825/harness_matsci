from __future__ import annotations

import unittest

from harness_matsci.benchmarks import BENCHMARK_BUILDERS, make_records


class BenchmarkTests(unittest.TestCase):
    def test_benchmarks_produce_both_labels(self) -> None:
        for benchmark in sorted(BENCHMARK_BUILDERS):
            records = make_records(benchmark, n=120, seed=7)
            labels = [record.label for record in records]
            self.assertEqual(len(records), 120)
            self.assertIn(0, labels, benchmark)
            self.assertIn(1, labels, benchmark)
            self.assertEqual(len({record.record_id for record in records}), len(records))
            self.assertTrue(all(record.benchmark == benchmark for record in records))


if __name__ == "__main__":
    unittest.main()
