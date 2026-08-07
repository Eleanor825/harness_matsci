from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from urllib.request import Request

from harness_matsci.direct_judge import LLMDirectJudge, _direct_judge_prompt
from harness_matsci.experiments import ExperimentSuiteConfig, run_experiment_suite
from harness_matsci.schema import ActionRecord


class DirectJudgeTests(unittest.TestCase):
    def _record(self, record_id: str = "r1") -> ActionRecord:
        return ActionRecord(
            record_id=record_id,
            benchmark="preferential_bo",
            split="test",
            visible_context="A pairwise materials optimization decision.",
            candidate_action="Select candidate A for the next comparison.",
            action_type="choose",
            evidence=["pairwise feedback is moderately consistent"],
            features={"verbal_confidence": 0.5},
            label=1,
        )

    def test_prompt_contains_only_visible_decision_information(self) -> None:
        prompt = _direct_judge_prompt(self._record())
        self.assertIn("candidate A", prompt)
        self.assertNotIn("label", prompt)
        self.assertNotIn("hidden objective", prompt.lower())

    def test_parses_score_and_reuses_cache(self) -> None:
        calls: list[Request] = []

        def request(request: Request, timeout: float) -> bytes:
            calls.append(request)
            return json.dumps(
                {"choices": [{"message": {"content": '{"p_success": 0.73, "route": "proceed"}'}}]}
            ).encode()

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "scores.json"
            judge = LLMDirectJudge(
                model="test-model",
                api_key="test-key",
                cache_path=cache,
                request_fn=request,
            )
            self.assertEqual(judge.score_records([self._record()]), {"r1": 0.73})
            self.assertEqual(judge.score_records([self._record()]), {"r1": 0.73})
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads(cache.read_text())["scores"]["r1"], 0.73)

    def test_uses_responses_wire_and_gpt54_defaults(self) -> None:
        requests: list[Request] = []

        def request(request: Request, timeout: float) -> bytes:
            requests.append(request)
            return json.dumps(
                {
                    "output_text": '{"p_success": 0.61, "route": "retrieve_more", "rationale": "uncertain"}',
                }
            ).encode()

        judge = LLMDirectJudge.from_env(
            environ={"OPENAI_API_KEY": "test-key"},
            request_fn=request,
        )
        self.assertEqual(judge.model, "gpt-5.6-luna")
        self.assertEqual(judge.base_url, "https://coding.beehears.com")
        self.assertEqual(judge.reasoning_effort, "xhigh")
        self.assertEqual(judge.score_records([self._record()]), {"r1": 0.61})
        self.assertTrue(requests[0].full_url.endswith("/responses"))
        body = json.loads(requests[0].data)
        self.assertEqual(body["model"], "gpt-5.6-luna")
        self.assertEqual(body["reasoning"], {"effort": "xhigh"})
        self.assertFalse(body["store"])

    def test_fake_direct_judge_is_reported_in_suite(self) -> None:
        def fake_judge(records: list[ActionRecord]) -> dict[str, float]:
            return {record.record_id: 0.8 if record.features.get("verbal_confidence", 0.0) > 0.5 else 0.2 for record in records}

        report = run_experiment_suite(
            ExperimentSuiteConfig(
                experiments=(1,),
                seeds=(3,),
                n_per_task=60,
                rhi_iterations=0,
                epochs=30,
            ),
            direct_judge=fake_judge,
        )
        methods = report["experiments"]["experiment_1_single_task"]["by_method"]
        self.assertIn("llm_direct_judge", methods)
        self.assertTrue(report["baselines"]["llm_direct_judge"]["enabled"])
        self.assertEqual(report["summary"]["n_single_runs"], 18)


if __name__ == "__main__":
    unittest.main()
