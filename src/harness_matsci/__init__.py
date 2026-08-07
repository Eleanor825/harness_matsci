"""Materials-science runtime uncertainty harness."""

from .campaign import CampaignConfig, run_campaign
from .benchmarks import make_records
from .direct_judge import DirectJudgeError, LLMDirectJudge
from .experiments import ExperimentSuiteConfig, run_experiment_suite, save_experiment_suite
from .paper_bootstrap import load_paper_action_records, run_paper_bootstrap_experiment
from .rhi import DeterministicTrajectoryProposer, JSONLLMHarnessProposer, train_rhi
from .schema import ActionRecord, GateDecision, HarnessSpec
from .models import LogisticGate
from .training import train_gate, evaluate_gate

__all__ = [
    "ActionRecord",
    "CampaignConfig",
    "ExperimentSuiteConfig",
    "make_records",
    "load_paper_action_records",
    "GateDecision",
    "HarnessSpec",
    "LogisticGate",
    "DeterministicTrajectoryProposer",
    "DirectJudgeError",
    "JSONLLMHarnessProposer",
    "LLMDirectJudge",
    "run_campaign",
    "save_experiment_suite",
    "run_experiment_suite",
    "run_paper_bootstrap_experiment",
    "train_rhi",
    "train_gate",
    "evaluate_gate",
]
