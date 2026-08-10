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
from .voi import VOI_SEED_HARNESS, fit_voi_model, train_voi_rhi, evaluate_voi
from .voi_experiments import VoIExperimentConfig, run_voi_experiment_suite, save_voi_experiment_suite

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
    "VoIExperimentConfig",
    "VOI_SEED_HARNESS",
    "evaluate_voi",
    "fit_voi_model",
    "run_voi_experiment_suite",
    "save_voi_experiment_suite",
    "train_voi_rhi",
    "train_rhi",
    "train_gate",
    "evaluate_gate",
]
