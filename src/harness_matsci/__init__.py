"""Materials-science runtime uncertainty harness."""

from .campaign import CampaignConfig, run_campaign
from .benchmarks import make_records
from .paper_bootstrap import load_paper_action_records, run_paper_bootstrap_experiment
from .rhi import DeterministicTrajectoryProposer, JSONLLMHarnessProposer, train_rhi
from .schema import ActionRecord, GateDecision, HarnessSpec
from .models import LogisticGate
from .training import train_gate, evaluate_gate

__all__ = [
    "ActionRecord",
    "CampaignConfig",
    "make_records",
    "load_paper_action_records",
    "GateDecision",
    "HarnessSpec",
    "LogisticGate",
    "DeterministicTrajectoryProposer",
    "JSONLLMHarnessProposer",
    "run_campaign",
    "run_paper_bootstrap_experiment",
    "train_rhi",
    "train_gate",
    "evaluate_gate",
]
