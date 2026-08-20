from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LogitMargin:
    positive_logit: float
    negative_logit: float
    margin: float
    preferred: str
    confidence: float
    uncertainty: float

    def to_json(self) -> dict[str, Any]:
        return {
            "positive_logit": self.positive_logit,
            "negative_logit": self.negative_logit,
            "margin": self.margin,
            "preferred": self.preferred,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
        }


def calibrated_confidence(margin: float, temperature: float = 1.0) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    value = abs(float(margin)) / temperature
    return 1.0 / (1.0 + math.exp(-min(60.0, value)))


def execute_probability(margin: float, temperature: float = 1.0) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    value = float(margin) / temperature
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def margin_to_uncertainty(margin: float, temperature: float = 1.0) -> float:
    return 1.0 - calibrated_confidence(margin, temperature)


class CausalLMLogitJudge:
    """Extract candidate-token logit margins from a local causal LM.

    The prompt must end immediately before a one-token A/B decision. The
    returned margin is an internal model signal, not a verbal confidence.
    """

    def __init__(
        self,
        model_name: str,
        *,
        positive_token: str = "A",
        negative_token: str = "B",
        temperature: float = 1.0,
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("logit extraction requires torch and transformers") from exc
        self.torch = torch
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.temperature = temperature
        positive_ids = self.tokenizer.encode(positive_token, add_special_tokens=False)
        negative_ids = self.tokenizer.encode(negative_token, add_special_tokens=False)
        if len(positive_ids) != 1 or len(negative_ids) != 1:
            raise ValueError("positive_token and negative_token must each tokenize to one token")
        self.positive_token = positive_token
        self.negative_token = negative_token
        self.positive_id = positive_ids[0]
        self.negative_id = negative_ids[0]

    def score(self, prompts: list[str], *, batch_size: int = 8) -> list[LogitMargin]:
        results: list[LogitMargin] = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.no_grad():
                logits = self.model(**encoded).logits
            lengths = encoded["attention_mask"].sum(dim=1).tolist()
            for row, length in enumerate(lengths):
                last = logits[row, int(length) - 1]
                positive = float(last[self.positive_id].detach().cpu())
                negative = float(last[self.negative_id].detach().cpu())
                margin = positive - negative
                confidence = calibrated_confidence(margin, self.temperature)
                results.append(
                    LogitMargin(
                        positive_logit=positive,
                        negative_logit=negative,
                        margin=margin,
                        preferred=self.positive_token if margin >= 0 else self.negative_token,
                        confidence=confidence,
                        uncertainty=1.0 - confidence,
                    )
                )
        return results
