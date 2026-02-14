import pytest
from ai_scientist.treesearch.hypothesis_tracker import (
    Hypothesis,
    HypothesisTracker,
    build_ablation_prompt_from_hypothesis,
)


class TestHypothesisDrivenAblation:
    def test_ablation_prompt_from_hypothesis(self):
        h = Hypothesis(
            claim="Multi-head attention is responsible for >60% of improvement",
            prediction="Removing multi-head attention drops accuracy by >8%",
            source_node_id="abc-123",
        )
        prompt = build_ablation_prompt_from_hypothesis(
            hypothesis=h,
            base_code="import torch\n# ... model code ...",
            previous_ablations=["remove_dropout", "reduce_layers"],
        )
        assert "attention" in prompt.lower()
        assert "remov" in prompt.lower()  # "removing" or "remove"
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_ablation_prompt_mentions_prediction(self):
        h = Hypothesis(
            claim="Dropout prevents overfitting",
            prediction="Removing dropout increases overfitting gap by >5%",
            source_node_id="xyz",
        )
        prompt = build_ablation_prompt_from_hypothesis(
            hypothesis=h,
            base_code="model code here",
            previous_ablations=[],
        )
        assert "5%" in prompt or "overfit" in prompt.lower()
