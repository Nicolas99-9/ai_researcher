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


class TestHypothesisEvidence:
    def test_hypothesis_updated_after_ablation(self):
        tracker = HypothesisTracker()
        h = Hypothesis(
            claim="Attention mechanism drives improvement",
            prediction="Removing attention drops accuracy by >8%",
            source_node_id="n1",
        )
        tracker.add(h)

        # Simulate ablation result
        h.update_with_evidence(
            result="Accuracy dropped from 0.92 to 0.71 (23% drop)",
            falsified=False,
            new_confidence=0.9,
            node_id="ablation_n1",
        )
        assert h.status == "supported"
        assert h.confidence == 0.9

    def test_hypothesis_falsified_after_ablation(self):
        tracker = HypothesisTracker()
        h = Hypothesis(
            claim="Skip connections are essential",
            prediction="Removing skip connections drops accuracy by >10%",
            source_node_id="n1",
        )
        tracker.add(h)

        h.update_with_evidence(
            result="Accuracy only dropped by 0.5%",
            falsified=True,
            new_confidence=0.1,
            node_id="ablation_n2",
        )
        assert h.status == "falsified"
