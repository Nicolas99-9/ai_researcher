import pytest
from ai_scientist.treesearch.hypothesis_tracker import Hypothesis, HypothesisTracker


class TestHypothesis:
    def test_create_hypothesis(self):
        h = Hypothesis(
            claim="Attention mechanism is responsible for >50% of improvement",
            prediction="Removing attention drops accuracy by >50% of the gain",
            source_node_id="abc-123",
            confidence=0.5,
        )
        assert h.claim != ""
        assert h.status == "untested"
        assert h.confidence == 0.5

    def test_update_after_falsification(self):
        h = Hypothesis(
            claim="Dropout is essential",
            prediction="Removing dropout drops accuracy by >5%",
            source_node_id="abc-123",
        )
        h.update_with_evidence(
            result="Accuracy dropped by 1%",
            falsified=True,
            new_confidence=0.1,
        )
        assert h.status == "falsified"
        assert h.confidence == 0.1
        assert len(h.evidence) == 1

    def test_update_after_support(self):
        h = Hypothesis(
            claim="Data augmentation helps",
            prediction="Removing augmentation drops accuracy by >3%",
            source_node_id="abc-123",
        )
        h.update_with_evidence(
            result="Accuracy dropped by 8%",
            falsified=False,
            new_confidence=0.85,
        )
        assert h.status == "supported"
        assert h.confidence == 0.85

    def test_serialization_roundtrip(self):
        h = Hypothesis(
            claim="test claim",
            prediction="test prediction",
            source_node_id="xyz",
        )
        d = h.to_dict()
        restored = Hypothesis.from_dict(d)
        assert restored.claim == h.claim
        assert restored.status == h.status


class TestHypothesisTracker:
    def test_add_hypothesis(self):
        tracker = HypothesisTracker()
        tracker.add(Hypothesis(
            claim="Method X works",
            prediction="Removing X hurts",
            source_node_id="n1",
        ))
        assert len(tracker) == 1

    def test_get_untested(self):
        tracker = HypothesisTracker()
        h1 = Hypothesis(claim="A", prediction="P1", source_node_id="n1")
        h2 = Hypothesis(claim="B", prediction="P2", source_node_id="n2")
        h2.status = "supported"
        tracker.add(h1)
        tracker.add(h2)
        untested = tracker.get_untested()
        assert len(untested) == 1
        assert untested[0].claim == "A"

    def test_get_by_status(self):
        tracker = HypothesisTracker()
        h1 = Hypothesis(claim="A", prediction="P1", source_node_id="n1")
        h1.status = "falsified"
        h2 = Hypothesis(claim="B", prediction="P2", source_node_id="n2")
        h2.status = "supported"
        tracker.add(h1)
        tracker.add(h2)
        assert len(tracker.get_by_status("falsified")) == 1
        assert len(tracker.get_by_status("supported")) == 1

    def test_format_for_prompt(self):
        tracker = HypothesisTracker()
        tracker.add(Hypothesis(
            claim="Attention is key",
            prediction="Remove attention → big drop",
            source_node_id="n1",
        ))
        text = tracker.format_for_prompt()
        assert "Attention" in text

    def test_format_for_paper(self):
        tracker = HypothesisTracker()
        h = Hypothesis(claim="Method works", prediction="Ablation shows drop", source_node_id="n1")
        h.update_with_evidence(result="5% drop confirmed", falsified=False, new_confidence=0.9)
        tracker.add(h)
        paper_text = tracker.format_for_paper()
        assert "supported" in paper_text.lower() or "Method works" in paper_text

    def test_serialization_roundtrip(self):
        tracker = HypothesisTracker()
        tracker.add(Hypothesis(claim="X", prediction="Y", source_node_id="n1"))
        d = tracker.to_dict()
        restored = HypothesisTracker.from_dict(d)
        assert len(restored) == len(tracker)


class TestHypothesisGeneration:
    def test_generate_hypothesis_prompt_structure(self):
        """Verify the hypothesis generation prompt has required components."""
        from ai_scientist.treesearch.hypothesis_tracker import build_hypothesis_generation_prompt
        prompt = build_hypothesis_generation_prompt(
            research_idea="Study the effect of attention mechanisms on small datasets",
            best_node_plan="Implemented multi-head attention with 4 heads",
            best_node_analysis="Model achieved 0.92 accuracy, outperforming baseline MLP at 0.78",
            best_node_code="import torch\n...",
        )
        assert "hypothesis" in prompt.lower() or "claim" in prompt.lower()
        assert "attention" in prompt.lower()
        assert "prediction" in prompt.lower() or "falsif" in prompt.lower()

    def test_hypothesis_func_spec_exists(self):
        """Verify the FunctionSpec for hypothesis generation exists."""
        from ai_scientist.treesearch.hypothesis_tracker import hypothesis_generation_spec
        assert hypothesis_generation_spec.name == "generate_hypotheses"
        assert "hypotheses" in hypothesis_generation_spec.json_schema["properties"]


class TestHypothesisGenerationTrigger:
    def test_tracker_initialized_and_usable(self):
        """HypothesisTracker should be initializable and hold hypotheses."""
        tracker = HypothesisTracker()
        assert len(tracker) == 0
        tracker.add(Hypothesis(claim="test", prediction="test pred", source_node_id="n1"))
        assert len(tracker) == 1
