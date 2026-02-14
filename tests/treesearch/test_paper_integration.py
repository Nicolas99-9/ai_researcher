import pytest
from ai_scientist.treesearch.hypothesis_tracker import Hypothesis, HypothesisTracker


class TestPaperIntegration:
    def test_hypothesis_results_formatted_for_paper(self):
        tracker = HypothesisTracker()

        h1 = Hypothesis(claim="Attention is key", prediction="Remove → >8% drop", source_node_id="n1")
        h1.update_with_evidence(result="23% drop confirmed", falsified=False, new_confidence=0.9)

        h2 = Hypothesis(claim="Skip connections essential", prediction="Remove → >10% drop", source_node_id="n2")
        h2.update_with_evidence(result="Only 0.5% drop", falsified=True, new_confidence=0.1)

        tracker.add(h1)
        tracker.add(h2)

        paper_text = tracker.format_for_paper()
        assert "Supported" in paper_text or "supported" in paper_text
        assert "Falsified" in paper_text or "falsified" in paper_text
        assert "Attention" in paper_text
        assert "Skip" in paper_text
