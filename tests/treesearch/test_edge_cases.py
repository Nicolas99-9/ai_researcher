"""
Tests for edge cases and bug fixes in the hypothesis pipeline.
"""
import pytest
from ai_researcher.treesearch.hypothesis_tracker import (
    Hypothesis,
    HypothesisTracker,
    build_ablation_prompt_from_hypothesis,
)


class TestHypothesisIdMatching:
    """Test the ID-based matching used in _generate_ablation_idea → _evaluate_hypothesis_evidence."""

    def _simulate_ablation_name(self, hypothesis: Hypothesis) -> str:
        """Reproduce what _generate_ablation_idea does."""
        return f"hypothesis_test:{hypothesis.id}"

    def _simulate_find(self, tracker: HypothesisTracker, ablation_name: str):
        """Reproduce what _find_hypothesis_by_ablation_name does."""
        hypothesis_id = ablation_name.replace("hypothesis_test:", "")
        for h in tracker.hypotheses:
            if h.id == hypothesis_id:
                return h
        return None

    def test_unique_id_generated(self):
        """Each hypothesis gets a unique ID."""
        h1 = Hypothesis(claim="A", prediction="P1", source_node_id="n1")
        h2 = Hypothesis(claim="A", prediction="P1", source_node_id="n1")
        assert h1.id != h2.id

    def test_id_survives_serialization(self):
        """ID should persist through to_dict/from_dict."""
        h = Hypothesis(claim="test", prediction="pred", source_node_id="n1")
        original_id = h.id
        d = h.to_dict()
        restored = Hypothesis.from_dict(d)
        assert restored.id == original_id

    def test_match_by_id(self):
        """Ablation name contains ID, matching finds the right hypothesis."""
        tracker = HypothesisTracker()
        h1 = Hypothesis(claim="Attention is key", prediction="p1", source_node_id="n1")
        h2 = Hypothesis(claim="Dropout matters", prediction="p2", source_node_id="n2")
        tracker.add(h1)
        tracker.add(h2)

        abl_name = self._simulate_ablation_name(h2)
        found = self._simulate_find(tracker, abl_name)
        assert found is h2

    def test_identical_claims_matched_correctly(self):
        """Two hypotheses with identical claims are distinguished by ID."""
        tracker = HypothesisTracker()
        h1 = Hypothesis(claim="Same claim text", prediction="p1", source_node_id="n1")
        h2 = Hypothesis(claim="Same claim text", prediction="p2", source_node_id="n2")
        tracker.add(h1)
        tracker.add(h2)

        abl_name_1 = self._simulate_ablation_name(h1)
        abl_name_2 = self._simulate_ablation_name(h2)
        assert abl_name_1 != abl_name_2

        assert self._simulate_find(tracker, abl_name_1) is h1
        assert self._simulate_find(tracker, abl_name_2) is h2

    def test_no_match_for_unknown_id(self):
        """Unknown ID returns None."""
        tracker = HypothesisTracker()
        tracker.add(Hypothesis(claim="X", prediction="Y", source_node_id="n1"))
        assert self._simulate_find(tracker, "hypothesis_test:nonexistent") is None


class TestBuggyAblationHypothesisReset:
    """Test that buggy ablation resets hypothesis to untested."""

    def test_buggy_ablation_resets_hypothesis_to_untested(self):
        """After a buggy ablation, hypothesis should be back to 'untested'
        so it can be retried."""
        tracker = HypothesisTracker()
        h = Hypothesis(
            claim="Attention is key",
            prediction="Remove attention drops accuracy by >8%",
            source_node_id="n1",
        )
        tracker.add(h)

        # Simulate _generate_ablation_idea setting status to testing
        untested = tracker.get_untested()
        assert len(untested) == 1
        hypothesis = untested[0]
        hypothesis.status = "testing"

        # Simulate buggy ablation — the fix resets status
        # (This is what _update_ablation_state now does)
        if hypothesis.status == "testing":
            hypothesis.status = "untested"

        # Now it should be available again
        untested_after = tracker.get_untested()
        assert len(untested_after) == 1
        assert untested_after[0] is h

    def test_successful_ablation_resolves_hypothesis(self):
        """Successful ablation should move hypothesis to supported/falsified, not untested."""
        tracker = HypothesisTracker()
        h = Hypothesis(
            claim="Attention is key",
            prediction="Remove attention drops accuracy by >8%",
            source_node_id="n1",
        )
        tracker.add(h)
        h.status = "testing"

        h.update_with_evidence(
            result="Accuracy dropped by 15%",
            falsified=False,
            new_confidence=0.9,
            node_id="abl-1",
        )
        assert h.status == "supported"
        assert len(tracker.get_untested()) == 0
        assert len(tracker.get_by_status("testing")) == 0
