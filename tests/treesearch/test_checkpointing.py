import json
import pytest
from ai_scientist.treesearch.scientific_memory import ScientificMemory
from ai_scientist.treesearch.hypothesis_tracker import Hypothesis, HypothesisTracker


class TestCheckpointing:
    def test_memory_json_roundtrip(self, make_node):
        memory = ScientificMemory()
        n = make_node(plan="test", metric_value=0.9)
        memory.record(n, stage_name="s1")

        data = json.dumps(memory.to_dict())
        restored = ScientificMemory.from_dict(json.loads(data))
        assert len(restored) == 1

    def test_tracker_json_roundtrip(self):
        tracker = HypothesisTracker()
        h = Hypothesis(claim="X works", prediction="Remove X → drop", source_node_id="n1")
        h.update_with_evidence(result="confirmed", falsified=False, new_confidence=0.8)
        tracker.add(h)

        data = json.dumps(tracker.to_dict())
        restored = HypothesisTracker.from_dict(json.loads(data))
        assert len(restored) == 1
        assert restored.hypotheses[0].status == "supported"

    def test_pickle_roundtrip(self, make_node):
        import pickle
        memory = ScientificMemory()
        n = make_node(plan="test", metric_value=0.9)
        memory.record(n, stage_name="s1")

        tracker = HypothesisTracker()
        tracker.add(Hypothesis(claim="test", prediction="pred", source_node_id="n1"))

        mem_bytes = pickle.dumps(memory)
        trk_bytes = pickle.dumps(tracker)
        assert len(pickle.loads(mem_bytes)) == 1
        assert len(pickle.loads(trk_bytes)) == 1
