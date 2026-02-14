import pytest
from ai_scientist.treesearch.scientific_memory import ExperimentRecord


class TestExperimentRecord:
    def test_create_from_successful_node(self, make_node):
        node = make_node(
            plan="train transformer with lr=0.001",
            code="import torch\n...",
            metric_value=0.85,
            analysis="Model converged well",
        )
        record = ExperimentRecord.from_node(node, stage_name="2_baseline_tuning_1_first_attempt")
        assert record.node_id == node.id
        assert record.plan == node.plan
        assert record.outcome == "success"
        assert record.failure_mode is None
        assert record.stage_name == "2_baseline_tuning_1_first_attempt"
        assert record.is_buggy is False

    def test_create_from_buggy_node_oom(self, make_node):
        node = make_node(
            plan="train huge model",
            is_buggy=True,
            exc_type="MemoryError",
            term_out="CUDA out of memory",
            analysis="Model too large for GPU",
        )
        record = ExperimentRecord.from_node(node, stage_name="1_initial_1_first")
        assert record.outcome == "failure"
        assert record.failure_mode == "MemoryError"
        assert record.is_buggy is True

    def test_create_from_buggy_node_nan(self, make_node):
        node = make_node(
            plan="train with lr=1.0",
            is_buggy=True,
            exc_type="ValueError",
            term_out="NaN detected",
            analysis="Learning rate too high",
        )
        record = ExperimentRecord.from_node(node, stage_name="1_initial_1_first")
        assert record.failure_mode == "ValueError"

    def test_serialization_roundtrip(self, make_node):
        node = make_node(plan="test plan", metric_value=0.9)
        record = ExperimentRecord.from_node(node, stage_name="1_initial_1_first")
        d = record.to_dict()
        restored = ExperimentRecord.from_dict(d)
        assert restored.node_id == record.node_id
        assert restored.plan == record.plan
        assert restored.outcome == record.outcome

    def test_record_from_ablation_node(self, make_node):
        node = make_node(
            plan="Ablation: remove dropout",
            ablation_name="remove_dropout",
            metric_value=0.78,
        )
        record = ExperimentRecord.from_node(node, stage_name="4_ablation_1_first")
        assert record.ablation_name == "remove_dropout"

    def test_record_from_hyperparam_node(self, make_node):
        node = make_node(
            plan="Increase learning rate to 0.01",
            hyperparam_name="learning_rate",
            metric_value=0.82,
        )
        record = ExperimentRecord.from_node(node, stage_name="2_baseline_1_first")
        assert record.hyperparam_name == "learning_rate"
