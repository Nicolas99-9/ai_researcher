import pytest
from ai_scientist.treesearch.scientific_memory import ExperimentRecord, ScientificMemory


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


class TestScientificMemory:
    @pytest.fixture
    def memory_with_records(self, make_node):
        memory = ScientificMemory()
        # Add successful nodes
        n1 = make_node(plan="train MLP", metric_value=0.7)
        memory.record(n1, stage_name="1_initial_1_first")

        n2 = make_node(plan="train Transformer", metric_value=0.85)
        memory.record(n2, stage_name="1_initial_1_first")

        # Add buggy nodes
        n3 = make_node(
            plan="train huge Transformer",
            is_buggy=True,
            exc_type="MemoryError",
            analysis="Model too large",
        )
        memory.record(n3, stage_name="1_initial_1_first")

        n4 = make_node(
            plan="train MLP with lr=1.0",
            is_buggy=True,
            exc_type="ValueError",
            analysis="NaN in loss due to high lr",
        )
        memory.record(n4, stage_name="2_baseline_1_first")

        # Add ablation node
        n5 = make_node(
            plan="Ablation: remove dropout",
            ablation_name="remove_dropout",
            metric_value=0.78,
        )
        memory.record(n5, stage_name="4_ablation_1_first")

        return memory

    def test_record_and_count(self, memory_with_records):
        assert len(memory_with_records) == 5

    def test_get_failures(self, memory_with_records):
        failures = memory_with_records.get_failures()
        assert len(failures) == 2
        assert all(r.is_buggy for r in failures)

    def test_get_failures_by_type(self, memory_with_records):
        oom_failures = memory_with_records.get_failures(failure_mode="MemoryError")
        assert len(oom_failures) == 1

    def test_get_successes(self, memory_with_records):
        successes = memory_with_records.get_successes()
        assert len(successes) == 3

    def test_get_by_stage(self, memory_with_records):
        stage1 = memory_with_records.get_by_stage("1_initial_1_first")
        assert len(stage1) == 3

    def test_get_ablation_records(self, memory_with_records):
        ablations = memory_with_records.get_ablations()
        assert len(ablations) == 1
        assert ablations[0].ablation_name == "remove_dropout"

    def test_format_for_prompt_returns_string(self, memory_with_records):
        prompt_text = memory_with_records.format_for_prompt(max_records=3)
        assert isinstance(prompt_text, str)
        assert len(prompt_text) > 0

    def test_format_for_prompt_respects_max_records(self, memory_with_records):
        prompt_text = memory_with_records.format_for_prompt(max_records=2)
        # Should contain at most 2 experiment entries
        assert prompt_text.count("---") <= 3  # 2 records + possible header

    def test_format_failures_for_prompt(self, memory_with_records):
        prompt_text = memory_with_records.format_failures_for_prompt()
        assert "MemoryError" in prompt_text or "Model too large" in prompt_text

    def test_serialization_roundtrip(self, memory_with_records):
        data = memory_with_records.to_dict()
        restored = ScientificMemory.from_dict(data)
        assert len(restored) == len(memory_with_records)

    def test_empty_memory_format(self):
        memory = ScientificMemory()
        prompt_text = memory.format_for_prompt()
        assert isinstance(prompt_text, str)

    def test_dedup_by_code_hash(self, make_node):
        memory = ScientificMemory()
        n1 = make_node(plan="plan A", code="same code")
        n2 = make_node(plan="plan B", code="same code")
        memory.record(n1, stage_name="s1")
        memory.record(n2, stage_name="s1")
        # Both should be stored (same code in different contexts is valid)
        assert len(memory) == 2
        # But get_unique_approaches should dedup
        unique = memory.get_unique_approaches()
        assert len(unique) <= 2
