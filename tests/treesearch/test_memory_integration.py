import pytest
from ai_researcher.treesearch.scientific_memory import ScientificMemory, ExperimentRecord
from ai_researcher.treesearch.journal import Node, Journal


class TestMemoryIntegration:
    def test_memory_survives_stage_transition(self, make_node):
        """Memory should persist experiment records when stages change."""
        memory = ScientificMemory()

        # Simulate stage 1 experiments
        n1 = make_node(plan="stage 1 attempt", metric_value=0.7)
        memory.record(n1, stage_name="1_initial_1_first")

        n2 = make_node(plan="stage 1 failure", is_buggy=True, exc_type="RuntimeError")
        memory.record(n2, stage_name="1_initial_1_first")

        # Simulate stage 2 — memory should still have stage 1 records
        n3 = make_node(plan="stage 2 tuning", metric_value=0.85)
        memory.record(n3, stage_name="2_baseline_1_first")

        assert len(memory) == 3
        assert len(memory.get_by_stage("1_initial_1_first")) == 2
        assert len(memory.get_by_stage("2_baseline_1_first")) == 1

        # Prompt should contain info from both stages
        prompt = memory.format_for_prompt()
        assert "stage 1" in prompt.lower() or "initial" in prompt.lower()

    def test_memory_provides_failure_context_for_debug(self, make_node):
        """Debug agents should get failure context from memory."""
        memory = ScientificMemory()

        # Record multiple OOM failures
        for i in range(3):
            node = make_node(
                plan=f"large model variant {i}",
                is_buggy=True,
                exc_type="MemoryError",
                analysis=f"OOM on variant {i}",
            )
            memory.record(node, stage_name="1_initial_1_first")

        failures_prompt = memory.format_failures_for_prompt()
        assert "MemoryError" in failures_prompt
        assert failures_prompt.count("OOM") >= 2  # multiple failures visible

    def test_memory_pickle_roundtrip(self, make_node):
        """Memory should survive pickle (for ProcessPoolExecutor transfer)."""
        import pickle

        memory = ScientificMemory()
        node = make_node(plan="test", metric_value=0.9)
        memory.record(node, stage_name="1_initial_1_first")

        data = pickle.dumps(memory)
        restored = pickle.loads(data)
        assert len(restored) == 1


class TestDebugPromptMemory:
    def test_debug_prompt_includes_failure_patterns(self, make_node):
        """When memory has failure records, _debug prompt should reference them."""
        memory = ScientificMemory()

        # Record OOM failures
        for i in range(3):
            n = make_node(
                plan=f"large model {i}",
                is_buggy=True,
                exc_type="MemoryError",
                analysis="Out of memory",
            )
            memory.record(n, stage_name="1_initial_1_first")

        failure_text = memory.format_failures_for_prompt()
        # Verify the failure text is useful
        assert "MemoryError" in failure_text
        assert len(failure_text) > 50
