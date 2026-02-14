import pytest
from ai_scientist.treesearch.journal import Node, Journal
from ai_scientist.treesearch.utils.metric import MetricValue, WorstMetricValue


@pytest.fixture
def make_node():
    """Factory fixture: create a Node with sensible defaults."""
    def _make(
        plan="test plan",
        code="print('hello')",
        is_buggy=False,
        metric_value=0.8,
        analysis="test analysis",
        exc_type=None,
        exc_info=None,
        term_out="output",
        ablation_name=None,
        hyperparam_name=None,
        parent=None,
    ):
        node = Node(
            plan=plan,
            code=code,
            is_buggy=is_buggy,
            analysis=analysis,
            ablation_name=ablation_name,
            hyperparam_name=hyperparam_name,
        )
        node._term_out = [term_out] if term_out else []
        node.exc_type = exc_type
        node.exc_info = exc_info
        if is_buggy:
            node.metric = WorstMetricValue()
        else:
            node.metric = MetricValue(
                value={
                    "metric_names": [
                        {
                            "metric_name": "accuracy",
                            "lower_is_better": False,
                            "description": "Test accuracy",
                            "data": [
                                {
                                    "dataset_name": "test",
                                    "final_value": metric_value,
                                    "best_value": metric_value,
                                }
                            ],
                        }
                    ]
                },
            )
        if parent:
            node.parent = parent
            parent.children.add(node)
        return node
    return _make


@pytest.fixture
def sample_journal(make_node):
    """A journal with 3 good nodes and 2 buggy nodes."""
    journal = Journal()
    root = make_node(plan="initial draft", code="v1 code", metric_value=0.7)
    journal.append(root)

    improved = make_node(
        plan="improved version", code="v2 code", metric_value=0.85, parent=root
    )
    journal.append(improved)

    buggy1 = make_node(
        plan="buggy attempt OOM",
        code="bad code 1",
        is_buggy=True,
        exc_type="MemoryError",
        exc_info={"error": "OOM on large model"},
        term_out="CUDA out of memory",
        analysis="Model too large for GPU memory",
        parent=root,
    )
    journal.append(buggy1)

    buggy2 = make_node(
        plan="buggy attempt NaN",
        code="bad code 2",
        is_buggy=True,
        exc_type="ValueError",
        exc_info={"error": "NaN in loss"},
        term_out="NaN detected in loss",
        analysis="Learning rate too high causing NaN",
        parent=improved,
    )
    journal.append(buggy2)

    good_leaf = make_node(
        plan="final version", code="v3 code", metric_value=0.92, parent=improved
    )
    journal.append(good_leaf)

    return journal
