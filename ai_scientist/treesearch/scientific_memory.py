"""
Scientific Memory System for AI Scientist v2.

Provides persistent, structured storage of all experiment outcomes (including
failures) across stages, enabling retrieval-augmented experiment design.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Any

from dataclasses_json import DataClassJsonMixin


@dataclass
class ExperimentRecord(DataClassJsonMixin):
    """A structured record of a single experiment for the Scientific Memory."""

    node_id: str = ""
    plan: str = ""
    stage_name: str = ""
    outcome: str = ""  # "success" | "failure"
    is_buggy: bool = False
    failure_mode: Optional[str] = None  # exc_type string
    analysis: str = ""
    term_out_snippet: str = ""  # first 500 chars of terminal output
    metric_summary: str = ""  # string representation of metric
    ablation_name: Optional[str] = None
    hyperparam_name: Optional[str] = None
    code_hash: str = ""  # sha256 of the code for dedup

    @classmethod
    def from_node(cls, node: Any, stage_name: str = "") -> ExperimentRecord:
        """Create an ExperimentRecord from a Node object."""
        outcome = "failure" if node.is_buggy else "success"
        failure_mode = node.exc_type if hasattr(node, "exc_type") and node.exc_type else None
        term_out = node.term_out if hasattr(node, "term_out") else ""
        term_out_snippet = term_out[:500] if term_out else ""
        metric_summary = str(node.metric) if hasattr(node, "metric") and node.metric else ""
        code = node.code if hasattr(node, "code") else ""
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16] if code else ""

        return cls(
            node_id=node.id,
            plan=node.plan if hasattr(node, "plan") else "",
            stage_name=stage_name,
            outcome=outcome,
            is_buggy=node.is_buggy if hasattr(node, "is_buggy") else False,
            failure_mode=failure_mode,
            analysis=node.analysis if hasattr(node, "analysis") else "",
            term_out_snippet=term_out_snippet,
            metric_summary=metric_summary,
            ablation_name=getattr(node, "ablation_name", None),
            hyperparam_name=getattr(node, "hyperparam_name", None),
            code_hash=code_hash,
        )
