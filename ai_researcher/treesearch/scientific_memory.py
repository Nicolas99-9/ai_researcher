"""
Scientific Memory System for AI Scientist v2.

Provides persistent, structured storage of all experiment outcomes (including
failures) across stages, enabling retrieval-augmented experiment design.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional, Any

from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger("ai-researcher")


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


@dataclass
class ScientificMemory(DataClassJsonMixin):
    """Persistent store of all experiment records across stages.

    Provides structured queries for retrieval-augmented experiment design:
    - Query failures by failure mode
    - Query successes by stage
    - Format records as prompt context
    - Dedup by code hash
    """

    records: list[ExperimentRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def record(self, node: Any, stage_name: str = "") -> ExperimentRecord:
        """Record an experiment from a Node. Returns the created record."""
        rec = ExperimentRecord.from_node(node, stage_name=stage_name)
        self.records.append(rec)
        logger.info(
            f"[ScientificMemory] Recorded experiment: node={rec.node_id}, "
            f"stage={rec.stage_name}, outcome={rec.outcome}, "
            f"failure_mode={rec.failure_mode}, code_hash={rec.code_hash}, "
            f"total_records={len(self.records)}"
        )
        return rec

    def get_failures(self, failure_mode: Optional[str] = None) -> list[ExperimentRecord]:
        """Get all failure records, optionally filtered by failure mode."""
        failures = [r for r in self.records if r.is_buggy]
        if failure_mode:
            failures = [r for r in failures if r.failure_mode == failure_mode]
        return failures

    def get_successes(self) -> list[ExperimentRecord]:
        """Get all successful experiment records."""
        return [r for r in self.records if not r.is_buggy]

    def get_by_stage(self, stage_name: str) -> list[ExperimentRecord]:
        """Get all records from a specific stage."""
        return [r for r in self.records if r.stage_name == stage_name]

    def get_ablations(self) -> list[ExperimentRecord]:
        """Get all ablation experiment records."""
        return [r for r in self.records if r.ablation_name is not None]

    def get_unique_approaches(self) -> list[ExperimentRecord]:
        """Deduplicated records by code hash, keeping the latest."""
        seen: dict[str, ExperimentRecord] = {}
        for r in self.records:
            if r.code_hash:
                seen[r.code_hash] = r  # later record overwrites
            else:
                seen[r.node_id] = r
        return list(seen.values())

    def format_for_prompt(self, max_records: int = 10) -> str:
        """Format memory records as a string suitable for LLM prompt injection."""
        if not self.records:
            return "No previous experiments recorded."

        # Prioritize: recent failures first, then recent successes
        failures = self.get_failures()[-max_records // 2:]
        successes = self.get_successes()[-(max_records - len(failures)):]
        selected = failures + successes

        if not selected:
            return "No previous experiments recorded."

        logger.info(
            f"[ScientificMemory] format_for_prompt: {len(failures)} failures, "
            f"{len(successes)} successes, {len(selected)} selected (max={max_records})"
        )
        lines = ["## Experiment Memory (previous attempts and outcomes)"]
        for r in selected[:max_records]:
            lines.append("---")
            lines.append(f"**Stage:** {r.stage_name}")
            lines.append(f"**Plan:** {r.plan[:200]}")
            lines.append(f"**Outcome:** {r.outcome}")
            if r.failure_mode:
                lines.append(f"**Failure Mode:** {r.failure_mode}")
            if r.analysis:
                lines.append(f"**Analysis:** {r.analysis[:200]}")
            if r.metric_summary and r.outcome == "success":
                lines.append(f"**Metric:** {r.metric_summary[:100]}")
        return "\n".join(lines)

    def format_failures_for_prompt(self, max_records: int = 5) -> str:
        """Format only failure records for prompt injection."""
        failures = self.get_failures()[-max_records:]
        if not failures:
            return "No failures recorded."

        lines = ["## Known Failure Modes"]
        for r in failures:
            lines.append("---")
            lines.append(f"**Plan:** {r.plan[:200]}")
            lines.append(f"**Failure:** {r.failure_mode or 'Unknown'}")
            lines.append(f"**Analysis:** {r.analysis[:200]}")
            if r.term_out_snippet:
                lines.append(f"**Output snippet:** {r.term_out_snippet[:150]}")
        return "\n".join(lines)
