"""
Hypothesis Tracker for AI Scientist v2.

Provides structured hypothesis formalization, tracking, and
falsification experiment design for hypothesis-driven ablation studies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from dataclasses_json import DataClassJsonMixin


@dataclass
class Hypothesis(DataClassJsonMixin):
    """A scientific hypothesis with tracking for falsification."""

    claim: str = ""
    prediction: str = ""  # testable prediction derived from claim
    source_node_id: str = ""  # which experiment generated this hypothesis
    status: str = "untested"  # "untested" | "supported" | "falsified" | "inconclusive" | "testing"
    confidence: float = 0.5  # 0.0 to 1.0
    evidence: list[dict] = field(default_factory=list)  # list of evidence dicts
    falsification_node_ids: list[str] = field(default_factory=list)

    def update_with_evidence(
        self,
        result: str,
        falsified: bool,
        new_confidence: float,
        node_id: str = "",
    ):
        """Update hypothesis with new experimental evidence."""
        self.evidence.append({
            "result": result,
            "falsified": falsified,
            "node_id": node_id,
        })
        self.confidence = max(0.0, min(1.0, new_confidence))
        if falsified:
            self.status = "falsified"
        else:
            self.status = "supported"
        if node_id:
            self.falsification_node_ids.append(node_id)


@dataclass
class HypothesisTracker(DataClassJsonMixin):
    """Tracks all hypotheses across the experiment lifecycle."""

    hypotheses: list[Hypothesis] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.hypotheses)

    def add(self, hypothesis: Hypothesis):
        self.hypotheses.append(hypothesis)

    def get_untested(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.status == "untested"]

    def get_by_status(self, status: str) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.status == status]

    def format_for_prompt(self) -> str:
        """Format hypotheses for injection into LLM prompts."""
        if not self.hypotheses:
            return "No hypotheses formulated yet."

        lines = ["## Current Hypotheses"]
        for i, h in enumerate(self.hypotheses, 1):
            lines.append(f"\n### Hypothesis {i} (status: {h.status}, confidence: {h.confidence:.1f})")
            lines.append(f"**Claim:** {h.claim}")
            lines.append(f"**Prediction:** {h.prediction}")
            if h.evidence:
                for ev in h.evidence:
                    lines.append(f"  - Evidence: {ev['result']} (falsified: {ev['falsified']})")
        return "\n".join(lines)

    def format_for_paper(self) -> str:
        """Format hypotheses and evidence for paper writing."""
        if not self.hypotheses:
            return "No hypotheses tested."

        lines = ["## Hypothesis Testing Results"]
        supported = self.get_by_status("supported")
        falsified = self.get_by_status("falsified")

        if supported:
            lines.append("\n### Supported Hypotheses")
            for h in supported:
                lines.append(f"- **{h.claim}** (confidence: {h.confidence:.2f})")
                lines.append(f"  Prediction: {h.prediction}")
                for ev in h.evidence:
                    lines.append(f"  Evidence: {ev['result']}")

        if falsified:
            lines.append("\n### Falsified Hypotheses")
            for h in falsified:
                lines.append(f"- **{h.claim}** (confidence: {h.confidence:.2f})")
                lines.append(f"  Prediction: {h.prediction}")
                for ev in h.evidence:
                    lines.append(f"  Evidence: {ev['result']}")

        return "\n".join(lines)
