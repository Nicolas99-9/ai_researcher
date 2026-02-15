"""
Hypothesis Tracker for AI Scientist v2.

Provides structured hypothesis formalization, tracking, and
falsification experiment design for hypothesis-driven ablation studies.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger("ai-researcher")


@dataclass
class Hypothesis(DataClassJsonMixin):
    """A scientific hypothesis with tracking for falsification."""

    claim: str = ""
    prediction: str = ""  # testable prediction derived from claim
    source_node_id: str = ""  # which experiment generated this hypothesis
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
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
        old_status = self.status
        old_confidence = self.confidence
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
        logger.info(
            f"[Hypothesis] id={self.id} updated: "
            f"status {old_status}→{self.status}, "
            f"confidence {old_confidence:.2f}→{self.confidence:.2f}, "
            f"falsified={falsified}, node_id={node_id}, "
            f"claim='{self.claim[:80]}', "
            f"evidence_result='{result[:120]}'"
        )


@dataclass
class HypothesisTracker(DataClassJsonMixin):
    """Tracks all hypotheses across the experiment lifecycle."""

    hypotheses: list[Hypothesis] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.hypotheses)

    def add(self, hypothesis: Hypothesis):
        self.hypotheses.append(hypothesis)
        logger.info(
            f"[HypothesisTracker] Added hypothesis id={hypothesis.id}: "
            f"claim='{hypothesis.claim[:80]}', "
            f"prediction='{hypothesis.prediction[:80]}', "
            f"source_node={hypothesis.source_node_id}, "
            f"total_hypotheses={len(self.hypotheses)}"
        )

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


# --- Hypothesis generation prompt and FunctionSpec ---

from .backend import FunctionSpec


hypothesis_generation_spec = FunctionSpec(
    name="generate_hypotheses",
    description="Generate testable scientific hypotheses from experimental results",
    json_schema={
        "type": "object",
        "properties": {
            "hypotheses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "A specific, testable claim about why the method works",
                        },
                        "prediction": {
                            "type": "string",
                            "description": "A concrete, falsifiable prediction. Must specify what ablation/change to make and what quantitative outcome is expected.",
                        },
                    },
                    "required": ["claim", "prediction"],
                },
                "maxItems": 5,
                "description": "List of hypotheses to test",
            },
        },
        "required": ["hypotheses"],
    },
)


hypothesis_evidence_spec = FunctionSpec(
    name="evaluate_hypothesis_evidence",
    description=(
        "Evaluate whether an ablation experiment's result supports or falsifies "
        "a hypothesis. IMPORTANT: 'supported' means the prediction came true "
        "(the data matches what was predicted). 'falsified' means the prediction "
        "did NOT come true (the data contradicts what was predicted)."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "prediction_came_true": {
                "type": "boolean",
                "description": (
                    "Did the experimental outcome match the hypothesis prediction? "
                    "True if the prediction was confirmed by the data (e.g., if the "
                    "prediction said 'accuracy will drop by >=10%' and accuracy did "
                    "drop by 10% or more, this is True). False if the prediction "
                    "was NOT confirmed (e.g., accuracy did not drop, or dropped less "
                    "than the predicted threshold)."
                ),
            },
            "confidence": {
                "type": "number",
                "description": (
                    "Confidence in the hypothesis after seeing this evidence "
                    "(0.0 to 1.0). Higher if the prediction clearly came true "
                    "or clearly failed. Lower if results are ambiguous."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "Step-by-step explanation: (1) What did the prediction say "
                    "would happen? (2) What actually happened in the experiment? "
                    "(3) Does the actual outcome match or contradict the prediction?"
                ),
            },
        },
        "required": ["prediction_came_true", "confidence", "reasoning"],
    },
)


def build_hypothesis_generation_prompt(
    research_idea: str,
    best_node_plan: str,
    best_node_analysis: str,
    best_node_code: str,
) -> str:
    """Build the prompt for LLM-based hypothesis generation."""
    return f"""You are a rigorous scientist analyzing experimental results. Your task is to generate specific, falsifiable hypotheses about WHY the current method works.

## Research Context
{research_idea}

## Best Method's Plan
{best_node_plan}

## Best Method's Analysis
{best_node_analysis}

## Best Method's Code (for reference)
```python
{best_node_code[:3000]}
```

## Instructions
Generate hypotheses that:
1. Make a specific CLAIM about which component or design choice is responsible for the method's performance.
2. Include a concrete PREDICTION that can be tested via an ablation experiment — specify exactly what to remove/change and what quantitative effect to expect.
3. Are falsifiable — it must be possible for the ablation to show the prediction is wrong.
4. Cover different aspects of the method (architecture, training procedure, data handling, etc.).

Do NOT generate vague hypotheses like "the model works because of good hyperparameters." Be specific: "The multi-head attention with 4 heads is responsible for >60% of the accuracy gain over the MLP baseline, because it captures pairwise feature interactions that a single linear layer cannot."
"""


def build_ablation_prompt_from_hypothesis(
    hypothesis: Hypothesis,
    base_code: str,
    previous_ablations: list[str],
) -> str:
    """Build a targeted ablation prompt from a specific hypothesis."""
    prev_str = ", ".join(previous_ablations) if previous_ablations else "None yet"
    return f"""You are an AI researcher conducting a TARGETED ablation study to test a specific hypothesis.

## Hypothesis to Test
**Claim:** {hypothesis.claim}
**Prediction:** {hypothesis.prediction}

## Your Task
Design and implement an ablation experiment that directly tests this hypothesis. Specifically:
1. Modify the base code to remove or disable the component identified in the claim.
2. Keep everything else identical to isolate the effect.
3. The experiment should either SUPPORT or FALSIFY the prediction.

## Base Code
```python
{base_code[:4000]}
```

## Previously Completed Ablations (do not repeat)
{prev_str}

## Important
- Change ONLY what the hypothesis targets. Do not change hyperparameters, training epochs, data splits, or anything else.
- The goal is to isolate the effect of the specific component mentioned in the claim.
- Print clear metrics that can be compared against the prediction.
"""
