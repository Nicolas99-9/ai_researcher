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
