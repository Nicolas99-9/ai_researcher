"""
Integration tests that make real LLM calls to verify the hypothesis pipeline works end-to-end.
Run with: OPENAI_API_KEY=... python -m pytest tests/treesearch/test_integration_real_llm.py -v -s
"""
import os
import pytest

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


class TestRealHypothesisGeneration:
    """Test hypothesis generation with real LLM calls."""

    def test_generate_hypotheses_via_query(self):
        """Call query() with hypothesis_generation_spec — the exact code path
        used by AgentManager._generate_hypotheses_from_stage3()."""
        from ai_researcher.treesearch.backend import query
        from ai_researcher.treesearch.hypothesis_tracker import (
            hypothesis_generation_spec,
            build_hypothesis_generation_prompt,
        )

        prompt = build_hypothesis_generation_prompt(
            research_idea="Study the effect of attention mechanisms on small tabular datasets",
            best_node_plan="Implemented multi-head attention with 4 heads on top of an MLP",
            best_node_analysis="Model achieved 0.92 accuracy, outperforming baseline MLP at 0.78",
            best_node_code="import torch\nimport torch.nn as nn\n\nclass AttentionMLP(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.attn = nn.MultiheadAttention(64, 4)\n        self.fc = nn.Linear(64, 10)\n",
        )

        # This is a string prompt — same as _generate_hypotheses_from_stage3 uses
        print(f"\n--- Prompt type: {type(prompt)} ---")
        print(f"--- Prompt length: {len(prompt)} ---")

        response = query(
            system_message=prompt,
            user_message=None,
            func_spec=hypothesis_generation_spec,
            model="gpt-4o-mini",
            temperature=0.7,
        )

        print(f"\n--- Response: {response} ---")
        print(f"--- Response type: {type(response)} ---")

        # Verify structure
        assert isinstance(response, dict), f"Expected dict, got {type(response)}"
        assert "hypotheses" in response, f"Missing 'hypotheses' key. Keys: {response.keys()}"
        assert isinstance(response["hypotheses"], list), f"hypotheses is not a list"
        assert len(response["hypotheses"]) > 0, "No hypotheses generated"

        for h in response["hypotheses"]:
            assert "claim" in h, f"Missing 'claim' in hypothesis: {h}"
            assert "prediction" in h, f"Missing 'prediction' in hypothesis: {h}"
            print(f"\n  Claim: {h['claim']}")
            print(f"  Prediction: {h['prediction']}")

    def test_evaluate_hypothesis_evidence_via_query(self):
        """Call query() with hypothesis_evidence_spec — the exact code path
        used by ParallelAgent._evaluate_hypothesis_evidence()."""
        from ai_researcher.treesearch.backend import query
        from ai_researcher.treesearch.hypothesis_tracker import hypothesis_evidence_spec

        # String prompt with explicit definitions — same format as _evaluate_hypothesis_evidence uses
        claim = "Multi-head attention with 4 heads is responsible for >60% of accuracy improvement"
        prediction = "Removing multi-head attention drops accuracy by >8%"
        metrics = "accuracy: 0.71 (was 0.92)"
        analysis = "After removing the multi-head attention layer and replacing with a simple linear projection, accuracy dropped from 0.92 to 0.71, a 23% absolute drop."
        term_out = "Epoch 50: train_loss=0.45, val_acc=0.71"

        prompt = (
            "You are evaluating whether an ablation experiment supports or falsifies a scientific hypothesis.\n\n"
            "CRITICAL DEFINITIONS:\n"
            "- SUPPORTED: The prediction CAME TRUE. The experimental data matches what the hypothesis predicted.\n"
            "  Example: Prediction says 'removing X drops accuracy by >=10%'. Accuracy dropped by 25%. "
            "The prediction came true -> SUPPORTED (prediction_came_true = true).\n"
            "- FALSIFIED: The prediction did NOT come true. The experimental data contradicts the prediction.\n"
            "  Example: Prediction says 'removing X drops accuracy by >=10%'. Accuracy only dropped 2%. "
            "The prediction did not come true -> FALSIFIED (prediction_came_true = false).\n\n"
            "IMPORTANT: A large effect that EXCEEDS the predicted threshold SUPPORTS the hypothesis. "
            "Only contradictions (opposite direction or below threshold) FALSIFY it.\n\n"
            f"## Hypothesis\n**Claim:** {claim}\n**Prediction:** {prediction}\n\n"
            f"## Experimental Results\n**Metrics:** {metrics}\n**Analysis:** {analysis}\n**Output:** {term_out}\n\n"
            "Now evaluate: did the prediction come true?"
        )

        print(f"\n--- Prompt type: {type(prompt)} ---")

        response = query(
            system_message=prompt,
            user_message=None,
            func_spec=hypothesis_evidence_spec,
            model="gpt-4o-mini",
            temperature=0.7,
        )

        print(f"\n--- Response: {response} ---")
        print(f"--- Response type: {type(response)} ---")

        assert isinstance(response, dict), f"Expected dict, got {type(response)}"
        assert "prediction_came_true" in response, f"Missing 'prediction_came_true'. Keys: {response.keys()}"
        assert "confidence" in response, f"Missing 'confidence'. Keys: {response.keys()}"
        assert "reasoning" in response, f"Missing 'reasoning'. Keys: {response.keys()}"
        assert isinstance(response["prediction_came_true"], bool), f"prediction_came_true is not bool: {type(response['prediction_came_true'])}"
        assert isinstance(response["confidence"], (int, float)), f"confidence is not numeric: {type(response['confidence'])}"

        print(f"\n  Prediction came true: {response['prediction_came_true']}")
        print(f"  Confidence: {response['confidence']}")
        print(f"  Reasoning: {response['reasoning']}")

        # Given the evidence (23% drop > 8% predicted), prediction_came_true should be True
        # But we just verify the structure — LLM judgment may vary


class TestRealEndToEnd:
    """Test the full hypothesis lifecycle: generate → ablation idea → evaluate."""

    def test_full_hypothesis_lifecycle(self):
        """Simulate the full flow: generate hypotheses, create ablation, evaluate results."""
        from ai_researcher.treesearch.backend import query
        from ai_researcher.treesearch.hypothesis_tracker import (
            Hypothesis,
            HypothesisTracker,
            hypothesis_generation_spec,
            hypothesis_evidence_spec,
            build_hypothesis_generation_prompt,
            build_ablation_prompt_from_hypothesis,
        )

        # Step 1: Generate hypotheses (like _generate_hypotheses_from_stage3)
        tracker = HypothesisTracker()

        prompt = build_hypothesis_generation_prompt(
            research_idea="Investigate whether learned feature interactions improve prediction on tabular data",
            best_node_plan="Used attention-based feature interaction layer",
            best_node_analysis="Achieved 0.88 AUC vs 0.81 baseline",
            best_node_code="class FeatureInteraction(nn.Module): ...",
        )

        response = query(
            system_message=prompt,
            user_message=None,
            func_spec=hypothesis_generation_spec,
            model="gpt-4o-mini",
            temperature=0.7,
        )

        print(f"\n--- Step 1: Generated {len(response.get('hypotheses', []))} hypotheses ---")

        for h_data in response.get("hypotheses", []):
            tracker.add(Hypothesis(
                claim=h_data["claim"],
                prediction=h_data["prediction"],
                source_node_id="test-node-1",
            ))

        assert len(tracker) > 0, "No hypotheses generated"
        print(f"Tracker has {len(tracker)} hypotheses")

        # Step 2: Pick first untested hypothesis for ablation (like _generate_ablation_idea)
        untested = tracker.get_untested()
        assert len(untested) > 0, "No untested hypotheses"
        hypothesis = untested[0]
        hypothesis.status = "testing"

        # Uses hypothesis.id — same as the real _generate_ablation_idea code
        ablation_name = f"hypothesis_test:{hypothesis.id}"
        ablation_desc = build_ablation_prompt_from_hypothesis(
            hypothesis=hypothesis,
            base_code="class FeatureInteraction(nn.Module):\n    def forward(self, x):\n        return self.attention(x)",
            previous_ablations=[],
        )
        print(f"\n--- Step 2: Ablation name: '{ablation_name}' ---")
        print(f"--- Hypothesis ID: '{hypothesis.id}' ---")
        print(f"--- Ablation desc length: {len(ablation_desc)} ---")

        # Step 3: Simulate ablation result and evaluate (like _find_hypothesis_by_ablation_name)
        hypothesis_id = ablation_name.replace("hypothesis_test:", "")
        matched_h = None
        for h in tracker.hypotheses:
            if h.id == hypothesis_id:
                matched_h = h
                break
        print(f"\n--- Step 3: Matched by ID: {matched_h is not None} ---")
        print(f"  hypothesis_id from name: '{hypothesis_id}'")
        print(f"  hypothesis.id: '{hypothesis.id}'")

        assert matched_h is not None, (
            f"HYPOTHESIS MATCHING FAILED!\n"
            f"  ablation_name: '{ablation_name}'\n"
            f"  hypothesis_id: '{hypothesis_id}'\n"
            f"  tracker IDs: {[h.id for h in tracker.hypotheses]}"
        )

        eval_prompt = (
            "You are evaluating whether an ablation experiment supports or falsifies a scientific hypothesis.\n\n"
            "CRITICAL DEFINITIONS:\n"
            "- SUPPORTED: The prediction CAME TRUE. The experimental data matches what the hypothesis predicted.\n"
            "- FALSIFIED: The prediction did NOT come true. The experimental data contradicts the prediction.\n\n"
            f"## Hypothesis\n**Claim:** {matched_h.claim}\n**Prediction:** {matched_h.prediction}\n\n"
            f"## Experimental Results\n"
            f"**Metrics:** AUC: 0.83\n"
            f"**Analysis:** After removing the feature interaction layer, AUC dropped from 0.88 to 0.83\n"
            f"**Output:** Test AUC: 0.83\n\n"
            "Now evaluate: did the prediction come true?"
        )

        eval_response = query(
            system_message=eval_prompt,
            user_message=None,
            func_spec=hypothesis_evidence_spec,
            model="gpt-4o-mini",
            temperature=0.7,
        )

        print(f"\n--- Step 3 result: {eval_response} ---")

        # Map prediction_came_true to falsified for update_with_evidence
        prediction_came_true = eval_response.get("prediction_came_true", True)
        matched_h.update_with_evidence(
            result=eval_response.get("reasoning", ""),
            falsified=not prediction_came_true,
            new_confidence=eval_response.get("confidence", 0.5),
            node_id="ablation-test-node",
        )

        print(f"\n--- Final hypothesis status: {matched_h.status} ---")
        print(f"--- Final confidence: {matched_h.confidence} ---")
        assert matched_h.status in ("supported", "falsified")

        # Step 4: Verify paper formatting works
        paper_text = tracker.format_for_paper()
        print(f"\n--- Paper text ---\n{paper_text}")
        assert len(paper_text) > 50


class TestQueryEdgeCases:
    """Test edge cases in how we call query()."""

    def test_query_with_none_user_message(self):
        """Both _generate_hypotheses_from_stage3 and _evaluate_hypothesis_evidence
        pass user_message=None. Verify this works."""
        from ai_researcher.treesearch.backend import query
        from ai_researcher.treesearch.hypothesis_tracker import hypothesis_evidence_spec

        response = query(
            system_message="Evaluate: claim='X is important', prediction='remove X drops metric by 10%', evidence='metric dropped by 15%'",
            user_message=None,
            func_spec=hypothesis_evidence_spec,
            model="gpt-4o-mini",
            temperature=0.7,
        )
        assert isinstance(response, dict)
        assert "prediction_came_true" in response

    def test_query_with_empty_strings_in_prompt(self):
        """_evaluate_hypothesis_evidence may have empty analysis/term_out.
        Verify prompt with empty values works."""
        from ai_researcher.treesearch.backend import query
        from ai_researcher.treesearch.hypothesis_tracker import hypothesis_evidence_spec

        prompt = (
            "You are evaluating whether an ablation experiment supports or falsifies a scientific hypothesis.\n\n"
            "CRITICAL DEFINITIONS:\n"
            "- SUPPORTED: The prediction CAME TRUE.\n"
            "- FALSIFIED: The prediction did NOT come true.\n\n"
            "## Hypothesis\n**Claim:** Test claim\n**Prediction:** Test prediction\n\n"
            "## Experimental Results\n**Metrics:** N/A\n**Analysis:** \n**Output:** \n\n"
            "Now evaluate: did the prediction come true?"
        )

        response = query(
            system_message=prompt,
            user_message=None,
            func_spec=hypothesis_evidence_spec,
            model="gpt-4o-mini",
            temperature=0.7,
        )
        assert isinstance(response, dict)
        assert "prediction_came_true" in response
