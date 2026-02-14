# AI-Researcher: Hypothesis-Driven Autonomous Scientific Discovery

> Built on top of [The AI Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) by Sakana AI.

AI-Researcher extends the AI Scientist-v2 pipeline with **hypothesis-driven science** capabilities. Where AI Scientist-v2 performs exploratory tree search over experiment code, AI-Researcher adds structured scientific reasoning: it remembers past experiments, generates falsifiable hypotheses from results, designs targeted ablation studies to test them, and evaluates evidence to update hypothesis status.

## What's New

AI-Researcher introduces two core modules and integrates them throughout the 4-stage BFTS pipeline:

### ScientificMemory

A persistent, structured store of all experiment outcomes (successes and failures) across stages. Every node evaluated by the pipeline is recorded with its plan, outcome, failure mode, metrics, terminal output snippet, and a code hash for deduplication.

- **Balanced prompt injection**: When generating new experiments, the LLM receives context from up to 10 prior experiments (5 failures + 5 successes), giving it awareness of what has been tried and what failed.
- **Failure-aware debugging**: Failure records are available for injection into debug prompts, helping the LLM avoid repeating known failure modes.
- **Code deduplication**: Records are keyed by SHA-256 code hash to identify duplicate approaches.

Key file: [`ai_researcher/treesearch/scientific_memory.py`](ai_researcher/treesearch/scientific_memory.py)

### HypothesisTracker

A structured system for managing scientific hypotheses through their full lifecycle: generation, testing, evaluation, and falsification/support.

- **LLM-based hypothesis generation**: After Stage 3 (creative research), the system uses the best-performing experiment to generate up to 5 specific, falsifiable hypotheses about *why* the method works.
- **Targeted ablation design**: In Stage 4, ablation experiments are designed to directly test hypotheses rather than performing generic ablations.
- **Evidence evaluation**: After each ablation completes, an LLM evaluates whether the evidence supports or falsifies the hypothesis, updating its confidence score.
- **Paper integration**: Hypothesis testing results are formatted for inclusion in the generated paper via `format_for_paper()`.

Key file: [`ai_researcher/treesearch/hypothesis_tracker.py`](ai_researcher/treesearch/hypothesis_tracker.py)

### Pipeline Integration

Both modules are woven into the existing BFTS pipeline:

| Component | File | What Changed |
|-----------|------|-------------|
| Instantiation | `agent_manager.py` | ScientificMemory and HypothesisTracker created at pipeline start |
| Experiment recording | `parallel_agent.py` | Every evaluated node is recorded to ScientificMemory |
| Prompt injection | `parallel_agent.py` | Memory context injected into experiment generation prompts |
| Hypothesis generation | `parallel_agent.py` | After Stage 3, hypotheses generated from best node |
| Ablation design | `parallel_agent.py` | Stage 4 ablations target specific hypotheses |
| Evidence evaluation | `parallel_agent.py` | Ablation results evaluated against hypothesis predictions |
| Buggy ablation recovery | `parallel_agent.py` | Failed hypothesis tests reset hypothesis to `untested` for retry |
| Checkpointing | `agent_manager.py` | Memory and hypotheses saved to pickle at each checkpoint |
| Summary output | `perform_experiments_bfts_with_agentmanager.py` | `hypothesis_summary.json` written after pipeline completion |

### New Files

```
ai_researcher/treesearch/scientific_memory.py   # ScientificMemory and ExperimentRecord
ai_researcher/treesearch/hypothesis_tracker.py   # HypothesisTracker, Hypothesis, prompt builders
tests/treesearch/test_scientific_memory.py      # Unit tests for ScientificMemory
tests/treesearch/test_hypothesis_tracker.py     # Unit tests for HypothesisTracker
tests/treesearch/test_hypothesis_ablation.py    # Ablation workflow tests
tests/treesearch/test_memory_integration.py     # Integration tests
tests/treesearch/test_checkpointing.py          # Checkpoint serialization tests
tests/treesearch/test_edge_cases.py             # Edge case tests
tests/treesearch/test_integration_real_llm.py   # Real LLM integration tests
tests/treesearch/test_paper_integration.py      # Paper formatting tests
findings.md                                      # Detailed comparison results
run_comparison.py                                # Comparison runner script
bfts_config_fast.yaml                            # Fast config (reduced iterations)
bfts_config_fullscale.yaml                       # Full-scale config with corrected model IDs
```

## Requirements

This code is designed to run on Linux or macOS with Python 3.11+. GPU support (NVIDIA CUDA) is recommended for ML experiment workloads but not required for the pipeline itself.

### Installation

```bash
# Create a new conda environment
conda create -n ai_researcher python=3.11
conda activate ai_researcher

# Install PyTorch (adjust for your setup)
# With CUDA:
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
# CPU-only (macOS):
conda install pytorch torchvision torchaudio -c pytorch

# Install PDF and LaTeX tools (needed for paper generation)
conda install anaconda::poppler
conda install conda-forge::chktex

# Install Python dependencies
pip install -r requirements.txt
```

### Supported Models and API Keys

#### OpenAI Models

Set the `OPENAI_API_KEY` environment variable. Used for feedback evaluation, paper writing, and optionally for code generation.

#### Claude Models via AWS Bedrock

Install additional packages and configure AWS credentials:

```bash
pip install anthropic[bedrock]
```

Set environment variables:
```bash
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_REGION_NAME="your-region"
```

**Note on model IDs:** When using Bedrock in cross-region inference profiles (e.g., `ap-northeast-1`), use the inference profile prefix:
```
apac.anthropic.claude-3-5-sonnet-20241022-v2:0
```
instead of `anthropic.claude-3-5-sonnet-20241022-v2:0`.

#### Gemini Models

Set the `GEMINI_API_KEY` environment variable for Gemini models through OpenAI API.

#### Semantic Scholar API

Optionally set `S2_API_KEY` for literature search and novelty checking during ideation and paper writing.

```bash
export OPENAI_API_KEY="your-key"
export S2_API_KEY="your-key"        # optional
```

## Generate Research Ideas

Before running the pipeline, generate research ideas using the ideation script:

```bash
python ai_researcher/perform_ideation_temp_free.py \
  --workshop-file "ai_researcher/ideas/my_research_topic.md" \
  --model gpt-4o-2024-05-13 \
  --max-num-generations 20 \
  --num-reflections 5
```

This produces a JSON file (e.g., `ai_researcher/ideas/my_research_topic.json`) containing structured research ideas with hypotheses, proposed experiments, and related work analysis.

See `ai_researcher/ideas/i_cant_believe_its_not_better.md` for the expected format of your topic description file.

## Run AI-Researcher Pipeline

### Configuration

The pipeline is configured via a YAML file. The default is `bfts_config.yaml`. Key parameters:

```yaml
agent:
  type: parallel
  num_workers: 4              # parallel exploration paths
  stages:
    stage1_max_iters: 20      # initial implementation
    stage2_max_iters: 12      # baseline tuning
    stage3_max_iters: 12      # creative research
    stage4_max_iters: 18      # hypothesis-driven ablation
  steps: 5
  k_fold_validation: 1
  multi_seed_eval:
    num_seeds: 3

  code:
    model: apac.anthropic.claude-3-5-sonnet-20241022-v2:0
    temp: 1.0
    max_tokens: 12000

  feedback:
    model: gpt-4o-2024-11-20
    temp: 0.5
    max_tokens: 8192

  search:
    max_debug_depth: 3
    debug_prob: 0.5
    num_drafts: 3
```

A fast configuration (`bfts_config_fast.yaml`) is available for quick testing with reduced iterations (12/6/6/6, 2 workers).

### Launch the Pipeline

```bash
python launch_scientist_bfts.py \
  --load_ideas "ai_researcher/ideas/my_research_topic.json" \
  --load_code \
  --add_dataset_ref \
  --model_writeup o1-preview-2024-09-12 \
  --model_citation gpt-4o-2024-11-20 \
  --model_review gpt-4o-2024-11-20 \
  --model_agg_plots o3-mini-2025-01-31 \
  --num_cite_rounds 20
```

To use a specific config file, modify `bfts_config.yaml` in the project root before launching.

### Pipeline Stages

The BFTS pipeline runs in 4 stages, each building on the previous:

1. **Stage 1 - Initial Implementation**: Multiple independent drafts of the experiment code. ScientificMemory begins recording all outcomes.
2. **Stage 2 - Baseline Tuning**: Iterative improvement of the best Stage 1 result. Memory context helps avoid repeating failed approaches.
3. **Stage 3 - Creative Research**: Exploratory modifications. After this stage, hypotheses are generated from the best-performing node.
4. **Stage 4 - Hypothesis-Driven Ablation**: Each ablation experiment targets a specific hypothesis. Results are evaluated as supporting or falsifying evidence.

### Output Structure

After completion, find results in `experiments/<timestamp_ideaname>/`:

```
logs/0-run/
  unified_tree_viz.html          # interactive tree visualization
  hypothesis_summary.json        # all hypotheses with evidence and status
  research_summary.json          # experiment results summary
  baseline_summary.json          # baseline metrics
  draft_summary.json             # draft-level summary
  experiment_results/            # per-node experiment outputs
```

### Running Comparisons

To compare AI-Researcher against vanilla AI Scientist-v2, use the comparison runner:

```bash
python run_comparison.py
```

This script runs the full pipeline with DEBUG-level logging and saves detailed logs to `comparison_runs/`. See `findings.md` for detailed comparison results from our evaluation.

## Architecture

```
ai_researcher/
  treesearch/
    scientific_memory.py          # ScientificMemory + ExperimentRecord
    hypothesis_tracker.py         # HypothesisTracker + Hypothesis + prompt builders
    parallel_agent.py             # Main agent logic (modified for memory + hypotheses)
    agent_manager.py              # Pipeline orchestrator (instantiation + checkpointing)
    perform_experiments_bfts_with_agentmanager.py  # Entry point (hypothesis summary output)
    backend/                      # LLM backends (Anthropic, OpenAI)
```

### How It Works

1. **Recording**: After each node is evaluated (success or failure), `ScientificMemory.record(node)` creates an `ExperimentRecord` with plan, outcome, metrics, failure mode, and code hash.

2. **Prompt injection**: When generating new experiments, `ScientificMemory.format_for_prompt()` returns a balanced sample of past experiments (failures + successes) for the LLM to learn from.

3. **Hypothesis generation**: After Stage 3, `_generate_hypotheses_from_results()` sends the best node's plan, analysis, and code to an LLM, which returns up to 5 structured hypotheses via function calling (`hypothesis_generation_spec`).

4. **Targeted ablation**: In Stage 4, `_generate_ablation_idea()` checks for untested hypotheses and generates ablation prompts targeting specific hypotheses via `build_ablation_prompt_from_hypothesis()`.

5. **Evidence evaluation**: After an ablation completes, `_evaluate_hypothesis_evidence()` asks an LLM to determine whether the result supports or falsifies the hypothesis (`hypothesis_evidence_spec`). The hypothesis status and confidence are updated accordingly.

6. **Recovery**: If a hypothesis-targeted ablation fails (buggy code), the hypothesis is reset to `untested` so it can be retried.

7. **Checkpointing**: ScientificMemory and HypothesisTracker are serialized to pickle at each checkpoint, surviving pipeline restarts.

## Evaluation Results

Full comparison results are documented in [`findings.md`](findings.md). Key highlights from the full-scale evaluation:

| Metric | AI-Researcher (Improved) | Vanilla AI Scientist-v2 |
|--------|-------------------------|------------------------|
| Total nodes explored | 56 | 67 |
| All 4 stages completed | Yes | Yes |
| Best accuracy | 1.0 (train + test) | CGS ~60.23 |
| Hypotheses generated | 5 | 0 (no hypothesis system) |
| Hypotheses correctly evaluated | 3/5 (60%) | N/A |
| ScientificMemory records | 56 | N/A |
| Memory prompt injections | 13 | N/A |
| Runtime | ~54 min | ~93 min |

### Known Limitations

- **Stage 3 reliability**: Creative research stage has high failure rates in both systems (0-100% buggy code rate)
- **Hypothesis evaluation accuracy**: LLM-based evaluation (gpt-4o-mini) achieves ~56-60% accuracy, sometimes confusing supported/falsified
- **failure_memory dead code**: The `format_failures_for_prompt()` debug injection path is never triggered in practice
- **Different problem domains**: The two runs used different research ideas, making direct metric comparison imperfect

## Citing

If you use AI-Researcher in your work, please cite both this project and the original AI Scientist-v2:

```bibtex
@article{aiscientist_v2,
  title={The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search},
  author={Yamada, Yutaro and Lange, Robert Tjarko and Lu, Cong and Hu, Shengran and Lu, Chris and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2504.08066},
  year={2025}
}
```

## Acknowledgements

- [The AI Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) by Sakana AI - the foundation this project builds upon
- [AIDE](https://github.com/WecoAI/aideml) - tree search component used within the AI Scientist

## License

This project inherits the AI Scientist Source Code License from the original AI Scientist-v2. By using this code, you are legally bound to clearly and prominently disclose the use of AI in any resulting scientific manuscripts or papers.
