"""
Run AI Scientist v2 pipeline and capture detailed logs.
Usage:
    python run_comparison.py --run-name improved   # on feature branch
    python run_comparison.py --run-name vanilla     # on main branch
"""
import argparse
import logging
import os
import sys
import json
from datetime import datetime
from pathlib import Path


def setup_logging(log_file: str):
    """Set up DEBUG-level logging to both file and console."""
    # Remove all existing handlers
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    # File handler — DEBUG level, captures everything
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Console handler — INFO level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    ))

    root.addHandler(fh)
    root.addHandler(ch)
    root.setLevel(logging.DEBUG)

    # Also set our specific logger
    ai_logger = logging.getLogger("ai-researcher")
    ai_logger.setLevel(logging.DEBUG)
    ai_logger.propagate = True

    return ai_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True, help="Name for this run (e.g. 'improved' or 'vanilla')")
    parser.add_argument("--idea-idx", type=int, default=0)
    parser.add_argument("--config", default="bfts_config_fast.yaml")
    args = parser.parse_args()

    os.environ["AI_SCIENTIST_ROOT"] = os.path.dirname(os.path.abspath(__file__))
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY environment variable must be set")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"comparison_runs/{args.run_name}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = str(run_dir / "full_debug.log")
    logger = setup_logging(log_file)
    logger.info(f"Starting {args.run_name} run, logs -> {log_file}")
    logger.info(f"Git branch: {os.popen('git branch --show-current').read().strip()}")
    logger.info(f"Git commit: {os.popen('git log --oneline -1').read().strip()}")

    # Load ideas
    ideas_path = "ai_researcher/ideas/i_cant_believe_its_not_better.json"
    with open(ideas_path) as f:
        ideas = json.load(f)
    idea = ideas[args.idea_idx]
    logger.info(f"Running idea [{args.idea_idx}]: {idea['Name']}")

    # Set up experiment directory
    idea_dir = str(run_dir / f"{idea['Name']}")
    os.makedirs(idea_dir, exist_ok=True)

    # Convert idea to markdown
    from ai_researcher.treesearch.bfts_utils import idea_to_markdown, edit_bfts_config_file

    idea_path_md = os.path.join(idea_dir, "idea.md")
    idea_to_markdown(idea, idea_path_md, None)

    # Load code template
    code_path = ideas_path.rsplit(".", 1)[0] + ".py"
    code = None
    if os.path.exists(code_path):
        with open(code_path) as f:
            code = f.read()
        idea["Code"] = code

    # Store idea json
    idea_path_json = os.path.join(idea_dir, "idea.json")
    with open(idea_path_json, "w") as f:
        json.dump(idea, f, indent=4)

    # Create config for this run
    config_path = edit_bfts_config_file(args.config, idea_dir, idea_path_json)
    logger.info(f"Config written to {config_path}")

    # Override logging AGAIN after config loading (config.py resets it to WARNING)
    # We need to patch the config module's logging setup
    ai_logger = logging.getLogger("ai-researcher")
    ai_logger.setLevel(logging.DEBUG)

    # Run the BFTS pipeline
    logger.info("=" * 60)
    logger.info(f"STARTING BFTS PIPELINE ({args.run_name})")
    logger.info("=" * 60)

    from ai_researcher.treesearch.perform_experiments_bfts_with_agentmanager import perform_experiments_bfts

    # Override logging one more time after all imports
    logging.getLogger("ai-researcher").setLevel(logging.DEBUG)

    try:
        perform_experiments_bfts(config_path)
        logger.info("BFTS pipeline completed successfully")
    except Exception as e:
        logger.error(f"BFTS pipeline failed: {e}", exc_info=True)

    # Save metadata
    meta = {
        "run_name": args.run_name,
        "timestamp": timestamp,
        "idea_idx": args.idea_idx,
        "idea_name": idea["Name"],
        "git_branch": os.popen("git branch --show-current").read().strip(),
        "git_commit": os.popen("git log --oneline -1").read().strip(),
        "log_file": log_file,
        "config": args.config,
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Run complete. Logs at: {log_file}")
    logger.info(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
