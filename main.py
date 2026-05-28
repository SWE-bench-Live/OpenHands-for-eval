from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from benchmarks.swebench.config import INFER_DEFAULTS
from benchmarks.swebench.run_infer import SWEBenchEvaluation
from benchmarks.utils.args_parser import add_prompt_path_argument, get_parser
from benchmarks.utils.critics import create_critic
from benchmarks.utils.evaluation_utils import get_default_on_result_writer
from benchmarks.utils.llm_config import load_llm_config
from benchmarks.utils.models import EvalMetadata
from openhands.sdk import get_logger


logger = get_logger(__name__)


def _load_yaml_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise ValueError(f"Config file {path} does not exist")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping")
    return data


def _agent_defaults(config: dict[str, Any]) -> dict[str, Any]:
    agent = dict(config.get("agent") or {})
    aliases = {
        "workers": "num_workers",
        "max_step": "max_iterations",
        "max_steps": "max_iterations",
    }
    for old, new in aliases.items():
        if old in agent and new not in agent:
            agent[new] = agent.pop(old)

    # Keep the command compact for local SWE-Live JSONL runs.
    agent.setdefault("workspace", "docker")
    agent.setdefault("num_workers", 1)
    agent.setdefault("max_iterations", 100)
    agent.setdefault("output_dir", "logs")
    return agent


def _bootstrap_config_path(argv: list[str] | None) -> str:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in raw_argv or "--help" in raw_argv:
        return "config/default.yaml"

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args(raw_argv)
    return args.config


def _model_slug(model: str) -> str:
    return model.replace("/", "__")


def _write_predictions(output_path: str, preds_path: str) -> None:
    predictions: dict[str, dict[str, str]] = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                instance_id = row["instance_id"]
                patch = (row.get("test_result") or {}).get("git_patch", "")
                predictions[instance_id] = {
                    "instance_id": instance_id,
                    "model_patch": patch,
                }

    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)


def main(argv: list[str] | None = None) -> None:
    config_path = _bootstrap_config_path(argv)
    config = _load_yaml_config(config_path)
    agent_defaults = _agent_defaults(config)

    parser = get_parser(add_llm_config=False)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--run-id", default="debug", help="Run identifier")
    parser.add_argument(
        "--instance-timeout",
        type=int,
        default=4 * 60 * 60,
        help="Maximum seconds per instance before recording a timeout",
    )
    add_prompt_path_argument(
        parser, str(Path(__file__).parent / "benchmarks/swebench/run_infer.py")
    )
    parser.set_defaults(**INFER_DEFAULTS)
    parser.set_defaults(**agent_defaults)
    args = parser.parse_args(argv)

    if not args.dataset:
        raise ValueError("--dataset must be provided")
    if args.n_critic_runs < 1:
        raise ValueError(f"n_critic_runs must be >= 1, got {args.n_critic_runs}")

    llm = load_llm_config(args.config)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    output_dir = os.path.join(args.output_dir, _model_slug(llm.model), args.run_id)
    os.makedirs(output_dir, exist_ok=True)

    critic = create_critic(args)
    enable_condenser = args.enable_condenser
    if args.disable_condenser:
        enable_condenser = False

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=output_dir,
        details={},
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=agent_defaults.get(
            "env_setup_commands", ["export PIP_CACHE_DIR=~/.cache/pip"]
        ),
        n_critic_runs=args.n_critic_runs,
        critic=critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        tool_preset=args.tool_preset,
        enable_delegation=args.enable_delegation,
        agent_type=args.agent_type,
        enable_condenser=enable_condenser,
        condenser_max_size=args.condenser_max_size,
        condenser_keep_first=args.condenser_keep_first,
    )

    evaluator = SWEBenchEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
        instance_timeout=args.instance_timeout,
    )
    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    preds_path = os.path.join(output_dir, "preds.json")
    _write_predictions(evaluator.output_path, preds_path)

    logger.info("Evaluation completed!")
    print(json.dumps({"output_json": evaluator.output_path, "preds_json": preds_path}))


if __name__ == "__main__":
    main()
