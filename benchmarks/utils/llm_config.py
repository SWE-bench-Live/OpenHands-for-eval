from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from openhands.sdk import LLM


def _extract_llm_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return raw

    if isinstance(raw.get("model"), dict):
        config = dict(raw["model"])
    elif isinstance(raw.get("agent"), dict) and isinstance(
        raw["agent"].get("model"), dict
    ):
        config = dict(raw["agent"]["model"])
    else:
        config = dict(raw)

    if "name" in config and "model" not in config:
        config["model"] = config.pop("name")
    if "api_base" in config and "base_url" not in config:
        config["base_url"] = config.pop("api_base")

    for token_field in ("max_input_tokens", "max_output_tokens"):
        if config.get(token_field) == 0:
            config.pop(token_field)

    return config


def _load_raw_config(config_path: Path) -> Any:
    with config_path.open("r", encoding="utf-8") as f:
        raw_text = f.read()

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(raw_text)

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Preserve pydantic's JSON validation errors for existing JSON callers.
        return LLM.model_validate_json(raw_text)


def load_llm_config(config_path: str | Path) -> LLM:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ValueError(f"LLM config file {config_path} does not exist")

    raw_config = _load_raw_config(config_path)
    if isinstance(raw_config, LLM):
        return raw_config

    return LLM.model_validate(_extract_llm_config(raw_config))
