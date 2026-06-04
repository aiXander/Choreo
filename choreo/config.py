"""Config resolution: packaged defaults ← optional config dir ← runtime overrides.

Choreo ships a complete, working configuration inside the package
(``choreo/defaults/`` — config.yaml + the four prompt yamls). That is the
canonical input schema. Callers customize it in two stacking layers, so
nothing is ever hardcoded per deployment:

  1. ``config_dir`` — a directory holding any subset of the five files.
     A ``config.yaml`` found there is DEEP-MERGED over the packaged default
     (specify only the keys you change); prompt yamls found there REPLACE
     the packaged ones wholesale (prompts don't merge meaningfully).
  2. ``overrides`` — a plain dict deep-merged on top of everything, for
     per-call dynamic values (e.g. an MCP tool call passing
     ``{"query": {"top_k": 3}}`` or ``{"models": {"pair_llm": "..."}}``).

Typical usage::

    from choreo.config import load_config, resolve_prompt_paths

    config = load_config(config_dir="worlds/wintercircus", overrides={"query": {"top_k": 3}})
    prompt_paths = resolve_prompt_paths(config_dir="worlds/wintercircus", config=config)
    run_query_match(query, pool, config, prompt_paths=prompt_paths, ...)
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .utils import DEFAULTS_DIR, DEFAULT_PROMPT_PATHS, PROMPT_FILENAMES, load_yaml

DEFAULT_CONFIG_PATH = DEFAULTS_DIR / "config.yaml"


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a deep copy of ``base``.

    Dicts merge key-by-key; any non-dict value (including lists) replaces the
    base value outright. Neither input is mutated.
    """
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def set_by_path(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``config["a"]["b"]["c"] = value`` from ``"a.b.c"``, creating
    intermediate dicts as needed. Used by the CLI's ``--set`` flag."""
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_config(
    config_dir: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the effective pipeline config (see module docstring for layering)."""
    config = load_yaml(str(DEFAULT_CONFIG_PATH))

    if config_dir is not None:
        candidate = Path(config_dir).expanduser() / "config.yaml"
        if candidate.exists():
            config = deep_merge(config, load_yaml(str(candidate)))

    if overrides:
        config = deep_merge(config, overrides)

    return config


def _first_existing(mapping: Dict[str, Any], *keys: str, default: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if value:
            return value
    return default


def resolve_prompt_paths(
    config_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Resolve the four prompt-file paths.

    Per prompt, precedence is: explicit path in the config dict
    (``prompt_files:``/``prompts:`` with ``<name>_prompt_path``/``<name>_prompt``
    keys) > ``<config_dir>/<name>_prompt.yaml`` if it exists > packaged default.
    """
    base = dict(DEFAULT_PROMPT_PATHS)
    if config_dir is not None:
        directory = Path(config_dir).expanduser()
        for key, fname in PROMPT_FILENAMES.items():
            candidate = directory / fname
            if candidate.exists():
                base[key] = str(candidate)

    prompt_sections: Dict[str, Any] = {}
    if config:
        for candidate_key in ("prompt_files", "prompts"):
            candidate = config.get(candidate_key)
            if isinstance(candidate, dict):
                prompt_sections.update(candidate)

    return {
        "sections": _first_existing(
            prompt_sections, "section_prompt_path", "section_prompt",
            default=base["sections"],
        ),
        "scoring": _first_existing(
            prompt_sections, "scoring_prompt_path", "scoring_prompt",
            default=base["scoring"],
        ),
        "introduction": _first_existing(
            prompt_sections, "introduction_prompt_path", "introduction_prompt",
            default=base["introduction"],
        ),
        "hyde": _first_existing(
            prompt_sections, "hyde_prompt_path", "hyde_prompt",
            default=base["hyde"],
        ),
    }
