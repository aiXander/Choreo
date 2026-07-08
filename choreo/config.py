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

Prompts resolve the same way, with one extra (highest-precedence) layer:
**inline prompt text** in the config dict, so a caller whose per-deployment
config lives in a database (no files at request time) can carry custom
prompts in the same overrides dict::

    config = load_config(overrides={
        "prompts": {"scoring_prompt_text": "..."},        # template string
        # "section_prompt_text" takes the full section-config YAML text
        # (or an already-parsed dict).
    })

Typical usage::

    from choreo.config import load_config, resolve_prompt_paths

    config = load_config(config_dir="worlds/my_community", overrides={"query": {"top_k": 3}})
    prompt_paths = resolve_prompt_paths(config_dir="worlds/my_community", config=config)
    run_query_match(query, pool, config, prompt_paths=prompt_paths, ...)
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

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


# Which yaml key inside each prompt file holds the template string.
_PROMPT_TEMPLATE_KEYS = {
    "scoring": "pair_scoring",
    "introduction": "introduction_generation",
    "hyde": "hyde_generation",
}

# Inline-text config key per prompt (under config["prompts"]/["prompt_files"]).
_INLINE_TEXT_KEYS = {
    "sections": "section_prompt_text",
    "scoring": "scoring_prompt_text",
    "introduction": "introduction_prompt_text",
    "hyde": "hyde_prompt_text",
}


def resolve_prompt_templates(
    config_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    prompt_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Resolve the four prompts to their in-memory shapes (no file IO needed
    downstream) — what the mode runners consume.

    Returns ``{"sections": <section-config dict>, "scoring": <template str>,
    "introduction": <template str>, "hyde": <template str>}``.

    Per prompt, precedence (highest first):
      1. **Inline text** in the config dict: ``prompts.<name>_prompt_text``
         (also accepted under ``prompt_files:``). For ``sections`` the value
         may be the section-config YAML *text* or an already-parsed dict; for
         the other three it is the template string itself. This is how an
         external app whose per-tenant config lives in a DB passes fully
         custom prompts without files at request time.
      2. Explicit ``prompt_paths`` entries (e.g. from the CLI's
         ``resolve_prompt_paths(config_dir=…)``).
      3. Path resolution via :func:`resolve_prompt_paths` (config path keys →
         ``config_dir`` files → packaged defaults).

    Cache correctness: scoring/intro/rerank LLM caches key on the full prompt
    and the HyDE cache key folds in a prompt fingerprint, so switching any of
    these templates invalidates the affected cached responses automatically.
    """
    paths = resolve_prompt_paths(config_dir=config_dir, config=config)
    if prompt_paths:
        paths.update({k: v for k, v in prompt_paths.items() if v})

    inline: Dict[str, Any] = {}
    if config:
        for candidate_key in ("prompt_files", "prompts"):
            candidate = config.get(candidate_key)
            if isinstance(candidate, dict):
                inline.update(candidate)

    resolved: Dict[str, Any] = {}
    for name in ("sections", "scoring", "introduction", "hyde"):
        inline_value = inline.get(_INLINE_TEXT_KEYS[name])
        if name == "sections":
            if isinstance(inline_value, dict):
                resolved[name] = inline_value
            elif isinstance(inline_value, str) and inline_value.strip():
                resolved[name] = yaml.safe_load(inline_value)
            else:
                resolved[name] = load_yaml(paths[name])
        else:
            if isinstance(inline_value, str) and inline_value.strip():
                resolved[name] = inline_value
            else:
                resolved[name] = load_yaml(paths[name])[_PROMPT_TEMPLATE_KEYS[name]]
    return resolved
