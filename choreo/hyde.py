"""HyDE (Hypothetical Document Embedding) descriptor generation.

Bridges vocabulary gap between source sections (e.g., needs) and target sections
(e.g., skills) by generating hypothetical target-vocabulary descriptions.

Split into a pure transform (``hyde_descriptors_for_sections`` — no disk IO,
reuse passed in via ``existing``) and a filesystem wrapper
(``generate_hyde_descriptors`` — preserves the historical per-cross-key
``hyde/<cross_key>.jsonl`` cache; pass ``cache_dir=None`` to run fully
in-memory, e.g. for a transient query atom).
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional

from .utils import save_jsonl, load_jsonl, ensure_dir, hash_text, parse_cross_key
from .llm import LLMWrapper
from .schemas import ExtractedSections, HydeDescriptors  # noqa: F401 — HydeDescriptors re-exported

__all__ = [
    "HydeDescriptors",
    "hyde_cache_key",
    "hyde_descriptors_for_sections",
    "generate_hyde_descriptors",
]


def hyde_cache_key(source_text: str, n_descriptors: int, cross_key: str) -> str:
    """Content-addressed cache key for one user's HyDE generation."""
    return hash_text(f"{source_text}|{n_descriptors}|{cross_key}")


def hyde_descriptors_for_sections(
    extracted_sections: List[ExtractedSections],
    cross_section_weights: Dict[str, float],
    hyde_config: Dict[str, Any],
    prompt_template: str,
    goal: str,
    llm_wrapper: LLMWrapper,
    model: str,
    sections_config: Optional[Dict[str, Any]] = None,
    existing: Optional[Dict[str, Dict[str, List[str]]]] = None,
    use_llm_cache: bool = True,
) -> Dict[str, List[HydeDescriptors]]:
    """Pure HyDE transform: sections in, descriptors out. No disk IO.

    Args:
        extracted_sections: Sections per user.
        cross_section_weights: The recipe's directional weights; one HyDE set is
            generated per cross key (e.g. ``needs_skills``).
        hyde_config: ``{n_descriptors: int}``.
        prompt_template: The ``hyde_generation`` template string.
        goal: Matching goal injected into the prompt.
        llm_wrapper / model: LLM plumbing.
        sections_config: Optional sections config used for guideline lookups.
        existing: Optional reuse map ``{cross_key: {cache_key: descriptors}}``
            (cache keys from :func:`hyde_cache_key`). Users whose key is present
            skip the LLM call. The caller decides where this came from.
        use_llm_cache: Pass False to bypass the LLM wrapper's response cache.

    Returns:
        Dict mapping cross_key to list of HydeDescriptors (one per user, in the
        same order as ``extracted_sections``).
    """
    if not cross_section_weights:
        return {}

    existing = existing or {}
    n_descriptors = hyde_config.get('n_descriptors', 1)
    result: Dict[str, List[HydeDescriptors]] = {}

    # Build guideline lookup from sections_config
    section_guidelines = {}
    if sections_config:
        for name, sec in sections_config.get("sections", {}).items():
            if isinstance(sec, dict) and "guideline" in sec:
                section_guidelines[name] = sec["guideline"]

    for cross_key in cross_section_weights:
        src_section, tgt_section = parse_cross_key(cross_key)
        print(f"Generating HyDE descriptors for {cross_key} (source={src_section}, target={tgt_section})")

        existing_for_key = existing.get(cross_key, {})

        # Separate cached and uncached
        prompts = []
        cache_keys_for_llm = []
        uncached_indices = []
        all_cache_keys = []

        for idx, es in enumerate(extracted_sections):
            source_text = es.sections.get(src_section, "Not specified")
            cache_key = hyde_cache_key(source_text, n_descriptors, cross_key)
            all_cache_keys.append(cache_key)

            if cache_key in existing_for_key:
                continue  # will use cached

            prompt = prompt_template.format(
                goal=goal,
                source_text=source_text,
                n_descriptors=n_descriptors,
                source_section=src_section,
                target_section=tgt_section,
                source_section_guideline=section_guidelines.get(src_section, "Not specified"),
                target_section_guideline=section_guidelines.get(tgt_section, "Not specified"),
            )
            prompts.append(prompt)
            cache_keys_for_llm.append(f"hyde_{cross_key}_{cache_key}" if use_llm_cache else None)
            uncached_indices.append(idx)

        print(f"  Cached: {len(extracted_sections) - len(uncached_indices)}, to generate: {len(uncached_indices)}")

        # Run LLM calls for uncached
        new_items: Dict[str, List[str]] = {}
        if prompts:
            llm_wrapper.set_component("hyde_generation")

            schema_hint = json.dumps({"descriptors": ["..."]})
            schema_hints = [schema_hint] * len(prompts)

            async def _async_batch():
                try:
                    responses = await llm_wrapper.batch_json_complete(
                        prompts=prompts,
                        model=model,
                        cache_keys=cache_keys_for_llm,
                        schema_hints=schema_hints,
                        batch_size=16,
                    )
                    return responses
                finally:
                    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
                    if tasks:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)

            try:
                responses = asyncio.run(_async_batch())

                for section_idx, response in zip(uncached_indices, responses):
                    try:
                        if isinstance(response, Exception):
                            raise response

                        descriptors = response.get('descriptors', [])
                        if isinstance(descriptors, str):
                            descriptors = [descriptors]
                        # Ensure correct length
                        descriptors = descriptors[:n_descriptors]
                        while len(descriptors) < n_descriptors:
                            descriptors.append(extracted_sections[section_idx].sections.get(src_section, "Not specified"))

                        new_items[all_cache_keys[section_idx]] = descriptors
                    except Exception as e:
                        print(f"  Error processing HyDE for {extracted_sections[section_idx].id}: {e}")
                        fallback = extracted_sections[section_idx].sections.get(src_section, "Not specified")
                        new_items[all_cache_keys[section_idx]] = [fallback] * n_descriptors
            except Exception as e:
                print(f"  Error in batch HyDE generation: {e}")
                # Fallback for all uncached
                for section_idx in uncached_indices:
                    cache_key = all_cache_keys[section_idx]
                    if cache_key not in new_items:
                        fallback = extracted_sections[section_idx].sections.get(src_section, "Not specified")
                        new_items[cache_key] = [fallback] * n_descriptors

        # Merge cached + new into final output
        merged = {**existing_for_key, **new_items}

        user_descriptors = []
        for idx, es in enumerate(extracted_sections):
            descriptors = merged.get(all_cache_keys[idx])
            if descriptors is None:
                # Should not happen, but fallback
                descriptors = [es.sections.get(src_section, "Not specified")] * n_descriptors

            user_descriptors.append(HydeDescriptors(
                user_id=es.id,
                source_section=src_section,
                target_section=tgt_section,
                descriptors=descriptors,
            ))

        result[cross_key] = user_descriptors
        print(f"  Generated HyDE descriptors for {len(user_descriptors)} users")

    return result


def generate_hyde_descriptors(
    extracted_sections: List[ExtractedSections],
    cross_section_weights: Dict[str, float],
    hyde_config: Dict[str, Any],
    prompt_template: str,
    goal: str,
    llm_wrapper: LLMWrapper,
    model: str,
    cache_dir: Optional[Path] = None,
    sections_config: Dict[str, Any] = None,
    force: bool = False,
) -> Dict[str, List[HydeDescriptors]]:
    """Filesystem wrapper around ``hyde_descriptors_for_sections``.

    Preserves the historical ``<cache_dir>/hyde/<cross_key>.jsonl`` content-hash
    cache. Pass ``cache_dir=None`` to skip disk entirely (e.g. transient query
    atoms).

    Returns:
        Dict mapping cross_key (e.g., "needs_skills") to list of HydeDescriptors
        (one per user, in same order as extracted_sections).
    """
    if not cross_section_weights:
        return {}

    n_descriptors = hyde_config.get('n_descriptors', 1)
    hyde_dir = ensure_dir(Path(cache_dir) / "hyde") if cache_dir else None

    # Load existing per-cross-key caches
    existing: Dict[str, Dict[str, List[str]]] = {}
    if hyde_dir and not force:
        for cross_key in cross_section_weights:
            cache_file = hyde_dir / f"{cross_key}.jsonl"
            if not cache_file.exists():
                continue
            try:
                existing[cross_key] = {
                    item['cache_key']: item['descriptors']
                    for item in load_jsonl(cache_file)
                }
                print(f"  Loaded {len(existing[cross_key])} cached HyDE descriptors for {cross_key}")
            except Exception as e:
                print(f"  Warning: Could not load HyDE cache for {cross_key}: {e}")

    result = hyde_descriptors_for_sections(
        extracted_sections=extracted_sections,
        cross_section_weights=cross_section_weights,
        hyde_config=hyde_config,
        prompt_template=prompt_template,
        goal=goal,
        llm_wrapper=llm_wrapper,
        model=model,
        sections_config=sections_config,
        existing=existing,
        use_llm_cache=not force,
    )

    # Persist the merged cache back to disk (existing + freshly generated).
    if hyde_dir:
        for cross_key, user_descriptors in result.items():
            src_section, _ = parse_cross_key(cross_key)
            items = dict(existing.get(cross_key, {}))
            for es, hd in zip(extracted_sections, user_descriptors):
                source_text = es.sections.get(src_section, "Not specified")
                cache_key = hyde_cache_key(source_text, n_descriptors, cross_key)
                items[cache_key] = hd.descriptors
            cache_file = hyde_dir / f"{cross_key}.jsonl"
            save_jsonl(
                [
                    {
                        'cache_key': cache_key,
                        'cross_key': cross_key,
                        'descriptors': descriptors,
                    }
                    for cache_key, descriptors in items.items()
                ],
                cache_file,
            )
            print(f"  Saved {len(items)} HyDE descriptors to cache")

    return result
