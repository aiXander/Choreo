"""HyDE (Hypothetical Document Embedding) descriptor generation.

Bridges vocabulary gap between source sections (e.g., needs) and target sections
(e.g., skills) by generating hypothetical target-vocabulary descriptions.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

from utils import save_jsonl, load_jsonl, ensure_dir, hash_text, parse_cross_key
from llm import LLMWrapper
from extract import ExtractedSections


@dataclass
class HydeDescriptors:
    """HyDE descriptors for a user's section, bridging to target vocabulary."""
    user_id: str
    source_section: str     # e.g., "needs"
    target_section: str     # e.g., "skills"
    descriptors: List[str]  # Always a list, even when n_descriptors=1


def generate_hyde_descriptors(
    extracted_sections: List[ExtractedSections],
    cross_section_weights: Dict[str, float],
    hyde_config: Dict[str, Any],
    prompt_template: str,
    goal: str,
    llm_wrapper: LLMWrapper,
    model: str,
    cache_dir: Path,
    sections_config: Dict[str, Any] = None,
    force: bool = False,
) -> Dict[str, List[HydeDescriptors]]:
    """Generate HyDE descriptors for all source sections in cross_section_weights.

    Returns:
        Dict mapping cross_key (e.g., "needs_skills") to list of HydeDescriptors
        (one per user, in same order as extracted_sections).
    """
    if not cross_section_weights:
        return {}

    n_descriptors = hyde_config.get('n_descriptors', 1)
    hyde_dir = ensure_dir(Path(cache_dir) / "hyde")
    result = {}

    # Build guideline lookup from sections_config
    section_guidelines = {}
    if sections_config:
        for name, sec in sections_config.get("sections", {}).items():
            if isinstance(sec, dict) and "guideline" in sec:
                section_guidelines[name] = sec["guideline"]

    for cross_key, weight in cross_section_weights.items():
        src_section, tgt_section = parse_cross_key(cross_key)
        print(f"Generating HyDE descriptors for {cross_key} (source={src_section}, target={tgt_section})")

        # Load existing cache
        cache_file = hyde_dir / f"{cross_key}.jsonl"
        existing_cache = {}
        if cache_file.exists() and not force:
            try:
                for item in load_jsonl(cache_file):
                    existing_cache[item['cache_key']] = item
                print(f"  Loaded {len(existing_cache)} cached HyDE descriptors")
            except Exception as e:
                print(f"  Warning: Could not load cache: {e}")

        # Separate cached and uncached
        prompts = []
        cache_keys_for_llm = []
        uncached_indices = []
        all_cache_keys = []

        for idx, es in enumerate(extracted_sections):
            source_text = es.sections.get(src_section, "Not specified")
            cache_key = hash_text(f"{source_text}|{n_descriptors}|{cross_key}")
            all_cache_keys.append(cache_key)

            if cache_key in existing_cache:
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
            cache_keys_for_llm.append(None if force else f"hyde_{cross_key}_{cache_key}")
            uncached_indices.append(idx)

        print(f"  Cached: {len(extracted_sections) - len(uncached_indices)}, to generate: {len(uncached_indices)}")

        # Run LLM calls for uncached
        new_cache_items = {}
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

                for resp_idx, (section_idx, response) in enumerate(zip(uncached_indices, responses)):
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

                        cache_key = all_cache_keys[section_idx]
                        new_cache_items[cache_key] = {
                            'cache_key': cache_key,
                            'user_id': extracted_sections[section_idx].id,
                            'cross_key': cross_key,
                            'descriptors': descriptors,
                        }
                    except Exception as e:
                        print(f"  Error processing HyDE for {extracted_sections[section_idx].id}: {e}")
                        cache_key = all_cache_keys[section_idx]
                        fallback = extracted_sections[section_idx].sections.get(src_section, "Not specified")
                        new_cache_items[cache_key] = {
                            'cache_key': cache_key,
                            'user_id': extracted_sections[section_idx].id,
                            'cross_key': cross_key,
                            'descriptors': [fallback] * n_descriptors,
                        }
            except Exception as e:
                print(f"  Error in batch HyDE generation: {e}")
                # Fallback for all uncached
                for section_idx in uncached_indices:
                    cache_key = all_cache_keys[section_idx]
                    if cache_key not in new_cache_items:
                        fallback = extracted_sections[section_idx].sections.get(src_section, "Not specified")
                        new_cache_items[cache_key] = {
                            'cache_key': cache_key,
                            'user_id': extracted_sections[section_idx].id,
                            'cross_key': cross_key,
                            'descriptors': [fallback] * n_descriptors,
                        }

        # Save new cache items
        if new_cache_items:
            all_items = list(existing_cache.values()) + list(new_cache_items.values())
            save_jsonl(all_items, cache_file)
            print(f"  Saved {len(all_items)} HyDE descriptors to cache")

        # Merge cached + new into final output
        merged_cache = {**existing_cache, **new_cache_items}

        user_descriptors = []
        for idx, es in enumerate(extracted_sections):
            cache_key = all_cache_keys[idx]
            if cache_key in merged_cache:
                descriptors = merged_cache[cache_key]['descriptors']
            else:
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
