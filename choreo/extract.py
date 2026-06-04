"""Extract structured sections from user profiles using LLM.

Split into a pure transform (``extract_sections`` — no disk IO, reuse passed in
via ``existing``) and a filesystem wrapper (``extract_sections_from_profiles``
— preserves the historical ``processed/sections.jsonl`` cache behavior).
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from .utils import load_yaml, save_jsonl, load_jsonl, ensure_dir, truncate_words, generate_schema_hint_from_sections, filter_active_sections
from .llm import LLMWrapper, run_coro_blocking
from .ingest import Profile
from .schemas import ExtractedSections, sections_from_dict  # noqa: F401 — re-exported

__all__ = [
    "ExtractedSections",
    "sections_from_dict",
    "build_extraction_prompt",
    "extract_sections",
    "extract_sections_from_profiles",
    "load_extracted_sections",
]


def build_extraction_prompt(profile_text: str, sections_config: Dict[str, Any], goal: str = "") -> str:
    """Build prompt for section extraction."""

    sections_desc = []
    for section_name, config in sections_config['sections'].items():
        guideline = config['guideline']
        max_words = config['max_words']
        sections_desc.append(f'"{section_name}": {guideline} (max {max_words} words)')

    sections_list = '\n'.join(f"  - {desc}" for desc in sections_desc)

    # Get the prompt template from config and format it
    prompt_template = sections_config.get('sections_prompt', '')
    prompt = prompt_template.format(
        goal=goal,
        profile_text=profile_text,
        sections_list=sections_list
    )

    return prompt


def extract_sections(
    profiles: List[Profile],
    sections_config: Dict[str, Any],
    model: str,
    llm_wrapper: LLMWrapper,
    goal: str = "",
    existing: Optional[Dict[str, Dict[str, str]]] = None,
    max_calls: Optional[int] = None,
    use_llm_cache: bool = True,
    failed_out: Optional[List[str]] = None,
) -> List[ExtractedSections]:
    """Pure extraction transform: profiles in, ExtractedSections out. No disk IO.

    Args:
        profiles: Profile objects (id, text, content hash).
        sections_config: Parsed + active-filtered section config (the
            ``sections`` mapping with guidelines/max_words and the prompt
            template).
        model: LLM model name.
        llm_wrapper: LLM wrapper instance (its own response cache is separate
            from the ``existing`` reuse below).
        goal: Matching goal injected into the prompt.
        existing: Optional reuse map ``{profile_hash: sections_dict}`` — any
            profile whose content hash is present is returned from here without
            an LLM call. The caller decides where this came from (FileStore,
            Neon, nothing).
        max_calls: Optional budget cap on the number of LLM extractions.
        use_llm_cache: Pass False to bypass the LLM wrapper's response cache
            (the ``--force`` path).
        failed_out: Optional list that collects the ids of profiles whose
            extraction failed (they are returned with "Not specified" sections
            so the pipeline keeps running, but adapters must NOT persist them —
            persisting would prevent the retry on the next run).

    Returns:
        One ExtractedSections per profile (order: cached first, then fresh —
        same as the historical behavior).
    """
    existing = existing or {}

    extracted_sections: List[ExtractedSections] = []
    uncached_profiles: List[Profile] = []

    for profile in profiles:
        if profile.hash in existing:
            # Hash match == identical content, so the cached sections describe
            # the profile as it is now — stamp them with its timestamp.
            extracted_sections.append(ExtractedSections(
                id=profile.id,
                sections=dict(existing[profile.hash]),
                hash=profile.hash,
                last_updated_at=profile.last_updated_at,
            ))
        else:
            uncached_profiles.append(profile)

    print(f"Using {len(extracted_sections)} cached extractions")

    if max_calls is not None and len(uncached_profiles) > max_calls:
        print(f"Limiting extraction to {max_calls} profiles due to budget")
        uncached_profiles = uncached_profiles[:max_calls]

    if not uncached_profiles:
        return extracted_sections

    # Prepare batch data
    prompts = []
    cache_keys = []
    schema_hints = []

    for profile in uncached_profiles:
        prompts.append(build_extraction_prompt(profile.text, sections_config, goal))
        cache_keys.append(f"extract_{profile.hash}" if use_llm_cache else None)
        schema_hints.append(generate_schema_hint_from_sections(sections_config))

    # Run batch extraction
    llm_wrapper.set_component("profile_extraction")

    try:
        responses = run_coro_blocking(llm_wrapper.batch_json_complete(
            prompts=prompts,
            model=model,
            cache_keys=cache_keys,
            schema_hints=schema_hints,
            batch_size=16,
        ))

        # Process batch responses
        failed_ids = []
        for profile, response in zip(uncached_profiles, responses):
            try:
                if isinstance(response, Exception):
                    raise response

                # Validate and truncate sections
                processed_sections = {}
                for section_name, config in sections_config['sections'].items():
                    raw_content = response.get(section_name, "Not specified")
                    max_words = config['max_words']
                    truncated_content = truncate_words(str(raw_content), max_words)
                    processed_sections[section_name] = truncated_content

                extracted_sections.append(ExtractedSections(
                    id=profile.id,
                    sections=processed_sections,
                    hash=profile.hash,
                    last_updated_at=profile.last_updated_at,
                ))

            except Exception as e:
                print(f"Error processing response for {profile.id}: {e}")
                failed_ids.append(profile.id)
                if failed_out is not None:
                    failed_out.append(profile.id)
                # Add empty sections to avoid breaking pipeline
                empty_sections = {
                    section_name: "Not specified"
                    for section_name in sections_config['sections'].keys()
                }
                extracted_sections.append(ExtractedSections(
                    id=profile.id,
                    sections=empty_sections,
                    hash=profile.hash,
                    last_updated_at=profile.last_updated_at,
                ))

        print(f"Done processing {len(extracted_sections)} LLM extractions")
        if failed_ids:
            print(
                f"⚠️  WARNING: extraction FAILED for {len(failed_ids)} profile(s): "
                f"{', '.join(failed_ids)}. These were filled with empty "
                f"('Not specified') sections and will produce meaningless "
                f"matches — re-run with --force to retry them."
            )

    except Exception as e:
        print(f"Error in batch extraction: {e}")
        # Fallback: add empty sections for all uncached profiles
        for profile in uncached_profiles:
            if failed_out is not None:
                failed_out.append(profile.id)
            empty_sections = {
                section_name: "Not specified"
                for section_name in sections_config['sections'].keys()
            }
            extracted_sections.append(ExtractedSections(
                id=profile.id,
                sections=empty_sections,
                hash=profile.hash,
                last_updated_at=profile.last_updated_at,
            ))

    return extracted_sections


def extract_sections_from_profiles(
    profiles: List[Profile],
    sections_config_path: str,
    model: str,
    llm_wrapper: LLMWrapper,
    processed_dir: str,
    budgets: Dict[str, int],
    goal: str = "",
    force: bool = False
) -> List[ExtractedSections]:
    """Filesystem wrapper around ``extract_sections``.

    Reuses previously processed sections from ``processed/sections.jsonl``
    (keyed by profile content hash) and appends newly extracted ones — the
    historical disk-cache behavior, now expressed as adapter logic around the
    pure transform.
    """
    sections_config = load_yaml(sections_config_path)
    sections_config = filter_active_sections(sections_config)
    processed_path = ensure_dir(processed_dir)
    sections_file = processed_path / "sections.jsonl"

    # Load existing processed sections if available (unless force flag is set)
    existing_by_hash: Dict[str, Dict[str, str]] = {}
    if sections_file.exists() and not force:
        try:
            existing_data = load_jsonl(sections_file)
            for item in existing_data:
                existing_by_hash[item['hash']] = item['sections']
            print(f"Loaded {len(existing_by_hash)} existing processed sections")
        except Exception as e:
            print(f"Warning: Could not load existing sections: {e}")
    elif force and sections_file.exists():
        print("Force flag set - ignoring existing sections")

    failed_ids: List[str] = []
    extracted_sections = extract_sections(
        profiles=profiles,
        sections_config=sections_config,
        model=model,
        llm_wrapper=llm_wrapper,
        goal=goal,
        existing=existing_by_hash,
        max_calls=budgets.get('extraction_llm_calls', 300),  # matches defaults/config.yaml
        use_llm_cache=not force,
        failed_out=failed_ids,
    )

    # Persist newly extracted sections (overwrite if force, append otherwise).
    # Failed extractions are deliberately NOT persisted so they retry next run.
    print("Saving...")
    failed_set = set(failed_ids)
    new_sections_data = [
        es.to_dict() for es in extracted_sections
        if es.hash not in existing_by_hash and es.id not in failed_set
    ]
    if new_sections_data:
        if force or not sections_file.exists():
            # Create new file or overwrite existing
            save_jsonl(new_sections_data, sections_file)
        else:
            # Append new data
            with open(sections_file, 'a') as f:
                for item in new_sections_data:
                    f.write(json.dumps(item) + '\n')

        print(f"Saved {len(new_sections_data)} extracted sections to {sections_file}")

    total_calls_made = len(new_sections_data)
    print(f"Total sections extracted: {len(extracted_sections)} (LLM calls: {total_calls_made})")
    return extracted_sections


def load_extracted_sections(processed_dir: str) -> List[ExtractedSections]:
    """Load previously extracted sections from disk."""
    processed_path = Path(processed_dir)
    sections_file = processed_path / "sections.jsonl"

    if not sections_file.exists():
        return []

    sections_data = load_jsonl(sections_file)
    return [ExtractedSections.from_dict(item) for item in sections_data]
