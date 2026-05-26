"""Extract structured sections from user profiles using LLM."""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

from utils import load_yaml, save_jsonl, load_jsonl, ensure_dir, truncate_words, generate_schema_hint_from_sections, filter_active_sections
from llm import LLMWrapper
from ingest import Profile

@dataclass
class ExtractedSections:
    """Extracted profile sections."""
    id: str
    sections: Dict[str, str]
    hash: str


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
    """
    Extract sections from all profiles using LLM.
    
    Args:
        profiles: List of Profile objects
        sections_config_path: Path to sections.yaml
        model: LLM model name
        llm_wrapper: LLM wrapper instance
        processed_dir: Directory to save processed sections
        budgets: Budget constraints
        
    Returns:
        List of ExtractedSections
    """
    sections_config = load_yaml(sections_config_path)
    sections_config = filter_active_sections(sections_config)
    processed_path = ensure_dir(processed_dir)
    sections_file = processed_path / "sections.jsonl"
    
    # Load existing processed sections if available (unless force flag is set)
    existing_sections = {}
    if sections_file.exists() and not force:
        try:
            existing_data = load_jsonl(sections_file)
            for item in existing_data:
                existing_sections[item['hash']] = item
            print(f"Loaded {len(existing_sections)} existing processed sections")
        except Exception as e:
            print(f"Warning: Could not load existing sections: {e}")
    elif force and sections_file.exists():
        print("Force flag set - ignoring existing sections")
    
    extracted_sections = []
    new_sections_data = []
    
    max_calls = budgets.get('extraction_llm_calls', 1000)
    
    # Separate profiles into cached and uncached
    cached_profiles = []
    uncached_profiles = []
    
    for profile in profiles:
        # Check if we already processed this profile (by hash)
        if profile.hash in existing_sections:
            existing_item = existing_sections[profile.hash]
            sections = ExtractedSections(
                id=profile.id,
                sections=existing_item['sections'],
                hash=profile.hash
            )
            cached_profiles.append(sections)
        else:
            uncached_profiles.append(profile)
    
    # Add cached profiles to results
    extracted_sections.extend(cached_profiles)
    print(f"Using {len(cached_profiles)} cached extractions")
    
    # Limit uncached profiles by budget
    if len(uncached_profiles) > max_calls:
        print(f"Limiting extraction to {max_calls} profiles due to budget")
        uncached_profiles = uncached_profiles[:max_calls]
    
    if uncached_profiles:
        # Prepare batch data
        prompts = []
        cache_keys = []
        schema_hints = []
        
        for profile in uncached_profiles:
            prompt = build_extraction_prompt(profile.text, sections_config, goal)
            prompts.append(prompt)
            
            cache_key = None if force else f"extract_{profile.hash}"
            cache_keys.append(cache_key)
            
            schema_hint = generate_schema_hint_from_sections(sections_config)
            schema_hints.append(schema_hint)
        
        # Run batch extraction
        llm_wrapper.set_component("profile_extraction")
        
        async def _async_batch_with_cleanup():
            """Run batch processing with proper async cleanup."""
            try:
                responses = await llm_wrapper.batch_json_complete(
                    prompts=prompts,
                    model=model,
                    cache_keys=cache_keys,
                    reasoning_effort="low",
                    schema_hints=schema_hints,
                    batch_size=16
                )
                return responses
            finally:
                # Force cleanup of any remaining tasks
                tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
                if tasks:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

        try:
            responses = asyncio.run(_async_batch_with_cleanup())
            
            # Process batch responses
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
                    
                    sections = ExtractedSections(
                        id=profile.id,
                        sections=processed_sections,
                        hash=profile.hash
                    )
                    extracted_sections.append(sections)
                    
                    # Prepare for saving
                    section_data = {
                        'id': profile.id,
                        'sections': processed_sections,
                        'hash': profile.hash
                    }
                    new_sections_data.append(section_data)
                    
                except Exception as e:
                    print(f"Error processing response for {profile.id}: {e}")
                    # Add empty sections to avoid breaking pipeline
                    empty_sections = {
                        section_name: "Not specified" 
                        for section_name in sections_config['sections'].keys()
                    }
                    sections = ExtractedSections(
                        id=profile.id,
                        sections=empty_sections,
                        hash=profile.hash
                    )
                    extracted_sections.append(sections)

            print(f"Done processing {len(extracted_sections)} LLM extractions")
                    
        except Exception as e:
            print(f"Error in batch extraction: {e}")
            # Fallback: add empty sections for all uncached profiles
            for profile in uncached_profiles:
                empty_sections = {
                    section_name: "Not specified" 
                    for section_name in sections_config['sections'].keys()
                }
                sections = ExtractedSections(
                    id=profile.id,
                    sections=empty_sections,
                    hash=profile.hash
                )
                extracted_sections.append(sections)
    
    # Save sections (overwrite if force, append otherwise)
    print("Saving...")
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
    
    extracted_sections = []
    for item in sections_data:
        sections = ExtractedSections(
            id=item['id'],
            sections=item['sections'],
            hash=item['hash']
        )
        extracted_sections.append(sections)
    
    return extracted_sections