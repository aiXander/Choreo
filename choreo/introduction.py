"""Introduction and conversation starter generation for matched pairs."""

from typing import List, Dict, Optional

from .llm import LLMWrapper, run_coro_blocking
from .utils import load_yaml, hash_text, is_absent
from .schemas import Edge, Introduction  # noqa: F401 — Introduction re-exported


# The response shapes an introduction call may legitimately return: the
# directional format the packaged prompt asks for, and the legacy single-`intro`
# format a custom `introduction_prompt_text` may still use. Declared to the LLM
# layer so a response matching NEITHER is a retryable failure that is never
# cached — instead of silently becoming placeholder prose on a published card.
INTRODUCTION_KEY_SETS = (
    ("intro_for_a", "intro_for_b", "starter_topics"),
    ("intro", "starter_topics"),
)


def _as_text(value) -> str:
    """Coerce one response field to trimmed text; '' for anything empty.

    Models occasionally answer with a list of bullets where a string was asked
    for, so lists are joined rather than str()'d into Python repr.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(_as_text(item) for item in value if _as_text(item)).strip()
    return str(value).strip()


def fallback_introduction(
    pair_id: str,
    user1: str,
    user2: str,
    user1_name: Optional[str] = None,
    user2_name: Optional[str] = None,
) -> Introduction:
    """Generic fallback when intro generation failed/was skipped for a pair."""
    name1 = user1_name or user1
    name2 = user2_name or user2
    return Introduction(
        pair_id=pair_id,
        user1=user1,
        user2=user2,
        intro=(
            f"For {name1}: You've been matched with {name2} — "
            f"explore how their skills could support your project.\n\n"
            f"For {name2}: You've been matched with {name1} — "
            f"explore how their skills could support your project."
        ),
        starter_topics=(
            "- Share what you're each building\n"
            "- Identify where your skills meet the other's needs\n"
            "- Plan a concrete next step"
        ),
    )


def attach_fallback_intro(edge: Edge, display_names: Optional[Dict[str, str]] = None) -> None:
    """Attach the generic fallback intro text directly onto an edge."""
    display_names = display_names or {}
    fallback = fallback_introduction(
        edge.pair_id, edge.user1, edge.user2,
        user1_name=display_names.get(edge.user1),
        user2_name=display_names.get(edge.user2),
    )
    edge.intro = fallback.intro
    edge.starter_topics = fallback.starter_topics


def build_introduction_prompt(
    user1_sections: Dict[str, str],
    user2_sections: Dict[str, str],
    user1_id: str,
    user2_id: str,
    instruction: str,
    prompt_template: str,
    goal: str,
    user1_name: Optional[str] = None,
    user2_name: Optional[str] = None,
) -> str:
    """Build prompt for introduction generation.

    ``user1_name``/``user2_name`` are the human display names woven into the
    prompt prose (defaults: the ids). Intros never need the model to output
    ids, so with names present the ids don't appear in the prompt at all —
    prose generated from uuids can't be repaired afterwards.
    """
    name1 = user1_name or user1_id
    name2 = user2_name or user2_id

    # Format sections nicely
    def format_sections(sections: Dict[str, str], display_name: str) -> str:
        lines = [f"Profile of {display_name}:"]
        for section_name, content in sections.items():
            if not is_absent(content):
                lines.append(f"  {section_name.title()}: {content}")
        return "\n".join(lines)

    user1_text = format_sections(user1_sections, name1)
    user2_text = format_sections(user2_sections, name2)

    # Build the complete prompt using template with all variables
    prompt = prompt_template.format(
        instruction=instruction,
        goal=goal,
        user_a_name=name1,
        user_b_name=name2,
        user1_text=user1_text,
        user2_text=user2_text
    )

    return prompt


def generate_introductions_for_matches(
    final_edges: List[Edge],
    sections_dict: Dict[str, Dict[str, str]],
    instruction: str,
    goal: str,
    introduction_config_path: str = None,
    llm_wrapper: LLMWrapper = None,
    model: str = None,
    force: bool = False,
    prompt_template: str = None,
    display_names: Optional[Dict[str, str]] = None,
) -> Dict[str, Introduction]:
    """
    Generate introductions and conversation starters for final matched pairs.

    Args:
        final_edges: List of final matched edges
        sections_dict: Dictionary mapping user_id to their sections
        instruction: Instruction for what kind of matching was done
        goal: Goal instruction for matching
        introduction_config_path: Path to introduction_prompt.yaml
        llm_wrapper: LLM wrapper instance
        model: LLM model name
        force: Force re-generation
        prompt_template: The introduction_generation template string itself
            (keeps the transform free of file IO; takes precedence over
            introduction_config_path)
        display_names: Optional {user_id: human name} map — names are used in
            the prompt AND in the assembled intro prose ("For <name>: …"), so
            uuid-keyed adapters get human-readable intros.

    Returns:
        Dictionary mapping pair_id to Introduction
    """
    display_names = display_names or {}
    # Load prompt template
    if prompt_template is None:
        introduction_config = load_yaml(introduction_config_path)
        prompt_template = introduction_config['introduction_generation']
    
    introductions = {}
    
    if not final_edges:
        return introductions
    
    # Prepare batch data
    prompts = []
    cache_keys = []
    valid_edges = []
    
    for edge in final_edges:
        # Get sections for both users
        user1_sections = sections_dict.get(edge.user1, {})
        user2_sections = sections_dict.get(edge.user2, {})
        
        if not user1_sections or not user2_sections:
            print(f"Warning: Missing sections for edge {edge.pair_id}")
            continue
        
        # Build introduction prompt
        prompt = build_introduction_prompt(
            user1_sections=user1_sections,
            user2_sections=user2_sections,
            user1_id=edge.user1,
            user2_id=edge.user2,
            instruction=instruction,
            prompt_template=prompt_template,
            goal=goal,
            user1_name=display_names.get(edge.user1),
            user2_name=display_names.get(edge.user2),
        )
        prompts.append(prompt)
        
        # Cache key = pair_id (readability) + hash of the full prompt, which
        # embeds both profiles' section CONTENT plus instruction/goal/template
        # — so an edited profile invalidates its intros automatically instead
        # of replaying stale prose. hash_text (sha256) — NOT the builtin
        # hash(), which is salted per process and would never hit across runs.
        cache_key = None if force else f"intro_{edge.pair_id}_{hash_text(prompt)}"
        cache_keys.append(cache_key)
        
        valid_edges.append(edge)
    
    if valid_edges:
        # Run batch introduction generation
        llm_wrapper.set_component("introduction_generation")

        try:
            responses = run_coro_blocking(llm_wrapper.batch_json_complete(
                prompts=prompts,
                model=model,
                cache_keys=cache_keys,
                progress_label="introduce",
                required_key_sets=INTRODUCTION_KEY_SETS,
            ))

            # Process batch responses
            for edge, response in zip(valid_edges, responses):
                try:
                    if isinstance(response, Exception):
                        raise response
                    if not isinstance(response, dict):
                        # None = cancelled at a batch deadline; anything else =
                        # the model answered with a non-object.
                        raise ValueError(
                            f"introduction response was {type(response).__name__}, "
                            "not a JSON object"
                        )

                    # Read the response — supports both directional and legacy
                    # formats. NO invented defaults: a missing field must reach
                    # `fallback_introduction` below, which says plainly that the
                    # pair was matched. Substituting cheerful filler here ("Great
                    # to meet you!") shipped two wintercircus cards on 2026-09-02
                    # that looked authored and said nothing — with the real,
                    # correctly generated intro sitting one JSON envelope away.
                    intro_for_a = _as_text(response.get('intro_for_a'))
                    intro_for_b = _as_text(response.get('intro_for_b'))
                    starter_topics = _as_text(response.get('starter_topics'))
                    if intro_for_a and intro_for_b:
                        name1 = display_names.get(edge.user1, edge.user1)
                        name2 = display_names.get(edge.user2, edge.user2)
                        intro = f"For {name1}: {intro_for_a}\n\nFor {name2}: {intro_for_b}"
                    else:
                        intro = _as_text(response.get('intro'))
                    if not intro or not starter_topics:
                        raise ValueError(
                            "introduction response is missing "
                            f"{'intro' if not intro else 'starter_topics'} "
                            f"(keys: {sorted(map(str, response))[:8]})"
                        )

                    # Create Introduction
                    introduction = Introduction(
                        pair_id=edge.pair_id,
                        user1=edge.user1,
                        user2=edge.user2,
                        intro=intro.strip(),
                        starter_topics=starter_topics
                    )
                    
                    introductions[edge.pair_id] = introduction
                    
                except Exception as e:
                    print(f"Error processing introduction response for edge {edge.pair_id}: {e}")
                    introductions[edge.pair_id] = fallback_introduction(
                        edge.pair_id, edge.user1, edge.user2,
                        user1_name=display_names.get(edge.user1),
                        user2_name=display_names.get(edge.user2),
                    )
                    continue
            
        except Exception as e:
            print(f"Error in batch introduction generation: {e}")
            return {}
    
    print(f"Successfully generated introductions for {len(introductions)} matches")
    return introductions