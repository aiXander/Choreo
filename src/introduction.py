"""Introduction and conversation starter generation for matched pairs."""

from typing import List, Dict, Any
from dataclasses import dataclass
import asyncio

from llm import LLMWrapper
from utils import load_yaml
from match import Edge


@dataclass
class Introduction:
    """Generated introduction for a matched pair."""
    pair_id: str
    user1: str
    user2: str
    intro: str
    starter_topics: str


def build_introduction_prompt(
    user1_sections: Dict[str, str],
    user2_sections: Dict[str, str],
    user1_id: str,
    user2_id: str,
    instruction: str,
    prompt_template: str,
    goal: str
) -> str:
    """Build prompt for introduction generation."""
    
    # Format sections nicely
    def format_sections(sections: Dict[str, str], user_id: str) -> str:
        lines = [f"Profile of {user_id}:"]
        for section_name, content in sections.items():
            if content and content.strip() and content != "Not specified":
                lines.append(f"  {section_name.title()}: {content}")
        return "\n".join(lines)
    
    user1_text = format_sections(user1_sections, user1_id)
    user2_text = format_sections(user2_sections, user2_id)
    
    # Build the complete prompt using template with all variables
    prompt = prompt_template.format(
        instruction=instruction,
        goal=goal,
        user_a_name=user1_id,
        user_b_name=user2_id,
        user1_text=user1_text,
        user2_text=user2_text
    )
    
    return prompt


def generate_introductions_for_matches(
    final_edges: List[Edge],
    sections_dict: Dict[str, Dict[str, str]],
    instruction: str,
    goal: str,
    introduction_config_path: str,
    llm_wrapper: LLMWrapper,
    model: str,
    force: bool = False
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
        
    Returns:
        Dictionary mapping pair_id to Introduction
    """
    # Load prompt template
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
            goal=goal
        )
        prompts.append(prompt)
        
        cache_key = None if force else f"intro_{edge.pair_id}_{hash(instruction)}"
        cache_keys.append(cache_key)
        
        valid_edges.append(edge)
    
    if valid_edges:
        # Run batch introduction generation
        llm_wrapper.set_component("introduction_generation")
        
        async def _async_generate_with_cleanup():
            """Run batch introduction generation with proper async cleanup."""
            try:
                responses = await llm_wrapper.batch_json_complete(
                    prompts=prompts,
                    model=model,
                    cache_keys=cache_keys
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
            responses = asyncio.run(_async_generate_with_cleanup())
            
            # Process batch responses
            for edge, response in zip(valid_edges, responses):
                try:
                    if isinstance(response, Exception):
                        raise response
                    
                    # Validate response
                    intro = str(response.get('intro', 'Great to meet you! Looking forward to our conversation.'))
                    starter_topics = str(response.get('starter_topics', '• Share your background • Discuss common interests'))
                    
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
                    # Create fallback introduction
                    introduction = Introduction(
                        pair_id=edge.pair_id,
                        user1=edge.user1,
                        user2=edge.user2,
                        intro=f"Hi {edge.user2}! I'm {edge.user1}. Looking forward to connecting with you.",
                        starter_topics="• Share your background • Discuss common interests • Talk about your goals"
                    )
                    introductions[edge.pair_id] = introduction
                    continue
            
        except Exception as e:
            print(f"Error in batch introduction generation: {e}")
            return {}
    
    print(f"Successfully generated introductions for {len(introductions)} matches")
    return introductions