#!/usr/bin/env python3
"""
User profile matching system using LLM embeddings and processing.
Created by Xander Steenbrugge -- xander@eden.art
"""

print(f"Starting!")

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import modules
from utils import load_yaml
from llm import LLMWrapper
from ingest import load_profiles
from extract import extract_sections_from_profiles
from embed import create_section_embeddings
from candidate import generate_similarity_matrix, CandidatePair
from score import score_pairs_with_llm, create_sections_dict
from match import create_matches
from report import generate_all_reports
from cost_tracker import get_cost_tracker
from visualize_similarity import create_similarity_plots
from tsne import create_tsne_plots

print(f"Loaded all modules, ready to go!")

def main(group_name: str = None, force: bool = False):
    """Main pipeline execution."""
    
    # Load environment variables
    load_dotenv()
    
    print("🚀 Starting prompt-mesh matching pipeline...")
    
    # Load configurations
    config = load_yaml("config/config.yaml")
    
    # Update paths for group-specific data if group_name provided
    if group_name:
        print(f"📁 Using group-specific data: {group_name}")
        base_data_dir = f"data/{group_name}"
        config['io']['raw_dir'] = f"{base_data_dir}/raw"
        config['io']['processed_dir'] = f"{base_data_dir}/processed"
        config['io']['embeds_dir'] = f"{base_data_dir}/embeds"
        config['io']['outputs_dir'] = f"{base_data_dir}/outputs"
        config['io']['cache_dir'] = f"{base_data_dir}/cache"
    sections_config_path = "config/section_prompt.yaml"
    prompts_config_path = "config/scoring_prompt.yaml"
    
    # Initialize LLM wrapper
    llm_wrapper = LLMWrapper(cache_dir=config['io']['cache_dir'])
    
    if force:
        print("🔄 Force flag set - all steps will be re-run, ignoring existing data")
    
    print("\n📁 Step 1: Ingesting profiles...")
    profiles = load_profiles(config['io']['raw_dir'])
    print(f"✅ Loaded {len(profiles)} profiles")
    
    print("\n🧠 Step 2: Extracting sections with LLM...")
    try:
        goal = config['instruction_prompt']['goal']
        extracted_sections = extract_sections_from_profiles(
            profiles=profiles,
            sections_config_path=sections_config_path,
            model=config['models']['extraction_llm'],
            llm_wrapper=llm_wrapper,
            processed_dir=config['io']['processed_dir'],
            budgets=config['budgets'],
            goal=goal,
            force=force
        )
        print(f"✅ Extracted sections for {len(extracted_sections)} profiles")
    except Exception as e:
        print(f"❌ Error extracting sections: {e}")
        return 1
    
    print("\n🔢 Step 3: Creating embeddings...")
    user_ids, section_names, embeddings = create_section_embeddings(
        extracted_sections=extracted_sections,
        embedding_model=config['models']['embedding'],
        embeds_dir=config['io']['embeds_dir'],
        force=force
    )
    print(f"✅ Created embeddings: {embeddings.shape}")
    
    print("\n📊 Step 3.5: Creating t-SNE visualizations...")
    try:
        tsne_results = create_tsne_plots(
            embeddings=embeddings,
            user_ids=user_ids,
            section_names=section_names,
            output_dir=config['io']['outputs_dir'],
            metric='cosine',
            perplexity=7
        )
        print(f"✅ Created t-SNE plots: {tsne_results['plots_dir']}")
    except Exception as e:
        print(f"❌ Error creating t-SNE visualizations: {e}")
        return 1
    
    print("\n🎯 Step 4: Generating similarity matrix...")
    try:
        similarity_matrix, user_ids_sorted, matrices_dict = generate_similarity_matrix(
            embeddings=embeddings,
            user_ids=user_ids,
            section_names=section_names,
            recipe_config=config['recipe']
        )
        print(f"✅ Generated similarity matrix for {len(user_ids_sorted)} users")
    except Exception as e:
        print(f"❌ Error generating similarity matrix: {e}")
        return 1
    
    print("\n⚡ Step 5: LLM pair scoring...")
    try:
        sections_dict = create_sections_dict(extracted_sections)
        
        # Get instruction from config
        instruction = config['recipe'].get('instruction', 'find good matches')
        goal = config['instruction_prompt']['goal']
        
        llm_scores = score_pairs_with_llm(
            similarity_matrix=similarity_matrix,
            user_ids=user_ids_sorted,
            sections_dict=sections_dict,
            instruction=instruction,
            goal=goal,
            prompts_config_path=prompts_config_path,
            llm_wrapper=llm_wrapper,
            model=config['models']['pair_llm'],
            max_n_llm_evaluations_per_profile=config['budgets']['max_n_llm_evaluations_per_profile'],
            global_cap=config['budgets']['max_pair_llm_calls'],
            force=force
        )
        print(f"✅ Scored {len(llm_scores)} pairs with LLM")
    except Exception as e:
        print(f"❌ Error scoring pairs: {e}")
        return 1
    
    print("\n🔗 Step 6: Greedy b-matching...")
    try:
        # Create candidate pairs from LLM scores only (no original candidates without LLM evaluation)
        scored_candidates = [
            CandidatePair.create(score.user1, score.user2, score.embed_score)
            for score in llm_scores.values()
        ]
        
        final_edges = create_matches(
            candidates=scored_candidates,
            llm_scores=llm_scores,
            all_user_ids=user_ids_sorted,
            matching_config=config['matching'],
            blending_config=config['blending']
        )
        print(f"✅ Created {len(final_edges)} final matches")
    except Exception as e:
        print(f"❌ Error creating matches: {e}")
        return 1
    
    print("\n📝 Step 7: Generating reports...")
    try:
        generate_all_reports(
            all_edges=final_edges,
            extracted_sections=extracted_sections,
            outputs_dir=config['io']['outputs_dir'],
            top_matches_per_user=config['matching']['b_max']
        )
        print(f"✅ Generated reports for all users")
    except Exception as e:
        print(f"❌ Error generating reports: {e}")
        return 1
    
    # Print LLM usage statistics
    stats = llm_wrapper.get_stats()
    print(f"\n📊 LLM Usage: {stats['total_calls']} total calls")
    
    # Print comprehensive cost summary
    cost_tracker = get_cost_tracker()
    cost_tracker.print_summary()
    
    # Save detailed cost report
    cost_report_path = Path(config['io']['outputs_dir']) / "cost_report.json"
    cost_tracker.save_detailed_report(str(cost_report_path))
    
    print("\n🎨 Step 8: Creating similarity visualizations...")
    try:
        plots_results = create_similarity_plots(
            matrices_dict=matrices_dict,
            user_ids=user_ids_sorted,
            recipe_config=config['recipe'],
            output_dir=config['io']['outputs_dir'],
            group_name=group_name
        )
        print(f"✅ Created similarity visualizations: {plots_results['plots_dir']}")
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        return 1
    
    print("\n🎉 Pipeline completed successfully!")
    print(f"📁 Check outputs in: {config['io']['outputs_dir']}")
    print(f"📊 Cohort summary: {config['io']['outputs_dir']}/cohort.json")
    print(f"💰 Cost report: {cost_report_path}")
    print(f"📊 t-SNE plots: {tsne_results['plots_dir']}")
    print(f"🎨 Similarity plots: {plots_results['plots_dir']}")
    
    return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prompt-mesh: User profile matching system")
    parser.add_argument("--group", type=str, help="Group name for data organization (e.g., 'group_name_01')")
    parser.add_argument("--force", action="store_true", help="Force re-run all steps, overwriting existing data")
    
    args = parser.parse_args()
    exit_code = main(group_name=args.group, force=args.force)
    sys.exit(exit_code)