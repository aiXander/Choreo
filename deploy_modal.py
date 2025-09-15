#!/usr/bin/env python3
"""
Modal deployment script for the Choreo matching system.
Deploys the pipeline as a serverless function on Modal.
"""

import modal
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Create Modal app
app = modal.App("choreo-matching")

# Define the image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "litellm>=1.76.2",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "tsne>=0.3.1",
        "python-dotenv>=0.19.0",
        "numba>=0.56.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
    ])
    .copy_local_dir("src", "/app/src")
    .copy_local_dir("config", "/app/config")
    .copy_local_file("main.py", "/app/main.py")
    .workdir("/app")
    .env({"PYTHONPATH": "/app:/app/src"})
)

# Create Modal volume for persistent storage
volume = modal.Volume.from_name("choreo-data", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/app/data": volume},
    cpu=4.0,  # CPU-only instance
    concurrency_limit=1,  # Only allow one concurrent execution
    timeout=3600,  # 1 hour timeout
    keep_warm=0,  # Scale down to 0 when not in use
    secrets=[modal.Secret.from_name("openai-secret")]  # Assuming you have OpenAI secrets
)
def run_matching(
    user_profiles: Dict[str, str],
    config_dict: Dict[str, Any],
    sections_config_path: str = "config/section_prompt.yaml",
    prompts_config_path: str = "config/scoring_prompt.yaml",
    introduction_config_path: str = "config/introduction_prompt.yaml",
    force: bool = False,
    group_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Modal function to run the matching pipeline.

    Args:
        user_profiles: Dictionary mapping user_id -> profile_text
        config_dict: Configuration dictionary for the pipeline
        sections_config_path: Path to section extraction config
        prompts_config_path: Path to scoring prompts config
        introduction_config_path: Path to introduction prompts config
        force: Whether to force re-run all steps
        group_name: Optional group name for organization

    Returns:
        Dict containing results, matches, and metadata
    """
    import tempfile
    import shutil
    from main import run_matching_pipeline

    # Create temporary directory for user profiles
    with tempfile.TemporaryDirectory() as temp_dir:
        profiles_dir = os.path.join(temp_dir, "profiles")
        os.makedirs(profiles_dir, exist_ok=True)

        # Write user profiles to temporary files
        for user_id, profile_text in user_profiles.items():
            profile_path = os.path.join(profiles_dir, f"{user_id}.txt")
            with open(profile_path, "w", encoding="utf-8") as f:
                f.write(profile_text)

        # Update config to use persistent volume for outputs
        if group_name:
            base_data_dir = f"/app/data/{group_name}"
        else:
            base_data_dir = "/app/data/default"

        config_dict['io']['processed_dir'] = f"{base_data_dir}/processed"
        config_dict['io']['embeds_dir'] = f"{base_data_dir}/embeds"
        config_dict['io']['outputs_dir'] = f"{base_data_dir}/outputs"
        config_dict['io']['cache_dir'] = f"{base_data_dir}/cache"

        # Ensure output directories exist
        for dir_path in [
            config_dict['io']['processed_dir'],
            config_dict['io']['embeds_dir'],
            config_dict['io']['outputs_dir'],
            config_dict['io']['cache_dir']
        ]:
            os.makedirs(dir_path, exist_ok=True)

        # Run the matching pipeline
        result = run_matching_pipeline(
            user_profiles_dir=profiles_dir,
            config_dict=config_dict,
            sections_config_path=sections_config_path,
            prompts_config_path=prompts_config_path,
            introduction_config_path=introduction_config_path,
            force=force,
            group_name=group_name
        )

        # Serialize matches to be JSON-serializable
        if result.get("success") and result.get("matches"):
            serialized_matches = []
            for match in result["matches"]:
                match_dict = {
                    "user1": match.user1,
                    "user2": match.user2,
                    "score": float(match.score) if hasattr(match, 'score') else 0.0,
                    "intro": getattr(match, 'intro', ""),
                    "starter_topics": getattr(match, 'starter_topics', ""),
                    "pair_id": getattr(match, 'pair_id', f"{match.user1}_{match.user2}")
                }
                serialized_matches.append(match_dict)
            result["matches"] = serialized_matches

        # Commit volume changes
        volume.commit()

        return result


@app.local_entrypoint()
def main(
    profiles_file: str,
    config_file: str = "config/config.yaml",
    group_name: str = None,
    force: bool = False
):
    """
    Local entrypoint for testing the Modal deployment.

    Args:
        profiles_file: Path to JSON file containing user profiles dict
        config_file: Path to config YAML file
        group_name: Optional group name
        force: Force re-run flag
    """
    import json
    import yaml

    # Load user profiles
    with open(profiles_file, "r") as f:
        user_profiles = json.load(f)

    # Load config
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    # Run the matching pipeline
    result = run_matching.remote(
        user_profiles=user_profiles,
        config_dict=config_dict,
        group_name=group_name,
        force=force
    )

    print("✅ Pipeline completed!")
    print(f"Success: {result.get('success', False)}")
    if result.get("success"):
        print(f"Profiles processed: {result.get('profiles_count', 0)}")
        print(f"Matches created: {result.get('stats', {}).get('matches_created', 0)}")
        print(f"LLM calls: {result.get('stats', {}).get('llm_calls', 0)}")
        print(f"Outputs dir: {result.get('outputs_dir', 'N/A')}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Choreo matching to Modal")
    parser.add_argument("profiles_file", help="Path to JSON file with user profiles")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--group", help="Group name for data organization")
    parser.add_argument("--force", action="store_true", help="Force re-run all steps")

    args = parser.parse_args()

    main(
        profiles_file=args.profiles_file,
        config_file=args.config,
        group_name=args.group,
        force=args.force
    )