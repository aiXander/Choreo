#!/usr/bin/env python3
"""
Modal deployment script for the Choreo matching system.
Deploys the pipeline as a serverless function on Modal.

DEPLOYMENT COMMANDS:
# Deploy
modal deploy deploy_modal.py

# Run matching pipeline
modal run deploy_modal.py::run_matching_pipeline --user-profiles-json=profiles.json --config-path=config/config.yaml

# Monitor deployments
modal app list
modal app logs choreo-matching

# Stop deployment
modal app stop choreo-matching
"""

import modal
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
import tempfile
import hashlib
import io
import mimetypes

# Modal app setup
root_dir = Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "numpy>=2.3.3",
        "scikit-learn>=1.7.2",
        "matplotlib>=3.10.6",
        "seaborn>=0.13.0",
        "litellm>=1.76.2",
        "pyyaml>=6.0.2",
        "tqdm>=4.66.2",
        "python-dotenv>=1.0.1",
        "boto3>=1.34.0"
    ])
    .add_local_dir("src", "/app/src", copy=True)
    .add_local_dir("config", "/app/config", copy=True)
    .add_local_file("main.py", "/app/main.py", copy=True)
    .add_local_python_source("src", copy=True)
    .workdir("/app")
    .env({"PYTHONPATH": "/app:/app/src"})
)

app = modal.App(
    name="profile-matching",
    secrets=[modal.Secret.from_name("eve-secrets-PROD")]
)


def upload_file_to_s3(s3_client, file_path: str, bucket_name: str, name: Optional[str] = None) -> tuple[str, str]:
    """
    Minimal S3 upload function - uploads a file and returns (file_url, sha_hash).

    Args:
        s3_client: Boto3 S3 client
        file_path: Path to file to upload
        bucket_name: S3 bucket name
        name: Optional custom name (otherwise uses SHA256 hash)

    Returns:
        Tuple of (file_url, sha_hash)
    """
    # Read file
    with open(file_path, "rb") as f:
        buffer = f.read()

    # Generate SHA256 hash
    hasher = hashlib.sha256()
    hasher.update(buffer)
    sha_hash = hasher.hexdigest()

    # Use provided name or SHA256 hash
    if not name:
        name = sha_hash

    # Detect file extension
    file_extension = Path(file_path).suffix
    if not file_extension:
        # Try to guess from mimetype
        mime_type, _ = mimetypes.guess_type(file_path)
        file_extension = mimetypes.guess_extension(mime_type) if mime_type else ""

    filename = f"{name}{file_extension}"

    # Detect content type
    content_type, _ = mimetypes.guess_type(filename)
    if not content_type:
        content_type = "application/octet-stream"

    # Check if file already exists
    try:
        s3_client.head_object(Bucket=bucket_name, Key=filename)
        # File exists, return URL
        file_url = f"https://{bucket_name}.s3.amazonaws.com/{filename}"
        return file_url, sha_hash
    except s3_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "404":
            raise e

    # Upload file
    file_bytes = io.BytesIO(buffer)
    s3_client.upload_fileobj(
        file_bytes,
        bucket_name,
        filename,
        ExtraArgs={
            "ContentType": content_type,
            "ContentDisposition": "inline"
        }
    )

    file_url = f"https://{bucket_name}.s3.amazonaws.com/{filename}"
    return file_url, sha_hash

# Create Modal volume for persistent storage
volume = modal.Volume.from_name("data_01", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/app/data": volume},
    cpu=2.0,
    max_containers=1,
    timeout=180,
    min_containers=0
)
def run_matching_pipeline(
    user_profiles_json: str,
    config_path: str = "config/config.yaml",
    group_name: Optional[str] = None,
    force: bool = False
) -> Dict[str, Any]:
    """
    Modal function to run the matching pipeline.

    Args:
        user_profiles_json: JSON string containing user profiles dict
        config_path: Path to config YAML file
        group_name: Optional group name for organization
        force: Whether to force re-run all steps

    Returns:
        Dict containing results, matches, and metadata
    """
    import yaml
    import boto3
    from main import run_matching_pipeline

    # Initialize S3 client
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION_NAME"),
    )
    bucket_name = os.getenv("AWS_BUCKET_NAME")

    # Parse user profiles from JSON string
    user_profiles = json.loads(user_profiles_json)

    # Load config file
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

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