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
import shutil
import uuid

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


def zip_and_upload_outputs(s3_client, outputs_dir: str, bucket_name: str, group_name: Optional[str] = None) -> Optional[str]:
    """
    Zip the outputs directory and upload it to S3.

    Args:
        s3_client: Boto3 S3 client
        outputs_dir: Path to the outputs directory to zip
        bucket_name: S3 bucket name
        group_name: Optional group name for naming the zip file

    Returns:
        S3 URL of the uploaded zip file, or None if upload fails
    """
    # Create a temporary zip file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
        temp_zip_path = temp_zip.name

    try:
        # Create zip archive of outputs directory
        zip_base_name = temp_zip_path.replace(".zip", "")
        shutil.make_archive(zip_base_name, 'zip', outputs_dir)

        # Generate a name for the zip file
        zip_name = f"outputs_{group_name}" if group_name else "outputs_default"

        # Upload zip to S3
        outputs_zip_url, _ = upload_file_to_s3(
            s3_client=s3_client,
            file_path=temp_zip_path,
            bucket_name=bucket_name
        )

        print(f"✅ Uploaded outputs zip to S3: {outputs_zip_url}")
        return outputs_zip_url

    except Exception as e:
        print(f"⚠️ Warning: Failed to upload outputs zip to S3: {e}")
        return None

    finally:
        # Clean up temporary zip file
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

# Create Modal volume for persistent storage
volume = modal.Volume.from_name("data_01", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/app/data": volume},
    cpu=2.0,
    max_containers=1,
    timeout=3600,
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
        # Generate unique UUID for this run to avoid conflicts
        run_uuid = str(uuid.uuid4())
        if group_name:
            base_data_dir = f"/app/data/{group_name}_{run_uuid}"
        else:
            base_data_dir = f"/app/data/default_{run_uuid}"

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

        try:
            # Run the matching pipeline
            result = run_matching_pipeline(
                user_profiles_dir=profiles_dir,
                config_dict=config_dict,
                force=force,
                group_name=group_name
            )

            print(f"✅ Pipeline result:")
            print(result)

            # Commit volume changes
            volume.commit()

            # Zip outputs directory and upload to S3
            outputs_zip_url = None
            if result.get("success") and result.get("outputs_dir"):
                outputs_zip_url = zip_and_upload_outputs(
                    s3_client=s3_client,
                    outputs_dir=result["outputs_dir"],
                    bucket_name=bucket_name,
                    group_name=group_name
                )

            # Return clean, consistent format
            if result.get("success") and result.get("cohort_summary"):
                # Add outputs_zip_url to cohort_summary
                cohort_summary = result["cohort_summary"]
                cohort_summary["outputs_zip_url"] = outputs_zip_url
                return cohort_summary
            elif result.get("success"):
                return {
                    "error": "Pipeline succeeded but cohort_summary not found in result",
                    "outputs_zip_url": outputs_zip_url
                }
            else:
                return {
                    "error": result.get("error", "Pipeline execution failed")
                }

        finally:
            # Clean up the unique data directory
            if os.path.exists(base_data_dir):
                try:
                    shutil.rmtree(base_data_dir)
                    print(f"✅ Cleaned up temporary directory: {base_data_dir}")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to clean up directory {base_data_dir}: {e}")