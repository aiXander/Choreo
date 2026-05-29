#!/usr/bin/env python3
"""
Modal deployment for the Choreo matching pipeline.

Runs the directional "who can help whom" matching pipeline (see CLAUDE.md) as a
serverless Modal function. Profiles are passed in as a JSON object
(``{user_id: profile_text}``); outputs (per-user markdown reports, cohort.json,
cost report, plots) are written to a persistent Modal Volume, zipped, and made
retrievable two ways:

  1. Local testing  — the ``local_entrypoint`` below downloads the outputs zip
     from the Volume straight back to the input folder. No AWS needed.
  2. Production      — if AWS_* env vars are present in the attached secret, the
     outputs zip is also uploaded to S3 and its URL returned in the result.

The remote function returns a JSON-serializable dict (the cohort summary plus
metadata), so it stays usable as an API building block.

SECRET
------
The attached secret ``choreo-secrets`` must contain ``OPENROUTER_API_KEY`` (all
LLM + embedding calls route through OpenRouter — see src/llm.py). To also enable
S3 upload, add AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION_NAME /
AWS_BUCKET_NAME to that same secret.

Create it from your local .env with:
    modal secret create choreo-secrets OPENROUTER_API_KEY=sk-or-...

COMMANDS
--------
# Deploy the function (callable via .remote / .spawn from other apps)
modal deploy deploy_modal.py

# Run a matching job from a local folder of .txt profiles (one file per user).
# Outputs are downloaded back to <folder>/outputs_modal.zip.
modal run deploy_modal.py --input-dir data/test4 --force

# Or run against a JSON file of {user_id: profile_text}
modal run deploy_modal.py --profiles-json profiles.json

# Monitor / manage
modal app list
modal app logs choreo-matching
modal volume ls choreo-data
"""

import modal
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import tempfile
import hashlib
import io
import mimetypes
import shutil
import uuid

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
# Dependency set mirrors pyproject.toml (plus boto3 for the optional S3 upload).
# matplotlib runs headless in the container -> force the Agg backend so the
# plotting steps don't try to reach a display.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "numpy>=2.3.3",
            "scikit-learn>=1.7.2",
            "matplotlib>=3.10.6",
            "seaborn>=0.13.0",
            "openai>=1.30.0",
            "pyyaml>=6.0.2",
            "tqdm>=4.66.2",
            "python-dotenv>=1.0.1",
            "boto3>=1.34.0",
        ]
    )
    .add_local_dir("src", "/app/src", copy=True)
    .add_local_dir("config", "/app/config", copy=True)
    .add_local_file("main.py", "/app/main.py", copy=True)
    .workdir("/app")
    .env({"PYTHONPATH": "/app:/app/src", "MPLBACKEND": "Agg"})
)

app = modal.App(
    name="choreo-matching",
    secrets=[modal.Secret.from_name("choreo-secrets")],
)

# Persistent storage for run artifacts (outputs, embeddings cache, etc.).
volume = modal.Volume.from_name("choreo-data", create_if_missing=True)
VOLUME_MOUNT = "/app/data"


# ---------------------------------------------------------------------------
# Optional S3 upload helpers (only used when AWS_BUCKET_NAME is configured)
# ---------------------------------------------------------------------------
def upload_file_to_s3(
    s3_client, file_path: str, bucket_name: str, name: Optional[str] = None
) -> tuple[str, str]:
    """Upload a file to S3 (keyed by SHA256 hash) and return (url, sha_hash)."""
    with open(file_path, "rb") as f:
        buffer = f.read()

    hasher = hashlib.sha256()
    hasher.update(buffer)
    sha_hash = hasher.hexdigest()

    if not name:
        name = sha_hash

    file_extension = Path(file_path).suffix
    if not file_extension:
        mime_type, _ = mimetypes.guess_type(file_path)
        file_extension = mimetypes.guess_extension(mime_type) if mime_type else ""

    filename = f"{name}{file_extension}"

    content_type, _ = mimetypes.guess_type(filename)
    if not content_type:
        content_type = "application/octet-stream"

    # Skip re-upload if the object already exists.
    try:
        s3_client.head_object(Bucket=bucket_name, Key=filename)
        return f"https://{bucket_name}.s3.amazonaws.com/{filename}", sha_hash
    except s3_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "404":
            raise e

    s3_client.upload_fileobj(
        io.BytesIO(buffer),
        bucket_name,
        filename,
        ExtraArgs={"ContentType": content_type, "ContentDisposition": "inline"},
    )
    return f"https://{bucket_name}.s3.amazonaws.com/{filename}", sha_hash


def _to_jsonable(obj):
    """Coerce numpy scalars/arrays to native Python types so the function's
    return value deserializes in a numpy-free local client (e.g. `modal run`)."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def maybe_upload_zip_to_s3(zip_path: str) -> Optional[str]:
    """Upload the outputs zip to S3 if AWS creds are configured; else no-op."""
    bucket_name = os.getenv("AWS_BUCKET_NAME")
    if not bucket_name:
        return None
    try:
        import boto3

        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION_NAME"),
        )
        url, _ = upload_file_to_s3(s3_client, zip_path, bucket_name)
        print(f"✅ Uploaded outputs zip to S3: {url}")
        return url
    except Exception as e:  # pragma: no cover - best-effort side channel
        print(f"⚠️ Warning: Failed to upload outputs zip to S3: {e}")
        return None


# ---------------------------------------------------------------------------
# Remote matching function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    cpu=2.0,
    timeout=3600,
    min_containers=0,
)
def run_matching_pipeline(
    user_profiles_json: str,
    config_path: str = "config/config.yaml",
    force: bool = False,
) -> Dict[str, Any]:
    """Run the Choreo matching pipeline on a JSON map of profiles.

    Args:
        user_profiles_json: JSON object string ``{user_id: profile_text}``.
        config_path: Path (inside the image) to the config YAML.
        force: Re-run every step, ignoring caches.

    Returns:
        JSON-serializable dict. On success it is the cohort summary augmented
        with ``success``, ``zip_volume_path`` (path inside the Volume, used by
        the local entrypoint to download results) and ``outputs_zip_url`` (S3
        URL, or None when S3 isn't configured). On failure: ``{"success": False,
        "error": ...}``.
    """
    import yaml
    from main import run_matching_pipeline as run_pipeline

    # Parse profiles -------------------------------------------------------
    try:
        user_profiles = json.loads(user_profiles_json)
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON in user_profiles_json: {e}"}

    if not user_profiles:
        return {"success": False, "error": "No user profiles provided."}

    # Load config ----------------------------------------------------------
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        return {"success": False, "error": f"Configuration file not found: {config_path}"}
    except yaml.YAMLError as e:
        return {"success": False, "error": f"Invalid YAML in configuration file: {e}"}

    config_dict.setdefault("io", {})

    # Unique run dir inside the persistent volume so concurrent/successive
    # runs never clobber each other.
    run_id = uuid.uuid4().hex[:12]
    run_dir_rel = f"run_{run_id}"
    base_data_dir = f"{VOLUME_MOUNT}/{run_dir_rel}"

    config_dict["io"]["processed_dir"] = f"{base_data_dir}/processed"
    config_dict["io"]["embeds_dir"] = f"{base_data_dir}/embeds"
    config_dict["io"]["outputs_dir"] = f"{base_data_dir}/outputs"
    config_dict["io"]["cache_dir"] = f"{base_data_dir}/cache"
    for dir_path in config_dict["io"].values():
        if dir_path.startswith(base_data_dir):
            os.makedirs(dir_path, exist_ok=True)

    # Materialize profiles as .txt files (filename = user ID) -------------
    with tempfile.TemporaryDirectory() as temp_dir:
        profiles_dir = os.path.join(temp_dir, "profiles")
        os.makedirs(profiles_dir, exist_ok=True)
        for user_id, profile_text in user_profiles.items():
            with open(os.path.join(profiles_dir, f"{user_id}.txt"), "w", encoding="utf-8") as f:
                f.write(profile_text)

        result = run_pipeline(
            user_profiles_dir=profiles_dir,
            config_dict=config_dict,
            force=force,
        )

    print("✅ Pipeline result keys:", list(result.keys()))

    if not result.get("success"):
        # Surface the failure with whatever context the pipeline provided.
        error_response = {"success": False, "error": result.get("error", "Pipeline execution failed")}
        for k in ("profiles_count", "min_required"):
            if k in result:
                error_response[k] = result[k]
        return error_response

    # Zip the outputs into the volume so they can be retrieved -------------
    outputs_dir = result.get("outputs_dir")
    zip_volume_path = None
    outputs_zip_url = None
    if outputs_dir and os.path.isdir(outputs_dir):
        zip_base = f"{base_data_dir}/outputs"  # -> {base}/outputs.zip
        shutil.make_archive(zip_base, "zip", outputs_dir)
        zip_path = f"{zip_base}.zip"
        zip_volume_path = f"{run_dir_rel}/outputs.zip"
        outputs_zip_url = maybe_upload_zip_to_s3(zip_path)

    # Persist everything written this run to the volume.
    volume.commit()

    # Assemble a JSON-safe response (cohort summary + retrieval metadata).
    cohort_summary = result.get("cohort_summary") or {}
    response = dict(cohort_summary) if isinstance(cohort_summary, dict) else {"cohort_summary": cohort_summary}
    response.update(
        {
            "success": True,
            "run_dir": run_dir_rel,
            "zip_volume_path": zip_volume_path,
            "outputs_zip_url": outputs_zip_url,
            "profiles_count": result.get("profiles_count"),
            "stats": result.get("stats"),
        }
    )
    return _to_jsonable(response)


# ---------------------------------------------------------------------------
# Local entrypoint — `modal run deploy_modal.py --input-dir data/test4`
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    input_dir: Optional[str] = None,
    profiles_json: Optional[str] = None,
    force: bool = False,
):
    """Run a matching job and download the outputs zip back locally.

    Provide either --input-dir (a folder of .txt profiles, filename = user ID)
    or --profiles-json (a JSON file of {user_id: profile_text}).
    """
    if not input_dir and not profiles_json:
        raise SystemExit("Provide --input-dir <folder of .txt> or --profiles-json <file.json>")

    if input_dir:
        folder = Path(input_dir).expanduser()
        raw = folder / "raw" if (folder / "raw").is_dir() else folder
        profiles = {p.stem: p.read_text(encoding="utf-8") for p in sorted(raw.glob("*.txt"))}
        dest_dir = folder
    else:
        profiles = json.loads(Path(profiles_json).expanduser().read_text(encoding="utf-8"))
        dest_dir = Path(profiles_json).expanduser().parent

    if not profiles:
        raise SystemExit("No profiles found.")

    print(f"📤 Submitting {len(profiles)} profiles to Modal (force={force})...")
    result = run_matching_pipeline.remote(json.dumps(profiles), force=force)
    print("\n📥 Result:")
    print(json.dumps(result, indent=2, default=str))

    zip_volume_path = result.get("zip_volume_path") if isinstance(result, dict) else None
    if result.get("success") and zip_volume_path:
        local_zip = Path(dest_dir) / "outputs_modal.zip"
        with open(local_zip, "wb") as f:
            for chunk in volume.read_file(zip_volume_path):
                f.write(chunk)
        print(f"\n✅ Downloaded outputs to: {local_zip}")
        print(f"   Unzip with: unzip -o '{local_zip}' -d '{dest_dir}/outputs_modal'")
    elif not result.get("success"):
        print(f"\n❌ Pipeline failed: {result.get('error')}")
