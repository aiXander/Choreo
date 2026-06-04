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
LLM + embedding calls route through OpenRouter — see choreo/llm.py). To also enable
S3 upload, add AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION_NAME /
AWS_BUCKET_NAME to that same secret.

Create it from your local .env with:
    modal secret create choreo-secrets OPENROUTER_API_KEY=sk-or-...

COMMANDS
--------
# Deploy the functions (callable via .remote / .spawn from other apps)
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

GRANULAR ENDPOINTS (persistent FileStore at groups/<group> on the Volume)
--------------------------------------------------------------------------
Besides the legacy full run, three mode endpoints are exposed for standalone
Modal use. (In the real product the app imports the runners directly with its
own data; these endpoints are the FileStore-backed convenience path.)

  upsert_profiles(user_profiles_json, group)   # Mode A: extract+HyDE+embed,
                                               # incremental via content hashes
  query_match(payload_json, group)             # Mode B: 1×M ranked shortlist
  batch_match(members_json, group)             # Mode C: M×N novel matches,
                                               # history on the Volume

e.g. from another app:
  f = modal.Function.from_name("choreo-matching", "upsert_profiles")
  f.remote(json.dumps({"alice": "profile text…"}), group="wintercircus")
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
            "python-dotenv>=1.0.1",
            "boto3>=1.34.0",
        ]
    )
    # The choreo package ships its default config + prompts as package data
    # (choreo/defaults/), so copying the package dir is all that's needed.
    .add_local_dir("choreo", "/app/choreo", copy=True)
    .add_local_file("main.py", "/app/main.py", copy=True)
    .workdir("/app")
    .env({"PYTHONPATH": "/app", "MPLBACKEND": "Agg"})
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
    config_overrides_json: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Run the Choreo matching pipeline on a JSON map of profiles.

    Args:
        user_profiles_json: JSON object string ``{user_id: profile_text}``.
        config_overrides_json: Optional JSON object deep-merged over the
            packaged default config (see choreo/config.py).
        force: Re-run every step, ignoring caches.

    Returns:
        JSON-serializable dict. On success it is the cohort summary augmented
        with ``success``, ``zip_volume_path`` (path inside the Volume, used by
        the local entrypoint to download results) and ``outputs_zip_url`` (S3
        URL, or None when S3 isn't configured). On failure: ``{"success": False,
        "error": ...}``.
    """
    from main import run_matching_pipeline as run_pipeline

    # Parse profiles -------------------------------------------------------
    try:
        user_profiles = json.loads(user_profiles_json)
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON in user_profiles_json: {e}"}

    if not user_profiles:
        return {"success": False, "error": "No user profiles provided."}

    # Load config (packaged defaults + optional overrides) ------------------
    try:
        config_dict = _load_config(config_overrides_json)
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON in config_overrides_json: {e}"}

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
# Granular mode endpoints (Mode A/B/C) — persistent FileStore per group
# ---------------------------------------------------------------------------
def _group_store(group: str):
    """FileStore rooted at the group's directory on the Volume."""
    from choreo.store import FileStore

    return FileStore(f"{VOLUME_MOUNT}/groups/{group}")


# Warm-container cache of each group's loaded pool (bundle + sections), keyed
# on the on-disk file signature so it invalidates automatically when
# upsert_profiles rewrites the embeds/sections (including writes committed by
# OTHER containers — endpoints call volume.reload() first).
_POOL_CACHE: Dict[str, tuple] = {}


def _pool_signature(store) -> tuple:
    sig = []
    for path in (
        Path(store.embeds_dir) / "vectors.npz",
        Path(store.embeds_dir) / "bundle_meta.json",
        store.sections_file,
    ):
        try:
            st = os.stat(path)
            sig.append((str(path), st.st_mtime_ns, st.st_size))
        except FileNotFoundError:
            sig.append((str(path), None, None))
    return tuple(sig)


def _get_group_pool(group: str, store):
    """Load (or reuse) the group's embeddings bundle + sections.

    Avoids re-reading the full-size vectors.npz and re-parsing sections.jsonl
    on every warm query — the pool only changes on upsert.
    """
    sig = _pool_signature(store)
    cached = _POOL_CACHE.get(group)
    if cached and cached[0] == sig:
        return cached[1], cached[2]
    bundle = store.get_embeddings()  # raises FileNotFoundError if absent
    sections = {s.id: s.sections for s in store.get_sections()}
    _POOL_CACHE[group] = (sig, bundle, sections)
    return bundle, sections


def _load_config(config_overrides_json: Optional[str] = None) -> Dict[str, Any]:
    """Packaged default config, deep-merged with optional JSON overrides."""
    from choreo.config import load_config

    overrides = json.loads(config_overrides_json) if config_overrides_json else None
    return load_config(overrides=overrides)


@app.function(image=image, volumes={VOLUME_MOUNT: volume}, cpu=2.0, timeout=3600)
def upsert_profiles(
    user_profiles_json: str,
    group: str = "default",
    config_overrides_json: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Mode A: upsert profiles into a group's persistent store.

    Extracts sections (content-hash cached — unchanged profiles skip the LLM),
    regenerates HyDE for the whole roster (cached per content hash), and
    re-embeds ONLY changed/new cells (content-hash diff). The refreshed
    sections + embeddings bundle persist on the Volume for query/batch calls.

    ``user_profiles_json`` values are either plain profile text or
    ``{"text": ..., "last_updated_at": "<ISO-8601>"}`` — the latter lets an
    external store propagate its own updated_at timestamps onto the derived
    sections/embeddings (defaults to "now" when omitted).
    """
    from choreo.utils import hash_text, filter_active_sections, load_yaml, utc_now_iso, DEFAULT_PROMPT_PATHS
    from choreo.ingest import Profile
    from choreo.llm import LLMWrapper
    from choreo.extract import extract_sections
    from choreo.hyde import generate_hyde_descriptors
    from choreo.embed import create_section_embeddings_bundle
    from pathlib import Path as _Path

    try:
        user_profiles = json.loads(user_profiles_json)
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON in user_profiles_json: {e}"}
    if not user_profiles:
        return {"success": False, "error": "No user profiles provided."}

    volume.reload()  # see writes committed by other containers
    config = _load_config(config_overrides_json)
    store = _group_store(group)
    models_cfg = config.get("models", {})
    llm_wrapper = LLMWrapper(
        cache_dir=str(store.cache_dir),
        reasoning_effort=models_cfg.get("reasoning_effort", "low"),
    )
    goal = config.get("instruction_prompt", {}).get("goal")

    now = utc_now_iso()
    profiles = []
    for user_id, value in user_profiles.items():
        if isinstance(value, dict):
            text = (value.get("text") or "").strip()
            ts = value.get("last_updated_at") or now
        else:
            text, ts = value.strip(), now
        profiles.append(Profile(id=user_id, text=text, hash=hash_text(text),
                                last_updated_at=ts))

    # Extract (upsert semantics: force only re-extracts the GIVEN profiles,
    # it never wipes the rest of the roster).
    sections_config = filter_active_sections(load_yaml(DEFAULT_PROMPT_PATHS["sections"]))
    existing_by_hash = {} if force else {s.hash: s.sections for s in store.get_sections()}
    failed: list = []
    extracted = extract_sections(
        profiles=profiles,
        sections_config=sections_config,
        model=models_cfg.get("extraction_llm"),
        llm_wrapper=llm_wrapper,
        goal=goal,
        existing=existing_by_hash,
        max_calls=config.get("budgets", {}).get("extraction_llm_calls", 300),
        use_llm_cache=not force,
        failed_out=failed,
    )
    store.put_sections([es for es in extracted if es.id not in set(failed)])

    # HyDE + embed over the FULL roster — content-hash caches make this
    # incremental (only new/changed users actually hit the APIs).
    all_sections = store.get_sections()
    cross_weights = config.get("recipe", {}).get("cross_section_weights", {}) or {}
    hyde_descriptors = {}
    if cross_weights:
        hyde_prompt_template = load_yaml(DEFAULT_PROMPT_PATHS["hyde"])["hyde_generation"]
        hyde_descriptors = generate_hyde_descriptors(
            extracted_sections=all_sections,
            cross_section_weights=cross_weights,
            hyde_config=config.get("hyde", {}),
            prompt_template=hyde_prompt_template,
            goal=goal,
            llm_wrapper=llm_wrapper,
            model=models_cfg.get("extraction_llm"),
            cache_dir=_Path(store.processed_dir),
            sections_config=load_yaml(DEFAULT_PROMPT_PATHS["sections"]),
            force=force,
        )

    bundle = create_section_embeddings_bundle(
        extracted_sections=all_sections,
        embedding_model=models_cfg.get("embedding"),
        embeds_dir=str(store.embeds_dir),
        hyde_descriptors=hyde_descriptors or None,
        force=force,
    )

    volume.commit()
    _POOL_CACHE.pop(group, None)  # next query/batch reloads the fresh pool
    return _to_jsonable({
        "success": True,
        "group": group,
        "upserted": [p.id for p in profiles],
        "failed": failed,
        "roster_size": len(bundle.user_ids),
        "embedding_model": bundle.embedding_model,
        "embedding_dim": bundle.dim,
    })


@app.function(image=image, volumes={VOLUME_MOUNT: volume}, cpu=2.0, timeout=1800)
def query_match(
    payload_json: str,
    group: str = "default",
    config_overrides_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Mode B: rank the group's stored pool against one query (the hot path).

    ``payload_json``: {"query": "raw text" | {"needs": "…"}, "top_k": …,
    "llm_rerank": …, "recipe_override": …, "exclude_ids": […]}. The pool is
    read from the group's Volume store and never re-embedded.
    """
    from choreo.llm import LLMWrapper
    from choreo.query import run_query_match_json

    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON in payload_json: {e}"}

    volume.reload()  # see writes committed by other containers (e.g. upserts)
    config = _load_config(config_overrides_json)
    store = _group_store(group)
    if not payload.get("pool"):  # honor a caller-supplied inline pool
        try:
            pool, pool_sections = _get_group_pool(group, store)
        except FileNotFoundError:
            return {"success": False,
                    "error": f"No embeddings for group '{group}' — call upsert_profiles first."}
        payload["pool"] = pool
        payload.setdefault("pool_sections", pool_sections)

    llm_wrapper = LLMWrapper(
        cache_dir=str(store.cache_dir),
        reasoning_effort=config.get("models", {}).get("reasoning_effort", "low"),
    )
    result = run_query_match_json(payload, config, llm_wrapper=llm_wrapper)
    if llm_wrapper.cache_writes:  # nothing to persist on a fully cache-hit call
        volume.commit()
    return _to_jsonable(result)


@app.function(image=image, volumes={VOLUME_MOUNT: volume}, cpu=2.0, timeout=3600)
def batch_match(
    members_json: str,
    group: str = "default",
    config_overrides_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Mode C: match a member subset against the group's pool, novel pairs only.

    ``members_json``: JSON array of member ids. Pairs surfaced within
    ``matching.novelty_window_months`` (from the group's match_history.jsonl
    on the Volume) are excluded; this run's new pairs are appended back.
    """
    from choreo.llm import LLMWrapper
    from choreo.batch_match import run_batch_match

    try:
        member_ids = json.loads(members_json)
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON in members_json: {e}"}
    if not isinstance(member_ids, list) or not member_ids:
        return {"success": False, "error": "members_json must be a non-empty JSON array of user ids"}

    volume.reload()  # see writes committed by other containers (e.g. upserts)
    config = _load_config(config_overrides_json)
    store = _group_store(group)

    try:
        pool, pool_sections = _get_group_pool(group, store)
    except FileNotFoundError:
        return {"success": False,
                "error": f"No embeddings for group '{group}' — call upsert_profiles first."}

    novelty_window = config.get("matching", {}).get("novelty_window_months", 6)
    excluded_pairs = store.get_match_history(window_months=novelty_window)

    llm_wrapper = LLMWrapper(
        cache_dir=str(store.cache_dir),
        reasoning_effort=config.get("models", {}).get("reasoning_effort", "low"),
    )

    try:
        result = run_batch_match(
            member_ids=member_ids,
            pool=pool,
            config=config,
            excluded_pairs=excluded_pairs,
            pool_sections=pool_sections,
            llm_wrapper=llm_wrapper,
        )
    except Exception as e:  # pylint: disable=broad-except
        return {"success": False, "error": f"batch_match failed: {e}"}

    store.put_matches(result.edges)
    volume.commit()
    return _to_jsonable({"success": True, "group": group, **result.to_dict()})


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
