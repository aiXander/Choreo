"""Generate embeddings for profile sections via OpenRouter.

The embed stage is split into a pure transform (``embed_sections`` — no disk
IO; reuse is content-hash based via an optional ``existing`` bundle, so adding
or removing one user re-embeds only that user's changed cells) and a
filesystem wrapper (``create_section_embeddings`` — preserves the historical
``embeds/`` directory layout and return tuple).
"""

import time
import numpy as np
from typing import Callable, List, Dict, Optional, Tuple

from .utils import ensure_dir, hash_text, is_absent
from .schemas import (
    EmbeddingsBundle,
    ExtractedSections,
    HydeDescriptors,
    hyde_content_hash,
)
from .cost_tracker import get_cost_tracker
from .llm import get_openrouter_client, extract_usage, DEFAULT_EMBEDDING_MODEL


# Transient error markers worth retrying: rate limits, 5xx server errors, and
# connection/timeout blips (common on live venue wifi). A 400 (bad request,
# e.g. batch too large) is NOT transient and should fail fast so it's visible.
_TRANSIENT_ERROR_MARKERS = (
    "rate limit", "rate_limit", "429", "too many requests", "throttle",
    "500", "502", "503", "504", "520", "521", "522", "524", "529",
    "internal server error", "bad gateway", "service unavailable",
    "gateway timeout", "overloaded", "timeout", "timed out",
    "connection", "connection error", "connection reset", "temporarily",
)


def _is_transient_embedding_error(error: Exception) -> bool:
    """Whether an embedding API error looks transient (worth retrying)."""
    # Honor an explicit HTTP status code if the SDK exposes one.
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        if status == 429 or 500 <= status < 600:
            return True
        if 400 <= status < 500:
            return False  # client error (e.g. 400 batch-too-large) — fail fast
    error_str = str(error).lower()
    return any(marker in error_str for marker in _TRANSIENT_ERROR_MARKERS)


# Embedding models known to be Matryoshka (MRL) trained — i.e. their leading
# dimensions can be safely kept + renormalized to produce a shorter, still-valid
# embedding. Truncating any other model would silently corrupt similarity, so
# MRL truncation (embedding_dimensions in config) is skipped unless the active
# model is listed here. Add slugs as you verify support.
MRL_CAPABLE_MODELS = {
    "google/gemini-embedding-2-preview",
}


def supports_mrl(model: str) -> bool:
    """Whether `model` is known to support Matryoshka (MRL) truncation."""
    return model in MRL_CAPABLE_MODELS


def get_embeddings(
    texts: List[str],
    model: str,
    max_retries: int = 4,
    retry_delay_base: float = 1.0,
) -> np.ndarray:
    """
    Get embeddings for a list of texts via OpenRouter's embeddings endpoint.

    Always fetches the model's full native dimensionality. Matryoshka (MRL)
    truncation to a smaller size is applied later, at computation time, via
    truncate_embeddings() — so the on-disk vectors stay full and the truncation
    size can be re-tuned without re-embedding.

    Transient failures (rate limits, 5xx, connection/timeout blips) are retried
    with exponential backoff so a single network hiccup on live wifi doesn't
    abort the whole pipeline. Non-transient errors (e.g. a 400 from too-large a
    batch) fail fast so they stay visible.

    Args:
        texts: List of text strings to embed
        model: Embedding model name (e.g. "google/gemini-embedding-2-preview")
        max_retries: Max retries on transient errors (default: 4)
        retry_delay_base: Base delay (s) for exponential backoff (default: 1.0)

    Returns:
        numpy array of shape (len(texts), embedding_dim)

    Empty/whitespace-only texts (e.g. a section a profile didn't fill) are never
    sent to the API — Google AI Studio rejects any batch containing an empty Part
    with a 400, which OpenRouter surfaces as a 200 carrying ``data: null`` rather
    than an exception. Such inputs are mapped to zero vectors, which read as "no
    signal" (cosine 0 with everyone) downstream.
    """
    model = model or DEFAULT_EMBEDDING_MODEL
    client = get_openrouter_client()

    # Split out empty inputs so the API only ever sees non-empty Parts.
    nonempty_idx = [i for i, t in enumerate(texts) if t and t.strip()]
    if not nonempty_idx:
        raise ValueError(
            "get_embeddings called with no non-empty texts; cannot infer "
            "embedding dimensionality for an all-empty batch."
        )
    api_texts = [texts[i] for i in nonempty_idx]

    last_error: Exception = None
    for attempt in range(max_retries + 1):
        try:
            # The OpenAI SDK defaults to encoding_format="base64", which OpenRouter's
            # Google AI Studio embedding provider rejects (400 → empty data → the SDK
            # raises "No embedding data received"). Force "float" to stay compatible.
            response = client.embeddings.create(
                model=model, input=api_texts, encoding_format="float"
            )

            # OpenRouter passes provider-side errors (e.g. a 400) back as a 200
            # body with ``data: null`` and an ``error`` field, so the SDK never
            # raises. Surface it as a clear exception instead of letting the
            # downstream comprehension fail with "'NoneType' object is not iterable".
            if response.data is None:
                err = getattr(response, "model_extra", {}) or {}
                raise RuntimeError(
                    f"Embedding API returned no data (model={model}): "
                    f"{err.get('error', response)}"
                )

            # Track cost using OpenRouter's native usage accounting (real cost in USD
            # credits, returned automatically). Falls back to 0 if not reported.
            cost_tracker = get_cost_tracker()
            try:
                usage = extract_usage(response)
                input_tokens = usage["prompt_tokens"] or len(texts)

                cost_tracker.record_call(
                    component="embeddings",
                    call_type="embedding",
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=0,  # Embeddings don't have output tokens
                    cost=usage["cost"] or 0.0,
                )
            except (AttributeError, KeyError, TypeError):
                # If cost tracking fails, continue without it
                print(f"Warning: Could not track cost for embedding call with model {model}")

            api_embeddings = np.array([item.embedding for item in response.data])

            # Scatter results back to original positions; empty inputs stay zero.
            embeddings = np.zeros((len(texts), api_embeddings.shape[1]))
            for slot, src in enumerate(nonempty_idx):
                embeddings[src] = api_embeddings[slot]

            n_empty = len(texts) - len(nonempty_idx)
            suffix = f" ({n_empty} empty → zero vectors)" if n_empty else ""
            print(f"Created embeddings with {model} of shape {embeddings.shape}{suffix}")

            return embeddings

        except Exception as e:
            last_error = e
            if attempt < max_retries and _is_transient_embedding_error(e):
                delay = retry_delay_base * (2 ** attempt)
                print(f"Transient embedding error (attempt {attempt + 1}/{max_retries + 1}), "
                      f"retrying in {delay:.1f}s: {e}")
                time.sleep(delay)
            else:
                print(f"Error getting embeddings: {e}")
                raise

    # Should not reach here; re-raise the last error defensively.
    raise last_error


def truncate_embeddings(arr: np.ndarray, dimensions: int) -> np.ndarray:
    """
    Matryoshka (MRL) truncation of stored full-size embeddings, applied at
    computation time.

    gemini-embedding-2 is MRL-trained: the most important information is packed
    into the leading dimensions, so keeping the first `dimensions` components and
    L2-renormalizing reproduces what the API returns for an equivalent
    output_dimensionality request (verified to ~1e-7). This lets us store full
    3072-dim vectors once and re-tune the working size for free.

    Truncates along the last axis, so it works for both the section embeddings
    (users × sections × dims) and HyDE embeddings (users × descriptors × dims).

    Args:
        arr: Embedding array with the embedding dimension as the last axis
        dimensions: Target size. None or >= current size returns arr unchanged.

    Returns:
        Truncated, unit-normalized array (or arr unchanged).
    """
    if not dimensions or dimensions >= arr.shape[-1]:
        return arr
    sliced = arr[..., :dimensions]
    norms = np.linalg.norm(sliced, axis=-1, keepdims=True)
    return sliced / np.clip(norms, 1e-12, None)


# Batch cap for embedding requests — Google AI Studio rejects larger batches
# with a 400 ("at most 100 requests can be in one batch").
EMBED_BATCH_SIZE = 100


def _embed_texts_batched(
    texts: List[str],
    embed_fn: Callable[[List[str]], np.ndarray],
) -> Optional[np.ndarray]:
    """Embed texts in API-sized batches. Returns None for an empty list."""
    if not texts:
        return None
    batches = []
    n_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batches.append(embed_fn(texts[i:i + EMBED_BATCH_SIZE]))
        if n_batches > 1:
            print(f"Processed batch {i // EMBED_BATCH_SIZE + 1}/{n_batches}")
    return np.vstack(batches)


def embed_sections(
    extracted_sections: List[ExtractedSections],
    embedding_model: str,
    hyde_descriptors: Optional[Dict[str, List[HydeDescriptors]]] = None,
    existing: Optional[EmbeddingsBundle] = None,
    embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
) -> EmbeddingsBundle:
    """Pure embed transform: sections (+ HyDE) in, EmbeddingsBundle out.

    Reuse is CONTENT-HASH based, not roster based: every (user, section) cell
    whose text hash matches the corresponding cell in ``existing`` keeps its
    stored vector; only changed/new cells hit the API. Adding or removing a
    user therefore never re-embeds anyone else. The caller decides where
    ``existing`` came from (FileStore, Neon, nothing).

    Args:
        extracted_sections: Sections per user (section order taken from the
            first profile; all profiles share the active-section structure).
        embedding_model: Embedding model slug (recorded as bundle provenance).
        hyde_descriptors: Optional ``{cross_key: [HydeDescriptors per user]}``.
        existing: Optional prior bundle to reuse vectors from. Ignored (with a
            warning) if it was produced by a different embedding model.
        embed_fn: Embedding callable ``texts -> (n, dim)`` (defaults to the
            OpenRouter ``get_embeddings``; injectable for tests).

    Returns:
        EmbeddingsBundle with full-size vectors, provenance and content hashes.
    """
    if not extracted_sections:
        raise ValueError("No extracted sections provided")

    if embed_fn is None:
        embed_fn = lambda texts: get_embeddings(texts, embedding_model)  # noqa: E731

    section_names = list(extracted_sections[0].sections.keys())
    user_ids = [profile.id for profile in extracted_sections]
    n_users, n_sections = len(user_ids), len(section_names)

    print(f"Creating embeddings for {n_users} users, {n_sections} sections each")

    # An existing bundle from a different model is unusable (vectors live in a
    # different space). Model migration is explicitly out of scope — fail soft
    # by re-embedding everything.
    if existing is not None and existing.embedding_model not in (None, embedding_model):
        print(f"⚠️  Existing embeddings were created with model "
              f"'{existing.embedding_model}', not '{embedding_model}' — ignoring them.")
        existing = None

    ex_user_idx = {u: i for i, u in enumerate(existing.user_ids)} if existing else {}
    ex_sec_idx = {s: i for i, s in enumerate(existing.section_names)} if existing else {}

    # ---- main section embeddings -----------------------------------------
    section_hashes: Dict[str, Dict[str, str]] = {}
    reused: List[Tuple[int, int, np.ndarray]] = []   # (user_idx, section_idx, vector)
    to_embed: List[Tuple[int, int, str]] = []        # (user_idx, section_idx, text)
    n_empty = 0

    for user_idx, profile in enumerate(extracted_sections):
        user_hashes: Dict[str, str] = {}
        for section_idx, section_name in enumerate(section_names):
            text = profile.sections.get(section_name, "")
            text_hash = hash_text(text)
            user_hashes[section_name] = text_hash

            if is_absent(text):
                # Absent section (empty OR the "Not specified" placeholder)
                # -> zero vector ("no signal"), no API call. Checked BEFORE
                # reuse so phantom placeholder vectors in older bundles get
                # zeroed out instead of carried forward.
                n_empty += 1
            elif (
                existing is not None
                and profile.id in ex_user_idx
                and section_name in ex_sec_idx
                and existing.section_hashes.get(profile.id, {}).get(section_name) == text_hash
            ):
                vector = existing.embeddings[ex_user_idx[profile.id], ex_sec_idx[section_name]]
                reused.append((user_idx, section_idx, vector))
            else:
                to_embed.append((user_idx, section_idx, text))
        section_hashes[profile.id] = user_hashes

    print(f"  Section cells: {len(reused)} reused, {len(to_embed)} to embed, {n_empty} empty")

    fresh = _embed_texts_batched([t for _, _, t in to_embed], embed_fn)

    # Resolve the embedding dim from whatever source is available.
    if fresh is not None:
        embedding_dim = fresh.shape[1]
    elif reused:
        embedding_dim = reused[0][2].shape[-1]
    elif existing is not None:
        embedding_dim = existing.embeddings.shape[-1]
    else:
        raise ValueError(
            "Cannot determine embedding dimensionality: all section texts are "
            "empty and no existing embeddings were provided."
        )
    if reused and reused[0][2].shape[-1] != embedding_dim:
        raise ValueError(
            f"Embedding dim mismatch: existing vectors have dim "
            f"{reused[0][2].shape[-1]} but '{embedding_model}' returned "
            f"{embedding_dim}. Re-embed from scratch (force)."
        )

    embeddings_array = np.zeros((n_users, n_sections, embedding_dim))
    for user_idx, section_idx, vector in reused:
        embeddings_array[user_idx, section_idx] = vector
    if fresh is not None:
        for (user_idx, section_idx, _), vector in zip(to_embed, fresh):
            embeddings_array[user_idx, section_idx] = vector

    # ---- HyDE embeddings ---------------------------------------------------
    hyde_embeddings: Dict[str, np.ndarray] = {}
    hyde_hashes: Dict[str, Dict[str, str]] = {}

    if hyde_descriptors:
        print(f"Embedding HyDE descriptors for {len(hyde_descriptors)} cross-section pairs...")

        for cross_key, user_descs in hyde_descriptors.items():
            n_desc = len(user_descs[0].descriptors)
            key_hashes: Dict[str, str] = {}
            section_embeds = np.zeros((n_users, n_desc, embedding_dim))

            ex_hyde = existing.hyde.get(cross_key) if existing is not None else None
            ex_hyde_hashes = existing.hyde_hashes.get(cross_key, {}) if existing is not None else {}

            hyde_to_embed: List[Tuple[int, int, str]] = []  # (user_idx, desc_idx, text)
            n_reused_rows = 0

            for user_idx, ud in enumerate(user_descs):
                desc_hash = hyde_content_hash(ud.descriptors)
                key_hashes[ud.user_id] = desc_hash

                if (
                    ex_hyde is not None
                    and ud.user_id in ex_user_idx
                    and ex_hyde_hashes.get(ud.user_id) == desc_hash
                    and ex_hyde.shape[1] == n_desc
                    and ex_hyde.shape[2] == embedding_dim
                ):
                    section_embeds[user_idx] = ex_hyde[ex_user_idx[ud.user_id]]
                    n_reused_rows += 1
                    continue

                for d, text in enumerate(ud.descriptors):
                    if not is_absent(text):
                        hyde_to_embed.append((user_idx, d, text))

            print(f"  {cross_key}: {n_users} users x {n_desc} descriptors "
                  f"({n_reused_rows} reused, {len(hyde_to_embed)} descriptors to embed)")

            fresh_hyde = _embed_texts_batched([t for _, _, t in hyde_to_embed], embed_fn)
            if fresh_hyde is not None:
                if fresh_hyde.shape[1] != embedding_dim:
                    raise ValueError(
                        f"HyDE embedding dim {fresh_hyde.shape[1]} does not match "
                        f"section embedding dim {embedding_dim}."
                    )
                for (user_idx, d, _), vector in zip(hyde_to_embed, fresh_hyde):
                    section_embeds[user_idx, d] = vector

            hyde_embeddings[cross_key] = section_embeds
            hyde_hashes[cross_key] = key_hashes

    # Per-user freshness provenance: each row's vectors reflect that user's
    # sections as of this timestamp (content hashes guarantee reused cells are
    # identical to the current text, so the current timestamp applies to them
    # too). Adapters compare these against their source data's updated_at
    # (utils.is_stale) to decide when an upsert/re-embed is needed.
    user_timestamps = {
        p.id: p.last_updated_at for p in extracted_sections if p.last_updated_at
    }

    return EmbeddingsBundle(
        user_ids=user_ids,
        section_names=section_names,
        embeddings=embeddings_array,
        hyde=hyde_embeddings,
        embedding_model=embedding_model,
        dim=int(embedding_dim),
        section_hashes=section_hashes,
        hyde_hashes=hyde_hashes,
        user_timestamps=user_timestamps,
    )


def _trust_legacy_bundle(
    existing: EmbeddingsBundle,
    extracted_sections: List[ExtractedSections],
    hyde_descriptors: Optional[Dict[str, List[HydeDescriptors]]],
) -> Optional[EmbeddingsBundle]:
    """Adopt a pre-refactor embeds dir (no content hashes) when safe.

    The old cache was roster-keyed: it was valid iff the user set and section
    list matched exactly. Reproduce that adapter-level decision here by
    synthesizing content hashes from the *current* texts, which makes every
    cell reusable for an unchanged roster. Returns None when the roster check
    fails (the transform will then re-embed everything, as the old code did).
    """
    section_names = list(extracted_sections[0].sections.keys())
    user_ids = [p.id for p in extracted_sections]
    if set(existing.user_ids) != set(user_ids) or existing.section_names != section_names:
        return None

    print("Adopting legacy embeddings dir (no content hashes) for unchanged roster")
    existing.section_hashes = {
        p.id: {name: hash_text(p.sections.get(name, "")) for name in section_names}
        for p in extracted_sections
    }
    if hyde_descriptors:
        for cross_key, user_descs in hyde_descriptors.items():
            ex_hyde = existing.hyde.get(cross_key)
            if ex_hyde is None or ex_hyde.shape[1] != len(user_descs[0].descriptors):
                continue
            existing.hyde_hashes[cross_key] = {
                ud.user_id: hyde_content_hash(ud.descriptors) for ud in user_descs
            }
    return existing


def create_section_embeddings_bundle(
    extracted_sections: List[ExtractedSections],
    embedding_model: str,
    embeds_dir: str,
    hyde_descriptors: Dict[str, List[HydeDescriptors]] = None,
    force: bool = False
) -> EmbeddingsBundle:
    """Filesystem wrapper around ``embed_sections``.

    Loads any existing bundle from ``embeds_dir`` as the reuse source, runs the
    content-hash-diffing transform, and persists the refreshed bundle back to
    the same layout (vectors.npz / ids.json / section_names.json /
    hyde_vectors.npz / bundle_meta.json).
    """
    embeds_path = ensure_dir(embeds_dir)

    if not extracted_sections:
        raise ValueError("No extracted sections provided")

    existing: Optional[EmbeddingsBundle] = None
    if force:
        print("Force flag set - regenerating all embeddings")
    else:
        try:
            existing = EmbeddingsBundle.load(embeds_path)
        except FileNotFoundError:
            existing = None
        except Exception as e:  # pylint: disable=broad-except
            print(f"Warning: Could not load existing embeddings: {e}")
            existing = None

        if existing is not None and not existing.section_hashes:
            existing = _trust_legacy_bundle(existing, extracted_sections, hyde_descriptors)

    bundle = embed_sections(
        extracted_sections=extracted_sections,
        embedding_model=embedding_model,
        hyde_descriptors=hyde_descriptors,
        existing=existing,
    )

    bundle.dump(embeds_path)
    print(f"Saved embeddings to {embeds_path}")
    print(f"Shape: {bundle.embeddings.shape} (users, sections, dims)")

    return bundle


def create_section_embeddings(
    extracted_sections: List[ExtractedSections],
    embedding_model: str,
    embeds_dir: str,
    hyde_descriptors: Dict[str, List[HydeDescriptors]] = None,
    force: bool = False
) -> Tuple[List[str], List[str], np.ndarray, Dict[str, np.ndarray]]:
    """Legacy tuple API over ``create_section_embeddings_bundle``.

    Returns:
        Tuple of (user_ids, section_names, embeddings_array, hyde_embeddings)
        - embeddings_array shape: (n_users, n_sections, embedding_dim)
        - hyde_embeddings: dict of cross_key -> (n_users, n_descriptors, embedding_dim)
    """
    bundle = create_section_embeddings_bundle(
        extracted_sections=extracted_sections,
        embedding_model=embedding_model,
        embeds_dir=embeds_dir,
        hyde_descriptors=hyde_descriptors,
        force=force,
    )
    return bundle.user_ids, bundle.section_names, bundle.embeddings, bundle.hyde


def load_embeddings_bundle(embeds_dir: str) -> EmbeddingsBundle:
    """Load the full embeddings bundle from disk (tolerates legacy dirs)."""
    return EmbeddingsBundle.load(embeds_dir)


def load_embeddings(embeds_dir: str) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Load embeddings from disk (legacy tuple API; see ``load_embeddings_bundle``).

    Returns:
        Tuple of (user_ids, section_names, embeddings_array)
    """
    try:
        bundle = EmbeddingsBundle.load(embeds_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Embedding files not found. Run embedding generation first."
        ) from exc
    return bundle.user_ids, bundle.section_names, bundle.embeddings