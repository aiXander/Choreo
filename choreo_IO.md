# Choreo Pipeline IO Specification

Reference document for wrapping the Choreo matching pipeline as an external tool call. Covers all inputs, outputs, filesystem artifacts, and the programmatic API surface.

---

## 1. Entry Points

### CLI Entry Point
```bash
python main.py --group <group_name> --force
```
- `--group`: String identifier for data isolation. Determines all IO paths via `data/{group}/...`
- `--force`: Boolean. Skips all caches, re-runs every step from scratch.
- `--config`: Path to config YAML (default: `config/config.yaml`)
- `--pipeline`: Pipeline name (default: `"matching"`, only option currently)

**Ref:** `main.py:564-616`

### Programmatic Entry Point (for wrapping)
```python
from main import run_matching_pipeline

result = run_matching_pipeline(
    user_profiles_dir="path/to/profiles/",   # Directory of .txt files
    config_dict={...},                         # Parsed config.yaml dict
    sections_config_path="config/section_prompt.yaml",
    scoring_prompt_path="config/scoring_prompt.yaml",
    introduction_prompt_path="config/introduction_prompt.yaml",
    hyde_prompt_path="config/hyde_prompt.yaml",
    force=False,
    group_name="my_group"                      # Optional
)
```
**Ref:** `main.py:496-523`

### Modal (Serverless) Entry Point
```python
# Input: JSON string of {user_id: profile_text, ...}
run_matching_pipeline(
    user_profiles_json='{"alice": "profile text...", "bob": "..."}',
    config_path="config/config.yaml",   # Optional, default shown
    force=False                          # Optional, default shown
)
```
The Modal function parses the JSON into a dict, writes profile texts to temp `.txt` files, loads the config YAML, and delegates to the programmatic `run_matching_pipeline()` (passing only `user_profiles_dir`, `config_dict`, and `force` — no prompt paths or group name). A unique UUID-based data directory on a Modal volume is used for intermediate files and cleaned up after each run. On success it returns `cohort_summary` (the same dict described in section 5 below) with `success: True` and `outputs_zip_url` added (the latter may be `None` if S3 upload fails).

**App name:** `profile-matching` (Modal secret: `eve-secrets-PROD`)

**Deployment & Invocation:**
```bash
# Deploy to Modal
modal deploy deploy_modal.py

# Invoke the deployed function
modal run deploy_modal.py::run_matching_pipeline --user-profiles-json=profiles.json --config-path=config/config.yaml

# Monitor deployments
modal app list
modal app logs profile-matching
```

**Ref:** `deploy_modal.py:185-343`

---

## 2. Required Inputs

### 2.1 User Profile Files

| Property | Value |
|----------|-------|
| Location | `data/{group}/raw/*.txt` |
| Format | One plain `.txt` file per user |
| Naming | Filename stem = user ID (e.g., `alice.txt` → ID `"alice"`) |
| Encoding | UTF-8 |
| Minimum | **4 profiles** required (hard-coded check in `main.py:192-205`) |
| Content | Free-form text describing the person's skills, projects, needs, etc. |

**Ref:** `src/ingest.py:27-59` — `load_profiles()` reads all `.txt` from `raw_dir`, creates `Profile(id, text, hash)` objects.

### 2.2 Configuration Files

All paths are relative to repo root. The pipeline requires these config files to exist:

| File | Purpose | Key Fields |
|------|---------|------------|
| `config/config.yaml` | Main pipeline config | `models`, `recipe`, `blending`, `matching`, `budgets`, `instruction_prompt`, `hyde`, `io` |
| `config/section_prompt.yaml` | Section extraction definitions | `sections` (with `active`, `guideline`, `max_words` per section), `sections_prompt` template |
| `config/scoring_prompt.yaml` | LLM pair scoring prompt | `pair_scoring` template string |
| `config/introduction_prompt.yaml` | Introduction generation prompt | `introduction_generation` template string |
| `config/hyde_prompt.yaml` | HyDE descriptor generation prompt | `hyde_generation` template string |

**Ref:** `main.py:36-41` for default paths, `main.py:121-155` for `resolve_prompt_paths()`

### 2.3 Environment Variables

| Variable | Required | Used By |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes | `src/llm.py` — OpenAI client initialization |

Loaded via `python-dotenv` from `.env` file at CLI startup (`main.py:534`).

### 2.4 Config YAML Schema (key fields)

```yaml
models:
  embedding: "text-embedding-3-small"    # OpenAI embedding model
  extraction_llm: "gpt-5.4"             # LLM for section extraction + HyDE
  pair_llm: "gpt-5.4"                   # LLM for pair scoring + introductions

instruction_prompt:
  goal: "string describing matching context"  # Injected into all LLM prompts

budgets:
  extraction_llm_calls: 100              # Max extraction calls
  max_pair_llm_calls: 300                # Global cap on pair scoring calls
  max_n_llm_evaluations_per_profile: 16  # Per-user cap on LLM evaluations
  n_profiles_to_score_together: 4        # Batch size for scoring groups

hyde:
  n_descriptors: 1                       # HyDE phrasings per source section

recipe:
  instruction: "string"                  # Scoring instruction
  section_weights:                       # Same-section similarity weights
    skills: -0.10                        # Negative = dissimilarity preferred
    vision: 0.30
    project: 0.10
    needs: 0.00
  cross_section_weights:                 # Cross-section (asymmetric) weights
    needs_skills: 0.80                   # Key format: "{source}_{target}"

blending:
  embed_weight: 0.35                     # Weight for embedding score
  llm_weight: 0.65                       # Weight for LLM score

matching:
  b_min: 2                               # Min connections per user
  b_max: 4                               # Max connections per user

io:                                      # Overridden by --group flag
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  embeds_dir: "data/embeds"
  outputs_dir: "data/outputs"
  cache_dir: "data/cache"
```

When `--group <name>` is passed, all `io.*` paths are overridden to `data/{name}/{subdir}`.

**Ref:** `main.py:94-110` — `apply_group_overrides()`

---

## 3. Pipeline Steps & Intermediate IO

All paths below assume `--group mygroup`, so base = `data/mygroup/`.

### Step 1: Ingest
- **Reads:** `data/mygroup/raw/*.txt`
- **Produces:** In-memory `List[Profile]` — `Profile(id: str, text: str, hash: str)`
- **No disk output**
- **Ref:** `src/ingest.py`

### Step 2: Section Extraction
- **Reads:** Profile texts, `config/section_prompt.yaml`
- **Produces:**
  - In-memory: `List[ExtractedSections]` — `ExtractedSections(id: str, sections: Dict[str, str], hash: str)`
  - Disk: `data/mygroup/processed/sections.jsonl`
    ```jsonl
    {"id": "alice", "sections": {"skills": "...", "vision": "...", "project": "...", "needs": "..."}, "hash": "abc123..."}
    ```
- **Cache key:** Profile content hash. Only re-extracts if profile text changed.
- **Ref:** `src/extract.py:43-234`

### Step 2.5: HyDE Descriptor Generation
- **Condition:** Only runs when `config.recipe.cross_section_weights` is non-empty
- **Reads:** Extracted sections, `config/hyde_prompt.yaml`, `config/section_prompt.yaml`
- **Produces:**
  - In-memory: `Dict[str, List[HydeDescriptors]]` — keyed by cross_key (e.g., `"needs_skills"`)
  - Disk: `data/mygroup/processed/hyde/{cross_key}.jsonl`
    ```jsonl
    {"cache_key": "...", "user_id": "alice", "cross_key": "needs_skills", "descriptors": ["hypothetical skill text..."]}
    ```
- **Ref:** `src/hyde.py:35-212`

### Step 3: Embedding Generation
- **Reads:** Extracted sections, HyDE descriptors (if any)
- **Produces:**
  - `data/mygroup/embeds/vectors.npz` — numpy array shape `(n_users, n_sections, embedding_dim)`
  - `data/mygroup/embeds/ids.json` — `["alice", "bob", ...]` (user ID ordering)
  - `data/mygroup/embeds/section_names.json` — `["skills", "vision", "project", "needs"]`
  - `data/mygroup/embeds/hyde_vectors.npz` — (if HyDE) keyed by cross_key, shape `(n_users, n_descriptors, embedding_dim)`
- **Cache:** Reuses existing embeddings if user set + section names haven't changed
- **Ref:** `src/embed.py:61-197`

### Step 3.5: t-SNE Visualization
- **Produces:** `data/mygroup/outputs/plots/tsne_*.png` (one plot per section)
- **Non-essential** — pipeline continues if this fails
- **Ref:** `src/tsne.py`

### Step 4: Similarity Matrix Generation
- **Reads:** Embeddings, HyDE embeddings, recipe config
- **Produces:** In-memory only:
  - `dir_similarity_matrix` — possibly asymmetric `(n_users, n_users)` numpy array
  - `similarity_matrix` — symmetric version `(dir + dir.T) / 2`
  - `user_ids_sorted` — user IDs in matrix order
  - `matrices_dict` — contains per-section matrices, cross-section matrices, weights, combined matrix
- **No disk output** at this step
- **Ref:** `src/candidate.py:29-255`

### Step 5: LLM Pair Scoring
- **Reads:** Similarity matrix, extracted sections, `config/scoring_prompt.yaml`
- **Produces:** In-memory `Dict[str, PairScore]` keyed by `pair_id`
  - `PairScore(pair_id: str, user1: str, user2: str, embed_score: float, score: float)`
  - `pair_id` is always alphabetically sorted: `stable_pair_id("bob", "alice") = "alice_bob"`
- **LLM cache:** Individual call results cached in `data/mygroup/cache/llm/` as JSON files
- **Ref:** `src/score.py:263-433`

### Step 6: B-Matching
- **Reads:** Candidate pairs, LLM scores, similarity matrix, blending/matching config
- **Produces:** In-memory `List[Edge]`
  ```python
  Edge(
      user1: str, user2: str, pair_id: str,
      final_weight: float,           # Blended score
      embed_score: float,            # Raw embedding similarity
      llm_score: float,              # Raw LLM score (0-1)
      embed_score_normalized: float, # Normalized embedding score
      llm_score_normalized: float,   # Normalized LLM score
      intro: str,                    # Filled in Step 7
      starter_topics: str            # Filled in Step 7
  )
  ```
- **Ref:** `src/match.py:237-287`

### Step 7: Introduction Generation
- **Reads:** Final edges, sections dict, `config/introduction_prompt.yaml`
- **Produces:** In-memory `Dict[str, Introduction]`
  - LLM returns JSON with either:
    - Directional: `{"intro_for_a": "...", "intro_for_b": "...", "starter_topics": "..."}`
    - Legacy: `{"intro": "...", "starter_topics": "..."}`
  - Introduction text is attached to Edge objects (`edge.intro`, `edge.starter_topics`)
- **LLM cache:** Results cached in `data/mygroup/cache/llm/`
- **Ref:** `src/introduction.py:57-191`

### Step 8: Report Generation
- **Produces:**
  - **Per-user reports:** `data/mygroup/outputs/{user_id}.json`
    ```json
    {
      "profile": "Profile of alice:\n\n**Skills:** ...\n**Vision:** ...",
      "matches": "### 1. bob\n\n**Match Score:** 0.742 ..."
    }
    ```
  - **Cohort summary:** `data/mygroup/outputs/cohort.json` — see Section 5 for full schema
- **Ref:** `src/report.py:183-267`

### Step 9: Visualization & Cost Report
- **Produces:**
  - `data/mygroup/outputs/plots/` — similarity heatmaps, score correlation plots
  - `data/mygroup/outputs/cost_report.json` — detailed API cost breakdown
- **Non-essential** — pipeline continues if visualization fails
- **Ref:** `src/visualize_similarity.py`, `src/score_correlation.py`, `src/cost_tracker.py`

---

## 4. Filesystem Layout (Complete)

```
data/{group}/
├── raw/                          # INPUT: User profiles
│   ├── alice.txt
│   ├── bob.txt
│   └── ...
├── processed/                    # INTERMEDIATE: Extracted data
│   ├── sections.jsonl            # Extracted sections per user
│   └── hyde/                     # HyDE descriptors (if cross_section_weights set)
│       └── needs_skills.jsonl
├── embeds/                       # INTERMEDIATE: Embedding vectors
│   ├── vectors.npz               # Shape: (n_users, n_sections, embedding_dim)
│   ├── hyde_vectors.npz          # Shape per key: (n_users, n_descriptors, embedding_dim)
│   ├── ids.json                  # User ID ordering
│   └── section_names.json        # Section name ordering
├── cache/                        # CACHE: LLM call results
│   └── llm/
│       ├── extract_<hash>.json
│       ├── hyde_<cross_key>_<hash>.json
│       ├── batch_score_<hash>.json
│       └── intro_<pair_id>_<hash>.json
└── outputs/                      # OUTPUT: Final results
    ├── alice.json                # Per-user report
    ├── bob.json
    ├── cohort.json               # Cohort summary (main output)
    ├── cost_report.json          # API cost breakdown
    └── plots/                    # Visualizations
        ├── tsne_skills.png
        ├── tsne_vision.png
        ├── similarity_*.png
        └── score_correlation_*.png
```

---

## 5. Return Value Schema (`_execute_matching_pipeline`)

On **success**, the pipeline returns:

```python
{
    "success": True,
    "matches": List[Edge],              # List of Edge dataclass instances
    "profiles_count": int,              # Number of input profiles
    "outputs_dir": str,                 # Path to outputs directory
    "cost_report_path": str,            # Path to cost_report.json
    "cohort_summary": {                 # The cohort.json content (see below)
        "overview": {
            "total_users": int,
            "total_edges": int,
            "average_degree": float,
            "edges_with_llm_scores": int
        },
        "degree_distribution": {
            "2": int,                   # Count of users with degree 2
            "3": int,                   # etc.
        },
        "score_statistics": {
            "final_weights": {"min": float, "max": float, "avg": float},
            "embedding_scores": {"min": float, "max": float, "avg": float},
            "llm_scores": {"min": float, "max": float, "avg": float}
        },
        "users": {
            "<user_id>": {
                "degree": int,
                "profile": "**Skills:** ...\n**Vision:** ...",
                "matches": [
                    {
                        "partner": str,
                        "weight": float,
                        "intro": str
                    }
                ]
            }
        }
    },
    "stats": {
        "llm_calls": int,
        "matches_created": int
    },
    "tsne_plots_dir": str,              # Optional, only if t-SNE succeeded
    "similarity_plots_dir": str         # Optional, only if plots succeeded
}
```

On **failure**:
```python
{
    "success": False,
    "error": str,                       # Human-readable error message
    "profiles_count": int,              # Optional, present for min-profile errors
    "min_required": int                 # Optional, present for min-profile errors (=4)
}
```

**Ref:** `main.py:446-465` (success), `main.py:195-205` (min-profile failure)

### Modal Return Value

**On success with `cohort_summary`** (normal case):
The Modal entry point returns just the `cohort_summary` dict with two extra fields:
```python
cohort_summary["success"] = True
cohort_summary["outputs_zip_url"] = "https://..." or None  # S3 URL of zipped outputs (None if upload failed)
```

**On success without `cohort_summary`** (edge case):
```python
{
    "success": True,
    "error": "Pipeline succeeded but cohort_summary not found in result",
    "outputs_zip_url": "https://..." or None
}
```

**On pipeline failure:**
```python
{
    "success": False,
    "error": str,                  # Human-readable error message
    "profiles_count": int,         # Optional, present for min-profile errors
    "min_required": int            # Optional, present for min-profile errors
}
```

**On JSON parse error:**
```python
{"success": False, "error": "Invalid JSON in user_profiles_json: ..."}
```

**On empty profiles:**
```python
{"success": False, "error": "No user profiles provided. Please provide at least 6 user profiles."}
```

**On config file error:**
```python
{"success": False, "error": "Configuration file not found: ..." or "Invalid YAML in configuration file: ..."}
```

**On unexpected exception:**
```python
{"success": False, "error": "Unexpected error in matching pipeline: <ExceptionType>: <message>"}
```

**Ref:** `deploy_modal.py:302-343`

---

## 6. Key Conventions for Wrapping

### Pair ID Stability
All pair IDs are alphabetically sorted: `stable_pair_id(a, b) = f"{min(a,b)}_{max(a,b)}"`. Any external lookup by pair must use this convention.

**Ref:** `src/utils.py:25-27`

### Caching Behavior
- **Profile hash:** SHA-256 of profile text content (first 16 hex chars). Drives extraction + HyDE caching.
- **LLM cache:** File-based in `data/{group}/cache/llm/`. Cache key format varies by step.
- **Embedding cache:** Reuses if user set + section names match exactly.
- **`--force` flag:** Bypasses ALL caches. Use when config changes require full re-computation.

### Minimum Group Size
Hard minimum of **4 profiles**. Pipeline returns early with error if fewer are provided.

**Ref:** `main.py:192-205`

### Active Sections
Only sections with `active: true` in `section_prompt.yaml` are extracted and embedded. This determines the dimensionality of the embedding tensor and which section_weights are valid.

**Ref:** `src/utils.py:101-104` — `filter_active_sections()`

### LLM Provider
Uses the **OpenAI Responses API** (`openai.responses.create()`) for completions and **LiteLLM** for embeddings. The LLM wrapper handles JSON extraction, retry with exponential backoff, and file-based caching.

**Ref:** `src/llm.py` (completions), `src/embed.py:15-58` (embeddings via LiteLLM)

### Edge Serialization
`Edge.to_dict()` produces a clean JSON-serializable dict. The `matches` field in the return value contains raw Edge dataclass instances (not dicts). The `cohort_summary` is already fully serializable.

**Ref:** `src/match.py:27-40`

---

## 7. Critical Files Reference

| File | What to check when wrapping |
|------|-----------------------------|
| `main.py:496-523` | `run_matching_pipeline()` — primary programmatic API |
| `main.py:158-465` | `_execute_matching_pipeline()` — full pipeline logic |
| `main.py:94-110` | `apply_group_overrides()` — how IO paths are derived from group name |
| `deploy_modal.py:185-343` | Modal entry point — shows how to pass profiles as JSON + get cohort_summary back |
| `src/ingest.py` | Profile loading contract (`.txt` files, stem = user ID) |
| `src/report.py:107-180` | `create_cohort_summary()` — schema of the main output |
| `src/match.py:14-40` | `Edge` dataclass — shape of individual match results |
| `src/extract.py:13-18` | `ExtractedSections` dataclass — shape of extracted profile data |
| `config/config.yaml` | Full config schema with all tuneable parameters |
| `config/section_prompt.yaml` | Active sections definition (drives embedding dimensions) |
