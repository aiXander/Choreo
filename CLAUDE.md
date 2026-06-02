# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Choreo is an AI-powered profile matching system. It extracts structured sections from free-text profiles, embeds them, and computes **directional** "who can help whom" similarity (one person's `needs` vs. another's `skills`), then refines with LLM pair scoring to produce mutually useful connections. Behavior is config-driven — switching matching modes requires no code edits.

## Commands

This project is managed with [uv](https://docs.astral.sh/uv/). `uv sync` creates the
`.venv` from `pyproject.toml` + `uv.lock`; prefix commands with `uv run` to use it.

```bash
# Install / sync the environment (reads pyproject.toml + uv.lock)
uv sync

# Run the matching pipeline. Two input modes:

# (a) Folder mode — point at any folder of profile .txt files; the group name is
#     derived from the folder name and all artifacts go inside it
#     (<folder>/processed, <folder>/embeds, <folder>/outputs, <folder>/cache).
uv run python main.py --input /path/to/folder --force

# (b) Group mode — reads data/<group>/raw, writes to data/<group>/.
uv run python main.py --group <group_name> --force

# Examples
uv run python main.py --input ~/cohorts/spring_2026   # folder mode
uv run python main.py --group real --force            # group mode, "real"
uv run python main.py --group test4                   # group mode with caching
uv run python main.py --list-pipelines                # show registered pipelines

# Modal deployment (serverless)
uv run modal deploy deploy_modal.py                              # deploy callable function
uv run modal run deploy_modal.py --input-dir data/test4 --force # run a job; downloads outputs_modal.zip locally
uv run modal run deploy_modal.py --profiles-json profiles.json  # or pass {user_id: text} JSON
```

Modal needs a `choreo-secrets` secret holding `OPENROUTER_API_KEY` (add `AWS_*`
keys to it to also push the outputs zip to S3). Outputs persist to the
`choreo-data` Volume; the local entrypoint pulls the zip back automatically. See
the header of `deploy_modal.py` for details.

`--force` re-runs every step, ignoring caches. Without it, unchanged profiles/sections/embeddings are reused.

## Architecture

The matching pipeline (steps as printed in `main.py`):

```
1.   INGEST    → Load .txt profiles from the raw dir (filename = user ID)
2.   EXTRACT   → LLM extracts active sections (skills, vision, project, needs)
2.5. HYDE      → LLM writes "ideal helper" descriptors in target vocabulary
                 (only when recipe.cross_section_weights is set)
3.   EMBED     → Embed each section + HyDE descriptors; optional MRL truncation
3.5. T-SNE     → Per-section cluster plots (non-fatal if it fails)
4.   CANDIDATE → Fuse same-section (symmetric) + cross-section (DIRECTIONAL)
                 similarity; also derive a symmetric matrix for matching
5.   SCORE     → LLM evaluates top candidate pairs (batched, budgeted)
6.   MATCH     → Greedy b-matching on blended embed+LLM scores (b_min/b_max)
7.   INTRO     → Directional introductions + starter topics per matched pair
8.   REPORT    → Per-user markdown + cohort.json
9.   VISUALIZE → Similarity heatmaps + score-correlation plots
```

### Directional matching (the core idea)

- **Same-section similarity** is symmetric (e.g. `project`↔`project`).
- **Cross-section similarity is asymmetric**: `cross[i][j]` = "how well can j's skills address i's needs". This is *not* symmetrized during computation — see `candidate.py:121`.
- **HyDE bridges the vocabulary gap**: a need ("make my installation respond to movement") is rewritten by an LLM into skill-vocabulary text, so it embeds close to the matching skills.
- `generate_similarity_matrix` returns both the directional matrix and a symmetric `(dir + dir.T)/2` matrix. The symmetric matrix drives scoring/matching; the directional matrix is intentionally retained (captured in `main.py` but not yet consumed) for future directional features such as user-centric mode.

### Key Data Flow

- Input: `<raw_dir>/*.txt` (one file per user, filename = user ID)
- Processing: `<processed_dir>/` (extracted sections + `hyde/*.jsonl`, cached)
- Embeddings: `<embeds_dir>/` (embeddings.npy, ids.json, section_names.json)
- Cache: `<cache_dir>/llm/` (LLM call cache)
- Output: `<outputs_dir>/` (per-user reports, cohort.json, cost_report.json, plots/, plots/raw_data/)
- Raw plot data: `<outputs_dir>/plots/raw_data/` — `.npz` dumps of the arrays behind each plot (t-SNE 2D coords, similarity matrices, per-pair scores), each labelled by `user_ids`/`pair_id` so points trace back to users. Written by `src/raw_data.py`, whose save functions are crash-safe (never propagate errors to the pipeline). Lets you edit data (drop outliers, rename labels) and re-export images without re-embedding/re-running t-SNE.

Paths resolve to `data/<group>/{raw,processed,embeds,outputs,cache}` in group mode, or `<input_dir>/{...}` in folder mode (see `apply_io_overrides` in `main.py`).

### Core Modules (src/)

| Module | Purpose |
|--------|---------|
| `ingest.py` | Load + hash raw `.txt` profiles |
| `extract.py` | LLM section extraction (active-section filtering) |
| `hyde.py` | HyDE descriptor generation (needs→skills vocabulary bridge) |
| `embed.py` | Multi-section + HyDE embeddings; MRL truncation helpers |
| `candidate.py` | Directional + symmetric similarity fusion |
| `score.py` | Batched, budgeted LLM pair scoring |
| `match.py` | Greedy b-matching, embed/LLM score blending |
| `introduction.py` | Directional intros + starter topics |
| `report.py` | Per-user markdown + cohort.json |
| `tsne.py`, `visualize_similarity.py`, `score_correlation.py` | Plots |
| `llm.py` | OpenRouter wrapper: caching, async batching, cost tracking |
| `cost_tracker.py` | API cost accounting |
| `utils.py` | YAML/JSON I/O, hashing, `stable_pair_id`, cosine matrix |

## Configuration

Main config: `config/config.yaml`. All LLM and embedding calls route through **OpenRouter** (OpenAI-compatible endpoint, via the `openai` SDK in `llm.py`). Models use OpenRouter slugs (`provider/model`); swap providers or per-phase models by editing the strings.

```yaml
models:
  embedding: "google/gemini-embedding-2-preview"
  embedding_dimensions: 768   # MRL truncation; null = full native size (3072)
  extraction_llm: "google/gemini-3.1-flash-lite"
  pair_llm: "google/gemini-3.1-flash-lite"
  reasoning_effort: "low"      # global default; xhigh|high|medium|low|minimal|none (null = model default). Ignored on non-reasoning models; pair scoring overrides to "medium".

hyde:
  n_descriptors: 1             # HyDE phrasings per source section

recipe:
  section_weights:             # same-section (symmetric); negative = dissimilarity preferred
    skills:  -0.10
    vision:   0.30
    project:  0.10
    needs:    0.00
  cross_section_weights:       # cross-section (DIRECTIONAL); "<source>_<target>"
    needs_skills: 0.80

blending:
  embed_weight: 0.35
  llm_weight:   0.65           # LLM scores dominate final ranking

matching:
  b_min: 2                     # min connections per user
  b_max: 4                     # max connections per user
  min_profiles_required: 2

budgets:
  max_pair_llm_calls: 300
  max_n_llm_evaluations_per_profile: 16
  n_profiles_to_score_together: 4
```

Prompts in `config/`:
- `section_prompt.yaml` — section definitions; each has an `active` flag (only active sections are extracted) and a `guideline`
- `hyde_prompt.yaml` — HyDE descriptor generation
- `scoring_prompt.yaml` — LLM pair scoring
- `introduction_prompt.yaml` — directional introduction generation

## Key Patterns

**Pair IDs**: alphabetically sorted for stability (`alice_bob`, not `bob_alice`) — `utils.stable_pair_id()`.

**Caching**: hash-based change detection at extraction, HyDE, and embedding steps. Embedding cache keys on the (user set, section names), so renaming/toggling sections invalidates old caches — use `--force` when in doubt.

**HyDE gating**: the HyDE step runs *only* when `recipe.cross_section_weights` is non-empty. Empty → fully symmetric mode (no HyDE), backward-compatible with social-connectivity matching.

**Cross-section key parsing**: keys are `"<source>_<target>"` split on the single `_`. Multi-word section names (e.g. `final_project`) would break parsing in both `hyde.py` and `candidate.py`.

**MRL truncation**: only applied to models in `MRL_CAPABLE_MODELS` (`embed.py`); other models keep full dims with a warning. Full vectors are always stored on disk, so the truncation size is re-tunable without re-embedding.

**Batching**: scoring evaluates `n_profiles_to_score_together` profiles per LLM call, generating N*(N-1)/2 pairs.

**Blending**: `final = embed_weight * embed_score + llm_weight * llm_score`.

## Switching matching modes

To move between need/skill matching and symmetric social-connectivity matching, edit config only — no code changes:
1. `section_prompt.yaml`: flip `active` flags on the relevant sections.
2. `config.yaml`: adjust `section_weights`; set/clear `cross_section_weights` (empty disables HyDE and directionality).
3. Swap `scoring_prompt.yaml` / `introduction_prompt.yaml` if the framing changes.

## Environment

Requires `.env` at the repo root:
```
OPENROUTER_API_KEY=sk-or-...
```
Get a key at https://openrouter.ai/settings/keys.
