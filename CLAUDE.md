# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Choreo is an AI-powered profile matching system — a **library of matchmaking
compute**, not a database. It extracts structured sections from free-text
profiles, embeds them, and computes **directional** "who can help whom"
similarity (one person's `needs` vs. another's `skills`), then refines with
LLM pair scoring to produce mutually useful connections. Behavior is
config-driven — switching matching modes requires no code edits.

Every stage is a pure transform with a declared IO schema; all persistence
lives in adapters (CLI/FileStore in this repo, or an external app's
own store). Three trigger shapes are supported: the classic full cohort run
(N×N), **query match** (1×M hot path: "find me a CTO who…"), and **subset
batch match** (M members × N pool, novel pairs only). Deep dives:
[docs/reference/stages_and_adapters.md](docs/reference/stages_and_adapters.md)
and [docs/reference/matching_modes.md](docs/reference/matching_modes.md).

Choreo is an installable, unopinionated package: external apps add it as a
dependency (`uv add <path-or-git>`; core deps are just numpy/openai/pyyaml/
dotenv — plotting lives behind the `choreo[plots]` extra) and import
`from choreo import run_query_match, load_config, …`. Defaults (config +
prompts) ship inside the package (`choreo/defaults/`), so it works from any
cwd with zero setup; everything is overridable per use-case (see
Configuration).

## Commands

This project is managed with [uv](https://docs.astral.sh/uv/). `uv sync` creates the
`.venv` from `pyproject.toml` + `uv.lock`; prefix commands with `uv run` to use it.

```bash
# Install / sync the environment (reads pyproject.toml + uv.lock)
uv sync

# Full cohort matching. Two input modes:

# (a) Folder mode — point at any folder of profile .txt files; the group name is
#     derived from the folder name and all artifacts go inside it
#     (<folder>/processed, <folder>/embeds, <folder>/outputs, <folder>/cache).
uv run python main.py --input /path/to/folder --force

# (b) Group mode — reads data/<group>/raw, writes to data/<group>/.
uv run python main.py --group <group_name> --force

# Query match (Mode B): rank the group's pre-built pool against one query
uv run python main.py --pipeline query_match --group test4 \
    --query '{"needs": "someone who can build the agent backend"}'   # or raw text

# Subset batch match (Mode C): members × pool, excluding recently surfaced pairs
uv run python main.py --pipeline batch_match --group test4 --members alice,bob

uv run python main.py --list-pipelines     # show registered pipelines

# Config overrides (see Configuration): overlay a config dir and/or set
# individual values by dotted path — applies to every pipeline.
uv run python main.py --group test4 --config-dir my_overrides/ \
    --set query.top_k=3 --set models.pair_llm=openai/gpt-5

# Tests (offline by default — fake LLM + embedder, no API key needed)
uv run pytest
RUN_LLM_TESTS=1 uv run pytest tests/test_e2e_regression.py   # live golden e2e (needs caches + key)

```

`--force` re-runs every step, ignoring caches. Without it, unchanged profiles/sections/embeddings are reused.

## Architecture

Three layers (see [docs/reference/stages_and_adapters.md](docs/reference/stages_and_adapters.md)):

```
Adapters (own ALL IO)              Orchestration (choreo/runners.py)    Core stages (pure transforms)
  main.py (CLI + FileStore)    →     run_full_match()            →     extract · hyde · embed ·
  Neon wrapper (external app)        run_query_match()                 similarity (rectangular) ·
                                     run_batch_match()                 score · match · introduce · report
```

- **Schemas** (`choreo/schemas.py`): every stage's IO is a dataclass with
  `to_dict`/`from_dict` (`ExtractedSections`, `EmbeddingsBundle`, `Edge`, …).
- **Stage registry** (`choreo/stages.py`): `describe_stage(name)` returns each
  stage's IO contract at runtime; each stage has `load`/`dump` disk helpers so
  stages chain **in-memory or via disk** — both first-class.
- **Store protocol** (`choreo/store.py`): `FileStore` is the reference adapter
  (the historical disk layout + `match_history.jsonl`); an external app
  implements the same protocol against its own DB. Embedding always happens
  in-repo; external stores only hold the bundle.
- **Entry at any stage**: raw `Profile`s, pre-sectioned input
  (`schemas.sections_from_dict`), or a pre-built `EmbeddingsBundle`.

The full cohort run: ingest → extract → HyDE → embed → similarity (square) →
LLM pair scoring → greedy b-matching → intros → report data (all inside
`run_full_match`), then adapter-side t-SNE, score-correlation plots, report
writing and similarity plots (`main.py`).

### Directional matching (the core idea)

- **Same-section similarity** is symmetric (e.g. `project`↔`project`).
- **Cross-section similarity is asymmetric**: `cross[i][j]` = "how well can j's
  skills address i's needs". Similarity is rectangular at the core
  (source set × target set, `candidate.py`); the square cohort path is the
  special case with no target passed and reduces **bit-exactly** to the legacy
  behavior. Symmetrization (`(dir+dir.T)/2`) happens only on the square path.
- **HyDE bridges the vocabulary gap**: a need ("make my installation respond
  to movement") is rewritten by an LLM into skill-vocabulary text, so it
  embeds close to the matching skills. `n_descriptors > 1` is supported
  end-to-end (max-pool over descriptor pairs).
- **Absent sections are neutral, not zero** — empty sections AND the
  `"Not specified"` extraction placeholder (`utils.is_absent`) embed to zero
  vectors, skip HyDE entirely, and are masked out of the per-pair fusion. This
  is what lets a query that only fills `needs` drop into the same machinery
  ([docs/reference/matching_modes.md](docs/reference/matching_modes.md)).

### Key Data Flow

- Input: `<raw_dir>/*.txt` (one file per user, filename = user ID)
- Processing: `<processed_dir>/` (extracted sections + `hyde/*.jsonl`, cached)
- Embeddings: `<embeds_dir>/` (vectors.npz, ids.json, section_names.json,
  hyde_vectors.npz, **bundle_meta.json** — provenance + per-cell content hashes)
- Cache: `<cache_dir>/llm/` (LLM call cache)
- Match history: `<base>/match_history.jsonl` (append-only; novelty input for batch mode)
- Output: `<outputs_dir>/` (per-user reports, cohort.json, cost_report.json,
  plots/, plots/raw_data/; batch-mode reports go to `<outputs_dir>/batch/`)
- Raw plot data: `<outputs_dir>/plots/raw_data/` — `.npz` dumps of the arrays
  behind each plot, written by `choreo/raw_data.py` (crash-safe), so plots can be
  re-exported without re-embedding.

Paths resolve to `data/<group>/{raw,processed,embeds,outputs,cache}` in group mode, or `<input_dir>/{...}` in folder mode (see `apply_io_overrides` in `main.py`).

### Core Modules (choreo/)

| Module | Purpose |
|--------|---------|
| `schemas.py` | Typed IO dataclasses for every stage (+ `sections_from_dict`) |
| `stages.py` | Stage registry: `describe_stage`, per-stage `load`/`dump` |
| `store.py` | `Store` protocol + `FileStore` reference adapter |
| `runners.py` | Public mode runners: `run_full_match` / `run_query_match` / `run_batch_match` |
| `query.py` | Mode B: query-as-partial-profile, LLM re-rank, JSON wrapper |
| `batch_match.py` | Mode C: rectangular selection, novelty exclusions |
| `ingest.py` | Load + hash raw `.txt` profiles |
| `extract.py` | LLM section extraction (pure transform + disk wrapper) |
| `hyde.py` | HyDE descriptors (pure transform + disk-cache wrapper) |
| `embed.py` | Embeddings: content-hash incremental reuse, MRL truncation |
| `candidate.py` | Rectangular + square fused similarity |
| `score.py` | Batched, budgeted LLM pair scoring (`excluded_pairs` aware) |
| `match.py` | Greedy b-matching (symmetric or asymmetric member/pool caps) |
| `introduction.py` | Directional intros + starter topics per matched pair |
| `report.py` | Pure `build_report_data` + `write_reports` adapter |
| `llm.py` | OpenRouter wrapper: caching, async batching, cost tracking |
| `config.py` | Config layering: packaged defaults ← config dir ← overrides dict |
| `tsne.py`, `visualize_similarity.py`, `score_correlation.py`, `raw_data.py` | Plots |
| `cost_tracker.py`, `utils.py` | Cost accounting; YAML/JSON IO, hashing, cosine |

## Configuration

The canonical config lives **inside the package**: `choreo/defaults/config.yaml`
plus the four prompt yamls. Nothing is cwd-relative. Three override layers
stack on top (lowest → highest), implemented in `choreo/config.py`:

1. **Packaged defaults** — `choreo/defaults/` (edit these in the checkout for
   local experiments; they're the single source of truth).
2. **Config dir** — `load_config(config_dir=…)` / CLI `--config-dir`: a folder
   with any subset of the five files. `config.yaml` is deep-merged (specify
   only changed keys); prompt yamls replace the packaged ones wholesale.
3. **Overrides dict** — `load_config(overrides={…})` / CLI repeated
   `--set dotted.key=value`: per-call dynamic values (this is what an external
   tool/MCP server passes through).

Prompt paths resolve via `resolve_prompt_paths(config_dir=…, config=…)` with
the same precedence (plus explicit `prompt_files:`/`prompts:` keys in the
config dict as a final escape hatch). One layer above paths, the runners
resolve prompt **content** via `resolve_prompt_templates(config_dir=…,
config=…, prompt_paths=…)` — inline text in the config dict
(`prompts.<name>_prompt_text`, e.g. `scoring_prompt_text`) takes precedence
over any file, so an external app can carry fully custom prompts in its DB
and pass them per call with no files at request time.

All LLM and embedding calls route through **OpenRouter** (OpenAI-compatible endpoint, via the `openai` SDK in `llm.py`). Models use OpenRouter slugs (`provider/model`); swap providers or per-phase models by editing the strings.

```yaml
models:
  embedding: "google/gemini-embedding-2-preview"
  embedding_dimensions: 1536  # MRL truncation; null = full native size (3072)
  extraction_llm: "minimax/minimax-m3"
  pair_llm: "minimax/minimax-m3"
  reasoning_effort: "low"           # global default for every phase
  pair_reasoning_effort: "low"      # pair scoring + query re-rank override

instruction_prompt:
  goal: "…"                    # matching goal injected into every prompt
  language: null               # pin output language of embedded artifacts
                               # (sections + HyDE); null = match each profile

hyde:
  n_descriptors: 1             # HyDE phrasings per source section

recipe:
  section_weights:             # same-section (symmetric); negative = dissimilarity preferred
    skills:  -0.10
    vision:   0.35
    project:  0.25
    needs:   -0.10
  cross_section_weights:       # cross-section (DIRECTIONAL); "<source>_<target>"
    needs_skills: 0.80

blending:
  embed_weight: 0.35
  llm_weight:   0.65           # LLM scores dominate final ranking

matching:
  b_min: 3                     # min connections per user (member side in batch mode)
  b_max: 4                     # max connections per user
  min_profiles_required: 2
  pool_b_max: null             # batch mode: optional pool-side degree cap
  novelty_window_months: 6     # batch mode: exclusion window for past matches

query:                         # Mode B defaults
  top_k: 4
  llm_rerank: true             # false = pure-embedding, cheaper
  # instruction: "…"           # query-mode override of recipe.instruction for
                               # the re-rank prompt (recipe.instruction is
                               # usually pair-framed prose that fights the
                               # directional query_scoring template); unset =
                               # fall back to recipe.instruction
  rerank_pool_multiplier: 4    # over-fetch: LLM re-ranks top_k*4 candidates,
                               # returns top_k (1 = legacy reorder-only)
  generate_intros: true        # true | int top-N | false
  rerank_max_retries: 0        # serial re-ask rounds for scores the model
                               # silently dropped from a chunk response. 0 =
                               # never: the wave is already dispatched, so each
                               # round is a whole extra round-trip to rescue ~1
                               # candidate; unscored ones drop out instead.
  rerank_deadline_s: null      # wall-clock budget for the re-rank wave; null =
                               # wait for every call. Set it to stop paying the
                               # pair model's tail — stragglers are cancelled
                               # (their tokens are still billed) and their
                               # candidates go unscored. Size from p50/p75, and
                               # leave room for >> top_k to land or the re-rank
                               # stops being selective.
  # NO packaged query.recipe: explicit-mapping queries should pass a per-call
  # recipe_override (same-section weights, empty cross = no query-path HyDE);
  # without one, queries fall back to the top-level `recipe`.
  # Prompt count == top_k exactly (ceil(top_k*multiplier / (chunk width)), and
  # chunk width == n_profiles_to_score_together - 1 == multiplier by default),
  # so concurrency.max_concurrent_llm_calls binds only above top_k. An unscored
  # candidate is DROPPED from the shortlist, never embed-ranked into it — the
  # over-fetch pool is the buffer that makes that safe (below top_k scored, it
  # falls back to embed-only rather than returning a short list).

budgets:
  max_pair_llm_calls: 1600
  max_n_llm_evaluations_per_profile: 24
  n_profiles_to_score_together: 5

concurrency:
  max_concurrent_llm_calls: 16   # single global cap on in-flight LLM calls
```

Prompts in `choreo/defaults/`:
- `section_prompt.yaml` — section definitions; each has an `active` flag (only active sections are extracted) and a `guideline`
- `hyde_prompt.yaml` — HyDE descriptor generation
- `scoring_prompt.yaml` — LLM pair scoring: `pair_scoring` (mutual, cohort/batch)
  plus the optional `query_scoring` key (directional Mode-B re-rank: candidate →
  query need, reciprocity off). Custom scoring prompts without `query_scoring`
  govern both paths (query mode falls back to the pair template); inline
  override key: `prompts.query_scoring_prompt_text`.
- `introduction_prompt.yaml` — directional introduction generation

## Key Patterns

**Pair IDs**: alphabetically sorted for stability (`alice_bob`, not `bob_alice`) — `utils.stable_pair_id()`.

**Caching**: hash-based change detection at extraction, HyDE, and embedding
steps. Embedding reuse is **content-hash addressed per (user, section) cell**
— adding/removing one user never re-embeds anyone else. Pre-refactor embeds
dirs (no `bundle_meta.json`) are adopted on first load when the roster matches.
LLM response caches (scoring, intros, query rerank) key on a **hash of the
full prompt**, so edited profile content invalidates automatically — never key
on roster/pair-id alone (a past bug: edited profiles silently replayed stale
scores). HyDE cache keys fold in a **prompt-context fingerprint**
(`hyde.hyde_context_fingerprint`: template + goal + model + language +
guidelines), so editing `hyde_prompt.yaml` or the goal regenerates descriptors
instead of replaying stale ones. Cache keys must use `utils.hash_text` (sha256), never the builtin
`hash()` (salted per process — entries would never hit across runs). Same trap with
`set` iteration order: anything that feeds an LLM prompt or cache key (scoring
group composition, b-matching backfill order) must iterate `sorted(...)`, or
grouping changes every process and the cache never hits
(`test_profile_grouping_deterministic_across_hash_seeds`).

**Timestamps (`last_updated_at`)**: every input/derived artifact carries an
optional ISO-8601 freshness timestamp — `Profile` (file mtime locally, or
caller-supplied), `ExtractedSections`, and per-user on the bundle
(`user_timestamps`, persisted in `bundle_meta.json`). Content hashes remain the
*internal* invalidation mechanism; timestamps are the **adapter-level**
freshness signal (`utils.is_stale(artifact_ts, source_ts)`) so an external
store (Neon `updated_at`) can pass its timestamps in
(`sections_from_dict(..., last_updated_at=…)` accepts
`{"text": …, "last_updated_at": …}` values) and compare them coming back out.

**HyDE gating**: the HyDE step runs *only* when `recipe.cross_section_weights` is non-empty. Empty → fully symmetric mode (no HyDE), backward-compatible with social-connectivity matching.

**Cross-section key parsing**: keys are `"<source>_<target>"` split on the single `_` (preferred alternative: `"<source>-><target>"`). Multi-word section names need the `->` form.

**MRL truncation**: only applied to models in `MRL_CAPABLE_MODELS` (`embed.py`); other models keep full dims with a warning. Full vectors are always stored on disk, so the truncation size is re-tunable without re-embedding.

**Batching**: scoring evaluates `n_profiles_to_score_together` profiles per LLM call, generating N*(N-1)/2 pairs.

**Concurrency**: every batched phase (extract, HyDE, scoring, query re-rank,
intros) routes through `LLMWrapper.batch_json_complete`, which is semaphore-gated
by `concurrency.max_concurrent_llm_calls` (default 16) — exactly that many calls
are in flight at once and the next fires the instant one returns (continuous
dispatch, not fire-a-window-then-await). One global knob; no per-stage batch sizes.

**Blending**: `final = embed_weight * embed_score + llm_weight * llm_score`.
Score normalization takes the reference distribution as an explicit input
(`utils.prepare_normalized_scores(reference_scores=…)`); only the legacy square
path derives it from the current matrix.

**Display names + scoring aliases**: all three runners accept
`display_names={user_id: name}` — names go into scoring/intro prompt prose
(and query mode's `"__query__"` pseudo-user can be named); every returned
field stays keyed by real id. Inside SCORING prompts, profiles and the pair
keys the model must echo are per-prompt aliases (`Q` for the query, `P1`/`P2`/…
in roster order — `build_batch_scoring_prompt` returns the `alias_of` map,
`get_pair_score` parses responses through it, with raw-id keys as fallback):
raw uuids in pair keys were attention noise and one transcription slip
silently dropped the candidate. Ids without a display name fall back to the
raw id as the profile's `name=` label, so pass display_names whenever ids are
opaque. Raw ids never appear in scoring pair keys or the JSON hint.

## Switching matching modes

To move between need/skill matching and symmetric social-connectivity matching, edit config only — no code changes (either in `choreo/defaults/` directly, or in a `--config-dir` overlay):
1. `section_prompt.yaml`: flip `active` flags on the relevant sections.
2. `config.yaml`: adjust `section_weights`; set/clear `cross_section_weights` (empty disables HyDE and directionality).
3. Swap `scoring_prompt.yaml` / `introduction_prompt.yaml` if the framing changes.

## Docs

`docs/` follows the TODO / reference / finished workflow. Current reference
docs: [stages_and_adapters.md](docs/reference/stages_and_adapters.md) (stage
contracts, bundle, FileStore, adapters),
[matching_modes.md](docs/reference/matching_modes.md) (cohort / query / batch
semantics). `choreo_IO.md` (repo root) is the external-integration IO spec.

## Environment

Requires `OPENROUTER_API_KEY` in the environment — for local runs, a `.env`
at the repo root:
```
OPENROUTER_API_KEY=sk-or-...
```
Get a key at https://openrouter.ai/settings/keys. External apps importing
choreo just set the env var (hosted wrappers: via their secret store); `llm.py` reads
it lazily at first client construction, no other key plumbing exists.
