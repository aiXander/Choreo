# Choreo: AI-Powered Directional Profile Matching

Choreo is a **library of matchmaking compute**, not a database. It extracts
structured sections from free-text profiles, embeds them, and computes
**directional** "who can help whom" similarity (one person's `needs` vs.
another's `skills`), then refines with LLM pair scoring to produce mutually
useful connections. All behavior is config-driven — switching matching modes
requires no code edits.

Every stage is a pure transform with a declared IO schema; all persistence
lives in adapters (the CLI + `FileStore` in this repo, or an external
app's own store).

## Three matching modes

| Mode | Shape | Trigger | Use case |
|------|-------|---------|----------|
| **Full cohort** | N×N | `main.py --group <g>` | match everyone in a community against everyone |
| **Query match** (B) | 1×M | `--pipeline query_match --query '…'` | hot path: "find me a CTO who…" against a pre-built pool |
| **Batch match** (C) | M×N | `--pipeline batch_match --members a,b` | periodic re-matching of a subset, surfacing only **novel** pairs |

Semantics in [docs/reference/matching_modes.md](docs/reference/matching_modes.md).

## Features

- **Directional cross-matching**: asymmetric need-to-skill similarity — "how well can B help A?" is computed independently from "how well can A help B?"
- **HyDE vocabulary bridging**: an LLM rewrites each need ("make my installation respond to movement") into skill-vocabulary text so it embeds close to the matching skills
- **Pure stages, pluggable IO**: every stage (extract · hyde · embed · similarity · score · match · introduce · report) is a pure transform with a discoverable IO contract; chain them in-memory or via disk
- **Incremental by content hash**: embedding reuse is addressed per (user, section) cell — adding or editing one profile never re-embeds anyone else
- **Novelty-aware batch matching**: append-only match history excludes recently surfaced pairs within a configurable window
- **Multi-signal blending**: directional embedding similarity fused with batched, budgeted LLM pair scoring
- **B-matching**: fair degree distribution (`b_min`–`b_max` connections per user, asymmetric member/pool caps in batch mode)
- **Config-only mode switching**: flip between need/skill matching and symmetric social-connectivity matching by editing YAML

## Quick Start

1. **Setup**
   ```bash
   cp .env.example .env   # add your OpenRouter key (https://openrouter.ai/settings/keys)
   uv sync                # creates .venv from pyproject.toml + uv.lock
   ```

2. **Add profiles** — one `.txt` per user, filename = user ID (`alice.txt` → "alice"):
   - **Folder mode**: any folder, via `--input` (artifacts written inside it)
   - **Group mode**: `data/<group>/raw/`, via `--group`

3. **Run**
   ```bash
   # Full cohort
   uv run python main.py --input /path/to/folder --force
   uv run python main.py --group <group_name> --force

   # Query match (pool must exist first): JSON section mapping or raw text
   uv run python main.py --pipeline query_match --group <g> \
       --query '{"needs": "someone who can build the agent backend"}'

   # Subset batch match (novel pairs only)
   uv run python main.py --pipeline batch_match --group <g> --members alice,bob

   uv run python main.py --list-pipelines
   ```

4. **Results** (under `data/<group>/outputs/` or `<folder>/outputs/`):
   - Per-user reports: `outputs/<user_id>.json` (`{"profile": md, "matches": md}`)
   - Cohort summary + costs: `outputs/cohort.json`, `outputs/cost_report.json`
   - Plots: `outputs/plots/` (+ re-exportable raw arrays in `plots/raw_data/`)
   - Batch-mode reports go to `outputs/batch/` (never clobber the full run)

5. **Tests** (offline by default — fake LLM + embedder, no API key needed)
   ```bash
   uv run pytest
   RUN_LLM_TESTS=1 uv run pytest tests/test_e2e_regression.py   # live golden e2e
   ```

## Architecture

Three layers — adapters own all IO, stages stay pure:

```
Adapters (own ALL IO)              Orchestration (choreo/runners.py)    Core stages (pure transforms)
  main.py (CLI + FileStore)    →     run_full_match()            →     extract · hyde · embed ·
  external app (own store)           run_query_match()                 similarity (rectangular) ·
                                     run_batch_match()                 score · match · introduce · report
```

- **Schemas** (`choreo/schemas.py`): every stage's IO is a dataclass with
  `to_dict`/`from_dict` (`ExtractedSections`, `EmbeddingsBundle`, `Edge`, …)
- **Stage registry** (`choreo/stages.py`): `describe_stage(name)` returns each
  stage's IO contract at runtime; per-stage `load`/`dump` helpers let stages
  chain in-memory **or** via disk
- **Store protocol** (`choreo/store.py`): `FileStore` is the reference adapter;
  an external app implements the same protocol against its own DB
- **Entry at any stage**: raw `Profile`s, pre-sectioned input, or a pre-built
  `EmbeddingsBundle`

Deep dive: [docs/reference/stages_and_adapters.md](docs/reference/stages_and_adapters.md).
Full IO spec for wrapping Choreo as an external tool: [choreo_IO.md](choreo_IO.md).

```python
# Programmatic API (what external apps import)
from choreo import run_full_match, run_query_match, run_batch_match, FileStore, load_config

run_full_match(profiles, config, store=FileStore("data/grp"))     # or sections, or a bundle
run_query_match({"needs": "a CTO great at agents"}, pool_bundle, config)
run_batch_match(["alice", "bob"], pool_bundle, config,
                excluded_pairs=store.get_match_history(window_months=6),
                pool_sections={s.id: s.sections for s in store.get_sections()})
```

## How the matching works

1. **Ingest** — load `.txt` profiles, content-hash for change detection
2. **Extract** — LLM pulls structured sections (default: `skills`, `vision`, `project`, `needs`; controlled by `active` flags in `section_prompt.yaml`)
3. **HyDE** — for each cross-section weight (e.g. `needs_skills`), an LLM rewrites the source section into target-section vocabulary; runs **only** when `cross_section_weights` is non-empty
4. **Embed** — sections → `(n_users, n_sections, dim)` tensor + HyDE tensors; full-size vectors stored on disk, MRL-truncated at compute time (re-tunable without re-embedding)
5. **Similarity** — rectangular at the core (source set × target set): same-section cosine (symmetric) + HyDE-bridged cross-section (asymmetric: `cross[i][j]` = "how well can j's skills address i's needs"), weight-fused into a directional matrix; the square cohort path symmetrizes `(dir + dir.T) / 2` for selection. Absent sections are neutral (masked), not zero — which is what lets a needs-only query drop into the same machinery
6. **Score** — batched LLM pair scoring (N profiles per call → N·(N−1)/2 pairs), budgeted, novelty-aware
7. **Match** — greedy b-matching on blended scores (`final = embed_weight·embed + llm_weight·llm`)
8. **Introduce + report** — directional intros (what each person offers the *other's* project) + per-user reports, cohort summary, cost report, plots

## Configuration

The canonical config ships inside the package (`choreo/defaults/config.yaml` +
four prompt yamls; annotated schema in [CLAUDE.md](CLAUDE.md)). Override per
use-case with a config dir (`--config-dir` / `load_config(config_dir=…)`:
config.yaml deep-merges, prompt files replace) and/or per-call overrides
(`--set dotted.key=value` / `load_config(overrides={…})`). All models are
OpenRouter slugs (`provider/model`):

```yaml
models:
  embedding: "google/gemini-embedding-2-preview"
  embedding_dimensions: 1536   # MRL truncation; null = full native size (3072)
  extraction_llm: "deepseek/deepseek-v4-flash-0731"
  pair_llm: "deepseek/deepseek-v4-flash-0731"
  reasoning_effort: "low"      # global default; pair_reasoning_effort overrides pair scoring

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
  llm_weight:   0.65

matching:
  b_min: 3                     # min connections per user (member side in batch mode)
  b_max: 4
  pool_b_max: null             # batch mode: optional pool-side degree cap
  novelty_window_months: 6     # batch mode: exclusion window for past matches

query:                         # Mode B defaults
  top_k: 5
  llm_rerank: true             # false = pure-embedding, cheaper
```

Prompt files (packaged in `choreo/defaults/`, overridable via a config dir or
inline `prompts.<name>_prompt_text`): `section_prompt.yaml` (section
definitions with `active` flags), `hyde_prompt.yaml`, `scoring_prompt.yaml`,
`introduction_prompt.yaml`. The templates are use-case-neutral by design —
deployment flavor belongs in `instruction_prompt.goal` + `recipe.instruction`.

### Switching matching modes (config only)

To move between need/skill matching and symmetric social-connectivity matching:

1. `section_prompt.yaml`: flip `active` flags on the relevant sections
2. `config.yaml`: adjust `section_weights`; set/clear `cross_section_weights`
   (empty disables HyDE and directionality entirely)
3. Swap `scoring_prompt.yaml` / `introduction_prompt.yaml` if the framing changes

## Module map (`choreo/`)

| Module | Purpose |
|--------|---------|
| `schemas.py` | Typed IO dataclasses for every stage |
| `stages.py` | Stage registry: `describe_stage`, per-stage `load`/`dump` |
| `store.py` | `Store` protocol + `FileStore` reference adapter |
| `runners.py` | Public mode runners: full / query / batch |
| `query.py` · `batch_match.py` | Mode B and Mode C logic |
| `ingest.py` · `extract.py` · `hyde.py` · `embed.py` | Profile → sections → descriptors → embeddings |
| `candidate.py` | Rectangular + square fused similarity |
| `score.py` · `match.py` | LLM pair scoring · greedy b-matching |
| `introduction.py` · `report.py` | Directional intros · report data + writing |
| `llm.py` · `cost_tracker.py` | OpenRouter wrapper (caching, batching) · cost accounting |
| `tsne.py` · `visualize_similarity.py` · `score_correlation.py` · `raw_data.py` | Plots + raw plot data |

## Key design decisions

- **Pair IDs** are alphabetically sorted for stability (`alice_bob`, never `bob_alice`) — `utils.stable_pair_id()`
- **Embedding ownership stays in-repo**: external stores hold the bundle (`EmbeddingsBundle.to_dict()`) and hand it back; bundles carry `embedding_model` + `dim` provenance, and a model mismatch raises or triggers a full re-embed
- **Determinism is load-bearing**: anything feeding an LLM prompt or cache key iterates in sorted order; cache keys use sha256 (`utils.hash_text`), never the salted builtin `hash()`
- **Directionality preserved at the core**: the asymmetric matrix is never symmetrized during computation — symmetrization is an aggregation choice on the square cohort path only
- **Backward compatible**: empty `cross_section_weights` → no HyDE, fully symmetric mode

## Requirements

- Python ≥ 3.11, managed with [uv](https://docs.astral.sh/uv/)
- OpenRouter API key (in `.env` as `OPENROUTER_API_KEY`)

## Embedding Choreo in your app

Choreo is a library, not a service — deploy it inside your own app. Implement
the `Store` protocol over your persistence (or reuse `FileStore`), layer your
deployment's framing over the packaged defaults with
`load_config(overrides=…)`, and call the runners. The full wrapping contract
is [choreo_IO.md](choreo_IO.md) §1.4.
