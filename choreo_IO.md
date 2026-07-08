# Choreo Pipeline IO Specification

Reference for wrapping Choreo as an external tool: all entry points, inputs,
outputs, filesystem artifacts and the programmatic API surface. Architecture
rationale lives in
[docs/reference/stages_and_adapters.md](docs/reference/stages_and_adapters.md)
and [docs/reference/matching_modes.md](docs/reference/matching_modes.md).

---

## 1. Entry Points

### 1.1 CLI (`main.py`) — three registered pipelines

```bash
uv run python main.py --group <group> [--force]                         # matching (full N×N)
uv run python main.py --input /path/to/folder [--force]                 # matching, folder mode
uv run python main.py --pipeline query_match --group <g> --query '…'    # Mode B (1×M)
uv run python main.py --pipeline batch_match --group <g> --members a,b  # Mode C (M×N, novel)
uv run python main.py --list-pipelines
```

- `--group`: data isolation under `data/<group>/…`. `--input`: any folder of
  `.txt` profiles (artifacts written inside it). `--force`: bypass all caches.
- `--query`: raw text, or a JSON section mapping (`'{"needs": "…"}'`).
- `--members`: comma-separated member ids for the batch side.
- `--config-dir <dir>`: overlay any subset of the five config/prompt yamls over
  the packaged defaults (`choreo/defaults/`). `--set dotted.key=value`
  (repeatable): override single config values on top. See §2.2.

### 1.2 Programmatic — the mode runners (`choreo/runners.py`)

The public API external apps import. Schema objects in, schema objects out;
an optional `FileStore` adds disk caching/persistence but is never required.

```python
from choreo import run_full_match, run_query_match, run_batch_match
from choreo import sections_from_dict, EmbeddingsBundle, FileStore
from choreo.config import load_config, resolve_prompt_paths, resolve_prompt_templates

config = load_config(config_dir=..., overrides={...})   # packaged defaults ← dir ← dict

# Full cohort — enter at ANY stage:
run_full_match(profiles, config, store=FileStore("data/grp"))      # raw text
run_full_match(sections_from_dict({uid: {sec: txt}}), config)      # pre-sectioned
run_full_match(bundle, config, sections=sections,                  # pre-embedded
               display_names={uid: "Name"})
# -> {edges, report_data, embeddings (bundle), llm_scores, introductions,
#     similarity{dir_matrix, sym_matrix, user_ids, matrices_dict}, …}

# Query (Mode B) — pool comes in as an argument, never re-embedded:
run_query_match({"needs": "a CTO great at agents"}, pool_bundle, config,
                pool_sections=…, recipe_override=…, top_k=…, llm_rerank=…,
                exclude_ids={...},                     # asker + novelty exclusions
                display_names={uid: "Name", "__query__": "Asker Name"},
                generate_intros=True)                  # True | int top-N | False
# -> QueryMatchResult{shortlist, query_sections, recipe, notes, …}

# Batch (Mode C) — members + history are caller-supplied:
run_batch_match(member_ids, pool_bundle, config,
                excluded_pairs=store.get_match_history(window_months=6),
                pool_sections=…, display_names={uid: "Name"})
# -> BatchMatchResult{edges, report_data (members only), new_pairs, …}
```

`display_names` (all three runners, optional): `{user_id: human name}` —
names are woven into the scoring/re-rank/intro prompts and the intro prose
("For <name>: …") while score JSON and every returned field stay keyed by id.
Pass it whenever ids are opaque (uuids); prose generated from raw ids cannot
be repaired afterwards. In query mode a `"__query__"` entry names the asker.

**Mode-B novelty**: there is deliberately no `excluded_pairs` on
`run_query_match` — for a 1×M query the asker is known, so recent-history
pairs reduce to candidate user ids. Build that set from your match history
(same `matching.novelty_window_months` window) and pass it as `exclude_ids`.

JSON-in/JSON-out wrapper for agent tool-calls:
`choreo.run_query_match_json(payload, config)` — payload takes `query` plus
either `store_dir` (FileStore path) or an inline `pool`
(`EmbeddingsBundle.to_dict()` shape); optional keys `top_k`, `llm_rerank`,
`generate_intros` (bool or int), `recipe_override`, `exclude_ids`,
`display_names`.

### 1.3 Stage-level access (`choreo/stages.py`)

Each stage is individually invokable with a discoverable contract:

```python
from choreo import stages
stages.list_stages()              # extract · hyde · embed · similarity · score · match · introduce · report
stages.describe_stage("embed")    # JSON-serializable input/output schema
spec = stages.get_stage("embed"); spec.run(...); spec.dump(out, path); spec.load(path)
```

`load`/`dump` define each stage's canonical disk format, so stages chain
in-memory **or** via files — both supported.

### 1.4 Embedding Choreo in a host app (the wrapping contract)

Choreo ships as a **library, not a service** — there is no Choreo-owned
deployment. A host app (serverless function, web backend, cron job) wraps it
by composing three things, all specified in this doc:

1. **Persistence** — implement the `Store` protocol (`choreo/store.py`)
   against your own DB, or reuse `FileStore` (§4) on a disk/volume. The store
   holds sections, the embeddings bundle, and match history.
2. **Config** — `load_config(overrides=…)` (§2.2) deep-merges a plain dict
   over the packaged defaults. Deployment flavor rides
   `instruction_prompt.goal` + `recipe.instruction`; whole prompts can be
   replaced inline via `prompts.<name>_prompt_text`.
3. **The runners** (§1.2) — Mode-A stage calls for incremental profile
   upkeep, `run_query_match` (Mode B), `run_batch_match` (Mode C), handing in
   the pool/bundle loaded from your store.

The host owns identity, authorization, scheduling, result delivery and any
warm caching; Choreo owns the matching compute. Bake the `choreo` package
directory into the host's image/venv — `defaults/` ships inside the package,
so `load_config()` works from any cwd. Return-value schemas for tool-shaped
wrappers are in §5; wrapping conventions in §6.

---

## 2. Required Inputs

### 2.1 Profiles
One UTF-8 `.txt` per user, filename stem = user ID (or a `{user_id: text}`
mapping / `sections_from_dict` for pre-sectioned input). Minimum cohort size:
`matching.min_profiles_required` (config; full run only).

### 2.2 Config files
The canonical config ships inside the package at `choreo/defaults/`:
`config.yaml` (models / recipe / blending / matching / budgets / hyde / query /
io) + the four prompt files (`section_prompt.yaml`, `scoring_prompt.yaml`,
`introduction_prompt.yaml`, `hyde_prompt.yaml`). See CLAUDE.md for the
annotated schema. All models are OpenRouter slugs; all LLM/embedding calls go
through OpenRouter (`choreo/llm.py`).

Override layering (`choreo/config.py`), lowest to highest precedence:
1. packaged defaults (`choreo/defaults/`);
2. `config_dir` — a directory holding any subset of the five files
   (`config.yaml` deep-merges; prompt yamls replace wholesale);
3. `overrides` — a dict deep-merged per call
   (`load_config(config_dir=…, overrides={"query": {"top_k": 3}})`).
Prompt paths resolve the same way via `resolve_prompt_paths(config_dir=…,
config=…)`, with explicit `prompt_files:`/`prompts:` keys in the config dict
taking final precedence.

**Inline prompt text** (highest-precedence prompt layer): the config dict can
carry prompt *content* directly — no files at request time — under
`prompts.<name>_prompt_text`:

```python
config = load_config(overrides={"prompts": {
    "scoring_prompt_text": "…{user_profiles_xml_formatted}…{json_format_hint}",
    "introduction_prompt_text": "…", "hyde_prompt_text": "…",
    "section_prompt_text": "<full section-config YAML text or parsed dict>",
}})
```

The runners resolve prompts through `resolve_prompt_templates(config_dir=…,
config=…, prompt_paths=…)` → `{"sections": dict, "scoring"/"introduction"/
"hyde": template str}`; inline text > explicit paths > config_dir files >
packaged defaults. Scoring/intro/re-rank LLM caches key on the full prompt and
the HyDE cache key folds in a prompt-context fingerprint, so switching prompt
text invalidates the affected cached responses automatically.

**Language pinning**: `instruction_prompt.language` (default null = match each
profile's own language) pins the output language of the artifacts that get
embedded (extracted sections + HyDE descriptors) — recommended for
mixed-language communities so cosine similarity stays comparable. Extraction
reuse is keyed on profile content, so switching language on an existing cohort
needs `force`; HyDE picks it up automatically.

### 2.3 Environment
`OPENROUTER_API_KEY` as an environment variable (a `.env` in the calling
process's cwd is auto-loaded via python-dotenv; hosted wrappers inject it via
their own secret store). There is no other key plumbing.

---

## 3. Stage IO contracts

The §1.3 registry is authoritative; summary:

| Stage | Input | Output |
|-------|-------|--------|
| `extract` | `list[Profile]` + sections config (+ optional `existing` `{hash: sections}`) | `list[ExtractedSections{id, sections, hash, last_updated_at?}]` |
| `hyde` | sections + cross weights + `{n_descriptors}` (+ optional `existing`) | `{cross_key: [HydeDescriptors{…, descriptors: list[str]}]}` |
| `embed` | sections (+ HyDE, + optional `existing` bundle) | `EmbeddingsBundle{user_ids, section_names, embeddings[N,S,D], hyde{key:[N,d,D]}, embedding_model, dim, section_hashes, hyde_hashes, user_timestamps}` |
| `similarity` | source bundle + target bundle + recipe | `SimilarityResult{dir_matrix[n_src,n_tgt], sym_matrix?, matrices_dict, source_ids, target_ids}` |
| `score` | sym matrix (or `selected_pairs`) + sections + budgets + `excluded_pairs?` | `{pair_id: PairScore}` |
| `match` | candidates + llm_scores + matching/blending cfg + `member_ids?`/`excluded_pairs?`/`reference_scores?` | `list[Edge]` |
| `introduce` | edges + sections | `{pair_id: Introduction}` |
| `report` | edges + sections (+ `scope_user_ids?`) | `{user_reports: {uid: {profile, matches}}, cohort_summary}` |

Reuse rules: `existing` arguments are how adapters hand cached data in —
extraction by profile hash, HyDE by content-hash cache key, embeddings by
per-(user, section) content hash (roster changes never trigger re-embeds).
An `existing` bundle from a different `embedding_model` is ignored (full
re-embed).

---

## 4. Filesystem Layout (FileStore)

```
data/<group>/
├── raw/*.txt                     # INPUT (filename = user ID)
├── processed/
│   ├── sections.jsonl            # extracted sections (append-only; last row per id wins)
│   └── hyde/<cross_key>.jsonl    # HyDE cache (content-hash keyed)
├── embeds/
│   ├── vectors.npz               # (n_users, n_sections, dim) full-size
│   ├── hyde_vectors.npz          # per cross_key: (n_users, n_descriptors, dim)
│   ├── ids.json · section_names.json
│   └── bundle_meta.json          # provenance + content hashes (absent on legacy dirs — adopted on load)
├── cache/llm/*.json              # LLM response cache (sha256-stable keys)
├── match_history.jsonl           # {pair_id, user1, user2, matched_at} — batch-mode novelty input
└── outputs/
    ├── <user>.json               # {"profile": md, "matches": md}
    ├── cohort.json · cost_report.json
    ├── batch/                    # batch-mode reports (never clobber the full run)
    └── plots/ (+ plots/raw_data/*.npz)
```

---

## 5. Return Value Schemas

### Full run (`_execute_matching_pipeline`)
On success
`{success: True, matches: list[Edge], profiles_count, outputs_dir,
cost_report_path, cohort_summary{overview, degree_distribution,
score_statistics, users}, stats}`; on failure `{success: False, error, …}`.
`Edge.to_dict()` →
`{user1, user2, pair_id, final_weight, embed_score, llm_score,
embed_score_normalized, llm_score_normalized, intro, starter_topics}`.

### Query match (`QueryMatchResult.to_dict()` / `run_query_match_json`)
```python
{"success": True,                      # json wrapper only
 "query_sections": {section: text},
 "shortlist": [{"rank", "user_id", "score", "embed_score",
                 "embed_score_normalized", "llm_score",  # None if rerank off/skipped
                 "intro", "starter_topics"}],  # intro empty beyond generate_intros=N
 "recipe": {...}, "llm_rerank_applied": bool,
 "pool_size": int, "notes": [str]}
```

The shortlist is always at most `top_k` rows; with `llm_rerank` on, the LLM
scored `top_k * query.rerank_pool_multiplier` embedding candidates first
(over-fetch — recovery, not just reorder) and this is the re-ranked top slice.

### Batch match (`BatchMatchResult.to_dict()`)
```python
{"edges": [Edge.to_dict()],
 "report_data": {"user_reports": {member: {...}}, "cohort_summary": {...}},
 "new_pairs": [{"pair_id", "user1", "user2", "final_weight"}],  # append to your history
 "member_ids": [...], "excluded_count": int}
```

---

## 6. Key Conventions for Wrapping

- **Pair IDs** are alphabetically sorted (`stable_pair_id(a, b)` →
  `"alice_bob"`); any external lookup must use this convention — including the
  `excluded_pairs` sets you pass in.
- **Match history is yours.** Choreo accepts `excluded_pairs` (Mode C /
  cohort) and returns `new_pairs`; the documented default window is
  `matching.novelty_window_months` (6) — apply it when building the set. For
  Mode B, map the asker's recent-history partners to `exclude_ids` (§1.2) —
  there is no pair-id mechanism on the query path by design.
- **Absent sections**: empty strings AND the literal `"Not specified"`
  placeholder are absent (`choreo.is_absent`) — they embed to zero vectors,
  skip HyDE entirely, and are masked out of the per-pair fusion as neutral.
  External stores can safely pass either shape for a missing section.
- **Embedding ownership**: always in-repo. Store the bundle
  (`EmbeddingsBundle.to_dict()` or the `embeds/` dir format) and hand it back
  as `existing` / `pool`; never embed externally. Bundles carry
  `embedding_model` + `dim` provenance; a model mismatch raises (query) or
  triggers a full re-embed (embed stage).
- **`--force` / `force=True`** bypasses all caches. Upsert-style wrappers
  should scope `force` to the given profiles only.
- **Caching**: profile-hash for extraction, content-hash for HyDE and
  per-cell embeddings; LLM response caches (scoring/intros/rerank) key on a
  sha256 of the **full prompt**, so edited profile content invalidates them
  automatically. Renaming/toggling sections invalidates embeddings via the
  hashes; use `--force` when in doubt.
- **Freshness timestamps (`last_updated_at`)**: optional ISO-8601 provenance
  carried end-to-end — `Profile` (file mtime locally, or supplied by the
  caller), `ExtractedSections.last_updated_at`,
  `EmbeddingsBundle.user_timestamps` (persisted in `sections.jsonl` /
  `bundle_meta.json`). Inject your store's `updated_at` via
  `sections_from_dict(..., last_updated_at=…)`;
  read it back from the returned objects and use `choreo.is_stale(artifact_ts,
  source_ts)` to decide when a profile needs re-upserting. Content hashes
  remain the internal invalidation mechanism — timestamps never trigger
  recomputation by themselves.
