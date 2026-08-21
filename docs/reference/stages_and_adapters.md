# Stages & adapters: how Choreo's granular architecture fits together

*(Built June 2026 in the "granular, incremental & pluggable" refactor.
Remaining follow-ups:
[../TODO/external_adapter_integration.md](../TODO/external_adapter_integration.md),
[../TODO/scoring_calibration.md](../TODO/scoring_calibration.md).)*

## The one-sentence model

Choreo is a **library of matchmaking compute**: every pipeline stage is a pure
transform with a declared input/output schema, and all persistence lives in
**adapters** that fetch existing data, hand it to the transforms as plain
arguments, and store whatever comes back.

```
Adapters (own ALL IO)                 Orchestration               Core stages (pure)
─────────────────────                 ─────────────               ──────────────────
CLI / FileStore   (main.py)     →     run_full_match()      →     extract · hyde · embed ·
Neon/Postgres     (external app)      run_query_match()           similarity (rectangular) ·
                                      run_batch_match()           score · match · introduce ·
                                      (choreo/runners.py)            report (build_report_data)
```

The external community platform (Neon/Postgres) is the *second* adapter and
lives **outside this repo**. It implements the same `Store` protocol
(`choreo/store.py`) that `FileStore` implements here, and otherwise just imports
the runners. **Embedding always happens inside this repo** — external stores
only hold the embeddings bundle and pass it back; they never embed.

## Why each piece exists

### Schemas (`choreo/schemas.py`) + stage registry (`choreo/stages.py`)

Every stage's currency is a dataclass with `to_dict`/`from_dict`
(`ExtractedSections`, `HydeDescriptors`, `EmbeddingsBundle`,
`SimilarityResult`, `PairScore`, `Edge`, `Introduction`). Plain dataclasses
were chosen over pydantic deliberately — dependency-light; revisit only if the
external caller needs runtime validation.

`stages.describe_stage(name)` returns a JSON-serializable description of any
stage's IO contract, so an external caller never reads Choreo source to learn
what a stage needs. Each stage also has `load`/`dump` helpers defining its
canonical disk format — **in-memory chaining and disk chaining are both
first-class**: a stage's output can be passed as a Python object or dumped to
disk and loaded by the next stage. The adapter-parity test
(`tests/test_adapter_parity.py`) locks the two styles to identical results.

### The embeddings bundle and content-hash reuse

`EmbeddingsBundle` carries `user_ids`/`section_names` (the array axes),
full-size vectors, source-side HyDE arrays, **provenance**
(`embedding_model`, native `dim`) and **per-cell content hashes**
(`section_hashes`, `hyde_hashes`).

The hashes are the key design move: embedding reuse is **content-addressed,
not roster-addressed**. `embed.embed_sections(…, existing=bundle)` re-embeds
only cells whose text hash changed — adding/removing one user never re-embeds
anyone else (the old cache required the exact same roster). A pre-refactor
embeds dir (no `bundle_meta.json`) is *adopted* on first load when the roster
matches (`embed._trust_legacy_bundle`), then upgraded with hashes.

Gotchas:
- A bundle from a different `embedding_model` is ignored entirely (model
  migration is explicitly out of scope; the provenance fields exist so a
  future `model_version` guard can be added without contract changes).
- Full vectors are always stored; MRL truncation happens on working copies at
  computation time (`embed.truncate_embeddings`), so the truncation size stays
  re-tunable without re-embedding.
- `bundle.subset(ids)` is the "get embeddings for an arbitrary subset"
  primitive that query (1×M) and batch (M×N) modes are built on.

### Rectangular similarity (`choreo/candidate.py`)

`compute_fused_similarity_matrix` is rectangular at the core:
`fused[i][j]` = "how well can **target j** help **source i**". Square cohort
mode is the special case where no target is passed — and that path calls the
*literal legacy* `cosine_matrix` because BLAS uses a symmetric kernel for
`A @ A.T` that differs from the rectangular `A @ B.T` by ~1 ULP; only this
keeps the legacy output **bit-exact** (locked by
`tests/test_stage_isolation.py::test_rectangular_reduces_to_square_exactly`).

Two semantics that must survive any future rewrite:
- **Absent section = neutral, not zero**: empty sections embed to zero
  vectors, are masked out of the per-pair fusion, and the denominator is the
  weight mass *actually present* for that pair. This is what lets a query that
  only fills `needs` drop into the same machinery.
- **Directional cross terms are never symmetrized** in rectangular mode;
  `(dir+dir.T)/2` only happens on the legacy square path
  (`generate_similarity_matrix`).
- HyDE supports `n_descriptors > 1` end-to-end (arrays `[N, n_desc, D]`,
  max-pool over descriptor pairs) — config just defaults to 1.

### Store protocol + FileStore (`choreo/store.py`)

`Store` is the contract an external persistence adapter satisfies:
`get/put_sections`, `get/put_embeddings`, `get_match_history`, `put_matches`.
`FileStore` is the reference implementation wrapping the historical
`{raw,processed,embeds,outputs,cache}` layout, plus the new
**`<base>/match_history.jsonl`** (append-only `{pair_id, user1, user2,
matched_at}` rows — the novelty input for batch mode; `base` = parent of
`outputs_dir`).

Nothing in the core stages calls a Store — only runners/adapters do, and only
when one is provided. `run_*` functions all work with `store=None` (fully
in-memory).

### Entry at any stage

- raw text → `list[Profile]` → everything runs;
- pre-sectioned input → `schemas.sections_from_dict({user: {section: text}})`
  → skips extraction;
- pre-embedded → pass the `EmbeddingsBundle` (+ `sections` for LLM phases) →
  skips extraction + HyDE + embedding.

`run_full_match` dispatches on the input type; equivalence across entry
points is tested.

## Adapters in this repo

- **CLI (`main.py`)** — argument parsing, IO path resolution
  (`apply_io_overrides`: folder mode vs group mode), plots and cost reporting
  around `run_full_match`. Three registered pipelines: `matching`,
  `query_match` (`--query`), `batch_match` (`--members`).
- **External/hosted adapters live outside this repo** (the repo used to ship
  its own Modal app, `deploy_modal.py` — retired 2026-07, git history is the
  archive; production deployments wrap the library per `choreo_IO.md` §1.4).
  One upsert semantic worth preserving for any wrapper: scope `force` to the
  *given* profiles only, never wipe the roster — use the pure
  `extract_sections` + `store.put_sections` instead of the appending disk
  wrapper.

## Non-obvious gotchas

- **Every OpenRouter LLM request carries `provider.data_collection="deny"`.**
  `_build_extra_body` owns this policy and must preserve it when adding optional
  reasoning parameters. This restricts provider data collection; it is not an
  assertion of zero data retention or EU residency.
- **LLM cache keys must be `utils.hash_text`, never builtin `hash()`** — the
  builtin is salted per process, so such cache entries can never hit across
  runs (this was a real pre-refactor bug in scoring/intro caching, fixed in
  the refactor).
- **LLM cache keys hash the full prompt**, not the roster/pair id — the prompt
  embeds the profiles' section content, so an edited profile invalidates its
  scoring/intro/rerank cache entries automatically. (A roster-keyed cache
  silently replayed stale scores after profile edits — fixed June 2026.)
- Failed extractions are returned with "Not specified" sections so the
  pipeline keeps running, but adapters must **not persist them** (they'd never
  retry) — `extract_sections(failed_out=…)` reports them.
- Selected pairs the LLM never scores (failed batches, exhausted retries,
  budget-capped grouping) are reported via
  `score_pairs_with_llm(unscored_out=…)` and kept in the matching candidate
  set with their **embedding-only weight** — they are never silently dropped.
- All LLM phases enter asyncio via `llm.run_coro_blocking`, which works both
  from plain sync code and from inside a host's running event loop (async web
  backend / MCP server) — bare `asyncio.run` would raise there and silently
  degrade scoring.
- `LLMWrapper(cache_dir=None)` disables the response cache (used for
  transient query atoms). `LLMWrapper.cache_writes` counts new cache files
  this run — adapters use it to skip persistence (e.g. Modal `volume.commit`)
  on fully cache-hit calls.
- Cross-section keys are split on a single `_` (`needs_skills`) or the
  preferred `->` form; multi-word section names need `->`.
- **Freshness timestamps**: `Profile` / `ExtractedSections` /
  `EmbeddingsBundle.user_timestamps` carry optional ISO-8601
  `last_updated_at` provenance end-to-end (persisted in `sections.jsonl` and
  `bundle_meta.json`). Content hashes drive internal reuse; the timestamps are
  the adapter-level freshness signal — external stores pass their `updated_at`
  in (`sections_from_dict(..., last_updated_at=…)`, Modal `upsert_profiles`
  dict values) and compare with `utils.is_stale` to decide when to re-upsert.

See [matching_modes.md](matching_modes.md) for the three trigger shapes built
on top of this, and `tests/` for the executable spec (offline, fake LLM +
embedder; `RUN_LLM_TESTS=1` gates the live golden e2e).
