# TODO — Make Choreo matchmaking granular, incremental & pluggable

**Status:** design / not yet implemented (drafted 2026-06-02, revised 2026-06-03).
**Goal:** evolve Choreo from a single monolithic, all-to-all, raw-text→reports batch
job into a set of **composable, IO-agnostic functions** that can be (a) driven
end-to-end from a folder of `.txt` profiles exactly as today, **and** (b) plugged
into an external app that owns its own data store (a Neon/Postgres DB) and feeds
profiles / sections / embeddings / past-match history in as plain arguments.

### Guiding principle: this repo stays unopinionated about persistence

Choreo is a **library of matchmaking compute**, not a database. The external app
(the community platform) owns all user data in Neon and wraps this repo with a thin
adapter that:

- pulls `user sections`, `section embeddings`, and `past matches per user` out of
  Neon,
- calls Choreo's core functions with that data **passed as arguments**,
- takes the returned values (new embeddings, match edges, intros) and writes them
  back to Neon.

So the refactor is fundamentally about **IO shape, not new storage**. The code will
always run in some process with a disk available (localhost, Modal container, …),
so **writing a stage's output to disk and having the next stage read it back is a
perfectly valid way to chain stages** — it's just not the *only* way. Each stage
must support both:

- **in-memory chaining** — pass the previous stage's return value straight in as a
  Python object, and
- **disk chaining** — a stage can persist its output in a declared format, and a
  later stage can load that same format.

The key enabler is that **every stage has a fixed, declared input schema and a
predictable output schema** (see §3.2, the *data-flow schema*). An external caller
(the Neon wrapper) fetches those schemas, formats whatever data it has into the
right shape, and triggers any stage it wants — either by handing Python objects in
or by writing files to a folder the stage reads. We pick whatever is most flexible
and extensible, not the most dogmatically pure.

A direct consequence: the caller must be able to **enter the pipeline at any
stage** — with raw text (do everything), with pre-extracted sections (skip
extraction), or with pre-computed embeddings (skip extraction + embedding, go
straight to similarity/scoring). The filesystem/`.txt` flow that exists today is
*one adapter* (the reference implementation) and must keep working unchanged; the
Neon wrapper is a *second adapter* living **outside this repo**. See §3.

> An unrelated prior design note ("LLM pair-scoring batch anchoring & global
> calibration") lives at the bottom of this file under **Appendix A** — it is
> still valid and was preserved verbatim. Don't delete it.

---

## 1. Where we're heading (the new product model)

Today the agent runs **one** pipeline: a folder of raw `.txt` profiles in →
extract → HyDE → embed → NxN similarity → score → b-match → intro → reports out.
Everything is glued inside one function (`_execute_matching_pipeline`, `main.py:178`).

The platform we're building instead needs **three distinct trigger shapes**, all
reading from a shared, slowly-updated community database:

| Mode | Trigger | Shape | Who gets matches | b-matching? | Match history? |
|------|---------|-------|------------------|-------------|----------------|
| **A. Profile upsert** | daily / on activity | per-user | — (writes DB) | no | no |
| **B. Query match** | agent tool-call ("find me a CTO who…") | **1 × M** | the *query* only | no — just top-N rank | optional |
| **C. Subset batch match** | weekly, paying members | **M × N** (M ⊆ pool) | the M members | yes (asymmetric) | **yes — novelty** |

- **A** replaces steps 1–3: profiles arrive already free-text *or* already
  sectioned; we extract (if needed), HyDE, embed, and **persist per-user** into a
  durable store. This is the "users are organically active, we update profiles
  daily" path.
- **B** is the hot path: one transient "need" (e.g. *"new CTO with skillset X"*)
  is treated as a one-row pseudo-profile, HyDE'd into skills vocabulary, embedded,
  and ranked against a **static, pre-built** community embedding DB. No re-embedding
  the community. Returns a ranked shortlist + intros. Must be cheap & fast.
- **C** runs the *full* matching machinery but over a **subset M** (paying members)
  as the "who needs matches" side, against the **full community pool** as the
  candidate side, excluding pairs already surfaced in prior runs ("N *novel*
  matches they weren't matched with before").

---

## 2. How the code is hooked up today (the coupling map)

The pipeline is a 9-step linear flow inside `_execute_matching_pipeline`
(`main.py:178–519`), orchestrated through a `PipelineRegistry` that currently holds
exactly one pipeline (`MatchingPipeline`, `main.py:522`). Data shapes:

```
load_profiles (ingest.py:27)            -> List[Profile]  (raw .txt, filename=id)
extract_sections_from_profiles          -> List[ExtractedSections{id, sections, hash}]
generate_hyde_descriptors (hyde.py:27)  -> {cross_key: [HydeDescriptors per user]}
create_section_embeddings (embed.py:197)-> (user_ids, section_names,
                                            embeddings[N,S,D], hyde_emb{key:[N,d,D]})
generate_similarity_matrix (candidate)  -> (dir[N,N], sym[N,N], ids, matrices_dict)
score_pairs_with_llm (score.py:282)     -> {pair_id: PairScore}
create_matches (match.py:237)           -> (edges, norm_embed, norm_llm)
generate_introductions_for_matches      -> {pair_id: Introduction}
generate_all_reports (report.py:184)    -> cohort_summary + per-user JSON
```

**The hard couplings that block granular operation (each is a concrete refactor target):**

1. **Embedding store is monolithic & roster-keyed.**
   `create_section_embeddings` writes one `vectors.npz` for the whole cohort and
   its cache is only reused when `set(existing_ids) == set(user_ids)` *and*
   `section_names` match (`embed.py:238–262`). **Adding or removing a single user
   re-embeds everyone.** There is no per-user, content-hash-addressable store and
   no "load embeddings for an arbitrary subset of user_ids" API
   (`load_embeddings`, `embed.py:338`, loads the whole blob only).

2. **Similarity is square & symmetric only.**
   `compute_fused_similarity_matrix` (`candidate.py:29`) assumes one user set:
   `cosine_matrix(section_embeddings)` does full NxN pairwise (`candidate.py:65`),
   the HyDE cross term is `src_norm @ tgt_norm.T` over the *same* users
   (`candidate.py:125`), and `generate_similarity_matrix` symmetrizes with
   `(dir+dir.T)/2` (`candidate.py:262`). There is **no rectangular
   source-set × target-set** path — required for both 1×M (query) and M×N (subset).

3. **Scoring/selection assume mutual square pairs.**
   `select_pairs_for_llm_scoring_optimal` (`score.py:180`) and the set-cover
   `create_profile_groups_from_pairs` (`score.py:25`) iterate the upper triangle of
   a single square matrix and pack *mutual* pairs. Query mode (one source) and
   subset mode (directed M→pool) don't fit this; there's no exclusion-set argument.

4. **b-matching is symmetric & global.**
   `greedy_b_matching` (`match.py:111`) applies `b_min`/`b_max` to *all* users
   symmetrically. Subset mode wants degree constraints on the **M side** only
   (members get N matches; popular pool candidates shouldn't be force-capped the
   same way), and needs an **excluded-pairs** input for novelty. Query mode skips
   b-matching entirely.

5. **No match-history persistence.**
   `cohort.json` is rewritten per run (`report.py:242`); nothing records *which
   pairs were already surfaced*. Mode C's "novel matches they weren't matched with
   before" has nothing to diff against.

6. **Score normalization is cohort-relative.**
   `prepare_normalized_scores` / `normalize_scores_with_reference_distribution`
   (`utils.py:149,220`) build the reference distribution from the **full square
   upper triangle** of the current run's matrix. For 1×M and M×N this both (a)
   breaks structurally (rectangular) and (b) makes scores non-comparable across
   runs of different size — a problem when results are stored/compared over time.

7. **Reports/intros are written for "all users in the edge set."**
   `generate_all_reports` (`report.py:184`) emits a file per user derived from
   `extracted_sections`. Query mode wants a *single ranked answer*; subset mode
   wants reports only for the M members. Intro generation (`introduction.py:57`) is
   fine to reuse per-pair but is currently only called on b-match edges.

8. **Entry points assume raw `.txt` on a filesystem.**
   `main.py` (folder/group mode) and `deploy_modal.py` both start from `.txt`
   files and write to a *throwaway* `run_<uuid>` dir on the Volume
   (`deploy_modal.py:234`) — no persistent DB, no reuse between calls, no
   upsert/query/batch endpoints.

**What's already reusable as-is (lean on these):**
- `extract_sections_from_profiles` caches per-profile by content hash
  (`extract.py:96`) — good for incremental upsert; just needs a "skip extraction,
  ingest pre-sectioned input" entry.
- HyDE caches per-user by content hash in `hyde/<cross_key>.jsonl` (`hyde.py:64`)
  — reusable for both upsert and per-query HyDE.
- `truncate_embeddings` (`embed.py:169`) is shape-agnostic (last-axis), works for
  query rows and HyDE alike. Full vectors are stored, so MRL size stays re-tunable.
- The directional cross-section math (`candidate.py:92–139`) is exactly the
  "query.needs → candidate.skills" operation we want — it just needs to accept a
  separate source row set.
- `LLMWrapper` batching/caching/cost-tracking (`llm.py`) is mode-agnostic.
- `PipelineRegistry` (`main.py:69`) already supports registering multiple named
  pipelines — the seam to add `query_match` and `batch_match`.
- `choreo_IO.md` (repo root) documents the current IO contract — keep it honest as
  these change.

---

## 3. Target architecture: schema-driven stages + adapters

Three layers. The dividing line that matters: **each stage is a transform with a
declared input/output schema; *how* its data is moved (in-memory object or file on
disk) is the adapter's choice, not baked into the stage.**

```
┌─ Adapters (own IO; one per data source) ───────────────────────────────┐
│  • Filesystem/CLI adapter  (main.py): .txt in  → stages → json/npz out  │  ← in this repo
│  • Modal adapter           (deploy_modal.py): same, on a Volume         │  ← in this repo
│  • Neon/Postgres adapter   (the app's wrapper)                          │  ← OUTSIDE this repo
│      fetches the stage schema → formats rows into it → calls the stage  │
│      (passing python objects OR writing files the stage reads)          │
└────────────────────────────────────────────────────────────────────────┘
                  │  (each stage: declared input schema → output schema)
┌─ Orchestration (mode runners; chain stages in-memory or via disk) ──────┐
│  run_full_match() · run_query_match() · run_batch_match()               │
└────────────────────────────────────────────────────────────────────────┘
                                  │
┌─ Core stages (schema in → schema out; can be invoked individually) ─────┐
│  extract · hyde · embed · similarity(rectangular) · score ·             │
│  match · introduce · build_report_data                                   │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.1 The data-flow schema (the central organizing idea)

Every stage already has a *de facto* fixed input shape and predictable output
shape (see the §2 data-shapes block). The plan is to make that **explicit and
discoverable**: each stage declares a typed `input_schema` and `output_schema`, and
the set of stages is exposed through a small **stage registry** the caller can
introspect at runtime.

The workflow for an external caller (e.g. the Neon wrapper) becomes:

1. **Fetch the schema** for the stage it wants to trigger (e.g. "similarity") —
   `describe_stage("similarity")` returns its input/output schema.
2. **Format its data** (whatever it has in Postgres) into that input schema.
3. **Invoke the stage**, passing Python objects directly *or* pointing the stage at
   a folder where it wrote the input files (both supported — see §3.3).
4. **Consume the output** in the declared output format and persist it however it
   likes (back to Neon, to disk, return to the agent).

Concretely, implement each stage as a small spec object:

```
Stage{
  name: str,
  input_schema:  <typed schema>,   # dataclass / TypedDict / pydantic / JSON-schema
  output_schema: <typed schema>,
  run(input) -> output,            # pure transform, in-memory
  # optional disk helpers the adapter may use:
  load(path) -> input,  dump(output, path) -> None
}
```

Use a single schema mechanism end-to-end (recommendation: dataclasses +
`to_dict`/`from_dict`, optionally exported as JSON Schema so a non-Python caller can
read it too). The win: the contract is **self-describing** — the external caller
never has to read Choreo's source to learn what a stage needs; it asks.

The concrete schemas (these already mostly exist as return values — formalize them):

| Stage | Input schema | Output schema |
|-------|-------------|---------------|
| `extract` | `{user_id: raw_text}` + sections-config | `list[ExtractedSections{id, sections, hash}]` |
| `hyde` | `ExtractedSections[]` + cross-weights + `n_descriptors` | `{cross_key: [HydeDescriptors{..., descriptors: list[str]}]}` |
| `embed` | `ExtractedSections[]` (+ optional HyDE, + optional `existing_embeddings` to reuse) | **embeddings bundle**: `{user_ids, section_names, embeddings[N,S,D], hyde{key:[N,d,D]}, embedding_model, dim}` |
| `similarity` | source bundle + target bundle + recipe | `{dir_matrix, (sym_matrix), matrices_dict, source_ids, target_ids}` |
| `score` | similarity + sections + budgets + `excluded_pairs?` | `{pair_id: PairScore}` |
| `match` | candidates + llm_scores + matching/blending cfg + `excluded_pairs?` | `list[Edge]` |
| `introduce` | `Edge[]` (or query→candidate pairs) + sections | `{pair_id: Introduction}` |
| `report` | `Edge[]` + sections (scope: which users) | report-data dict (the caller writes it) |

Design rules for every stage:
- **Pure transform at heart:** `run(input) -> output` does no IO. Disk read/write is
  an *optional* helper (`load`/`dump`) the adapter calls — never hidden inside
  `run`. This is what lets the same stage chain in-memory or via files.
- **Optional reuse, passed in:** instead of "load cache from disk if present,"
  accept an optional `existing_*` argument (e.g. `existing_embeddings`) and recompute
  only what's missing or hash-changed. The caller decides where "existing" came
  from (Neon, files, nothing).
- **Embeddings carry provenance:** `embedding_model` + native `dim` ride inside the
  embeddings bundle so MRL truncation (`embed.py:169`) stays consistent across
  stored-vs-fresh vectors. (Model-migration handling is explicitly out of scope for
  now — see §5 note.)
- **Stable ordering:** stages that build dense `[N,S,D]` arrays use an explicit
  `section_names` and `user_ids` order from the input, so a subset pulled from Neon
  lines up with the array axes deterministically.

### 3.2 Query as a *partial profile* (the clean way to support query-driven matching)

A natural-language query ("find me a CTO in the network who's great at agentic
engineering") should be **the full input context** for a match run, *without*
forcing it through the whole profile recipe. The clean model: **a query is just a
partial `ExtractedSections`** — a pseudo-user (`id="__query__"`) with only some
sections populated. This drops straight into the existing machinery because the
per-pair fusion already **treats an absent section as neutral, not as similarity 0**
(`candidate.py:80–143`). So a query with only `needs` filled simply matches via the
`needs_skills` cross-weight (+ any same-section `needs` weight) and ignores the rest.

Two ways the caller supplies a query, both supported:

- **(a) Explicit section mapping (default, cheapest):** the tool/caller stipulates
  which section(s) the query text maps to, e.g.
  `{"needs": "a CTO great at agentic engineering"}`. No extraction LLM call. The
  query is HyDE'd + embedded like any partial profile.
- **(b) Auto-expand via extraction:** hand raw query text to the `extract` stage to
  populate one or more sections (reusing the existing extractor) when the caller
  wants richer structure than a single section.

Either way, allow a **per-call recipe override**: a query usually wants a different
recipe than full-profile matching (e.g. `cross_section_weights: {needs_skills: 1.0}`
with same-section weights zeroed, since the query has no vision/project to compare).
The recipe is already a plain config dict, so accept an override argument on the
query runner rather than hardcoding. This makes "match on which section(s)" a
caller-controlled knob, not a code change.

### 3.3 Multiple HyDE descriptors per query/need (already supported — keep it)

The directional machinery is **already built for `n` descriptors per source
section**, not one: `hyde.n_descriptors` config drives generation
(`hyde.py:48`), embeddings are stored as `[N, n_desc, D]` (`embed.py:322–327`), and
the cross-similarity **max-pools over all descriptor pairs** (`candidate.py:122–127`).
Today config just sets `n_descriptors: 1`. The task is therefore mainly to **avoid
regressing this** in the query path: the query atom must carry `descriptors:
list[str]` (the `HydeDescriptors` dataclass already does, `hyde.py:18`) and the
1×M similarity must keep the max-pool loop. Add a test at `n_descriptors > 1`
(see WS7) to lock it in.

### 3.4 The filesystem store stays — as an adapter, not the core

The current disk behavior (`vectors.npz`, `sections.jsonl`, `hyde/*.jsonl`,
hash-based skip) is **kept and refactored behind a small storage interface** so the
standalone `.txt` workflow runs exactly as now. It becomes the *reference adapter*
that implements the same "give me existing data / here's new data to persist"
contract the Neon adapter implements — just backed by files. It is also the
canonical implementation of each stage's `load`/`dump` disk format (§3.1), so
disk-chaining between stages has one source of truth. No external user is forced to
adopt it.

---

## 4. Refactor plan (workstreams)

Ordered so each unblocks the next. WS1 + WS2 are the foundation; B and C ride on top.

### WS1 — Schema-driven stages: separate transform from IO  *(foundation)*

Make each stage a pure transform with a declared input/output schema (§3.1) and an
optional disk `load`/`dump`. The existing file persistence is preserved as the
`FileStore` adapter. The point is **not** a new store — it's clean, self-describing
stage boundaries that chain in-memory or via disk.

- [ ] **Formalize the stage schemas + a `describe_stage` / stage registry.** Pick
      one schema mechanism (recommend dataclasses with `to_dict`/`from_dict`, JSON
      Schema export optional) and define the §3.1 table as real types. Expose a
      registry so a caller can fetch a stage's input/output schema at runtime.
- [ ] **Split each disk-coupled stage into transform + disk helpers.** Today
      `create_section_embeddings` (`embed.py:197`) and
      `extract_sections_from_profiles` (`extract.py:43`) interleave compute with
      load/save and hash-skip logic. Pull out a pure transform
      (e.g. `embed_sections(extracted, model, existing=None) -> embeddings-bundle`)
      and move `.npz`/`.jsonl` read/write into `load`/`dump` helpers the adapter
      calls — never inside the transform.
- [ ] **Define a tiny `Store`/adapter protocol** with methods like
      `get_sections(ids)`, `get_embeddings(ids)`, `get_match_history(ids)`,
      `put_embeddings(...)`, `put_matches(...)`. Provide **one implementation in
      this repo**: `FileStore` (wraps current disk layout, owns the stage disk
      formats). The Neon implementation is the app's, outside this repo.
- [ ] **Make embedding reuse content-hash based, not roster-based.** Today cache is
      reused only when `set(user_ids)` matches exactly (`embed.py:238`) — adding one
      user re-embeds everyone. The transform should diff per-section `content_hash`
      against the passed-in `existing` and recompute just the deltas, regardless of
      who else is in the set. (Lets the Neon adapter store one embedding per section
      and pass back only what it has.)
- [ ] **`get_embeddings(ids)` for arbitrary subsets.** Replace whole-blob
      `load_embeddings` (`embed.py:338`) with subset assembly: given any `user_ids`,
      return the dense `[N,S,D]` bundle in a fixed `section_names` order. Critical for
      query (1×M) and subset (M×N) modes pulling a slice of the community.
- [ ] **Entry-at-any-stage:** ingest pre-sectioned input
      (`{user_id: {section: text}}` → `ExtractedSections`) bypassing
      `load_profiles`/extraction, and accept the embeddings-bundle directly bypassing
      embedding. (See WS6 for the public signatures.) Embedding always happens
      **inside this repo** via the `embed` stage — Neon only stores the bundle and
      hands it back; it never embeds itself.

### WS2 — Rectangular (source × target) similarity  *(foundation)*

Generalize similarity so source and target user sets can differ.

- [ ] Refactor `compute_fused_similarity_matrix` (`candidate.py:29`) to accept
      `source_embeddings` and `target_embeddings` (+ their HyDE), computing
      same-section terms as `source_sec @ target_sec.T` and the directional cross
      term as `source_hyde @ target_sec.T`. Square mode = pass the same set twice
      (must reduce *exactly* to current behavior — keep a regression check).
- [ ] Drop the forced symmetrization for rectangular use; keep
      `(dir+dir.T)/2` only for the legacy square cohort path
      (`generate_similarity_matrix`, `candidate.py:262`).
- [ ] Keep per-pair masking/normalization semantics (`candidate.py:80–143`) — the
      "absent section is neutral, not zero" behavior must survive the rewrite.

### WS3 — Mode B: Query match (1 × M)  *(the hot path)*

Built on the **query-as-partial-profile** model (§3.2) — reuses the directional
machinery rather than a separate code path.

- [ ] New `src/query.py`: build a transient query atom as a partial
      `ExtractedSections` (`id="__query__"`). Accept **either** an explicit section
      mapping (`{"needs": "<query text>"}`, default — no extraction call) **or** raw
      text routed through the `extract` stage to auto-populate sections (§3.2 a/b).
- [ ] HyDE the query's source section(s) into target vocabulary (reuse
      `generate_hyde_descriptors` for the single atom; no disk cache needed),
      supporting **`n_descriptors > 1`** (§3.3). Embed need + HyDE via the `embed`
      stage (embedding stays in-repo).
- [ ] **Candidate pool comes in as an argument**: `run_query_match` accepts the
      embeddings-bundle for the pool directly (Neon adapter passes pulled rows;
      `FileStore` passes loaded vectors). Never re-embeds the pool.
- [ ] **Per-call recipe override** (§3.2): the caller can pass a query-specific
      recipe (e.g. `{cross_section_weights: {needs_skills: 1.0}}`) so matching keys
      on just the section(s) the query populated. Falls back to config recipe.
- [ ] Rank: compute 1×M directional similarity (WS2) against the pool, take top-K
      (`top_k` configurable).
- [ ] **LLM re-rank of the top-K is ON by default** (decision confirmed). Reuse the
      `build_batch_scoring_prompt` framing (`score.py:132`) as query-vs-candidate,
      **no** set-cover / b-match. Allow disabling via config for a pure-embedding,
      cheaper path.
- [ ] Output (returned, not written): a ranked shortlist (id, score, why) + per-
      candidate intro (reuse `generate_introductions_for_matches`, query as user1).
      The caller persists/returns it however it likes.
- [ ] Register a `QueryMatchPipeline` in `PIPELINE_REGISTRY` and expose a thin
      JSON-in/JSON-out wrapper for the agent tool-call (goes through the Neon adapter
      in production).

### WS4 — Mode C: Subset batch match (M × N) with novelty

- [ ] **Members & pool are caller-supplied** (decision confirmed): `run_batch_match`
      takes an explicit M list (ids/sections/embeddings) for the "who needs matches"
      side and an explicit pool (ids/sections/embeddings) for the candidate side —
      for both input and output. Choreo never reads a `tier` flag or decides who's a
      member; that's app data.
- [ ] **Match history is an INPUT, not a store this repo owns.** `run_batch_match`
      accepts `excluded_pairs: set[pair_id]` (or `dict[user_id, set[partner_id]]`)
      as an argument — the caller builds it from Neon's `past_matches` table.
      **Novelty window default = last 6 months, configurable** via a config key
      (e.g. `matching.novelty_window_months: 6`). For the standalone `FileStore`
      path, the adapter applies this window against `match_history.jsonl`; for the
      Neon path the app applies it when building `excluded_pairs` (the config value
      is still the documented default it should honor).
- [ ] Thread that `excluded_pairs` set through pair selection (`score.py:180`),
      group building, and `greedy_b_matching` (`match.py:111`) so already-surfaced
      pairs are skipped → "N novel matches."
- [ ] Make b-matching asymmetric: degree targets bind on the **M (member) side**;
      pool candidates get a separate (looser/optional) cap to avoid one popular
      person saturating everyone. Generalize `greedy_b_matching` accordingly.
- [ ] Subset-aware selection/set-cover: restrict scored pairs to (member × pool),
      not the full upper triangle.
- [ ] **Return** report-data only for the M members, plus the new pairs surfaced
      this run (so the caller can append them to its own history). The `FileStore`
      adapter writes them to disk; the Neon adapter writes them to Postgres.
- [ ] Register a `BatchMatchPipeline`.

### WS5 — Cross-run score stability & config

- [ ] Decide normalization strategy that doesn't depend on cohort size
      (gap #6): either compute the reference distribution over the passed-in pool /
      let the caller supply stable reference stats, or switch to an
      absolute/rubric-anchored score (ties into Appendix A's "shared calibration
      anchors"). Either way the reference must be an explicit input, not derived
      from the current run's square matrix — needed so query/subset scores are
      comparable over time.
- [ ] Config surface for modes: `query` (top_k, llm_rerank on/off, pool filter),
      `batch` (member set source, novelty lookback, asymmetric b-params). The
      recipe/HyDE config already drives directionality — keep mode switching
      config-only where possible (per CLAUDE.md "switching matching modes").

### WS6 — Public function signatures & adapters

The orchestration layer (§3) is the public API external apps import. Keep it small
and typed; runners take Python objects in and return Python objects out (callers may
still chain via disk using each stage's `load`/`dump`).

- [ ] **Define the three mode runners** as importable functions taking schema
      objects and returning schema objects (sketch):
      - `run_full_match(sections|embeddings, config, excluded_pairs=None) -> {edges, report_data, embeddings}`
      - `run_query_match(query, pool_embeddings, config, recipe_override=None, top_k=...) -> {shortlist}`
      - `run_batch_match(members, pool, config, excluded_pairs) -> {edges, report_data, new_pairs}`
      Each accepts an optional `Store` for the convenience/standalone case but never
      requires one.
- [ ] **`report.py` → return data, then write.** Split `generate_all_reports`
      (`report.py:184`, currently writes per-user JSON + `cohort.json` to disk) into
      a pure `build_report_data(...) -> dict` plus an adapter that persists it. The
      Neon caller wants the dict; the CLI wants the files.
- [ ] **Stage helpers** for entry-at-any-stage:
      `sections_from_dict({user_id: {section: text}}) -> list[ExtractedSections]`
      (skip `load_profiles`/extraction) and direct acceptance of the
      embeddings-bundle (skip embedding).
- [ ] **CLI adapter (`main.py`) keeps working unchanged** for `.txt` folder/group
      runs — it just becomes a `FileStore` + `run_full_match` composition.
- [ ] **Modal adapter:** expose `upsert_profiles`, `query_match`, `batch_match` (+
      keep the legacy full-run for back-compat). For standalone Modal use these can
      back onto a `FileStore` on the `choreo-data` Volume; in the real product the
      app calls the library directly with Neon data, so the Volume persistence is
      optional, not the design center. (Replaces the throwaway `run_<uuid>` layout,
      `deploy_modal.py:234`.)
- [ ] Keep `choreo_IO.md` and `CLAUDE.md` updated as these contracts land
      (per docs workflow: durable bits → `docs/reference/`).

### WS7 — Test scripts (per-stage, per-mode, and end-to-end)

Primary fixture: **`data/test4`** (4 profiles). `matching.min_profiles_required`
is 2 in config, so 4 profiles is enough to exercise full and batch modes. Put tests
under `tests/` and make them runnable with `uv run pytest` (and a couple of
runnable `__main__` scripts for manual inspection). Keep network/LLM calls
**cached** (the existing hash caches make reruns cheap) and gate live-LLM tests so
the suite can run offline against fixtures.

- [ ] **Schema round-trip tests:** for every stage, assert `from_dict(to_dict(x))`
      is identity and that `dump`→`load` reproduces the object (locks the §3.1
      contract and the disk format).
- [ ] **Per-stage isolation tests** on `test4`, each fed the *previous* stage's
      saved output (proving stages compose via disk *and* in-memory):
      `extract` · `hyde` (assert `n_descriptors=1` **and** `>1`, §3.3) · `embed`
      (assert bundle shape + provenance) · `similarity` (assert **rectangular**
      source×target reduces to the square result when source==target, the WS2
      regression check) · `score` (with and without `excluded_pairs`) · `match`
      (asymmetric b, exclusion honored) · `introduce` · `report`.
- [ ] **Mode B (query) test:** explicit-section-mapping query + auto-expand query,
      both against the `test4` pool; assert top-K ordering, that LLM re-rank runs by
      default, and that a `recipe_override` changes results. Include an
      `n_descriptors>1` query case.
- [ ] **Mode C (batch) test:** a 2-member M against the 4-profile pool with a seeded
      `excluded_pairs`; assert excluded pairs never appear and members get novel
      matches.
- [ ] **Incremental-embedding test:** embed `test4`, change one profile's section,
      re-run; assert only the changed section re-embeds and unchanged vectors are
      byte-identical (locks the content-hash reuse from WS1).
- [ ] **End-to-end regression:** the existing full `.txt` run on `data/test4` still
      produces equivalent `cohort.json` / per-user reports after the refactor
      (golden-file compare, tolerant to score float jitter).
- [ ] **Adapter parity test:** `FileStore`-driven run == in-memory-objects-driven
      run for the same inputs (proves the two chaining styles agree).

---

## 5. Decisions (confirmed) & remaining open questions

**Confirmed (2026-06-03):**
- **Persistence backend is the caller's.** Neon owns the data; this repo only ships
  a `FileStore` adapter for standalone `.txt` runs. In-memory matmul over a passed-in
  pool is the V1 compute (fine for hundreds/low-thousands). Disk-as-intermediate
  between stages is allowed wherever it's the most flexible option (§3).
- **Embedding ownership: entirely inside this repo.** The `embed` stage (existing
  functions) does all embedding; **Neon only stores** the resulting bundle and hands
  it back. The app never embeds itself. → `embed` is a first-class public stage.
- **Query LLM re-rank: ON by default** (config-disableable for a cheaper pure-
  embedding path).
- **Novelty window: last 6 months, configurable** (`matching.novelty_window_months`).
- **Member (M) source for batch mode: caller passes explicit ids/profiles/embeddings**
  for both input and output. Choreo never reads a `tier` flag.

**Still open:**
- **Schema mechanism:** dataclasses (+JSON-Schema export) vs pydantic for the §3.1
  stage schemas. Recommendation: dataclasses to stay dependency-light; revisit if
  the Neon caller wants runtime validation.
- **`Store`/adapter protocol surface:** confirm the §3.1 stage table is the complete
  set of contracts the Neon adapter must satisfy.
- **Cross-run score normalization (gap #6 / WS5):** pool-relative vs caller-supplied
  reference stats vs absolute rubric — needs a call once query/batch land.

**Noted for later (explicitly out of scope now):**
- **Embedding model migration:** when the embedding model changes, every stored
  vector is stale and must be re-embedded. Not handled now. The embeddings bundle
  carries `embedding_model` + `dim` (§3.1) so a future `model_version` guard +
  bulk re-embed path can be added without reworking the contract.

---

## Appendix A — LLM pair-scoring: batch anchoring & global calibration (prior note, preserved)

Context: investigated whether to bump `budgets.n_profiles_to_score_together` (currently 4)
and, more deeply, how the batched LLM scoring interacts with global ranking quality.
Nothing changed in code yet — this is the design residue to revisit.

### How scoring batches are built today (the relevant mechanics)

- `select_pairs_for_llm_scoring_optimal` (`src/score.py:180`) picks which pairs to score,
  bounded by `max_n_llm_evaluations_per_profile` (per-user) and `max_pair_llm_calls` (global).
- `create_profile_groups_from_pairs` (`src/score.py:25`) then packs the *selected* pairs into
  user-groups via greedy **set-cover**: seed each group from the highest-priority uncovered
  pair, grow it to size `n` by adding whoever covers the most additional uncovered pairs.
  Each group = **one LLM call** scoring all C(n,2) selected pairs among its members.
- `n = max(2, n_profiles_to_score_together)` (`src/score.py:53`) — clamped min 2, no upper bound.
- Each pair is scored **exactly once** (marked `covered`, never re-scored — `src/score.py:104`).
- Final rank blends LLM score 0.65 + embedding score 0.35 (`blending`); the embedding term is
  globally calibrated (cosine over whole cohort) and already damps LLM anchoring noise.
- Scoring prompt (`config/scoring_prompt.yaml`) asks for **absolute, independent** per-pair
  scores against a fixed rubric (0.8–1.0 / 0.5–0.7 / …) — but the model inevitably uses the
  rest of the batch as a contrast frame regardless.

### `n_profiles_to_score_together` 4 → 5

- Safe; runs correctly. Changes pairs/call from C(4,2)=6 to C(5,2)=10.
- Fewer LLM calls to cover the same pairs (denser batches), slightly bigger prompt per call.
- On `gemini-3.5-flash` the attention-dilution cost at 5 (even ~8) is negligible.
- **Verdict: nearly neutral.** Low risk, but it does NOT deliver better global calibration
  (see below). Batch size is a cost knob, not a quality knob. Leave at 4 unless tinkering.

### The real issue: quality-stratified batches → drifting anchor frame

- Because seeds are similarity-ordered, **early batches are dense high-similarity clusters and
  later batches are lower-similarity leftovers.** The in-batch contrast frame drifts downward
  across batches, and since each pair is scored once, that frame is baked into its score.
  This can cause cross-batch rank inversions (a strong pair compressed inside a strong cluster
  losing to a mediocre pair lifted as "best of a weak batch").

### Why random batching (the tempting fix) is the wrong lever

Idea considered: keep selection, then randomly permute so each batch is a random sample.
Rejected for two reasons:

1. **Fights the set-cover optimization.** Selected pairs are a sparse graph (~27% density at
   N≈60, 16 evals/profile). A *random* group of 5 users captures ~2.7 selected pairs instead
   of ~8–10 → ~2–3× more LLM calls for the same coverage, or dropped coverage (the
   `UNSCORED` warning path, `src/score.py:115`).
2. **Trades correctable bias for uncorrectable noise.** Randomization makes the anchor frame
   unbiased-on-average but high-variance per pair. With one-shot-per-pair scoring there's no
   way to average it out — noisy ranks are worse than consistently-tilted ranks.

Also: shuffling batch *order* (vs membership) does nothing — anchoring lives in membership.

### Better levers to revisit (ranked)

1. **Shared calibration anchors (best ROI):** put the *same* frame in every batch — 1–2 fixed
   reference profiles in each group and/or few-shot rubric exemplars in `scoring_prompt.yaml`
   ("this pair = 0.9 because…, this = 0.3 because…"). Equalizes anchoring across all batches at
   ~constant cost; low-bias AND low-variance. For a one-shot scorer, shared anchors dominate
   random anchors.
2. **Global re-rank pass:** after clustered scoring, take the global top-K pairs across all
   batches and score them together in 1–2 final calibration batches. Puts the actual
   contenders (the only ones whose order affects matching) into one shared frame. Cheap.
3. **Strengthen absolute rubric thresholds** so the model leans on external anchors over
   batch-relative contrast. Partial, cheapest.
4. (Expensive) score each pair in 2 batches and average — directly cuts anchoring variance,
   but doubles cost and breaks the score-once design.
