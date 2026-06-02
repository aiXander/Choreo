# TODO — Make Choreo matchmaking granular & incremental

**Status:** design / not yet implemented (drafted 2026-06-02).
**Goal:** evolve Choreo from a single monolithic, all-to-all, raw-text→reports batch
job into a set of composable services that operate against a *persistent community
profile/embedding store*, so the platform can trigger small, targeted matchmaking
operations on demand.

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

## 3. Refactor plan (workstreams)

Ordered so each unblocks the next. WS1 + WS2 are the foundation; B and C ride on top.

### WS1 — Persistent, per-user profile + embedding store  *(foundation)*

Replace the monolithic `vectors.npz` with a content-addressable per-user store so
profiles can be upserted one at a time and any subset loaded on demand.

- [ ] Define an on-disk schema (per user), e.g. `store/users/<user_id>.json` +
      `store/vectors/<user_id>.npz`, holding: `sections`, per-section
      `content_hash`, per-section embedding, HyDE descriptors + embeddings per
      `cross_key`, **`embedding_model`**, **native `dim`**, `section_names`,
      `updated_at`. (Record the model+dim so queries embed compatibly and MRL
      truncation stays consistent — see gap #6 / WS5.)
- [ ] New module `src/store.py` (working name) with:
      `upsert_user(...)`, `get_users(ids) -> (ids, section_names, emb[N,S,D], hyde)`,
      `all_user_ids()`, `delete_user(id)`. `get_users` is the replacement for
      `load_embeddings` and must assemble the dense array for an **arbitrary subset**
      in a fixed section order.
- [ ] Make embedding generation incremental: re-embed a section **only** when its
      `content_hash` changed (today it's all-or-nothing on roster equality,
      `embed.py:238`). Keep storing full-dim vectors.
- [ ] Migration/back-compat: a one-shot importer from the existing
      `data/<group>/embeds` blob into the new store (or keep `load_embeddings` for
      the legacy monolithic pipeline and have the store layer wrap it).

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

- [ ] New `src/query.py`: build a transient "query atom" from a free-text need
      (one `needs` section, `id="__query__"`), HyDE it into target vocabulary
      (reuse `generate_hyde_descriptors` for a single item, no disk cache needed),
      embed need + HyDE with the **store's** model/dims.
- [ ] Rank: load candidate pool via `store.get_users(...)`, compute 1×M directional
      similarity (WS2), take top-K.
- [ ] Optional LLM re-rank of the top-K only (reuse `build_batch_scoring_prompt`
      framing from `score.py:132` but query-vs-candidate, **no** set-cover/b-match).
- [ ] Output: a single ranked shortlist (id, score, why) + optional per-candidate
      intro (reuse `generate_introductions_for_matches`, treating the query as user1).
- [ ] Register a `QueryMatchPipeline` in `PIPELINE_REGISTRY` and expose it as a
      function with a tight, JSON-in/JSON-out signature for the agent tool-call.
- [ ] Target: no community re-embedding; cold load from store only.

### WS4 — Mode C: Subset batch match (M × N) with novelty

- [ ] Match-history store: persist surfaced pairs (by `stable_pair_id`) with a
      timestamp / run id, e.g. `store/match_history.jsonl`. Add a helper to fetch
      the exclusion set for a given member set and lookback window.
- [ ] Thread an `excluded_pairs` set through pair selection (`score.py:180`),
      group building, and `greedy_b_matching` (`match.py:111`) so already-surfaced
      pairs are skipped → "N novel matches."
- [ ] Make b-matching asymmetric: degree targets bind on the **M (member) side**;
      pool candidates get a separate (looser/optional) cap to avoid one popular
      person saturating everyone. Generalize `greedy_b_matching` accordingly.
- [ ] Subset-aware selection/set-cover: restrict scored pairs to (member × pool),
      not the full upper triangle.
- [ ] Reports only for the M members; append the run's matches to match-history.
- [ ] Register a `BatchMatchPipeline`.

### WS5 — Cross-run score stability & config

- [ ] Decide normalization strategy that doesn't depend on cohort size
      (gap #6): either persist global reference stats in the store and normalize
      against those, or switch to an absolute/rubric-anchored score (ties into
      Appendix A's "shared calibration anchors"). Needed so query/subset scores are
      comparable over time.
- [ ] Config surface for modes: `query` (top_k, llm_rerank on/off, pool filter),
      `batch` (member set source, novelty lookback, asymmetric b-params). The
      recipe/HyDE config already drives directionality — keep mode switching
      config-only where possible (per CLAUDE.md "switching matching modes").

### WS6 — Entry points / serving surface

- [ ] Add an "ingest pre-sectioned profiles" path (skip `load_profiles` +
      `extract_sections_from_profiles`): accept `{user_id: {section: text}}` and
      build `ExtractedSections` directly. This is the daily-upsert input.
- [ ] Modal: replace the throwaway `run_<uuid>` layout (`deploy_modal.py:234`) with
      a **persistent** store on the `choreo-data` Volume, and expose 3 functions:
      `upsert_profiles`, `query_match`, `batch_match` (+ keep the legacy full-run for
      back-compat). `query_match` must load-not-rebuild.
- [ ] Keep `choreo_IO.md` and `CLAUDE.md` updated as these contracts land
      (per docs workflow: durable bits → `docs/reference/`).

---

## 4. Open questions / decisions to confirm with Xander

- **Store backend:** flat files on disk/Volume (simplest, matches current style) vs
  SQLite vs a vector DB. Given community scale (hundreds, not millions) and the
  "static snapshot at query time" framing, flat per-user files + in-memory matmul
  is probably enough for V1 — confirm.
- **Query LLM re-rank default:** on (better quality, slower/cost) or off (pure
  embedding rank) for the agent tool-call?
- **Novelty window:** exclude *ever-surfaced* pairs, or a rolling window
  (e.g. last 8 weeks)? Re-surfacing after a cooldown may be desirable.
- **Member (M) source for batch mode:** explicit list passed in, or a flag on the
  stored profile (e.g. `tier: paying`)?
- **Embedding model migration:** when the embedding model changes, the whole store
  must be re-embedded. Worth a `model_version` guard + bulk re-embed path.

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
