# Matching modes: full cohort, query (1×M), subset batch (M×N)

*(Companion to [stages_and_adapters.md](stages_and_adapters.md) — that file
covers the plumbing; this one covers the three product trigger shapes.)*

| Mode | Runner | Shape | b-matching? | Match history? |
|------|--------|-------|-------------|----------------|
| Cohort (full) | `run_full_match` | N × N | yes (symmetric) | optional `excluded_pairs` |
| **A. Profile upsert** | Modal `upsert_profiles` / extract+hyde+embed stages | per-user | no | no |
| **B. Query match** | `run_query_match` (choreo/query.py) | 1 × M | no — top-K rank | optional `exclude_ids` |
| **C. Subset batch** | `run_batch_match` (choreo/batch_match.py) | M × N | yes (asymmetric) | **yes — novelty input** |

## Mode B: query as a partial profile

A query ("find me a CTO great at agentic engineering") is a pseudo-user
(`__query__`) with only some sections filled — it drops into the normal
directional machinery because absent sections are masked as neutral. Two input
shapes:

- **explicit mapping** (default, no extraction call): `{"needs": "<text>"}`
- **raw text** → auto-expanded through the extract stage ("Not specified"
  results are mapped to empty = absent, unlike the cohort path which embeds
  them literally).

Key behaviors and why:
- **The pool is an argument** (`EmbeddingsBundle`), never re-embedded. Only the
  one-row query atom is HyDE'd + embedded per call → cheap hot path.
- **Per-call recipe override**: a query has no vision/project to compare, so
  `config.query.recipe` defaults to pure `cross_section_weights:
  {needs_skills: 1.0}`; callers can override per call
  (`recipe_override`). Cross keys whose source section the query didn't fill
  are dropped for that call (noted in `result.notes`).
- **LLM re-rank is ON by default** (`query.llm_rerank`), reusing the pair
  scoring prompt with only query↔candidate pairs requested (no set-cover, no
  b-matching). Requires `pool_sections`; silently skipped with a note
  otherwise. Final score = `embed_weight * embed_norm + llm_weight *
  llm_score` — deliberately blending the **raw** LLM score, unlike the
  cohort/batch remap (`normalize_scores_with_reference_distribution`): a
  shortlist's top-K embed scores are near-identical, so the cohort remap would
  compress the LLM signal to nothing and make the re-rank a no-op. Query
  scores rank within one shortlist; they are not comparable with cohort/batch
  `final_weight` values.
- Returns a `QueryMatchResult` (shortlist + intros) — **returned, never
  written**. JSON wrapper for agent tool-calls: `query.run_query_match_json`
  (accepts `store_dir` for the FileStore path or an inline pool dict).

## Mode C: subset batch with novelty

`run_batch_match(member_ids, pool, config, excluded_pairs, pool_sections=…)`:

- **Members are caller-supplied ids** (⊆ pool). Choreo never reads a `tier`
  flag or decides who's a member — that's app data.
- **Match history is an input, not a store this repo owns.** The adapter
  builds `excluded_pairs` from its history honoring
  `matching.novelty_window_months` (default 6): FileStore reads
  `match_history.jsonl`; the Neon app applies the same window to its
  `past_matches`. Excluded pairs are filtered at *selection*, *scoring* and
  *matching* (defense in depth).
- **Pair selection is rectangular** (`select_pairs_rectangular`): pairs
  restricted to member × pool, self-pairs skipped, member↔member pairs scored
  once with the two directional entries averaged, round-robin over members so
  budgets are spent fairly.
- **b-matching is asymmetric** (`greedy_b_matching(member_ids=…,
  pool_b_max=…)`): `b_min`/`b_max` bind on members only; pool users get the
  optional looser `matching.pool_b_max` cap (None = uncapped) so one popular
  person can't saturate every member. Phase-3 force-fill may still exceed caps
  for members stuck below `b_min` — intentional, inherited from the legacy
  algorithm.
- Returns edges + **report data for the members only** + `new_pairs` (this
  run's surfaced pairs, for the caller to append to its history). The CLI
  pipeline writes reports to `outputs/batch/` (so they never clobber the full
  cohort run's reports) and appends to `match_history.jsonl` itself.

## Cross-run score stability

`utils.prepare_normalized_scores` takes the reference distribution as an
**explicit input** (`reference_scores`). Rectangular modes pass their own
member×pool (or 1×M) similarity values; only the legacy square path still
derives the reference from the current matrix's upper triangle. Callers that
need scores comparable over time should pass stable reference stats from their
store. (An absolute/rubric-anchored LLM score is still an open lever — see
[../TODO/scoring_calibration.md](../TODO/scoring_calibration.md).)

## Config surface (config.yaml)

```yaml
matching:
  pool_b_max: null            # Mode C pool-side degree cap
  novelty_window_months: 6    # Mode C exclusion window (adapters apply it)
query:
  top_k: 5
  llm_rerank: true            # false = pure-embedding, cheaper
  generate_intros: true
  recipe: {…}                 # query-default recipe (cross-only)
```

CLI: `--pipeline query_match --query '…'` · `--pipeline batch_match --members a,b`.
