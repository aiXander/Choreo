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
- **The pool is an argument** (`EmbeddingsBundle`), never re-embedded. At most
  the one-row query atom is embedded (and HyDE'd, when cross weights ask for
  it) per call → cheap hot path.
- **Per-call recipe override** (`recipe_override`): the recommended caller
  shape is an explicit section mapping plus same-section weights with empty
  cross weights — legs embed directly against the matching pool sections, so
  the query path makes **zero** LLM calls before the re-rank. There is no
  packaged `query.recipe` anymore; without an override, queries fall back to
  the top-level `recipe` (whose cross terms exist for batch mode and DO
  HyDE-expand the filled query sections). Cross keys whose source section the
  query didn't fill are dropped for that call (noted in `result.notes`).
- **`pool_sections` can be lazy** (`sections_provider`): instead of passing
  section text for the whole pool up front, pass a callable
  `(user_ids) -> {user_id: sections}` — it's invoked once with the
  over-fetched re-rank candidate ids after the embedding cut, so a store-backed
  adapter only materializes text for the ~`top_k × multiplier` survivors.
  Explicit `pool_sections` wins; with neither, LLM hops skip with a note.
- **LLM re-rank is ON by default** (`query.llm_rerank`), reusing the batch
  scoring machinery with only query↔candidate pairs requested (no set-cover,
  no b-matching) and the **`query_scoring` template** — a directional variant
  (candidate → query need, reciprocity explicitly off) that lives as a second
  key in `scoring_prompt.yaml`; custom scoring prompts without that key fall
  back to their pair template (`config.resolve_prompt_templates`). Requires `pool_sections`; silently skipped with a note
  otherwise. Final score = `embed_weight * embed_norm + llm_weight *
  llm_score` — deliberately blending the **raw** LLM score, unlike the
  cohort/batch remap (`normalize_scores_with_reference_distribution`): a
  shortlist's top-K embed scores are near-identical, so the cohort remap would
  compress the LLM signal to nothing and make the re-rank a no-op. Query
  scores rank within one shortlist; they are not comparable with cohort/batch
  `final_weight` values.
- **The re-rank pool is over-fetched** (`query.rerank_pool_multiplier`,
  default 4): the LLM scores `top_k * multiplier` embedding candidates and the
  shortlist is the re-ranked top `top_k` — so a good match the embedding stage
  ranked just below the cut can be *recovered*, not merely reordered. Set the
  multiplier to 1 for the legacy reorder-only behavior. Intros are only
  generated for the final `top_k` (or fewer — `generate_intros` also accepts
  an int N to cap the intro wave at the rows an adapter will render).
- **Mode-B novelty needs no `excluded_pairs` mechanism**: for a 1×M query the
  asker is known, so "pairs recently surfaced for this asker" reduces to a set
  of candidate user ids — build it from your match history and pass it as
  `exclude_ids` (identical semantics: a novelty-excluded candidate is not
  surfaced at all). Don't build a parallel pair-id mechanism for query mode.
- **`display_names`** (`{user_id: name}`, optional on all three runners): maps
  opaque ids (uuids) to human names inside the scoring/re-rank/intro prompts —
  prose speaks names; the score JSON the model RETURNS is keyed by short
  per-prompt aliases (`Q`/`P1`/`P2`…, see `build_batch_scoring_prompt`) and
  translated back, so returned fields stay keyed by real id. Ids without a
  display name fall back to the raw id as the profile's name label — pass
  names whenever ids are opaque. Include a `{"__query__": <asker name>}` entry
  so query intros address the asker by name. Post-hoc string surgery on
  generated prose is a losing game; pass names in.
- Returns a `QueryMatchResult` (shortlist + intros) — **returned, never
  written**. JSON wrapper for agent tool-calls: `query.run_query_match_json`
  (accepts `store_dir` for the FileStore path or an inline pool dict, plus
  `exclude_ids` / `display_names` / `generate_intros` payload keys).

## Mode C: subset batch with novelty

`run_batch_match(member_ids, pool, config, excluded_pairs, pool_sections=…)`:

- **Members are caller-supplied ids** (⊆ pool). Choreo never reads a `tier`
  flag or decides who's a member — that's app data.
- **`pool_sections` can be lazy here too** (`sections_provider`): invoked once
  after budgeted pair selection with union(users in selected pairs,
  member_ids) — the only users whose text the LLM scoring, intros and member
  reports read (members ride along because reports render zero-pair members).
  One of `pool_sections` / `sections_provider` is required.
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
  novelty_window_months: 6    # exclusion window (adapters apply it — Mode C
                              # excluded_pairs, Mode B exclude_ids)
query:
  top_k: 4
  llm_rerank: true            # false = pure-embedding, cheaper
  rerank_pool_multiplier: 4   # over-fetch factor for the re-rank (1 = reorder-only)
  generate_intros: true       # true | int top-N | false
  # no packaged query.recipe — pass recipe_override per call, or fall back
  # to the top-level `recipe`
```

CLI: `--pipeline query_match --query '…'` · `--pipeline batch_match --members a,b`.
