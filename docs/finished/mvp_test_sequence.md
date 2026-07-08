# TODO — MVP end-to-end test sequence (post-integration acceptance run)

**Status:** not yet run. Execute once choreo is wired into the agent
(motherbrain exposes the Modal endpoints as MCP tools via the
`call(tool_key, identity, args)` dispatch; the webapp imports choreo directly
against Neon). This is the acceptance script for the **full production
configuration** — live OpenRouter key, deployed Modal app, real Volume.

It deliberately exercises the components added/fixed in the June 2026 polish
pass, each of which has a failure mode that unit tests can't see in the real
deployment: freshness timestamps, content-aware LLM cache invalidation,
unscored-pair fallback, async-host safety, the warm pool cache +
`volume.reload()`, and conditional `volume.commit()`.

Conventions used below:
- `GROUP=mvp_acceptance_<date>` — always a **fresh group name**, never reuse.
- "endpoint calls" = `modal.Function.from_name("choreo-matching", "<fn>").remote(...)`
  or the MCP tool wrapping it — run each phase through the same path the agent
  will actually use.
- Keep the Modal app logs open in a second terminal:
  `modal app logs choreo-matching`.

---

## Phase 0 — Deploy + smoke

1. `uv run modal deploy deploy_modal.py` (fresh deploy of the current code).
2. Confirm secret wiring: call `query_match` for a nonexistent group → expect
   clean `{"success": false, "error": "No embeddings for group ... — call
   upsert_profiles first."}` (not a traceback).

## Phase 1 — Cold upsert (Mode A) with external timestamps

Use 6–10 realistic profiles (the Wintercircus-style bios), **with explicit
timestamps** in the new dict shape, as the Neon wrapper will send them:

```json
{"alice": {"text": "<bio>", "last_updated_at": "2026-06-01T10:00:00+00:00"},
 "bob":   {"text": "<bio>", "last_updated_at": "2026-06-02T09:30:00+00:00"},
 "carol": "<plain text — no timestamp, should default to call time>"}
```

Verify:
- [ ] Response: `success`, `roster_size`, `embedding_model`, `embedding_dim`
      all sane; `failed` is empty.
- [ ] On the Volume (`modal volume get choreo-data groups/$GROUP/...` or a
      debug shell): `embeds/bundle_meta.json` →
      `user_timestamps.alice == 2026-06-01T10:00:00+00:00` (the **supplied**
      value, not call time), `user_timestamps.carol` ≈ call time.
- [ ] `processed/sections.jsonl` rows carry the same `last_updated_at`.
- [ ] Logs show every cell embedded fresh (`0 reused, N to embed`).

## Phase 2 — Incremental upsert (the content-hash + timestamp contract)

Re-send **the same payload** but with: alice's text *edited* (and a bumped
timestamp), bob byte-identical but with a **newer timestamp**, carol untouched.

Verify:
- [ ] Logs: only alice is extracted/embedded (`reused` count covers everyone
      else). **Bob's identical text must NOT re-embed** despite the newer
      timestamp — timestamps are provenance, content hashes drive compute.
- [ ] `bundle_meta.json`: alice's and bob's `user_timestamps` updated; carol's
      unchanged.
- [ ] `is_stale()` round-trip (the Neon-side freshness check the wrapper will
      run): `choreo.is_stale(bundle.user_timestamps["alice"],
      neon_row.updated_at)` is `False` after the upsert, `True` if you bump
      the Neon row again without upserting. This is the loop the agent uses
      to decide *when* to call upsert — verify both directions once.

## Phase 3 — Query match hot path (Mode B) + warm-cache behavior

1. First query: `{"query": {"needs": "someone who can build the agent
   backend"}, "top_k": 5}`.
   - [ ] `success`, `llm_rerank_applied: true`, shortlist has intros, `notes`
         empty (in particular **no** "llm_rerank requested but pool_sections
         not provided" — the endpoint must be passing sections from the pool
         cache).
2. **Identical** query again, same warm container (back-to-back call).
   - [ ] Visibly faster; logs show cached LLM results
         (`Found N cached results`) and **no** `vectors.npz` reload.
   - [ ] Logs do NOT show a `volume.commit()`-related write for this call
         (conditional commit: nothing new was cached). If logging is too
         coarse, verify instead that repeated cache-hit queries don't grow
         the Volume.
   - *(sprint 2026-07-07)* The re-rank now **over-fetches**: with defaults it
     LLM-scores `top_k * 3` embedding candidates and returns the re-ranked top
     `top_k` — so a shortlist row that was NOT in the embedding top-5 is
     expected recovery behavior, not a bug, and rerank cost per query is ~3×
     the old wave. A note `"Re-ranked N embedding candidates (over-fetch …)"`
     appears in `notes`.
3. Same query with `"llm_rerank": false` → `llm_rerank_applied: false`,
   `llm_score: null` in shortlist rows, still ranked sensibly.
4. `"exclude_ids": ["<top hit from #1>"]` → that user absent, next-best
   promoted.
5. `"display_names": {"<top hit>": "Some Name", "__query__": "Asker"}` →
   intro prose says "For Asker: …" and uses the candidate's name; no raw ids
   in the intro text (sprint F4).
6. Sanity-check ranking quality by eye: the top hits should actually address
   the stated need (this is the product, not just the plumbing).

## Phase 4 — Stale-cache regression (the bug class we fixed)

This is the single most important new check. With caches **warm** from
Phase 3:

1. Upsert an edit to a profile that appeared in the Phase-3 shortlist —
   change their `skills` so they become clearly *better* (or clearly worse)
   for the Phase-3 query.
2. Re-run the **identical** query from Phase 3 (no force flags anywhere).

Verify:
- [ ] The shortlist/intros reflect the **edited** profile text (rank moves in
      the expected direction; intro mentions the new skill). A stale-cache
      regression shows up as identical scores/intros to Phase 3 — that's a
      fail.
- [ ] Logs confirm fresh LLM calls for chunks containing the edited user
      (their prompt hash changed), while untouched chunks may still cache-hit.
- [ ] Repeat the same edit-then-rerun probe via `batch_match` once Phase 5 is
      set up (same invalidation contract, different code path).

## Phase 5 — Batch match (Mode C): novelty + history

1. `batch_match(["alice","bob"], group=$GROUP)`.
   - [ ] Edges returned with intros; `report_data.user_reports` contains
         **members only**; `new_pairs` non-empty; `excluded_count: 0`.
   - [ ] Volume: `match_history.jsonl` grew by exactly `len(new_pairs)` rows.
2. Run the identical call again immediately.
   - [ ] `excluded_count == len(history)` for these members; returned pairs
         are **disjoint** from run 1 (novelty window working), or empty with
         a clear degree/“no eligible pairs” story if the pool is small.
3. Pass duplicate member ids `["alice","alice","bob"]` → behaves identically
   to `["alice","bob"]` (dedupe guard).
4. Window expiry: temporarily set
   `config_overrides_json={"matching": {"novelty_window_months": 0.0001}}`
   (≈ minutes) → previously surfaced pairs become eligible again.

## Phase 6 — Unscored-pair fallback (embedding-only edges)

Force the budget-starved path with overrides, e.g.
`{"budgets": {"max_pair_llm_calls": 1}}` on a `batch_match` call:

- [ ] Logs show the `pair(s) left UNSCORED` / `fall back to embedding-only
      weight` warnings.
- [ ] The result still contains edges for members whose pairs were never LLM
      scored — `llm_score: 0.0` / `llm_score_normalized: 0`, `final_weight`
      equal to the normalized embed score. **No member silently ends up with
      zero matches** because of the budget.

## Phase 7 — Async-host safety (the MCP integration path)

Run a query **through the agent**, i.e. from inside motherbrain's running
event loop (the MCP tool handler), not from a sync script:

- [ ] `llm_rerank_applied: true` and intros present. The historical failure
      mode is *silent*: `asyncio.run` raising inside the host loop was
      swallowed and the result degraded to embedding-only with rerank
      skipped — so explicitly assert rerank/intros actually ran, don't just
      check `success`.
- [ ] Also exercise the webapp's direct-import path (`run_query_match` from
      its async backend) once it exists — same assertion.

## Phase 8 — Multi-container / multi-writer consistency

1. Force a second container (e.g. two concurrent `query_match` calls, or
   bump the app to scale out, or simply wait for a cold start).
2. Upsert a brand-new profile from container A, then immediately query for
   something only that new profile can satisfy from container B.
   - [ ] The new user appears in B's shortlist (`volume.reload()` + pool-cache
         signature invalidation working across containers).
3. Confirm the pool cache isn't serving a stale roster after upsert:
   `pool_size` in the query result must equal the new roster size.

## Phase 9 — Config override surface (what the agent will actually tune)

One call each, verifying the override is *observably* applied:
- [ ] `{"query": {"top_k": 3}}` → 3 rows.
- [ ] `{"models": {"pair_reasoning_effort": "low"}}` → visible in OpenRouter
      dashboard / cost report deltas (the old hardcoded "medium" ignored
      this — that's the regression to watch).
- [ ] `{"blending": {"embed_weight": 1.0, "llm_weight": 0.0}}` → ranking
      equals pure embedding order from the `llm_rerank: false` run.

## Phase 10 — Cost + wrap-up

- [ ] Pull `cost_report.json` / OpenRouter usage for the whole sequence;
      sanity-check per-query cost on the hot path (this is the number that
      scales with agent usage).
- [ ] Note observed warm-query latency (target: dominated by the LLM rerank,
      not by pool loading).
- [ ] Delete or archive the `$GROUP` directory on the Volume.
- [ ] Record outcomes per checkbox here; promote anything durable
      (gotchas, real latencies/costs, deviations) into
      [reference/stages_and_adapters.md](../reference/stages_and_adapters.md) /
      [reference/matching_modes.md](../reference/matching_modes.md), then move
      this file to `docs/finished/`.

---

## Quick matrix — new component → phase that proves it

| Component (June 2026 pass) | Proven by |
|---|---|
| `last_updated_at` injection + propagation (`user_timestamps`) | 1, 2 |
| Timestamps ≠ compute trigger (hashes still rule) | 2 (bob) |
| `is_stale()` adapter loop | 2 |
| Prompt-hash LLM cache invalidation (scores **and** intros) | 4 |
| `unscored_out` → embedding-only edges survive | 6 |
| `run_coro_blocking` in async host | 7 |
| Warm pool cache + truncation memo | 3 (call 2) |
| `volume.reload()` cross-container freshness | 8 |
| Conditional `volume.commit()` | 3 (call 2) |
| Inline-pool precedence over store pool | optional: send a tiny inline `pool` in a query payload → results come from it, not the group |
| `pair_reasoning_effort` config key | 9 |
| Member-id dedupe / novelty window | 5 |
