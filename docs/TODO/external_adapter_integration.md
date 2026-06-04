# TODO — External (Neon) adapter integration follow-ups

**Status:** open items carried over when the granular-refactor plan closed out
(June 2026; husk in `docs/finished/01_todo.md`). The architecture these items
validate is documented in
[reference/stages_and_adapters.md](../reference/stages_and_adapters.md).
The actual Neon adapter + Modal tool now being built lives in the motherbrain
repo — see `motherbrain/docs/TODO/04_integrate_choreo_tool.md` (choreo stays
unopinionated; all Neon/Modal/world plumbing is motherbrain-side).

- [ ] **Confirm the `Store` protocol surface against the real Neon wrapper.**
      `choreo/store.py` defines `get/put_sections`, `get/put_embeddings`,
      `get_match_history`, `put_matches` and the stage registry
      (`stages.describe_all_stages()`) declares every stage contract. When the
      community platform's Neon adapter is actually written, verify this is
      the *complete* set of contracts it needs — extend the protocol here if
      gaps surface (e.g. partial-bundle fetches, per-section embedding rows).

- [ ] **Embedding model migration** (explicitly out of scope in the refactor):
      when `models.embedding` changes, every stored vector is stale. The
      bundle already carries `embedding_model` + `dim` provenance and a
      mismatched bundle is ignored (full re-embed), but there is no managed
      bulk re-embed path or `model_version` guard for external stores yet.

- [ ] **Query intro phrasing:** intros for query mode address the pseudo-user
      as `__query__` in the generated prose ("For __query__: …"). Fine for
      machine consumption; if these intros ever go user-facing verbatim, add a
      display-name substitution at the product layer or a `query_label`
      parameter to `run_query_match`.
