# Improvement sprint — algorithm audit findings & prioritized upgrades

**Status:** Phase 0 + the safe Track 4/5 integration-surface items **SHIPPED
2026-07-07** (see §1 residue + §5/§6 markers; `choreo_IO.md` updated in the
same change per the shipping contract). Remaining: Track 1 (eval harness —
build FIRST), Track 2 (HyDE upgrade), Track 3 (scoring/blending), the deferred
Track 4/5 stragglers, and discovery mode. Written 2026-07-07 after a full-repo
audit, in the context of Choreo becoming the core matchmaking + serendipity
engine inside the motherbrain community-agent platform
(`motherbrain/docs/TODO/01_choreo_matchmaking_integration.md`, hereafter
**plan 01**; profile input spec: plan 11 §11 in
`motherbrain/docs/TODO/11_collective_memory_architecture.md`).

**What this doc absorbs:** the FUTURE-TODO tail of `upgrade_plan.md` (now
archived in `docs/finished/`; multi-descriptor HyDE, community-aware HyDE,
bidirectional HyDE — all re-scoped below) and the recommendations of
[scoring_calibration.md](scoring_calibration.md) (kept as the analysis appendix
for Track 3; retire it to `finished/` when that track ships). Discovery mode is
its own plan at [discovery_mode.md](discovery_mode.md) (relocated from the repo
root during this audit and merged with this sprint's positioning notes).
IO-affecting items are mirrored into motherbrain plan 01 (aligned 2026-07-07 —
see the **IO-surface impact summary** at the end of §5/Track 4).

**Reading order for the implementing agent:** this file →
[reference/matching_modes.md](../reference/matching_modes.md) →
[reference/stages_and_adapters.md](../reference/stages_and_adapters.md) →
the specific modules named per track.

---

## 0. The algorithm as-built (one paragraph, for orientation)

Profiles → LLM section extraction (`skills/vision/project/needs`,
`extract.py`) → HyDE rewrite of `needs` into skills-vocabulary
(`hyde.py`, 1 descriptor) → embeddings with per-cell content-hash reuse
(`embed.py`, gemini-embedding-2 + MRL truncation) → rectangular fused
similarity: symmetric same-section terms + a directional `needs→skills` cross
term with per-pair absence masking (`candidate.py`) → budgeted batched LLM pair
scoring via greedy set-cover groups (`score.py`) → min-max-normalized blending
(0.35 embed / 0.65 LLM) → greedy b-matching (`match.py`) → directional intros
(`introduction.py`). Three trigger shapes: full cohort (N×N), query (1×M hot
path, `query.py`), subset batch (M×N with novelty exclusions,
`batch_match.py`). In production (plan 01) the extract stage is bypassed —
the memory engine's librarian delivers pre-normalized facet digests, and
Choreo runs (HyDE?) + embed + match only.

---

## 1. Findings — Phase 0 residue (ALL SHIPPED 2026-07-07)

All Phase-0 fixes landed with offline tests (`tests/test_sprint_phase0.py`)
and doc/IO-spec updates in the same change. What a future agent needs:

- **F1 over-fetch (shipped):** `run_query_match` now LLM-re-ranks
  `top_k * query.rerank_pool_multiplier` (default 3) embedding candidates and
  returns the re-ranked top `top_k` — recovery, not just reorder. Multiplier 1
  = legacy behavior. Intros stay limited to the final K.
- **F2 absent-text gate (shipped):** canonical `utils.is_absent()` (empty /
  whitespace / `"Not specified"`); applied in `embed.embed_sections` (checked
  BEFORE content-hash reuse so phantom placeholder vectors in old bundles get
  zeroed on next upsert) and in `hyde_descriptors_for_sections` (absent source
  → no LLM call, empty descriptors → masked out). Normalizing to `""` at
  extraction output remains the cleaner long-term shape — still open, low
  priority now that every consumer gates on `is_absent`.
- **F3 HyDE cache key (shipped):** `hyde_cache_key(source_text, n, cross_key,
  context_fingerprint)` where `hyde_context_fingerprint()` hashes template +
  goal + model + language + section guidelines. Editing the HyDE prompt/goal/
  model now regenerates descriptors (one-time disk-cache invalidation on
  deploy was accepted). NOTE: extraction's LLM cache is still keyed on
  `extract_{profile.hash}` only — same bug class, unfixed; extraction prompt
  edits need `--force` (documented in config.yaml).
- **F4 `display_names` (shipped):** optional `{user_id: name}` on all three
  runners + `run_query_match_json` payload; threaded into scoring/re-rank
  prompts (`<profile id name>`, score JSON keyed by id) and intro prompts +
  prose + fallbacks (ids don't appear at all). `{"__query__": asker}` names
  the query side (the old `query_label` idea was dropped — the map entry
  covers it). Without the kwarg, prompts are byte-identical → caches warm.
- **F5 Mode-B novelty (shipped as docs):** mapping documented in
  [reference/matching_modes.md](../reference/matching_modes.md) +
  `choreo_IO.md` — adapters pass the asker's recent partners as `exclude_ids`;
  no pair-id mechanism on the query path. The optional `recipe_override`
  kwarg for `run_batch_match` was NOT added (config overrides express it).
- **F7 hygiene (shipped):** `json_format_hint` is real JSON now
  (`{"a_b": "0..1"}` via `json.dumps` — the fake-LLM test responders parse
  this shape); CLAUDE.md config sample synced to actual defaults; root strays
  archived (`upgrade_plan.md`, `analysis_report.md`, `final_test.md` →
  `docs/finished/`) and dead pre-refactor scripts deleted (`replot_v2.py`,
  `debug_async.py` — in git history if ever needed).

### F6. Blending rests on fragile min-max remaps (OPEN — see Track 3)
`utils.normalize_scores_with_reference_distribution` maps LLM scores into the
min-max range of the selected candidates' embed scores; embed scores are
min-max normalized against a reference distribution. Two consequences: single
outliers set the scale, and the query path had to **deliberately bypass** the
remap (see the NOTE in `query.py`) — so cohort/batch and query mode now have
divergent blending semantics. Not a bug per se, but the root cause of the
open calibration question in [scoring_calibration.md](scoring_calibration.md).
Addressed structurally in Track 3.

---

## 2. Track 1 — Eval harness (the enabler; build first)

Every remaining track is a *quality* change to prompts, HyDE, or blending —
and today there is **no way to measure match quality** beyond eyeballing
reports. Tuning prompts blind is how quality regressions ship. Small, concrete
harness:

1. **Fixture cohort** (`evals/fixtures/`): ~20 authored profiles, deliberately
   diverse (tech/art/business/craft, some sparse, some multilingual, some with
   quirky needs). `discovery_mode.md` §7 needs the same fixture — author once,
   share. Keep `data/test8` as the small smoke cohort.
2. **Gold labels** (`evals/gold/`): a strong judge model (via OpenRouter, e.g.
   a frontier reasoning model at high effort) scores all C(20,2) pairs
   directionally with a written rationale, cached to disk = the gold matrix.
   Overlay ~15 hand-curated pairs (must-match / must-not-match / one-way-only)
   as hard assertions. Regenerate only deliberately (it IS the benchmark).
3. **Metrics** (`evals/run.py`, a plain script):
   - *Retrieval quality* — nDCG@k / recall@k of the **embedding stage alone**
     against gold, per variant. This isolates HyDE changes without paying for
     the full pipeline.
   - *Scoring quality* — Spearman correlation of pair-LLM scores vs gold;
     score histogram (detects the everything-is-0.7 clustering failure).
   - *End-to-end* — precision of final edges vs gold top pairs; per-user
     "got at least one gold-good match" coverage.
   - *Stability* — rank churn across two identical runs (should be ~0 given
     the determinism work already done).
   - *Cost/latency* per mode, from the existing `cost_tracker`.
4. **Variant switching is free**: everything under test is already
   config-driven (`--config-dir` overlay per variant), and the content-hash
   caches make re-runs that share stages cheap.

Deliverable: one command produces a comparison table for N config variants.
Est. 1–2 focused days. Everything in Tracks 2–3 lands with a before/after row.

---

## 3. Track 2 — HyDE upgrade (the headline: multiple descriptors per need)

Current state: `n_descriptors: 1`; the data path (list-based descriptors,
max-pool over descriptor pairs in `candidate.py`) supports k>1 end-to-end but
has never run in production. Raising the int is the trivial part — the value
is in *what the descriptors are* and *how they aggregate*.

### 3.1 Decompose, don't paraphrase (prompt semantics change)
The current prompt asks for k phrasings of "the ideal match". k paraphrases of
the same sentence buy little — they cluster in embedding space. The upgrade:
ask the LLM to **decompose the needs section into up to k distinct sub-needs /
solution angles**, one descriptor each ("make my installation respond to
movement" → computer-vision descriptor, capacitive-sensing descriptor,
creative-technologist descriptor). This is what actually widens the net, and
it matches how needs really arrive (a facet digest contains several needs).
Prompt rewrite checklist (`defaults/hyde_prompt.yaml`):
- decompose into *distinct* angles; explicitly forbid near-duplicates;
- if the source contains fewer than k real needs, return fewer descriptors
  (variable-length output; the fixed-k padding fallback in `hyde.py` that
  pads with raw source text should become "pad with nothing" — zero vectors
  mask out naturally);
- embedding-friendly style: dense, concrete, noun-phrase-heavy, 1–2 sentences,
  no filler ("I am able to…");
- keep the existing wide-net instruction (obvious + unexpected angles);
- fix the typo ("a ideal match") while in there.

### 3.2 Pooling semantics as a config knob
Global hard max (current) is right for "any strong hook" matching, but has an
upward-drifting noise floor as k grows, and it can't express "covers most of
my needs". Add `hyde.pooling: max | mean_of_max | softmax` (softmax =
logsumexp with temperature ≈ a soft max):
- `max` — best single angle; keep as default for intro-driven matchmaking.
- `mean_of_max` — per-source-descriptor max over target, then mean = coverage
  ("how much of what I need does this person address"); interesting for
  co-founder-type queries.
- `softmax(τ)` — the robust middle ground; τ in config.
Implementation is a few lines in `candidate.compute_fused_similarity_matrix`
where the descriptor loop currently does `np.maximum`. Evaluate all three on
the Track-1 retrieval metric before choosing defaults.

### 3.3 Diversity guard (no extra LLM cost)
After embedding descriptors, drop any descriptor with cosine > ~0.92 against
an earlier descriptor of the same (user, cross_key) — the LLM will sometimes
paraphrase despite instructions, and duplicates waste the k budget and bias
max-pooling. Cheap post-filter in `embed.embed_sections`'s HyDE branch or as
a step in `hyde.py`.

### 3.4 `matched_via` — surface which descriptor fired
Record the argmax descriptor index per (i, j) cell alongside the cross matrix
(`matrices_dict`), and thread the winning descriptor **text** onto edges /
query shortlist rows as `matched_via`. Two consumers: (a) the scoring and
intro prompts can cite the actual bridge ("matched via: needs someone who can
do computer vision for interactive installations") — grounds the LLM in the
evidence that produced the candidate; (b) agents/reports can explain *why* a
match surfaced. Big explainability win for the platform ("the brain suggests
WHY, not just WHO"). IO note: surfacing the winning *text* requires the bundle
to carry descriptor texts — add an additive `hyde_texts` field to
`EmbeddingsBundle` (`{cross_key: {user_id: [descriptor, …]}}`, included in
`to_dict`/`dump`; legacy bundles load with it empty and `matched_via` degrades
to a descriptor index). External stores that persist descriptor text
(motherbrain's `world_user_profile_hyde` table, plan 01 §3.3) populate it for
free.

### 3.5 Community-aware HyDE (recall lever, adapter-friendly)
The best vocabulary bridge depends on what the pool actually offers (the
original design insight in `upgrade_plan.md`). Add an optional
`community_context: str` slot to the HyDE prompt + `run_query_match` /
Mode-A entry points — the **adapter** supplies a compact skills-landscape
digest (one cached LLM summary of the pool's skills sections; motherbrain
already builds a community pulse, plan 11 §9.2). Choreo stays unopinionated:
it's just a prompt slot, empty by default. Cache correctness comes free once
F3 lands (key = full-prompt hash → new context invalidates descriptors);
adapters bound refresh frequency (regenerate digest on ~10% roster change) so
member-side descriptors don't churn weekly.

### 3.6 Two eval-decided questions (don't pre-commit)
- **Member-side HyDE skip** (plan 11 §11.5): production facets arrive with
  needs already phrased in skills-vocabulary, so Mode A might embed facet text
  directly and halve its LLM work. Run as a Track-1 variant: descriptors on
  both sides vs query-side only.
- **Bidirectional HyDE** (skills → needs-vocabulary as a second bridge):
  doubles member-side HyDE cost; only pursue if the retrieval metric shows a
  recall gap that 3.1–3.5 don't close.

---

## 4. Track 3 — LLM scoring & blending quality

Companion analysis: [scoring_calibration.md](scoring_calibration.md) (batch
anchoring mechanics, why random batching is the wrong fix). The levers, in
its ROI order, made concrete:

### 4.1 Rubric anchoring + few-shot exemplars (`defaults/scoring_prompt.yaml`)
Add 2–3 calibrated exemplar pairs with scores and one-line rationales
("0.9 because her embedded-systems skills directly unblock his LED wall;
0.35 because both 'do AI' but neither fills a stated gap"). Add explicit
anti-clustering instructions: use the full 0–1 range, steps of 0.05, "most
pairs in a community are < 0.5". Add one line that matters a lot for the
Wintercircus/Zwerm context: **generic domain overlap (both "work with AI") is
not a match** — score specific complementarity, not shared buzzwords. This is
the cheapest calibration lever and also the "better llm prompts" ask.

### 4.2 Shared calibration anchors per batch
Inject 1–2 fixed synthetic reference profiles into every scoring group (their
pair scores are parsed and discarded). Every batch then shares a contrast
frame, converting the drifting-anchor bias documented in scoring_calibration
into a constant offset. ~Constant cost (one extra profile per prompt).

### 4.3 Directional pair scores (eval lever, architectural payoff)
The prompt already makes the model reason about both directions, then throws
the asymmetry away in a single number. Ask for
`{"a_to_b": …, "b_to_a": …}` per pair and derive the symmetric edge weight in
code (`max`/`mean`/harmonic — config). Payoffs: batch mode can rank by the
direction that matters (pool→member help); reports/intros get honest
one-way-match framing; and the stored directional scores become training
signal for the future learned matcher (plan 11 §11's outcome loop).
`upgrade_plan.md` deliberately chose single-score ("LLM is better at holistic
judgment") — treat that as a hypothesis and let the Track-1 Spearman metric
decide, not dogma.

### 4.4 Replace the min-max remaps with rank-based normalization
Swap the affine remap chain (F6) for ECDF/percentile normalization of embed
scores against the reference distribution, and blend **raw rubric-anchored LLM
scores** directly (they become absolute once 4.1/4.2 land). This unifies
cohort/query blending semantics (the query-mode divergence NOTE in `query.py`
disappears), makes stored `final_weight`s comparable across runs — which
`choreo_matches.final_weight` in plan 01 silently assumes — and removes
outlier sensitivity. Keep the legacy path behind a config flag for one release
(`blending.normalization: rank | legacy_minmax`) with a regression eval row.

### 4.5 Global top-K calibration pass (optional, cheap)
After batched scoring, re-score the global top ~2·b_max·N candidate pairs
together in 1–2 shared-frame calls and blend with round-1 scores. Only the
contenders' relative order affects matching, so this buys cross-batch
consistency exactly where it matters. Ship behind config
(`budgets.global_rerank_pairs: 0 = off`).

---

## 5. Track 4 — Integration surface (what plan 01 actually needs from Choreo)

Items 1–3 **SHIPPED 2026-07-07** (residue below); item 4 stays deferred.

1. **Inline prompt-text overrides (shipped):**
   `resolve_prompt_templates(config_dir=…, config=…, prompt_paths=…)` in
   `config.py` returns `{"sections": dict, "scoring"/"introduction"/"hyde":
   template str}`; inline `prompts.<name>_prompt_text` config keys beat paths
   beat packaged defaults. All three runners + `deploy_modal.upsert_profiles`
   consume templates (no file IO downstream); `worlds.settings.choreo` can now
   carry prompt text directly. `section_prompt_text` accepts YAML text or a
   parsed dict.
2. **`display_names` (shipped)** — see §1 F4.
3. **Language pinning (shipped):** `instruction_prompt.language` (default
   null = match each profile's language) renders an `{output_language}` line
   in the extraction + HyDE templates and is part of the HyDE cache
   fingerprint. Gotcha: switching language on an existing cohort needs
   `--force` for extraction (its reuse is content-hash keyed); HyDE picks it
   up automatically.
4. **Soft novelty (nice-to-have, DEFERRED)** — binary `excluded_pairs` is the
   only history mechanism; a generic `pair_weight_multipliers: {pair_id: float}`
   input on batch mode would let adapters *decay* recently-surfaced-but-
   unacted pairs instead of hard-excluding, and boost pairs flagged by
   outcome feedback. One multiplication in `compute_final_weights`. Defer
   until an adapter asks, but keep in mind when touching `match.py`.

### IO-surface impact summary (cross-repo alignment, 2026-07-07)

Which sprint items change Choreo's external surface, and what the motherbrain
wrapper does about it — plan 01
(`motherbrain/docs/TODO/01_choreo_matchmaking_integration.md`) was updated in
lockstep on this date. Rule: `choreo_IO.md` documents what IS — update it as
each item **lands**, never ahead of code. Everything below is
backward-compatible (additive kwargs/fields or flag-gated semantics); no
existing caller breaks.

| Sprint item | Status | Surface change | Wrapper impact (plan 01) |
|---|---|---|---|
| F4 `display_names` | ✅ shipped | new optional kwarg on all three runners + JSON-wrapper payload key | §4.1/§4.2 pass roster names in; the Task-B dependency is now available |
| §5 item 1 inline prompts | ✅ shipped | config keys `prompts.<name>_prompt_text` + `resolve_prompt_templates` | `worlds.settings.choreo` can carry prompt text; plan 01 §4.3 updated |
| F1 over-fetch, language, `generate_intros: top_n`, F2/F3 fixes | ✅ shipped | config-only / internal (`query.rerank_pool_multiplier`, `instruction_prompt.language`, int-valued `generate_intros`) | none — new config keys ride the existing overrides layer |
| 3.4 `matched_via` + bundle `hyde_texts` | open (Track 2) | new fields on shortlist rows / `Edge.to_dict`; additive bundle field (legacy bundles load fine) | persisted via new `choreo_matches.details` jsonb |
| 3.1/3.2 multi-descriptor HyDE | open (Track 2) | `hyde[cross_key]` stays `(n_users, k, dim)` but k>1, per-user counts vary (zero-padded) | per-descriptor rows needed → plan 01 §3.3 gained `world_user_profile_hyde` |
| 4.3 directional pair scores (if adopted) | open (Track 3) | additive fields on `PairScore`/`Edge` | `choreo_matches.details` |
| 4.4 rank normalization | open (Track 3) | `final_weight` becomes cross-run comparable — semantic change, no shape change, flag-gated | plan 01 §3.3 notes weights are within-run ordinals until this lands |

The alignment pass also caught a **pre-existing plan-01 schema bug**
independent of this sprint: its original `world_user_profile_sections` stored
ONE vector per (user, section) cell and claimed it embedded the HyDE text.
That cannot rebuild an `EmbeddingsBundle` — content embeddings (same-section
terms + the target side of cross terms) and HyDE embeddings (source side,
keyed by *cross_key*, not section) are **separate arrays**. Plan 01 §3.3 now
splits them into two tables.

---

## 6. Track 5 — Hot-path efficiency (query mode)

`find_matches` is the agent-facing, seconds-matter path. Current LLM
round-trips for top_k=5: HyDE (1) → rerank (~2 chunked) → intros (5), in
three serial waves.

1. **Decision (2026-07-07): scoring and intro generation stay SEPARATE LLM
   steps — do not fuse.** A fused score+intro completion was considered and
   rejected: one job per call; the score-contamination risk (the judge biased
   toward eloquently-introducible candidates) isn't worth the wave saved.
   Latency levers within that constraint:
   - With F1's over-fetch active (the recommended default), the intro wave
     genuinely depends on the re-rank outcome — keep it serial and **shrink**
     it instead: `generate_intros: top_n` (item 2) caps the third wave at the
     candidates an adapter will actually render.
   - Without over-fetch (`rerank_pool_multiplier: 1`), the intro set equals
     the embedding shortlist (re-rank only reorders it), so the re-rank and
     intro waves can run **concurrently** — one `gather` over both batches
     through the existing semaphore, intros attached to the re-ranked order
     afterwards. Same latency win fusion promised, zero prompt coupling.
2. **`generate_intros: top_n`** — ✅ SHIPPED 2026-07-07: `generate_intros`
   accepts an int (intros only for the final N) in addition to bool, on the
   kwarg, the config key and the JSON payload.
3. **Parallel embedding batches (OPEN)** — `_embed_texts_batched` (`embed.py`)
   runs 100-text batches sequentially; a thread pool over batches cuts Mode-A
   latency for large pools. Minor; do opportunistically.

---

## 7. Track 6 — Discovery mode (serendipity beyond pairs)

Moved to its own plan: **[discovery_mode.md](discovery_mode.md)** (the full V1
MVP + V2 roadmap, relocated from the repo root and merged with this sprint's
positioning notes). The coupling that remains this sprint's concern: the
Track-1 fixture cohort doubles as its test cohort, and Track-2's HyDE work
feeds its §6.7 complement-attraction upgrade.

---

## 8. Suggested sequencing

| Phase | Contents | Size | Gate |
|---|---|---|---|
| 0 | ✅ **DONE 2026-07-07** — F1 over-fetch, F2 absent-text gate, F3 HyDE cache key, F4 display_names, F7 hygiene + doc moves (tests: `tests/test_sprint_phase0.py`) | — | shipped with choreo_IO.md update |
| 1 | Track 1 eval harness + fixture + gold labels — **the next phase to build** | 1–2 days | baseline table exists |
| 2 | Track 2 HyDE upgrade (3.1–3.5), eval 3.6 variants | 2–3 days | retrieval metric ≥ baseline |
| 3 | Track 3 scoring & blending (4.1→4.4; 4.5 optional) | ~2 days | Spearman + e2e ≥ baseline; blending unified |
| 4 | ✅ **mostly DONE 2026-07-07** — inline prompts, language, `generate_intros: top_n` shipped; still open: soft novelty (§5.4), parallel embed batches (§6.3) — both deferred/opportunistic | — | plan 01 wishlist closed |
| 5 | Discovery mode V1 ([discovery_mode.md](discovery_mode.md)) | separate sprint | its own §5.8 baseline |

Phases 2 and 3 are independent after Phase 1 and can run in parallel. Every
phase ends with the standard docs pass: condense the completed section here,
promote durable residue into
[reference/matching_modes.md](../reference/matching_modes.md) /
[reference/stages_and_adapters.md](../reference/stages_and_adapters.md), and
keep [mvp_test_sequence.md](mvp_test_sequence.md) honest (F1/F4 change a few
of its assertions: shortlist provenance, intro phrasing).

**Shipping contract with motherbrain plan 01:** any phase that ships an
IO-surface item (§5 summary table) MUST update `choreo_IO.md` in the **same
change** — plan 01 builds against that spec, not against this plan, and
treats a `choreo_IO.md` diff as the signal that an item is actually
available. An IO-affecting phase is not done until `choreo_IO.md` reflects
it.

---

## 9. Explicitly out of scope (with revisit triggers)

- **Learned matching embedding** (train proximity = match quality on facet
  texts + outcome records) — plan 11 §11 owns the research track; Choreo's
  contribution for now is Track 3's directional scores + `matched_via`
  provenance landing in stored matches, i.e. the training data.
- **ILP/optimal collective matching** to replace greedy b-matching — no
  evidence greedy is the quality bottleneck; revisit if Track-1 e2e metrics
  show good pairs being crowded out by degree caps.
- **SQL-side ANN retrieval** — plan 01 §3.3 documents pgvector as storage
  only until a world passes ~1k members; in-memory rectangular similarity is
  fine at current scale.
- **Embedding model migration tooling** — still parked in
  [external_adapter_integration.md](external_adapter_integration.md).
