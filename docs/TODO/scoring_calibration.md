# TODO — LLM pair-scoring calibration (batch anchoring & cross-run stability)

**Status:** design residue, not yet implemented. Carried over from the
granular-refactor plan (now in `docs/finished/01_todo.md`) when it closed out
in June 2026.

## Open question: cross-run score normalization endgame

The refactor made the normalization reference an **explicit input**
(`utils.prepare_normalized_scores(reference_scores=…)`; rectangular modes pass
their member×pool / 1×M similarity values — see
[reference/matching_modes.md](../reference/matching_modes.md)). Still open:
whether to go further and switch to an **absolute / rubric-anchored** LLM
score so stored scores are comparable across runs without any reference
bookkeeping. That decision ties directly into the anchoring analysis below.

---

## Appendix A — LLM pair-scoring: batch anchoring & global calibration (prior note, preserved)

Context: investigated whether to bump `budgets.n_profiles_to_score_together` (currently 4)
and, more deeply, how the batched LLM scoring interacts with global ranking quality.
Nothing changed in code yet — this is the design residue to revisit.

### How scoring batches are built today (the relevant mechanics)

- `select_pairs_for_llm_scoring_optimal` (`choreo/score.py:180`) picks which pairs to score,
  bounded by `max_n_llm_evaluations_per_profile` (per-user) and `max_pair_llm_calls` (global).
- `create_profile_groups_from_pairs` (`choreo/score.py:25`) then packs the *selected* pairs into
  user-groups via greedy **set-cover**: seed each group from the highest-priority uncovered
  pair, grow it to size `n` by adding whoever covers the most additional uncovered pairs.
  Each group = **one LLM call** scoring all C(n,2) selected pairs among its members.
- `n = max(2, n_profiles_to_score_together)` (`choreo/score.py:53`) — clamped min 2, no upper bound.
- Each pair is scored **exactly once** (marked `covered`, never re-scored — `choreo/score.py:104`).
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
   `UNSCORED` warning path, `choreo/score.py:115`).
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
