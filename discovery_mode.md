# Discovery Mode — Plan

A new pipeline mode (`--pipeline discovery`) that, instead of producing N warm
1:1 introductions, reads a whole cohort's onboarding inputs and surfaces:

- the **emergent themes / clusters of ideas** in the group,
- concrete **project seeds & collaboration proposals** that combine *different
  people with complementary skills around a shared interest/value*, and
- a facilitator-facing **"state of the community" overview** (themes, the people
  who connect them, and the skill gaps to fill).

Target use: the start of a hackathon / residency / Wintercircus cohort, right
after everyone has done an agent onboarding call. Cohort sizes are small
(~8–50 people).

> This document is a design plan only. No code is written here.

---

## 1. Design thesis

The hard part, as stated: a *great* project usually needs **cohesion on the
"why"** (shared interests, values, curiosities) and **diversity on the "how"**
(different, complementary skills). Naive clustering of a single "goals"
embedding fails because it optimizes only cohesion and is blind to
complementarity — it surfaces echo chambers, not teams.

Two design commitments fall out of this:

1. **Separate the two axes.** Use one embedding-derived signal for *cohesion*
   (interests/vision/ideas) and a different one for *complementarity* (skills).
   Cluster primarily on cohesion; use complementarity to *score and rank* the
   resulting groups. Do **not** collapse both into one signed similarity number
   and cluster on that — clustering a signed blend is semantically muddy and
   hard to interpret. (This is the one place we deliberately *don't* reuse the
   matching recipe's single fused matrix as-is.)

2. **Over-generate cheaply, judge expensively — but bounded.** Mirror the
   matching pipeline's budget discipline exactly. There the flow is: compute an
   N×N similarity matrix (cheap) → select the top candidate *pairs* per profile
   under a budget → spend LLM calls only on those. Here it becomes: embed
   everything (cheap) → generate many candidate *groups* and score them with
   pure-numpy cohesion/diversity metrics (cheap) → spend a **bounded** number of
   LLM calls only on the top-ranked, de-duplicated groups. We never do O(N²) or
   O(2^N) LLM work.

This keeps the cost profile identical in spirit to matching: **LLM cost grows
linearly with N (extraction) plus a fixed cap (group analysis); the
combinatorial work stays in cheap embedding/linear-algebra space.**

---

## 2. What we reuse vs. what is new

### Reuse as-is (no changes)

| Primitive | Module | Role in discovery |
|-----------|--------|-------------------|
| `load_profiles` → `Profile` | `src/ingest.py` | Identical ingest of `<raw_dir>/*.txt`. |
| `extract_sections_from_profiles` | `src/extract.py` | Bounded, cached, async section extraction. **Linear in N.** Driven by a section config — so we just point it at discovery-flavored sections. |
| `LLMWrapper.batch_json_complete` | `src/llm.py` | The async/cached/cost-tracked/retrying batch workhorse for the group-analysis and synthesis passes (same way `score.py` and `introduction.py` use it). |
| `get_embeddings` | `src/embed.py` | Low-level batch embedder. Reusable directly for idea-atom embedding (see §4.3). |
| `cost_tracker`, `utils` (`cosine_matrix`, `hash_text`, JSON/JSONL/YAML IO, `ensure_dir`, `filter_active_sections`, `truncate_words`) | `src/cost_tracker.py`, `src/utils.py` | Unchanged. |
| `compute_combined_distances` + t-SNE plotting | `src/tsne.py` | Visual sanity check, now colored by discovered cluster. |
| Pipeline plumbing: `BasePipeline`, `PipelineRegistry`, `PipelineContext`, `apply_io_overrides`, `resolve_prompt_paths`, `--group`/`--input`/`--force` | `main.py` | The mode hooks in here as a registered pipeline — see §6. |

### Reuse with a small adaptation

| Primitive | Module | Adaptation |
|-----------|--------|-----------|
| `compute_fused_similarity_matrix` | `src/candidate.py` | Call it with a **discovery recipe** to get the cohesion (affinity) matrix from interest/vision/idea sections. It already returns per-section matrices in `matrices_dict` — we read the `skills` section matrix straight out of there for the complementarity axis. No code change needed; just a different recipe + we consume more of its existing output. |
| `create_section_embeddings` | `src/embed.py` | Reusable verbatim for **person-level** discovery (fixed `users × sections × dims` tensor). For **idea-atom-level** discovery the fixed reshape doesn't fit a variable number of atoms per person, so atoms use `get_embeddings` + a thin new flatten/index routine instead (§4.3). |
| HyDE (`generate_hyde_descriptors`) + directional cross-matrix | `src/hyde.py`, `src/candidate.py` | Optional advanced path (§5): bridge "idea/need" → "skills that would realize it" to recruit complementary collaborators around an idea. Pure reuse, gated like it already is on `cross_section_weights`. |

### New components

| New file | Responsibility | Modeled on |
|----------|----------------|------------|
| `src/discover.py` | Pure numpy/sklearn: build the cohesion & complementarity matrices, generate candidate groups, score them (cohesion × diversity), rank + de-dupe. **No LLM calls.** | the cheap pre-filter half of `candidate.py` / `score.py` |
| `src/analyze.py` | The bounded LLM passes: (a) per-group analysis → spark score + named theme + project seeds + gaps; (b) one cohort-level synthesis pass. Batched via `batch_json_complete`. | `score.py` + `introduction.py` |
| `src/discovery_report.py` | Emit `discovery.json` (machine) + `discovery_report.md` (human) + cluster-colored plots. | `report.py` |
| `config/discovery_prompt.yaml` | Prompt templates: `group_analysis`, `cohort_synthesis` (and `idea_atoms` if used). | `scoring_prompt.yaml`, `introduction_prompt.yaml`, `hyde_prompt.yaml` |
| `config/config_discovery.yaml` *(or a `discovery:` block in `config.yaml`)* | Discovery recipe, group-size bounds, objective weights, budgets, prompt-file overrides. | `config.yaml` |
| `DiscoveryPipeline` class in `main.py` | Registers the mode; runs ingest/extract/embed verbatim then branches into discovery steps. | `MatchingPipeline` |

---

## 3. Pipeline at a glance

```
1.  INGEST    → reuse load_profiles  (unchanged)
2.  EXTRACT   → reuse extract_sections_from_profiles with discovery sections
                (skills + interests + vision + ideas/needs)            [LLM, linear in N]
2a. IDEA ATOMS → (recommended) split each profile's ideas/interests into a
                 bounded list of atomic "idea units"                    [LLM, linear in N]
3.  EMBED     → reuse get_embeddings (atoms) / create_section_embeddings (persons)
                                                                        [cheap]
4.  SIGNALS   → cohesion matrix (interests/vision/ideas) via discovery recipe;
                complementarity from the skills section matrix          [cheap, numpy]
5.  CLUSTER   → cluster idea-atoms into THEMES and/or grow candidate
                people-GROUPS by seed-expansion                         [cheap, sklearn/numpy]
6.  SCORE+RANK→ score each candidate group by cohesion × skill-diversity;
                de-dupe overlapping groups; keep top G under budget     [cheap, numpy]
7.  ANALYZE   → LLM judges each of the top-G groups → spark score, theme
                name, 1–3 concrete project seeds, missing-skill gaps    [LLM, BOUNDED ≤ G]
8.  SYNTHESIZE→ one LLM pass over the surviving themes/seeds → cohort
                "state of the community" overview                       [LLM, ~1 call]
9.  REPORT    → discovery.json + discovery_report.md + cluster plots    [cheap]
```

Steps 1–3 are shared with matching and **cache-compatible**: extraction and
embeddings are mode-agnostic, so if the section set matches, both modes can run
on the same cohort and reuse caches. (Embedding cache keys on the user set +
section names, so a different active-section set lands in its own cache — run
with `--force` when switching section configs in place, or use a separate
group/folder.)

---

## 4. The core algorithm (steps 4–6 in detail)

### 4.1 The two signals

From the embeddings we derive two N×N matrices over people (and, in the
idea-atom variant, distances over atoms):

- **Cohesion / affinity `A`** — "how aligned are these two on what they care
  about?" Built by calling `compute_fused_similarity_matrix` with a discovery
  recipe that puts **positive** weight on `interests`, `vision`, `ideas` (the
  "why" sections) and **zero** weight on `skills`. This reuses the existing
  fusion + normalization code unchanged.

- **Complementarity `C`** — "how *different* are their skill sets?" Read the
  `skills` per-section similarity matrix out of `matrices_dict['section_matrices']`
  (already computed by the same call) and use `C = 1 − S_skills`. High `C` =
  complementary capabilities. (Note the elegant parallel: matching already
  expresses "skills should differ" as a *negative* `section_weights.skills`;
  discovery just makes that the explicit second axis instead of folding it in.)

### 4.2 Candidate-group generation — **seed expansion** (recommended)

Hard partitioning (k-means / a single cut of agglomerative clustering) is the
wrong model: a person can belong to several project seeds, and we want
**overlapping, size-bounded, interpretable** groups. Proposed primary method,
which needs no new dependencies (pure numpy):

For each seed (each person, or each strong affinity edge):
1. Start a group `G = {seed}`.
2. Repeatedly add the not-yet-included person `c` that **maximizes affinity to
   the current group** (mean of `A[c, g]` for `g ∈ G`) *subject to* the group's
   cohesion staying above a floor.
3. Stop when `|G|` hits `group_size.max` or no candidate keeps cohesion above
   the floor; discard if `|G| < group_size.min`.
4. Collect the set; de-dupe identical sets at the end.

This yields many overlapping candidate groups that are all internally cohesive
by construction. Diversity is *not* forced during growth — it's used to **rank**
afterward (§4.4), so we don't trade away cohesion to chase diversity.

**Baseline / sanity alternative:** `sklearn.cluster.AgglomerativeClustering`
(or `SpectralClustering`) on the affinity distance `1 − A` (both accept a
precomputed affinity/distance and are already available via the sklearn dep).
This gives a clean hard partition useful for the cohort map and t-SNE coloring,
but it can't express overlap, so it's the secondary view, not the group source.
HDBSCAN / Louvain community detection are optional upgrades but add a dependency
(`hdbscan` / `networkx`) and are less robust at very small N — noted as future
options, not the v1 default.

### 4.3 Idea-atom theme clustering — **the recommended core for "clusters of ideas"**

The user's framing is literally "find interesting clusters of *ideas*." Treating
each **person** as one point under-resolves this — one person often carries
several distinct ideas. So the recommended core operates on **idea atoms**:

1. **Extract atoms (step 2a, LLM, still linear in N):** one extra extraction
   call per profile asks the LLM to decompose the person's
   interests/ideas/project into a *bounded* list (e.g. ≤5) of atomic idea units,
   each a short self-contained phrase, tagged with the source person. Bounded
   list → still O(N) calls, batched and cached like every other extraction.
2. **Embed atoms (cheap):** flatten all atoms across all people into one list,
   call `get_embeddings` directly, and keep an `atom → person` index. *(This is
   where `create_section_embeddings`'s fixed `users × sections` reshape doesn't
   fit — `get_embeddings` is the reusable piece; the flatten/index is a few new
   lines.)*
3. **Cluster atoms into themes (cheap):** cluster the atom embeddings (cosine).
   Each cluster is a **theme** = a cluster of ideas contributed by potentially
   several different people.
4. **Back out people & skills per theme:** map a theme's atoms to their source
   people → the theme's *participant set*. Now check the participants' `skills`
   embeddings for diversity/coverage. A theme with many distinct contributors
   and complementary skills is a strong project seed; a theme that's one
   person's pet idea, or a crowd with identical skills, ranks lower.

This directly solves the stated problem: **a theme is defined by clustered idea
fragments (shared interest) while its participant set's skill spread measures
complementarity** — the two axes stay separate and legible.

v1 can ship person-level seed-expansion (§4.2) first for simplicity, with
idea-atom theming (§4.3) as the headline capability; they share all downstream
scoring, analysis, and reporting code.

### 4.4 Scoring & ranking candidate groups/themes (cheap, pre-LLM)

For each candidate group `G` (whether from seed-expansion or an atom theme's
participant set), compute with pure numpy:

- `cohesion(G)` = mean pairwise affinity within `G` (on `A`); for atom themes,
  also the tightness of the atom cluster.
- `diversity(G)` = mean pairwise skill distance within `G` (mean of `C`), i.e.
  how complementary the skills are.
- `size sanity` = penalty outside `[group_size.min, group_size.max]`.

Combine into a single rankable score, config-driven (mirrors `recipe` weights):

```
group_score = cohesion_weight * cohesion(G) + diversity_weight * diversity(G)
```

with a hard **cohesion floor** (a group with no shared thread is noise, not a
seed) gating before diversity is even considered.

Then **de-dupe for variety** before spending LLM budget: greedily keep
top-scoring groups while rejecting any new group whose member overlap (Jaccard)
with an already-kept group exceeds `overlap_jaccard_max`. This is the discovery
analogue of the matching pipeline's "spread coverage across users" concern, and
it ensures the LLM sees ~G *distinct* seeds rather than 20 variations of the
same clique. Keep the top `max_group_llm_calls` survivors.

---

## 5. The LLM passes (steps 7–8, bounded)

### 5.1 Per-group analysis (≤ G calls, batched)

For each surviving group, build one prompt containing the members' extracted
sections (reusing `score.py`'s XML profile formatting) and the theme's idea
atoms, and ask the LLM for structured JSON:

- `spark_score` (0–1): is this a genuinely interesting, non-obvious
  combination? — the LLM judgment the cheap metrics can't make.
- `theme`: a short name + one-line description of the shared thread.
- `project_seeds`: 1–3 concrete proposals that *specifically* leverage who is in
  the group — "X's projection-mapping + Y's sensor fabrication + Z's
  community-organizing → a movement-reactive installation for the Kaaibar
  opening" — naming which person brings what (same specificity bar the
  introduction prompt already enforces: "be specific about WHICH skill meets
  WHICH need," no vague "you share interests").
- `missing`: the one or two skills/roles the seed needs but the group lacks —
  feeds gap analysis and "who else to pull in."

Batched and budgeted exactly like `score_pairs_with_llm`: a single
`batch_json_complete` over all surviving groups, cached per group signature,
capped by `budgets.max_group_llm_calls`. `spark_score` lets us drop groups the
cheap geometry liked but the LLM finds boring before they reach the report.

### 5.2 Cohort synthesis (~1 call)

A final pass receives the surviving themes + their seeds + the cohort skill
inventory and writes the narrative overview: the top themes, **connectors**
(people who bridge several themes — derivable from group overlap), notable
unmatched/loner profiles, and **collective gaps** (skills repeatedly listed in
`missing`). This is the facilitator's "state of the community" briefing. Bounded
to one (or a few, if the cohort is large) calls.

---

## 6. Integration & invocation

Use the existing registry — it's purpose-built for exactly this (`main.py`
already has `PipelineRegistry`, `--pipeline`, and `--list-pipelines`):

- Add `DiscoveryPipeline(BasePipeline)` with `name = "discovery"`, registered
  next to `MatchingPipeline`. Its `run()` reuses `apply_io_overrides` +
  `resolve_prompt_paths` (extended to resolve the discovery prompt files), runs
  ingest/extract/embed via the existing functions, then calls the new
  `discover` → `analyze` → `discovery_report` steps.
- Invoke:
  ```bash
  python main.py --pipeline discovery --input ~/cohorts/hackathon_2026 --force
  python main.py --pipeline discovery --group wintercircus
  python main.py --list-pipelines        # now shows matching + discovery
  ```
- Optional ergonomic wrapper: a tiny `analyze_community.py` that just calls
  `main(pipeline_name="discovery", ...)`, so the mode has a memorable entry
  point — but the registry is the real mechanism, no plumbing is duplicated.
- Modal: add a `run_discovery_pipeline` entry alongside the existing matching one
  in `deploy_modal.py`, delegating to the same pipeline (same profiles-as-JSON
  contract).

---

## 7. Configuration sketch

A `discovery:` block (or standalone `config_discovery.yaml` chosen via
`--config`), in the existing config style:

```yaml
models:                      # reuse existing model slugs
  embedding: "google/gemini-embedding-2-preview"
  embedding_dimensions: 768
  extraction_llm: "google/gemini-3.1-flash-lite"
  analysis_llm:   "google/gemini-3.1-flash-lite"   # group analysis + synthesis

discovery:
  cohesion_recipe:           # the "why" axis → affinity matrix A (reuses candidate.py)
    section_weights:
      interests: 0.40
      vision:    0.35
      ideas:     0.25
      skills:    0.00
  complementarity_section: skills     # → C = 1 - S_skills
  granularity: idea_atoms             # "idea_atoms" | "person"
  idea_atoms:
    max_atoms_per_profile: 5
  group_size: { min: 2, max: 5 }
  objective:
    cohesion_weight: 0.6
    diversity_weight: 0.4
    cohesion_floor: 0.25              # hard gate before ranking
  candidate_groups:
    overlap_jaccard_max: 0.6          # de-dupe for variety
  budgets:
    extraction_llm_calls: 100
    max_group_llm_calls: 40           # the bounded LLM cap (≈ #seeds judged)
  synthesis: true

prompt_files:
  section_prompt:   config/discovery_section_prompt.yaml   # adds interests/ideas sections
  discovery_prompt: config/discovery_prompt.yaml
```

Everything that defines *what makes a good seed* (axis weights, size bounds,
cohesion floor, objective blend, budgets) is config, not code — same philosophy
as the matching `recipe`/`blending`/`matching` blocks, so Xander can tune it per
cohort without edits.

### Sections for discovery

Reuse the section-config mechanism (`active` flags + `guideline` + `max_words`).
The matching sections are `skills / vision / project / needs`. For discovery, a
discovery section config keeps `skills` and `vision`, and adds/activates:

- `interests` — themes, curiosities, domains this person is drawn to (the
  cohesion axis, distinct from long-term `vision`).
- `ideas` — things they want to build / wish existed / would jump on (the raw
  material for idea atoms and project seeds), distinct from present-tense
  `project`.

Because extraction is fully config-driven, this is a new YAML, not new code.

---

## 8. Outputs

Written to the existing `<outputs_dir>` (so `--group`/`--input` layout is
unchanged):

- **`discovery.json`** (machine-readable): list of themes/groups, each with
  members, their idea atoms, `spark_score`, `cohesion`/`diversity`,
  `project_seeds`, `missing` skills; plus cohort-level connectors and gaps.
- **`discovery_report.md`** (the deliverable): the facilitator narrative —
  "6 emergent themes, 10 concrete project seeds with named people and the why,
  the connectors, the gaps." This is the artifact a hackathon/residency host
  reads aloud or pins to a wall.
- **`plots/`**: idea-atom / person t-SNE colored by discovered theme (reuse
  `tsne.py`), and an affinity heatmap (reuse `visualize_similarity.py`), so
  cluster quality is eyeballable — the discovery analogue of matching's
  score-correlation plots.
- **`cost_report.json`**: reuse `cost_tracker` unchanged.

---

## 9. Scaling & cost (the explicit requirement)

| Stage | Work | LLM calls | Scaling |
|-------|------|-----------|---------|
| Extract sections | 1 call/profile (batched, cached) | N | **linear** |
| Extract idea atoms | 1 call/profile (batched, cached) | N | **linear** |
| Embed | atoms+sections, batched | 0 (embeddings) | linear, cheap |
| Signals (A, C) | N×N cosine | 0 | quadratic *but pure numpy*, trivial at our N |
| Cluster + generate groups | atom clustering / seed expansion | 0 | cheap |
| Score + rank + de-dupe | numpy over candidate groups | 0 | cheap |
| Group analysis | 1 call/surviving group | **≤ `max_group_llm_calls`** | **capped constant** |
| Synthesis | cohort overview | ~1 | constant |

Total LLM cost ≈ `2N + min(#groups, cap) + 1` — **linear extraction plus a fixed
analysis cap**, never combinatorial. This is the same cost shape as matching and
holds from an 8-person residency to a few-hundred-person cohort.

---

## 10. Risks, tradeoffs & decisions to confirm

- **Person-level vs. idea-atom granularity.** Idea-atoms answer the brief most
  directly and resolve multi-idea people, but add one extraction pass and a
  custom embedding/index path. *Recommendation:* ship person-level seed-expansion
  as v1 scaffolding, then idea-atom theming as the headline — they share all
  downstream code. (Confirm which to build first.)
- **Clustering method at small N.** HDBSCAN/Louvain are tempting but fragile at
  N≈8–15 and add deps. *Recommendation:* seed-expansion (numpy, overlap-friendly,
  size-bounded) as the group source + agglomerative (existing sklearn) for the
  partition map. Revisit community detection only if cohorts grow large.
- **No ground truth.** Quality is subjective, so lean on the plots + exposed
  scores + fully config-driven knobs for human-in-the-loop tuning (exactly how
  matching exposes recipe weights and correlation plots today).
- **Cohesion-floor / objective-weight sensitivity.** These decide whether seeds
  skew "safe echo chamber" vs. "ambitious mashup." Surface both `cohesion` and
  `diversity` per group in `discovery.json` so the balance is auditable and
  tunable.
- **Don't cluster the signed fused matrix.** Reusing matching's single
  `final = embed·w + llm·w` blend for *clustering* would re-introduce the exact
  cohesion/complementarity confound we set out to avoid. We reuse the fusion code
  but keep the two axes separate. (Stated so a future contributor doesn't
  "simplify" it back.)
- **Optional HyDE idea→skill bridge (§2, §5).** An idea-anchored variant —
  take each strong idea as a nucleus and recruit complementary skills via the
  existing directional HyDE cross-matrix — is arguably an even more direct
  "project proposal" generator. It's pure reuse of `hyde.py` + `candidate.py`'s
  directional path and can layer on later as an alternative `granularity` mode.

---

## 11. Suggested build order

1. **Scaffold the mode.** `DiscoveryPipeline` in `main.py` (register + CLI),
   reuse ingest/extract/embed, stub `discover`/`analyze`/`discovery_report`.
   Add the discovery section + prompt + config files. Run end-to-end on
   `data/real` producing a placeholder report.
2. **Cheap half (`src/discover.py`).** Build A and C from the discovery recipe;
   person-level seed-expansion; cohesion×diversity scoring; Jaccard de-dupe.
   Validate via t-SNE/heatmap before any LLM spend.
3. **LLM half (`src/analyze.py`).** Per-group analysis (spark score + theme +
   seeds + gaps) via `batch_json_complete`; then the synthesis pass.
4. **Report (`src/discovery_report.py`).** `discovery.json` + `discovery_report.md`
   + cluster-colored plots.
5. **Idea-atom granularity.** Add the atom extraction pass (step 2a) + the
   `get_embeddings` flatten/index path + atom clustering; route the same scoring/
   analysis/report code through it. Make `granularity` config-switchable.
6. **(Optional) HyDE idea→skill anchored variant** and Modal entry point.
```
