# Discovery Mode — engineered serendipity (V1 lean MVP + V2 roadmap)

**Status:** plan / not started. Originally drafted pre-refactor at the repo
root; relocated here 2026-07-07 during the improvement-sprint audit and merged
with that sprint's positioning notes. Paths modernized to the current package
layout (`src/` → `choreo/`, `config/` → `choreo/defaults/` + config-dir
overlays).

**Positioning (from [improvement_sprint.md](improvement_sprint.md)):** this is
the **serendipity half** of the platform's "matchmaking and serendipity
engine" — the capability Murmura / show-night orchestration actually needs
(groups + project seeds, not 1:1 intros). Run it as its own sprint, sequenced
after the improvement sprint, coupled through shared infrastructure:

- The improvement sprint's **Track-1 fixture cohort doubles as this plan's
  test cohort** (§7 below names the missing fixture as the first task —
  author once, share).
- The sprint's Track-2 HyDE work (sub-need decomposition, `matched_via`
  provenance) feeds §6.7 complement-attraction directly.
- Build the V1 lean MVP **plus the LLM-dump baseline** exactly as specified
  below; resist V2 features until each is A/B-benchmarked (§5.8).

---

A new pipeline mode (`--pipeline discovery`) that, instead of producing N warm
1:1 introductions, reads a whole cohort's onboarding inputs and engineers
**serendipity**: it surfaces a curated set of project seeds and collaboration
provocations — concrete "the four of you should build X this weekend" sparks.

Target use: the first night of a weekend hackathon / a residency / a Wintercircus
cohort, right after everyone has done an agent onboarding call. Cohort sizes are
small (~8–50 people).

**This document is split into two phases.** §5 (**V1**) is the leanest thing that
tests the core hypothesis end-to-end, reuses the existing codebase, and can be
benchmarked against a dead-simple baseline. §6 (**V2**) scopes every refinement —
including the fixes for the three known failure modes — as *not-yet-implemented*
upgrades, each paired with the V1 result it would improve. Build V1, benchmark it,
then add V2 features one at a time and measure each.

---

## 1. The core thesis: a geometry engine + a meaning engine

- **Embeddings + similarity = a geometry engine.** It answers *where to look*:
  which threads are adjacent, which are far, which corner of the room is crowded.
  It can enumerate and score thousands of candidate groupings — cheaply, in numpy.
- **The LLM = a meaning engine.** It answers *what it means*: given an odd
  assortment of threads and the event context, what could these specific people
  actually *make* together? It names the latent theme and turns adjacency into a
  buildable provocation.

Three observations shape the design:

- **(a) Pure similarity collapses to the mean.** Maximizing goal-similarity yields
  bland blobs. So similarity shapes the process **softly** (a probability
  landscape) and is **never** the thing we maximize.
- **(b) People arrive with threads, not ideas.** The *idea* is an **output** the
  LLM synthesizes from a *collision* of threads — never an input we cluster.
- **(c) The loudest signal is the least useful.** Instrumental goals ("I want to
  network") are high-volume and high-similarity. We do **not** classify them out
  (the line is a gradient an LLM can't reliably draw). In V2, geometry demotes them
  via density; in V1 we accept some obvious groups and let the LLM label them.

---

## 2. The atom model (the V1 substrate)

Everything is built on one unit: the **atom**.

> An **atom** is a single, self-contained, interesting statement about a person,
> phrased to stand alone and embed cleanly. *"Ran a 60m audio-reactive LED
> installation on a Raspberry Pi." "Plays modular synth on the side." "Wants paying
> customers for his payments API." "Curious about citizen science."*

We do **not** tag atoms (instrumental/latent, skill/value) — they are just
statements, and geometry sorts them. A person contributes a **variable-length list**
of atoms (soft target ~3–8). The landscape is the union of everyone's atoms.

Why atoms beat one vector per person: a buried "plays modular synth" gets **equal
geometric standing** with a loud "I want customers," and a person can enter
different groups through different facets of themselves.

**Lean handling of variable length.** The "non-trivial" part is only hard if you
force atoms into the fixed `(users × sections × dims)` tensor that
`create_section_embeddings` produces. We **don't**. We embed a *flat list* of all
atom texts with `embed.get_embeddings` and keep an integer owner index. No reshape,
no per-section bookkeeping — simpler than the section tensor, not harder.

```
AtomTable:
  texts:      List[str]          # length A (total atoms in cohort)
  owner:      np.ndarray[int]    # length A; owner[i] = person index of atom i
  person_ids: List[str]          # person index -> user id
  E:          np.ndarray[A, d]   # L2-normalized atom embeddings
  S:          np.ndarray[A, A]   # atom×atom cosine = E @ E.T (utils.cosine_matrix)
```

A **group** is a set of atom indices. A **team** is the set of distinct owners.
**Invariant:** one atom per person per group (mask an owner once picked), so a
k-atom group is always k distinct people; facet-level participation lives *across*
groups, not within one.

---

## 3. Why this beats dumping all profiles into one LLM call

The geometry engine does the relational/combinatorial reasoning a single LLM call
is bad at: it gives **equal standing to quiet threads** (attention ignores them),
it can **enumerate thousands of groupings** without mode-collapsing to a handful of
clichés, and **adding/removing a member is a cheap matrix update**. The LLM does the
one thing geometry can't — the semantic leap from adjacency to a named, buildable
idea. V1 is designed to be measured against exactly the "LLM dump" baseline (§5.8).

---

## 4. End-to-end shape (V1)

```
INGEST    → load .txt profiles                    (reuse ingest.load_profiles)
ATOMIZE   → 1 batched LLM call/profile → atom list (NEW fn, reuses batch_json_complete)
EMBED     → embed all atoms flat → AtomTable        (reuse embed.get_embeddings)
SAMPLE    → N tempered random walks → groups          (NEW, pure numpy)
SELECT    → coherence gate + diversity + coverage → M  (NEW, pure numpy)
PROPOSE   → ≤M batched LLM calls → provocation JSON     (reuse batch_json_complete)
RANK      → top K by spark_score
REPORT    → discovery.json + discovery_report.md + atom t-SNE
```

LLM cost = `P (atomize) + M (propose)`. No HyDE, no pair-scoring, no synthesis —
cheaper than the existing matching pipeline.

---

## 5. V1 — the lean MVP

### 5.1 Scope

**In:** flat atoms, single-temperature stochastic sampling with a mean↔max pooling
knob, a coherence gate, diversity + coverage selection, one LLM provocation pass,
a report, and an atom-landscape plot. Two dials only: temperature `T` and pooling
`λ`. **Out (→ V2):** everything in §6.

### 5.2 The algorithm, step by step

All of 5.2 is **pure numpy, no LLM, no new dependencies.** Build it and eyeball the
t-SNE landscape *before* spending on the LLM pass.

**Landscape (once):**
```
1. atoms = extract_atoms(profiles)                     # §5.3 — [(owner_idx, text)]
2. E     = get_embeddings([t for _,t in atoms], model) # embed.get_embeddings
   E     = E / ||E||                                    # L2 normalize rows
3. S     = E @ E.T                                       # utils.cosine_matrix
4. owner = int array of length A
```

**Grow one group (single-temperature tempered walk):**
```
def grow_group(rng, T, lam):
    k    = rng.integers(size_min, size_max + 1)
    seed = rng.integers(A)                       # uniform seed (V1)
    G, used = [seed], {owner[seed]}
    while len(G) < k:
        cand = [c for c in range(A) if owner[c] not in used]   # 1 atom/person
        if not cand: break
        sims = S[np.ix_(cand, G)]                              # |cand| × |G|
        pool = (1-lam) * sims.mean(1) + lam * sims.max(1)      # λ: 0=mean … 1=max
        z    = (pool - pool.mean()) / (pool.std() + 1e-9)      # per-step z-score
        p    = softmax(z / T)                                   # Boltzmann
        c    = rng.choice(cand, p=p)
        G.append(c); used.add(owner[c])
    return G
```
- `T` is the one temperature dial. `T→0` echo chamber; `T→∞` noise; moderate `T`
  coherent-with-surprises.
- `λ` is the one character dial. Default `λ=0` (mean → tight, safe groups). Cranking
  `λ` toward 1 (max-pooling drift) is where cross-domain serendipity comes from —
  it's a single line, so V1 exposes it, but defaults safe. (Drift's failure mode and
  its guards are V2; see §6.)

**Gate (kill noise):**
```
def coherent(G):
    sub = S[np.ix_(G, G)].copy(); np.fill_diagonal(sub, -inf)
    return np.all(sub.max(1) >= coherence_floor)   # single-linkage: every member
                                                    # is close to ≥1 other member
```

**Sample N, then select M (anti-collapse by diversity, not by a quality score):**
```
groups    = dedup( [grow_group(rng, T, lam) for _ in range(N)] )   # by frozenset(owner)
survivors = [g for g in groups if coherent(g)]
rng.shuffle(survivors)                              # NOT sorted by similarity (obs. a)
selected  = []
for g in survivors:
    members = {owner[i] for i in g}
    if all(jaccard(members, members_of(s)) <= overlap_jaccard_max for s in selected):
        selected.append(g)
    if len(selected) >= llm_judge_pool: break       # M
# coverage — nobody leaves without a provocation
for p in (all_persons - covered(selected)):
    add any survivor group containing p              # (relax M slightly)
```

Deliberately, V1 has **no contrast/novelty/density score** — the only geometric
quality signal is the coherence gate; diversity selection prevents near-duplicate
groups, and the **LLM's `spark_score` does all quality ranking.** This keeps V1 a
clean test ("does stochastic diverse sampling + LLM judging beat a dump?") and
turns every V2 geometric score into a separately measurable improvement.

### 5.3 Atomization (the one new LLM step)

A new `extract_atoms_from_profiles` — modeled on
`extract.extract_sections_from_profiles` but with a **list-valued output**. Simplest
implementation: call `llm_wrapper.batch_json_complete` directly (one prompt per
profile, like `score.py`), prompt returns `{"atoms": ["…", "…"]}`.

- Prompt (`choreo/defaults/discovery_atoms_prompt.yaml`): *"Extract the distinct,
  interesting, self-contained elements of this person — projects, skills,
  obsessions, side hobbies, what they're building, and **what they want or need
  help with**. Each as a standalone sentence that reads on its own. Surface the
  orthogonal and the quiet, not just the headline. ~3–8 elements. Do not invent."*
  (Including "what they want" seeds complementarity cheaply — a need-atom can sit
  near a skill-atom.)
- Cache per profile via `utils.hash_text(profile_text)` (reuse the cache-key pattern
  from extract.py), so re-runs are free and `--force` re-atomizes. Heed the
  improvement sprint's F3 lesson: the cache key must also cover the rendered
  prompt template + model, or prompt iteration silently replays stale atoms.
- Embeddings cached as one `.npz` in `embeds_dir`; recompute if the atom set changes.

### 5.4 LLM proposal pass (the only analysis call)

One batched call over the M selected groups (driven via `run_coro_blocking`,
exactly like `score.py` — async-host safe). Prompt = members' atoms + their full
profiles + the **event context** (theme, venue/vibe, time budget, ethos — the
constraint is *generative*). Stance is a **provocation, not a prediction**.
Structured JSON per group:

- `spark_score` (0–1) — the taste judgment geometry can't make;
- `theme` — the latent thread, named;
- `proposal` — names *which person's atom brings what* (no vague "you share
  interests");
- `first_step` — the smallest thing they could do in the first hour;
- `missing` — a skill/role the group lacks (recorded; V2 turns this into a loop).

Cached per group signature (sorted owner ids + atom texts), capped by
`budgets.max_group_llm_calls`. Rank by `spark_score`; return top `K`.

Single-responsibility caution (mirrors the improvement sprint's §6.1 decision
to keep scoring and intro generation separate): `spark_score` here is
**self-graded** — the model rates the proposal it just wrote, so scores will
skew high across the board. Acceptable for V1 since ranking is relative and
every group shares the bias; if the §5.8 eval shows poor separation between
strong and weak provocations, split judging into its own cheap pass (one call
ranking all M generated proposals comparatively, proposal text as input)
before reaching for V2 §6.6's geometric scores.

### 5.5 Reuse map

| Need | Reuse / extend |
|------|----------------|
| Load profiles | `ingest.load_profiles` — as-is |
| Atomize | **NEW** `extract_atoms_from_profiles` (in `choreo/extract.py` or new `choreo/atomize.py`), reuses `LLMWrapper.batch_json_complete`, `utils.hash_text`, `get_cache_path` |
| Embed atoms | `embed.get_embeddings` — as-is (**not** `create_section_embeddings`); optional `truncate_embeddings` |
| Similarity | `utils.cosine_matrix` — as-is |
| Sample/gate/select | **NEW** `choreo/sampler.py` — pure numpy |
| Proposal pass | **NEW** `choreo/analyze.py` — reuses `batch_json_complete` + `run_coro_blocking` |
| Report | **NEW** `choreo/discovery_report.py` — mirrors `report.py`; reuse `utils` IO |
| Landscape plot | `tsne.py` — thin adapter for a flat `(A, d)` array |
| Cost | `cost_tracker` — as-is |
| Pipeline plumbing | `main.py`: `BasePipeline`, `PipelineRegistry`, `apply_io_overrides`, `resolve_prompt_paths`, `--group/--input/--force/--pipeline/--list-pipelines` — register a `DiscoveryPipeline` |

New surface is small: ~3 modules + 1 extract function + 2 prompt files + 1 pipeline
class.

### 5.6 Configuration sketch (minimal)

Ships as new keys in `choreo/defaults/config.yaml` + two new packaged prompt
yamls (overridable per deployment via the standard `--config-dir` overlay):

```yaml
models:
  embedding:      "qwen/qwen3-embedding-8b"
  embedding_dimensions: 768
  extraction_llm: "<models.extraction_llm default>"   # atomization
  analysis_llm:   "<models.pair_llm default>"          # proposals

discovery:
  atoms: { max_atoms_per_profile: 8 }

  sampler:
    n_groups: 3000                 # N — compute knob; pure numpy, scale freely
    group_size: { min: 3, max: 5 }
    temperature: 0.8               # the one T dial
    pooling: 0.0                   # λ: 0=mean (safe). Crank toward 1 for drift.

  selection:
    coherence_floor: 0.20
    llm_judge_pool: 30             # M — distinct candidates sent to the LLM
    overlap_jaccard_max: 0.6
    coverage_min_per_person: 1

  output: { return_top: 12 }       # K

  event_context: |
    A weekend hackathon at Wintercircus, Ghent. Playful, hands-on, community-driven
    vibe-coding around music, visuals, physical/social spaces. Bias toward things
    buildable in 2 days that could live on afterward.

  budgets: { atomize_llm_calls: 100, max_group_llm_calls: 30 }

prompt_files:
  atoms_prompt:     choreo/defaults/discovery_atoms_prompt.yaml
  discovery_prompt: choreo/defaults/discovery_prompt.yaml
```

### 5.7 Outputs

- **`discovery.json`**: each returned group with `members` (people), `atoms`
  (entering facets), `spark_score`, `theme`, `proposal`, `first_step`, `missing`.
- **`discovery_report.md`**: ranked provocations, each naming who brings what +
  first step + missing skill.
- **`plots/`**: atom t-SNE colored by person and by selected-group membership.
- **`cost_report.json`**: `cost_tracker`, unchanged.

### 5.8 Baseline & evaluation (the point of staying lean)

To know whether any of this works, V1 ships with a **baseline** to beat:

- **Baseline = the LLM dump.** Feed all atoms (or all profiles) + event context to
  the LLM in one/few calls; ask for `K` group provocations in the *same JSON
  format*. (A `--baseline dump` flag on the same pipeline.)
- **Compare V1 vs. baseline on, for the same K:**
  - *coverage* — fraction of people appearing in ≥1 provocation;
  - *non-obviousness* — mean intra-group atom similarity (lower = more cross-domain);
    plus a count of cross-domain vs. same-domain groups;
  - *facilitator/LLM-judge rating* of spark and buildability (qualitative).
- Every V2 feature is then an A/B against this V1 number, on real cohort data — so we
  add complexity only where it measurably moves coverage / non-obviousness / rating.

### 5.9 V1 build order

1. **Scaffold:** `DiscoveryPipeline` (register + CLI), reuse ingest/embed, stub
   `atomize`/`sampler`/`analyze`/`discovery_report`, add the two prompt files. Run a
   placeholder end-to-end. Test cohort = the improvement sprint's Track-1 fixture
   (~20 diverse profiles — author it there if it doesn't exist yet; see Risks).
2. **Atomize + landscape:** `extract_atoms_from_profiles`, `AtomTable`, atom t-SNE.
   **Eyeball the landscape before any LLM scoring.**
3. **Sampler + selection (§5.2):** single-T tempered walk, coherence gate, diversity
   + coverage. Eyeball groups on the landscape.
4. **Proposal pass (§5.4) + report (§5.7).**
5. **Baseline + eval harness (§5.8).** Measure V1 vs. dump on the fixture + a real
   cohort. *Stop here and learn before building V2.*

---

## 6. V2 — scoped, not implemented

Each item names the **V1 weakness it addresses**, a one-line **sketch**, its **cost**,
and (where relevant) which Gemini critique it answers. Add them one at a time and
A/B against the §5.8 numbers — do not batch them.

### 6.1 Cohort synthesis — "state of the room" (cheap, likely first add)
*Weakness:* V1 returns provocations but no cohort-level narrative. *Sketch:* 1 extra
LLM call over the returned groups + atom inventory → dominant threads, connectors,
collective gaps. *Cost:* +1 LLM call.

### 6.2 Temperature portfolio + annealing
*Weakness:* a single `T` bets on one energy. *Sketch:* sample across cool/warm/hot
bands, tag each group with its band (cool=teams, hot=wildcards); optionally anneal
`T_start→T_end` within a walk (nucleation). *Cost:* 0 LLM, pure numpy.

### 6.3 Outlier (density) seeding
*Weakness:* uniform seeding under-samples the misfits that produce the best
collisions. *Sketch:* precompute `dens[i] = Σ_j max(S[i,j],0)`; seed `p ∝
exp(−dens/T_seed)`. *Cost:* 0 LLM.

### 6.4 Max-pooling drift **+ its guards** (answers Gemini #2)
*Weakness:* high-`λ` drift produces the most exciting cross-domain groups but can
create "chain-link" groups whose extremes share no vocabulary — the LLM then
hallucinates a contrived connection. *Sketch:* turn up `λ`, **and ship the guards
with it** — replace the single-linkage gate with a **weakest-link gate** (max
spanning tree over `S_G`, require its minimum edge ≥ floor) and a **diameter cap**
(reject `max_{i,j}(1−S[i,j]) > diameter_max`), turning contrast into a bounded
inverted-U ("diverse but not shattered"). *Cost:* 0 LLM.

### 6.5 Theme anchor projection (answers Gemini #2 **and** #3)
*Weakness:* (a) raw density penalties would punish the *core event theme* (at a
climate residency, "carbon tracking" is dense *because it's the point*); (b) groups
lack a guaranteed shared center of gravity. *Sketch:* embed `event_context` → `θ`;
measure **contrast in the theme-orthogonal residual** `a⊥ = a − (a·θ̂)θ̂` (diversity
in the dimensions that *aren't* "we're all here for climate"), while the theme
provides cohesion. One mechanism fixes both density-eats-theme and chain-link
(everyone connects through the theme). *Cost:* 0 LLM.

### 6.6 Contrast & novelty scores
*Weakness:* V1 does no geometric quality ranking (LLM judges everything). *Sketch:*
add `contrast` (mean pairwise distance, bounded per 6.4) and `novelty` (off-centroid
+ low theme-residual density per 6.5) as a *loose* pre-ranking before the LLM, plus a
**wildcard quota** (force-keep high-novelty groups the metrics under-rate). *Cost:* 0
LLM. Measure whether geometric pre-ranking beats pure LLM judging.

### 6.7 Complement-attraction via HyDE (answers Gemini #1)
*Weakness:* like-attracts-like sampling clusters similar skills, not complementary
ones (React dev ≠ near designer). *Sketch:* give each atom a HyDE-derived "complement
vector" (what would *complete* it; reuse `hyde.py`) and blend into growth:
`p(c) ∝ exp((α·sim + β·complement)/T)`. *Cost:* +1 HyDE LLM pass over atoms.
*Note:* build on the improvement sprint's Track-2 HyDE upgrade (sub-need
decomposition + pooling knob) rather than the single-descriptor V0 — the
"distinct solution angles" semantics are exactly what a complement vector wants.

### 6.8 Missing→fill loop (answers Gemini #1)
*Weakness:* V1 just records the LLM's `missing` field. *Sketch:* embed the missing-
role text, nearest-neighbor among *unattached* people → concrete "pull in Sarah."
Turns a flagged gap into a geometric suggestion. *Cost:* 0–1 LLM.

### 6.9 Path-as-explanation
*Sketch:* emit the max-similarity spanning tree edges as the human-readable "why it
cohered" ("art-student ↔ robotics 0.68, bridged to producer via sound 0.61").
*Cost:* 0 LLM. Builds facilitator trust.

### 6.10 Connector / free-agent detection
*Sketch:* centrality on `S` → connectors; people who recur across many groups'
`missing` lists → free agents to deploy. *Cost:* 0 LLM.

### 6.11 Iterative refinement (light MCMC / genetic)
*Sketch:* take top groups, swap the weakest-marginal member for a hot-sampled
alternative, re-score. Cheap matrix ops; polishes raw samples. *Cost:* 0 LLM.

### 6.12 Runner entry point
*Sketch:* a `run_discovery` entry mirroring the mode runners in
`choreo/runners.py` (and, host-side, a
possible third tool_key — that decision belongs to
`motherbrain/docs/TODO/01_choreo_matchmaking_integration.md`, not here).

---

## 7. Risks & open questions

- **Variable-length atoms are the real engineering cost** — but the flat-array
  approach (§2) keeps it small. The `AtomTable` is the single source of truth; build
  it first and route every step through it. *Open:* atom-set caching when profiles
  change partially.
- **Atom quality = onboarding quality** (garbage in, garbage out). The orthogonal-
  thread digging happens **upstream**, out of scope here; this pipeline assumes
  profiles already contain such material.
- **V1 tests the *weakest* form of the hypothesis** (mean-pooling → tightish
  groups). The most exciting cross-domain serendipity is expected to need V2's drift
  (§6.4) — so don't over-conclude from V1 alone; the point of V1 is the harness and
  the baseline.
- **No fixture cohort exists yet.** The vivid multi-domain cohort needed to validate
  serendipity must be authored (~20 `.txt` profiles); current `data/*` groups are
  small and homogeneous. Shared task with the improvement sprint's Track 1
  ([improvement_sprint.md](improvement_sprint.md) §2) — whichever sprint runs
  first authors it.
- **Don't sample on a signed fused matrix.** Keep similarity a *soft probability
  landscape*; collapsing it into a single maximized score recreates mean-collapse.
- **Randomness is a feature** — output varies run-to-run by design (fresh best-of-N
  draw). To pin a batch, persist its `discovery.json`. (Note: `Workflow`-style
  reproducibility isn't a goal here, but seed the RNG from config so a facilitator
  *can* reproduce a batch.)
- **No external frontend.** Output is the markdown report + JSON; the schema is owned
  by this doc.
