# Choreo Upgrade Plan: Directional Cross-Matching with HyDE

## Architectural Vision

Choreo's matching should be **directional by default**: every score answers "how valuable is person B *for person A*?" rather than "how good is this pair?" This is the fundamental primitive. All matching modes are built on top of directional scores:

| Mode | How it uses directional scores | Status |
|------|-------------------------------|--------|
| **Batch-directional** (demo tomorrow) | Compute asymmetric embedding scores for all pairs. Aggregate into symmetric edge weights. Run b-matching so everyone gets ~equal matches. | **IMPLEMENT NOW** |
| **User-centric** (near-future) | Given user A + optional need query, rank all others by `score_A→B`. No graph optimization, just a ranked list. | FUTURE TODO |
| **Collective-optimal** (longer-term) | Optimize total graph value: `maximize Σ value(edge)` with diversity/bridging/coverage constraints. Operates from the group's POV, not any individual's. | FUTURE TODO |

### What "directional" means concretely

The cross-section similarity `needs_skills` produces an **asymmetric matrix**: `cross_sim[i][j]` = "how well do j's skills match i's needs" ≠ `cross_sim[j][i]`. **Do NOT symmetrize this.** The asymmetry IS the directional signal.

For batch mode, the final edge weight used by b-matching is symmetric: `edge_weight(A,B) = 0.5 * dir_embed[A→B] + 0.5 * dir_embed[B→A]`. The asymmetric matrix is preserved for reports/introductions.

The LLM scoring step produces a **single score per pair** (not directional). The embedding-level asymmetry is the more reliable directional signal; the LLM excels at holistic judgment ("is this a good match?") rather than calibrated directional numbers. Directionality in reports and introductions comes from the directional intro prompt, where the LLM reasons about who helps whom from profiles alone.

---

## Files to Read Before Implementing

1. **`config/section_prompt.yaml`** — current section definitions + prompt template. Will be rewritten.
2. **`config/config.yaml`** — current pipeline config with weights/budgets. Will be partially rewritten.
3. **`config/scoring_prompt.yaml`** — current scoring prompt. Will be rewritten.
4. **`config/introduction_prompt.yaml`** — current intro prompt. Will be rewritten.
5. **`src/extract.py`** — extraction logic. Key: `build_extraction_prompt()` (line 21), `extract_sections_from_profiles()` (line 43). Needs active-filtering.
6. **`src/embed.py`** — embedding generation. Needs to embed HyDE descriptors alongside regular sections.
7. **`src/candidate.py`** (196 lines) — similarity computation. Key: `compute_fused_similarity_matrix()` (line 29), `apply_recipe()` (line 138). Needs cross-section weights + **asymmetric** output.
8. **`main.py`** — pipeline orchestration. Needs new HyDE step + pass new data through pipeline.

**No changes needed**: `src/score.py`, `src/match.py`, `src/llm.py`, `src/ingest.py`, `src/report.py`, `deploy_modal.py`.

---

## Problem

The current pipeline uses **same-section cosine similarity** (skills↔skills, goals↔goals) to find candidates. Even with negative weights on capabilities (preferring dissimilarity), this is a blunt heuristic — "different skills" ≠ "skills that address your needs."

We need **cross-section matching**: compare person A's *needs* against person B's *skills*. This is fundamentally a different operation than what exists today. And it is **inherently directional** — what B's skills can do for A's needs is not the same as what A's skills can do for B's needs.

### The Embedding Framing Problem & HyDE

Naive cross-section cosine sim won't work well because needs and skills are phrased differently ("I need someone to help with my VJ set" vs "I build audio-reactive lasers"). This is the classic **vocabulary mismatch** problem in information retrieval.

**Hypothetical Document Embeddings (HyDE)** is the established solution: instead of embedding the raw query text and hoping it lands near the right documents, you use an LLM to generate a **hypothetical document written in the voice/vocabulary of the target**. This hypothetical document is then embedded and used for similarity search. Because it's written in the target's vocabulary, it produces high cosine similarity with actual matches.

### Our Approach: Lightweight HyDE at the Section Level

**Key design decision**: The `needs` section captures raw, authentic needs as the user expressed them — NOT rephrased as skill descriptors. This preserves the original intent and context in the raw profile sections, which is critical because:

1. The most valuable matches often come from **unexpected** skill-need connections that wouldn't be captured by naive rephrasing
2. Forcing needs into skill-descriptor vocabulary at extraction time loses nuance ("I need help making my installation feel more alive" → "animation programming"? "physical computing"? "kinetic sculpture"? The right answer depends on what the community actually offers)
3. Raw needs are more useful for LLM scoring context and reports

Instead, vocabulary bridging happens in a **separate HyDE step** between extraction and embedding. For any section that appears as a **source** in a `cross_section_weights` pair (e.g., `needs` in `needs_skills`), the system generates HyDE descriptors — skill-vocabulary phrasings that bridge the embedding space to the target section. These HyDE descriptors are what get embedded and used for cross-section similarity. The LLM prompt for this should be in the config and should attempt to generate HyDE descriptors with broad coverage in embedding space since the current version will only generate a single one per **source** (this will be extended later).

**`n_descriptors` (configurable, default 1)**: Controls how many HyDE phrasings are generated per section. Even with `n_descriptors=1`, the output is always a **list** (length 1), so the code path is uniform. When `n_descriptors > 1` (future enhancement), the LLM generates multiple alternative phrasings per need, each embedded independently, with max-pooling at similarity time. This casts a wider semantic net without diluting precision. But for v1, `n_descriptors=1` keeps things simple while the data structures are ready for expansion.

---

## Design Principles

1. **All behavior changes live in config** — the Python code should never need editing to switch between use-cases. The code reads config and adapts.
2. **No sections are deleted** — every section lives in `section_prompt.yaml` with an `active: true/false` flag.
3. **Cross-section similarity is a generic feature** — the code supports arbitrary `source_target` cross-section pairs defined in config.
4. **HyDE is driven by config** — any section that appears as a source in `cross_section_weights` automatically gets HyDE descriptors generated. No hardcoding of which sections need HyDE.
5. **Directionality is the default** — cross-section matrices are asymmetric. Symmetry is an aggregation choice made at the matching layer, not baked into the similarity computation.
6. **Lists by default** — HyDE descriptors are always stored as lists, even when `n_descriptors=1`. This keeps all code paths uniform and ready for multi-descriptor expansion.

---

## Implementation Plan

### Step 1: Extend Section Config (`config/section_prompt.yaml`)

Keep all existing sections, add new ones, use `active` flag to control which are used:

```yaml
sections:
  # --- Original sections (deactivated for need↔skill mode) ---
  capabilities:
    active: false
    guideline: "Extract the concrete skills, tools, and resources this person can contribute to others and the group. Use action-oriented, specific terms (e.g., Python, fundraising, facilitation, woodworking, video editing)."
    max_words: 80
  interests:
    active: false
    guideline: "Extract the main topics, themes, and domains this person is curious about, follows, or enjoys discussing—regardless of expertise."
    max_words: 80
  goals:
    active: false
    guideline: "Summarize this person's current projects and near- to mid-term goals. Capture what they are building, seeking, or aiming toward. What drives them?"
    max_words: 80
  persona:
    active: false
    guideline: "Capture broader psychological, contextual, and human factors about this person not captured in the previous sections: personality traits, collaboration tendencies, values, cultural background, location/timezone, logistical constraints, quirks, or personal trivia that are relevant for the goal."
    max_words: 120

  # --- New sections for need↔skill matching ---
  skills:
    active: true
    guideline: "Extract the concrete skills, tools, techniques, and resources this person can contribute. Use specific, action-oriented terms (e.g., Python, projection mapping, sound design, fabrication, facilitation). Focus on what they could teach or do for others."
    max_words: 100
  project:
    active: true
    guideline: "Describe this person's current project for finals: what they're building, its current state, the vision, and what makes it unique. Be specific about the medium, technology, and creative direction."
    max_words: 120
  needs:
    active: true
    guideline: "Extract what this person's project still needs — gaps, challenges, missing expertise, resources, or help they're looking for. Capture the ACTUAL need as expressed, preserving context and nuance. Do NOT rephrase as generic skill categories — keep the original intent (e.g., 'make my installation respond to audience movement' not just 'computer vision')."
    max_words: 100
```

**Note**: The `needs` section no longer tries to rephrase as skill descriptors. It captures raw needs faithfully. The vocabulary bridging happens in the HyDE step (Step 3).

### Step 2: Code Changes for Section Filtering (`src/extract.py`)

Modify the extraction step to respect `active` flags:

1. Read `active` flag from each section config — only extract sections where `active: true`
2. No `n_descriptors` logic here — initial extraction produces plain text for all sections that faithfully reflects profile context.

#### Exact code locations

**`src/extract.py:21` — `build_extraction_prompt()`**: This function iterates over `sections_config['sections'].items()` (line 25) to build the prompt. Add active-section filtering here — before the loop, filter to only `active: true` sections (default `true` if key missing).

**`src/extract.py:43` — `extract_sections_from_profiles()`**: The sections config is loaded at line 67. The filtering should happen right after this load — create a filtered copy containing only active sections, then pass that to `build_extraction_prompt()`.

**Critical**: The response processing loop at lines 165-169 also iterates over `sections_config['sections'].items()`. This loop must use the same filtered sections dict.

**Helper function** (place in `src/utils.py` or `src/extract.py`):

```python
def filter_active_sections(sections_config: dict) -> dict:
    """Return a copy of sections_config with only active sections."""
    filtered = {k: v for k, v in sections_config['sections'].items() if v.get('active', True)}
    return {**sections_config, 'sections': filtered}
```

Call it right after `load_yaml()` at line 67.

**Also filter in**: `generate_schema_hint_from_sections()` (`src/utils.py:101`) and `generate_json_structure_from_sections()` (`src/utils.py:109`). Since they're called with the sections_config dict, if you filter the dict before passing it, these will automatically only include active sections.

### Step 3: NEW — HyDE Descriptor Generation (`src/hyde.py`)

**This is a new pipeline step** between extraction and embedding. For any section that appears as a **source** in a `cross_section_weights` pair, generate HyDE descriptors that bridge the vocabulary gap to the target section. Try to run these LLM calls in parallel when possible, but always use a single LLM call per section (even in the future case where n_descriptors > 1, this will be one LLM call per source section).

#### How it works

Given `cross_section_weights: { needs_skills: 0.85 }`, the system identifies `needs` as a source section that needs HyDE transformation toward the `skills` vocabulary space.

For each user, the HyDE generator:
1. Takes the raw extracted **source** text, including the original config prompt that extracted that section from the profile.
2. Asks an LLM to extract a HyDE descriptor that matches the **source** section, using the config/hyde_prompt.yaml 
3. Returns a **list** of `n_descriptors` HyDE phrasings (default 1)

#### Config

In `config/config.yaml`:
```yaml
hyde:
  n_descriptors: 1   # Number of HyDE phrasings per source section (default 1)
```

In `config/hyde_prompt.yaml`:
```yaml
hyde_generation: |
  Context: {goal}

  You are generating a Hypothetical Document Embedding (HyDE) — a text written in the
  vocabulary of the TARGET side of a match, to bridge the semantic gap between what someone
  needs and what someone else offers.

  Given this person's project needs:
  <needs>
  {source_text}
  </needs>

  Generate {n_descriptors} skill/technique descriptor(s) that describe what the IDEAL HELPER
  would have on their profile. Write as if you're describing the helper's skills, not the
  seeker's needs.

  Rules:
  - Use specific terms (good for embedding similarity)
  - Use the vocabulary of skills/expertise, not requests
  - Each descriptor should be independent and self-contained
  - Cast a wide net — include both obvious and unexpected skill angles that could address these needs: good matches / solutions can come from a wide variety of skills and capabilities. Don't overly collapse the possible matching space.

  Return JSON:
  {{"descriptors": ["descriptor 1", ...]}}
```

#### Data structure

```python
@dataclass
class HydeDescriptors:
    """HyDE descriptors for a user's section, bridging to target vocabulary."""
    user_id: str
    source_section: str     # e.g., "needs"
    target_section: str     # e.g., "skills"
    descriptors: List[str]  # Always a list, even when n_descriptors=1
```

#### Implementation (`src/hyde.py`)

```python
def generate_hyde_descriptors(
    extracted_sections: List[ExtractedSections],
    cross_section_weights: Dict[str, float],
    hyde_config: dict,
    prompt_template: str,
    goal: str,
    llm_wrapper,
    model: str,
    cache_dir: Path,
    force: bool = False,
) -> Dict[str, List[HydeDescriptors]]:
    """Generate HyDE descriptors for all source sections in cross_section_weights.

    Returns: dict mapping cross_key (e.g., "needs_skills") to list of HydeDescriptors
    (one per user, in same order as extracted_sections).
    """
```

The function:
1. Identifies source sections from `cross_section_weights` keys (e.g., `"needs_skills"` → source is `"needs"`)
2. For each source section, collects the raw text from all users
3. Batches LLM calls to generate HyDE descriptors (using `llm_wrapper.batch_json_complete()`)
4. Caches results per user (keyed by hash of raw section text)
5. Returns `HydeDescriptors` with `descriptors` as a list of length `n_descriptors`

**Caching**: Use the same hash-based caching pattern as extraction. Cache key = hash of (source_text, n_descriptors, prompt_template). Store in `data/{group}/processed/hyde/`.

### Step 4: Embed HyDE Descriptors (`src/embed.py`)

The embedding step now handles two types of data:
1. **Regular section embeddings** → main 3D tensor `(n_users, n_sections, dim)` — unchanged
2. **HyDE descriptor embeddings** → separate dict `hyde_embeddings: Dict[str, np.ndarray]` with shape `(n_users, n_descriptors, dim)` per cross-section key

Even with `n_descriptors=1`, the HyDE embeddings have shape `(n_users, 1, dim)` — always 3D, always list-based. This means the candidate code has a single code path regardless of `n_descriptors`.

#### Changes to `create_section_embeddings()`

Add a new parameter: `hyde_descriptors: Dict[str, List[HydeDescriptors]] = None`

After embedding regular sections into the main tensor, embed HyDE descriptors:

```python
hyde_embeddings = {}  # cross_key → (n_users, n_descriptors, dim)

if hyde_descriptors:
    for cross_key, user_descriptors in hyde_descriptors.items():
        n_desc = len(user_descriptors[0].descriptors)  # same for all users
        section_embeds = np.zeros((n_users, n_desc, embed_dim))
        for d in range(n_desc):
            variant_texts = [ud.descriptors[d] for ud in user_descriptors]
            section_embeds[:, d, :] = get_embeddings(variant_texts, model)
        hyde_embeddings[cross_key] = section_embeds
```

Return `hyde_embeddings` alongside the existing return values. Save to disk alongside the main tensor.

### Step 5: Cross-Section Similarity — DIRECTIONAL (`src/candidate.py`)

**THIS IS THE KEY CHANGE.** The cross-section similarity matrix is **asymmetric by design**. `cross_sim[i][j]` means "how well do j's skills address i's needs." Do NOT average with the transpose.

**New config key** in `config.yaml` under `recipe`:
```yaml
recipe:
  section_weights:
    skills:   0.00
    project:  0.15
    needs:    0.00
  cross_section_weights:
    needs_skills: 0.85     # A's needs ↔ B's skills — DIRECTIONAL
```

#### Core change in `compute_fused_similarity_matrix()`

**Add parameters**: `cross_section_weights: Dict[str, float] = None`, `hyde_embeddings: Dict[str, np.ndarray] = None`

Insert after the same-section fusion loop (after line 70):

```python
# Cross-section similarity — DIRECTIONAL (not symmetrized)
# Uses HyDE embeddings for the source side, regular embeddings for the target side.
# cross_matrix[i][j] = "how well can j help i" (based on j's skills matching i's HyDE-bridged needs)
cross_weights = cross_section_weights or {}
hyde = hyde_embeddings or {}

for cross_key, weight in cross_weights.items():
    src_section, tgt_section = cross_key.split("_")  # e.g., "needs", "skills"

    # Source side: use HyDE embeddings (vocabulary-bridged toward target)
    # Shape: (n_users, n_descriptors, dim) — always 3D, n_descriptors >= 1
    src_emb = hyde[cross_key]  # e.g., hyde["needs_skills"]

    # Target side: use regular section embeddings
    tgt_idx = section_names.index(tgt_section)
    tgt_emb = embeddings[:, tgt_idx:tgt_idx+1, :]  # (n_users, 1, dim)

    n_users = src_emb.shape[0]
    n_src_desc = src_emb.shape[1]
    n_tgt_desc = tgt_emb.shape[1]  # always 1 for regular sections

    # Normalize
    src_norm = src_emb / (np.linalg.norm(src_emb, axis=2, keepdims=True) + 1e-8)
    tgt_norm = tgt_emb / (np.linalg.norm(tgt_emb, axis=2, keepdims=True) + 1e-8)

    # Max-pooled cross-similarity (ASYMMETRIC)
    # cross_matrix[i][j] = max over (k, l) of cos_sim(src_i_k, tgt_j_l)
    # With n_descriptors=1, this is just a single matmul. With n_descriptors>1,
    # it finds the best-matching descriptor pair — same code path either way.
    cross_matrix = np.full((n_users, n_users), -np.inf)
    for k in range(n_src_desc):
        for l in range(n_tgt_desc):
            pair_sim = src_norm[:, k, :] @ tgt_norm[:, l, :].T
            cross_matrix = np.maximum(cross_matrix, pair_sim)

    # *** DO NOT SYMMETRIZE ***
    # cross_matrix[i][j] != cross_matrix[j][i] and that's the point.
    # cross_matrix[i][j] = "j can help i" (j's skills match i's needs)
    # cross_matrix[j][i] = "i can help j" (i's skills match j's needs)

    fused_matrix += weight * cross_matrix
    total_weight += weight
```

**Complexity note**: With `n_descriptors=1`, the nested loop executes exactly once (1×1) — it's just `src_norm[:, 0, :] @ tgt_norm[:, 0, :].T`. When `n_descriptors` increases later, the same code handles it automatically via max-pooling.

#### Updated return value

```python
return fused_matrix, {
    'section_matrices': section_matrices,
    'cross_section_matrices': cross_section_matrices,  # NEW: dict of asymmetric matrices
    'combined_matrix': fused_matrix,                    # ASYMMETRIC when cross-section weights present
}
```

#### Downstream: `generate_similarity_matrix()` return

When cross-section weights are present, `fused_matrix` is asymmetric. The function should return both:
- `dir_similarity_matrix`: the raw asymmetric matrix (for directional reporting)
- `sym_similarity_matrix`: `(dir + dir.T) / 2` (for b-matching and candidate selection)

Both get passed downstream. The symmetric version is used where the current code expects a symmetric matrix. The directional version is stored for introductions and reports.

### Step 6: Update Main Config (`config/config.yaml`)

```yaml
models:
  embedding: "text-embedding-3-small"
  extraction_llm: "gpt-5.2"
  pair_llm: "gpt-5.2"

instruction_prompt:
  goal: "We are matching community residents who are working on their finals projects. Each person has specific skills and project needs. We want to find pairs where one person's skills can directly help the other's project needs."

budgets:
  extraction_llm_calls: 100
  max_pair_llm_calls: 300
  max_n_llm_evaluations_per_profile: 16
  n_profiles_to_score_together: 4

hyde:
  n_descriptors: 1

recipe:
  instruction: "Score this match based on how directly each person's skills can address the other's project needs. The best matches are where both people bring something the other needs. Consider specificity — a vague overlap is less valuable than a concrete skill meeting a concrete need."
  section_weights:
    skills:   0.00
    project:  0.15
    needs:    0.00
  cross_section_weights:
    needs_skills: 0.85

blending:
  embed_weight: 0.35
  llm_weight:   0.65

matching:
  b_min: 2
  b_max: 4

io:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  embeds_dir: "data/embeds"
  outputs_dir: "data/outputs"
  cache_dir: "data/cache"
```

### Step 7: Update Scoring Prompt (`config/scoring_prompt.yaml`)

The LLM scorer produces a **single score per pair** — no directional split. The embedding-level asymmetry handles directionality. The LLM focuses on holistic match quality.

```yaml
pair_scoring: |
  Context: {goal}

  Your task is to score potential collaboration pairs based on how well each person's skills can address the other's project needs.
  "{instruction}"

  For each pair, evaluate:
  1. Can person A's skills help person B's project needs? How directly?
  2. Can person B's skills help person A's project needs? How directly?
  3. How specific and actionable is the potential collaboration?

  Score reflects overall match quality — pairs where both people can help each other score highest, but strong one-directional help is also very valuable.
  
  Score 0.8-1.0: Strong, specific skill↔need matches in at least one direction
  Score 0.5-0.7: Relevant experience that could partially help
  Score 0.2-0.4: Weak or indirect connection
  Score 0.0-0.1: No meaningful skill↔need alignment

  {user_profiles_xml_formatted}

  Return compact JSON with scores for all pairs:
  {json_format_hint}
```

### Step 8: Update Introduction Prompt (`config/introduction_prompt.yaml`)

Introductions are **directional** — each person learns what the OTHER can do for THEM. This is config-only, high value.

```yaml
introduction_generation: |
  Context: {goal}

  Your task is to generate collaboration guidance for a matched pair of residents.
  "{instruction}"

  This pair ({user_a_name} and {user_b_name}) has been matched because their skills and project needs are complementary.

  Create:
  1. A directional introduction for {user_a_name} explaining what {user_b_name} can specifically offer their project
  2. A directional introduction for {user_b_name} explaining what {user_a_name} can specifically offer their project
  3. Specific, actionable starter topics — concrete things they could work on together

  Be specific about WHICH skills match WHICH needs. Vague "you have shared interests" is not useful.

  <profiles>
  {user_a_name} profile:
  {user1_text}
  ---
  {user_b_name} profile:
  {user2_text}
  </profiles>

  Return compact JSON:
  {{
    "intro_for_a": "What {user_b_name} can offer {user_a_name}'s project (addressed to {user_a_name})",
    "intro_for_b": "What {user_a_name} can offer {user_b_name}'s project (addressed to {user_b_name})",
    "starter_topics": "* topic1 * topic2 * topic3 ..."
  }}
```

**NOTE**: This changes the introduction JSON schema from `{intro, starter_topics}` to `{intro_for_a, intro_for_b, starter_topics}`. The code in `src/introduction.py` that parses the response needs a minor update. For the existing `Edge.intro` field, store both intros concatenated: `intro = f"For {user1}: {intro_for_a}\n\nFor {user2}: {intro_for_b}"`

---

## Files to Modify (Summary)

| File | Change | Type |
|------|--------|------|
| `config/section_prompt.yaml` | Add `active` flag to all sections, add `skills`/`project`/`needs` sections | Config |
| `config/config.yaml` | New goal, instruction, section_weights, cross_section_weights, hyde config | Config |
| `config/scoring_prompt.yaml` | Updated scoring prompt (single score, needs↔skills aware) | Config |
| `config/introduction_prompt.yaml` | Directional intros (`intro_for_a`, `intro_for_b`) | Config |
| `config/hyde_prompt.yaml` | **NEW** — HyDE descriptor generation prompt | Config |
| `src/extract.py` | Filter active sections (~10 lines) | Code |
| `src/hyde.py` | **NEW** — HyDE descriptor generation module | Code |
| `src/embed.py` | Embed HyDE descriptors alongside regular sections | Code |
| `src/candidate.py` | Cross-section weights, **asymmetric** fused matrix, return both dir and sym matrices | Code |
| `src/introduction.py` | Parse `intro_for_a`/`intro_for_b` from LLM response | Code |
| `main.py` | New HyDE pipeline step, pass data through, handle new intro format | Code |

## What Stays the Same

- `src/score.py` — LLM pair scoring (single score per pair, driven by prompt template)
- `src/match.py` — greedy b-matching + blending (unchanged algorithm, operates on symmetric similarity matrix)
- `src/report.py` — report generation (unchanged, reads Edge objects)
- `src/llm.py` — LLM wrapper (unchanged)
- `src/ingest.py` — profile loading (unchanged)
- `src/utils.py` — utilities (unchanged, but `filter_active_sections` helper added)
- `src/cost_tracker.py` — cost tracking (unchanged)
- `src/visualize_similarity.py` — similarity heatmap plots (unchanged)
- `src/tsne.py` — t-SNE visualization (unchanged)
- `src/score_correlation.py` — score correlation plots (unchanged)
- `deploy_modal.py` — Modal deployment (unchanged)

---

## Pipeline Data Flow

```
main.py  load_profiles()              → List[Profile]
    ↓
main.py  extract_sections_from_profiles()   → List[ExtractedSections]
    │    ├── FILTER active sections
    │    └── needs captured as raw text (no rephrasing)
    ↓
main.py  generate_hyde_descriptors()   → Dict[str, List[HydeDescriptors]]    ← NEW STEP
    │    ├── reads cross_section_weights to find source sections
    │    ├── LLM generates skill-vocabulary HyDE phrasings per user
    │    └── always returns list of descriptors (len=n_descriptors, default 1)
    ↓
main.py  create_section_embeddings()   → (user_ids, section_names, embeddings, hyde_embeddings)
    │    ├── regular sections → main tensor (n_users, n_sections, dim)
    │    └── HyDE descriptors → hyde_embeddings dict {cross_key: (n_users, n_descriptors, dim)}
    ↓
main.py  generate_similarity_matrix()
    │    ├── same-section cosine sim (symmetric)
    │    ├── cross-section HyDE sim (ASYMMETRIC)    ← KEY CHANGE
    │    └── returns: dir_matrix (asymmetric), sym_matrix (averaged)
    ↓
main.py  score_pairs_with_llm()
    │    ├── pair selection uses sym_matrix
    │    └── single score per pair (unchanged format)
    ↓
main.py  create_matches()
    │    ├── final_weight uses symmetric blended scores
    │    └── b-matching runs on symmetric weights (UNCHANGED algorithm)
    ↓
main.py  generate_introductions()
    │    └── generates directional intros: "what B offers A" + "what A offers B"
    ↓
main.py  generate_all_reports()        → per-user markdown reports
```

---

## Why HyDE Instead of Rephrasing at Extraction Time

The earlier plan had needs extracted directly as skill descriptors. This was simpler but had a fundamental limitation: **the best vocabulary bridge depends on what the community actually offers**. A need like "make my installation respond to the audience" could map to "computer vision," "capacitive sensing," "ultrasonic rangefinding," or "Kinect programming" — and which framing works best depends on which skills actually exist in the group.

With a separate HyDE step:
1. **Raw needs preserve context** — the LLM scorer and intro generator see the authentic need, not a lossy transformation
2. **HyDE is independently tunable** — you can improve the HyDE prompt without re-extracting all profiles
3. **n_descriptors scales cleanly** — going from 1 to 3 descriptors later requires zero code changes (same list-based data path), just a config change + re-running the HyDE step
4. **Community-aware HyDE (future)** — the HyDE prompt could be enhanced to include a summary of available skills in the community, generating descriptors specifically tuned to the actual skill landscape. This is impossible if vocabulary bridging is baked into extraction.

---

## Important Notes for Implementation

- **Clear the cache** after switching modes — run with `--force`.
- The `cross_section_weights` config key is new — if absent or empty, behavior is identical to the original (fully symmetric, backward compatible). No HyDE step runs.
- The `active` flag defaults to `true` if not specified, preserving backward compatibility.
- **HyDE descriptors are always lists** — even with `n_descriptors=1`, the structure is `["single descriptor"]`. This means embed.py always creates `(n_users, n_descriptors, dim)` arrays and candidate.py always runs max-pooling (which with `n_descriptors=1` is just a single matmul). No branching on list vs string.
- **HyDE generation uses the same LLM wrapper** — batched, cached, cost-tracked. Same pattern as extraction.

---

## FUTURE TODO: User-Centric Mode

**Not implementing now, but the directional infrastructure built above makes this straightforward.**

User-centric mode: given a single focal user A (and optionally a need query), rank all other users by directional embedding score. No b-matching, no graph optimization — just a ranked list.

What's needed:
- CLI flag: `--user alice` or `--user alice --query "need help with projection mapping"`
- Skip full N×N matrix — only compute one row of the directional matrix (A vs all others)
- Skip b-matching — just return top-K ranked by directional score
- LLM scoring only for the focal user's top-K candidates
- Single-user report output

## FUTURE TODO: Collective-Optimal Mode

**Not implementing now. Requires different optimization objective.**

Collective mode: optimize the matching graph for maximum total value with diversity/bridging/coverage constraints. Would use ILP or similar optimization rather than greedy b-matching.

## FUTURE TODO: Enhanced HyDE

- **n_descriptors > 1**: Generate multiple HyDE phrasings per need. Already supported by the data structures (list-based), just change `hyde.n_descriptors` in config.
- **Community-aware HyDE**: Pass a summary of available community skills into the HyDE prompt, so descriptors are tuned to what's actually available. This would be a major recall improvement.
- **Bidirectional HyDE**: Also generate HyDE descriptors for the target side (skills → need-vocabulary). Currently only the source side is HyDE-transformed.

## Switching Between Use-Cases

To go from "need↔skill matching" back to the original "social connectivity" mode, only config changes are needed:

1. In `section_prompt.yaml`: flip `active` flags (capabilities/interests/goals/persona → true, skills/project/needs → false)
2. In `config.yaml`: restore original goal, instruction, section_weights; remove/empty `cross_section_weights`
3. In `scoring_prompt.yaml` and `introduction_prompt.yaml`: swap back to original prompts

**No Python code changes required to switch modes.** When `cross_section_weights` is empty, no HyDE step runs and the pipeline operates in fully symmetric mode, identical to the original behavior.
