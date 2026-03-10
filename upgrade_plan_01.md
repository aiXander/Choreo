# Choreo Upgrade Plan: Configurable Cross-Matching

## Files to Read Before Implementing

The following files (in-order) provide most of the needed context for understanding this project. Read only when needed.

1. **`config/section_prompt.yaml`** (27 lines) — current section definitions + prompt template. Will be rewritten.
2. **`config/config.yaml`** (31 lines) — current pipeline config with weights/budgets. Will be partially rewritten.
3. **`config/scoring_prompt.yaml`** (12 lines) — current scoring prompt. Will be rewritten.
4. **`config/introduction_prompt.yaml`** (27 lines) — current intro prompt. Will be rewritten.
5. **`src/extract.py`** (255 lines) — extraction logic. Key: `build_extraction_prompt()` (line 21), `extract_sections_from_profiles()` (line 43). Both need active-filtering + n_descriptors support.
6. **`src/embed.py`** — embedding generation. Needs changes to handle multi-descriptor sections (embed each variant separately, return additional `multi_desc_embeddings` dict).
7. **`src/candidate.py`** (196 lines) — similarity computation. Key: `compute_fused_similarity_matrix()` (line 29), `apply_recipe()` (line 138). Both need cross-section weights + multi-descriptor max-pooling support.
8. **`main.py`** — pipeline orchestration. Needs to pass `multi_desc_embeddings` dict from embed step through to candidate step.

**Likely no changes needed**: `src/score.py`, `src/match.py`, `src/report.py`, `src/introduction.py`, `src/llm.py`, `src/utils.py`, `src/ingest.py`, `deploy_modal.py`.

---

## Problem

The current pipeline uses **same-section cosine similarity** (skills↔skills, goals↔goals) to find candidates. Even with negative weights on capabilities (preferring dissimilarity), this is a blunt heuristic — "different skills" ≠ "skills that address your needs."

We need **cross-section matching**: compare person A's *needs* against person B's *skills*. This is fundamentally a different operation than what exists today.

### The Embedding Framing Problem

Naive cross-section cosine sim won't work well because needs and skills are phrased differently ("I need someone to pimp my VJ set" vs "I build audio-reactive lasers"). The fix: extract needs **as skill descriptors** at extraction time, so both sections share the same semantic space. To increase recall, the LLM should generate **multiple alternative skill descriptors per need** — controlled by a configurable `n_descriptors` hyperparameter.

## Key Insight from Index Network

Index Network's "intent-driven" model centers matching on **what you want** rather than **who you are**. Their "stakes with reasoning" pattern — where agents provide explanatory justification for each match — maps directly to how our LLM scoring step should work. The multi-signal approach (multiple agents/perspectives converging on a match) validates our hybrid embedding + LLM architecture. We borrow the core philosophy (intent/need-driven discovery) without their protocol complexity.

## Design Principles

1. **All behavior changes live in config** — the Python code should never need editing to switch between use-cases (e.g., "AI art residency social matching" vs "finals project need↔skill matching"). The code reads config and adapts.
2. **No sections are deleted** — every section (old and new) lives in `section_prompt.yaml` with an `active: true/false` flag. Inactive sections are skipped during extraction and embedding but remain in the file ready to be toggled back on.
3. **Cross-section similarity is a generic feature** — the code supports arbitrary `source_target` cross-section pairs defined in config. `needs_skills` is just one instance.
4. **Need descriptor expansion is configurable** — a per-section `n_descriptors` parameter controls how many alternative phrasings the LLM generates for that section's content, giving the embedding more surface area to match against.

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
    n_descriptors: 3
    guideline: "Extract the specific skills, tools, techniques, and expertise this person's project still needs but they currently lack. IMPORTANT: phrase these as skill/technique descriptors (e.g., 'audio-reactive programming', 'laser control systems', 'physical computing') NOT as requests (e.g., NOT 'someone who can help with...'). This ensures needs are directly comparable to skills."
    max_words: 100
```

**`n_descriptors` (new, optional, default 1)**: When set to N > 1, the extraction LLM generates N alternative phrasings for that section. For `needs`, this means each need gets expanded into multiple skill-descriptor variants (e.g., "audio-reactive visuals" → ["audio-reactive programming", "real-time visual effects", "sound-driven projection mapping"]). These variants are generated in a single LLM call and stored as **separate strings** in the extracted data. Each variant is then **embedded independently**, producing N embedding vectors per user for that section. At similarity time, cross-section matching uses **max-pooling** across descriptor variants: `sim(A, B) = max over k of cos_sim(A_needs_k, B_skills)`. This avoids the semantic dilution problem of concatenating variants into a single embedding (which creates a blurry centroid that may not be close to any individual descriptor in embedding space) and ensures that a strong match on *any* single variant surfaces properly.

### Step 2: Code Changes for Section Filtering (`src/extract.py`)

Modify the extraction step to:
1. Read `active` flag from each section config — only extract sections where `active: true`
2. Read `n_descriptors` — if > 1, append to the extraction prompt for that section: instruct the LLM to generate N alternative phrasings as a JSON list of strings (not a single text blob)
3. Store the N descriptor variants as a **list of strings** in `ExtractedSections.sections[section_name]` (instead of a single string). For sections with `n_descriptors=1` (the default), the value remains a single string for backward compatibility.

**This is a small change**: filter the sections dict before building the extraction prompt, and conditionally append a "generate N alternative phrasings" instruction to the per-section guideline. The response parsing must handle list-valued sections for multi-descriptor fields.

#### Exact code locations

**`src/extract.py:21` — `build_extraction_prompt()`**: This function iterates over `sections_config['sections'].items()` (line 25) to build the prompt. Add active-section filtering here — before the loop, filter to only `active: true` sections (default `true` if key missing). For sections with `n_descriptors > 1`, append an instruction like `"Generate {n} alternative skill-descriptor phrasings. Return this section as a JSON list of {n} strings, each being an independent descriptor variant."` to the guideline text inside the loop (line 27-28).

**`src/extract.py:43` — `extract_sections_from_profiles()`**: This is the main extraction entry point. The sections config is loaded at line 67 (`sections_config = load_yaml(sections_config_path)`). The filtering should happen right after this load — create a filtered copy of `sections_config` containing only active sections, then pass that filtered config to `build_extraction_prompt()` (called at line 122).

**Critical**: The response processing loop at lines 165-169 also iterates over `sections_config['sections'].items()` to validate/truncate the LLM output. This loop must use the same filtered sections dict, otherwise it will look for section keys the LLM didn't produce.

**Also filter in**: `generate_schema_hint_from_sections()` (`src/utils.py:101`) and `generate_json_structure_from_sections()` (`src/utils.py:109`) — both iterate `sections_config['sections'].keys()`. Since they're called with the sections_config dict (extract.py line 128), if you filter the dict before passing it, these will automatically only include active sections. Alternatively, add a small helper:

```python
def filter_active_sections(sections_config: dict) -> dict:
    """Return a copy of sections_config with only active sections."""
    filtered = {k: v for k, v in sections_config['sections'].items() if v.get('active', True)}
    return {**sections_config, 'sections': filtered}
```

Place this in `src/extract.py` (or `src/utils.py`) and call it right after `load_yaml()` at line 67.

### Step 3: Code Changes for Multi-Descriptor Embedding (`src/embed.py`)

**Code changes needed** — `embed.py` must now handle sections where `ExtractedSections.sections[name]` is a list of N strings (multi-descriptor sections) rather than a single string. It might be cleanest to make this a list by default and then just dynamically read its length.

- `src/embed.py:84` — `section_names = list(extracted_sections[0].sections.keys())` — still derives sections from `ExtractedSections` objects (auto-filtered to active sections). No change needed here.
- `src/embed.py:114-118` — the text collection loop must detect multi-descriptor sections (value is a list) and embed each variant separately.

**New embedding shape**: The current 3D tensor `(n_users, n_sections, dim)` cannot represent sections with multiple descriptors. Two options:

**Option A (recommended)**: Keep the main 3D tensor for single-descriptor sections. Store multi-descriptor embeddings in a separate dict: `multi_desc_embeddings: Dict[str, np.ndarray]` with shape `(n_users, n_descriptors, dim)` per section. Return this dict alongside the main tensor. This minimizes changes to all code that consumes the main tensor.

**Option B**: Expand to 4D `(n_users, n_sections, max_n_descriptors, dim)` with padding. Cleaner but requires changes everywhere the tensor is indexed.

**Recommended approach (Option A)**:
```python
# In the embedding loop:
multi_desc_embeddings = {}  # section_name → (n_users, n_descriptors, dim)

for section_name in section_names:
    texts = []
    for profile in extracted_sections:
        val = profile.sections.get(section_name, "")
        if isinstance(val, list):
            texts.append(val)  # list of N strings
        else:
            texts.append([val])  # wrap single string in list for uniform handling

    if isinstance(extracted_sections[0].sections.get(section_name), list):
        # Multi-descriptor: embed each variant, store separately
        n_desc = len(texts[0])
        section_embeds = np.zeros((len(texts), n_desc, embed_dim))
        for d in range(n_desc):
            variant_texts = [t[d] for t in texts]
            section_embeds[:, d, :] = embed_texts(variant_texts)
        multi_desc_embeddings[section_name] = section_embeds
    else:
        # Single descriptor: embed normally into the main tensor
        flat_texts = [t[0] for t in texts]
        embeddings[:, sec_idx, :] = embed_texts(flat_texts)
```

The main tensor and `multi_desc_embeddings` dict are both saved to disk and passed to the candidate step. The function signature of `create_section_embeddings()` must be updated to return the additional dict.

**Also update**: `main.py` where `create_section_embeddings()` is called (~line 218) — must capture and pass through the `multi_desc_embeddings` dict to `generate_similarity_matrix()`.

### Step 4: Add Cross-Section Similarity (`src/candidate.py`)

Extend `compute_fused_similarity_matrix()` to support cross-section pairs. This is the core algorithmic addition.

**New config key** in `config.yaml` under `recipe`:
```yaml
recipe:
  section_weights:         # same-section cosine similarity (existing mechanism)
    skills:   0.00         # same skills — not useful for need↔skill mode
    project:  0.15         # similar projects = shared context value
    needs:    0.00         # same needs — not useful
  cross_section_weights:   # NEW: cross-section cosine similarity
    needs_skills: 0.85     # A's needs ↔ B's skills — the KEY signal
```

#### Exact code locations — call chain to modify (3 functions, bottom-up)

**1. `src/candidate.py:29-83` — `compute_fused_similarity_matrix()`** — the core function to extend.

Current signature (line 29-33):
```python
def compute_fused_similarity_matrix(
    embeddings: np.ndarray,  # shape: (n_users, n_sections, embedding_dim)
    section_names: List[str],
    section_weights: Dict[str, float]
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
```

**Add parameters**: `cross_section_weights: Dict[str, float] = None`, `multi_desc_embeddings: Dict[str, np.ndarray] = None`

The `multi_desc_embeddings` dict maps section names to arrays of shape `(n_users, n_descriptors, dim)` for sections with `n_descriptors > 1`. Sections in the main `embeddings` tensor have `n_descriptors=1`.

The existing same-section loop runs at lines 52-59 (iterates `section_names`, computes `cosine_matrix()` per section) and the fusion loop at lines 65-70 (applies `section_weights`). Insert the cross-section loop **after line 70** (after the same-section fusion loop, before the normalization at line 73). The new cross-section code block:

```python
# Cross-section similarity with max-pooling over descriptor variants (e.g., needs↔skills)
cross_weights = cross_section_weights or {}
multi_desc = multi_desc_embeddings or {}

for cross_key, weight in cross_weights.items():
    src_section, tgt_section = cross_key.split("_")  # e.g., "needs", "skills"

    # Get source embeddings — may be multi-descriptor
    if src_section in multi_desc:
        src_emb = multi_desc[src_section]  # (n_users, n_descriptors, dim)
    else:
        src_idx = section_names.index(src_section)
        src_emb = embeddings[:, src_idx:src_idx+1, :]  # (n_users, 1, dim)

    # Get target embeddings — may be multi-descriptor
    if tgt_section in multi_desc:
        tgt_emb = multi_desc[tgt_section]  # (n_users, n_descriptors, dim)
    else:
        tgt_idx = section_names.index(tgt_section)
        tgt_emb = embeddings[:, tgt_idx:tgt_idx+1, :]  # (n_users, 1, dim)

    n_users = src_emb.shape[0]
    n_src_desc = src_emb.shape[1]
    n_tgt_desc = tgt_emb.shape[1]

    # Normalize all descriptor vectors
    src_norm = src_emb / np.linalg.norm(src_emb, axis=2, keepdims=True)
    tgt_norm = tgt_emb / np.linalg.norm(tgt_emb, axis=2, keepdims=True)

    # Compute max-pooled cross-similarity:
    # For each user pair (i, j), find the best-matching descriptor pair
    # cross_sim[i, j] = max over (k, l) of cos_sim(src_i_k, tgt_j_l)
    cross_matrix = np.zeros((n_users, n_users))
    for k in range(n_src_desc):
        for l in range(n_tgt_desc):
            # (n_users, dim) @ (dim, n_users) → (n_users, n_users)
            pair_sim = src_norm[:, k, :] @ tgt_norm[:, l, :].T
            cross_matrix = np.maximum(cross_matrix, pair_sim)

    # Make bidirectional: average forward (A's needs↔B's skills) and backward
    cross_matrix_sym = (cross_matrix + cross_matrix.T) / 2

    fused_matrix += weight * cross_matrix_sym
    total_weight += weight
```

**Complexity note**: The nested loop over `(n_src_desc, n_tgt_desc)` is tiny — typically 3×1 or 3×3 — so it's effectively free. Each inner iteration is a vectorized `(n_users, dim) @ (dim, n_users)` matmul.

Also update the `matrices_dict` return value (lines 77-81) to include cross-section matrices for visualization.

**2. `src/candidate.py:138-167` — `apply_recipe()`** — reads `section_weights` from config and calls `compute_fused_similarity_matrix()`.

Current code at line 155: `section_weights = recipe_config.get('section_weights', {})`. Add right after:
```python
cross_section_weights = recipe_config.get('cross_section_weights', {})
```

Then pass both to `compute_fused_similarity_matrix()` at lines 161-165 (add `cross_section_weights=cross_section_weights, multi_desc_embeddings=multi_desc_embeddings` to the call).

**3. `src/candidate.py:169-196` — `generate_similarity_matrix()`** — calls `apply_recipe()`. **Signature change needed**: add `multi_desc_embeddings: Dict[str, np.ndarray] = None` parameter and pass it through to `apply_recipe()`. The `recipe_config` dict already flows through for `cross_section_weights`.

**Validation**: Add a check in `compute_fused_similarity_matrix()` that all section names in `cross_section_weights` keys exist in `section_names`. The split pattern is `"needs_skills"` → `["needs", "skills"]`. Warn and skip if a referenced section doesn't exist.

**Note**: The existing `cosine_matrix()` helper (`src/utils.py:11`) computes **same-section** pairwise similarity (square matrix). For cross-section, you need the raw `src_norm @ tgt_norm.T` multiplication as shown above — don't use `cosine_matrix()` for this.

### Step 5: Update Main Config (`config/config.yaml`)

The full config for "finals project need↔skill matching" mode:

```yaml
models:
  embedding: "text-embedding-3-small"
  extraction_llm: "gpt-5.2"
  pair_llm: "gpt-5.2"

instruction_prompt:
  goal: "We are matching community residents who are working on their finals projects. Each person has specific skills and project needs. We want to find pairs where one person's skills can directly help the other's project, ideally with mutual benefit — both people helping each other."

budgets:
  extraction_llm_calls: 100
  max_pair_llm_calls: 300
  max_n_llm_evaluations_per_profile: 16
  n_profiles_to_score_together: 4

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

### Step 6: Update Scoring Prompt (`config/scoring_prompt.yaml`)

```yaml
pair_scoring: |
  Context: {goal}

  Your task is to score potential collaboration pairs based on how well each person's skills can address the other's project needs.
  "{instruction}"

  For each pair, evaluate:
  1. Can person A's skills help person B's project needs? (0-1)
  2. Can person B's skills help person A's project needs? (0-1)
  3. How specific and actionable is the potential collaboration?

  Score reflects MUTUAL benefit — pairs where both people can help each other score highest.

  {user_profiles_xml_formatted}

  Return compact JSON with scores for all pairs:
  {json_format_hint}
```

### Step 7: Update Introduction Prompt (`config/introduction_prompt.yaml`)

```yaml
introduction_generation: |
  Context: {goal}

  Your task is to generate collaboration guidance for a matched pair of residents.
  "{instruction}"

  This pair ({user_a_name} and {user_b_name}) has been matched because their skills and project needs are complementary. Create:
  1. A concise mutual introduction explaining what each person can offer the other's project
  2. Specific, actionable starter topics — concrete things they could work on together in the next few days

  <profiles>
  {user_a_name} profile:
  {user1_text}
  ---
  {user_b_name} profile:
  {user2_text}
  </profiles>

  Return compact JSON:
  {{
    "intro": "A mutual introduction explaining how {user_a_name} and {user_b_name} can help each other's projects (neutral framing)",
    "starter_topics": "• topic1 • topic2 • topic3 ..."
  }}
```

## Files to Modify (Summary)

| File | Change | Lines to touch | Type |
|------|--------|---------------|------|
| `config/section_prompt.yaml` | Replace entire `sections:` block — add `active` flag to all existing sections, add `skills`/`project`/`needs` sections, add `n_descriptors` to `needs` | Full rewrite | Config |
| `config/config.yaml` | Replace `instruction_prompt.goal`, `recipe.instruction`, `recipe.section_weights`; add `recipe.cross_section_weights`; change `matching.b_min` to 2 | ~10 lines | Config |
| `config/scoring_prompt.yaml` | Replace `pair_scoring` template text | Full rewrite | Config |
| `config/introduction_prompt.yaml` | Replace `introduction_generation` template text | Full rewrite | Config |
| `src/extract.py` | Filter sections by `active` flag after `load_yaml()` (line 67); modify `build_extraction_prompt()` loop (line 25) for `n_descriptors`; parse multi-descriptor responses as lists; ensure response processing loop (line 165) uses filtered config | ~20 lines | Code |
| `src/embed.py` | Detect multi-descriptor sections (list values in extracted data); embed each variant separately into `multi_desc_embeddings` dict `{section: (n_users, n_desc, dim)}`; update return signature to include this dict; save/load the dict alongside the main tensor | ~30 lines | Code |
| `src/candidate.py` | Add `cross_section_weights` + `multi_desc_embeddings` params to `compute_fused_similarity_matrix()` (line 29); add cross-section max-pooling loop after line 70; read `cross_section_weights` in `apply_recipe()` after line 155; pass `multi_desc_embeddings` through `generate_similarity_matrix()` → `apply_recipe()` → `compute_fused_similarity_matrix()` | ~35 lines | Code |
| `main.py` | Capture `multi_desc_embeddings` dict from `create_section_embeddings()` (~line 218); pass it through to `generate_similarity_matrix()` (~line 244) | ~5 lines | Code |

## What Stays the Same (no reads needed by coding agent)

- `src/score.py` — LLM pair scoring (driven by prompt template, no code change)
- `src/match.py` — greedy b-matching + blending (unchanged algorithm, operates on similarity matrix output)
- `src/report.py` — report generation (unchanged, reads `ExtractedSections` and match edges)
- `src/introduction.py` — introduction generation (driven by prompt template, no code change)
- `src/llm.py` — LLM wrapper with caching and batching (unchanged)
- `src/utils.py` — utility functions (unchanged, but `generate_schema_hint_from_sections()` at line 101 and `generate_json_structure_from_sections()` at line 109 will automatically respect active-only sections IF the filtered config dict is passed to them)
- `src/ingest.py` — profile loading from disk (unchanged)
- `src/cost_tracker.py` — cost tracking (unchanged)
- `src/visualize_similarity.py` — similarity heatmap plots (unchanged)
- `src/tsne.py` — t-SNE visualization (unchanged)
- `src/score_correlation.py` — score correlation plots (unchanged)
- `deploy_modal.py` — Modal serverless deployment (unchanged)

## Data flow diagram with code references

```
main.py:179  load_profiles()         → List[Profile]
    ↓
main.py:201  extract_sections_from_profiles()  → List[ExtractedSections]
    │         ├── src/extract.py:67   load_yaml(sections_config_path) ← FILTER active HERE
    │         ├── src/extract.py:122  build_extraction_prompt()       ← n_descriptors HERE
    │         └── src/extract.py:165  response processing loop        ← parse lists for multi-desc
    ↓
main.py:218  create_section_embeddings()       → (user_ids, section_names, embeddings, multi_desc_embeddings)  ← UPDATED RETURN
    │         ├── src/embed.py:84    section_names from extracted data (auto-filtered)
    │         ├── src/embed.py:114   single-desc sections → main tensor (n_users, n_sections, dim)
    │         └── src/embed.py:NEW   multi-desc sections → multi_desc_embeddings dict {name: (n_users, n_desc, dim)}
    ↓
main.py:244  generate_similarity_matrix(multi_desc_embeddings=...)  ← PASS THROUGH
    │         ├── src/candidate.py:188  apply_recipe(recipe_config, multi_desc_embeddings)
    │         │    ├── line 155: reads section_weights         ← existing
    │         │    ├── line 155+: reads cross_section_weights  ← NEW
    │         │    └── line 161: calls compute_fused_similarity_matrix()
    │         └── src/candidate.py:29   compute_fused_similarity_matrix()
    │              ├── lines 52-59: same-section cosine sim    ← existing
    │              ├── lines 62-70: weighted fusion             ← existing
    │              └── after 70:    cross-section MAX-POOLED sim loop  ← NEW
    ↓
main.py:263  score_pairs_with_llm()   → uses scoring_prompt.yaml (config-only change)
    ↓
main.py:290  create_matches()         → final_edges (unchanged algorithm)
    ↓
main.py:327  generate_introductions() → uses introduction_prompt.yaml (config-only change)
    ↓
main.py:358  generate_all_reports()   → markdown reports (unchanged)
```

## Switching Between Use-Cases

To go from "need↔skill matching" back to the original "social connectivity" mode, only config changes are needed:

1. In `section_prompt.yaml`: flip `active` flags (capabilities/interests/goals/persona → true, skills/project/needs → false)
2. In `config.yaml`: restore original `instruction_prompt.goal`, `recipe.instruction`, `section_weights` (with the original 4 sections), remove/empty `cross_section_weights`, restore `b_min: 3`
3. In `scoring_prompt.yaml` and `introduction_prompt.yaml`: swap back to original prompts

**No Python code changes required to switch modes.** The code is fully config-driven.

To make this even easier, consider keeping versioned config directories (e.g., `config/modes/social/`, `config/modes/needs_skills/`) and selecting via a CLI flag like `--mode needs_skills`. This is optional but would make switching a one-liner.

## Important Notes

- **Clear the cache** after switching modes — section names change so old cached extractions/embeddings are invalid. Run with `--force`.
- The `cross_section_weights` config key is new — if absent or empty, behavior is identical to the original (backward compatible).
- The `active` flag defaults to `true` if not specified, preserving backward compatibility with the existing config.
- The `n_descriptors` parameter defaults to 1 if not specified, preserving backward compatibility.
- The bidirectional averaging of the cross matrix means high scores require MUTUAL complementarity. If you want to also value one-directional help, you could weight forward and backward differently, but symmetric is a good default.
- **`n_descriptors` tuning**: Start with 3 for `needs`. Higher values cast a wider semantic net but increase the number of embedding API calls (N calls per multi-descriptor section instead of 1). Since max-pooling picks the best match per variant pair, adding more variants improves recall without diluting precision — but at diminishing returns. Test with 1 vs 3 vs 5 to find the sweet spot.
