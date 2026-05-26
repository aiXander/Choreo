# Choreo: AI-Powered Directional Profile Matching

A flexible user profile matching system that uses LLM embeddings, HyDE (Hypothetical Document Embeddings), and directional cross-section similarity to create meaningful connections between people. The system matches users where one person's skills can directly address the other's project needs.

## TODO

Loose ends from the HyDE / directional-matching work, to pick up later (roughly highest-impact first):

1. **Use or drop the directional matrix.** `main.py:329` captures `dir_similarity_matrix` (the asymmetric "who can help whom" signal — the whole point of not symmetrizing in `candidate.py`), but only the symmetrized matrix flows into scoring/matching/intros/reports. Either consume it (order directional intros by it, or surface a directional help score in the report) or drop the extra return value so it isn't dead computation.
2. **Existing groups need `--force`.** Embedding cache keys on (user set, section names); the section rename invalidates old caches anyway, but don't trust a stale `data/{group}/embeds`.
3. **Commit hygiene.** This work-in-progress mixes the feature with model repricing, a Modal signature change, and regenerated `README.md`/`analysis_report.md` (the latter looks like a run artifact — decide if it belongs in git or `.gitignore`). Consider splitting doc regen from code before pushing.

_Resolved: starter-topics bullet char unified on `•`; cross-key parsing consolidated into `utils.parse_cross_key` (now `->`-aware for multi-word sections); error-path intros use the dual-direction format; `deploy_modal.py` dead `group_name` param removed._

## Features

- **Directional Cross-Matching**: Asymmetric need-to-skill matching — "how well can B help A?" is computed independently from "how well can A help B?"
- **HyDE Vocabulary Bridging**: Automatically bridges the semantic gap between needs ("make my installation respond to movement") and skills ("computer vision, sensor integration") using Hypothetical Document Embeddings
- **Configurable Sections**: Active/inactive flags on extraction sections — switch between use-cases (need/skill matching vs. social connectivity) via config alone
- **Multi-Signal Blending**: Combines directional embedding similarity with LLM pair scoring
- **Smart Budgeting**: Configurable LLM call limits and hash-based caching at every step
- **B-Matching Algorithm**: Ensures fair degree distribution across users
- **Directional Introductions**: Each person learns what the other can specifically offer their project
- **Extensible**: All behavior changes live in config — no code edits needed to switch matching modes

## Quick Start

1. **Setup Environment**
   ```bash
   cp .env.example .env
   # Add your OpenRouter API key to .env (https://openrouter.ai/settings/keys)
   pip install -e .
   ```

2. **Add User Profiles** (one `.txt` file per user; filename becomes the user ID, e.g. `alice.txt` -> "alice")
   - **Folder mode**: keep them in any folder you like and point Choreo at it with `--input` (below).
   - **Group mode**: place them in `data/{group_name}/raw/` and use `--group`.

3. **Configure Matching**
   - Edit `config/config.yaml` to adjust models, budgets, weights, and matching parameters
   - Modify `config/section_prompt.yaml` to customize profile sections (with `active` flags)
   - Update `config/scoring_prompt.yaml` and `config/introduction_prompt.yaml` for prompts
   - Adjust `config/hyde_prompt.yaml` to tune HyDE descriptor generation

4. **Run Matching**
   ```bash
   # Folder mode: group name derived from the folder; outputs go inside it.
   python main.py --input /path/to/folder --force

   # Group mode: reads data/<group>/raw, writes to data/<group>/.
   python main.py --group <group_name> --force
   ```

5. **View Results** (under `<folder>/outputs/` in folder mode, or `data/{group_name}/outputs/` in group mode)
   - Individual reports: `…/outputs/{user_id}.md`
   - Cohort summary: `…/outputs/cohort.json`
   - Visualizations: `…/outputs/plots/`

## How It Works: The Matching Algorithm

The system implements a 9-step pipeline that transforms raw user profiles into directional, skill-need-aware connections:

### Step 1: Profile Ingestion
- Load raw text files from `data/{group}/raw/` (one `.txt` file per user)
- Each filename becomes a user ID
- Content hashing for change detection

### Step 2: LLM Section Extraction
- Use LLM to extract structured sections from each profile
- Only **active** sections are extracted (controlled via `active` flag in `section_prompt.yaml`)
- Default active sections for need/skill matching:
  - **Skills**: Concrete tools, techniques, and expertise the person can contribute
  - **Vision**: Broader direction — values, long-term interests, collaboration style
  - **Project**: Current project description — what they're building, its state, and next step
  - **Needs**: Project gaps framed in concrete skill vocabulary (wanted-ad style, so they embed close to others' skills)
- Smart caching prevents re-processing unchanged profiles

### Step 2.5: HyDE Descriptor Generation (NEW)
- **Only runs when `cross_section_weights` are configured** (e.g., `needs_skills: 0.85`)
- For each user's needs, an LLM generates a **hypothetical skill descriptor** — text written in the vocabulary of the target section (skills), describing what the ideal helper's profile would look like
- This bridges the vocabulary gap: "make my installation respond to audience movement" becomes a skill-vocabulary descriptor that will have high embedding similarity with "computer vision, motion sensors, interactive installations"
- Produces a list of `n_descriptors` HyDE phrasings per user (default 1, configurable)
- Results are cached per user keyed by source text hash

### Step 3: Multi-Section Embedding
- Generate vector embeddings for each user's active sections -> 3D tensor `(n_users, n_sections, embedding_dim)`
- **Additionally embed HyDE descriptors** into a separate tensor per cross-section pair: `(n_users, n_descriptors, embedding_dim)`
- Uses OpenRouter embedding models (default: `google/gemini-embedding-2-preview`)

### Step 3.5: t-SNE Visualization
- Generate t-SNE plots showing user clusters in embedding space per section

### Step 4: Directional Similarity Matrix Generation
- **Same-section similarity** (symmetric): cosine similarity within each section (e.g., project-to-project)
- **Cross-section similarity** (ASYMMETRIC): HyDE-bridged needs vs. regular skills embeddings
  - `cross_sim[i][j]` = "how well can j's skills address i's needs" (using i's HyDE descriptor vs. j's skill embedding)
  - `cross_sim[j][i]` = "how well can i's skills address j's needs" (different value!)
  - With `n_descriptors > 1`, max-pooling finds the best-matching descriptor pair
- Weighted fusion produces a **directional** fused matrix
- A **symmetric** version `(dir + dir.T) / 2` is derived for b-matching and candidate selection

### Step 5: Smart LLM Pair Scoring
- Candidate pairs selected using the symmetric similarity matrix
- LLM evaluates each pair holistically — produces a **single score per pair** (not directional)
- The embedding-level asymmetry handles directionality; the LLM excels at holistic "is this a good match?" judgment
- Batch processing for efficiency

### Step 6: Greedy B-Matching
- Blend normalized embedding scores + LLM scores: `final = embed_weight * embed + llm_weight * llm`
- Run greedy b-matching on **symmetric** blended scores
- Every user gets between `b_min` and `b_max` connections

### Step 7: Directional Introduction Generation
- For each matched pair, generate **directional introductions**:
  - `intro_for_a`: What person B can specifically offer person A's project
  - `intro_for_b`: What person A can specifically offer person B's project
- Plus concrete starter topics for collaboration

### Step 8: Report Generation
- Per-user markdown reports with directional match reasoning
- Cohort summary JSON with network statistics

### Step 9: Visualization & Analytics
- Similarity heatmaps for same-section and cross-section matrices
- Score correlation plots (embedding vs. LLM scores)

## Configuration

### Main Config (`config/config.yaml`)

All models are routed through OpenRouter (use `provider/model` slugs):

```yaml
models:
  embedding: "google/gemini-embedding-2-preview"
  embedding_dimensions: 768   # MRL truncation; null = full native size (3072)
  extraction_llm: "google/gemini-3.1-flash-lite"
  pair_llm: "google/gemini-3.1-flash-lite"
  enable_reasoning: false   # true only for reasoning-capable models

instruction_prompt:
  goal: "We are matching community residents who are working on their finals projects..."

hyde:
  n_descriptors: 1   # HyDE phrasings per source section (ready for >1 in future)

recipe:
  section_weights:        # Same-section similarity weights (negative = dissimilarity preferred)
    skills:   -0.10
    vision:    0.30
    project:   0.10
    needs:     0.00
  cross_section_weights:  # Cross-section similarity weights (DIRECTIONAL)
    needs_skills: 0.80   # A's needs vs B's skills

blending:
  embed_weight: 0.35
  llm_weight:   0.65

matching:
  b_min: 2
  b_max: 4
```

### Section Config (`config/section_prompt.yaml`)

Each section has an `active` flag:
```yaml
sections:
  capabilities:
    active: false          # Deactivated for need/skill mode
    guideline: "..."
  skills:
    active: true           # Active for need/skill mode
    guideline: "..."
```

### Switching Between Matching Modes

To switch from **need/skill matching** to a symmetric **social-connectivity** mode, only config changes are needed:

1. In `section_prompt.yaml`: define/activate the sections you want to match on (e.g. interests, goals, persona) via their `active` flags, and deactivate the rest.
2. In `config.yaml`: set `section_weights` for those sections; remove/empty `cross_section_weights`.
3. Swap scoring and introduction prompts to match the new framing.

**No Python code changes required.** When `cross_section_weights` is empty, no HyDE step runs and the pipeline operates in fully symmetric mode.

## Technical Architecture

```
main.py                # Pipeline orchestration
├── src/ingest.py      # Profile loading & validation
├── src/extract.py     # LLM section extraction (active-section filtering)
├── src/hyde.py        # HyDE descriptor generation (vocabulary bridging)
├── src/embed.py       # Multi-section + HyDE embedding generation
├── src/candidate.py   # Directional similarity fusion & candidate generation
├── src/score.py       # Batched LLM pair scoring
├── src/match.py       # Greedy b-matching algorithm
├── src/introduction.py # Directional introduction generation
├── src/report.py      # Report generation & templating
├── src/visualize_similarity.py  # Similarity heatmaps
├── src/tsne.py        # t-SNE visualization
├── src/llm.py         # LLM wrapper with caching & cost tracking
├── src/utils.py       # Utilities, score normalization, I/O helpers
└── src/cost_tracker.py # API cost tracking

config/
├── config.yaml              # Main pipeline configuration
├── section_prompt.yaml      # Section definitions with active flags
├── scoring_prompt.yaml      # LLM pair scoring prompt
├── introduction_prompt.yaml # Directional introduction prompt
└── hyde_prompt.yaml         # HyDE descriptor generation prompt
```

## Key Design Decisions

- **Pair IDs**: Always alphabetically sorted for stability (`alice_bob` not `bob_alice`) — see `utils.stable_pair_id()`
- **Caching**: Hash-based change detection at every step (extraction, HyDE, embeddings) prevents re-processing
- **Lists by default**: HyDE descriptors are always stored as lists, even with `n_descriptors=1`. Same code path handles 1 or many descriptors.
- **Directionality preserved**: The asymmetric cross-section matrix is never symmetrized during computation — symmetry is an aggregation choice made at the matching layer
- **Backward compatible**: When `cross_section_weights` is absent/empty, the pipeline behaves identically to the original symmetric mode

## Requirements

- Python 3.9+
- OpenRouter API key (in `.env` as `OPENROUTER_API_KEY`)
- See `pyproject.toml` for full dependency list

## Deployment

```bash
# Local
python main.py --group <group_name> --force

# Modal (serverless)
modal deploy deploy_modal.py
modal run deploy_modal.py::run_matching_pipeline --user-profiles-json=profiles.json
```
