TODO:

- properly scan code for dependencies (remove unneeded ones) and update pyproject.toml
- add ability for bigger projects / brainstorms / ideas to emerge from the profile + context
- create "teams" / "groups" and assign them brainstorm prompts / topics.



Idea generation Pipeline

1. Embedding based cohort sampling (eg make sure every user is part of 3-5 cohorts):
- Based on each users' embeddings for skills/interests/goals/persona
- Run a greedy loop over all users, where at each step you sample the best "team match" based on embedding matrix (using appropriate weights like aligned interests/goals but complementary skills) and keeping count of how often each user has already been paired (b-min spread)
- Add some algorithmic noise into the "team match" ranking scores at each sampling step to add entropy (sometimes match less aligned people also). "alignment_noise" parameter (0-1)

2. Cohort-level ideation (N idea seeds x n_cohorts):

Prompt per cohort: “Given these 3–5 profiles and this venue/context brief, propose N short, concrete project seeds with target outcomes in about 50 words.”
Temperature sweep: T ∈ {0.7, 0.9, 1.1} across shards to inject entropy.

3. Seed embedding & dedup (cheap):

Embed all seed briefs; cluster via HDBSCAN or agglomerative; within each cluster choose medoid; optionally keep 1–3 variants as “modes”.

Seed consolidation (medium LLM):

For each cluster, call LLM to merge similar seeds into a crisp brief (title, purpose, deliverables in 48–72 hours, resource needs, success signals, roles).

Seed scoring (cheap+medium):

Compute novelty: distance from theme centroids + KL divergence vs global topic distribution.

Compute context fit: cosine(sim(seed, event_context_embeddings)).

Quick LLM rubric check (short JSON): feasibility/excitement/impact (0–1).

Pareto select K seeds across novelty × context_fit × feasibility (submodular coverage; see below).

Assignment (medium):

For each seed, rank users by fused similarity (Interests 40 / Goals 30 / Skills 20 / Personality 10) times role-fit heuristics from the brief (e.g., “needs audio dev + facilitator”). Then run coverage-aware b-matching (users→seeds) with constraints:

Each user assigned to at least a_min seeds (e.g., 1–2).

Each seed gets a minimum viable team (roles filled) and a soft cap.

Why it works: The novelty comes from combinatorial sampling of contrasting but coherent micro-cohorts; the reduce stage trims chaos into a tidy set of briefs.




#####################################################################################


# Usage:
python main.py --group test4 --force

# AI-Powered Profile Matching

A flexible user profile matching system that uses LLM embeddings and processing to create meaningful connections between people. The system supports multiple "recipes" for different types of matching (overlap, complement, debate) and generates personalized reports for each user to start introductions.

## Features

- **Multi-modal Matching**: Combines embedding similarity with LLM refinement
- **Flexible Recipes**: Support for overlap, complement, and debate matching strategies  
- **Smart Budgeting**: Configurable LLM call limits and caching
- **B-matching Algorithm**: Ensures fair degree distribution across users
- **Rich Reports**: Personalized markdown reports with conversation starters
- **Extensible**: Easy to add new matching recipes and customize prompts

## Quick Start

1. **Setup Environment**
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   pip install -e .
   ```

2. **Add User Profiles**
   - Place user profile text files in `data/group_name/raw/` (one `.txt` file per user)
   - Filename becomes the user ID (e.g., `alice.txt` → user ID "alice")

3. **Configure Matching** 
   - Edit `config/config.yaml` to adjust models, budgets, and matching parameters
   - Modify `config/section_prompts.yaml` to customize profile extraction
   - Update `config/scoring_prompt.yaml` to customize the scoring prompt

4. **Run Matching**
   ```bash
   python main.py
   ```

5. **View Results**
   - Individual reports: `data/outputs/{user_id}.md` 
   - Cohort summary: `data/outputs/cohort.json`
   - Raw edges: `data/graphs/edges.jsonl`

## How It Works: The Matching Algorithm

This system implements a sophisticated 8-step pipeline that transforms raw user profiles into meaningful connections:

### Step 1: Profile Ingestion 📁
- Load raw text files from `data/raw/` (one `.txt` file per user)
- Each filename becomes a user ID (e.g., `alice.txt` → user "alice")
- Create Profile objects with content hashing for change detection

### Step 2: LLM Section Extraction 🧠
- Use LLM to analyze each profile and extract structured sections:
  - **Skills**: Technical abilities and expertise
  - **Interests**: Hobbies, topics of interest, passions
  - **Goals**: Professional/personal objectives and aspirations  
  - **Personality**: Communication style, work preferences, values
- Smart caching prevents re-processing unchanged profiles
- Configurable word limits per section to manage costs

### Step 3: Multi-Section Embedding 🔢
- Generate vector embeddings for each user's sections separately
- Creates a 3D tensor: `(n_users, n_sections, embedding_dim)`
- Uses OpenAI's text-embedding models by default
- Embeddings capture semantic similarity within each section type

### Step 4: Similarity Matrix Generation 🎯
- Compute cosine similarity matrices for each section independently
- Apply **recipe-based weighting** to combine sections:
  - **Overlap**: Similar interests (40%) + goals (30%) + skills (20%) + personality (10%)
  - **Complement**: Shared interests/goals but different skills
  - **Debate**: Same topics but contrasting perspectives
- Result: Single fused similarity matrix capturing relationship potential

### Step 5: Smart LLM Pair Scoring ⚡
- **Intelligent pair selection**: Use greedy algorithm to select optimal subset of pairs for expensive LLM evaluation
- **Per-user budgeting**: Each user gets evaluated against their top N 'best-match' candidates (configurable)
- **Batch processing**: Evaluate multiple pairs in parallel for speed
- LLM generates:
  - Match quality score (0-1)
  - Personalized introduction text
  - Conversation starter topics

### Step 6: Greedy B-Matching 🔗
- Blend embedding scores + LLM scores
- Run **greedy b-matching algorithm** to create fair matches:
  - Every user gets between `b_min` and `b_max` connections
  - Greedily select highest-weighted edges first
  - Backfill users below minimum degree requirement
- Ensures balanced network where no one is over/under-connected

### Step 7: Personalized Reports 📝
- Generate markdown reports for each user listing their matches
- Include match reasoning, conversation starters, and contact details
- Create cohort summary with network statistics and visualizations

### Step 8: Visualization & Analytics 🎨
- Generate t-SNE plots showing user clusters in embedding space
- Create similarity heatmaps for different sections

## Matching Recipes

The system supports different strategies for matching through configurable "recipes":

### Overlap Recipe (Default)
Find users with similar interests and complementary skills
```yaml
section_weights:
  skills: 0.20      # Some skill overlap helpful
  interests: 0.40   # Strong interest alignment
  goals: 0.30       # Shared objectives
  personality: 0.10 # Compatible styles
```

## Technical Architecture

The system is built with modularity and extensibility in mind:

```
main.py              # Pipeline orchestration & async management
├── ingest.py        # Profile loading & validation
├── extract.py       # LLM section extraction with batching  
├── embed.py         # Multi-section embedding generation
├── candidate.py     # Similarity fusion & candidate generation
├── score.py         # Intelligent LLM pair scoring
├── match.py         # Greedy b-matching algorithm
├── report.py        # Report generation & templating
├── visualize.py     # t-SNE plots & similarity heatmaps
├── llm.py           # LLM wrapper with caching & rate limiting
└── utils.py         # Mathematical utilities & I/O helpers
```

## Requirements

- Python 3.9+
- API keys for LLM providers (OpenAI, Anthropic, etc.)
- See `pyproject.toml` for full dependency list