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



TODO:

- properly scan code for dependencies (remove unneeded ones) and update pyproject.toml
- add ability for bigger projects / brainstorms / ideas to emerge from the profile + context
- create "teams" / "groups" and assign them brainstorm prompts / topics.



Idea generation Pipeline (Yet To build)

initialization:
- from each user profile/bio, use LLM to extract skills/interests/goals/persona sections (text, max 100 words per section)
- For persona, it might be a good idea to also task the LLM with extracting 3 adjectives from a fixed list (e.g., facilitator, finisher, explorer…) instead of inventing persona descriptions free-form (too open-ended)
- embed each section using an embedding model like text-embedding-3-large

1. Embedding based cohort sampling (eg make sure every user is part of 3-5 cohorts):
- Based on each users' embeddings for skills/interests/goals/persona, assemble cohorts (teams) of 3-5 people that would work well together. Think about the fact that “Complementary skills” via negative cosine is relatively weak; complementarity is coverage, not simple dissimilarity ---> can we do better than pure negative cosine sim?
- To do this, run a greedy loop over all users, where at each step you sample the best "team match" based on a combined embedding matrix (using appropriate weights like aligned interests/goals but complementary skills) and keeping count of how often each user has already been assigned to teams with other people, try to create diverse cohorts that mix and match different subgroups
- b-min spread formally: a bipartite b-matching where each user has cap b_user (e.g., 3–5 cohorts) and each other user has exposure limits to avoid the same pairs repeating (pairwise cap).
- Add some algorithmic noise into the "team match" ranking scores at each sampling step to add entropy (sometimes match less aligned people also). "alignment_noise" parameter (0-1). Sample with Gumbel-top-k instead of ad-hoc noise: draw teams proportional to exp(Score/τ). This gives principled entropy without pathological choices.


2. Cohort-level ideation (N idea seeds x n_cohorts):

Prompt per cohort: “Given these 3–5 profiles and this venue/context brief, propose N short, concrete project seeds with target outcomes in 50 words.”
Important: embeddings are very sensitive to text length (longer text leads to more averaged embeddings) so we have to make sure these seed descriptions are all very similar in length.
Temperature sweep: T ∈ {0.7, 0.9, 1.1} across shards to inject entropy. The idea here is that running the same LLM call N times will lead to very similar results, but asking the LLM to generate N different ideas in one pass will force it to diversify. Keep the ideas high level and basic, we don't need details yet at this point, just high level narrative / outcomes.

upgrades:
- Prompt ensemble (“three voices”): for each cohort, generate with 3 distinct rubrics:
  - Feasible-in-72h PM (resources, stakeholders, demo).
  - Wild artist (provocation, narrative, spectacle).
  - Systems/impact (measurements, externalities, handoff).
Ask for 2 ideas per rubric → 6 seeds/ cohort with built-in diversity.

3. Seed idea embedding & dedup (embeddings):

Embed all seed briefs (full cosine similarity matrix)
Create a full seed graph, where edge weights are the similarity scores for each pair
Then run community detection: groups of closely connected ideas (themes).

Run Leiden algorithm: “are these nodes more connected to each other than to the rest?”
Output = partition of seeds into clusters (communities).
Implemented in igraph, networkx, or leidenalg in Python.

Seed consolidation (LLM):

For each cluster, call LLM to consolidate / merge similar seeds into a crisp brief (title, purpose, deliverables in 48–72 hours, resource needs, success signals, roles). This lets the LLM leverage the entropy (diversity) of the previous step and combine strengths from different seeds into stronger versions.
Its fine in this step if some seeds are included in multiple clusters.

Idea scoring (LLM):
Run LLM calls to rank multiple refined ideas in one pass, creating relative rankings eg B > C > A.
Run enough LLM calls to have each idea scored a few times. Assign points to relative rankings and create a global idea rank.

Algorithm outputs the final top-n ideas.

(Optional) Assignment:

For each final idea, rank users by fused similarity (Interests 40 / Goals 30 / Skills 20 / Personality 10) times role-fit heuristics from the brief (e.g., “needs audio dev + facilitator”). Then run coverage-aware b-matching (users→seeds) with constraints:

Each user assigned to at least a_min seeds (e.g., 1–2).
Each seed gets a minimum viable team (roles filled) and a soft cap.

Why it works: The novelty comes from combinatorial sampling of contrasting but coherent micro-cohorts; the reduce stage trims chaos into a tidy set of briefs.


#####################################################################################
#####################################################################################


Current repo guidelines:

# Usage:
python main.py --group test4 --force
