



TODO:

- activate reasoning for LLM pair scoring to give it more tokens before scoring
- run LLM pairing on the actual profile instead of the sections?
- properly scan code for dependencies (remove unneeded ones) and update pyproject.toml
- add ability for bigger projects / brainstorms / ideas to emerge from the profile + context
- create "teams" / "groups" and assign them brainstorm prompts / topics.

#####################################################################################


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