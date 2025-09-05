# TODO:

- create 6 nice, diverse profiles
- make all llm calls run in parallel


- merge
matching:
  b_min: 2
  b_max: 4

report:
  top_matches_per_user: 5


- investigate embedding scoring model: weights, recipe, instructions, ...

- normalize all embedding and LLM scores to same range before combining
- fix combining of llm-score with embedding score:
Current: if llm_score is missing, you set final_weight = embed_score (not scaled by embed_weight).
Consequence: with embed_weight < 1, pairs with LLM get compressed by weights, while pairs without LLM are not—skewing comparisons.


Future optimizations:
- LLM-scoring is generating score, introduction & topics in a single call for each candidate pair: lots of tokens
 ---> Better approach: first generate scores in batches of eg 4 profiles at once, generating 6 scores per call
 ---> Then aggregate all scores, rank and then generate the final introduction & topics

- run LLM pairing on the actual profile instead of the sections?


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

## Matching Recipes

### Overlap (Default)
Matches users with similar interests, goals, and complementary skills.
```yaml
recipe:
  name: "overlap"
  section_weights:
    skills: 0.20
    interests: 0.40
    goals: 0.30
    personality: 0.10
  dissimilar_sections: []
```

### Complement
Matches users with aligned interests/goals but different skillsets.
```yaml
recipe:
  name: "complement"
  section_weights:
    interests: 0.45
    goals: 0.35
    skills: 0.20
    personality: 0.0
  dissimilar_sections: ["skills"]
```

### Debate
Matches users with aligned topics but different perspectives.
```yaml
recipe:
  name: "debate"  
  section_weights:
    interests: 0.5
    goals: 0.2
    personality: 0.3
    skills: 0.0
  dissimilar_sections: ["personality"]
```

## Architecture

```
main.py              # Pipeline orchestration
├── ingest.py        # Load .txt profiles → Profile objects
├── extract.py       # LLM: profile → structured sections  
├── embed.py         # Generate embeddings per section
├── candidate.py     # Fused similarity + top-K candidates
├── score.py         # LLM pair scoring for top pairs
├── match.py         # Greedy b-matching algorithm
├── report.py        # Generate user reports + cohort summary
├── llm.py           # LiteLLM wrapper with caching
└── utils.py         # Cosine similarity, I/O helpers
```

## Configuration

- **Budgets**: Control LLM usage and costs
- **Matching**: Set degree bounds (b_min, b_max) and candidate pool size
- **Blending**: Weight embedding vs LLM scores in final ranking
- **Models**: Choose embedding and LLM models (supports OpenAI, Anthropic, etc.)

## Data Flow

1. **Raw profiles** (.txt) → **Extracted sections** (skills, interests, goals, personality)
2. **Section embeddings** → **Fused similarity matrix** (recipe-based)
3. **Top-K candidates** → **LLM pair scoring** → **Final edge weights**
4. **Greedy b-matching** → **User reports** + **Cohort summary**

## Extending the System

- **New Recipes**: Add new section weighting schemes and dissimilar sections
- **Custom Sections**: Modify `sections.yaml` to extract different profile aspects  
- **Alternative Matching**: Replace greedy b-matching with min-cost flow or other algorithms
- **Rich Outputs**: Extend reports with visualizations, export formats, etc.

## Requirements

- Python 3.9+
- API keys for LLM providers (OpenAI, Anthropic, etc.)
- See `pyproject.toml` for full dependency list