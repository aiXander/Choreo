# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Choreo is an AI-powered user profile matching system that creates meaningful connections between people. It uses LLM embeddings for semantic similarity and LLM-based pair scoring to match users with aligned interests/goals but complementary skills.

## Commands

```bash
# Install
pip install -e .

# Run matching pipeline (main usage)
python main.py --group <group_name> --force

# Examples
python main.py --group real --force    # Run on "real" group
python main.py --group test4           # Run with caching

# Modal deployment (serverless)
modal deploy deploy_modal.py
modal run deploy_modal.py::run_matching_pipeline --user-profiles-json=profiles.json
```

## Architecture

The system runs an 8-step pipeline:

```
1. INGEST   → Load .txt profiles from data/{group}/raw/
2. EXTRACT  → LLM extracts sections: capabilities, interests, goals, persona
3. EMBED    → Generate embeddings per section (3D tensor: users × sections × dims)
4. CANDIDATE → Compute per-section similarity matrices, fuse via recipe weights
5. SCORE    → LLM evaluates top candidate pairs (batched, budgeted)
6. MATCH    → Greedy b-matching: blend embed + LLM scores, enforce b_min/b_max
7. INTRO    → Generate personalized introductions and conversation starters
8. REPORT   → Output markdown reports + cohort.json + visualizations
```

### Key Data Flow

- Input: `data/{group}/raw/*.txt` (one file per user, filename = user ID)
- Processing: `data/{group}/processed/` (extracted sections, cached)
- Embeddings: `data/{group}/embeds/` (embeddings.npy, ids.json, section_names.json)
- Cache: `data/{group}/cache/llm/` (LLM call cache)
- Output: `data/{group}/outputs/` (reports, cohort.json, plots)

### Core Modules (src/)

| Module | Purpose |
|--------|---------|
| `llm.py` | LLM wrapper with caching, cost tracking, async support |
| `extract.py` | Section extraction from profiles via LLM |
| `embed.py` | Multi-section embedding generation |
| `candidate.py` | Similarity matrices, recipe-based fusion |
| `score.py` | Batched LLM pair scoring with budget constraints |
| `match.py` | Greedy b-matching algorithm |
| `report.py` | Markdown report generation |

## Configuration

Main config: `config/config.yaml`

```yaml
models:
  embedding: "text-embedding-3"
  extraction_llm: "gpt-5"
  pair_llm: "gpt-5"

recipe:
  section_weights:
    capabilities: -0.20    # Negative = dissimilarity preferred
    interests:     0.40
    goals:         0.30
    persona:       0.10

blending:
  embed_weight: 0.35
  llm_weight:   0.65       # LLM scores dominate final ranking

matching:
  b_min: 3                 # Min connections per user
  b_max: 4                 # Max connections per user
```

Prompts in `config/`:
- `section_prompt.yaml` - Section extraction guidelines
- `scoring_prompt.yaml` - Pair evaluation prompt template
- `introduction_prompt.yaml` - Introduction generation template

## Key Patterns

**Pair IDs**: Always alphabetically sorted for stability (`alice_bob` not `bob_alice`) - see `utils.stable_pair_id()`

**Caching**: Profile hash-based change detection prevents re-processing unchanged profiles

**Batching**: Scoring evaluates N profiles together (default 4) generating N*(N-1)/2 pairs per LLM call

**Blending Formula**: `final_score = embed_weight * embed_score + llm_weight * llm_score`

## Environment

Requires `.env` with:
```
OPENAI_API_KEY=sk-...
```
