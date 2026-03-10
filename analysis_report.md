# Choreo: Architecture Analysis & Refactoring Roadmap

**From batch matchmaker to modular AI-powered community intelligence toolkit**

---

## 1. Current System: What We Have

Choreo today is a well-structured **batch matching pipeline** that:

1. Ingests raw text profiles from disk
2. Extracts 4 fixed sections (capabilities, interests, goals, persona) via LLM
3. Embeds each section into a 3D tensor `(users × sections × dims)`
4. Computes per-section cosine similarity, fuses via fixed linear weights
5. Selects top candidate pairs, scores them with LLM calls
6. Runs greedy b-matching to enforce degree constraints
7. Generates introductions and markdown reports

The pipeline is monolithic-in-spirit: it runs end-to-end on a batch of profiles, producing a single set of pairwise matches. Every component assumes the full cohort is present at once.

---

## 2. Limitations of the Current Approach

### 2.1 The Linear Combination Problem

The current "recipe" defines match quality as:

```
score = -0.20 × sim(capabilities) + 0.40 × sim(interests) + 0.30 × sim(goals) + 0.10 × sim(persona)
```

This has several deep problems:

**A. Complementarity ≠ Dissimilarity.** A negative weight on capabilities cosine similarity assumes that people with *different* skill vectors make good teammates. But "different" in embedding space is not "complementary." A Python ML engineer and a Rust systems programmer are dissimilar, but not necessarily complementary for any given project. A Python ML engineer and a product designer with UX research skills are also dissimilar, but *are* complementary for building an AI product. The linear combination cannot distinguish these cases.

**B. Non-linear match dynamics.** Real match quality has interaction effects. Two people who share an interest in climate tech AND one has policy expertise while the other has engineering skills is a *multiplicative* signal, not an additive one. The linear combination treats each axis independently — it cannot capture "this combination of axes is what makes this pair special."

**C. Context-blindness.** The same pair might be a great match for a hackathon team but a poor match for a reading group. The weights are static — there's no mechanism to condition the matching objective on the *purpose* of the connection.

**D. No learned signal.** The weights are hand-tuned. There's no feedback loop from successful matches back into the scoring function. As the system processes more cohorts, it doesn't get better at predicting match quality.

### 2.2 The Bagging Problem

Each section (capabilities, interests, goals, persona) collapses a potentially rich, multi-faceted signal into a single embedding vector:

- A person who knows "Python, Rust, Figma, video editing" gets a single capabilities vector that averages all of these into a blurry centroid
- When searching for "someone who knows Figma," this averaged vector has low cosine similarity with a pure "Figma" query vector because the signal is diluted by Python, Rust, etc.
- **Granularity is lost.** You cannot answer "who in this community has experience with facilitation?" without re-running LLM extraction or building custom queries

This is the classic bag-of-features problem: the embedding model does its best to compress the paragraph, but individual skills/interests become invisible once averaged.

### 2.3 The Batch-Only Architecture

The current system assumes:
- All profiles arrive simultaneously
- The full pipeline runs end-to-end
- Results are files on disk
- No persistent state between runs

In practice, communities are *incremental*: new members join one at a time, existing members update their profiles, and matching queries arrive continuously. Re-running the entire pipeline for each new member is wasteful and doesn't support real-time use cases.

### 2.4 Single-Use Output

The pipeline produces one artifact: a set of pairwise matches with introductions. But community builders need many different *queries* against the same profile data:

- "Find the 3 best people to introduce to this new member"
- "Assemble a team of 5 for this project brief, ensuring role coverage"
- "What skills are missing in this existing team for this goal?"
- "Who are the bridge connectors between these two sub-communities?"
- "Generate seed teams for a hackathon with maximum diversity"

The current architecture cannot serve these queries without major pipeline modifications each time.

---

## 3. Target Capabilities

Choreo should become a **community intelligence toolkit** — a set of composable primitives that AI agents can orchestrate to answer diverse questions about people and their potential connections. The core capabilities:

### 3.1 Profile Intelligence Layer
- **Structured extraction** with fine-grained, enumerable attributes (individual skills, individual interests) alongside free-text summaries
- **Persistent profile store** (e.g., MongoDB) with incremental updates
- **Multi-resolution embeddings**: both per-attribute vectors AND section-level summaries
- **Queryable**: "find everyone who knows Rust" should be a simple filter, not a full pipeline run

### 3.2 Matching Engine (Multi-Objective)
- **Pairwise scoring** that goes beyond linear combinations — learned or LLM-conditioned match functions
- **Context-aware matching**: the same profiles matched differently for "hackathon team" vs. "mentorship pair" vs. "reading group"
- **Constraint-aware**: respect degree limits, diversity requirements, role coverage, existing relationships
- **Incremental**: add one new profile, get matches without recomputing everything

### 3.3 Team Formation Engine
- **Seed team assembly**: given a project brief, find the optimal N-person team
- **Gap analysis**: given a team + goal, identify what capabilities/perspectives are missing
- **Role assignment**: map people to roles within a project context
- **Diversity optimization**: ensure teams aren't echo chambers

### 3.4 Community Analytics
- **Sub-community detection**: cluster the network, find natural groupings
- **Bridge identification**: who connects disparate clusters?
- **Capability mapping**: what skills/interests does the community have? What's missing?
- **Temporal dynamics**: how is the community evolving? Who's drifting?

### 3.5 Agent-Facing API
- **Programmatic interface** (not CLI-only) that agents can call with structured queries
- **Composable operations**: embed, score, match, filter, rank — each independently callable
- **Streaming results**: for real-time applications
- **Explainable outputs**: every recommendation includes reasoning traces

---

## 4. Proposed Architecture

### 4.1 High-Level Design

```
┌────────────────────────────────────────────────────-──────┐
│                    AGENT / API LAYER                      │
│  (FastAPI / tool-calling interface / MCP server)          │
│                                                           │
│  Endpoints:                                               │
│  - ingest_profile(text) → ProfileID                       │
│  - query_matches(profile_id, context, constraints)        │
│  - form_team(project_brief, constraints)                  │
│  - analyze_gap(team_ids, goal)                            │
│  - find_by_attribute(skill="Rust")                        │
│  - community_stats()                                      │
│  - explain_match(pair_id)                                 │
└─────────────────────┬─────────────────────────────────-───┘
                      │
┌─────────────────────▼───────────────────────────────────-─┐
│                   QUERY PLANNER                           │
│  Interprets high-level requests, orchestrates primitives  │
│  Decides: embed-only? Need LLM scoring? How many?         │
└─────────────────────┬───────────────────────────────────-─┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                 COMPOSABLE PRIMITIVES                   │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Extract  │  │  Embed   │  │  Score   │  │  Match   │ │
│  │          │  │          │  │          │  │          │ │
│  │ LLM-based│  │ Multi-res│  │ Pairwise │  │ Constrai-│ │
│  │ section  │  │ attribute│  │ embed +  │  │ ned graph│ │
│  │ + attrib │  │ + section│  │ LLM +    │  │ optimi-  │ │
│  │ extract  │  │ vectors  │  │ learned  │  │ zation   │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │  Filter  │  │   Rank   │  │ Generate │  │ Analyze  │ │
│  │          │  │          │  │          │  │          │ │
│  │ Attribute│  │ Multi-   │  │ Intros,  │  │ Clusters,│ │
│  │ + vector │  │ criteria │  │ briefs,  │  │ gaps,    │ │
│  │ search   │  │ sorting  │  │ reports  │  │ bridges  │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  PERSISTENCE LAYER                      │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │   Profile   │  │   Vector     │  │   Relationship  │ │
│  │   Store     │  │   Index      │  │   Graph         │ │
│  │  (MongoDB)  │  │  (pgvector/  │  │  (edges, scores │ │
│  │             │  │   Pinecone/  │  │   history)      │ │
│  │  raw text   │  │   Qdrant)    │  │                 │ │
│  │  extracted  │  │              │  │                 │ │
│  │  sections   │  │  per-attrib  │  │                 │ │
│  │  attributes │  │  per-section │  │                 │ │
│  │  metadata   │  │  vectors     │  │                 │ │
│  └─────────────┘  └──────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 4.2 The Extraction Overhaul: From Bags to Structured Attributes

**Current:**
```json
{
  "capabilities": "Python, ML, systems design, facilitation, video editing",
  "interests": "climate tech, governance, AI safety, music production"
}
```

**Proposed: Dual-layer extraction**
```json
{
  "capabilities": {
    "summary": "Full-stack ML engineer with facilitation and creative production skills",
    "summary_embedding": [0.12, -0.34, ...],
    "attributes": [
      {"name": "Python", "category": "programming", "level": "expert", "embedding": [...]},
      {"name": "machine learning", "category": "technical", "level": "expert", "embedding": [...]},
      {"name": "systems design", "category": "technical", "level": "proficient", "embedding": [...]},
      {"name": "facilitation", "category": "soft_skill", "level": "proficient", "embedding": [...]},
      {"name": "video editing", "category": "creative", "level": "intermediate", "embedding": [...]}
    ]
  },
  "interests": {
    "summary": "Deeply curious about the intersection of technology and governance",
    "summary_embedding": [0.08, 0.22, ...],
    "attributes": [
      {"name": "climate tech", "category": "domain", "embedding": [...]},
      {"name": "governance", "category": "domain", "embedding": [...]},
      {"name": "AI safety", "category": "domain", "embedding": [...]},
      {"name": "music production", "category": "creative", "embedding": [...]}
    ]
  }
}
```

**Why this matters:**
- **Queryable attributes**: "find everyone who knows Rust" is now a filter, not a semantic search
- **Per-attribute embeddings**: similarity between "Python" and "data science" is meaningful; similarity between "Python + facilitation + video editing" averaged together is not
- **Category taxonomy**: enables structured reasoning about skill coverage in teams
- **Proficiency levels**: a team needs an *expert* in ML, not just someone who mentioned it
- **Backward-compatible**: the summary + summary_embedding preserve the current pipeline's behavior

### 4.3 Multi-Resolution Similarity

Replace the single cosine similarity per section with a richer similarity computation:

**Level 1: Attribute-level matching (fine-grained)**
For each attribute of user A, find the nearest attribute of user B in the same section. This produces a *set* of (attribute_A, attribute_B, similarity) triples rather than a single number.

```python
def attribute_level_similarity(user_a_attrs, user_b_attrs):
    """Returns set of best attribute matches between two users."""
    matches = []
    for attr_a in user_a_attrs:
        best_match = max(user_b_attrs, key=lambda b: cosine(attr_a.embedding, b.embedding))
        matches.append((attr_a, best_match, cosine(attr_a.embedding, best_match.embedding)))
    return matches
```

This enables:
- "Alice's Python expertise aligns with Bob's data engineering" (specific match reasoning)
- "Alice has facilitation skills that nobody on Bob's current team has" (gap detection)
- Coverage scoring: what fraction of a project's required skills does this team cover?

**Level 2: Section-level similarity (current behavior, preserved)**
The summary embeddings provide the same coarse-grained signal we have today.

**Level 3: Cross-section interactions (new)**
Sometimes the most interesting signal is *cross-section*: Alice's *skills* align with Bob's *goals* (she can help him achieve what he wants). This requires computing similarity between different sections of different users:

```python
cross_section_sim = cosine(alice.capabilities.summary_embedding,
                           bob.goals.summary_embedding)
```

### 4.4 Beyond Linear: Context-Conditioned Match Scoring

Replace the fixed linear combination with a **context-conditioned scoring function**.

**Option A: LLM-as-judge with structured context (near-term, practical)**

Instead of a static recipe, pass the *matching context* to the LLM scorer:

```yaml
# Old: static recipe
recipe:
  section_weights: {capabilities: -0.20, interests: 0.40, goals: 0.30, persona: 0.10}

# New: context-aware query
query:
  context: "Forming a 48-hour hackathon team to build a climate data dashboard"
  matching_objective: "complementary skills with aligned motivation"
  must_have_coverage: ["frontend", "data engineering", "domain expertise"]
  nice_to_have: ["design", "presentation skills"]
```

The LLM scorer receives this context alongside the profiles and produces richer output:

```json
{
  "score": 0.82,
  "reasoning": "Alice's data engineering + Bob's frontend skills cover 2/3 required areas. Shared climate interest provides motivation alignment.",
  "complementarity_signal": ["Alice:data_engineering ↔ Bob:frontend"],
  "overlap_signal": ["shared: climate_tech interest"],
  "risk_factors": ["both are introverted — may need a facilitator"]
}
```

**Option B: Learned scoring function (medium-term, data-dependent)**

Once the system has processed enough cohorts with outcome feedback (did this introduction lead to a conversation? a project? a lasting collaboration?), train a lightweight model:

```
Input features per pair:
  - Per-section cosine similarities (4 floats)
  - Cross-section similarities (4×4 = 16 floats)
  - Attribute overlap count, complement count
  - Profile length ratio, section balance similarity
  - Community graph features (shared connections, cluster distance)

Output: P(successful_connection | context_type)
```

This could be as simple as a gradient-boosted tree or a small MLP. The key is that the *features* we feed it are richer than a single linear combination allows.

**Option C: Embedding-space learned metric (longer-term)**

Learn a projection that maps profile embeddings into a space where "good match" proximity is directly captured. This is metric learning / contrastive learning territory — requires significant outcome data but would eliminate the need for explicit feature engineering.

### 4.5 Incremental Processing Architecture

**Current flow:**
```
All profiles → Full pipeline → All matches
```

**Proposed flow:**
```
New profile → Extract → Embed → Store in DB
                                     ↓
                              Query: "find matches for this profile"
                                     ↓
                              Vector search (top-K candidates from DB)
                                     ↓
                              LLM scoring (only top-K, not all pairs)
                                     ↓
                              Return ranked matches
```

Key changes:
1. **Profile ingestion is decoupled from matching.** A profile can be ingested and stored without immediately finding matches.
2. **Vector search replaces full similarity matrix.** For N existing profiles and 1 new one, we do K vector lookups instead of computing an N×N matrix.
3. **LLM scoring is on-demand.** Only invoked for the top candidates from vector search, not pre-computed for all pairs.
4. **Historical matches persist.** The relationship graph accumulates over time, informing future matching (e.g., "don't re-match people who already know each other").

**Storage design:**

```
MongoDB / PostgreSQL:
  profiles: {
    _id, raw_text, extracted_sections, attributes[],
    hash, created_at, updated_at, group_ids[]
  }

  relationships: {
    pair_id, user1_id, user2_id,
    scores: {embed, llm, final},
    context, introduction, status,
    created_at, feedback{}
  }

Vector Index (Qdrant / pgvector / Pinecone):
  - Per-section summary vectors (indexed by user_id + section)
  - Per-attribute vectors (indexed by user_id + section + attribute_name)
  - Supports filtered search: "find nearest to X where category='programming'"
```

### 4.6 Team Formation as a First-Class Operation

Team formation is fundamentally different from pairwise matching — it requires reasoning about *sets* of people, not just pairs.

**Algorithm: Context-Aware Team Assembly**

```
Input:
  - project_brief: "Build a climate data dashboard in 48 hours"
  - team_size: 4-5
  - required_roles: ["frontend", "data_eng", "domain_expert"]
  - optional_roles: ["design", "facilitation"]
  - constraints: {diversity_min: 0.3, max_from_same_org: 2}
  - pool: all profiles or a filtered subset

Algorithm:
  1. ROLE EXTRACTION: LLM analyzes project_brief → required capabilities per role
  2. CANDIDATE RETRIEVAL: For each role, vector search for top-K candidates
  3. COMPATIBILITY MATRIX: Score all candidate pairs (embed + optional LLM)
  4. TEAM SEARCH:
     - Enumerate candidate teams via beam search or constraint programming
     - Score each team on:
       a. Role coverage (are all required roles filled?)
       b. Internal compatibility (avg pairwise score)
       c. Diversity (embedding spread, attribute diversity)
       d. Balance (no single person overloaded)
  5. LLM VALIDATION: Top-3 candidate teams evaluated by LLM for coherence
  6. OUTPUT: Ranked teams with role assignments and reasoning
```

**Gap Analysis (related operation):**

```
Input:
  - existing_team: [profile_ids]
  - goal: "Launch an AI-powered governance platform"

Algorithm:
  1. Extract required capabilities from goal
  2. Map existing team members to capabilities (attribute matching)
  3. Identify uncovered capabilities
  4. Search profile DB for candidates who fill gaps
  5. Rank candidates by: gap coverage × team compatibility
```

---

## 5. Refactoring Plan: Phased Approach

### Phase 1: Decouple & Persist (Foundation)

**Goal:** Break the monolithic pipeline into independently callable modules with persistent storage.

| Task | Description |
|------|-------------|
| **Profile Store** | Abstract profile storage behind an interface (`ProfileStore`) with implementations for file-based (current) and MongoDB. Profiles are ingested, extracted, and stored independently of matching. |
| **Embedding Store** | Abstract vector storage behind `VectorStore` interface. Implementations: in-memory numpy (current), pgvector/Qdrant for production. Support incremental insert/update. |
| **Relationship Store** | Persist match results, scores, and feedback. Enables historical queries and prevents re-matching. |
| **Decouple extraction from pipeline** | `extract_profile(text) → ExtractedProfile` becomes a standalone operation that writes to the profile store. |
| **Decouple embedding from pipeline** | `embed_profile(extracted) → vectors` becomes standalone, writes to vector store. |
| **Incremental candidate retrieval** | `find_candidates(profile_id, top_k) → CandidatePair[]` uses vector search instead of full matrix computation. Keep batch matrix computation as an alternative for small cohorts. |

### Phase 2: Rich Extraction & Multi-Resolution Embeddings

**Goal:** Move from bagged sections to structured attributes with per-attribute embeddings.

| Task | Description |
|------|-------------|
| **Dual-layer extraction** | Modify extraction prompt to produce both summary text and enumerated attributes with categories and levels. |
| **Attribute embedding** | Embed each individual attribute in addition to section summaries. |
| **Attribute-level similarity** | Implement fine-grained matching: best-match pairs between users' attributes, coverage scoring, gap detection. |
| **Cross-section similarity** | Compute Alice's skills vs. Bob's goals, enabling "can this person help that person achieve their objectives?" |
| **Backward compatibility** | Keep section-level summaries and the current recipe system working alongside the new fine-grained system. |

### Phase 3: Context-Aware Scoring

**Goal:** Replace static recipes with context-conditioned match functions.

| Task | Description |
|------|-------------|
| **Query objects** | Define `MatchQuery` with context, objective, constraints, and optional recipe override. |
| **Context-aware LLM scoring** | Pass matching context to the LLM scorer. Output includes structured reasoning, not just a number. |
| **Multi-objective blending** | Instead of a single `final_score`, produce a score vector: `{compatibility, complementarity, coverage, novelty}`. Let the query specify how to weight these. |
| **Feedback collection** | Store match outcomes (accepted/rejected, led to conversation, led to project). Build a dataset for learned scoring. |

### Phase 4: Team Formation & Community Analytics

**Goal:** Add set-level operations beyond pairwise matching.

| Task | Description |
|------|-------------|
| **Team assembly** | Given a project brief and constraints, search for optimal teams using beam search over candidate combinations. |
| **Gap analysis** | Given a team and a goal, identify missing capabilities and find candidates to fill them. |
| **Community mapping** | Cluster the profile graph, identify sub-communities, find bridge connectors. |
| **Diversity scoring** | Quantify how diverse a team/cohort is across multiple axes. |

### Phase 5: Agent-Facing API & MCP Server

**Goal:** Expose Choreo as a tool that AI agents can call.

| Task | Description |
|------|-------------|
| **FastAPI service** | REST API wrapping all operations. Stateless request handling against the persistent stores. |
| **MCP server** | Expose Choreo tools for Claude Code and other MCP-compatible agents. Tools: `ingest_profile`, `find_matches`, `form_team`, `analyze_gap`, `search_by_attribute`, `community_stats`. |
| **Streaming** | For long-running operations (team formation with LLM scoring), stream partial results. |
| **Batch + realtime** | Preserve the batch pipeline for cohort processing, add single-profile realtime path. |

---

## 6. Key Design Decisions & Trade-offs

### 6.1 Storage: Embedded vs. External

| Approach | Pros | Cons |
|----------|------|------|
| **File-based (current)** | Zero setup, portable, works with Modal | No concurrent access, no indexing, full reload |
| **SQLite + numpy** | Still portable, adds querying, concurrent reads | Limited vector search, still single-machine |
| **MongoDB + Qdrant** | Full-featured, scalable, proper vector search | Infrastructure overhead, deployment complexity |
| **PostgreSQL + pgvector** | Single DB for everything, mature ecosystem | Vector search less optimized than purpose-built |

**Recommendation:** Start with SQLite + in-memory vectors (Phase 1), migrate to PostgreSQL + pgvector or MongoDB + Qdrant when scale demands it. The key is defining clean interfaces (`ProfileStore`, `VectorStore`) so the swap is painless.

### 6.2 Attribute Extraction: Schema-Free vs. Taxonomy

| Approach | Pros | Cons |
|----------|------|------|
| **Free-form attributes** | Captures anything, no maintenance | Inconsistent naming, hard to aggregate |
| **Fixed taxonomy** | Consistent, queryable, comparable | Misses novel skills, maintenance burden |
| **LLM-normalized + taxonomy** | Best of both: extract freely, then normalize | Extra LLM call, still imperfect |

**Recommendation:** Use LLM-normalized attributes. The extraction LLM produces free-form attributes, then a lightweight normalization step maps them to canonical forms (using embedding similarity to a maintained but extensible skill ontology). Unknown attributes are kept as-is and flagged for ontology expansion.

### 6.3 Scoring: Pure Embedding vs. Hybrid vs. Pure LLM

| Approach | Cost | Latency | Quality |
|----------|------|---------|---------|
| **Embedding-only** | ~$0.001/query | <100ms | Good for similarity, weak on complementarity |
| **Hybrid (current)** | ~$0.10/pair | ~2s/pair | Better nuance, but expensive at scale |
| **Adaptive** | Variable | Variable | Best: use embedding for filtering, LLM for top-K |

**Recommendation:** Keep the hybrid approach but make it adaptive. Embedding-only for initial filtering (top-50 candidates), LLM scoring for the top-10 finalists. Allow queries to specify their quality/cost trade-off.

### 6.4 Matching Algorithm: Greedy vs. Optimal

The current greedy b-matching is fast but suboptimal. For small cohorts (<100), optimal matching via integer linear programming (ILP) is feasible:

```python
# Optimal b-matching via ILP
maximize: Σ w_ij * x_ij
subject to:
  b_min ≤ Σ_j x_ij ≤ b_max  for all users i
  x_ij ∈ {0, 1}
```

**Recommendation:** Keep greedy as default (fast, good enough), add ILP solver (e.g., `scipy.optimize.milp` or `PuLP`) as an option for smaller cohorts where optimality matters. For team formation, use beam search with pruning.

---

## 7. Use Case Matrix

How each target capability maps to the proposed architecture:

| Use Case | Primitives Used | Phase |
|----------|----------------|-------|
| **New member introductions** | Extract → Embed → Vector search → LLM score top-K → Generate intro | Phase 1 |
| **Hackathon team seeding** | Extract roles from brief → Attribute search → Team assembly → LLM validation | Phase 4 |
| **Skill gap analysis** | Attribute extraction → Coverage mapping → Candidate search | Phase 2+4 |
| **Community health check** | Embedding clustering → Bridge detection → Diversity metrics | Phase 4 |
| **Mentorship matching** | Context-aware scoring ("mentor has expertise in mentee's goals") → Cross-section sim | Phase 3 |
| **Project-person fit** | Embed project brief → Attribute matching → Role scoring | Phase 2+3 |
| **Batch cohort matching** | Current pipeline (preserved, optimized) | Phase 1 |
| **"Who knows X?"** | Attribute filter + vector search | Phase 2 |
| **Meeting facilitation** | Find participants → Identify shared interests → Generate discussion prompts | Phase 3+5 |
| **Governance delegation** | Attribute + goal matching → "Who is qualified to represent interest X?" | Phase 3+4 |
| **Idea generation** | Team formation → Cohort ideation → Seed embedding & clustering (per README roadmap) | Phase 4+ |

---

## 8. Community Building, Civic Participation & Collective Governance

Choreo's evolution into a community intelligence toolkit has implications beyond matchmaking:

### 8.1 Community Onboarding

When a new member joins a community (DAO, civic org, open-source project, co-working space), Choreo can:
- Instantly identify the 3-5 people most relevant to connect with, *given the community's current needs*
- Generate contextual introductions: "You should meet Alice — she's also working on climate policy and could use your data viz skills for her current project"
- Identify which working groups / sub-communities the person would thrive in

### 8.2 Collective Intelligence Amplification

For deliberative processes (citizen assemblies, participatory budgeting, community governance):
- **Diverse panel assembly**: form discussion groups that maximize perspective diversity while ensuring sufficient common ground for productive conversation
- **Expertise routing**: when a policy question arises, find the community members with relevant domain expertise
- **Blind spot detection**: "this working group is entirely composed of engineers — consider adding perspectives from affected communities"

### 8.3 Emergent Project Discovery

The README already outlines an idea generation pipeline. With the refactored architecture, this becomes:
1. **Micro-cohort sampling**: use multi-resolution embeddings to form small groups with creative tension (aligned interests, diverse skills)
2. **Context-aware ideation**: each cohort gets a tailored prompt based on their collective capabilities and the community's stated priorities
3. **Cross-pollination**: ideas from one cohort are fed as prompts to another, enabling emergent recombination
4. **Feasibility scoring**: match ideas to available human capital — "this idea requires skills X, Y, Z; we have X and Y covered, Z is a gap"

### 8.4 Relationship Graph as Community Memory

Over time, the relationship graph becomes a valuable asset:
- **Trust networks**: who has successfully collaborated before? Weight future matching by historical collaboration success.
- **Knowledge maps**: the aggregate of all profiles reveals the community's collective capabilities, blind spots, and emerging interests
- **Evolution tracking**: how are individual profiles and the community's composition changing over time?

---

## 9. Technical Priorities (Ordered)

1. **Define storage interfaces** (`ProfileStore`, `VectorStore`, `RelationshipStore`) — abstract away file I/O
2. **Implement dual-layer extraction** — structured attributes alongside summaries
3. **Per-attribute embeddings + vector index** — enable fine-grained search
4. **Single-profile ingestion path** — decouple from batch pipeline
5. **Query-based matching API** — `find_matches(profile_id, context, constraints)`
6. **Context-aware LLM scoring** — pass matching purpose to the scorer
7. **Cross-section similarity** — skills↔goals, capabilities↔needs
8. **Team formation primitive** — beam search over candidate combinations
9. **Gap analysis** — team + goal → missing capabilities → candidate search
10. **Agent-facing API** (FastAPI + MCP) — expose everything as callable tools
11. **Feedback loop** — collect match outcomes, build training data for learned scoring
12. **Community analytics** — clustering, bridges, capability mapping

---

## 10. Summary

Choreo today is a solid batch matching pipeline. To become the "swiss army knife" for community intelligence, it needs three fundamental shifts:

1. **From bags to structured attributes**: extract individual skills/interests as first-class queryable entities with their own embeddings, instead of collapsing everything into section-level summaries
2. **From static recipes to context-conditioned scoring**: match quality depends on *why* you're matching — the scoring function must be parameterized by the matching context, not hardcoded
3. **From batch pipeline to composable primitives with persistent state**: every step (extract, embed, score, match, analyze) should be independently callable against a persistent profile database, not a monolithic end-to-end run

These changes transform Choreo from "run this pipeline on a folder of text files" into "ask any question about how these people could work together" — which is what AI agents need to empower community building, civic participation, and collective governance at scale.
