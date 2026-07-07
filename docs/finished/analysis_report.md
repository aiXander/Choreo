# Choreo: Architecture Analysis & Refactoring Roadmap

**From batch matchmaker to directional cross-matching with HyDE, toward a modular AI-powered community intelligence toolkit**

---

## 1. Current System: What We Have

Choreo is a **directional cross-matching pipeline** that:

1. Ingests raw text profiles from disk
2. Extracts configurable active sections (skills, project, needs) via LLM
3. Generates HyDE (Hypothetical Document Embedding) descriptors to bridge vocabulary between needs and skills
4. Embeds each section + HyDE descriptors into tensors
5. Computes **directional cross-section similarity**: "how well can B's skills address A's needs?" (asymmetric)
6. Symmetrizes for candidate selection, then scores top pairs with LLM
7. Runs greedy b-matching to enforce degree constraints
8. Generates **directional introductions** ("what B can offer A's project" and vice versa) and markdown reports

### Key Architectural Properties

- **Directional by default**: Cross-section similarity produces asymmetric matrices. `score[A->B]` != `score[B->A]`. Symmetry is an aggregation choice at the matching layer.
- **HyDE vocabulary bridging**: Instead of comparing needs and skills directly (which fails due to vocabulary mismatch), the system generates hypothetical skill descriptors from needs, then compares those to actual skills in embedding space.
- **Config-driven behavior**: All section definitions, weights, and matching modes live in YAML config. Switching between need/skill matching and social connectivity requires zero code changes.
- **Active section filtering**: Sections have `active: true/false` flags. Only active sections are extracted, embedded, and used for similarity.

---

## 2. What Changed: From Linear Same-Section to Directional Cross-Section

### 2.1 The Old Approach (Symmetric, Same-Section)

```
score = -0.20 x sim(capabilities) + 0.40 x sim(interests) + 0.30 x sim(goals) + 0.10 x sim(persona)
```

Problems:
- **Complementarity != Dissimilarity**: Negative weight on capabilities assumes "different skills = complementary." But "different" in embedding space is not "useful for your project."
- **No cross-section signal**: Comparing skills-to-skills and needs-to-needs misses the real question: "do your skills match my needs?"
- **Symmetric by construction**: Cannot express "B helps A more than A helps B."

### 2.2 The New Approach (Directional, Cross-Section with HyDE)

```
score[i][j] = 0.15 x sim(project_i, project_j) + 0.85 x cross_sim(needs_i -> skills_j)
```

Where `cross_sim(needs_i -> skills_j)` uses HyDE-bridged embeddings:

1. **Raw needs preserved**: "make my installation respond to audience movement" (authentic, context-rich)
2. **HyDE bridges vocabulary**: LLM generates skill-vocabulary descriptor: "computer vision, motion tracking, interactive installation design, sensor integration"
3. **Cross-section cosine**: HyDE descriptor embedding vs. actual skills embedding produces high similarity when there's a real match

The result is an **asymmetric matrix**: `cross_sim[i][j]` = "j can help i" != `cross_sim[j][i]` = "i can help j".

For b-matching: `edge_weight(A,B) = 0.5 * dir[A->B] + 0.5 * dir[B->A]` (symmetrized at the matching layer, not before).

### 2.3 Why HyDE Instead of Rephrasing at Extraction Time

The earlier plan had needs extracted directly as skill descriptors. This was simpler but fundamentally limited:

- **The best vocabulary bridge depends on what the community offers.** "Make my installation respond to the audience" could map to "computer vision," "capacitive sensing," "ultrasonic rangefinding," or "Kinect programming" — which framing works best depends on which skills actually exist in the group.
- **Raw needs preserve context** for LLM scoring and introductions
- **HyDE is independently tunable** — improve the HyDE prompt without re-extracting all profiles
- **`n_descriptors` scales cleanly** — going from 1 to 3 descriptors requires zero code changes, just a config change

---

## 3. Current Pipeline Data Flow

```
main.py  load_profiles()                    -> List[Profile]
    |
main.py  extract_sections_from_profiles()   -> List[ExtractedSections]
    |    |-- FILTER active sections
    |    |-- needs captured as raw text (no rephrasing)
    |
main.py  generate_hyde_descriptors()         -> Dict[str, List[HydeDescriptors]]
    |    |-- reads cross_section_weights to find source sections
    |    |-- LLM generates skill-vocabulary HyDE phrasings per user
    |    |-- always returns list of descriptors (len=n_descriptors)
    |
main.py  create_section_embeddings()         -> (user_ids, section_names, embeddings, hyde_embeddings)
    |    |-- regular sections -> main tensor (n_users, n_sections, dim)
    |    |-- HyDE descriptors -> hyde_embeddings dict {cross_key: (n_users, n_descriptors, dim)}
    |
main.py  generate_similarity_matrix()
    |    |-- same-section cosine sim (symmetric)
    |    |-- cross-section HyDE sim (ASYMMETRIC)
    |    |-- returns: dir_matrix (asymmetric), sym_matrix (averaged)
    |
main.py  score_pairs_with_llm()
    |    |-- pair selection uses sym_matrix
    |    |-- single score per pair (not directional)
    |
main.py  create_matches()
    |    |-- final_weight uses symmetric blended scores
    |    |-- b-matching runs on symmetric weights
    |
main.py  generate_introductions()
    |    |-- generates directional intros: "what B offers A" + "what A offers B"
    |
main.py  generate_all_reports()              -> per-user markdown reports
```

---

## 4. Matching Modes

The system supports multiple matching modes via config alone:

### Need/Skill Matching (Current Default)

```yaml
sections:
  skills:   { active: true }
  project:  { active: true }
  needs:    { active: true }

recipe:
  section_weights:
    skills: 0.00, project: 0.15, needs: 0.00
  cross_section_weights:
    needs_skills: 0.85
```

Directional: "who can help whom?" Cross-section HyDE similarity dominates.

### Social Connectivity (Legacy Mode)

```yaml
sections:
  capabilities: { active: true }
  interests:    { active: true }
  goals:        { active: true }
  persona:      { active: true }

recipe:
  section_weights:
    capabilities: -0.20, interests: 0.40, goals: 0.30, persona: 0.10
  # No cross_section_weights -> no HyDE, fully symmetric
```

Symmetric: "who is most aligned?" Same-section similarity with configurable weights.

---

## 5. Remaining Limitations & Future Directions

### 5.1 The Bagging Problem (Partially Addressed)

Each section still collapses multi-faceted signals into a single embedding. A person who knows "Python, Rust, Figma, video editing" gets one skills vector. HyDE helps on the cross-section axis (needs find the right skills), but within-section search still suffers from averaging.

**Future**: Dual-layer extraction — enumerated per-attribute embeddings alongside section summaries.

### 5.2 Single HyDE Descriptor

Currently `n_descriptors=1`. A single HyDE phrasing may not cover all semantic angles of a complex need. The data structures already support `n_descriptors > 1` with max-pooling — it's a config change away.

**Future**: `n_descriptors: 3` for wider semantic coverage. Community-aware HyDE that includes a summary of available skills.

### 5.3 LLM Scoring is Symmetric

The LLM produces one score per pair. The directional signal comes entirely from embeddings. For some use cases, directional LLM scores ("how valuable is B for A specifically?") would be more informative.

**Future**: Optional directional LLM scoring mode.

### 5.4 Batch-Only Architecture

The pipeline assumes all profiles arrive simultaneously. Re-running for each new member is wasteful.

**Future**: Incremental ingestion, vector search for candidate retrieval, persistent profile store.

### 5.5 No Learned Signal

Weights are hand-tuned. No feedback loop from successful matches.

**Future**: Collect match outcomes, train lightweight scoring models.

---

## 6. Target Capabilities (Longer-Term Roadmap)

### 6.1 Matching Modes Built on Directional Scores

| Mode | How it uses directional scores | Status |
|------|-------------------------------|--------|
| **Batch-directional** | Compute asymmetric embedding scores, symmetrize for b-matching | **IMPLEMENTED** |
| **User-centric** | Given user A, rank all others by `score_A->B`. No graph optimization. | FUTURE |
| **Collective-optimal** | Maximize total graph value with diversity/bridging constraints via ILP. | FUTURE |

### 6.2 Profile Intelligence Layer
- Structured extraction with per-attribute embeddings
- Persistent profile store (MongoDB/PostgreSQL)
- Queryable: "find everyone who knows Rust" as a filter

### 6.3 Context-Aware Scoring
- Pass matching context to LLM scorer (hackathon team vs. mentorship pair)
- Multi-objective score vectors: {compatibility, complementarity, coverage, novelty}
- Feedback collection for learned scoring

### 6.4 Team Formation Engine
- Given a project brief, find optimal N-person teams via beam search
- Gap analysis: team + goal -> missing capabilities -> candidate search
- Role assignment and diversity optimization

### 6.5 Community Analytics
- Sub-community detection via embedding clustering
- Bridge identification between clusters
- Capability mapping and temporal dynamics

### 6.6 Agent-Facing API
- FastAPI / MCP server exposing composable operations
- Streaming for long-running operations
- Batch + real-time paths

---

## 7. Refactoring Plan: Phased Approach

### Phase 1: Foundation (COMPLETED)
- Directional cross-section similarity with HyDE
- Active section filtering
- Asymmetric similarity matrices
- Directional introductions
- Config-driven mode switching

### Phase 2: Rich Extraction & Multi-Resolution Embeddings
| Task | Description |
|------|-------------|
| **Dual-layer extraction** | Enumerated attributes with categories and levels alongside section summaries |
| **Per-attribute embeddings** | Individual skill/interest vectors for fine-grained matching |
| **Multi-descriptor HyDE** | `n_descriptors > 1` for wider semantic coverage |
| **Community-aware HyDE** | Include community skill summary in HyDE prompt |

### Phase 3: Context-Aware Scoring
| Task | Description |
|------|-------------|
| **Query objects** | `MatchQuery` with context, objective, constraints |
| **Directional LLM scoring** | Optional per-direction LLM scores |
| **Multi-objective blending** | Score vectors instead of single numbers |
| **Feedback collection** | Store match outcomes for learned scoring |

### Phase 4: Team Formation & Community Analytics
| Task | Description |
|------|-------------|
| **Team assembly** | Beam search over candidate combinations given project briefs |
| **Gap analysis** | Team + goal -> missing capabilities -> candidate search |
| **Community mapping** | Clustering, bridge detection, capability mapping |
| **Diversity scoring** | Multi-axis diversity quantification |

### Phase 5: Agent-Facing API & Persistence
| Task | Description |
|------|-------------|
| **Persistent stores** | ProfileStore, VectorStore, RelationshipStore abstractions |
| **Incremental ingestion** | Add profiles without full pipeline re-run |
| **FastAPI / MCP server** | Expose all operations as callable tools |
| **Streaming** | Partial results for long-running operations |

---

## 8. Key Design Decisions

### Storage
Start with file-based (current), migrate to PostgreSQL + pgvector when scale demands. Clean interfaces (`ProfileStore`, `VectorStore`) make the swap painless.

### Scoring Strategy
Adaptive hybrid: embedding-only for initial filtering (fast, cheap), LLM scoring for top-K finalists (nuanced, expensive). Queries can specify quality/cost trade-off.

### Matching Algorithm
Greedy b-matching as default (fast, good enough). ILP solver as option for smaller cohorts where optimality matters. For team formation, beam search with pruning.

### Directionality
Cross-section matrices are asymmetric by construction. Symmetry is always an explicit aggregation step, never implicit. This preserves the directional signal for reporting and future user-centric mode.

---

## 9. Summary

Choreo has evolved from a symmetric same-section matching pipeline to a **directional cross-matching system** with HyDE vocabulary bridging. The key shifts:

1. **From same-section to cross-section**: Matching needs against skills instead of skills against skills
2. **From symmetric to directional**: "How well can B help A?" is computed independently from "How well can A help B?"
3. **From vocabulary mismatch to HyDE bridging**: LLM-generated hypothetical documents bridge the semantic gap between how people express needs and how they describe skills
4. **From hardcoded sections to config-driven**: Active flags, cross-section weights, and HyDE config make all behavior changes config-only

The next evolution — structured per-attribute embeddings, context-aware scoring, team formation, and agent-facing APIs — builds directly on this directional foundation.
