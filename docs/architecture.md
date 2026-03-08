# Merkraum Architecture

*A dual-store knowledge graph with belief tracking and neuroscience-inspired memory consolidation for AI agents.*

## The Problem

AI agents built on Large Language Models are stateless between sessions. Without persistent external memory, each invocation begins with amnesia. Vector stores (RAG) solve retrieval but not reasoning:

1. **No causal reasoning.** Semantic similarity is not structural relationship. An agent can't traverse from a regulation through its organizational impact to the consulting engagement that addresses it.
2. **No contradiction detection.** When new information conflicts with existing knowledge, a vector store treats both as valid entries with high cosine similarity.
3. **No temporal validity.** Knowledge has a shelf life. Vector stores lack the machinery to manage belief lifecycle.

## Why Not Mem0?

We started with [Mem0](https://github.com/mem0ai/mem0), an established memory layer for AI agents (23,000+ GitHub stars). After deploying it against real operational data, the results were unusable:

- **1,812 distinct relationship types** — including `WORKS_ON`, `WORKS_AT`, `WORKS_WITH`, `WORKS_FOR`, `WORKING_ON`, all semantically overlapping
- **335 distinct node labels** — including `PERSON`, `Person`, `person`, `HUMAN`, `INDIVIDUAL`
- The graph was unqueryable: any Cypher query needed to enumerate hundreds of types

**Root cause:** Open-vocabulary LLM extraction without schema constraints produces unbounded schema drift. This is architectural, not incidental. Apple's ODKE+ research confirms: ontology-guided extraction achieves 98.8% precision; unconstrained extraction cannot achieve usable precision at scale.

## The Solution: Fixed Schema + Dual Store

Merkraum enforces a fixed schema at extraction time:

- **10 node types:** Person, Organization, Project, Concept, Regulation, Event, Belief, Artifact, Interview, Quote
- **16 relationship types:** SUPPORTS, CONTRADICTS, COMPLEMENTS, SUPERSEDES, EXTENDS, REFINES, CREATED_BY, AFFILIATED_WITH, APPLIES, IMPLEMENTS, PARTICIPATED_IN, PRODUCES, REFERENCES, TEMPORAL, MENTIONS, PART_OF

The type system was derived from organizational cybernetics requirements but is general enough for most knowledge domains.

### Dual-Store Architecture

```
         MCP Client (Claude / Cursor / ChatGPT)
                      |
                      | MCP (HTTP or stdio)
                      v
            ┌──────────────────┐
            │  Merkraum Server │
            │  (Python + MCP)  │
            └───────┬──────┬───┘
                    |      |
               ┌────┘      └────┐
               v                v
         ┌──────────┐    ┌──────────┐
         │  Neo4j   │    │  Qdrant  │
         │ (graph)  │    │ (vector) │
         └──────────┘    └──────────┘
```

- **Neo4j** handles structural relationships, graph traversal, and Cypher queries
- **Qdrant** (with FastEmbed) handles semantic similarity search — no external API keys needed
- Queries hit both stores: semantic search finds relevant content, graph traversal reveals structural context

### Why Two Stores?

Vector search and graph traversal answer fundamentally different questions:

| Capability | Vector Store | Knowledge Graph |
|-----------|-------------|-----------------|
| "What's similar to X?" | Strong | Weak |
| "How does X relate to Y?" | Weak | Strong |
| "What contradicts X?" | Cannot | Native |
| "What superseded X?" | Cannot | Native |
| "What changed since last week?" | Difficult | Native (temporal edges) |

Using both gives agents both retrieval quality and reasoning capability.

## Belief Tracking

Every knowledge claim in Merkraum can become a **Belief** — a first-class node with:

- **Confidence score** (0.0-1.0): How certain is this?
- **Status**: active, uncertain, contradicted, superseded, consolidated
- **Provenance**: What cycle/source created this belief?
- **Contradiction detection**: When a new belief conflicts with an existing one, both are linked with a CONTRADICTS edge and flagged for resolution

This means an agent can ask: "What do I believe about X? How confident am I? Has anything contradicted this?"

### Belief Lifecycle

```
     New information
           |
           v
    ┌─────────────┐
    │   Extract    │  (LLM classifies entities + relationships)
    │  & Ingest    │
    └──────┬──────┘
           |
           v
    ┌─────────────┐     ┌──────────────┐
    │   Active     │────>│ Contradicted │  (new evidence conflicts)
    │  (conf 0.8)  │     └──────────────┘
    └──────┬──────┘
           |
           v
    ┌─────────────┐     ┌──────────────┐
    │  Uncertain   │────>│  Superseded  │  (newer belief replaces)
    │  (conf 0.5)  │     └──────────────┘
    └──────┬──────┘
           |
           v
    ┌─────────────┐
    │ Consolidated │  (merged with similar beliefs)
    └─────────────┘
```

## Graph Dreaming

Inspired by neuroscience research on memory consolidation (McClelland et al., O'Reilly & Frank), Merkraum includes a "dreaming" protocol with three operations:

1. **Replay** (analogous to hippocampal replay): Random walks through the graph surface unexpected connections between distant knowledge clusters
2. **Consolidation** (analogous to hippocampus-to-neocortex transfer): Episodic beliefs are merged into abstract, generalizable beliefs
3. **Reflection** (analogous to Default Mode Network): Structural health analysis — orphan detection, hub over-centralization, schema discipline

These operations run periodically and produce measurable improvements in graph coherence.

## MCP Integration

Merkraum is an MCP server. Any MCP-compatible client can use it directly:

| Tool | What it does |
|------|-------------|
| `search_knowledge` | Semantic search across the knowledge graph |
| `traverse_graph` | Walk relationships from any entity |
| `add_knowledge` | Add a structured entity (no LLM needed) |
| `add_relationship` | Link two entities with a typed relationship |
| `ingest_knowledge` | Extract entities from free text (requires OpenAI key) |
| `list_beliefs` | View beliefs by status |
| `get_graph_stats` | Node and edge counts by type |
| `health_check` | Verify connectivity |

## Self-Hosting

Everything runs locally. No cloud dependencies required:

```bash
git clone https://github.com/nhilbert/merkraum.git
cd merkraum
docker compose up -d
```

Three containers: Merkraum server, Neo4j, and Qdrant. Total RAM: ~4 GB minimum.

## Background

Merkraum was developed as part of the [Viable System Generator (VSG)](https://merkraum.de) project — an autonomous AI agent built on Stafford Beer's Viable System Model. The knowledge graph architecture was designed to solve the memory problem that arises when an AI agent needs to maintain coherent knowledge across thousands of operational cycles.

The architecture is described in detail in a technical paper: *A Dual-Store Knowledge Graph with Neuroscience-Inspired Memory Consolidation for an Autonomous Cybernetic Agent* (Hilbert & VSG, 2026).

## License

Business Source License 1.1 (BSL 1.1). Free for personal and non-commercial use.

---

Built by [Supervision Rheinland](https://merkraum.de)
