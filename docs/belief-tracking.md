# Belief Tracking

*Developer guide for epistemic belief tracking in Merkraum.*

## Why Beliefs?

Standard AI memory systems store facts as flat key-value pairs or vector embeddings. When new information contradicts existing knowledge, both versions coexist silently — the agent has no mechanism to detect the conflict, let alone resolve it.

Merkraum treats knowledge claims as **beliefs** — propositions with confidence levels, provenance, and lifecycle state. This gives agents the ability to:

- Know *how confident* they are about a claim
- Detect when new evidence *contradicts* existing knowledge
- Track which beliefs have been *superseded* by newer information
- Audit the *origin* of any belief (when it was created, from what source)

Few agent memory systems expose these as user-facing primitives. Merkraum's approach is distinguished by its fixed schema discipline and explicit graph edges (CONTRADICTS, SUPERSEDES) that make epistemic changes auditable — not just tracked internally.

## Creating Beliefs

### Via MCP: `add_knowledge` with `node_type="Belief"`

```json
{
  "tool": "add_knowledge",
  "arguments": {
    "name": "React Server Components reduce client bundle size by 30-50%",
    "summary": "Based on Next.js migration data from three production apps",
    "node_type": "Belief",
    "confidence": 0.8
  }
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | yes | The belief proposition. Should be a falsifiable claim, not a vague observation. |
| `summary` | string | yes | Supporting evidence or context. |
| `node_type` | string | yes | Must be `"Belief"` for belief tracking. |
| `confidence` | float | no | 0.0 = speculation, 0.5 = uncertain, 0.7 = inferred (default), 0.9 = stated fact, 1.0 = axiomatic. |

**Writing good belief propositions:**

| Good (falsifiable) | Bad (vague) |
|---|---|
| "PostgreSQL outperforms MongoDB for write-heavy workloads under 10TB" | "PostgreSQL is good" |
| "Team velocity drops 20% during Q4 due to holiday schedules" | "Q4 is slower" |
| "The EU AI Act Article 6 deadline is August 2025" | "AI regulation is coming" |

### Via MCP: `ingest_knowledge`

For unstructured text, `ingest_knowledge` extracts entities and relationships automatically using an LLM. Beliefs are created when the extraction pipeline identifies knowledge claims in the text.

```json
{
  "tool": "ingest_knowledge",
  "arguments": {
    "text": "After migrating our auth service from JWT to session tokens, login latency dropped from 340ms to 80ms. The team consensus is that JWTs were adding unnecessary overhead for server-side rendered apps."
  }
}
```

This is async — the server returns a `job_id` immediately. Poll with `check_ingestion_status` to track completion.

**What happens during ingestion:**

1. LLM extracts entities (typed nodes) and relationships from the text
2. For each extracted Belief node, the contradiction detection pipeline runs automatically
3. Semantic search finds existing beliefs with similar content
4. If a conflict is detected, a `CONTRADICTS` edge is created between the two beliefs
5. If the new belief supersedes an old one, the old belief is marked `superseded`

## Querying Beliefs

### `list_beliefs`

```json
{
  "tool": "list_beliefs",
  "arguments": {
    "status": "active"
  }
}
```

**Status values:**

| Status | Meaning |
|--------|---------|
| `active` | Current, trusted beliefs (default) |
| `uncertain` | Beliefs with insufficient evidence or conflicting signals |
| `contradicted` | Beliefs that conflict with another active belief |
| `superseded` | Beliefs replaced by newer, more accurate information |
| `consolidated` | Beliefs merged during graph dreaming (memory consolidation) |

**Response format:**

```json
{
  "beliefs": [
    {
      "name": "React Server Components reduce client bundle size by 30-50%",
      "confidence": 0.8,
      "status": "active",
      "source_cycle": "Z1194",
      "created_at": "2026-03-08T10:30:00Z"
    }
  ],
  "count": 1,
  "status_filter": "active"
}
```

### Finding beliefs about a topic

Use `search_knowledge` with a natural language query:

```json
{
  "tool": "search_knowledge",
  "arguments": {
    "query": "React performance optimization"
  }
}
```

This performs semantic search across the entire knowledge graph, including beliefs. Results include similarity scores, allowing the agent to find relevant beliefs even when exact keywords don't match.

### Traversing belief relationships

Use `traverse_graph` to walk from a belief to its supporting evidence, contradictions, and related entities:

```json
{
  "tool": "traverse_graph",
  "arguments": {
    "entity": "React Server Components reduce client bundle size by 30-50%"
  }
}
```

Returns all connected nodes grouped by relationship type — what SUPPORTS this belief, what CONTRADICTS it, what it SUPERSEDES, etc.

## Contradiction Detection

When a new belief is ingested, Merkraum automatically checks it against existing active beliefs:

```
New belief arrives
       |
       v
Semantic search (vector similarity)
       |
       v
Filter to active beliefs in graph
       |
       v
LLM comparison (pairwise)
       |
       v
Classification:
  IDENTICAL    → skip (duplicate)
  COMPATIBLE   → add normally
  CONTRADICTION → create CONTRADICTS edge, flag both
  SUPERSESSION → mark old belief superseded
  REFINEMENT   → create REFINES edge
```

**Example — contradiction detected:**

1. Existing belief: *"JWT tokens are stateless and reduce server load"* (confidence 0.8)
2. New ingestion text: *"After removing JWTs, our server load dropped 40% because we eliminated per-request signature verification"*
3. Pipeline extracts a belief that contradicts the existing one
4. Result: both beliefs remain in the graph, linked by a `CONTRADICTS` edge
5. `list_beliefs(status="contradicted")` now returns both

The system does not automatically resolve contradictions — that requires human or agent judgment. The value is in *surfacing* the conflict rather than silently storing both versions.

## Belief Lifecycle

```
            ┌─────────────┐
    ingest → │   Active    │ ← reactivate
            │ (conf 0.8)  │
            └──────┬──────┘
                   │
        ┌──────────┼──────────┐
        v          v          v
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │Uncertain │ │Contradict│ │Superseded│
  │(conf 0.5)│ │   -ed    │ │(replaced)│
  └──────────┘ └──────────┘ └──────────┘
        │
        v
  ┌──────────────┐
  │ Consolidated │  (merged during dreaming)
  └──────────────┘
```

**Transitions:**

| From | To | Trigger |
|------|-----|---------|
| (new) | active | Ingestion with no contradiction found |
| active | contradicted | New belief conflicts with this one |
| active | superseded | New belief replaces this one (SUPERSEDES edge) |
| active | uncertain | Confidence drops below threshold or mixed evidence |
| uncertain | active | Additional evidence raises confidence |
| uncertain | consolidated | Graph dreaming merges episodic beliefs into abstractions |
| superseded | (terminal) | Old belief retained for audit trail but marked inactive |

## Integration Patterns

### Pattern 1: Pre-action belief check

Before taking action based on stored knowledge, query beliefs to check for contradictions:

```
Agent: "Before I recommend JWT for auth, let me check my beliefs."
→ list_beliefs(status="active")
→ search_knowledge("JWT authentication performance")
→ Finds contradicted belief about JWT server load
→ Adjusts recommendation based on current evidence
```

### Pattern 2: Periodic belief review

Schedule regular reviews of uncertain and contradicted beliefs:

```
Agent: "What beliefs need my attention?"
→ list_beliefs(status="uncertain")     # unresolved evidence
→ list_beliefs(status="contradicted")  # active conflicts
→ Investigate each, update confidence or resolve
```

### Pattern 3: Knowledge provenance audit

When a decision is questioned, trace back to the beliefs that informed it:

```
Agent: "Why did I recommend session tokens over JWT?"
→ traverse_graph("session tokens outperform JWT for SSR apps")
→ See: SUPERSEDES "JWT tokens are stateless and reduce server load"
→ See: SUPPORTS evidence from auth service migration
→ Full audit trail of how the recommendation evolved
```

## Confidence Calibration

Confidence scores are subjective but should be applied consistently:

| Score | Meaning | Example |
|-------|---------|---------|
| 0.1-0.3 | Speculation or hypothesis | "This might be caused by network latency" |
| 0.4-0.6 | Uncertain, mixed evidence | "Some benchmarks show improvement, others don't" |
| 0.7 | Inferred from context (default) | "Based on three similar cases, this pattern holds" |
| 0.8-0.9 | Stated fact with source | "The documentation confirms this behavior" |
| 1.0 | Axiomatic, definitional | "HTTP 200 means success" |

Beliefs at 0.3 or below may not surface in standard queries. Beliefs at 0.9+ should have verifiable sources.

## Graph Dreaming and Beliefs

Merkraum includes a neuroscience-inspired memory consolidation protocol ("dreaming") that operates on beliefs:

- **Replay**: Random walks through the graph surface unexpected connections between beliefs
- **Consolidation**: Multiple episodic beliefs about the same topic are merged into a single, higher-confidence abstract belief
- **Reflection**: Structural analysis identifies orphaned beliefs, over-centralized hubs, and schema violations

Consolidation is the most relevant to belief tracking. Example:

```
Before consolidation:
  Belief: "Project Alpha deadline slip was caused by unclear requirements" (conf 0.7)
  Belief: "Project Beta delay traced to requirements ambiguity" (conf 0.6)
  Belief: "Project Gamma timeline overrun from vague spec" (conf 0.8)

After consolidation:
  Belief: "Projects consistently slip when requirements are ambiguous" (conf 0.9, consolidated)
  ← SUPERSEDES all three original beliefs (retained as audit trail)
```

The agent's belief set evolves from scattered observations to coherent, high-confidence generalizations — without losing the original evidence.

## Comparison with Other Approaches

| Feature | Merkraum | Hindsight | Mem0 | Plain RAG |
|---------|----------|-----------|------|-----------|
| Contradiction detection | Automatic (LLM + graph edges) | Confidence decay (alpha-formula) | LLM-mediated (26% gain) | None |
| Confidence tracking | Per-belief (0.0-1.0) | Per-opinion (alpha-updated) | None | None |
| Belief lifecycle | 5 states + transitions | Active/decayed | None | None |
| Supersession | Explicit SUPERSEDES edge | Implicit (confidence overwrite) | Silent overwrite | Duplicate entries |
| Audit trail | Full provenance (cycle, source, timestamp) | Limited (PostgreSQL log) | Limited | None |
| Memory consolidation | Graph dreaming (replay + consolidation) | None | None | None |
| Schema discipline | Fixed (10 types, 16 relationships) | Open vocabulary | Open vocabulary (drift) | N/A |
| Graph traversal | Native Cypher (Neo4j) | PostgreSQL queries | Neo4j (open-vocab) | N/A |
| Regulatory readiness | EU AI Act Art. 13 compatible | No explicit compliance | No explicit compliance | None |

---

*See also: [Architecture](architecture.md) for the dual-store design, [README](../README.md) for quick start.*
