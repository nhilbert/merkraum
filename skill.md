---
name: merkraum
version: 1.2.0
description: |
  Merkraum is a structured knowledge graph memory layer for AI agents.
  It provides persistent, queryable memory with belief tracking, contradiction
  detection, and provenance — accessible via MCP tools directly inside Claude Code.
  Use Merkraum when you need memory that persists across sessions, tracks confidence,
  or models relationships between entities.
license: BSL 1.1
metadata:
  author: Supervision Rheinland (Dr. Norman Hilbert)
  homepage: https://merkraum.de
  documentation: https://www.merkraum.de/docs
  mcp_endpoint: https://agent.nhilbert.de/mcp/merkraum/
  rest_base: https://agent.nhilbert.de/api/merkraum
  discovery: https://agent.nhilbert.de/api/merkraum/api/discover
  well_known: https://agent.nhilbert.de/.well-known/skill.md
  agent_spec: https://agent.nhilbert.de/api/merkraum/AGENT_SPEC.json
---

# Merkraum — Knowledge Graph Memory for Claude Code

Merkraum gives Claude Code agents persistent, structured memory. Unlike flat
key-value stores, Merkraum treats knowledge as propositions: entities with
typed relationships, beliefs with confidence scores, and explicit contradiction
tracking. Memory persists across sessions, decays with time when appropriate,
and is queryable by semantic similarity or graph structure.

## When to Use

Use Merkraum when:

- You need to remember facts, decisions, or context across Claude Code sessions
- You want to track beliefs with confidence and see when they contradict
- You need to model relationships between people, projects, concepts, or events
- You are building an agent that should accumulate knowledge over time
- You want to recall what you knew at a prior point in time (audit trail)

Do not use Merkraum for:
- Temporary session state (use local variables or CLAUDE.md instead)
- Large binary blobs or raw file storage
- Bulk operational logs or telemetry (only ingest knowledge worth remembering)

## MCP Setup (Claude Code)

Add to your MCP client configuration (`~/.claude/mcp_settings.json` or project-level):

```json
{
  "mcpServers": {
    "merkraum": {
      "url": "https://agent.nhilbert.de/mcp/merkraum/",
      "auth": {
        "type": "oauth2",
        "provider": "cognito",
        "domain": "merkraum.auth.eu-central-1.amazoncognito.com"
      }
    }
  }
}
```

Alternatively, use a Personal Access Token (PAT) for non-interactive environments:

```json
{
  "mcpServers": {
    "merkraum": {
      "url": "https://agent.nhilbert.de/mcp/merkraum/",
      "headers": {
        "Authorization": "Bearer mk_pat_<your_token>"
      }
    }
  }
}
```

Create a PAT at https://merkraum.de → Settings → Access Tokens. Scopes: `read`, `search`, `ingest`, `projects`.

## Core Concepts

### Projects
Each user has a default project (auto-provisioned on first connection, keyed to
your Cognito identity). You can create additional projects for topic isolation.
All knowledge is scoped to a project — agents cannot read across project
boundaries without explicit grants.

### Node Types
`Person`, `Organization`, `Project`, `Concept`, `Regulation`, `Event`,
`Belief`, `Artifact`, `Interview`, `Quote`

### Relationship Types
`SUPPORTS`, `CONTRADICTS`, `COMPLEMENTS`, `SUPERSEDES`, `EXTENDS`, `REFINES`,
`CREATED_BY`, `AFFILIATED_WITH`, `APPLIES`, `IMPLEMENTS`, `PARTICIPATED_IN`,
`PRODUCES`, `REFERENCES`, `TEMPORAL`, `MENTIONS`, `PART_OF`

### Beliefs
A `Belief` node is a proposition with:
- `confidence` (0.0–1.0) — how certain the agent is
- `status` — `active`, `uncertain`, `contradicted`, or `superseded`
- `provenance` — source label for traceability
- `knowledge_type` — `fact` (permanent), `state` (temporary), `rule` (policy),
  `belief` (subjective), `memory` (episodic)
- `valid_until` — optional expiry date for temporal knowledge

## MCP Tool Reference

All tools use your default project unless `project` is specified.

### Memory Retrieval

**`search_knowledge`** — Semantic search across the knowledge graph.
```
query: str           # natural language query
top_k: int = 5       # results to return (max 20)
project: str = None  # project ID (default: personal space)
```
Use first — retrieve before producing. Score ≥ 0.35 indicates a strong match.

**`traverse_graph`** — Walk graph from an entity, following relationships.
```
entity: str          # entity name to start from
depth: int = 2       # hops to traverse (max 4)
project: str = None
```
Use after search to explore the neighborhood of a matched entity.

**`list_beliefs`** — List beliefs filtered by status.
```
status: str = "active"   # active | uncertain | contradicted | superseded
knowledge_type: str = None  # fact | state | rule | belief | memory
project: str = None
```
Use to surface active knowledge or review contradictions before acting.

**`query_nodes`** — Query entities by type.
```
node_type: str = None   # Person | Concept | Event | ... (None = all)
limit: int = 50         # max 200
project: str = None
```
Use to enumerate all entities of a given type.

### Memory Writing

**`add_knowledge`** — Add a single entity to the graph.
```
name: str             # unique entity name
summary: str          # what this entity is or means
node_type: str = "Concept"   # see node types above
confidence: float = 0.7      # for Belief nodes only
project: str = None
```
Use for atomic facts, entities, or beliefs worth remembering.
If the project's PII Gateway is active (mode ≠ off), entities are scanned for
personally identifiable information. In block mode, a `pii_detected` error is
returned. In warn mode, `pii_findings` are included in the response.

**`add_relationship`** — Link two entities with a typed relationship.
```
entity_a: str
entity_b: str
relationship_type: str   # see relationship types above
reason: str = None       # why this relationship holds
project: str = None
```
Use after add_knowledge to model how entities relate.

**`update_belief`** — Update confidence, status, or summary of an existing belief.
```
name: str
confidence: float = None   # new confidence (0.0–1.0)
status: str = None         # active | superseded
summary: str = None        # new summary text
project: str = None
```
Use when evidence changes your assessment of a proposition.

**`ingest_knowledge`** — Free-text ingestion via LLM extraction (async).
```
text: str       # raw text to extract knowledge from (max 10,240 chars)
project: str = None
```
Returns a `job_id`. Use `check_ingestion_status` to poll for completion.
Use for longer texts where structured entity extraction is needed.
PII Gateway applies to extracted entities — if mode is "block" and PII is
found, the job fails with a `pii_detected` error in the job result.

**`check_ingestion_status`** — Poll an async ingestion job.
```
job_id: str     # returned by ingest_knowledge
```

### Belief Maintenance

**`consolidate_beliefs`** — Resolve a contradiction between two beliefs.
```
belief_a: str    # name of first belief
belief_b: str    # name of second belief
resolution: str  # free-text explanation of how to resolve
project: str = None
```
Use to close contradictions surfaced by `list_beliefs(status="contradicted")`.

**`expire_nodes`** — Apply managed forgetting: mark nodes past `valid_until`.
```
project: str = None
dry_run: bool = False   # preview without mutating
```
Use periodically to keep knowledge fresh. Respects `valid_until` dates.

**`renew_node`** — Extend the validity of a node that would otherwise expire.
```
name: str
days: int = 90    # extend by N days from today
project: str = None
```

### Certainty Governance

**`certainty_decay`** — Apply time-based confidence decay to active beliefs.
```
project: str = None
dry_run: bool = False
```
Use in periodic maintenance cycles to reduce confidence of aging beliefs.

**`certainty_review`** — Surface beliefs whose confidence has decayed below threshold.
```
threshold: float = 0.4
project: str = None
```
Use to find beliefs that need human review or reconfirmation.

**`certainty_stats`** — Confidence distribution statistics.
```
project: str = None
```

### Audit Trail

**`get_history`** — Retrieve the change history of an entity.
```
entity: str
limit: int = 20
project: str = None
```

**`reconstruct_entity`** — Reconstruct entity state at a past point in time.
```
entity: str
timestamp: str    # ISO 8601 datetime
project: str = None
```

**`verify_audit_chain`** — Verify the hash-chain integrity of the audit log.
```
project: str = None
limit: int = 100
```

### PII Gateway

**`get_pii_config`** — Get the current PII Gateway configuration for a project.
```
project: str = None
```
Returns the active PII mode, language, and descriptions of available modes.
Use to check what PII protection is active before ingesting sensitive data.

**`set_pii_mode`** — Configure PII detection mode and language for a project.
```
pii_mode: str        # "block" | "warn" | "log" | "off"
pii_language: str = None   # "auto" | "en" | "de" (default: unchanged)
project: str = None
```
Controls how PII is handled during `add_knowledge` and `ingest_knowledge`:
- **block**: Entities containing PII are rejected (error returned)
- **warn**: Entities are stored but PII findings are returned in the response
- **log**: Entities are stored, PII findings logged silently to the audit trail
- **off**: No PII detection

Detected PII types (GDPR-critical): person names, email addresses, phone numbers,
IBAN codes, credit cards, IP addresses, locations, dates of birth, medical license
numbers, and German tax IDs.

Language detection is automatic by default. Set explicitly for projects with
known single-language content.

### Maintenance

**`deduplicate_edges`** — Remove duplicate relationships in the graph.
```
project: str = None
dry_run: bool = False
```

**`get_graph_stats`** — Graph statistics: node counts by type, edge totals.
```
project: str = None
```

**`get_usage`** — Current usage vs. tier limits.
```
project: str = None
```

**`health_check`** — Verify backends (Neo4j, Qdrant) are healthy.

## Recommended Workflows

### Session Start: Retrieve Before Producing
```
1. search_knowledge("<current topic>", top_k=5)
2. If score ≥ 0.35: traverse_graph("<matched entity>", depth=2)
3. list_beliefs(status="active") — review active beliefs on topic
4. list_beliefs(status="contradicted") — check for open contradictions
5. Now produce output informed by existing memory
```

### Recording a Decision or Finding
```
1. add_knowledge(name="<decision>", summary="<what was decided and why>",
                 node_type="Belief", confidence=0.85)
2. add_relationship(entity_a="<decision>", entity_b="<context entity>",
                    relationship_type="REFERENCES", reason="<why linked>")
3. If this supersedes a prior belief:
   update_belief(name="<old belief>", status="superseded")
   add_relationship(entity_a="<new>", entity_b="<old>",
                    relationship_type="SUPERSEDES")
```

### Ingesting a Document or Long Text
```
1. ingest_knowledge(text="<content>") → returns job_id
2. check_ingestion_status(job_id=<id>) — poll until status="completed"
3. search_knowledge("<topic>") to verify extraction
```

### Setting Up PII Protection
```
1. get_pii_config() — check current PII mode
2. set_pii_mode(pii_mode="warn") — enable PII detection in warn mode
3. add_knowledge(...) — PII findings returned in response if detected
4. Review findings, then optionally: set_pii_mode(pii_mode="block")
```

### Periodic Maintenance
```
1. certainty_decay() — reduce confidence of aging beliefs
2. certainty_review(threshold=0.4) — surface beliefs needing attention
3. expire_nodes(dry_run=True) — preview what would be forgotten
4. expire_nodes() — apply managed forgetting
5. deduplicate_edges() — clean up graph
```

## What Not to Ingest

Following the principle that memory is for knowledge worth keeping (not operational telemetry):

- **YES**: Decisions, new findings, beliefs, entity relationships, contradictions resolved, novel insights
- **NO**: Routine status updates, timer values, cycle counters, acknowledgements, raw logs, temporary context

## Safety

- All data is project-isolated. Agents cannot access other projects without grants.
- Authentication is mandatory in production (OAuth 2.0 PKCE or PAT).
- Node limits are enforced server-side per pricing tier.
- All data stored in EU (eu-central-1 Frankfurt). No data leaves EU.
- Every tool call is audit-logged with timestamp, user ID, and parameters.
- **PII Gateway**: Configurable per project — scans entities at ingestion time for
  personally identifiable information (GDPR Art. 5 data minimization). Supports
  English and German with automatic language detection. See `get_pii_config` and
  `set_pii_mode` tools.

## Pricing Tiers

| Tier       | Node Limit | Price         |
|------------|-----------|---------------|
| Free       | 100       | EUR 0/month   |
| Pro        | 1,000     | EUR 19/month  |
| Team       | 5,000     | EUR 49/month  |
| Enterprise | 50,000    | Custom        |

Free tier is sufficient for personal agent memory experiments. Check your
current usage with `get_usage()`.

## Discovery

An MCP-compatible agent can discover Merkraum's full capabilities at runtime:

```
GET https://agent.nhilbert.de/api/merkraum/api/discover
```

This returns the live capability manifest including available tools, schema,
authentication requirements, and pricing tiers — without requiring authentication.

## Contact

- Website: https://merkraum.de
- Operator: Supervision Rheinland, Bonn (Dr. Norman Hilbert)
- Support: info@merkraum.de
