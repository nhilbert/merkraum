# Merkraum

<!-- mcp-name: io.github.nhilbert/merkraum -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io)
[![EU Hosted](https://img.shields.io/badge/hosting-EU%20(Frankfurt)-blue.svg)](#)

Auditable knowledge memory for AI agents. Know what your AI believes — and prove it.

## Why Merkraum

AI agents accumulate knowledge but can't explain what they know or why they changed their mind. When your compliance officer asks "what does your AI believe about customer X, and when did that change?" — most memory systems have no answer.

Merkraum gives your AI agent a structured, inspectable knowledge graph where every fact has provenance, every belief has a confidence score, and contradictions surface instead of hiding.

**What makes it different:**
- **Belief tracking with confidence** — not just storage, but epistemic reasoning. Beliefs have confidence scores, contradictions are explicit, supersession is tracked.
- **Full audit trail** — tamper-evident hash chain. Reconstruct entity state at any past point in time.
- **Managed forgetting** — knowledge expires via `valid_until`. Certainty decays over time. Stale beliefs surface for review.
- **PII Gateway** — scan ingested content for personally identifiable information (EN + DE). Block, warn, log, or off. No competitor offers this.
- **Graph dreaming** — periodic consolidation discovers patterns, resolves contradictions, and strengthens connections while your agent is idle.

## Quick Start — Hosted (Recommended)

Sign up at [app.merkraum.de](https://app.merkraum.de) and create a Personal Access Token (PAT) under Settings → Access Tokens.

**Claude Desktop / claude.ai** (`~/.claude/claude_desktop_config.json` or MCP settings):
```json
{
  "mcpServers": {
    "merkraum": {
      "url": "https://mcp.merkraum.de/mcp",
      "headers": {
        "Authorization": "Bearer mk_pat_<your_token>"
      }
    }
  }
}
```

**Claude Code** (`~/.claude/mcp_settings.json`):
```json
{
  "mcpServers": {
    "merkraum": {
      "url": "https://mcp.merkraum.de/mcp",
      "headers": {
        "Authorization": "Bearer mk_pat_<your_token>"
      }
    }
  }
}
```

**Cursor** (MCP settings):
```
URL: https://mcp.merkraum.de/mcp
Headers: Authorization: Bearer mk_pat_<your_token>
```

**OpenCode**:
```json
{
  "mcp": {
    "merkraum": {
      "type": "remote",
      "url": "https://mcp.merkraum.de/mcp",
      "headers": {
        "Authorization": "Bearer mk_pat_<your_token>"
      }
    }
  }
}
```

Start asking your agent to remember things.

## Quick Start — Self-Hosted

```bash
git clone https://github.com/nhilbert/merkraum.git
cd merkraum
cp .env.example .env
docker compose up -d
```

Verify the server is reachable:

```bash
curl -sS http://localhost:8090/health | jq .
```

Then configure your MCP client with `http://localhost:8090/mcp` as the URL (no auth needed in local mode).

## Core Capabilities

| Capability | Description |
|---|---|
| **Belief tracking** | Confidence scores, contradiction detection, belief supersession. Your AI doesn't silently overwrite knowledge — it reasons about change. |
| **Fixed schema** | 10 node types, 16 relationship types. Every operation is traceable and auditable. |
| **Contradiction detection** | Typed relationships (CONTRADICTS, SUPERSEDES, SUPPORTS) make epistemic conflicts explicit. |
| **Managed forgetting** | Nodes have optional `valid_until` dates. Certainty decay reduces confidence over time. Stale beliefs surface for review. |
| **Full audit trail** | Hash-chained operation log. Reconstruct any entity's state at any past point in time. Verify chain integrity. |
| **PII Gateway** | Configurable privacy filter (block/warn/log/off). Detects names, emails, phone numbers, IBANs, and more. English + German with auto-detection. |
| **Graph dreaming** | Periodic consolidation discovers patterns, weaves connections, and strengthens the knowledge graph during idle time. |
| **Hybrid search** | Semantic retrieval (Qdrant vectors) with deterministic text-search fallback for higher recall. |
| **EU-hosted** | AWS Frankfurt (eu-central-1). Data never leaves the EU. GDPR Art. 17 (right to deletion) built in. |
| **MCP compatible** | Works with Claude, Cursor, ChatGPT, OpenCode, and any MCP-compatible client. |

## MCP Tools

All tools enforce project-level ACL. PAT auth requires appropriate scopes (`read`, `search`, `write`, or `admin`).

### Memory Retrieval

| Tool | Scope | Description |
|------|-------|-------------|
| `search_knowledge` | `search` | Semantic search across your knowledge graph |
| `traverse_graph` | `read` | Walk relationships from any entity |
| `list_beliefs` | `read` | View beliefs by status (active, uncertain, contradicted, superseded) |
| `query_nodes` | `read` | List entities, optionally filtered by type |
| `get_graph_stats` | `read` | Node and edge counts by type |
| `get_usage` | `read` | Usage metrics and tier limits |

### Memory Writing

| Tool | Scope | Description |
|------|-------|-------------|
| `add_knowledge` | `write` | Add a structured entity (no LLM needed) |
| `add_relationship` | `write` | Link two entities with a typed relationship |
| `update_belief` | `write` | Update confidence, status, or summary of an existing belief |
| `ingest_knowledge` | `write` | Extract entities from free text via LLM (async) |
| `check_ingestion_status` | `read` | Poll async ingestion jobs |

### Belief Maintenance

| Tool | Scope | Description |
|------|-------|-------------|
| `consolidate_beliefs` | `write` | Resolve a contradiction between two beliefs |
| `expire_nodes` | `write` | Apply managed forgetting (mark nodes past `valid_until`) |
| `renew_node` | `write` | Extend or reset node validity windows |
| `certainty_decay` | `write` | Apply time-based confidence decay to beliefs |
| `certainty_review` | `read` | Surface beliefs needing review (confidence below threshold) |
| `certainty_stats` | `read` | Confidence distribution statistics |

### Audit Trail

| Tool | Scope | Description |
|------|-------|-------------|
| `get_history` | `read` | Retrieve the change history of an entity |
| `reconstruct_entity` | `read` | Reconstruct entity state at a past point in time |
| `verify_audit_chain` | `read` | Verify the hash-chain integrity of the audit log |

### PII Gateway

| Tool | Scope | Description |
|------|-------|-------------|
| `get_pii_config` | `read` | View PII Gateway settings (mode, language) |
| `set_pii_mode` | `write` | Configure PII detection (block/warn/log/off) |

### Maintenance

| Tool | Scope | Description |
|------|-------|-------------|
| `deduplicate_edges` | `write` | Remove duplicate relationships in the graph |
| `health_check` | `read` | Verify Neo4j and Qdrant connectivity |

## Schema

**Node types**: Person, Organization, Project, Concept, Regulation, Event, Belief, Artifact, Interview, Quote

**Relationship types**: SUPPORTS, CONTRADICTS, COMPLEMENTS, SUPERSEDES, EXTENDS, REFINES, CREATED_BY, AFFILIATED_WITH, APPLIES, IMPLEMENTS, PARTICIPATED_IN, PRODUCES, REFERENCES, TEMPORAL, MENTIONS, PART_OF

**Knowledge types**: fact (permanent), state (temporary), rule (policy), belief (subjective), memory (episodic)

## Configuration

Copy `.env.example` to `.env` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `MERKRAUM_BACKEND` | `neo4j_qdrant` | Backend type |
| `NEO4J_URI` | `bolt://neo4j:7687` | Neo4j connection |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `merkraum-local` | Neo4j password |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant connection |
| `MERKRAUM_PORT` | `8090` | MCP server HTTP port |
| `OPENAI_API_KEY` | (none) | Optional: enables LLM-powered text ingestion |
| `AUTH_REQUIRED` | `true` | JWT/PAT authentication (use `DEV_MODE=true` for local bypass) |
| `ALLOW_DEFAULT_PROJECT` | `false` | Prevent accidental cross-tenant data exposure |

## Architecture

```
Claude / Cursor / ChatGPT / OpenCode
        |
        | MCP (HTTP/SSE)
        v
┌──────────────────────┐
│   Merkraum Server    │
│  (Python + FastMCP)  │
│  + REST API (Flask)  │
│  + PII Gateway       │
└───────┬──────┬───────┘
        |      |
   ┌────┘      └────┐
   v                v
┌──────┐      ┌────────┐
│Neo4j │      │ Qdrant │
│(graph)│     │(vector)│
└──────┘      └────────┘
```

## Documentation

- [Architecture](docs/architecture.md) — dual-store design, schema, competitive landscape
- [Belief Tracking](docs/belief-tracking.md) — developer guide for epistemic belief tracking
- [Compliance Guide](docs/compliance-guide.md) — EU AI Act, PLD, GDPR compliance mapping
- [EpistBench](docs/benchmark.md) — epistemological memory benchmark results
- [Security Review](docs/security-review-2026-03.md) — detailed security assessment
- [Deployment](docs/deployment_setup.md) — production deployment guide

## System Requirements (Self-Hosted)

- Docker and Docker Compose (or Python 3.11+ with Neo4j and Qdrant installed separately)
- 4 GB RAM minimum
- Linux, macOS, or Windows with WSL2

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement.

If you use Merkraum in a regulated environment (healthcare, finance, legal), you are solely responsible for ensuring compliance with applicable laws and standards. See the [Compliance Guide](docs/compliance-guide.md) for EU AI Act, PLD, and GDPR mapping.

## License

MIT License. Free for personal and commercial use. Attribution required. See [LICENSE](LICENSE) for details.

---

Built by [Supervision Rheinland](https://merkraum.de) | Bonn, Germany
