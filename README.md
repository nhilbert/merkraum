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

## Core capabilities

- **Belief tracking** — confidence scores, contradiction detection, belief supersession. Your AI doesn't silently overwrite knowledge — it reasons about change.
- **Fixed schema** — 10 node types, 16 relationship types. Every operation is traceable and auditable. No open-vocabulary chaos.
- **Contradiction detection** — typed relationships (CONTRADICTS, SUPERSEDES, SUPPORTS) make epistemic conflicts explicit, not implicit.
- **Hybrid search behavior** — semantic retrieval stays primary, with deterministic text-search fallback for higher recall when vector seeds are sparse.
- **Full audit trail** — every knowledge operation logged with who, what, when, and why. Built for environments where traceability matters.
- **EU-hosted** — AWS Frankfurt (eu-central-1). Data never leaves the EU. GDPR Art. 17 (right to deletion) built in. No CLOUD Act exposure.
- **Self-hosted option** — everything runs locally via Docker Compose. Own your knowledge infrastructure.
- **MCP compatible** — works with Claude, Cursor, ChatGPT, and any MCP-compatible client.
- **Graph dreaming** — periodic consolidation discovers patterns and strengthens connections while your agent is idle.

## Search and Vector Indexing (Update 2026-03-14)

- New entities are now vector-indexed during `write_entities`, so semantic search works immediately after ingest.
- Existing projects can be reindexed with:
  - `POST /api/projects/<project_id>/vectors/reindex`
  - optional body: `{ "limit": 5000 }` (max `10000`)
- Graph search supports explicit query modes via `/api/graph`:
  - `search_mode=semantic` (default)
  - `search_mode=text`
  - plus `q`, `hops`, `top` parameters for query-centered subgraphs.

## Quick start

```bash
git clone https://github.com/nhilbert/merkraum.git
cd merkraum
docker compose up -d
```

Then configure your MCP client:

**Claude Desktop** (`~/.claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "merkraum": {
      "url": "http://localhost:8090/mcp"
    }
  }
}
```

**Cursor** (MCP settings):
```
URL: http://localhost:8090/mcp
```

That's it. Start asking your agent to remember things.

## OpenCode setup

Use OpenCode as an MCP client for Merkraum (local or hosted).

1. Install OpenCode:

```bash
# npm (recommended)
npm install -g opencode-ai

# or Go
go install github.com/opencode-ai/opencode@latest
```

2. Create `opencode.json` in the directory where you start `opencode`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "merkraum": {
      "type": "remote",
      "url": "http://localhost:8090/mcp"
    }
  }
}
```

For hosted Merkraum, use your remote URL and token:

```json
{
  "$schema": "https://opencode.ai/config.json",
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

3. Verify connectivity:

```bash
opencode mcp list
```

If your MCP server is OAuth-enabled, you can also run:

```bash
opencode mcp auth merkraum
```

4. Start OpenCode and test:

```bash
opencode
```

Example prompts:
- `List all available tools`
- `Search for "Bundeskartellamt" in the knowledge graph`

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_knowledge` | Semantic search across your knowledge graph |
| `traverse_graph` | Walk relationships from any entity |
| `add_knowledge` | Add a structured entity (no LLM needed) |
| `add_relationship` | Link two entities with a typed relationship |
| `update_belief` | Update a belief's confidence, status, or summary — human-auditable correction |
| `get_usage` | Usage metrics and tier limits (nodes, edges, quota percentage) |
| `ingest_knowledge` | Extract entities from free text (requires OpenAI API key) |
| `check_ingestion_status` | Poll async ingestion jobs |
| `list_beliefs` | View beliefs by status (active, uncertain, contradicted) |
| `query_nodes` | List entities, optionally filtered by type |
| `get_graph_stats` | Node and edge counts by type |
| `health_check` | Verify Neo4j and Qdrant connectivity |

## Schema

**Node types**: Person, Organization, Project, Concept, Regulation, Event, Belief, Artifact, Interview, Quote

**Relationship types**: SUPPORTS, CONTRADICTS, COMPLEMENTS, SUPERSEDES, EXTENDS, REFINES, CREATED_BY, AFFILIATED_WITH, APPLIES, IMPLEMENTS, PARTICIPATED_IN, PRODUCES, REFERENCES, TEMPORAL, MENTIONS, PART_OF

## Configuration

Copy `.env.example` to `.env` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `MERKRAUM_BACKEND` | `neo4j_qdrant` | Backend type |
| `NEO4J_URI` | `bolt://neo4j:7687` | Neo4j connection (use service name in Docker) |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `merkraum-local` | Neo4j password |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant connection |
| `MERKRAUM_PORT` | `8090` | MCP server HTTP port |
| `OPENAI_API_KEY` | (none) | Optional: enables LLM-powered text ingestion |

### Security defaults (v1.1)

- `AUTH_REQUIRED` defaults to **enabled** when unset. Use `DEV_MODE=true` for local development bypass.
- `ALLOW_DEFAULT_PROJECT` defaults to **disabled** to reduce accidental cross-tenant data exposure.
- MCP OAuth config is fail-closed in non-dev mode: set `COGNITO_AWS_REGION`, `COGNITO_USER_POOL_ID`, `COGNITO_AUTH_DOMAIN`, `MCP_BASE_URL`, and `MCP_ALLOWED_CLIENT_IDS` (or `MCP_COGNITO_CLIENT_ID`).
- Dynamic OAuth client registration (`/register`) is disabled by default; explicitly set `MCP_ENABLE_DYNAMIC_CLIENT_REGISTRATION=true` only when protected by additional controls.

## System requirements

- Docker and Docker Compose
- 4 GB RAM minimum
- Any modern OS (Linux, macOS, Windows with WSL2)

## Architecture

```
Claude / Cursor / ChatGPT
        |
        | MCP (HTTP)
        v
┌──────────────────┐
│  Merkraum Server │
│  (Python + MCP)  │
└───────┬──────┬───┘
        |      |
   ┌────┘      └────┐
   v                v
┌──────┐      ┌────────┐
│Neo4j │      │ Qdrant │
│(graph)│     │(vector)│
└──────┘      └────────┘
```

## Running without Docker (development)

```bash
# Start Neo4j and Qdrant containers only
docker compose -f docker-compose.yml up neo4j qdrant -d

# Install Python dependencies
pip install -r requirements.txt

# Run MCP server directly
python merkraum_mcp_server.py --transport http

# Or use stdio transport for Claude Desktop
python merkraum_mcp_server.py
```

## Documentation

- [Architecture](docs/architecture.md) — dual-store design, schema, competitive landscape
- [Belief Tracking](docs/belief-tracking.md) — developer guide for epistemic belief tracking
- [Compliance Guide](docs/compliance-guide.md) — EU AI Act, PLD, GDPR compliance mapping
- [Public Security Readiness](docs/security-public-readiness.md) — checklist for safe public sharing (secrets/privacy)
- [EpistBench](docs/benchmark.md) — epistemological memory benchmark (baseline results)
- [Security Review (2026-03)](docs/security-review-2026-03.md) — detailed application security assessment and remediation roadmap

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement.

**Merkraum is not a production-grade service.** It is an experimental knowledge memory layer for AI agents. Use it at your own risk. The authors and contributors are not liable for any claim, damages, or other liability arising from the use of this software. See the [LICENSE](LICENSE) file for the full legal terms.

If you use Merkraum in a regulated environment (healthcare, finance, legal), you are solely responsible for ensuring compliance with applicable laws and standards. Merkraum does not provide legal, compliance, or security guarantees.

## License

MIT License. Free for personal and commercial use. Attribution required. See [LICENSE](LICENSE) for details.

---

Built by [Supervision Rheinland](https://merkraum.de) | Grounded in Stafford Beer's Viable System Model
