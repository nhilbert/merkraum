# Merkraum

Personal knowledge memory for AI agents. A self-hosted knowledge graph with semantic search, belief tracking, and MCP integration.

## What it does

Merkraum gives your AI agent (Claude, ChatGPT, Cursor) a persistent, structured memory:

- **Knowledge graph** (Neo4j) — entities, relationships, typed connections
- **Semantic search** (Qdrant + FastEmbed) — find knowledge by meaning, not keywords
- **Belief tracking** — confidence scores, contradiction detection, supersession
- **Fixed schema** — 10 node types, 16 relationship types, auditable and explainable
- **MCP native** — works with any MCP-compatible client
- **No cloud required** — everything runs locally via Docker

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

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_knowledge` | Semantic search across your knowledge graph |
| `traverse_graph` | Walk relationships from any entity |
| `add_knowledge` | Add a structured entity (no LLM needed) |
| `add_relationship` | Link two entities with a typed relationship |
| `ingest_knowledge` | Extract entities from free text (requires OpenAI API key) |
| `check_ingestion_status` | Poll async ingestion jobs |
| `list_beliefs` | View beliefs by status (active, uncertain, contradicted) |
| `query_nodes` | List entities, optionally filtered by type |
| `get_graph_stats` | Node and edge counts by type |
| `health_check` | Verify Neo4j and Qdrant connectivity |

## Node types

Person, Organization, Project, Concept, Regulation, Event, Belief, Artifact, Interview, Quote

## Relationship types

SUPPORTS, CONTRADICTS, COMPLEMENTS, SUPERSEDES, EXTENDS, REFINES, CREATED_BY, AFFILIATED_WITH, APPLIES, IMPLEMENTS, PARTICIPATED_IN, PRODUCES, REFERENCES, TEMPORAL, MENTIONS, PART_OF

## Configuration

Copy `merkraum_env_local.template` to `.env` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `MERKRAUM_BACKEND` | `neo4j_qdrant` | Backend type |
| `NEO4J_URI` | `bolt://neo4j:7687` | Neo4j connection (use service name in Docker) |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `merkraum-local` | Neo4j password |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant connection |
| `MERKRAUM_PORT` | `8090` | MCP server HTTP port |
| `OPENAI_API_KEY` | (none) | Optional: enables LLM-powered text ingestion |

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
pip install -r requirements-merkraum.txt

# Run MCP server directly
python merkraum_mcp_server.py --transport http

# Or use stdio transport for Claude Desktop
python merkraum_mcp_server.py
```

## License

Business Source License 1.1 (BSL 1.1). Free for personal and non-commercial use. See LICENSE for details.

---

Built by [Supervision Rheinland](https://merkraum.de) | Powered by the Viable System Model
