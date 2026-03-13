# Merkraum — Structured Knowledge Graph Memory for AI Agents

> Machine-addressable memory with belief tracking, contradiction detection, and provenance.

## What Merkraum Does

Merkraum provides persistent, structured memory for AI agents via a knowledge graph (Neo4j) and semantic/vector search (Qdrant or Pinecone, depending on deployment). Unlike flat key-value memory stores, Merkraum tracks **beliefs** — propositions with confidence scores, provenance, and explicit contradiction relationships.

### Core Capabilities

- **Semantic Search**: Vector-based retrieval across your knowledge graph
- **Hybrid Query Strategy**: Semantic-first retrieval with deterministic text-search fallback for recall-critical use cases
- **Knowledge Ingestion**: Extract entities and relationships from text via LLM pipeline
- **Graph Traversal**: Multi-hop exploration of entity relationships
- **Belief Tracking**: Store beliefs with confidence, status (active/uncertain/contradicted/superseded), and provenance
- **Contradiction Detection**: Automatic identification of conflicting beliefs
- **Multi-Project Isolation**: Separate knowledge spaces per project

### Supported Operations

| Operation | Method | Endpoint | Description |
|-----------|--------|----------|-------------|
| Search | GET | `/api/search?q=<query>` | Semantic vector search |
| Graph Search | GET | `/api/graph?q=<query>&search_mode=semantic|text` | Query-centered subgraph retrieval (supports `hops` and `top`) |
| Ingest (structured) | POST | `/api/ingest` | Write entities and relationships |
| Ingest (text) | POST | `/api/ingest/text` | LLM extraction + graph write |
| Traverse | GET | `/api/traverse/<entity>` | Multi-hop graph walk |
| Beliefs | GET | `/api/beliefs` | List beliefs by status |
| Stats | GET | `/api/stats` | Graph statistics |
| Graph | GET | `/api/graph` | Full graph for visualization |
| Nodes | GET | `/api/nodes` | Query nodes by type |
| Health | GET | `/api/health` | Service health check |
| Discover | GET | `/api/discover` | Machine-readable capabilities |
| Vector Reindex | POST | `/api/projects/<id>/vectors/reindex` | Rebuild vector index for existing project nodes |

## Vector Freshness

- New nodes are vector-indexed during entity writes (`write_entities`) to keep semantic retrieval up to date.
- Existing/legacy nodes can be reindexed per project via `POST /api/projects/<id>/vectors/reindex`.
- Reindex response includes: `upserted`, `failed`, `total_nodes`, `truncated`, and `limit`.

## Schema

### Node Types
Person, Organization, Project, Concept, Regulation, Event, Belief, Artifact, Interview, Quote

### Relationship Types
SUPPORTS, CONTRADICTS, COMPLEMENTS, SUPERSEDES, EXTENDS, REFINES, CREATED_BY, AFFILIATED_WITH, APPLIES, IMPLEMENTS, PARTICIPATED_IN, PRODUCES, REFERENCES, TEMPORAL, MENTIONS, PART_OF

## Authentication

All endpoints require a valid JWT token from AWS Cognito.

- **Header**: `Authorization: Bearer <jwt_token>`
- **Provider**: AWS Cognito (eu-central-1)
- **Flow**: OAuth 2.0 Authorization Code with PKCE

## Pricing Tiers

| Tier | Node Limit | Description |
|------|-----------|-------------|
| Free | 100 nodes | Evaluation and personal projects |
| Pro | 1,000 nodes | Individual professionals |
| Team | 5,000 nodes | Small teams |
| Enterprise | 50,000 nodes | Organizations |

## MCP Server

Merkraum also exposes tools via the Model Context Protocol (MCP) for direct integration with Claude, Cursor, and other MCP-compatible clients.

- **Endpoint**: `https://agent.nhilbert.de/mcp/merkraum/`
- **Auth**: OAuth 2.0 PKCE (Cognito)
- **Tools**: search, ingest, traverse, beliefs, stats

## Getting Started

1. **Discover**: `GET https://agent.nhilbert.de/api/merkraum/api/discover` — inspect available capabilities
2. **Authenticate**: Obtain a JWT token via Cognito OAuth flow
3. **Search**: `GET /api/search?q=your+query` — find existing knowledge
4. **Ingest**: `POST /api/ingest/text` — add knowledge from text
5. **Traverse**: `GET /api/traverse/entity_name` — explore graph connections

All API paths are relative to the base URL: `https://agent.nhilbert.de/api/merkraum`

## Safety Rules

- All data is project-isolated. Agents cannot access other projects without explicit grants.
- Node limits are enforced server-side per pricing tier.
- Authentication is mandatory in production.
- Rate limits apply to prevent abuse.

## Contact

- **Website**: https://merkraum.de
- **Operator**: Supervision Rheinland, Bonn (Dr. Norman Hilbert)
