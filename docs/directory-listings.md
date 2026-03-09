# MCP Directory Listings — Submission Guide

*Prepared Z1241. GTM execution for developer discovery (SUP-118).*

## Submission Metadata (reuse across directories)

- **Name**: merkraum
- **Tagline**: Auditable knowledge memory for AI agents
- **Description**: Know what your AI believes — and prove it. Structured knowledge graph with belief tracking, contradiction detection, and full audit trail. 10 MCP tools for Claude, Cursor, and any MCP client. Self-hosted via Docker Compose (Neo4j + Qdrant). EU-hosted option available (Frankfurt). BSL 1.1.
- **Repository**: https://github.com/nhilbert/merkraum
- **Website**: https://merkraum.de
- **Category**: Database / Knowledge Management
- **License**: BSL 1.1
- **Contact**: info@merkraum.de
- **Tools count**: 10

## Directory Submissions

### 1. PulseMCP (highest priority — 8,594 servers, community hub)

**URL**: https://www.pulsemcp.com/submit
**Process**: Select "MCP Server" → paste GitHub URL → submit.
**Input needed**: `https://github.com/nhilbert/merkraum`
**Time**: ~30 seconds.
**Status**: READY — submit now.

### 2. mcpservers.org (free listing, broad reach)

**URL**: https://mcpservers.org/submit
**Fields**:
- Server Name: `merkraum`
- Short Description: `Auditable knowledge memory for AI agents. Belief tracking, contradiction detection, and full audit trail. 10 MCP tools. Self-hosted via Docker Compose.`
- Link: `https://github.com/nhilbert/merkraum`
- Category: `database`
- Contact Email: `info@merkraum.de`
**Tier**: Free (skip the $39 premium for now).
**Time**: ~1 minute.
**Status**: READY — submit now.

### 3. Glama (largest directory — 18,497 servers)

**URL**: https://glama.ai/mcp/servers → click "Add Server"
**Process**: Browser-based form. Details extracted from GitHub repo automatically.
**Input needed**: GitHub URL.
**Time**: ~1 minute.
**Status**: READY — submit now.

### 4. Official MCP Registry (auto-propagates to PulseMCP weekly)

**URL**: https://registry.modelcontextprotocol.io
**Process**: Requires `mcp-publisher` CLI + PyPI package.
**Steps**:
1. Publish merkraum to PyPI: `pip install hatch && hatch build && hatch publish` (needs PyPI API token)
2. Clone registry repo: `git clone https://github.com/modelcontextprotocol/registry`
3. Build publisher: `make publisher`
4. Login: `./bin/mcp-publisher login github`
5. Create server.json (see below)
6. Publish: `./bin/mcp-publisher publish`
**Naming**: `io.github.nhilbert/merkraum`
**Status**: BLOCKED — needs PyPI account + API token. Highest long-term value (canonical source, auto-synced to PulseMCP).

### 5. Smithery (developer-focused)

**URL**: https://smithery.ai/new
**Process**: CLI-based (`smithery mcp publish`) or web form.
**Status**: DEFERRED — lower priority, revisit after top 3.

## server.json (for Official MCP Registry)

```json
{
  "name": "io.github.nhilbert/merkraum",
  "description": "Auditable knowledge memory for AI agents. Belief tracking, contradiction detection, and full audit trail via MCP.",
  "repository": {
    "url": "https://github.com/nhilbert/merkraum"
  },
  "version": "1.0.0",
  "packages": [
    {
      "registryType": "docker",
      "identifier": "nhilbert/merkraum",
      "version": "latest",
      "transport": [
        {
          "type": "http",
          "port": 8090,
          "path": "/mcp"
        }
      ],
      "environmentVariables": [
        { "name": "NEO4J_URI", "description": "Neo4j bolt URI", "required": true },
        { "name": "NEO4J_USER", "description": "Neo4j username", "required": true },
        { "name": "NEO4J_PASSWORD", "description": "Neo4j password", "required": true },
        { "name": "QDRANT_URL", "description": "Qdrant REST URL", "required": true },
        { "name": "OPENAI_API_KEY", "description": "For LLM-powered text ingestion", "required": false }
      ]
    }
  ]
}
```

## Competitive context

Cognee is already listed on PulseMCP (7 tools, $7.5M seed). Hindsight (Vectorize) is growing fast with MIT license. Being absent from directories while competitors are listed = lost developer discovery. Every day without listings is a day developers find Cognee or Hindsight instead.

## Next steps after initial listings

1. Monitor listing approval times (PulseMCP ~1 week, others vary).
2. After PyPI publish: submit to Official MCP Registry for canonical listing.
3. Track referral traffic via Plausible (already installed on merkraum.de).
4. Push Docker image to Docker Hub (`nhilbert/merkraum`) for Official Registry docker transport.
