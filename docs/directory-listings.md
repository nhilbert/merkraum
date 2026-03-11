# MCP Directory Listings — Submission Guide

*Prepared Z1241. Updated Z1348 (server.json validated, mcp-publisher installed, submission paths verified).*

## Submission Metadata (reuse across directories)

- **Name**: merkraum
- **Tagline**: Auditable knowledge memory for AI agents
- **Description**: Know what your AI believes — and prove it. Structured knowledge graph with belief tracking, contradiction detection, and full audit trail. 12 MCP tools for Claude, Cursor, and any MCP client. Self-hosted via Docker Compose (Neo4j + Qdrant). EU-hosted option available (Frankfurt). BSL 1.1.
- **Repository**: https://github.com/nhilbert/merkraum
- **Website**: https://merkraum.de
- **Category**: Database / Knowledge Management
- **License**: BSL 1.1
- **Contact**: info@merkraum.de
- **Tools count**: 12

## Directory Submissions

### 1. PulseMCP (highest priority — 8,610+ servers, community hub)

**URL**: https://www.pulsemcp.com/submit
**Process**: Select "MCP Server" → paste GitHub URL → submit.
**Input needed**: `https://github.com/nhilbert/merkraum`
**Time**: ~30 seconds.
**Status**: NOT LISTED (verified Z1348). READY — submit via web form.
**Note**: PulseMCP is a read-only aggregator — no API submission. Also auto-ingests from Official MCP Registry weekly.

### 2. mcpservers.org (free listing, broad reach)

**URL**: https://mcpservers.org/submit
**Fields**:
- Server Name: `merkraum`
- Short Description: `Auditable knowledge memory for AI agents. Belief tracking, contradiction detection, and full audit trail. 12 MCP tools. Self-hosted via Docker Compose.`
- Link: `https://github.com/nhilbert/merkraum`
- Category: `database`
- Contact Email: `info@merkraum.de`
**Tier**: Free (skip the $39 premium for now).
**Time**: ~1 minute.
**Status**: NOT LISTED (verified Z1348). READY — submit via web form.

### 3. Glama (largest directory — 18,967+ servers)

**URL**: https://glama.ai/mcp/servers → click "Add Server"
**Process**: Browser-based form. Details extracted from GitHub repo automatically.
**Input needed**: `https://github.com/nhilbert/merkraum`
**Time**: ~1 minute.
**Status**: NOT LISTED (verified Z1348). READY — submit via web form.

### 4. Official MCP Registry (canonical — auto-propagates to PulseMCP weekly)

**URL**: https://registry.modelcontextprotocol.io
**Tooling prepared (Z1348)**:
- `server.json` created in repo root — VALIDATED by mcp-publisher
- `mcp-publisher` CLI installed in user-local PATH (for example `~/.local/bin/mcp-publisher`)
- README has `<!-- mcp-name: io.github.nhilbert/merkraum -->` verification comment
- `pyproject.toml` has `name = "merkraum"` matching registry identifier

**Steps (maintainer action needed for PyPI)**:
1. Create PyPI account at https://pypi.org/account/register/ (if not existing)
2. Generate API token at https://pypi.org/manage/account/token/
3. Build + publish: `cd /path/to/merkraum && pip install hatch && hatch build && hatch publish` (enter token when prompted)
4. Login to registry: `mcp-publisher login github` (uses maintainer GitHub auth)
5. Publish to registry: `cd /path/to/merkraum && mcp-publisher publish`

**Status**: BLOCKED on PyPI account + API token. Highest long-term value (canonical source, auto-synced to PulseMCP).

### 5. Smithery (developer-focused)

**URL**: https://smithery.ai/new
**Process**: CLI-based (`smithery mcp publish`) or web form.
**Status**: DEFERRED — lower priority, revisit after top 3.

## Competitive context

Cognee is already listed on PulseMCP (7 tools, $7.5M seed). Hindsight (Vectorize) is growing fast with MIT license. Being absent from directories while competitors are listed = lost developer discovery. Every day without listings is a day developers find Cognee or Hindsight instead.

## Maintainer action summary (total ~3 minutes)

**Immediate (web forms, ~2 min total)**:
1. Open https://www.pulsemcp.com/submit → paste `https://github.com/nhilbert/merkraum` → submit (~30s)
2. Open https://mcpservers.org/submit → fill fields above → submit (~1 min)
3. Open https://glama.ai/mcp/servers → "Add Server" → paste GitHub URL → submit (~30s)

**When ready (PyPI, ~5 min)**:
4. Create PyPI account + API token → run `hatch build && hatch publish` → then `mcp-publisher publish`

## Next steps after initial listings

1. Monitor listing approval times (PulseMCP ~1 week, others vary).
2. After PyPI publish: submit to Official MCP Registry for canonical listing.
3. Track referral traffic via Plausible (already installed on merkraum.de).
4. Push Docker image to Docker Hub (`nhilbert/merkraum`) for Official Registry docker transport.
