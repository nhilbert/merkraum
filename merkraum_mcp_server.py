#!/usr/bin/env python3
"""
Merkraum MCP Server — Knowledge Graph for Claude, Cursor, and any MCP client.

Open-source standalone server using Neo4j CE + Qdrant (local Docker deployment).
No cloud services, no API keys needed for vector operations (uses FastEmbed).

Architecture:
    Claude Desktop / Cursor → stdio or HTTP → this server → Neo4j + Qdrant (local Docker)

Quick start:
    1. docker compose up -d
    2. pip install fastmcp neo4j qdrant-client fastembed
    3. python merkraum_mcp_server.py                    # stdio (Claude Desktop)
    4. python merkraum_mcp_server.py --transport http    # HTTP (Cursor, remote)

Claude Desktop config (~/.claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "merkraum": {
          "command": "python",
          "args": ["/path/to/merkraum_mcp_server.py"]
        }
      }
    }

v1.0 — Z1144 (2026-03-07). Extracted from vsg_mcp_server.py for open-source use.
"""

import os
import sys
import json
import time
import uuid
import queue
import logging
import threading
import asyncio
import argparse
from functools import partial
from typing import Optional

# Ensure project root is on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from fastmcp import FastMCP
from merkraum_backend import create_adapter, BackendAdapter, NODE_TYPES, RELATIONSHIP_TYPES, TIER_LIMITS

# --- Configuration ---

DEFAULT_BACKEND = os.environ.get("MERKRAUM_BACKEND", "neo4j_qdrant")
DEFAULT_PROJECT = os.environ.get("MERKRAUM_PROJECT", "default")
HTTP_PORT = int(os.environ.get("MERKRAUM_PORT", "8090"))
HTTP_HOST = os.environ.get("MERKRAUM_HOST", "127.0.0.1")

# LLM for knowledge extraction (optional — enables ingest_knowledge)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EXTRACTION_MODEL = os.environ.get("MERKRAUM_EXTRACTION_MODEL", "gpt-4o-mini")

# --- Logging ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("merkraum-mcp")

# --- Backend Singleton ---

_adapter: Optional[BackendAdapter] = None
_adapter_lock = threading.Lock()


def _get_adapter() -> BackendAdapter:
    """Lazy-initialize the backend adapter."""
    global _adapter
    if _adapter is not None:
        return _adapter
    with _adapter_lock:
        if _adapter is not None:
            return _adapter
        logger.info("Initializing backend: %s", DEFAULT_BACKEND)
        _adapter = create_adapter(DEFAULT_BACKEND)
        _adapter.connect()
        logger.info("Backend connected: %s", DEFAULT_BACKEND)
        return _adapter


# --- Async Helper ---

def _run_sync(func, *args, **kwargs):
    """Run a synchronous function in a thread executor."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, partial(func, *args, **kwargs))


# --- MCP Server ---

mcp = FastMCP("Merkraum Knowledge Graph")


# --- Tools ---

@mcp.tool()
async def search_knowledge(
    query: str,
    top_k: int = 5,
) -> dict:
    """Search your knowledge graph using semantic similarity.
    Returns ranked results with content, scores, and metadata.

    Args:
        query: What to search for (natural language)
        top_k: Number of results to return (1-20, default 5)
    """
    top_k = max(1, min(20, top_k))
    adapter = _get_adapter()
    start = time.time()
    try:
        results = await _run_sync(
            adapter.vector_search, query, top_k, DEFAULT_PROJECT
        )
        return {
            "results": results,
            "count": len(results),
            "query": query,
            "duration_ms": int((time.time() - start) * 1000),
        }
    except Exception as e:
        logger.error("search_knowledge failed: %s", e)
        return {"error": str(e)}


@mcp.tool()
async def traverse_graph(
    entity: str,
    depth: int = 2,
) -> dict:
    """Walk the knowledge graph from an entity, following relationships.
    Returns connected nodes and edges up to the specified depth.

    Args:
        entity: Entity name to start from (e.g. "Stafford Beer", "autopoiesis")
        depth: How many hops to traverse (1-4, default 2)
    """
    depth = max(1, min(4, depth))
    adapter = _get_adapter()
    start = time.time()
    try:
        result = await _run_sync(adapter.traverse, entity, DEFAULT_PROJECT, depth)
        result["duration_ms"] = int((time.time() - start) * 1000)
        return result
    except Exception as e:
        logger.error("traverse_graph failed: %s", e)
        return {"error": str(e)}


@mcp.tool()
async def list_beliefs(
    status: str = "active",
) -> dict:
    """List beliefs in the knowledge graph.
    Beliefs are atomic propositions with confidence scores.

    Args:
        status: Filter by status — active, uncertain, superseded, contradicted
    """
    if status not in ("active", "uncertain", "superseded", "contradicted"):
        status = "active"
    adapter = _get_adapter()
    start = time.time()
    try:
        beliefs = await _run_sync(adapter.get_beliefs, DEFAULT_PROJECT, status)
        return {
            "beliefs": beliefs,
            "count": len(beliefs),
            "status_filter": status,
            "duration_ms": int((time.time() - start) * 1000),
        }
    except Exception as e:
        logger.error("list_beliefs failed: %s", e)
        return {"error": str(e)}


@mcp.tool()
async def get_graph_stats() -> dict:
    """Get knowledge graph statistics: node counts by type, edge counts
    by relationship type, and totals."""
    adapter = _get_adapter()
    start = time.time()
    try:
        stats = await _run_sync(adapter.get_stats, DEFAULT_PROJECT)
        stats["duration_ms"] = int((time.time() - start) * 1000)
        return stats
    except Exception as e:
        logger.error("get_graph_stats failed: %s", e)
        return {"error": str(e)}


@mcp.tool()
async def query_nodes(
    node_type: str = None,
    limit: int = 50,
) -> dict:
    """Query entities in the knowledge graph, optionally filtered by type.

    Args:
        node_type: Filter by type (Person, Organization, Project, Concept,
            Regulation, Event, Belief, Artifact, Interview, Quote). None = all.
        limit: Max results (1-200, default 50)
    """
    if node_type and node_type not in NODE_TYPES:
        return {"error": f"Unknown node_type: {node_type}. Valid: {NODE_TYPES}"}
    limit = max(1, min(200, limit))
    adapter = _get_adapter()
    start = time.time()
    try:
        nodes = await _run_sync(adapter.query_nodes, node_type, DEFAULT_PROJECT, limit)
        return {
            "nodes": nodes,
            "count": len(nodes),
            "node_type_filter": node_type,
            "duration_ms": int((time.time() - start) * 1000),
        }
    except Exception as e:
        logger.error("query_nodes failed: %s", e)
        return {"error": str(e)}


@mcp.tool()
async def add_knowledge(
    name: str,
    summary: str,
    node_type: str = "Concept",
    confidence: float = 0.7,
) -> dict:
    """Add a single entity to the knowledge graph.
    For structured facts — use ingest_knowledge for free text.

    Args:
        name: Entity name (e.g. "Viable System Model", "Norman Hilbert")
        summary: Brief description
        node_type: One of Person, Organization, Project, Concept, Regulation,
            Event, Belief, Artifact, Interview, Quote
        confidence: For Beliefs only (0.0-1.0, default 0.7)
    """
    if node_type not in NODE_TYPES:
        return {"error": f"Unknown node_type: {node_type}. Valid: {NODE_TYPES}"}
    adapter = _get_adapter()
    entity = {"name": name, "summary": summary, "node_type": node_type}
    if node_type == "Belief":
        entity["confidence"] = max(0.0, min(1.0, confidence))
    start = time.time()
    try:
        written = await _run_sync(
            adapter.write_entities, [entity], "manual", "user", DEFAULT_PROJECT
        )
        # Also upsert to vector store for semantic search
        await _run_sync(
            adapter.vector_upsert,
            f"{DEFAULT_PROJECT}:{name}",
            f"{name}: {summary}",
            {"name": name, "node_type": node_type, "source": "manual"},
            DEFAULT_PROJECT,
        )
        return {
            "created": written > 0,
            "name": name,
            "node_type": node_type,
            "duration_ms": int((time.time() - start) * 1000),
        }
    except Exception as e:
        logger.error("add_knowledge failed: %s", e)
        return {"error": str(e)}


@mcp.tool()
async def add_relationship(
    entity_a: str,
    entity_b: str,
    relationship_type: str,
    reason: str = "",
) -> dict:
    """Link two entities with a typed relationship.
    Both entities must already exist in the graph.

    Args:
        entity_a: Source entity name
        entity_b: Target entity name
        relationship_type: One of SUPPORTS, CONTRADICTS, COMPLEMENTS, SUPERSEDES,
            EXTENDS, REFINES, CREATED_BY, AFFILIATED_WITH, APPLIES, IMPLEMENTS,
            PARTICIPATED_IN, PRODUCES, REFERENCES, TEMPORAL, MENTIONS, PART_OF
        reason: Why this relationship exists (optional)
    """
    if relationship_type not in RELATIONSHIP_TYPES:
        return {"error": f"Unknown type: {relationship_type}. Valid: {RELATIONSHIP_TYPES}"}
    adapter = _get_adapter()
    rel = {
        "source": entity_a,
        "target": entity_b,
        "type": relationship_type,
        "reason": reason[:200] if reason else "",
        "confidence": 0.9,
    }
    start = time.time()
    try:
        written = await _run_sync(
            adapter.write_relationships, [rel], "manual", "user", DEFAULT_PROJECT
        )
        return {
            "created": written > 0,
            "source": entity_a,
            "target": entity_b,
            "type": relationship_type,
            "duration_ms": int((time.time() - start) * 1000),
        }
    except Exception as e:
        logger.error("add_relationship failed: %s", e)
        return {"error": str(e)}


@mcp.tool()
async def update_belief(
    name: str,
    confidence: float = None,
    status: str = None,
    summary: str = None,
) -> dict:
    """Update an existing belief's confidence, status, or summary.
    This is how humans audit and correct the knowledge graph.

    Args:
        name: Belief name (must already exist)
        confidence: New confidence score (0.0-1.0). None = no change.
        status: New status — 'active' or 'superseded'. None = no change.
        summary: New summary text. None = no change.
    """
    adapter = _get_adapter()
    start = time.time()
    try:
        result = await _run_sync(
            adapter.update_belief, name, DEFAULT_PROJECT,
            confidence=confidence, status=status, summary=summary,
        )
        result["duration_ms"] = int((time.time() - start) * 1000)
        return result
    except Exception as e:
        logger.error("update_belief failed: %s", e)
        return {"error": str(e)}


@mcp.tool()
async def get_usage(
    tier: str = "free",
) -> dict:
    """Get knowledge graph usage metrics and tier limits.
    Shows current node/edge counts and how much of the tier quota is used.

    Args:
        tier: Pricing tier — free (100 nodes), pro (1000), team (5000),
            enterprise (50000). Default: free.
    """
    tier = tier.lower()
    if tier not in TIER_LIMITS:
        tier = "free"
    adapter = _get_adapter()
    start = time.time()
    try:
        usage = await _run_sync(adapter.get_usage, DEFAULT_PROJECT)
        node_limit = TIER_LIMITS[tier]
        usage_pct = round(usage["nodes"] / node_limit * 100, 1) if node_limit else 0
        return {
            "nodes": usage["nodes"],
            "edges": usage["edges"],
            "node_limit": node_limit,
            "usage_pct": usage_pct,
            "tier": tier,
            "duration_ms": int((time.time() - start) * 1000),
        }
    except Exception as e:
        logger.error("get_usage failed: %s", e)
        return {"error": str(e)}


# --- Async Ingestion (requires LLM API key) ---

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_job_queue: queue.Queue = None
MAX_JOBS = 50
MAX_INGEST_LENGTH = 10240  # 10KB


def _ensure_worker():
    global _job_queue
    if _job_queue is not None:
        return
    _job_queue = queue.Queue()

    def _worker():
        while True:
            job_id, text = _job_queue.get()
            _run_ingestion_job(job_id, text)
            _job_queue.task_done()

    t = threading.Thread(target=_worker, daemon=True, name="ingestion-worker")
    t.start()


def _run_ingestion_job(job_id: str, text: str):
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    start = time.time()
    try:
        result = _extract_and_write(text)
        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = result
            _jobs[job_id]["duration_ms"] = int((time.time() - start) * 1000)
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["duration_ms"] = int((time.time() - start) * 1000)


def _extract_and_write(text: str) -> dict:
    """Extract entities/relationships from text via LLM, write to graph."""
    import urllib.request
    import urllib.error

    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not set — LLM extraction unavailable. "
                "Use add_knowledge and add_relationship for manual entry."}

    prompt = f"""Extract structured knowledge from this text. Return JSON:
{{
  "entities": [
    {{"name": "...", "node_type": "...", "summary": "..."}}
  ],
  "relationships": [
    {{"source": "...", "target": "...", "type": "...", "reason": "..."}}
  ]
}}

Valid node_types: {json.dumps(NODE_TYPES)}
Valid relationship types: {json.dumps(RELATIONSHIP_TYPES)}

Text:
{text[:8000]}"""

    req_body = json.dumps({
        "model": EXTRACTION_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=req_body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    content = data["choices"][0]["message"]["content"]
    extracted = json.loads(content)

    adapter = _get_adapter()
    entities = extracted.get("entities", [])
    relationships = extracted.get("relationships", [])

    ent_count = adapter.write_entities(entities, "ingest", "extraction", DEFAULT_PROJECT)
    rel_count = adapter.write_relationships(relationships, "ingest", "extraction", DEFAULT_PROJECT)

    # Upsert entities to vector store
    for ent in entities:
        name = ent.get("name", "")
        summary = ent.get("summary", "")
        if name:
            adapter.vector_upsert(
                f"{DEFAULT_PROJECT}:{name}",
                f"{name}: {summary}",
                {"name": name, "node_type": ent.get("node_type", "Concept"),
                 "source": "extraction"},
                DEFAULT_PROJECT,
            )

    return {
        "entities_written": ent_count,
        "relationships_written": rel_count,
        "entities_extracted": len(entities),
        "relationships_extracted": len(relationships),
    }


@mcp.tool()
async def ingest_knowledge(
    text: str,
) -> dict:
    """Ingest free text into the knowledge graph using LLM extraction.
    Extracts entities and relationships automatically.
    Requires OPENAI_API_KEY environment variable.

    This is async — returns a job_id. Use check_ingestion_status to poll.
    For simple facts, prefer add_knowledge and add_relationship (no LLM needed).

    Args:
        text: Text to extract knowledge from (max 10KB)
    """
    if len(text) > MAX_INGEST_LENGTH:
        return {"error": f"Text too long ({len(text)} bytes). Max: {MAX_INGEST_LENGTH}"}
    if not text.strip():
        return {"error": "Empty text"}

    _ensure_worker()
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        # Evict old completed jobs
        if len(_jobs) >= MAX_JOBS:
            completed = [k for k, v in _jobs.items()
                         if v["status"] in ("completed", "failed")]
            for k in completed[:len(_jobs) - MAX_JOBS + 1]:
                del _jobs[k]
        _jobs[job_id] = {
            "status": "queued",
            "created": time.time(),
            "text_len": len(text),
            "result": None,
            "error": None,
        }
    _job_queue.put((job_id, text))
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Ingestion queued. Use check_ingestion_status to poll for results.",
    }


@mcp.tool()
async def check_ingestion_status(job_id: str) -> dict:
    """Check the status of an async ingestion job.

    Args:
        job_id: The job ID returned by ingest_knowledge
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return {"error": f"Unknown job_id: {job_id}"}
    response = {
        "job_id": job_id,
        "status": job["status"],
        "text_len": job["text_len"],
        "elapsed_s": int(time.time() - job["created"]),
    }
    if job["status"] == "completed":
        response["result"] = job["result"]
        response["duration_ms"] = job.get("duration_ms")
    elif job["status"] == "failed":
        response["error"] = job["error"]
        response["duration_ms"] = job.get("duration_ms")
    return response


@mcp.tool()
async def health_check() -> dict:
    """Check if the knowledge graph backends (Neo4j + Qdrant) are healthy."""
    adapter = _get_adapter()
    start = time.time()
    healthy = adapter.is_healthy()
    return {
        "healthy": healthy,
        "backend": DEFAULT_BACKEND,
        "project": DEFAULT_PROJECT,
        "extraction_available": bool(OPENAI_API_KEY),
        "duration_ms": int((time.time() - start) * 1000),
    }


# --- Main ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merkraum MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"],
                        default="stdio",
                        help="Transport mode (default: stdio for Claude Desktop)")
    parser.add_argument("--port", type=int, default=HTTP_PORT,
                        help=f"HTTP port (default: {HTTP_PORT})")
    parser.add_argument("--host", default=HTTP_HOST,
                        help=f"HTTP host (default: {HTTP_HOST})")
    args = parser.parse_args()

    tool_count = len(asyncio.run(mcp.list_tools()))
    logger.info("Merkraum MCP Server starting")
    logger.info("Backend: %s | Project: %s | Tools: %d",
                DEFAULT_BACKEND, DEFAULT_PROJECT, tool_count)
    logger.info("LLM extraction: %s",
                "available" if OPENAI_API_KEY else "unavailable (set OPENAI_API_KEY)")

    if args.transport == "http":
        logger.info("Transport: HTTP on %s:%d", args.host, args.port)
        mcp.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
            path="/mcp",
            stateless_http=True,
            json_response=True,
        )
    else:
        logger.info("Transport: stdio")
        mcp.run(transport="stdio")
