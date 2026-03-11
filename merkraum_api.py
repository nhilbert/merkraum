#!/usr/bin/env python3
"""
Merkraum REST API Server — HTTP interface for the knowledge graph.

Wraps BackendAdapter from merkraum_backend.py and exposes endpoints
consumed by the React frontend (merkraum-front).

Includes Cognito JWT authentication for all endpoints.

Authentication:
    - AUTH_REQUIRED env var controls if tokens are validated (default: false for local dev)
    - When AUTH_REQUIRED=true: All requests must have valid Authorization: Bearer header
    - When AUTH_REQUIRED=false or unset: Endpoints work without authentication
    - For production deployments, always set AUTH_REQUIRED=true

v1.0 — SUP-94
v1.1 — SUP-95 (2026-03-11): Added Cognito JWT validation
v1.2 — SUP-96 (2026-03-11): Made authentication configurable via AUTH_REQUIRED env var
"""

import argparse
import importlib
import json
import logging
import os
import sys
import urllib.request
import urllib.error
from functools import lru_cache
from typing import cast

from flask import Flask, jsonify, request

from merkraum_backend import (
    create_adapter, NODE_TYPES, RELATIONSHIP_TYPES, Neo4jBaseAdapter,
    NodeLimitExceeded, TIER_LIMITS,
)
from jwt_auth import get_cognito_validator, require_auth, optional_auth

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global adapter instance — created once at startup.
adapter: Neo4jBaseAdapter | None = None


# ---------------------------------------------------------------------------
# CORS — applied to every response
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS = {
    "https://app.merkraum.de",
    "http://localhost:3000",
    "http://localhost:5173",
}

MAX_GRAPH_LIMIT = 1000
MAX_NODES_LIMIT = 500
MAX_SEARCH_TOP = 50
MAX_TRAVERSE_DEPTH = 5


def _allowed_origins() -> set[str]:
    configured = os.environ.get("CORS_ALLOWED_ORIGINS")
    if not configured:
        return ALLOWED_ORIGINS
    return {x.strip() for x in configured.split(",") if x.strip()}


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "")
    allowed_origins = _allowed_origins()
    if origin and origin in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Methods"] = (
        "GET, POST, PATCH, OPTIONS, PUT, DELETE"
    )
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Authorization, X-Requested-With"
    )
    response.headers["Access-Control-Max-Age"] = "3600"
    return response


@app.route("/api/<path:path>", methods=["OPTIONS"])
@app.route("/api", methods=["OPTIONS"])
def handle_preflight(path=""):
    """Handle CORS preflight requests. No authentication required for OPTIONS."""
    origin = request.headers.get("Origin", "")
    if origin and origin not in _allowed_origins():
        return _error("CORS origin not allowed", 403)
    return jsonify({}), 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_id() -> str:
    """Extract project_id from query params, defaulting to 'default'."""
    return request.args.get("project", "default") or "default"


def _is_auth_required() -> bool:
    return os.environ.get("AUTH_REQUIRED", "false").lower() in ("true", "1", "yes")


def _is_production_env() -> bool:
    env = (os.environ.get("APP_ENV") or os.environ.get("FLASK_ENV") or "").strip().lower()
    return env in {"prod", "production"}


def _split_csv_env(name: str) -> set[str]:
    raw = os.environ.get(name, "")
    return {x.strip() for x in raw.split(",") if x.strip()}


@lru_cache(maxsize=1)
def _project_group_acl() -> dict[str, set[str]]:
    raw = os.environ.get("PROJECT_GROUP_ACL_JSON", "{}")
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}
        out: dict[str, set[str]] = {}
        for project, groups in parsed.items():
            if isinstance(project, str) and isinstance(groups, list):
                out[project] = {str(g).strip() for g in groups if str(g).strip()}
        return out
    except Exception:
        logger.warning("Invalid PROJECT_GROUP_ACL_JSON; ignoring")
        return {}


@lru_cache(maxsize=1)
def _project_user_acl() -> dict[str, set[str]]:
    raw = os.environ.get("PROJECT_USER_ACL_JSON", "{}")
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}
        out: dict[str, set[str]] = {}
        for project, users in parsed.items():
            if isinstance(project, str) and isinstance(users, list):
                out[project] = {str(u).strip() for u in users if str(u).strip()}
        return out
    except Exception:
        logger.warning("Invalid PROJECT_USER_ACL_JSON; ignoring")
        return {}


def _is_project_allowed(project: str) -> bool:
    if not _is_auth_required():
        return True

    user_id = getattr(request, "user_id", None)
    groups = set(getattr(request, "groups", []) or [])
    if not user_id:
        return False

    if project == "default" and os.environ.get("ALLOW_DEFAULT_PROJECT", "true").lower() in ("true", "1", "yes"):
        return True

    admin_groups = _split_csv_env("ADMIN_GROUPS")
    if admin_groups and groups.intersection(admin_groups):
        return True

    if project == user_id or project.startswith(f"{user_id}:"):
        return True

    if user_id in _project_user_acl().get(project, set()):
        return True

    if groups.intersection(_project_group_acl().get(project, set())):
        return True

    return False


def _deny_if_project_forbidden(project: str):
    if _is_project_allowed(project):
        return None
    return _error(f"Forbidden project access: '{project}'", 403)


def _actor() -> str:
    return (
        getattr(request, "username", None)
        or getattr(request, "user_id", None)
        or "api"
    )


def _error(message: str, status: int = 500):
    return jsonify({"error": message}), status


def _get_all_edges(adp, project_id: str, limit: int = 1000) -> list:
    """Query all relationships for a project directly via Neo4j driver."""
    edges = []
    with adp._driver.session() as session:
        records = session.run(
            """
            MATCH (a {project_id: $pid})-[r]->(b {project_id: $pid})
            WHERE any(lbl IN labels(a) WHERE lbl IN $node_types)
              AND any(lbl IN labels(b) WHERE lbl IN $node_types)
            RETURN a.name AS source_name, b.name AS target_name,
                   labels(a)[0] AS source_type, labels(b)[0] AS target_type,
                   a.node_id AS source_node_id, b.node_id AS target_node_id,
                   type(r) AS type, r.confidence AS confidence, r.reason AS reason
            LIMIT $limit
            """,
            pid=project_id,
            limit=limit,
            node_types=list(NODE_TYPES),
        )
        for rec in records:
            edges.append(
                {
                    "source_name": rec["source_name"],
                    "target_name": rec["target_name"],
                    "source_type": rec["source_type"],
                    "target_type": rec["target_type"],
                    "source_node_id": rec["source_node_id"],
                    "target_node_id": rec["target_node_id"],
                    "type": rec["type"],
                    "confidence": rec["confidence"] or 0,
                    "reason": rec["reason"] or "",
                }
            )
    return edges


def _map_stats(raw: dict) -> dict:
    """Map adapter stats dict to frontend format.

    Adapter returns:
        {nodes: {NodeType: count}, edges: {RelType: count},
         total_nodes, total_edges}

    Frontend expects:
        {entities, relationships, beliefs, contradictions, nodes_by_type}
    """
    nodes_by_type = raw.get("nodes", {})
    return {
        "entities": raw.get("total_nodes", 0),
        "relationships": raw.get("total_edges", 0),
        "beliefs": nodes_by_type.get("Belief", 0),
        "contradictions": raw.get("edges", {}).get("CONTRADICTS", 0),
        "nodes_by_type": nodes_by_type,
    }


def _map_belief(b: dict) -> dict:
    """Map adapter belief dict to frontend format."""
    return {
        "name": b.get("name", ""),
        "summary": b.get("summary", ""),
        "confidence": b.get("confidence", 0),
        "status": b.get("status", "active"),
        "source": b.get("cycle", b.get("source_cycle", "")),
    }


def _map_node_for_graph(n: dict) -> dict:
    """Map a query_nodes result to the frontend graph node format."""
    node_type = n.get("type", "Concept")
    node_name = n.get("name") or ""
    node_id = n.get("node_id") or f"{node_type}:{node_name}"
    return {
        "id": node_id,
        "node_id": node_id,
        "name": node_name,
        "type": node_type,
        "summary": n.get("summary", ""),
        "confidence": n.get("confidence"),
        # val drives node size in react-force-graph; beliefs slightly larger
        "val": 2 if node_type == "Belief" else 1,
    }


def _map_edge_for_graph(e: dict) -> dict:
    """Map an edge dict to the frontend graph link format."""
    source_id = e.get("source_node_id") or f"{e.get('source_type', 'Concept')}:{e.get('source_name', '')}"
    target_id = e.get("target_node_id") or f"{e.get('target_type', 'Concept')}:{e.get('target_name', '')}"
    return {
        "source": source_id,
        "target": target_id,
        "source_name": e.get("source_name", ""),
        "target_name": e.get("target_name", ""),
        "source_type": e.get("source_type", ""),
        "target_type": e.get("target_type", ""),
        "type": e.get("type", ""),
        "reason": e.get("reason", ""),
    }


# ---------------------------------------------------------------------------
# LLM-based text extraction
# ---------------------------------------------------------------------------

_NODE_TYPES_LIST = ", ".join(NODE_TYPES)
_REL_TYPES_LIST = ", ".join(RELATIONSHIP_TYPES)

EXTRACTION_SYSTEM_PROMPT = f"""You are a knowledge extraction engine for Merkraum, a knowledge graph tool.

Your task: Extract structured entities and relationships from the given text.
Use ONLY the following fixed vocabulary.

## NODE TYPES (use exactly these labels):
{chr(10).join(f"- {t}" for t in NODE_TYPES)}

## RELATIONSHIP TYPES (use exactly these labels):
{chr(10).join(f"- {t}" for t in RELATIONSHIP_TYPES)}

## RULES:
1. Each entity must be ATOMIC: one concept per node. Do NOT merge multiple concepts.
2. Each Belief must be a SINGLE falsifiable proposition, max 200 characters.
3. Set confidence for Beliefs: 0.9 = directly stated as fact, 0.7 = inferred, 0.5 = speculative.
4. Prefer canonical entity names. Use full names for people.
5. Extract AT MOST 15 entities and 20 relationships per text passage.
6. If the text contains no extractable knowledge, return empty arrays.
7. Do NOT invent relationships not stated or clearly implied in the text.

## OUTPUT FORMAT:
Return a JSON object with two arrays: "entities" and "relationships".

{{
  "entities": [
    {{
      "name": "canonical name",
      "node_type": "one of the node types above",
      "summary": "one-paragraph description, max 500 chars",
      "confidence": 0.9
    }}
  ],
  "relationships": [
    {{
      "source": "entity name",
      "target": "entity name",
      "type": "one of the relationship types above",
      "confidence": 0.8,
      "reason": "brief explanation, max 200 chars"
    }}
  ]
}}

Return ONLY valid JSON. No explanation, no markdown fences."""


def _get_openai_key() -> str | None:
    """Get OpenAI API key from environment."""
    return os.environ.get("OPENAI_API_KEY") or _load_env_value("OPENAI_API_KEY")


def _load_env_value(key: str) -> str | None:
    """Load a single value from .env file."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return None
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _llm_extract(text: str, api_key: str, model: str = "gpt-4o-mini") -> dict:
    """Call OpenAI to extract entities and relationships from text."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
        "max_completion_tokens": 8000,
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract entities and relationships from the following text:\n\n{text[:8000]}"},
        ],
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
        content = data["choices"][0]["message"].get("content", "")
        if not content:
            return {"entities": [], "relationships": []}
        result = json.loads(content)
        # Validate structure
        entities = result.get("entities", [])
        relationships = result.get("relationships", [])
        # Filter to valid types
        valid_node_types = set(NODE_TYPES)
        valid_rel_types = set(RELATIONSHIP_TYPES)
        entities = [e for e in entities if e.get("node_type") in valid_node_types]
        relationships = [r for r in relationships if r.get("type") in valid_rel_types]
        return {"entities": entities, "relationships": relationships}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
@require_auth
def health():
    """Health check — returns adapter connectivity status.

    Requires: Authorization: Bearer <cognito_jwt_token> (when AUTH_REQUIRED=true)
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    try:
        healthy = adapter.is_healthy()
        status = "ok" if healthy else "degraded"
        return jsonify({"status": status, "adapter": type(adapter).__name__}), (
            200 if healthy else 503
        )
    except Exception as exc:
        logger.exception("Health check failed")
        return _error(str(exc), 503)


@app.route("/api/projects", methods=["GET"])
@require_auth
def projects():
    """List all project IDs that have data in the graph.

    Requires: Authorization: Bearer <cognito_jwt_token> (when AUTH_REQUIRED=true)
    Returns a sorted list of project_id strings.
    """
    adp = adapter
    if adp is None:
        return _error("Adapter not initialized", 503)
    try:
        with adp._driver.session() as session:
            records = session.run(
                "MATCH (n) WHERE n.project_id IS NOT NULL "
                "RETURN DISTINCT n.project_id AS pid ORDER BY pid"
            )
            project_ids = [rec["pid"] for rec in records]
            if _is_auth_required():
                project_ids = [pid for pid in project_ids if _is_project_allowed(pid)]
        return jsonify(project_ids)
    except Exception as exc:
        logger.exception("projects listing failed")
        return _error(str(exc))


@app.route("/api/stats", methods=["GET"])
@require_auth
def stats():
    """Graph stats in frontend format: {entities, relationships, beliefs, contradictions}.

    Requires: Authorization: Bearer <cognito_jwt_token> (when AUTH_REQUIRED=true)
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    project = _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    try:
        raw = adapter.get_stats(project_id=project)
        return jsonify(_map_stats(raw))
    except Exception as exc:
        logger.exception("stats failed for project=%s", project)
        return _error(str(exc))


@app.route("/api/usage", methods=["GET"])
@require_auth
def usage():
    """Usage metrics for a project — node/edge counts and tier limits.

    Query params:
        project: project id (default: "default")
        tier: pricing tier — free, pro, team, enterprise (default: "free")

    Returns:
        {nodes, edges, node_limit, usage_pct, tier}
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    project = _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    tier = request.args.get("tier", "free") or "free"
    if tier not in TIER_LIMITS:
        return _error(
            f"Invalid tier '{tier}'. Must be one of: {', '.join(sorted(TIER_LIMITS))}",
            400,
        )
    try:
        raw = adapter.get_usage(project_id=project)
        node_limit = TIER_LIMITS[tier]
        nodes = raw["nodes"]
        return jsonify({
            "nodes": nodes,
            "edges": raw["edges"],
            "node_limit": node_limit,
            "usage_pct": round(nodes / node_limit * 100, 1) if node_limit else 0,
            "tier": tier,
        })
    except Exception as exc:
        logger.exception("usage failed for project=%s", project)
        return _error(str(exc))


@app.route("/api/beliefs", methods=["GET"])
@require_auth
def beliefs():
    """Beliefs list filtered by status.

    Requires: Authorization: Bearer <cognito_jwt_token> (when AUTH_REQUIRED=true)

    Query params:
        project: project id (default: "default")
        status: active | uncertain | contradicted | superseded (default: "active")
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    project = _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    status = request.args.get("status", "active") or "active"
    valid_statuses = {"active", "uncertain", "contradicted", "superseded"}
    if status not in valid_statuses:
        return _error(
            f"Invalid status '{status}'. Must be one of: {', '.join(sorted(valid_statuses))}",
            400,
        )
    try:
        raw = adapter.get_beliefs(project_id=project, status=status)
        return jsonify([_map_belief(b) for b in raw])
    except Exception as exc:
        logger.exception("beliefs failed for project=%s status=%s", project, status)
        return _error(str(exc))


@app.route("/api/graph", methods=["GET"])
@require_auth
def graph():
    """All nodes + relationships for force-graph visualization.

    Query params:
        project: project id (default: "default")
        limit: max edges to return (default: 500)
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    project = _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    try:
        limit = int(request.args.get("limit", 500))
    except (TypeError, ValueError):
        limit = 500
    limit = max(1, min(limit, MAX_GRAPH_LIMIT))

    try:
        raw_nodes = adapter.query_nodes(node_type=None, project_id=project, limit=limit)
        raw_edges = _get_all_edges(adapter, project, limit=limit)

        nodes = [_map_node_for_graph(n) for n in raw_nodes]
        links = [_map_edge_for_graph(e) for e in raw_edges]

        return jsonify({"nodes": nodes, "links": links})
    except Exception as exc:
        logger.exception("graph failed for project=%s", project)
        return _error(str(exc))


@app.route("/api/nodes", methods=["GET"])
@require_auth
def nodes():
    """Query nodes, optionally filtered by type.

    Query params:
        project: project id (default: "default")
        type: node type label, e.g. Belief, Concept, Person (optional)
        limit: max results (default: 100)
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    project = _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    node_type = request.args.get("type") or None
    try:
        limit = int(request.args.get("limit", 100))
    except (TypeError, ValueError):
        limit = 100
    limit = max(1, min(limit, MAX_NODES_LIMIT))

    try:
        results = adapter.query_nodes(
            node_type=node_type, project_id=project, limit=limit
        )
        return jsonify(results)
    except Exception as exc:
        logger.exception("nodes failed for project=%s type=%s", project, node_type)
        return _error(str(exc))


@app.route("/api/traverse/<path:entity>", methods=["GET"])
@require_auth
def traverse(entity: str):
    """Multi-hop graph traversal from a named entity.

    URL:   /api/traverse/<entity name>
    Query params:
        project: project id (default: "default")
        depth: traversal depth (default: 2)
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    project = _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    try:
        depth = int(request.args.get("depth", 2))
    except (TypeError, ValueError):
        depth = 2
    depth = max(1, min(depth, MAX_TRAVERSE_DEPTH))

    try:
        result = adapter.traverse(
            entity_name=entity, project_id=project, max_depth=depth
        )
        return jsonify(result)
    except Exception as exc:
        logger.exception(
            "traverse failed for entity=%s project=%s", entity, project
        )
        return _error(str(exc))


@app.route("/api/ingest", methods=["POST"])
@require_auth
def ingest():
    """Ingest entities and/or relationships into the graph.

    Requires: Authorization: Bearer <cognito_jwt_token> (when AUTH_REQUIRED=true)

    JSON body:
        {
            "entities": [{name, node_type, summary, confidence?, ...}],
            "relationships": [{source, target, type, reason?, confidence?}],
            "source": "string label for provenance",
            "project": "project_id"
        }
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)

    body = request.get_json(silent=True)
    if not body:
        return _error("Request body must be JSON", 400)

    project = body.get("project") or "default"
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    source = body.get("source") or "api"
    entities = body.get("entities") or []
    relationships = body.get("relationships") or []

    if not isinstance(entities, list) or not isinstance(relationships, list):
        return _error("'entities' and 'relationships' must be arrays", 400)

    # Resolve node limit from tier (passed in body or default "free")
    tier = body.get("tier") or "free"
    node_limit = TIER_LIMITS.get(tier)

    try:
        entities_written = 0
        relationships_written = 0

        if entities:
            entities_written = adapter.write_entities(
                entities,
                source_cycle=source,
                source_type="api",
                project_id=project,
                node_limit=node_limit,
            )

        if relationships:
            relationships_written = adapter.write_relationships(
                relationships,
                source_cycle=source,
                source_type="api",
                project_id=project,
            )

        return jsonify(
            {
                "entities_written": entities_written,
                "relationships_written": relationships_written,
                "project": project,
            }
        )
    except NodeLimitExceeded as exc:
        return jsonify({
            "error": "node_limit_exceeded",
            "message": str(exc),
            "current": exc.current,
            "limit": exc.limit,
            "attempted": exc.attempted,
        }), 429
    except Exception as exc:
        logger.exception("ingest failed for project=%s", project)
        return _error(str(exc))


@app.route("/api/search", methods=["GET"])
@require_auth
def search():
    """Vector (semantic) search.

    Query params:
        q: search query text (required)
        project: project id (default: "default")
        top: number of results (default: 5)
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)

    query = request.args.get("q", "").strip()
    if not query:
        return _error("Query parameter 'q' is required", 400)

    project = _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    try:
        top = int(request.args.get("top", 5))
    except (TypeError, ValueError):
        top = 5
    top = max(1, min(top, MAX_SEARCH_TOP))

    try:
        results = adapter.vector_search(
            query_text=query, top_k=top, project_id=project
        )
        return jsonify(results)
    except Exception as exc:
        logger.exception("search failed for q=%s project=%s", query, project)
        return _error(str(exc))


@app.route("/api/ingest/text", methods=["POST"])
@require_auth
def ingest_text():
    """Extract entities and relationships from raw text via LLM, then ingest.

    Requires: Authorization: Bearer <cognito_jwt_token> (when AUTH_REQUIRED=true)

    This is the core merkraum pipeline: text -> extraction -> knowledge graph.

    JSON body:
        {
            "text": "raw text to extract knowledge from (required, max 8000 chars)",
            "project": "project_id (default: 'default')",
            "source": "provenance label (default: 'text_ingestion')"
        }

    Returns extracted entities and relationships, plus ingestion counts.
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)

    body = request.get_json(silent=True)
    if not body:
        return _error("Request body must be JSON", 400)

    text = (body.get("text") or "").strip()
    if not text:
        return _error("'text' field is required and must be non-empty", 400)
    if len(text) > 16000:
        return _error("Text too long (max 16000 characters)", 400)

    project = body.get("project") or "default"
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    source = body.get("source") or "text_ingestion"

    api_key = _get_openai_key()
    if not api_key:
        return _error(
            "OPENAI_API_KEY not configured. Set it in .env or environment.",
            503,
        )

    # Step 1: LLM extraction
    try:
        extracted = _llm_extract(text, api_key)
    except urllib.error.HTTPError as exc:
        logger.exception("LLM extraction HTTP error")
        return _error(f"LLM extraction failed: HTTP {exc.code}", 502)
    except Exception as exc:
        logger.exception("LLM extraction failed")
        return _error(f"LLM extraction failed: {exc}", 502)

    entities = extracted.get("entities", [])
    relationships = extracted.get("relationships", [])

    if not entities and not relationships:
        return jsonify({
            "extracted": {"entities": [], "relationships": []},
            "ingested": {"entities_written": 0, "relationships_written": 0},
            "project": project,
            "message": "No extractable knowledge found in the text.",
        })

    # Resolve node limit from tier
    tier = body.get("tier") or "free"
    node_limit = TIER_LIMITS.get(tier)

    # Step 2: Ingest into graph
    try:
        entities_written = 0
        relationships_written = 0

        if entities:
            entities_written = adapter.write_entities(
                entities,
                source_cycle=source,
                source_type="text_extraction",
                project_id=project,
                node_limit=node_limit,
            )

        if relationships:
            relationships_written = adapter.write_relationships(
                relationships,
                source_cycle=source,
                source_type="text_extraction",
                project_id=project,
            )

        return jsonify({
            "extracted": {
                "entities": entities,
                "relationships": relationships,
            },
            "ingested": {
                "entities_written": entities_written,
                "relationships_written": relationships_written,
            },
            "project": project,
        })
    except NodeLimitExceeded as exc:
        return jsonify({
            "extracted": {
                "entities": entities,
                "relationships": relationships,
            },
            "ingested": {"entities_written": 0, "relationships_written": 0},
            "project": project,
            "error": "node_limit_exceeded",
            "message": str(exc),
            "current": exc.current,
            "limit": exc.limit,
        }), 429
    except Exception as exc:
        logger.exception("Ingestion failed after extraction for project=%s", project)
        return jsonify({
            "extracted": {
                "entities": entities,
                "relationships": relationships,
            },
            "ingested": {"entities_written": 0, "relationships_written": 0},
            "project": project,
            "error": f"Extraction succeeded but ingestion failed: {exc}",
        }), 207  # Multi-Status: partial success


@app.route("/api/relationship", methods=["POST"])
@require_auth
def add_relationship_api():
    """Add or update a relationship between two existing nodes."""
    if adapter is None:
        return _error("Adapter not initialized", 503)

    body = request.get_json(silent=True) or {}
    source = (body.get("source") or "").strip()
    target = (body.get("target") or "").strip()
    source_type = (body.get("source_type") or "").strip() or None
    target_type = (body.get("target_type") or "").strip() or None
    rel_type = (body.get("type") or "").strip()
    if not source or not target or not rel_type:
        return _error("'source', 'target', and 'type' are required", 400)

    reason = (body.get("reason") or "").strip()
    confidence = body.get("confidence", 0.7)
    project = body.get("project") or "default"
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    actor = _actor()

    result = adapter.add_relationship(
        source=source,
        target=target,
        rel_type=rel_type,
        project_id=project,
        reason=reason,
        confidence=confidence,
        source_type=source_type,
        target_type=target_type,
        actor=actor,
    )
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/relationship", methods=["DELETE"])
@require_auth
def delete_relationship_api():
    """Delete a relationship between two nodes."""
    if adapter is None:
        return _error("Adapter not initialized", 503)

    body = request.get_json(silent=True) or {}
    source = (body.get("source") or "").strip()
    target = (body.get("target") or "").strip()
    source_type = (body.get("source_type") or "").strip() or None
    target_type = (body.get("target_type") or "").strip() or None
    rel_type = (body.get("type") or "").strip()
    if not source or not target or not rel_type:
        return _error("'source', 'target', and 'type' are required", 400)

    project = body.get("project") or "default"
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    actor = _actor()
    result = adapter.delete_relationship(
        source=source,
        target=target,
        rel_type=rel_type,
        project_id=project,
        source_type=source_type,
        target_type=target_type,
        actor=actor,
    )
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/node", methods=["DELETE"])
@require_auth
def delete_node_api():
    """Delete one node (and attached edges) with audit/history and vector sync."""
    if adapter is None:
        return _error("Adapter not initialized", 503)

    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    node_type = (body.get("node_type") or "").strip() or None
    if not name:
        return _error("'name' is required", 400)

    project = body.get("project") or "default"
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    actor = _actor()
    result = adapter.delete_node(name=name, project_id=project, node_type=node_type, actor=actor)
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/node", methods=["PATCH"])
@require_auth
def update_node_api():
    """Update node attributes and/or rename node with vector re-embedding."""
    if adapter is None:
        return _error("Adapter not initialized", 503)

    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return _error("'name' is required", 400)

    project = body.get("project") or "default"
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    new_name = (body.get("new_name") or "").strip() or None
    node_type = (body.get("node_type") or "").strip() or None
    updates = body.get("updates") or {}
    if not isinstance(updates, dict):
        return _error("'updates' must be an object", 400)

    actor = _actor()
    result = adapter.update_node(
        name=name,
        project_id=project,
        updates=updates,
        new_name=new_name,
        node_type=node_type,
        actor=actor,
    )
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/nodes/merge", methods=["POST"])
@require_auth
def merge_nodes_api():
    """Merge two nodes by keeping one and removing the other."""
    if adapter is None:
        return _error("Adapter not initialized", 503)

    body = request.get_json(silent=True) or {}
    keep_name = (body.get("keep_name") or "").strip()
    remove_name = (body.get("remove_name") or "").strip()
    keep_node_id = (body.get("keep_node_id") or "").strip() or None
    remove_node_id = (body.get("remove_node_id") or "").strip() or None
    keep_type = (body.get("keep_type") or "").strip() or None
    remove_type = (body.get("remove_type") or "").strip() or None
    if not keep_name or not remove_name:
        return _error("'keep_name' and 'remove_name' are required", 400)

    project = body.get("project") or "default"
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    actor = _actor()
    result = adapter.merge_nodes(
        keep_name=keep_name,
        remove_name=remove_name,
        keep_node_id=keep_node_id,
        remove_node_id=remove_node_id,
        keep_type=keep_type,
        remove_type=remove_type,
        project_id=project,
        actor=actor,
    )
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _load_secrets():
    """Load secrets from AWS Secrets Manager into environment variables.

    Reads merkraum/config secret (JSON with NEO4J_PASSWORD, OPENAI_API_KEY).
    Falls back silently to existing env vars / .env for local development.
    """
    try:
        boto3 = importlib.import_module("boto3")
        client = boto3.client("secretsmanager", region_name="eu-central-1")
        resp = client.get_secret_value(SecretId="merkraum/config")
        secrets = json.loads(resp["SecretString"])
        for key, value in secrets.items():
            os.environ.setdefault(key, value)
        logger.info("Loaded %d secrets from AWS Secrets Manager", len(secrets))
    except Exception as exc:
        logger.info("Secrets Manager not available (%s), using env vars", exc)


def _init_adapter():
    """Create and connect the adapter. Called once at startup."""
    global adapter
    try:
        adapter = cast(Neo4jBaseAdapter, create_adapter())
        adapter.connect()
        logger.info("Adapter connected: %s", type(adapter).__name__)
    except Exception as exc:
        logger.error("Failed to initialize adapter: %s", exc)
        # Don't crash — endpoints will return 503 until adapter is healthy.
        adapter = None


def _init_cognito_auth():
    """Initialize Cognito JWT validation. Called once at startup."""
    validator = get_cognito_validator()
    if validator:
        setattr(app, "_cognito_validator", validator)
        logger.info(
            "Cognito JWT validator initialized for pool: %s (region: %s)",
            validator.user_pool_id,
            validator.aws_region,
        )
    else:
        logger.warning(
            "Cognito JWT validation not configured. "
            "Set COGNITO_USER_POOL_ID and COGNITO_AWS_REGION to enable."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Merkraum REST API server",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8083,
        help="Port to listen on (default: 8083)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode",
    )
    args = parser.parse_args()

    _load_secrets()

    if _is_production_env():
        if not _is_auth_required():
            raise RuntimeError(
                "Security baseline failed: AUTH_REQUIRED must be true in production"
            )
        if not os.environ.get("COGNITO_CLIENT_ID"):
            raise RuntimeError(
                "Security baseline failed: COGNITO_CLIENT_ID is required in production"
            )
    _init_adapter()
    _init_cognito_auth()

    logger.info(
        "Starting Merkraum API on %s:%d (debug=%s)",
        args.host,
        args.port,
        args.debug,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
