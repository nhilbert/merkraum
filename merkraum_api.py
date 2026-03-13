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
import urllib.request
import urllib.error
from functools import lru_cache
from typing import cast

from merkraum_acl import is_auth_required, is_project_allowed, split_csv_env

from flask import Flask, jsonify, request

from merkraum_backend import (
    create_adapter, NODE_TYPES, RELATIONSHIP_TYPES, Neo4jBaseAdapter,
    NodeLimitExceeded, TIER_LIMITS,
)
from jwt_auth import get_cognito_validator, require_auth, require_scope, PATValidator

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
    return is_auth_required()


def _is_production_env() -> bool:
    env = (os.environ.get('APP_ENV') or os.environ.get('FLASK_ENV') or '').strip().lower()
    return env in {'prod', 'production'}


def _split_csv_env(name: str) -> set[str]:
    return split_csv_env(name)


def _is_project_allowed(project: str) -> bool:
    user_id = getattr(request, 'user_id', None)
    groups = set(getattr(request, 'groups', []) or [])
    pat_projects = getattr(request, 'pat_projects', None)
    pat_all_projects = getattr(request, 'pat_all_projects', None)
    return is_project_allowed(
        project, user_id, groups,
        pat_projects=pat_projects,
        pat_all_projects=pat_all_projects,
    )


def _deny_if_project_forbidden(project: str):
    if _is_project_allowed(project):
        return None
    return _error(f"Forbidden project access: '{project}'", 403)


def _actor() -> str:
    return (
        getattr(request, 'username', None)
        or getattr(request, 'user_id', None)
        or 'api'
    )


def _error(message: str, status: int = 500):
    return jsonify({'error': message}), status



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
# Discovery endpoints (unauthenticated — public by design)
# ---------------------------------------------------------------------------

@app.route("/api/discover", methods=["GET"])
def discover():
    """Machine-readable discovery endpoint — returns capabilities, auth, schema.

    No authentication required. This is how agents find out what Merkraum offers.
    """
    return jsonify({
        "name": "Merkraum",
        "version": "1.0",
        "description": "Structured knowledge graph memory for AI agents with belief tracking and contradiction detection.",
        "base_url": "https://agent.nhilbert.de/api/merkraum",
        "mcp_url": "https://agent.nhilbert.de/mcp/merkraum/",
        "skill_md": "https://agent.nhilbert.de/.well-known/skill.md",
        "authentication": {
            "type": "oauth2",
            "provider": "aws_cognito",
            "region": "eu-central-1",
            "flows": ["authorization_code_pkce"],
            "header": "Authorization: Bearer <jwt_token>",
        },
        "schema": {
            "node_types": list(NODE_TYPES),
            "relationship_types": list(RELATIONSHIP_TYPES),
        },
        "tiers": {k: {"node_limit": v} for k, v in TIER_LIMITS.items()},
        "endpoints": [
            {"path": "/api/search", "method": "GET", "auth": True, "description": "Semantic vector search"},
            {"path": "/api/ingest", "method": "POST", "auth": True, "description": "Ingest structured entities and relationships"},
            {"path": "/api/ingest/text", "method": "POST", "auth": True, "description": "Extract knowledge from text via LLM and ingest"},
            {"path": "/api/traverse/<entity>", "method": "GET", "auth": True, "description": "Multi-hop graph traversal from entity"},
            {"path": "/api/beliefs", "method": "GET", "auth": True, "description": "List beliefs by status"},
            {"path": "/api/stats", "method": "GET", "auth": True, "description": "Graph statistics"},
            {"path": "/api/graph", "method": "GET", "auth": True, "description": "Full graph data for visualization"},
            {"path": "/api/nodes", "method": "GET", "auth": True, "description": "Query nodes by type"},
            {"path": "/api/node", "method": "PATCH", "auth": True, "description": "Update node attributes"},
            {"path": "/api/node", "method": "DELETE", "auth": True, "description": "Delete node and attached edges"},
            {"path": "/api/relationship", "method": "POST", "auth": True, "description": "Add or update relationship"},
            {"path": "/api/relationship", "method": "DELETE", "auth": True, "description": "Delete relationship"},
            {"path": "/api/nodes/merge", "method": "POST", "auth": True, "description": "Merge two nodes"},
            {"path": "/api/health", "method": "GET", "auth": True, "description": "Service health check"},
            {"path": "/api/usage", "method": "GET", "auth": True, "description": "Usage metrics and tier limits"},
            {"path": "/api/projects", "method": "GET", "auth": True, "description": "List projects (detail=true for metadata)"},
            {"path": "/api/projects", "method": "POST", "auth": True, "description": "Create a new project (knowledge space)"},
            {"path": "/api/projects/<id>", "method": "GET", "auth": True, "description": "Get project metadata and usage"},
            {"path": "/api/projects/<id>", "method": "PATCH", "auth": True, "description": "Update project metadata"},
            {"path": "/api/projects/<id>", "method": "DELETE", "auth": True, "description": "Delete project and all data"},
        ],
        "operator": {
            "name": "Supervision Rheinland",
            "contact": "Dr. Norman Hilbert",
            "website": "https://merkraum.de",
        },
    })


@app.route("/.well-known/skill.md", methods=["GET"])
def well_known_skill():
    """Serve skill.md at the well-known path for agent discovery."""
    skill_path = os.path.join(os.path.dirname(__file__), "skill.md")
    if not os.path.exists(skill_path):
        return _error("skill.md not found", 404)
    with open(skill_path) as f:
        content = f.read()
    return content, 200, {"Content-Type": "text/markdown; charset=utf-8"}


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
@require_scope("projects")
def projects():
    """List projects with metadata.

    Requires: Authorization: Bearer <cognito_jwt_token> (when AUTH_REQUIRED=true)

    Query params:
        detail: if "true", return full project metadata from ProjectMeta nodes.
                Otherwise return sorted list of project_id strings (legacy).

    Returns: list of project metadata dicts (detail=true) or project_id strings.
    """
    adp = adapter
    if adp is None:
        return _error("Adapter not initialized", 503)
    try:
        detail = request.args.get("detail", "false").lower() in ("true", "1", "yes")
        if detail:
            owner = None
            if _is_auth_required():
                owner_filter = getattr(request, "user_id", None)
                admin_groups = _split_csv_env("ADMIN_GROUPS")
                groups = set(getattr(request, "groups", []) or [])
                # Admins see all projects; non-admins see own projects only
                if not (admin_groups and groups.intersection(admin_groups)):
                    owner = owner_filter
            project_list = adp.list_projects(owner=owner)

            # Auto-provision: if authenticated user has no projects, create one
            if not project_list and owner:
                username = getattr(request, "username", None) or "My Space"
                display_name = username.split("@")[0].replace(".", " ").title()
                try:
                    auto_result = adp.create_project(
                        project_id=owner,
                        name=f"{display_name}'s Knowledge Space",
                        owner=owner,
                        description="Auto-created on first login",
                        tier="free",
                    )
                    logger.info("Auto-provisioned project '%s' for user '%s'",
                                owner, username)
                    project_list = adp.list_projects(owner=owner)
                except ValueError:
                    # Project already exists (race condition) — just re-list
                    project_list = adp.list_projects(owner=owner)

            # Enrich with usage stats
            for proj in project_list:
                pid = proj.get("project_id", "")
                if _is_project_allowed(pid):
                    try:
                        usage = adp.get_usage(project_id=pid)
                        proj["nodes"] = usage.get("nodes", 0)
                        proj["edges"] = usage.get("edges", 0)
                    except Exception:
                        proj["nodes"] = 0
                        proj["edges"] = 0
            return jsonify(project_list)

        # Legacy mode: just project_id strings
        with adp._driver.session() as session:
            records = session.run(
                "MATCH (n) WHERE n.project_id IS NOT NULL "
                "RETURN DISTINCT n.project_id AS pid ORDER BY pid"
            )
            project_ids = [rec["pid"] for rec in records]
            if _is_auth_required():
                project_ids = [pid for pid in project_ids if _is_project_allowed(pid)]

        # Auto-provision in legacy mode too
        if not project_ids and _is_auth_required():
            user_id = getattr(request, "user_id", None)
            if user_id:
                username = getattr(request, "username", None) or "My Space"
                display_name = username.split("@")[0].replace(".", " ").title()
                try:
                    adp.create_project(
                        project_id=user_id,
                        name=f"{display_name}'s Knowledge Space",
                        owner=user_id,
                        description="Auto-created on first login",
                        tier="free",
                    )
                    logger.info("Auto-provisioned project '%s' for user '%s'",
                                user_id, username)
                    project_ids = [user_id]
                except ValueError:
                    project_ids = [user_id]

        return jsonify(project_ids)
    except Exception as exc:
        logger.exception("projects listing failed")
        return _error(str(exc))


@app.route("/api/projects", methods=["POST"])
@require_auth
@require_scope("projects")
def create_project():
    """Create a new project (knowledge space).

    Requires: Authorization: Bearer <cognito_jwt_token> (when AUTH_REQUIRED=true)

    Body (JSON):
        name: display name (required)
        project_id: unique identifier (optional — derived from name if omitted)
        description: project description (optional)
        tier: pricing tier — free, pro, team, enterprise (default: "free")

    Returns: created project metadata.
    """
    adp = adapter
    if adp is None:
        return _error("Adapter not initialized", 503)
    body = request.get_json(silent=True) or {}
    name = body.get("name", "").strip()
    if not name:
        return _error("'name' is required", 400)

    # Derive project_id from name if not provided
    project_id = body.get("project_id", "").strip()
    if not project_id:
        import re
        project_id = re.sub(r"[^a-z0-9_-]", "-", name.lower())
        project_id = re.sub(r"-+", "-", project_id).strip("-")
    if not project_id:
        return _error("Could not derive a valid project_id", 400)
    if project_id == "default":
        return _error("Cannot create project with reserved id 'default'", 400)

    owner = getattr(request, "user_id", None) or "anonymous"
    description = body.get("description", "")
    tier = body.get("tier", "free")

    try:
        result = adp.create_project(
            project_id=project_id, name=name, owner=owner,
            description=description, tier=tier,
        )
        return jsonify(result), 201
    except ValueError as exc:
        return _error(str(exc), 409)
    except Exception as exc:
        logger.exception("create_project failed")
        return _error(str(exc))


@app.route("/api/projects/<path:project_id>", methods=["GET"])
@require_auth
@require_scope("projects")
def get_project(project_id):
    """Get project metadata and usage.

    Requires: Authorization: Bearer <cognito_jwt_token> (when AUTH_REQUIRED=true)
    """
    adp = adapter
    if adp is None:
        return _error("Adapter not initialized", 503)
    denied = _deny_if_project_forbidden(project_id)
    if denied:
        return denied
    try:
        proj = adp.get_project(project_id)
        if not proj:
            return _error(f"Project '{project_id}' not found", 404)
        # Enrich with usage
        usage = adp.get_usage(project_id=project_id)
        proj["nodes"] = usage.get("nodes", 0)
        proj["edges"] = usage.get("edges", 0)
        proj["node_limit"] = TIER_LIMITS.get(proj.get("tier", "free"), 100)
        return jsonify(proj)
    except Exception as exc:
        logger.exception("get_project failed for %s", project_id)
        return _error(str(exc))


@app.route("/api/projects/<path:project_id>", methods=["PATCH"])
@require_auth
@require_scope("projects")
def update_project(project_id):
    """Update project metadata.

    Requires: Authorization: Bearer <cognito_jwt_token> (when AUTH_REQUIRED=true)

    Body (JSON):
        name: new display name (optional)
        description: new description (optional)
        tier: new tier (optional)
    """
    adp = adapter
    if adp is None:
        return _error("Adapter not initialized", 503)
    denied = _deny_if_project_forbidden(project_id)
    if denied:
        return denied
    body = request.get_json(silent=True) or {}
    try:
        result = adp.update_project(
            project_id=project_id,
            name=body.get("name"),
            description=body.get("description"),
            tier=body.get("tier"),
        )
        if not result.get("updated"):
            return _error(result.get("error", "Update failed"), 404)
        return jsonify(result)
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        logger.exception("update_project failed for %s", project_id)
        return _error(str(exc))


@app.route("/api/projects/<path:project_id>", methods=["DELETE"])
@require_auth
@require_scope("projects")
def delete_project(project_id):
    """Delete a project and all its data.

    Requires: Authorization: Bearer <cognito_jwt_token> (when AUTH_REQUIRED=true)
    WARNING: This is irreversible. Deletes all nodes, relationships, and metadata.
    """
    adp = adapter
    if adp is None:
        return _error("Adapter not initialized", 503)
    denied = _deny_if_project_forbidden(project_id)
    if denied:
        return denied
    if project_id == "default":
        return _error("Cannot delete the default project", 400)
    try:
        counts = adp.delete_project_data(project_id=project_id)
        return jsonify({"deleted": True, "project_id": project_id, **counts})
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        logger.exception("delete_project failed for %s", project_id)
        return _error(str(exc))


@app.route("/api/stats", methods=["GET"])
@require_auth
@require_scope("read")
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
@require_scope("read")
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
@require_scope("read")
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
@require_scope("read")
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
@require_scope("read")
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
@require_scope("read")
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
@require_scope("ingest")
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
@require_scope("search")
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
@require_scope("ingest")
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
@require_scope("write")
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
@require_scope("write")
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
@require_scope("write")
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
@require_scope("write")
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
@require_scope("write")
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
# PAT Management Endpoints (Cognito JWT auth only — tokens don't manage themselves)
# ---------------------------------------------------------------------------


@app.route("/api/tokens", methods=["POST"])
@require_auth
def create_token():
    """Create a new Personal Access Token.

    Requires Cognito JWT auth (humans manage tokens).
    Returns the full token plaintext ONCE in the response.

    JSON body:
        {
            "name": "My Token",
            "scopes": ["read", "write", "search"],
            "projects": ["project-uuid"],
            "all_projects": false,
            "expires_in_days": 90
        }
    """
    # PATs cannot create other PATs (only Cognito users can)
    if getattr(request, "pat_scopes", None) is not None:
        return _error("PATs cannot create tokens. Use Cognito JWT auth.", 403)

    pat_validator = getattr(current_app, "_pat_validator", None)
    if not pat_validator:
        return _error("PAT system not configured", 503)

    body = request.get_json(silent=True)
    if not body:
        return _error("Request body must be JSON", 400)

    name = (body.get("name") or "").strip()
    if not name:
        return _error("'name' is required", 400)

    scopes = body.get("scopes") or ["read", "search"]
    if not isinstance(scopes, list):
        return _error("'scopes' must be an array", 400)

    projects = body.get("projects") or []
    if not isinstance(projects, list):
        return _error("'projects' must be an array", 400)

    all_projects = bool(body.get("all_projects", False))

    expires_in_days = body.get("expires_in_days")
    expires_at = None
    if expires_in_days is not None:
        try:
            days = int(expires_in_days)
            if days < 1 or days > 365:
                return _error("'expires_in_days' must be between 1 and 365", 400)
            from datetime import datetime as dt, timezone as tz, timedelta
            expires_at = (dt.now(tz.utc) + timedelta(days=days)).isoformat()
        except (TypeError, ValueError):
            return _error("'expires_in_days' must be an integer", 400)

    tier = body.get("tier") or "free"

    try:
        result = pat_validator.create_token(
            owner_id=request.user_id,
            name=name,
            scopes=scopes,
            projects=projects,
            all_projects=all_projects,
            expires_at=expires_at,
            tier=tier,
        )
        logger.info(
            "PAT created: prefix=%s owner=%s scopes=%s",
            result["token_prefix"], request.user_id, scopes,
        )
        return jsonify(result), 201
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        logger.exception("PAT creation failed")
        return _error(f"Token creation failed: {exc}")


@app.route("/api/tokens", methods=["GET"])
@require_auth
def list_tokens():
    """List all PATs for the authenticated user (never returns plaintext)."""
    if getattr(request, "pat_scopes", None) is not None:
        return _error("PATs cannot list tokens. Use Cognito JWT auth.", 403)

    pat_validator = getattr(current_app, "_pat_validator", None)
    if not pat_validator:
        return _error("PAT system not configured", 503)

    tokens = pat_validator.list_tokens(request.user_id)
    return jsonify(tokens)


@app.route("/api/tokens/<token_prefix>", methods=["DELETE"])
@require_auth
def revoke_token(token_prefix: str):
    """Revoke a PAT by its prefix."""
    if getattr(request, "pat_scopes", None) is not None:
        return _error("PATs cannot revoke tokens. Use Cognito JWT auth.", 403)

    pat_validator = getattr(current_app, "_pat_validator", None)
    if not pat_validator:
        return _error("PAT system not configured", 503)

    revoked = pat_validator.revoke_token(request.user_id, token_prefix)
    if revoked:
        logger.info(
            "PAT revoked: prefix=%s owner=%s", token_prefix, request.user_id,
        )
        return jsonify({"ok": True, "message": "Token revoked"})
    return _error("Token not found or already revoked", 404)


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


def _init_pat_auth():
    """Initialize PAT validation using the adapter's Neo4j driver. Called once at startup."""
    if adapter is None:
        logger.warning("Adapter not initialized — PAT auth disabled")
        return
    driver = getattr(adapter, "_driver", None)
    if not driver:
        logger.warning("No Neo4j driver available — PAT auth disabled")
        return
    try:
        PATValidator.ensure_constraints(driver)
        pat_validator = PATValidator(driver)
        setattr(app, "_pat_validator", pat_validator)
        logger.info("PAT validator initialized (Neo4j-backed)")
    except Exception as exc:
        logger.error("Failed to initialize PAT validator: %s", exc)


def main():
    parser = argparse.ArgumentParser(
        description="Merkraum REST API server",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
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
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not _is_auth_required():
        raise RuntimeError(
            "Security baseline failed: AUTH_REQUIRED must be true when binding to non-loopback host"
        )
    _init_adapter()
    _init_cognito_auth()
    _init_pat_auth()

    logger.info(
        "Starting Merkraum API on %s:%d (debug=%s)",
        args.host,
        args.port,
        args.debug,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
