#!/usr/bin/env python3
"""
Merkraum REST API Server — HTTP interface for the knowledge graph.

Wraps BackendAdapter from merkraum_backend.py and exposes endpoints
consumed by the React frontend (merkraum-front).

v1.0 — SUP-94
"""

import argparse
import logging
import sys

from flask import Flask, jsonify, request

from merkraum_backend import create_adapter

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
adapter = None


# ---------------------------------------------------------------------------
# CORS — applied to every response
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS = {
    "https://app.merkraum.de",
    "http://localhost:3000",
    "http://localhost:5173",
}


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    else:
        # Fallback: allow all origins during development.
        # For production, tighten this by removing the else branch.
        response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = (
        "GET, POST, OPTIONS, PUT, DELETE"
    )
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Authorization, X-Requested-With"
    )
    response.headers["Access-Control-Max-Age"] = "3600"
    return response


@app.route("/api/<path:path>", methods=["OPTIONS"])
@app.route("/api", methods=["OPTIONS"])
def handle_preflight(path=""):
    """Handle CORS preflight requests."""
    return jsonify({}), 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_id() -> str:
    """Extract project_id from query params, defaulting to 'default'."""
    return request.args.get("project", "default") or "default"


def _error(message: str, status: int = 500):
    return jsonify({"error": message}), status


def _get_all_edges(adp, project_id: str, limit: int = 1000) -> list:
    """Query all relationships for a project directly via Neo4j driver."""
    edges = []
    with adp._driver.session() as session:
        records = session.run(
            """
            MATCH (a {project_id: $pid})-[r]->(b {project_id: $pid})
            RETURN a.name AS source, b.name AS target, type(r) AS type,
                   r.confidence AS confidence, r.reason AS reason
            LIMIT $limit
            """,
            pid=project_id,
            limit=limit,
        )
        for rec in records:
            edges.append(
                {
                    "source": rec["source"],
                    "target": rec["target"],
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
        {entities, relationships, beliefs, contradictions}
    """
    return {
        "entities": raw.get("total_nodes", 0),
        "relationships": raw.get("total_edges", 0),
        "beliefs": raw.get("nodes", {}).get("Belief", 0),
        "contradictions": raw.get("edges", {}).get("CONTRADICTS", 0),
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
    return {
        "id": n.get("name", ""),
        "name": n.get("name", ""),
        "type": node_type,
        "summary": n.get("summary", ""),
        "confidence": n.get("confidence"),
        # val drives node size in react-force-graph; beliefs slightly larger
        "val": 2 if node_type == "Belief" else 1,
    }


def _map_edge_for_graph(e: dict) -> dict:
    """Map an edge dict to the frontend graph link format."""
    return {
        "source": e.get("source", ""),
        "target": e.get("target", ""),
        "type": e.get("type", ""),
        "reason": e.get("reason", ""),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    """Health check — returns adapter connectivity status."""
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


@app.route("/api/stats", methods=["GET"])
def stats():
    """Graph stats in frontend format: {entities, relationships, beliefs, contradictions}."""
    if adapter is None:
        return _error("Adapter not initialized", 503)
    project = _project_id()
    try:
        raw = adapter.get_stats(project_id=project)
        return jsonify(_map_stats(raw))
    except Exception as exc:
        logger.exception("stats failed for project=%s", project)
        return _error(str(exc))


@app.route("/api/beliefs", methods=["GET"])
def beliefs():
    """Beliefs list filtered by status.

    Query params:
        project: project id (default: "default")
        status: active | uncertain | contradicted | superseded (default: "active")
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    project = _project_id()
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
def graph():
    """All nodes + relationships for force-graph visualization.

    Query params:
        project: project id (default: "default")
        limit: max edges to return (default: 500)
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    project = _project_id()
    try:
        limit = int(request.args.get("limit", 500))
    except (TypeError, ValueError):
        limit = 500

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
    node_type = request.args.get("type") or None
    try:
        limit = int(request.args.get("limit", 100))
    except (TypeError, ValueError):
        limit = 100

    try:
        results = adapter.query_nodes(
            node_type=node_type, project_id=project, limit=limit
        )
        return jsonify(results)
    except Exception as exc:
        logger.exception("nodes failed for project=%s type=%s", project, node_type)
        return _error(str(exc))


@app.route("/api/traverse/<path:entity>", methods=["GET"])
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
    try:
        depth = int(request.args.get("depth", 2))
    except (TypeError, ValueError):
        depth = 2

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
def ingest():
    """Ingest entities and/or relationships into the graph.

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
    source = body.get("source") or "api"
    entities = body.get("entities") or []
    relationships = body.get("relationships") or []

    if not isinstance(entities, list) or not isinstance(relationships, list):
        return _error("'entities' and 'relationships' must be arrays", 400)

    try:
        entities_written = 0
        relationships_written = 0

        if entities:
            entities_written = adapter.write_entities(
                entities,
                source_cycle=source,
                source_type="api",
                project_id=project,
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
    except Exception as exc:
        logger.exception("ingest failed for project=%s", project)
        return _error(str(exc))


@app.route("/api/search", methods=["GET"])
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
    try:
        top = int(request.args.get("top", 5))
    except (TypeError, ValueError):
        top = 5

    try:
        results = adapter.vector_search(
            query_text=query, top_k=top, project_id=project
        )
        return jsonify(results)
    except Exception as exc:
        logger.exception("search failed for q=%s project=%s", query, project)
        return _error(str(exc))


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _init_adapter():
    """Create and connect the adapter. Called once at startup."""
    global adapter
    try:
        adapter = create_adapter()
        adapter.connect()
        logger.info("Adapter connected: %s", type(adapter).__name__)
    except Exception as exc:
        logger.error("Failed to initialize adapter: %s", exc)
        # Don't crash — endpoints will return 503 until adapter is healthy.
        adapter = None


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

    _init_adapter()

    logger.info(
        "Starting Merkraum API on %s:%d (debug=%s)",
        args.host,
        args.port,
        args.debug,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
