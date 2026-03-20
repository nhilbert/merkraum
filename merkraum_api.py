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
import queue
import threading
import time
import urllib.request
import urllib.error
from functools import lru_cache
from typing import cast

from merkraum_llm import llm_extract, get_provider_info

from merkraum_acl import is_auth_required, is_project_allowed, split_csv_env

from flask import Flask, jsonify, request, current_app

from merkraum_backend import (
    create_adapter, NODE_TYPES, RELATIONSHIP_TYPES, KNOWLEDGE_TYPES,
    Neo4jBaseAdapter, NodeLimitExceeded, TIER_LIMITS,
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

MAX_GRAPH_LIMIT = 5000
MAX_NODES_LIMIT = 500
MAX_SEARCH_TOP = 50
MAX_TRAVERSE_DEPTH = 5
MAX_GRAPH_HOPS = 3
MAX_GRAPH_EXPAND_LIMIT = 100
MAX_VECTOR_REINDEX_LIMIT = 10000


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


def _token_has_scope(scope: str) -> bool:
    """Check PAT scope; Cognito JWT users are treated as fully scoped."""
    pat_scopes = getattr(request, 'pat_scopes', None)
    if pat_scopes is None:
        return True
    return scope in pat_scopes or 'admin' in pat_scopes


def _get_all_edges(adp, project_id: str, limit: int = 1000) -> list:
    """Query all relationships for a project directly via Neo4j driver."""
    edges = []
    with adp._driver.session() as session:
        records = session.run(
            """
            MATCH (a {project_id: $pid})-[r]->(b {project_id: $pid})
            WHERE any(lbl IN labels(a) WHERE lbl IN $node_types)
              AND any(lbl IN labels(b) WHERE lbl IN $node_types)
              AND a.expired_at IS NULL AND b.expired_at IS NULL
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


def _get_contradiction_pairs(adp, project_id: str) -> list:
    """Query actual CONTRADICTS relationship pairs from Neo4j.

    Returns a list of dicts, each with belief_a, belief_b, reason, and relationship metadata.
    Uses directed match with DISTINCT to avoid duplicate pairs.
    """
    pairs = []
    with adp._driver.session() as session:
        records = session.run(
            """
            MATCH (a:Belief {project_id: $pid})-[r:CONTRADICTS]->(b:Belief {project_id: $pid})
            WHERE a.active = true AND b.active = true
            RETURN a.name AS a_name, a.summary AS a_summary,
                   a.confidence AS a_confidence, a.source_cycle AS a_cycle,
                   a.node_id AS a_node_id,
                   b.name AS b_name, b.summary AS b_summary,
                   b.confidence AS b_confidence, b.source_cycle AS b_cycle,
                   b.node_id AS b_node_id,
                   r.reason AS reason, r.confidence AS rel_confidence,
                   type(r) AS rel_type
            """,
            pid=project_id,
        )
        for rec in records:
            pairs.append({
                "belief_a": {
                    "name": rec["a_name"] or "",
                    "summary": rec["a_summary"] or "",
                    "confidence": rec["a_confidence"] or 0,
                    "status": "contradicted",
                    "source": rec["a_cycle"] or "",
                    "node_id": rec["a_node_id"] or "",
                },
                "belief_b": {
                    "name": rec["b_name"] or "",
                    "summary": rec["b_summary"] or "",
                    "confidence": rec["b_confidence"] or 0,
                    "status": "contradicted",
                    "source": rec["b_cycle"] or "",
                    "node_id": rec["b_node_id"] or "",
                },
                "reason": rec["reason"] or "Conflicting evidence detected",
                "rel_confidence": rec["rel_confidence"] or 0,
            })
    return pairs


def _get_semantic_subgraph(adp, project_id: str, query: str, *, limit: int, hops: int, top: int) -> dict:
    """Build a query-centered subgraph via semantic seeds + N-hop neighborhood."""
    seed_hits = adp.vector_search(query_text=query, top_k=top, project_id=project_id)
    seed_names: list[str] = []
    for hit in seed_hits:
        metadata = hit.get("metadata") if isinstance(hit, dict) else None
        if not isinstance(metadata, dict):
            continue
        name = metadata.get("name")
        if isinstance(name, str) and name.strip():
            seed_names.append(name.strip())

    # Preserve order while deduplicating.
    seen = set()
    deduped_seeds: list[str] = []
    for name in seed_names:
        if name in seen:
            continue
        seen.add(name)
        deduped_seeds.append(name)

    if not deduped_seeds:
        return {"nodes": [], "links": [], "meta": {"mode": "semantic_subgraph", "query": query, "seed_count": 0, "hops": hops}}

    # Variable length in Cypher needs literal bounds, so hops is validated and interpolated.
    depth = max(0, min(hops, MAX_GRAPH_HOPS))
    node_records = []
    with adp._driver.session() as session:
        node_records = list(session.run(
            f"""
            MATCH (s {{project_id: $pid}})
            WHERE s.name IN $seed_names AND s.expired_at IS NULL
            MATCH p=(s)-[*0..{depth}]-(n {{project_id: $pid}})
            WHERE any(lbl IN labels(n) WHERE lbl IN $node_types)
              AND n.expired_at IS NULL
            WITH DISTINCT n
            LIMIT $limit
            RETURN elementId(n) AS eid,
                   n.name AS name,
                   labels(n)[0] AS node_type,
                   n.summary AS summary,
                   n.node_id AS node_id,
                   n.confidence AS confidence
            """,
            pid=project_id,
            seed_names=deduped_seeds,
            node_types=list(NODE_TYPES),
            limit=limit,
        ))

    nodes = [
        {
            "name": rec.get("name") or "",
            "type": rec.get("node_type") or "Concept",
            "summary": rec.get("summary") or "",
            "node_id": rec.get("node_id"),
            "confidence": rec.get("confidence"),
        }
        for rec in node_records
    ]

    element_ids = [rec.get("eid") for rec in node_records if rec.get("eid")]
    if not element_ids:
        return {"nodes": [], "links": [], "meta": {"mode": "semantic_subgraph", "query": query, "seed_count": len(deduped_seeds), "hops": depth}}

    links = []
    with adp._driver.session() as session:
        edge_records = session.run(
            """
            UNWIND $eids AS eid
            MATCH (n)
            WHERE elementId(n) = eid
            WITH collect(n) AS nodes
            UNWIND nodes AS a
            MATCH (a)-[r]->(b)
            WHERE b IN nodes
            RETURN a.name AS source_name,
                   b.name AS target_name,
                   labels(a)[0] AS source_type,
                   labels(b)[0] AS target_type,
                   a.node_id AS source_node_id,
                   b.node_id AS target_node_id,
                   type(r) AS type,
                   r.confidence AS confidence,
                   r.reason AS reason
            LIMIT $limit
            """,
            eids=element_ids,
            limit=limit,
        )
        links = [dict(rec) for rec in edge_records]

    return {
        "nodes": [_map_node_for_graph(n) for n in nodes],
        "links": [_map_edge_for_graph(e) for e in links],
        "meta": {
            "mode": "semantic_subgraph",
            "query": query,
            "seed_count": len(deduped_seeds),
            "hops": depth,
            "requested_limit": limit,
            "returned_nodes": len(nodes),
            "returned_links": len(links),
            "truncated": len(nodes) >= limit,
            "cap_reason": "node_limit" if len(nodes) >= limit else None,
        },
    }


def _get_text_subgraph(adp, project_id: str, query: str, *, limit: int, hops: int, top: int) -> dict:
    """Build a query-centered subgraph via text seeds + N-hop neighborhood."""
    normalized_search = (query or "").strip().lower()
    if not normalized_search:
        return {"nodes": [], "links": [], "meta": {"mode": "text_subgraph", "query": query, "seed_count": 0, "hops": hops}}

    seed_records = []
    with adp._driver.session() as session:
        seed_records = list(session.run(
            """
            MATCH (n {project_id: $pid})
            WHERE any(lbl IN labels(n) WHERE lbl IN $node_types)
              AND n.expired_at IS NULL
              AND (
                toLower(coalesce(n.name, '')) CONTAINS $search_query
                OR toLower(coalesce(n.summary, '')) CONTAINS $search_query
              )
            RETURN DISTINCT n.name AS name
            ORDER BY
              CASE WHEN toLower(coalesce(n.name, '')) = $search_query THEN 0 ELSE 1 END,
              size(coalesce(n.name, '')) ASC,
              n.name ASC
            LIMIT $top
            """,
            pid=project_id,
            node_types=list(NODE_TYPES),
            search_query=normalized_search,
            top=top,
        ))

    seed_names = [str(rec.get("name") or "").strip() for rec in seed_records]
    seed_names = [name for name in seed_names if name]
    if not seed_names:
        return {"nodes": [], "links": [], "meta": {"mode": "text_subgraph", "query": query, "seed_count": 0, "hops": hops}}

    # Variable length in Cypher needs literal bounds, so hops is validated and interpolated.
    depth = max(0, min(hops, MAX_GRAPH_HOPS))
    node_records = []
    with adp._driver.session() as session:
        node_records = list(session.run(
            f"""
            MATCH (s {{project_id: $pid}})
            WHERE s.name IN $seed_names AND s.expired_at IS NULL
            MATCH p=(s)-[*0..{depth}]-(n {{project_id: $pid}})
            WHERE any(lbl IN labels(n) WHERE lbl IN $node_types)
              AND n.expired_at IS NULL
            WITH DISTINCT n
            LIMIT $limit
            RETURN elementId(n) AS eid,
                   n.name AS name,
                   labels(n)[0] AS node_type,
                   n.summary AS summary,
                   n.node_id AS node_id,
                   n.confidence AS confidence
            """,
            pid=project_id,
            seed_names=seed_names,
            node_types=list(NODE_TYPES),
            limit=limit,
        ))

    nodes = [
        {
            "name": rec.get("name") or "",
            "type": rec.get("node_type") or "Concept",
            "summary": rec.get("summary") or "",
            "node_id": rec.get("node_id"),
            "confidence": rec.get("confidence"),
        }
        for rec in node_records
    ]

    element_ids = [rec.get("eid") for rec in node_records if rec.get("eid")]
    if not element_ids:
        return {"nodes": [], "links": [], "meta": {"mode": "text_subgraph", "query": query, "seed_count": len(seed_names), "hops": depth}}

    links = []
    with adp._driver.session() as session:
        edge_records = session.run(
            """
            UNWIND $eids AS eid
            MATCH (n)
            WHERE elementId(n) = eid
            WITH collect(n) AS nodes
            UNWIND nodes AS a
            MATCH (a)-[r]->(b)
            WHERE b IN nodes
            RETURN a.name AS source_name,
                   b.name AS target_name,
                   labels(a)[0] AS source_type,
                   labels(b)[0] AS target_type,
                   a.node_id AS source_node_id,
                   b.node_id AS target_node_id,
                   type(r) AS type,
                   r.confidence AS confidence,
                   r.reason AS reason
            LIMIT $limit
            """,
            eids=element_ids,
            limit=limit,
        )
        links = [dict(rec) for rec in edge_records]

    return {
        "nodes": [_map_node_for_graph(n) for n in nodes],
        "links": [_map_edge_for_graph(e) for e in links],
        "meta": {
            "mode": "text_subgraph",
            "query": query,
            "seed_count": len(seed_names),
            "hops": depth,
            "requested_limit": limit,
            "returned_nodes": len(nodes),
            "returned_links": len(links),
            "truncated": len(nodes) >= limit,
            "cap_reason": "node_limit" if len(nodes) >= limit else None,
        },
    }


def _get_node_expansion(
    adp,
    project_id: str,
    *,
    node_id: str,
    node_name: str,
    node_type: str,
    limit: int,
) -> dict:
    """Expand one node by fetching up to N direct neighbors and connecting edges."""
    with adp._driver.session() as session:
        seed_record = session.run(
            """
            MATCH (s {project_id: $pid})
            WHERE any(lbl IN labels(s) WHERE lbl IN $node_types)
              AND s.expired_at IS NULL
              AND (
                ($node_id <> '' AND coalesce(s.node_id, '') = $node_id)
                OR
                ($node_name <> '' AND s.name = $node_name
                  AND ($node_type = '' OR $node_type IN labels(s)))
              )
            RETURN elementId(s) AS eid,
                   s.name AS name,
                   labels(s)[0] AS node_type,
                   s.summary AS summary,
                   s.node_id AS node_id,
                   s.confidence AS confidence
            LIMIT 1
            """,
            pid=project_id,
            node_types=list(NODE_TYPES),
            node_id=node_id,
            node_name=node_name,
            node_type=node_type,
        ).single()

        if not seed_record:
            return {
                "nodes": [],
                "links": [],
                "meta": {
                    "mode": "node_expand",
                    "seed_node_id": node_id or None,
                    "seed_name": node_name or None,
                    "requested_limit": limit,
                    "max_neighbor_limit": MAX_GRAPH_EXPAND_LIMIT,
                    "returned_nodes": 0,
                    "returned_links": 0,
                    "returned_neighbors": 0,
                    "total_neighbors": 0,
                    "truncated": False,
                    "cap_reason": None,
                },
            }

        seed_eid = seed_record.get("eid")
        total_neighbors_record = session.run(
            """
            MATCH (s)
            WHERE elementId(s) = $seed_eid
            MATCH (s)-[]-(n {project_id: $pid})
            WHERE any(lbl IN labels(n) WHERE lbl IN $node_types)
              AND n.expired_at IS NULL
            RETURN count(DISTINCT n) AS total
            """,
            seed_eid=seed_eid,
            pid=project_id,
            node_types=list(NODE_TYPES),
        ).single()
        total_neighbors = int((total_neighbors_record or {}).get("total") or 0)

        neighbor_records = list(
            session.run(
                """
                MATCH (s)
                WHERE elementId(s) = $seed_eid
                MATCH (s)-[]-(n {project_id: $pid})
                WHERE any(lbl IN labels(n) WHERE lbl IN $node_types)
                  AND n.expired_at IS NULL
                WITH DISTINCT n
                LIMIT $limit
                RETURN elementId(n) AS eid,
                       n.name AS name,
                       labels(n)[0] AS node_type,
                       n.summary AS summary,
                       n.node_id AS node_id,
                       n.confidence AS confidence
                """,
                seed_eid=seed_eid,
                pid=project_id,
                node_types=list(NODE_TYPES),
                limit=limit,
            )
        )

        seed_node = {
            "name": seed_record.get("name") or "",
            "type": seed_record.get("node_type") or "Concept",
            "summary": seed_record.get("summary") or "",
            "node_id": seed_record.get("node_id"),
            "confidence": seed_record.get("confidence"),
        }
        neighbor_nodes = [
            {
                "name": rec.get("name") or "",
                "type": rec.get("node_type") or "Concept",
                "summary": rec.get("summary") or "",
                "node_id": rec.get("node_id"),
                "confidence": rec.get("confidence"),
            }
            for rec in neighbor_records
            if rec.get("eid") and rec.get("eid") != seed_eid
        ]

        included_eids = [seed_eid] + [
            rec.get("eid")
            for rec in neighbor_records
            if rec.get("eid") and rec.get("eid") != seed_eid
        ]

        edge_limit = max(limit * 4, MAX_GRAPH_EXPAND_LIMIT)
        edge_records = list(
            session.run(
                """
                UNWIND $eids AS eid
                MATCH (n)
                WHERE elementId(n) = eid
                WITH collect(n) AS nodes
                UNWIND nodes AS a
                MATCH (a)-[r]->(b)
                WHERE b IN nodes
                RETURN a.name AS source_name,
                       b.name AS target_name,
                       labels(a)[0] AS source_type,
                       labels(b)[0] AS target_type,
                       a.node_id AS source_node_id,
                       b.node_id AS target_node_id,
                       type(r) AS type,
                       r.confidence AS confidence,
                       r.reason AS reason
                LIMIT $edge_limit
                """,
                eids=included_eids,
                edge_limit=edge_limit,
            )
        )

    return {
        "nodes": [_map_node_for_graph(n) for n in [seed_node, *neighbor_nodes]],
        "links": [_map_edge_for_graph(dict(rec)) for rec in edge_records],
        "meta": {
            "mode": "node_expand",
            "seed_node_id": seed_node.get("node_id"),
            "seed_name": seed_node.get("name"),
            "requested_limit": limit,
            "max_neighbor_limit": MAX_GRAPH_EXPAND_LIMIT,
            "returned_neighbors": len(neighbor_nodes),
            "total_neighbors": total_neighbors,
            "returned_nodes": len(neighbor_nodes) + 1,
            "returned_links": len(edge_records),
            "truncated": total_neighbors > len(neighbor_nodes),
            "cap_reason": (
                "server_neighbor_limit"
                if total_neighbors > len(neighbor_nodes) and limit >= MAX_GRAPH_EXPAND_LIMIT
                else "request_limit"
                if total_neighbors > len(neighbor_nodes)
                else None
            ),
        },
    }


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
    result = {
        "name": b.get("name", ""),
        "summary": b.get("summary", ""),
        "confidence": b.get("confidence", 0),
        "status": b.get("status", "active"),
        "source": b.get("cycle", b.get("source_cycle", "")),
    }
    # Include audit-relevant fields when available
    if b.get("valid_until") is not None:
        result["valid_until"] = b["valid_until"]
    if b.get("vsm_level") is not None:
        result["vsm_level"] = b["vsm_level"]
    if b.get("node_id") is not None:
        result["node_id"] = b["node_id"]
    if b.get("updated_at") is not None:
        result["updated_at"] = b["updated_at"]
    if b.get("knowledge_type") is not None:
        result["knowledge_type"] = b["knowledge_type"]
    return result


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

## VSM LEVEL CLASSIFICATION:
Classify each entity by its organizational function using the Viable System Model (S1-S5):
- S1 (Operational): Task-specific facts, in-progress data, current values, scratch notes. Fast-cycling.
- S2 (Coordination): Rules, procedures, process descriptions, scheduling, "how things connect." Valid while processes exist.
- S3 (Control): Performance metrics, quality assessments, resource capacities, priorities. Updated at reviews.
- S4 (Strategic): Environmental models, competitive intelligence, market trends, research findings. Slow-cycling.
- S5 (Identity): Core values, policy rules, identity claims, normative commitments. Near-permanent.
Set "vsm_level" to one of: S1, S2, S3, S4, S5. When uncertain, omit it (null).
Heuristics: task data → S1, process docs → S2, metrics/assessments → S3, external research → S4, values/policy → S5.

## TEMPORAL VALIDITY:
For entities with a known expiration or temporal boundary, set "valid_until" (ISO 8601 date).
- Events: set valid_until to the event date (knowledge becomes historical after).
- Regulations with enforcement dates: set valid_until to when they take effect or are superseded.
- Market data, statistics, competitive assessments: set valid_until ~90 days from the text date.
- Beliefs about current state: set valid_until ~30 days if the domain changes rapidly.
- Permanent concepts, people, organizations: omit valid_until (null = no expiration).
- Only set valid_until when the text provides temporal cues. When uncertain, omit it.
- If vsm_level is set but valid_until is omitted, a default TTL is applied based on the level (S1=30d, S2=90d, S3=180d, S4=365d, S5=none).

## KNOWLEDGE TYPE CLASSIFICATION:
Classify each entity by the epistemological nature of its knowledge using "knowledge_type":
- fact: Permanent world facts that are objectively true and rarely change. E.g., "Berlin is the capital of Germany", "Stafford Beer created the VSM". High confidence, typically no expiry.
- state: Temporary current-state facts that describe how things are NOW but will change. E.g., "Merkraum has 0 users", "Budget is at 70%". Should have valid_until set.
- rule: Rules, policies, procedures, or operational constraints. E.g., "S3 review every 10 cycles", "GDPR applies to EU data". Changes rarely.
- belief: Subjective assessments, opinions, or uncertain propositions. E.g., "Competitive window is 3-8 months", "Agent memory market will consolidate". Always has confidence < 1.0.
- memory: Episodic memories of specific events or actions. E.g., "Norman sent directive on Mar 16", "Security incident occurred at Z1294". Temporal, tied to a moment.
Set "knowledge_type" to one of: fact, state, rule, belief, memory. When uncertain, omit it (null).
Heuristics: established truths → fact, current metrics/status → state, documented procedures → rule, assessments/predictions → belief, past events/actions → memory.

## CONTRADICTION RULES (for CONTRADICTS relationships):
8. Only use CONTRADICTS when two beliefs are about the SAME subject with SAME scope.
9. Different scopes are NOT contradictions. Examples of different scopes:
   - "German market grew 30%" vs "International market declined" — different geographic scopes
   - "Revenue increased in Q1" vs "Revenue decreased in Q4" — different time periods
   - "Product A sales up" vs "Product B sales down" — different subjects
10. Before emitting CONTRADICTS, verify: same entity, same scope, same time frame, genuinely incompatible claims.
11. Include the specific reason in the "reason" field explaining WHY the beliefs conflict.

## OUTPUT FORMAT:
Return a JSON object with two arrays: "entities" and "relationships".

{{
  "entities": [
    {{
      "name": "canonical name",
      "node_type": "one of the node types above",
      "summary": "one-paragraph description, max 500 chars",
      "confidence": 0.9,
      "valid_until": "2026-06-30T00:00:00Z or null",
      "vsm_level": "S1 or S2 or S3 or S4 or S5 or null",
      "knowledge_type": "fact or state or rule or belief or memory or null"
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


def _llm_extract_text(text: str) -> dict:
    """Extract entities and relationships from text using configured LLM provider.

    Uses merkraum_llm module for provider-agnostic extraction (Bedrock or OpenAI).
    """
    user_prompt = f"Extract entities and relationships from the following text:\n\n{text[:8000]}"
    api_key = _get_openai_key()

    result = llm_extract(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
        api_key=api_key,
    )

    # Filter to valid types
    entities = result.get("entities", [])
    relationships = result.get("relationships", [])
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
        "agent_spec": "https://agent.nhilbert.de/.well-known/agent-spec.json",
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
        "llm": get_provider_info(),
        "tiers": {k: {"node_limit": v} for k, v in TIER_LIMITS.items()},
        "endpoints": [
            {"path": "/api/search", "method": "GET", "auth": True, "description": "Semantic vector search"},
            {"path": "/api/ingest", "method": "POST", "auth": True, "description": "Ingest structured entities and relationships"},
            {"path": "/api/ingest/text", "method": "POST", "auth": True, "description": "Extract knowledge from text via LLM and ingest"},
            {"path": "/api/traverse/<entity>", "method": "GET", "auth": True, "description": "Multi-hop graph traversal from entity"},
            {"path": "/api/beliefs", "method": "GET", "auth": True, "description": "List beliefs by status"},
            {"path": "/api/stats", "method": "GET", "auth": True, "description": "Graph statistics"},
            {"path": "/api/graph", "method": "GET", "auth": True, "description": "Full graph data for visualization"},
            {"path": "/api/graph/expand", "method": "GET", "auth": True, "description": "Expand one node by loading direct neighbors"},
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
            {"path": "/api/projects/<id>/vectors/reindex", "method": "POST", "auth": True, "description": "Rebuild vector index for one project"},
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


@app.route("/.well-known/agent-spec.json", methods=["GET"])
def well_known_agent_spec():
    """Serve AGENT_SPEC.json — machine-readable integration specification."""
    spec_path = os.path.join(os.path.dirname(__file__), "AGENT_SPEC.json")
    if not os.path.exists(spec_path):
        return _error("AGENT_SPEC.json not found", 404)
    with open(spec_path) as f:
        content = f.read()
    return content, 200, {"Content-Type": "application/json; charset=utf-8"}


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
        project_id = re.sub(r"[^a-z0-9-]", "-", name.lower())
        project_id = re.sub(r"-+", "-", project_id).strip("-")
    if not project_id:
        return _error("Could not derive a valid project_id", 400)
    if project_id == "default":
        return _error("Cannot create project with reserved id 'default'", 400)
    # The user-supplied suffix (after 'sub:') must not contain '_' because
    # Qdrant collection names use '_' as the ':' replacement.
    suffix = project_id.split(':', 1)[-1] if ':' in project_id else project_id
    if '_' in suffix:
        return _error("Project name must not contain '_' (underscores)", 400)

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
        dreaming_enabled: bool (optional)
        dreaming_schedule: "manual" | "daily" | "weekly" (optional)
        dreaming_config: dict with replay_walks, replay_hops, consolidation_threshold (optional)
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
            dreaming_enabled=body.get("dreaming_enabled"),
            dreaming_schedule=body.get("dreaming_schedule"),
            dreaming_config=body.get("dreaming_config"),
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


@app.route("/api/projects/<path:project_id>/dreaming", methods=["GET"])
@require_auth
@require_scope("read")
def get_dreaming_config(project_id):
    """Get dreaming configuration for a project.

    Returns: dreaming_enabled, dreaming_schedule, dreaming_config
    """
    adp = adapter
    if adp is None:
        return _error("Adapter not initialized", 503)
    denied = _deny_if_project_forbidden(project_id)
    if denied:
        return denied
    try:
        project = adp.get_project(project_id)
        if not project:
            return _error(f"Project '{project_id}' not found", 404)
        dreaming_config_raw = project.get("dreaming_config")
        if isinstance(dreaming_config_raw, str):
            try:
                dreaming_config_raw = json.loads(dreaming_config_raw)
            except (ValueError, TypeError):
                dreaming_config_raw = {}
        return jsonify({
            "project_id": project_id,
            "dreaming_enabled": project.get("dreaming_enabled", False),
            "dreaming_schedule": project.get("dreaming_schedule", "manual"),
            "dreaming_config": dreaming_config_raw or {
                "replay_walks": 3,
                "replay_hops": 5,
                "consolidation_threshold": 0.75,
            },
        })
    except Exception as exc:
        logger.exception("get_dreaming_config failed for %s", project_id)
        return _error(str(exc))


@app.route("/api/projects/<path:project_id>/dreaming", methods=["PATCH"])
@require_auth
@require_scope("projects")
def update_dreaming_config(project_id):
    """Update dreaming configuration for a project.

    Body (JSON):
        dreaming_enabled: bool (optional)
        dreaming_schedule: "manual" | "daily" | "weekly" (optional)
        dreaming_config: dict with replay_walks, replay_hops, consolidation_threshold (optional)
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
            dreaming_enabled=body.get("dreaming_enabled"),
            dreaming_schedule=body.get("dreaming_schedule"),
            dreaming_config=body.get("dreaming_config"),
        )
        if not result.get("updated"):
            return _error(result.get("error", "Update failed"), 404)
        # Return clean dreaming config
        dreaming_config_val = result.get("dreaming_config")
        if isinstance(dreaming_config_val, str):
            try:
                dreaming_config_val = json.loads(dreaming_config_val)
            except (ValueError, TypeError):
                dreaming_config_val = {}
        return jsonify({
            "project_id": project_id,
            "dreaming_enabled": result.get("dreaming_enabled", False),
            "dreaming_schedule": result.get("dreaming_schedule", "manual"),
            "dreaming_config": dreaming_config_val or {},
        })
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        logger.exception("update_dreaming_config failed for %s", project_id)
        return _error(str(exc))


@app.route("/api/projects/<path:project_id>/vectors/reindex", methods=["POST"])
@require_auth
@require_scope("projects")
def reindex_project_vectors(project_id):
    """Rebuild vector index for a project.

    Requires: Authorization: Bearer <cognito_jwt_token> (when AUTH_REQUIRED=true)

    Body (JSON, optional):
        limit: max number of nodes to reindex (default: 5000, max: 10000)
        cleanup_legacy_ids: delete legacy vector ids (`project:name`,
            `project:node_type:name`) after canonical upsert (default: false)
    """
    adp = adapter
    if adp is None:
        return _error("Adapter not initialized", 503)
    denied = _deny_if_project_forbidden(project_id)
    if denied:
        return denied

    body = request.get_json(silent=True) or {}
    try:
        limit = int(body.get("limit", 5000))
    except (TypeError, ValueError):
        limit = 5000
    limit = max(1, min(limit, MAX_VECTOR_REINDEX_LIMIT))
    cleanup_raw = body.get("cleanup_legacy_ids", False)
    if isinstance(cleanup_raw, str):
        cleanup_legacy_ids = cleanup_raw.strip().lower() in ("true", "1", "yes", "on")
    elif isinstance(cleanup_raw, (int, float)):
        cleanup_legacy_ids = bool(cleanup_raw)
    else:
        cleanup_legacy_ids = bool(cleanup_raw)

    try:
        result = adp.reindex_project_vectors(
            project_id=project_id,
            limit=limit,
            cleanup_legacy_ids=cleanup_legacy_ids,
        )
        status = 207 if result.get("failed", 0) > 0 else 200
        return jsonify(result), status
    except Exception as exc:
        logger.exception("project vector reindex failed for %s", project_id)
        return _error(str(exc))


# ---------------------------------------------------------------------------
# Sample data seeding — TechFlow GmbH demo project
# ---------------------------------------------------------------------------

SAMPLE_PROJECT_ID_SUFFIX = "techflow-demo"
SAMPLE_PROJECT_NAME = "TechFlow GmbH — AI Deployment"

SAMPLE_ENTITIES = [
    {"name": "TechFlow GmbH", "node_type": "Organization",
     "summary": "450-person company in Munich and Berlin, using AI for customer service since January 2026."},
    {"name": "Maria Weber", "node_type": "Person",
     "summary": "CTO of TechFlow GmbH, champion of the AI-First Support initiative."},
    {"name": "AI-First Support", "node_type": "Project",
     "summary": "ChatGPT-based customer service automation at TechFlow GmbH, launched January 2026."},
    {"name": "Betriebsrat TechFlow", "node_type": "Organization",
     "summary": "Works council at TechFlow GmbH, consulted during AI deployment planning."},
    # Beliefs — Phase 1 (initial success)
    {"name": "AI reduced ticket resolution time by 40%", "node_type": "Belief",
     "summary": "Initial deployment of AI agents reduced ticket resolution time by 40%.",
     "confidence": 0.85},
    {"name": "Customer satisfaction improved from 3.2 to 4.1", "node_type": "Belief",
     "summary": "Customer satisfaction scores improved from 3.2 to 4.1 out of 5 after AI deployment.",
     "confidence": 0.82},
    # Beliefs — Phase 2 (contradicting evidence from Q2 review)
    {"name": "Complex escalations increased by 25%", "node_type": "Belief",
     "summary": "While simple tickets improved, complex escalations increased by 25% since AI launch.",
     "confidence": 0.78, "status": "active"},
    {"name": "Support team satisfaction dropped to 2.9", "node_type": "Belief",
     "summary": "Employee satisfaction in the support team dropped from 3.8 to 2.9 — agents feel deskilled.",
     "confidence": 0.80, "status": "active"},
    {"name": "Overall CSAT declined to 3.6 when including complex cases", "node_type": "Belief",
     "summary": "When counting complex tickets, overall CSAT dropped from 4.1 to 3.6.",
     "confidence": 0.75, "status": "active"},
    # Beliefs — Phase 3 (VSM diagnostic)
    {"name": "Dr. Meier", "node_type": "Person",
     "summary": "Organizational consultant hired by TechFlow to assess the AI transition."},
    {"name": "AI deployment optimized S1 but neglected S2 and S3", "node_type": "Belief",
     "summary": "Operations (S1) improved but coordination (S2) and quality control (S3) of human-AI handoff were neglected.",
     "confidence": 0.88, "status": "active"},
]

SAMPLE_RELATIONSHIPS = [
    {"source": "Maria Weber", "target": "TechFlow GmbH",
     "type": "AFFILIATED_WITH", "reason": "CTO", "confidence": 0.95},
    {"source": "Maria Weber", "target": "AI-First Support",
     "type": "PRODUCES", "reason": "Championed the initiative", "confidence": 0.90},
    {"source": "AI-First Support", "target": "TechFlow GmbH",
     "type": "PART_OF", "reason": "Company project", "confidence": 0.95},
    {"source": "Betriebsrat TechFlow", "target": "TechFlow GmbH",
     "type": "PART_OF", "reason": "Works council", "confidence": 0.95},
    {"source": "Betriebsrat TechFlow", "target": "AI-First Support",
     "type": "CONSTRAINS", "reason": "Signed Betriebsvereinbarung on AI use", "confidence": 0.85},
    # Phase 1 beliefs support the project
    {"source": "AI reduced ticket resolution time by 40%", "target": "AI-First Support",
     "type": "SUPPORTS", "reason": "Initial deployment metric", "confidence": 0.85},
    {"source": "Customer satisfaction improved from 3.2 to 4.1", "target": "AI-First Support",
     "type": "SUPPORTS", "reason": "Customer feedback improvement", "confidence": 0.82},
    # Phase 2 contradictions
    {"source": "Complex escalations increased by 25%", "target": "AI reduced ticket resolution time by 40%",
     "type": "CONTRADICTS", "reason": "Resolution time improvement masked escalation increase", "confidence": 0.78},
    {"source": "Overall CSAT declined to 3.6 when including complex cases",
     "target": "Customer satisfaction improved from 3.2 to 4.1",
     "type": "CONTRADICTS", "reason": "Including complex tickets reverses the CSAT improvement", "confidence": 0.75},
    {"source": "Support team satisfaction dropped to 2.9", "target": "AI-First Support",
     "type": "CONSTRAINS", "reason": "Employee deskilling risk", "confidence": 0.80},
    # Phase 3 — VSM diagnostic
    {"source": "Dr. Meier", "target": "TechFlow GmbH",
     "type": "APPLIES", "reason": "Hired as organizational consultant", "confidence": 0.95},
    {"source": "AI deployment optimized S1 but neglected S2 and S3", "target": "AI-First Support",
     "type": "CONSTRAINS", "reason": "VSM diagnostic finding", "confidence": 0.88},
]


@app.route("/api/projects/seed-sample", methods=["POST"])
@require_auth
@require_scope("projects")
def seed_sample_project():
    """Create a demo project pre-loaded with TechFlow GmbH sample data.

    The sample project demonstrates belief tracking, contradiction detection,
    and graph traversal using a realistic consulting scenario.

    Returns the created project metadata and ingestion counts.
    """
    adp = adapter
    if adp is None:
        return _error("Adapter not initialized", 503)

    owner = getattr(request, "user_id", None) or "anonymous"
    project_id = f"{owner}:{SAMPLE_PROJECT_ID_SUFFIX}"

    # Check if sample project already exists
    existing = adp.get_project(project_id)
    if existing:
        return _error(
            f"Sample project already exists ('{project_id}'). "
            "Delete it first if you want to re-create it.",
            409,
        )

    try:
        # 1. Create the project
        adp.create_project(
            project_id=project_id,
            name=SAMPLE_PROJECT_NAME,
            owner=owner,
            description=(
                "Demo project: AI deployment at TechFlow GmbH. "
                "Shows belief tracking, contradiction detection, and VSM diagnostics."
            ),
            tier="free",
        )

        # 2. Write entities
        entities_written = adp.write_entities(
            SAMPLE_ENTITIES,
            source_cycle="sample-seed",
            source_type="sample",
            project_id=project_id,
        )

        # 3. Write relationships
        relationships_written = adp.write_relationships(
            SAMPLE_RELATIONSHIPS,
            source_cycle="sample-seed",
            source_type="sample",
            project_id=project_id,
        )

        return jsonify({
            "project_id": project_id,
            "name": SAMPLE_PROJECT_NAME,
            "entities_written": entities_written,
            "relationships_written": relationships_written,
            "message": "Sample project created with TechFlow GmbH demo data.",
        }), 201

    except ValueError as exc:
        return _error(str(exc), 409)
    except Exception as exc:
        logger.exception("seed_sample_project failed")
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
    """Beliefs list filtered by status and/or knowledge type.

    Requires: Authorization: Bearer <cognito_jwt_token> (when AUTH_REQUIRED=true)

    Query params:
        project: project id (default: "default")
        status: active | uncertain | contradicted | superseded (default: "active")
        knowledge_type: fact | state | rule | belief | memory (optional)
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    project = _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    status = request.args.get("status", "active") or "active"
    knowledge_type = request.args.get("knowledge_type") or None
    valid_statuses = {"active", "uncertain", "contradicted", "superseded", "consolidated"}
    if status not in valid_statuses:
        return _error(
            f"Invalid status '{status}'. Must be one of: {', '.join(sorted(valid_statuses))}",
            400,
        )
    try:
        raw = adapter.get_beliefs(project_id=project, status=status,
                                  knowledge_type=knowledge_type)
        return jsonify([_map_belief(b) for b in raw])
    except Exception as exc:
        logger.exception("beliefs failed for project=%s status=%s", project, status)
        return _error(str(exc))


@app.route("/api/beliefs", methods=["PATCH"])
@require_auth
@require_scope("write")
def update_belief_api():
    """Update a belief's confidence, status, summary, or valid_until.

    Requires: Authorization: Bearer <cognito_jwt_token> (when AUTH_REQUIRED=true)

    JSON body:
        {
            "name": "belief name (required)",
            "project": "project_id (default: 'default')",
            "confidence": 0.0-1.0 (optional),
            "status": "active|uncertain|contradicted|superseded" (optional),
            "summary": "updated summary text" (optional),
            "valid_until": "ISO 8601 datetime" (optional)
        }
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)

    body = request.get_json(silent=True)
    if not body:
        return _error("Request body must be JSON", 400)

    name = (body.get("name") or "").strip()
    if not name:
        return _error("'name' field is required", 400)

    project = body.get("project") or "default"
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied

    actor = getattr(request, "username", None) or getattr(request, "user_id", None) or "api"

    try:
        result = adapter.update_belief(
            name=name,
            project_id=project,
            confidence=body.get("confidence"),
            status=body.get("status"),
            summary=body.get("summary"),
            valid_until=body.get("valid_until"),
            actor=actor,
        )
        if result.get("updated"):
            return jsonify(result)
        else:
            code = 404 if "not found" in result.get("error", "") else 400
            return jsonify(result), code
    except Exception as exc:
        logger.exception("update_belief failed for name=%s project=%s", name, project)
        return _error(str(exc))


@app.route("/api/beliefs/consolidate", methods=["POST"])
@require_auth
@require_scope("write")
def consolidate_beliefs_api():
    """Resolve a contradiction between two beliefs with a user explanation.

    Creates a synthesis belief, marks both originals as 'consolidated',
    removes the CONTRADICTS relationship, adds SUPERSEDES links.

    Requires: Authorization: Bearer <cognito_jwt_token> (when AUTH_REQUIRED=true)

    JSON body:
        {
            "belief_a": "first belief name (required)",
            "belief_b": "second belief name (required)",
            "resolution": "free-text explanation of the real situation (required)",
            "project": "project_id (default: 'default')",
            "new_name": "optional name for the synthesis belief"
        }
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)

    body = request.get_json(silent=True)
    if not body:
        return _error("Request body must be JSON", 400)

    belief_a = (body.get("belief_a") or "").strip()
    belief_b = (body.get("belief_b") or "").strip()
    resolution = (body.get("resolution") or "").strip()

    if not belief_a:
        return _error("'belief_a' field is required", 400)
    if not belief_b:
        return _error("'belief_b' field is required", 400)
    if not resolution:
        return _error("'resolution' field is required", 400)

    project = body.get("project") or "default"
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied

    actor = getattr(request, "username", None) or getattr(request, "user_id", None) or "api"

    result = adapter.consolidate_beliefs(
        belief_a_name=belief_a,
        belief_b_name=belief_b,
        resolution_text=resolution,
        project_id=project,
        new_name=body.get("new_name"),
        actor=actor,
    )
    if result.get("ok"):
        return jsonify(result)
    else:
        code = 404 if "not found" in result.get("error", "") else 400
        return jsonify(result), code


@app.route("/api/contradictions", methods=["GET"])
@require_auth
@require_scope("read")
def contradictions():
    """Contradiction pairs — returns actual CONTRADICTS relationship pairs.

    Each pair contains both beliefs and the relationship reason.

    Query params:
        project: project id (default: "default")
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    project = _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    try:
        pairs = _get_contradiction_pairs(adapter, project)
        return jsonify(pairs)
    except Exception as exc:
        logger.exception("contradictions failed for project=%s", project)
        return _error(str(exc))


@app.route("/api/history", methods=["GET"])
@require_auth
@require_scope("read")
def history():
    """Audit history — chronological trail of all mutations.

    Requires: Authorization: Bearer <token> (when AUTH_REQUIRED=true)

    Query params:
        project: project id (default: from auth context)
        entity: filter by entity name (optional)
        operation_type: filter by type e.g. 'update_belief', 'entity_upsert' (optional)
        since: ISO 8601 timestamp — entries after this time (optional)
        until: ISO 8601 timestamp — entries before this time (optional)
        limit: max entries (default 50, max 200)
        offset: pagination offset (default 0)
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    project = _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied

    entity = request.args.get("entity")
    operation_type = request.args.get("operation_type")
    since = request.args.get("since")
    until = request.args.get("until")

    try:
        limit = min(int(request.args.get("limit", 50)), 200)
    except (ValueError, TypeError):
        limit = 50
    try:
        offset = max(int(request.args.get("offset", 0)), 0)
    except (ValueError, TypeError):
        offset = 0

    try:
        result = adapter.get_history(
            project_id=project,
            entity_name=entity,
            operation_type=operation_type,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
        )
        return jsonify(result)
    except Exception as exc:
        logger.exception("history failed for project=%s", project)
        return _error(str(exc))


@app.route("/api/graph", methods=["GET"])
@require_auth
@require_scope("read")
def graph():
    """Graph data for visualization.

    Query params:
        project: project id (default: "default")
        limit: max nodes/edges to return (default: 500)
        q: optional semantic query. When set, returns query-centered subgraph.
        search_mode: semantic (default) or text
        hops: neighborhood depth for semantic subgraph (default: 1, max: 3)
        top: number of semantic seed hits (default: 12, max: 50)
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

    query = (request.args.get("q") or "").strip()
    search_mode = (request.args.get("search_mode") or "semantic").strip().lower()
    if search_mode not in {"semantic", "text"}:
        return _error("Invalid search_mode. Use 'semantic' or 'text'.", 400)
    try:
        hops = int(request.args.get("hops", 1))
    except (TypeError, ValueError):
        hops = 1
    hops = max(0, min(hops, MAX_GRAPH_HOPS))

    try:
        top = int(request.args.get("top", 12))
    except (TypeError, ValueError):
        top = 12
    top = max(1, min(top, MAX_SEARCH_TOP))

    if query and not _token_has_scope("search"):
        return _error("Token lacks required scope: search", 403)

    try:
        if query:
            if search_mode == "text":
                return jsonify(_get_text_subgraph(adapter, project, query, limit=limit, hops=hops, top=top))
            return jsonify(_get_semantic_subgraph(adapter, project, query, limit=limit, hops=hops, top=top))

        raw_nodes = adapter.query_nodes(node_type=None, project_id=project, limit=limit)
        raw_edges = _get_all_edges(adapter, project, limit=limit)

        nodes = [_map_node_for_graph(n) for n in raw_nodes]
        links = [_map_edge_for_graph(e) for e in raw_edges]

        return jsonify(
            {
                "nodes": nodes,
                "links": links,
                "meta": {
                    "mode": "full",
                    "requested_limit": limit,
                    "returned_nodes": len(nodes),
                    "returned_links": len(links),
                    "truncated": len(nodes) >= limit,
                    "cap_reason": "node_limit" if len(nodes) >= limit else None,
                },
            }
        )
    except Exception as exc:
        logger.exception("graph failed for project=%s", project)
        return _error(str(exc))


@app.route("/api/graph/expand", methods=["GET"])
@require_auth
@require_scope("read")
def graph_expand():
    """Expand a single node with direct neighbors.

    Query params:
        project: project id (default: "default")
        node_id: node identifier (preferred)
        node_name: fallback node name if node_id is unavailable
        node_type: optional fallback node type when matching by name
        limit: max neighbors to return (default/max: 100)
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)

    project = _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied

    node_id = (request.args.get("node_id") or "").strip()
    node_name = (request.args.get("node_name") or "").strip()
    node_type = (request.args.get("node_type") or "").strip()
    if not node_id and not node_name:
        return _error("Query parameter 'node_id' or 'node_name' is required", 400)

    try:
        limit = int(request.args.get("limit", MAX_GRAPH_EXPAND_LIMIT))
    except (TypeError, ValueError):
        limit = MAX_GRAPH_EXPAND_LIMIT
    limit = max(1, min(limit, MAX_GRAPH_EXPAND_LIMIT))

    try:
        return jsonify(
            _get_node_expansion(
                adapter,
                project,
                node_id=node_id,
                node_name=node_name,
                node_type=node_type,
                limit=limit,
            )
        )
    except Exception as exc:
        logger.exception("graph_expand failed for project=%s node_id=%s", project, node_id)
        return _error(str(exc))


@app.route("/api/nodes", methods=["GET"])
@require_auth
@require_scope("read")
def nodes():
    """Query nodes, optionally filtered by type, VSM level, and knowledge type.

    Query params:
        project: project id (default: "default")
        type: node type label, e.g. Belief, Concept, Person (optional)
        vsm_level: VSM system level filter, e.g. S1, S2, S3, S4, S5 (optional)
        knowledge_type: knowledge type filter, e.g. fact, state, rule, belief, memory (optional)
        limit: max results (default: 100)
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    project = _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    node_type = request.args.get("type") or None
    vsm_level = request.args.get("vsm_level") or None
    knowledge_type = request.args.get("knowledge_type") or None
    try:
        limit = int(request.args.get("limit", 100))
    except (TypeError, ValueError):
        limit = 100
    limit = max(1, min(limit, MAX_NODES_LIMIT))

    try:
        results = adapter.query_nodes(
            node_type=node_type, project_id=project, limit=limit,
            vsm_level=vsm_level, knowledge_type=knowledge_type,
        )
        return jsonify(results)
    except Exception as exc:
        logger.exception("nodes failed for project=%s type=%s", project, node_type)
        return _error(str(exc))


@app.route("/api/nodes/expiring", methods=["GET"])
@require_auth
@require_scope("read")
def nodes_expiring():
    """Find nodes with valid_until set that are expiring soon or already expired.

    Query params:
        project: project id (default: "default")
        horizon: days ahead to look (default: 30)
        limit: max results (default: 100)
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    project = _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    try:
        horizon = int(request.args.get("horizon", 30))
    except (TypeError, ValueError):
        horizon = 30
    horizon = max(1, min(horizon, 365))
    try:
        limit = int(request.args.get("limit", 100))
    except (TypeError, ValueError):
        limit = 100
    limit = max(1, min(limit, MAX_NODES_LIMIT))

    try:
        results = adapter.query_expiring(
            project_id=project, horizon_days=horizon, limit=limit,
        )
        return jsonify({"expiring": results, "horizon_days": horizon,
                        "total": len(results)})
    except Exception as exc:
        logger.exception("nodes_expiring failed for project=%s", project)
        return _error(str(exc))


@app.route("/api/nodes/expire", methods=["POST"])
@require_auth
@require_scope("write")
def nodes_expire():
    """Enforce managed forgetting: expire nodes past their valid_until date.

    JSON body (all optional):
        project: project id (default: from JWT)
        dry_run: if true, return what would be expired without changing (default: false)

    Returns list of expired (or would-be-expired) nodes.
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    body = request.get_json(silent=True) or {}
    project = body.get("project") or _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    dry_run = body.get("dry_run", False)
    actor = getattr(request, "username", None) or getattr(request, "user_id", "api")

    try:
        result = adapter.expire_nodes(
            project_id=project, dry_run=dry_run, actor=actor,
        )
        return jsonify(result)
    except Exception as exc:
        logger.exception("nodes_expire failed for project=%s", project)
        return _error(str(exc))


@app.route("/api/nodes/renew", methods=["POST"])
@require_auth
@require_scope("write")
def nodes_renew():
    """Renew a node's validity — extend or reset valid_until.

    JSON body:
        name (required): node name
        project: project id (default: from JWT)
        extend_days: number of days to extend from now
        new_valid_until: explicit ISO datetime for valid_until
        node_type: optional node type to narrow match

    Exactly one of extend_days or new_valid_until must be provided.
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    body = request.get_json(silent=True) or {}
    name = body.get("name")
    if not name or not isinstance(name, str) or not name.strip():
        return _error("'name' is required", 400)
    name = name.strip()

    project = body.get("project") or _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied

    extend_days = body.get("extend_days")
    new_valid_until = body.get("new_valid_until")
    if extend_days is None and new_valid_until is None:
        return _error("Provide 'extend_days' or 'new_valid_until'", 400)

    if extend_days is not None:
        try:
            extend_days = int(extend_days)
            if extend_days < 1 or extend_days > 3650:
                return _error("'extend_days' must be between 1 and 3650", 400)
        except (TypeError, ValueError):
            return _error("'extend_days' must be an integer", 400)

    node_type = body.get("node_type")
    actor = getattr(request, "username", None) or getattr(request, "user_id", "api")

    try:
        result = adapter.renew_node(
            name=name, project_id=project, extend_days=extend_days,
            new_valid_until=new_valid_until, node_type=node_type, actor=actor,
        )
        if result.get("error"):
            return _error(result["error"], 404 if "not found" in result["error"] else 400)
        return jsonify(result)
    except Exception as exc:
        logger.exception("nodes_renew failed for project=%s name=%s", project, name)
        return _error(str(exc))


# --- Certainty Management (SUP-163) ---

@app.route("/api/certainty/decay", methods=["POST"])
@require_auth
@require_scope("write")
def certainty_decay():
    """Apply time-based confidence decay to active beliefs.

    Decay rates vary by knowledge_type. Facts are exempt.
    Use dry_run=true (default) to preview changes.

    JSON body (all optional):
        project: project id (default: from JWT)
        dry_run: if true, return what would change without applying (default: true)
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    body = request.get_json(silent=True) or {}
    project = body.get("project") or _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    dry_run = body.get("dry_run", True)
    actor = getattr(request, "username", None) or getattr(request, "user_id", "api")

    try:
        result = adapter.apply_confidence_decay(
            project_id=project, dry_run=dry_run, actor=actor,
        )
        return jsonify(result)
    except Exception as exc:
        logger.exception("certainty_decay failed for project=%s", project)
        return _error(str(exc))


@app.route("/api/certainty/review", methods=["GET"])
@require_auth
@require_scope("read")
def certainty_review():
    """Get beliefs needing review based on certainty governance rules.

    Returns categorized items: stale, low_confidence, type_mismatch,
    approaching_expiry, unclassified.

    Query params:
        project: project id (default: from JWT)
        limit: max items per category (default: 50)
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    project = _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied
    try:
        limit = int(request.args.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(limit, 200))

    try:
        result = adapter.get_certainty_review_queue(
            project_id=project, limit=limit,
        )
        return jsonify(result)
    except Exception as exc:
        logger.exception("certainty_review failed for project=%s", project)
        return _error(str(exc))


@app.route("/api/certainty/stats", methods=["GET"])
@require_auth
@require_scope("read")
def certainty_stats():
    """Confidence distribution statistics for certainty governance.

    Returns confidence histogram, type-confidence cross-tab,
    staleness distribution, and governance health summary.

    Query params:
        project: project id (default: from JWT)
    """
    if adapter is None:
        return _error("Adapter not initialized", 503)
    project = _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied

    try:
        result = adapter.get_certainty_stats(project_id=project)
        return jsonify(result)
    except Exception as exc:
        logger.exception("certainty_stats failed for project=%s", project)
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

        actor = getattr(request, "username", None) or getattr(request, "user_id", None) or "api"

        if entities:
            entities_written = adapter.write_entities(
                entities,
                source_cycle=source,
                source_type="api",
                project_id=project,
                node_limit=node_limit,
                actor=actor,
            )

        if relationships:
            relationships_written = adapter.write_relationships(
                relationships,
                source_cycle=source,
                source_type="api",
                project_id=project,
                actor=actor,
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

    # Check provider-specific prerequisites
    provider_info = get_provider_info()
    if provider_info["provider"] == "openai" and not _get_openai_key():
        return _error(
            "OPENAI_API_KEY not configured. Set it in .env or environment, "
            "or switch to Bedrock provider (MERKRAUM_LLM_PROVIDER=bedrock).",
            503,
        )

    # Step 1: LLM extraction (provider-agnostic via merkraum_llm)
    try:
        extracted = _llm_extract_text(text)
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
        actor = getattr(request, "username", None) or getattr(request, "user_id", None) or "text_extraction"

        if entities:
            entities_written = adapter.write_entities(
                entities,
                source_cycle=source,
                source_type="text_extraction",
                project_id=project,
                node_limit=node_limit,
                actor=actor,
            )

        if relationships:
            relationships_written = adapter.write_relationships(
                relationships,
                source_cycle=source,
                source_type="text_extraction",
                project_id=project,
                actor=actor,
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
# Dreaming — async graph dreaming with SSE progress streaming
# ---------------------------------------------------------------------------

from merkraum_dreaming import dream as _dream_engine

# In-memory dream job store (same pattern as MCP ingestion jobs)
_dream_jobs: dict[str, dict] = {}
_dream_jobs_lock = threading.Lock()
_dream_queue: queue.Queue | None = None
MAX_DREAM_JOBS = 20


def _ensure_dream_worker():
    """Lazy-create the dreaming background worker thread."""
    global _dream_queue
    if _dream_queue is not None:
        return
    _dream_queue = queue.Queue()

    def _worker():
        while True:
            job_id = _dream_queue.get()
            _run_dream_job(job_id)
            _dream_queue.task_done()

    t = threading.Thread(target=_worker, daemon=True, name="dreaming-worker")
    t.start()


def _run_dream_job(job_id: str):
    """Execute a dream session, storing progress messages for SSE streaming."""
    with _dream_jobs_lock:
        job = _dream_jobs.get(job_id)
        if not job:
            return
        job["status"] = "running"

    adp = adapter
    if not adp:
        with _dream_jobs_lock:
            job["status"] = "failed"
            job["error"] = "Backend not available"
        return

    start = time.time()
    try:
        gen = _dream_engine(
            adp,
            project_id=job["project_id"],
            phases=job.get("phases"),
            replay_hops=job.get("replay_hops", 5),
            replay_walks=job.get("replay_walks", 3),
            consolidation_threshold=job.get("consolidation_threshold", 0.75),
            consolidation_dry_run=job.get("consolidation_dry_run", False),
            seed=job.get("seed"),
        )
        result = None
        try:
            while True:
                progress = next(gen)
                with _dream_jobs_lock:
                    job.setdefault("messages", []).append(progress)
        except StopIteration as e:
            result = e.value

        with _dream_jobs_lock:
            job["status"] = "completed"
            job["result"] = result or {}
            job["duration_ms"] = int((time.time() - start) * 1000)
    except Exception as e:
        logger.error("Dream job %s failed: %s", job_id, e, exc_info=True)
        with _dream_jobs_lock:
            job["status"] = "failed"
            job["error"] = str(e)
            job["duration_ms"] = int((time.time() - start) * 1000)


@app.route("/api/dream", methods=["POST"])
@require_auth
@require_scope("write")
def dream_trigger():
    """Trigger a dream session (async). Returns job_id for status polling / SSE.

    Request body (all optional):
        phases: ["replay", "consolidation", "reflection"]
        replay_hops: int (default 5)
        replay_walks: int (default 3)
        consolidation_threshold: float (default 0.75)
        consolidation_dry_run: bool (default false)
        seed: str (optional starting entity for replay)
    """
    project = _project_id()
    denied = _deny_if_project_forbidden(project)
    if denied:
        return denied

    body = request.get_json(silent=True) or {}

    # Limit concurrent dream jobs per project
    with _dream_jobs_lock:
        active = [j for j in _dream_jobs.values()
                  if j["project_id"] == project and j["status"] in ("queued", "running")]
        if active:
            return _error("A dream session is already running for this project", 409)

        # Evict old jobs
        if len(_dream_jobs) >= MAX_DREAM_JOBS:
            oldest = sorted(_dream_jobs.keys(),
                            key=lambda k: _dream_jobs[k].get("created_at", "")
                            )[:MAX_DREAM_JOBS // 2]
            for k in oldest:
                if _dream_jobs[k]["status"] not in ("queued", "running"):
                    del _dream_jobs[k]

        import uuid
        job_id = str(uuid.uuid4())[:12]
        _dream_jobs[job_id] = {
            "job_id": job_id,
            "project_id": project,
            "status": "queued",
            "created_at": time.time(),
            "messages": [],
            "phases": body.get("phases"),
            "replay_hops": min(body.get("replay_hops", 5), 10),
            "replay_walks": min(body.get("replay_walks", 3), 5),
            "consolidation_threshold": body.get("consolidation_threshold", 0.75),
            "consolidation_dry_run": body.get("consolidation_dry_run", False),
            "seed": body.get("seed"),
        }

    _ensure_dream_worker()
    _dream_queue.put(job_id)

    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.route("/api/dream/status", methods=["GET"])
@require_auth
@require_scope("read")
def dream_status():
    """Check dream session status. Query param: job_id (required).

    Returns status, messages since last_index, and result if complete.
    Query params:
        job_id: str (required)
        since: int (message index to start from, for incremental polling)
    """
    job_id = request.args.get("job_id")
    if not job_id:
        return _error("job_id parameter required", 400)

    with _dream_jobs_lock:
        job = _dream_jobs.get(job_id)
        if not job:
            return _error("Dream job not found", 404)

        project = job["project_id"]
        denied = _deny_if_project_forbidden(project)
        if denied:
            return denied

        since = int(request.args.get("since", 0))
        messages = job.get("messages", [])[since:]

        resp = {
            "job_id": job_id,
            "status": job["status"],
            "messages": messages,
            "message_count": len(job.get("messages", [])),
        }
        if job["status"] == "completed":
            resp["result"] = job.get("result", {})
            resp["duration_ms"] = job.get("duration_ms")
        elif job["status"] == "failed":
            resp["error"] = job.get("error")

    return jsonify(resp)


@app.route("/api/dream/stream", methods=["GET"])
@require_auth
@require_scope("read")
def dream_stream():
    """SSE (Server-Sent Events) stream of dream progress messages.

    Query param: job_id (required).
    Streams messages as they arrive, then sends a 'complete' event.
    """
    from flask import Response

    job_id = request.args.get("job_id")
    if not job_id:
        return _error("job_id parameter required", 400)

    with _dream_jobs_lock:
        job = _dream_jobs.get(job_id)
        if not job:
            return _error("Dream job not found", 404)

        project = job["project_id"]
        denied = _deny_if_project_forbidden(project)
        if denied:
            return denied

    def generate():
        last_index = 0
        while True:
            with _dream_jobs_lock:
                job = _dream_jobs.get(job_id)
                if not job:
                    yield f"data: {json.dumps({'event': 'error', 'detail': 'Job not found'})}\n\n"
                    return

                messages = job.get("messages", [])[last_index:]
                status = job["status"]

            for msg in messages:
                yield f"data: {json.dumps(msg)}\n\n"
                last_index += 1

            if status == "completed":
                with _dream_jobs_lock:
                    result = job.get("result", {})
                yield f"data: {json.dumps({'event': 'complete', 'result': result})}\n\n"
                return
            elif status == "failed":
                with _dream_jobs_lock:
                    error = job.get("error", "Unknown error")
                yield f"data: {json.dumps({'event': 'error', 'detail': error})}\n\n"
                return

            time.sleep(0.5)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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
