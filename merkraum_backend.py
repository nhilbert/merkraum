#!/usr/bin/env python3
"""
Merkraum BackendAdapter — Abstract interface for graph + vector operations.

SUP-93: Enables swapping between different backend configurations:
  - Local Neo4j + Pinecone (current VSG setup)
  - Neo4j Aura + Pinecone managed (production Merkraum)
  - Neo4j + Qdrant (self-hosted alternative)

The adapter abstracts graph and vector operations so the MCP server,
CLI tools, and dreaming engine can work against any backend configuration
without code changes.

v1.0 — Z1134 (2026-03-07)
v1.1 — Z1336 (2026-03-11): Refactored — shared Neo4j graph ops extracted
       into Neo4jBaseAdapter, eliminating ~500 lines of duplication.
v1.2 — Z1340 (2026-03-11): Connection dedup — _connect_neo4j() and
       _load_neo4j_credentials() lifted into Neo4jBaseAdapter.
"""

import os
import json
import urllib.request
import urllib.error
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class NodeLimitExceeded(Exception):
    """Raised when a write would exceed the project's node limit."""

    def __init__(self, current: int, limit: int, attempted: int):
        self.current = current
        self.limit = limit
        self.attempted = attempted
        super().__init__(
            f"Node limit exceeded: {current}/{limit} nodes "
            f"(attempted to add {attempted})"
        )


# Tier-based node limits.  The API layer maps user tiers to these values.
TIER_LIMITS = {
    "free": 100,
    "pro": 1_000,
    "team": 5_000,
    "enterprise": 50_000,
}


# --- Fixed Schema (shared across all backends) ---

NODE_TYPES = [
    "Person", "Organization", "Project", "Concept",
    "Regulation", "Event", "Belief", "Artifact",
    "Interview", "Quote",
]

RELATIONSHIP_TYPES = [
    "SUPPORTS", "CONTRADICTS", "COMPLEMENTS",
    "SUPERSEDES", "EXTENDS", "REFINES",
    "CREATED_BY", "AFFILIATED_WITH", "APPLIES",
    "IMPLEMENTS", "PARTICIPATED_IN", "PRODUCES",
    "REFERENCES", "TEMPORAL",
    "MENTIONS", "PART_OF",
]

SYMMETRIC_TYPES = {"CONTRADICTS", "COMPLEMENTS"}

VSM_LEVELS = ["S1", "S2", "S3", "S4", "S5"]

# Default TTLs by VSM level (days). None = no expiration.
VSM_DEFAULT_TTL_DAYS = {
    "S1": 30,    # Operational — fast-cycling task data
    "S2": 90,    # Coordination — valid while processes exist
    "S3": 180,   # Control — performance metrics, superseded not deleted
    "S4": 365,   # Strategic — environmental models, must be refreshed
    "S5": None,  # Identity — near-permanent, only explicit S5 revision
}


class BackendAdapter(ABC):
    """Abstract interface for Merkraum graph + vector backends."""

    @abstractmethod
    def connect(self):
        """Establish connections to graph and vector backends."""

    @abstractmethod
    def close(self):
        """Close all connections."""

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if both graph and vector backends are reachable."""

    # --- Graph Operations ---

    @abstractmethod
    def write_entities(self, entities: list, source_cycle: str,
                       source_type: str = "extraction",
                       project_id: str = "default",
                       node_limit: Optional[int] = None) -> int:
        """Write entities to the graph. Returns count of entities written.

        Args:
            node_limit: If set, raises NodeLimitExceeded when the project's
                current node count + new entities would exceed this limit.
        """

    @abstractmethod
    def write_relationships(self, relationships: list, source_cycle: str,
                            source_type: str = "extraction",
                            project_id: str = "default") -> int:
        """Write relationships to the graph. Returns count written."""

    @abstractmethod
    def query_nodes(self, node_type: Optional[str] = None,
                    project_id: str = "default",
                    limit: int = 100,
                    vsm_level: Optional[str] = None) -> list:
        """Query nodes, optionally filtered by type, project, and VSM level."""

    @abstractmethod
    def traverse(self, entity_name: str, project_id: str = "default",
                 max_depth: int = 2) -> dict:
        """Multi-hop graph traversal from an entity. Returns nodes and edges."""

    @abstractmethod
    def get_beliefs(self, project_id: str = "default",
                    status: str = "active") -> list:
        """Get beliefs filtered by status (active, uncertain, contradicted, superseded)."""

    @abstractmethod
    def get_stats(self, project_id: str = "default") -> dict:
        """Get node/edge counts by type for a project."""

    @abstractmethod
    def delete_project_data(self, project_id: str) -> dict:
        """Delete all nodes and relationships for a project. Returns counts."""

    @abstractmethod
    def update_belief(self, name: str, project_id: str = "default",
                      confidence: Optional[float] = None,
                      status: Optional[str] = None,
                      summary: Optional[str] = None) -> dict:
        """Update an existing belief's confidence, status, or summary.

        Args:
            name: Belief name (must exist).
            confidence: New confidence score (0.0-1.0). None = no change.
            status: New status — 'active', 'superseded'. None = no change.
            summary: New summary text. None = no change.

        Returns:
            {"updated": bool, "name": str, "changes": dict}
        """

    @abstractmethod
    def update_node(self, name: str, project_id: str = "default",
                    updates: Optional[dict] = None,
                    new_name: Optional[str] = None,
                    node_type: Optional[str] = None,
                    actor: str = "api") -> dict:
        """Update a node and keep history/audit metadata."""

    @abstractmethod
    def delete_node(self, name: str, project_id: str = "default",
                    node_type: Optional[str] = None,
                    actor: str = "api") -> dict:
        """Delete a node and its relationships with history/audit metadata."""

    @abstractmethod
    def add_relationship(self, source: str, target: str, rel_type: str,
                         project_id: str = "default",
                         reason: str = "",
                         confidence: float = 0.7,
                         source_type: Optional[str] = None,
                         target_type: Optional[str] = None,
                         actor: str = "api") -> dict:
        """Create or update a relationship between two existing nodes."""

    @abstractmethod
    def delete_relationship(self, source: str, target: str, rel_type: str,
                            project_id: str = "default",
                            source_type: Optional[str] = None,
                            target_type: Optional[str] = None,
                            actor: str = "api") -> dict:
        """Delete a relationship between two nodes."""

    @abstractmethod
    def merge_nodes(self, keep_name: str, remove_name: str,
                    project_id: str = "default",
                    keep_type: Optional[str] = None,
                    remove_type: Optional[str] = None,
                    keep_node_id: Optional[str] = None,
                    remove_node_id: Optional[str] = None,
                    actor: str = "api") -> dict:
        """Merge remove_name into keep_name with history/audit metadata."""

    @abstractmethod
    def get_usage(self, project_id: str = "default") -> dict:
        """Get usage metrics for a project.

        Returns:
            {"nodes": int, "edges": int}
        """

    @abstractmethod
    def reindex_project_vectors(self, project_id: str = "default",
                                limit: int = 5000,
                                cleanup_legacy_ids: bool = False) -> dict:
        """Rebuild vector entries for project nodes.

        Returns:
            {"project_id": str, "total_nodes": int, "upserted": int,
             "failed": int, "legacy_deleted": int, "limit": int, "truncated": bool}
        """

    # --- Project Management ---

    @abstractmethod
    def create_project(self, project_id: str, name: str,
                       owner: str, description: str = "",
                       tier: str = "free") -> dict:
        """Create a new project with metadata.

        Returns:
            {"created": True, "project_id": str, "name": str, ...}
        Raises ValueError if project_id already exists.
        """

    @abstractmethod
    def get_project(self, project_id: str) -> Optional[dict]:
        """Get project metadata. Returns None if not found."""

    @abstractmethod
    def update_project(self, project_id: str,
                       name: Optional[str] = None,
                       description: Optional[str] = None,
                       tier: Optional[str] = None,
                       dreaming_enabled: Optional[bool] = None,
                       dreaming_schedule: Optional[str] = None,
                       dreaming_config: Optional[dict] = None) -> dict:
        """Update project metadata. Returns updated project dict."""

    @abstractmethod
    def list_projects(self, owner: Optional[str] = None) -> list:
        """List projects, optionally filtered by owner.

        Returns list of project metadata dicts.
        """

    # --- Audit History ---

    @abstractmethod
    def get_history(self, project_id: str = "default",
                    entity_name: Optional[str] = None,
                    operation_type: Optional[str] = None,
                    since: Optional[str] = None,
                    until: Optional[str] = None,
                    limit: int = 50,
                    offset: int = 0) -> dict:
        """Retrieve audit history from OperationLog and HistoryLog.

        Args:
            project_id: Scope to this project.
            entity_name: Filter to a specific entity (matches HistoryLog.entity_name).
            operation_type: Filter by operation type (e.g. 'update_belief', 'entity_upsert').
            since: ISO 8601 timestamp — only entries after this time.
            until: ISO 8601 timestamp — only entries before this time.
            limit: Max entries to return (capped at 200).
            offset: Pagination offset.

        Returns:
            {"entries": [...], "total": int, "limit": int, "offset": int}
        """

    # --- Vector Operations ---

    @abstractmethod
    def vector_search(self, query_text: str, top_k: int = 5,
                      project_id: str = "default",
                      namespace: Optional[str] = None) -> list:
        """Semantic search. Returns list of {id, score, content, metadata}."""

    @abstractmethod
    def vector_upsert(self, vector_id: str, text: str, metadata: dict,
                      project_id: str = "default",
                      namespace: Optional[str] = None) -> bool:
        """Upsert a vector with text embedding. Returns success."""

    @abstractmethod
    def vector_delete(self, vector_id: str, project_id: str = "default",
                      namespace: Optional[str] = None) -> bool:
        """Delete a vector by ID. Returns success."""


class Neo4jBaseAdapter(BackendAdapter):
    """Base adapter with shared Neo4j graph operations and connection logic.

    Subclasses must implement:
      - connect(), close(), is_healthy()
      - vector_search(), vector_upsert()

    Subclasses should call _load_neo4j_credentials() and _connect_neo4j()
    from their connect() method. The Neo4j driver is at self._driver after
    _connect_neo4j().
    """

    _driver = None
    _neo4j_uri = None
    _neo4j_user = None
    _neo4j_password = None

    def _load_neo4j_credentials(self):
        """Load Neo4j credentials from environment if not already set.

        Returns the env dict so subclasses can read vendor-specific keys.
        """
        env = _load_env()
        if not self._neo4j_uri:
            self._neo4j_uri = env.get("NEO4J_URI", "bolt://localhost:7687")
            self._neo4j_user = env.get("NEO4J_USER", "neo4j")
            self._neo4j_password = env.get("NEO4J_PASSWORD", "")
        return env

    def _connect_neo4j(self):
        """Connect to Neo4j and verify connectivity."""
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self._neo4j_uri,
                auth=(self._neo4j_user, self._neo4j_password),
            )
            self._driver.verify_connectivity()
        except Exception as e:
            logger.error("Neo4j connection failed: %s", e)
            raise

    def _operation_id(self) -> str:
        from uuid import uuid4
        return uuid4().hex

    def _vector_id_for_node(self, project_id: str, name: str,
                            node_type: Optional[str] = None,
                            node_id: Optional[str] = None) -> str:
        if node_id:
            return f"{project_id}:{node_id}"
        if node_type:
            return f"{project_id}:{node_type}:{name}"
        return f"{project_id}:{name}"

    def _vector_text_for_node(self, name: str, summary: str) -> str:
        return f"{name}: {summary or ''}".strip()

    def _legacy_vector_ids_for_node(self, project_id: str, name: str,
                                    node_type: str) -> list[str]:
        """Legacy vector ids kept for migration cleanup."""
        return [
            f"{project_id}:{name}",
            f"{project_id}:{node_type}:{name}",
        ]

    def _dedupe_vector_results(self, results: list, project_id: str,
                               top_k: int) -> list:
        """Deduplicate semantic search results and keep the highest score."""
        best_by_key: dict[tuple, dict] = {}
        for item in results:
            score = float(item.get("score") or 0.0)
            metadata = item.get("metadata") or {}
            node_id = (metadata.get("node_id") or "").strip()
            if node_id:
                key = ("node_id", node_id)
            else:
                name = (metadata.get("name") or "").strip().lower()
                node_type = (metadata.get("node_type") or "").strip() or "Concept"
                metadata_project = (metadata.get("project_id") or project_id).strip()
                if name:
                    key = ("fallback", metadata_project, node_type, name)
                else:
                    key = ("id", str(item.get("id", "")))

            existing = best_by_key.get(key)
            if existing is None or score > float(existing.get("score") or 0.0):
                best_by_key[key] = item

        deduped = sorted(
            best_by_key.values(),
            key=lambda x: float(x.get("score") or 0.0),
            reverse=True,
        )
        return deduped[:top_k]

    def _filter_expired_results(self, results: list,
                                project_id: str) -> list:
        """Post-filter vector search results to exclude expired nodes."""
        if not results or not self._driver:
            return results
        names = []
        for r in results:
            meta = r.get("metadata") or {}
            name = (meta.get("name") or "").strip()
            if name:
                names.append(name)
        if not names:
            return results
        expired_names = set()
        try:
            with self._driver.session() as session:
                records = session.run(
                    """
                    MATCH (n {project_id: $pid})
                    WHERE n.name IN $names AND n.expired_at IS NOT NULL
                    RETURN n.name AS name
                    """,
                    pid=project_id,
                    names=names,
                )
                for rec in records:
                    expired_names.add(rec["name"])
        except Exception:
            return results
        if not expired_names:
            return results
        return [r for r in results
                if (r.get("metadata") or {}).get("name", "").strip()
                not in expired_names]

    def _log_operation(self, tx, op_id: str, op_type: str, actor: str,
                       project_id: str, payload: dict,
                       before_state: Optional[dict] = None,
                       after_state: Optional[dict] = None,
                       status: str = "committed"):
        tx.run(
            """
            CREATE (op:OperationLog {
                id: $id,
                type: $type,
                actor: $actor,
                project_id: $project_id,
                payload_json: $payload_json,
                before_json: $before_json,
                after_json: $after_json,
                status: $status,
                created_at: $now
            })
            """,
            id=op_id,
            type=op_type,
            actor=actor,
            project_id=project_id,
            payload_json=json.dumps(payload, ensure_ascii=True),
            before_json=json.dumps(before_state, ensure_ascii=True)
            if before_state is not None else None,
            after_json=json.dumps(after_state, ensure_ascii=True)
            if after_state is not None else None,
            status=status,
            now=datetime.now(timezone.utc).isoformat(),
        )

    def _log_history(self, tx, op_id: str, entity_name: str, project_id: str,
                     action: str, snapshot: dict, actor: str):
        tx.run(
            """
            CREATE (:HistoryLog {
                operation_id: $operation_id,
                entity_name: $entity_name,
                project_id: $project_id,
                action: $action,
                actor: $actor,
                snapshot_json: $snapshot_json,
                created_at: $now
            })
            """,
            operation_id=op_id,
            entity_name=entity_name,
            project_id=project_id,
            action=action,
            actor=actor,
            snapshot_json=json.dumps(snapshot, ensure_ascii=True),
            now=datetime.now(timezone.utc).isoformat(),
        )

    def _ensure_project_node_ids(self, project_id: str):
        """Backfill stable node_id for legacy nodes missing it."""
        with self._driver.session() as session:
            records = session.run(
                """
                MATCH (n {project_id: $pid})
                WHERE n.node_id IS NULL
                  AND any(lbl IN labels(n) WHERE lbl IN $node_types)
                RETURN n.name AS name, labels(n)[0] AS node_type
                """,
                pid=project_id,
                node_types=list(NODE_TYPES),
            )
            for rec in records:
                name = rec.get("name") or ""
                node_type = rec.get("node_type") or "Concept"
                node_id = _string_to_uuid(f"{project_id}:{node_type}:{name}")
                session.run(
                    f"MATCH (n:{node_type} {{project_id: $pid, name: $name}}) "
                    "SET n.node_id = $node_id",
                    pid=project_id,
                    name=name,
                    node_id=node_id,
                )

    def write_entities(self, entities, source_cycle, source_type="extraction",
                       project_id="default", node_limit=None,
                       actor="ingestion"):
        if node_limit is not None and entities:
            usage = self.get_usage(project_id)
            if usage["nodes"] + len(entities) > node_limit:
                raise NodeLimitExceeded(
                    current=usage["nodes"],
                    limit=node_limit,
                    attempted=len(entities),
                )
        now = datetime.now(timezone.utc).isoformat()
        written = 0
        with self._driver.session() as session:
            for ent in entities:
                try:
                    node_type = ent.get("node_type", "Concept")
                    if node_type not in NODE_TYPES:
                        continue
                    node_id = _string_to_uuid(f"{project_id}:{node_type}:{ent['name']}")
                    valid_until = ent.get("valid_until") or None
                    vsm_level = ent.get("vsm_level") or None
                    if vsm_level and vsm_level not in VSM_LEVELS:
                        vsm_level = None
                    # Apply default TTL from VSM level if no explicit valid_until
                    if not valid_until and vsm_level:
                        ttl_days = VSM_DEFAULT_TTL_DAYS.get(vsm_level)
                        if ttl_days is not None:
                            valid_until = (
                                datetime.now(timezone.utc) +
                                timedelta(days=ttl_days)
                            ).isoformat()
                    params = dict(
                        name=ent["name"],
                        summary=ent.get("summary", ""),
                        now=now,
                        source_cycle=source_cycle,
                        source_type=source_type,
                        project_id=project_id,
                        node_id=node_id,
                        valid_until=valid_until,
                        vsm_level=vsm_level,
                    )

                    tx = session.begin_transaction()
                    try:
                        # Capture before-state (None if entity is new)
                        before_rec = tx.run(
                            f"""
                            MATCH (n:{node_type} {{name: $name, project_id: $project_id}})
                            RETURN properties(n) AS props
                            """,
                            name=ent["name"],
                            project_id=project_id,
                        ).single()
                        before_state = dict(before_rec["props"]) if before_rec else None
                        action = "update_entity" if before_state else "create_entity"

                        if node_type == "Belief":
                            params["confidence"] = ent.get("confidence", 0.7)
                            tx.run(
                                """
                                MERGE (n:Belief {name: $name, project_id: $project_id})
                                ON CREATE SET
                                    n.summary = $summary, n.created_at = $now,
                                    n.updated_at = $now, n.source_cycle = $source_cycle,
                                    n.source_type = $source_type, n.active = true,
                                    n.status = 'active', n.confidence = $confidence,
                                    n.project_id = $project_id, n.node_id = $node_id,
                                    n.valid_until = $valid_until,
                                    n.vsm_level = $vsm_level
                                ON MATCH SET
                                    n.summary = $summary, n.updated_at = $now,
                                    n.node_id = coalesce(n.node_id, $node_id),
                                    n.confidence = CASE WHEN $confidence > n.confidence
                                        THEN $confidence ELSE n.confidence END,
                                    n.valid_until = CASE WHEN $valid_until IS NOT NULL
                                        THEN $valid_until ELSE n.valid_until END,
                                    n.vsm_level = CASE WHEN $vsm_level IS NOT NULL
                                        THEN $vsm_level ELSE n.vsm_level END
                                """,
                                **params,
                            )
                        else:
                            tx.run(
                                f"""
                                MERGE (n:{node_type} {{name: $name, project_id: $project_id}})
                                ON CREATE SET
                                    n.summary = $summary, n.created_at = $now,
                                    n.updated_at = $now, n.source_cycle = $source_cycle,
                                    n.source_type = $source_type,
                                    n.project_id = $project_id, n.node_id = $node_id,
                                    n.valid_until = $valid_until,
                                    n.vsm_level = $vsm_level
                                ON MATCH SET
                                    n.summary = CASE WHEN size($summary) > size(coalesce(n.summary, ''))
                                        THEN $summary ELSE n.summary END,
                                    n.node_id = coalesce(n.node_id, $node_id),
                                    n.updated_at = $now,
                                    n.valid_until = CASE WHEN $valid_until IS NOT NULL
                                        THEN $valid_until ELSE n.valid_until END,
                                    n.vsm_level = CASE WHEN $vsm_level IS NOT NULL
                                        THEN $vsm_level ELSE n.vsm_level END
                                """,
                                **params,
                            )

                        # Log the operation
                        op_id = self._operation_id()
                        payload = {
                            "name": ent["name"],
                            "node_type": node_type,
                            "source_cycle": source_cycle,
                            "source_type": source_type,
                        }
                        after_payload = {
                            "name": ent["name"],
                            "node_type": node_type,
                            "summary": ent.get("summary", ""),
                            "valid_until": valid_until,
                            "vsm_level": vsm_level,
                        }
                        if node_type == "Belief":
                            after_payload["confidence"] = ent.get("confidence", 0.7)
                        self._log_history(tx, op_id, ent["name"], project_id,
                                          action,
                                          before_state if before_state else payload,
                                          actor)
                        self._log_operation(tx, op_id, "entity_upsert", actor,
                                            project_id, payload=payload,
                                            before_state=before_state,
                                            after_state=after_payload)
                        tx.commit()
                    except Exception as inner_exc:
                        tx.rollback()
                        raise inner_exc

                    vec_meta = {
                        "name": ent["name"],
                        "node_type": node_type,
                        "source": source_type,
                        "project_id": project_id,
                        "node_id": node_id,
                    }
                    if vsm_level:
                        vec_meta["vsm_level"] = vsm_level
                    vector_ok = self.vector_upsert(
                        self._vector_id_for_node(
                            project_id,
                            ent["name"],
                            node_type=node_type,
                            node_id=node_id,
                        ),
                        self._vector_text_for_node(
                            ent["name"],
                            ent.get("summary", ""),
                        ),
                        vec_meta,
                        project_id=project_id,
                    )
                    if not vector_ok:
                        logger.warning(
                            "Vector upsert failed for entity '%s' (project=%s)",
                            ent.get("name"),
                            project_id,
                        )
                    written += 1
                except Exception as e:
                    logger.warning("Entity write failed for %s: %s", ent.get("name"), e)
        return written

    def write_relationships(self, relationships, source_cycle,
                            source_type="extraction",
                            project_id="default", actor="ingestion"):
        now = datetime.now(timezone.utc).isoformat()
        written = 0
        with self._driver.session() as session:
            for rel in relationships:
                try:
                    rel_type = rel.get("type", "REFERENCES")
                    if rel_type not in RELATIONSHIP_TYPES:
                        continue
                    valid_until = rel.get("valid_until") or None
                    params = dict(
                        source=rel["source"],
                        target=rel["target"],
                        confidence=rel.get("confidence", 0.7),
                        reason=rel.get("reason", ""),
                        source_cycle=source_cycle,
                        source_type=source_type,
                        now=now,
                        project_id=project_id,
                        valid_from=rel.get("valid_from", now),
                        valid_until=valid_until,
                    )

                    tx = session.begin_transaction()
                    try:
                        # Capture before-state of relationship if it exists
                        before_rec = tx.run(
                            f"""
                            MATCH (a {{name: $source, project_id: $project_id}})
                                  -[r:{rel_type}]->
                                  (b {{name: $target, project_id: $project_id}})
                            RETURN properties(r) AS props
                            """,
                            source=rel["source"],
                            target=rel["target"],
                            project_id=project_id,
                        ).single()
                        before_state = dict(before_rec["props"]) if before_rec else None

                        cypher = f"""
                        MATCH (a {{name: $source, project_id: $project_id}})
                        MATCH (b {{name: $target, project_id: $project_id}})
                        MERGE (a)-[r:{rel_type}]->(b)
                        ON CREATE SET
                            r.confidence = $confidence, r.reason = $reason,
                            r.source_cycle = $source_cycle, r.source_type = $source_type,
                            r.created_at = $now, r.active = true,
                            r.weight = $confidence, r.valid_from = $valid_from,
                            r.valid_until = $valid_until, r.expired_at = null
                        ON MATCH SET
                            r.confidence = CASE WHEN $confidence > r.confidence
                                THEN $confidence ELSE r.confidence END,
                            r.updated_at = $now,
                            r.weight = CASE WHEN $confidence > r.weight
                                THEN $confidence ELSE r.weight END
                        """
                        result = tx.run(cypher, **params)
                        summary = result.consume()

                        # Log the operation
                        op_id = self._operation_id()
                        rel_name = f"{rel['source']}->{rel['target']}:{rel_type}"
                        payload = {
                            "source": rel["source"],
                            "target": rel["target"],
                            "type": rel_type,
                            "confidence": rel.get("confidence", 0.7),
                            "reason": rel.get("reason", ""),
                        }
                        action = "update_relationship" if before_state else "create_relationship"

                        created = summary.counters.relationships_created > 0
                        updated = summary.counters.properties_set > 0

                        # Always log — even no-op MERGEs get an audit entry
                        if created or updated:
                            written += 1
                        op_type_detail = "relationship_upsert"
                        if not created and not updated:
                            action = "noop_relationship"
                        self._log_history(tx, op_id, rel_name, project_id,
                                          action,
                                          before_state or payload, actor)
                        self._log_operation(tx, op_id, op_type_detail,
                                            actor, project_id, payload=payload,
                                            before_state=before_state,
                                            after_state=payload)

                        if created and rel_type in SYMMETRIC_TYPES:
                            inverse = f"""
                            MATCH (a {{name: $target, project_id: $project_id}})
                            MATCH (b {{name: $source, project_id: $project_id}})
                            MERGE (a)-[r:{rel_type}]->(b)
                            ON CREATE SET
                                r.confidence = $confidence, r.reason = $reason,
                                r.source_cycle = $source_cycle,
                                r.source_type = $source_type,
                                r.created_at = $now, r.active = true,
                                r.weight = $confidence, r.valid_from = $valid_from
                            """
                            tx.run(inverse, **params)

                        tx.commit()
                    except Exception as inner_exc:
                        tx.rollback()
                        raise inner_exc
                except Exception as e:
                    logger.warning("Relationship write failed %s->%s: %s",
                                   rel.get("source"), rel.get("target"), e)
        return written

    def query_nodes(self, node_type=None, project_id="default", limit=100,
                    vsm_level=None, include_expired=False):
        self._ensure_project_node_ids(project_id)
        results = []
        with self._driver.session() as session:
            vsm_filter = ""
            if vsm_level and vsm_level in VSM_LEVELS:
                vsm_filter = " AND n.vsm_level = $vsm_level"
            expired_filter = "" if include_expired else " AND n.expired_at IS NULL"
            if node_type and node_type in NODE_TYPES:
                cypher = f"""
                MATCH (n:{node_type} {{project_id: $pid}})
                WHERE true{vsm_filter}{expired_filter}
                RETURN n.name AS name, n.summary AS summary,
                       labels(n)[0] AS type, n.created_at AS created,
                       n.node_id AS node_id, n.confidence AS confidence,
                       n.valid_until AS valid_until, n.vsm_level AS vsm_level
                ORDER BY n.updated_at DESC LIMIT $limit
                """
            else:
                cypher = f"""
                MATCH (n {{project_id: $pid}})
                WHERE any(lbl IN labels(n) WHERE lbl IN $node_types){vsm_filter}{expired_filter}
                RETURN n.name AS name, n.summary AS summary,
                       labels(n)[0] AS type, n.created_at AS created,
                       n.node_id AS node_id, n.confidence AS confidence,
                       n.valid_until AS valid_until, n.vsm_level AS vsm_level
                ORDER BY n.updated_at DESC LIMIT $limit
                """
            params = {"pid": project_id, "limit": limit}
            if vsm_level and vsm_level in VSM_LEVELS:
                params["vsm_level"] = vsm_level
            if not (node_type and node_type in NODE_TYPES):
                params["node_types"] = list(NODE_TYPES)
            records = session.run(cypher, **params)
            for rec in records:
                entry = {
                    "name": rec["name"],
                    "summary": rec["summary"],
                    "type": rec["type"],
                    "created": rec["created"],
                    "node_id": rec.get("node_id"),
                    "confidence": rec.get("confidence"),
                }
                if rec.get("valid_until"):
                    entry["valid_until"] = rec["valid_until"]
                if rec.get("vsm_level"):
                    entry["vsm_level"] = rec["vsm_level"]
                results.append(entry)
        return results

    def query_expiring(self, project_id="default", horizon_days=30, limit=100):
        """Find nodes with valid_until set and expiring within horizon_days.

        Also returns already-expired nodes (valid_until < now) that are still active.
        """
        now = datetime.now(timezone.utc).isoformat()
        horizon = (datetime.now(timezone.utc) +
                   timedelta(days=horizon_days)).isoformat()
        results = []
        with self._driver.session() as session:
            cypher = """
            MATCH (n {project_id: $pid})
            WHERE n.valid_until IS NOT NULL
              AND n.valid_until <= $horizon
              AND any(lbl IN labels(n) WHERE lbl IN $node_types)
            RETURN n.name AS name, n.summary AS summary,
                   labels(n)[0] AS type, n.valid_until AS valid_until,
                   n.created_at AS created, n.updated_at AS updated,
                   n.node_id AS node_id, n.confidence AS confidence,
                   n.vsm_level AS vsm_level,
                   CASE WHEN n.valid_until < $now THEN true ELSE false END AS expired
            ORDER BY n.valid_until ASC LIMIT $limit
            """
            records = session.run(
                cypher, pid=project_id, now=now, horizon=horizon,
                limit=limit, node_types=list(NODE_TYPES),
            )
            for rec in records:
                entry = {
                    "name": rec["name"],
                    "summary": rec["summary"],
                    "type": rec["type"],
                    "valid_until": rec["valid_until"],
                    "created": rec["created"],
                    "updated": rec["updated"],
                    "node_id": rec.get("node_id"),
                    "confidence": rec.get("confidence"),
                    "expired": rec["expired"],
                }
                if rec.get("vsm_level"):
                    entry["vsm_level"] = rec["vsm_level"]
                results.append(entry)
        return results

    def expire_nodes(self, project_id="default", dry_run=False, actor="system"):
        """Enforce managed forgetting: mark expired nodes as inactive.

        Finds all nodes where valid_until < now and expired_at is not set.
        Sets expired_at timestamp and active=false (for Beliefs).
        Non-destructive — nodes remain in graph with full audit trail.

        Args:
            project_id: Scope to a specific project.
            dry_run: If True, return what would be expired without changing anything.
            actor: Who triggered the expiration (for audit trail).

        Returns:
            dict with 'expired' list and 'total' count.
        """
        now = datetime.now(timezone.utc).isoformat()
        expired_nodes = []
        op_id = self._operation_id()

        with self._driver.session() as session:
            # Find all nodes past valid_until that haven't been expired yet
            find_cypher = """
            MATCH (n {project_id: $pid})
            WHERE n.valid_until IS NOT NULL
              AND n.valid_until < $now
              AND n.expired_at IS NULL
              AND any(lbl IN labels(n) WHERE lbl IN $node_types)
            RETURN n.name AS name, n.summary AS summary,
                   labels(n)[0] AS type, n.valid_until AS valid_until,
                   n.node_id AS node_id, n.vsm_level AS vsm_level,
                   n.confidence AS confidence, n.status AS status,
                   n.active AS active
            ORDER BY n.valid_until ASC
            LIMIT 500
            """
            records = list(session.run(
                find_cypher, pid=project_id, now=now,
                node_types=list(NODE_TYPES),
            ))

            if dry_run:
                for rec in records:
                    expired_nodes.append({
                        "name": rec["name"],
                        "type": rec["type"],
                        "valid_until": rec["valid_until"],
                        "vsm_level": rec.get("vsm_level"),
                        "summary": rec["summary"],
                    })
                return {"expired": expired_nodes, "total": len(expired_nodes),
                        "dry_run": True}

            # Expire each node in a transaction with audit trail
            for rec in records:
                name = rec["name"]
                node_type = rec["type"]
                tx = session.begin_transaction()
                try:
                    before_state = {
                        "name": name, "type": node_type,
                        "valid_until": rec["valid_until"],
                        "active": rec.get("active"),
                        "status": rec.get("status"),
                    }

                    # Set expired_at; for Beliefs also set active=false, status=expired
                    if node_type == "Belief":
                        tx.run(
                            """
                            MATCH (n:Belief {name: $name, project_id: $pid})
                            SET n.expired_at = $now,
                                n.active = false,
                                n.status = 'expired',
                                n.updated_at = $now
                            """,
                            name=name, pid=project_id, now=now,
                        )
                    else:
                        tx.run(
                            f"""
                            MATCH (n:{node_type} {{name: $name, project_id: $pid}})
                            SET n.expired_at = $now,
                                n.updated_at = $now
                            """,
                            name=name, pid=project_id, now=now,
                        )

                    after_state = {
                        "name": name, "type": node_type,
                        "expired_at": now,
                        "active": False if node_type == "Belief" else rec.get("active"),
                        "status": "expired" if node_type == "Belief" else rec.get("status"),
                    }

                    self._log_history(tx, op_id, name, project_id,
                                      "expire_node", before_state, actor)
                    self._log_operation(tx, op_id, "expire_node", actor,
                                        project_id,
                                        payload={"name": name, "type": node_type,
                                                 "valid_until": rec["valid_until"]},
                                        before_state=before_state,
                                        after_state=after_state)
                    tx.commit()
                    expired_nodes.append({
                        "name": name, "type": node_type,
                        "valid_until": rec["valid_until"],
                        "vsm_level": rec.get("vsm_level"),
                    })
                except Exception:
                    tx.rollback()
                    # Continue with other nodes — don't fail entire batch

        return {"expired": expired_nodes, "total": len(expired_nodes),
                "dry_run": False, "operation_id": op_id}

    def renew_node(self, name, project_id="default", extend_days=None,
                   new_valid_until=None, node_type=None, actor="api"):
        """Extend or reset the valid_until of a node (managed renewal).

        Either extend_days (add N days from now) or new_valid_until (explicit
        ISO date) must be provided. Also clears expired_at if the node was
        previously expired.

        Args:
            name: Node name.
            project_id: Project scope.
            extend_days: Number of days to extend from now.
            new_valid_until: Explicit new ISO datetime for valid_until.
            node_type: Optional node label to narrow match.
            actor: Who triggered the renewal.

        Returns:
            dict with renewal result.
        """
        if extend_days is None and new_valid_until is None:
            return {"renewed": False, "name": name,
                    "error": "Provide extend_days or new_valid_until"}
        if extend_days is not None:
            extend_days = max(1, min(3650, int(extend_days)))
            computed_until = (
                datetime.now(timezone.utc) + timedelta(days=extend_days)
            ).isoformat()
        else:
            computed_until = new_valid_until

        now = datetime.now(timezone.utc).isoformat()
        op_id = self._operation_id()
        label_clause = f":{node_type}" if node_type and node_type in NODE_TYPES else ""

        with self._driver.session() as session:
            tx = session.begin_transaction()
            try:
                before_rec = tx.run(
                    f"""
                    MATCH (n{label_clause} {{name: $name, project_id: $pid}})
                    RETURN n.valid_until AS valid_until, n.expired_at AS expired_at,
                           n.active AS active, n.status AS status,
                           labels(n)[0] AS type
                    """,
                    name=name, pid=project_id,
                ).single()

                if before_rec is None:
                    tx.rollback()
                    return {"renewed": False, "name": name,
                            "error": f"Node '{name}' not found"}

                before_state = dict(before_rec)
                actual_type = before_rec["type"]
                was_expired = before_rec["expired_at"] is not None

                # Update valid_until, clear expired_at, reactivate if Belief
                set_parts = [
                    "n.valid_until = $new_until",
                    "n.expired_at = null",
                    "n.updated_at = $now",
                ]
                if actual_type == "Belief" and was_expired:
                    set_parts.append("n.active = true")
                    set_parts.append("n.status = 'active'")

                tx.run(
                    f"""
                    MATCH (n{label_clause} {{name: $name, project_id: $pid}})
                    SET {', '.join(set_parts)}
                    """,
                    name=name, pid=project_id, new_until=computed_until, now=now,
                )

                after_state = {
                    "valid_until": computed_until,
                    "expired_at": None,
                    "active": True if (actual_type == "Belief" and was_expired)
                    else before_rec.get("active"),
                    "status": "active" if (actual_type == "Belief" and was_expired)
                    else before_rec.get("status"),
                }

                self._log_history(tx, op_id, name, project_id,
                                  "renew_node", before_state, actor)
                self._log_operation(tx, op_id, "renew_node", actor,
                                    project_id,
                                    payload={"name": name,
                                             "extend_days": extend_days,
                                             "new_valid_until": computed_until},
                                    before_state=before_state,
                                    after_state=after_state)
                tx.commit()
                return {"renewed": True, "name": name,
                        "new_valid_until": computed_until,
                        "was_expired": was_expired,
                        "operation_id": op_id}
            except Exception as exc:
                tx.rollback()
                return {"renewed": False, "name": name, "error": str(exc)}

    def traverse(self, entity_name, project_id="default", max_depth=2,
                 include_expired=False):
        nodes = {}
        edges = []
        with self._driver.session() as session:
            expired_clause = "" if include_expired else \
                " AND start.expired_at IS NULL AND ALL(n IN nodes(path) WHERE n.expired_at IS NULL)"
            cypher = """
            MATCH path = (start {name: $name, project_id: $pid})-[*1..""" + str(max_depth) + """]->(end)
            WHERE end.project_id = $pid""" + expired_clause + """
            RETURN path
            LIMIT 100
            """
            records = session.run(cypher, name=entity_name, pid=project_id)
            for rec in records:
                path = rec["path"]
                for node in path.nodes:
                    nid = node.element_id
                    if nid not in nodes:
                        n_entry = {
                            "name": node.get("name", ""),
                            "type": list(node.labels)[0] if node.labels else "Unknown",
                            "summary": node.get("summary", ""),
                        }
                        if node.get("vsm_level"):
                            n_entry["vsm_level"] = node["vsm_level"]
                        nodes[nid] = n_entry
                for rel in path.relationships:
                    edges.append({
                        "source": rel.start_node.get("name", ""),
                        "target": rel.end_node.get("name", ""),
                        "type": rel.type,
                        "confidence": rel.get("confidence", 0),
                    })
        return {"nodes": list(nodes.values()), "edges": edges}

    def get_beliefs(self, project_id="default", status="active"):
        beliefs = []
        with self._driver.session() as session:
            if status == "active":
                cypher = """
                MATCH (b:Belief {project_id: $pid})
                WHERE b.active = true
                  AND (b.status IS NULL OR b.status = 'active')
                RETURN b.name AS name, b.summary AS summary,
                       b.confidence AS confidence, b.source_cycle AS cycle,
                       b.valid_until AS valid_until, b.vsm_level AS vsm_level,
                       b.node_id AS node_id, b.updated_at AS updated_at,
                       b.status AS status
                ORDER BY b.confidence DESC
                """
            else:
                cypher = """
                MATCH (b:Belief {project_id: $pid})
                WHERE b.status = $status
                RETURN b.name AS name, b.summary AS summary,
                       b.confidence AS confidence, b.source_cycle AS cycle,
                       b.valid_until AS valid_until, b.vsm_level AS vsm_level,
                       b.node_id AS node_id, b.updated_at AS updated_at,
                       b.status AS status
                ORDER BY b.updated_at DESC
                """
            records = session.run(cypher, pid=project_id, status=status)
            for rec in records:
                beliefs.append({
                    "name": rec["name"],
                    "summary": rec["summary"],
                    "confidence": rec["confidence"],
                    "cycle": rec["cycle"],
                    "valid_until": rec["valid_until"],
                    "vsm_level": rec["vsm_level"],
                    "node_id": rec["node_id"],
                    "updated_at": rec["updated_at"],
                    "status": rec["status"] or "active",
                })
        return beliefs

    def get_stats(self, project_id="default"):
        stats = {"nodes": {}, "edges": {}, "total_nodes": 0, "total_edges": 0,
                 "vsm_levels": {}}
        with self._driver.session() as session:
            for nt in NODE_TYPES:
                result = session.run(
                    f"MATCH (n:{nt} {{project_id: $pid}}) RETURN count(n) AS c",
                    pid=project_id,
                )
                count = result.single()["c"]
                if count > 0:
                    stats["nodes"][nt] = count
                    stats["total_nodes"] += count
            for rt in RELATIONSHIP_TYPES:
                result = session.run(
                    f"""MATCH (a {{project_id: $pid}})-[r:{rt}]->(b {{project_id: $pid}})
                    RETURN count(r) AS c""",
                    pid=project_id,
                )
                count = result.single()["c"]
                if count > 0:
                    stats["edges"][rt] = count
                    stats["total_edges"] += count
            # VSM level distribution
            result = session.run(
                """MATCH (n {project_id: $pid})
                WHERE n.vsm_level IS NOT NULL
                  AND any(lbl IN labels(n) WHERE lbl IN $node_types)
                RETURN n.vsm_level AS level, count(n) AS c
                ORDER BY n.vsm_level""",
                pid=project_id, node_types=list(NODE_TYPES),
            )
            for rec in result:
                stats["vsm_levels"][rec["level"]] = rec["c"]
        return stats

    def delete_project_data(self, project_id):
        if project_id == "default":
            raise ValueError("Cannot delete the VSG's own knowledge project")
        counts = {"nodes_deleted": 0, "relationships_deleted": 0}
        with self._driver.session() as session:
            result = session.run(
                "MATCH (n {project_id: $pid}) DETACH DELETE n RETURN count(n) AS c",
                pid=project_id,
            )
            counts["nodes_deleted"] = result.single()["c"]
            # Also remove the ProjectMeta node
            session.run(
                "MATCH (pm:ProjectMeta {project_id: $pid}) DELETE pm",
                pid=project_id,
            )
        return counts

    # --- Project Management (SUP-134) ---

    def create_project(self, project_id, name, owner, description="",
                       tier="free"):
        if tier not in TIER_LIMITS:
            raise ValueError(f"Invalid tier: {tier}. Valid: {list(TIER_LIMITS.keys())}")
        now = datetime.now(timezone.utc).isoformat()
        with self._driver.session() as session:
            # Check if project already exists
            existing = session.run(
                "MATCH (pm:ProjectMeta {project_id: $pid}) RETURN pm",
                pid=project_id,
            ).single()
            if existing:
                raise ValueError(f"Project '{project_id}' already exists")
            session.run(
                """
                CREATE (pm:ProjectMeta {
                    project_id: $pid,
                    name: $name,
                    description: $desc,
                    owner: $owner,
                    tier: $tier,
                    created_at: $now,
                    updated_at: $now
                })
                """,
                pid=project_id, name=name, desc=description,
                owner=owner, tier=tier, now=now,
            )
        return {
            "created": True, "project_id": project_id, "name": name,
            "description": description, "owner": owner, "tier": tier,
            "created_at": now, "updated_at": now,
        }

    def get_project(self, project_id):
        with self._driver.session() as session:
            result = session.run(
                "MATCH (pm:ProjectMeta {project_id: $pid}) RETURN pm",
                pid=project_id,
            ).single()
            if not result:
                return None
            props = dict(result["pm"])
            return props

    def update_project(self, project_id, name=None, description=None,
                       tier=None, dreaming_enabled=None,
                       dreaming_schedule=None, dreaming_config=None):
        VALID_SCHEDULES = {"manual", "daily", "weekly"}
        if tier is not None and tier not in TIER_LIMITS:
            raise ValueError(f"Invalid tier: {tier}. Valid: {list(TIER_LIMITS.keys())}")
        if dreaming_schedule is not None and dreaming_schedule not in VALID_SCHEDULES:
            raise ValueError(f"Invalid dreaming_schedule: {dreaming_schedule}. Valid: {sorted(VALID_SCHEDULES)}")
        now = datetime.now(timezone.utc).isoformat()
        set_clauses = ["pm.updated_at = $now"]
        params: dict = {"pid": project_id, "now": now}
        if name is not None:
            set_clauses.append("pm.name = $name")
            params["name"] = name
        if description is not None:
            set_clauses.append("pm.description = $desc")
            params["desc"] = description
        if tier is not None:
            set_clauses.append("pm.tier = $tier")
            params["tier"] = tier
        if dreaming_enabled is not None:
            set_clauses.append("pm.dreaming_enabled = $dreaming_enabled")
            params["dreaming_enabled"] = bool(dreaming_enabled)
        if dreaming_schedule is not None:
            set_clauses.append("pm.dreaming_schedule = $dreaming_schedule")
            params["dreaming_schedule"] = dreaming_schedule
        if dreaming_config is not None:
            import json as _json
            set_clauses.append("pm.dreaming_config = $dreaming_config")
            params["dreaming_config"] = _json.dumps(dreaming_config)

        with self._driver.session() as session:
            result = session.run(
                f"MATCH (pm:ProjectMeta {{project_id: $pid}}) "
                f"SET {', '.join(set_clauses)} "
                f"RETURN pm",
                **params,
            ).single()
            if not result:
                return {"updated": False, "error": f"Project '{project_id}' not found"}
            props = dict(result["pm"])
            # Parse dreaming_config back from JSON string
            if "dreaming_config" in props and isinstance(props["dreaming_config"], str):
                try:
                    import json as _json
                    props["dreaming_config"] = _json.loads(props["dreaming_config"])
                except (ValueError, TypeError):
                    pass
            return {"updated": True, **props}

    def list_projects(self, owner=None):
        with self._driver.session() as session:
            if owner:
                records = session.run(
                    "MATCH (pm:ProjectMeta {owner: $owner}) "
                    "RETURN pm ORDER BY pm.created_at DESC",
                    owner=owner,
                )
            else:
                records = session.run(
                    "MATCH (pm:ProjectMeta) "
                    "RETURN pm ORDER BY pm.created_at DESC",
                )
            return [dict(rec["pm"]) for rec in records]

    def get_usage(self, project_id="default"):
        with self._driver.session() as session:
            node_result = session.run(
                "MATCH (n {project_id: $pid}) RETURN count(n) AS c",
                pid=project_id,
            )
            node_count = node_result.single()["c"]
            edge_result = session.run(
                "MATCH (a {project_id: $pid})-[r]->(b {project_id: $pid}) "
                "RETURN count(r) AS c",
                pid=project_id,
            )
            edge_count = edge_result.single()["c"]
        return {"nodes": node_count, "edges": edge_count}

    def reindex_project_vectors(self, project_id="default", limit=5000,
                                cleanup_legacy_ids=False):
        self._ensure_project_node_ids(project_id)
        safe_limit = max(1, int(limit))
        stats = {
            "project_id": project_id,
            "total_nodes": 0,
            "upserted": 0,
            "failed": 0,
            "legacy_deleted": 0,
            "limit": safe_limit,
            "truncated": False,
        }
        with self._driver.session() as session:
            records = session.run(
                """
                MATCH (n {project_id: $pid})
                WHERE any(lbl IN labels(n) WHERE lbl IN $node_types)
                RETURN n.name AS name,
                       labels(n)[0] AS node_type,
                       n.summary AS summary,
                       n.node_id AS node_id
                ORDER BY n.updated_at DESC, n.name ASC
                LIMIT $limit
                """,
                pid=project_id,
                node_types=list(NODE_TYPES),
                limit=safe_limit,
            )
            for rec in records:
                name = (rec.get("name") or "").strip()
                node_type = rec.get("node_type") or "Concept"
                if not name:
                    continue
                node_id = rec.get("node_id") or _string_to_uuid(
                    f"{project_id}:{node_type}:{name}"
                )
                summary = rec.get("summary") or ""
                stats["total_nodes"] += 1
                ok = self.vector_upsert(
                    self._vector_id_for_node(
                        project_id, name, node_type=node_type, node_id=node_id
                    ),
                    self._vector_text_for_node(name, summary),
                    {
                        "name": name,
                        "node_type": node_type,
                        "source": "vector_reindex",
                        "project_id": project_id,
                    },
                    project_id=project_id,
                )
                if ok:
                    stats["upserted"] += 1
                else:
                    stats["failed"] += 1
                if cleanup_legacy_ids:
                    for legacy_id in self._legacy_vector_ids_for_node(
                        project_id, name, node_type
                    ):
                        if self.vector_delete(legacy_id, project_id=project_id):
                            stats["legacy_deleted"] += 1

            total_result = session.run(
                """
                MATCH (n {project_id: $pid})
                WHERE any(lbl IN labels(n) WHERE lbl IN $node_types)
                RETURN count(n) AS c
                """,
                pid=project_id,
                node_types=list(NODE_TYPES),
            ).single()
            total_nodes_in_project = int(total_result["c"]) if total_result else 0
            stats["truncated"] = total_nodes_in_project > stats["total_nodes"]
        return stats

    def update_belief(self, name, project_id="default", confidence=None,
                      status=None, summary=None, valid_until=None,
                      actor="api"):
        valid_statuses = ("active", "uncertain", "contradicted", "superseded", "consolidated")
        if status is not None and status not in valid_statuses:
            return {"updated": False, "name": name,
                    "error": f"Invalid status: {status}. Valid: {valid_statuses}"}
        if confidence is not None:
            confidence = max(0.0, min(1.0, float(confidence)))

        now = datetime.now(timezone.utc).isoformat()
        changes = {}
        set_clauses = ["b.updated_at = $now"]

        if confidence is not None:
            set_clauses.append("b.confidence = $confidence")
            changes["confidence"] = confidence
        if status is not None:
            set_clauses.append("b.status = $status")
            if status in ("superseded", "consolidated"):
                set_clauses.append("b.active = false")
            elif status in ("active", "uncertain", "contradicted"):
                set_clauses.append("b.active = true")
            changes["status"] = status
        if summary is not None:
            set_clauses.append("b.summary = $summary")
            changes["summary"] = summary
        if valid_until is not None:
            set_clauses.append("b.valid_until = $valid_until")
            changes["valid_until"] = valid_until

        if not changes:
            return {"updated": False, "name": name, "error": "No changes specified"}

        op_id = self._operation_id()
        with self._driver.session() as session:
            tx = session.begin_transaction()
            try:
                # Capture before-state
                before_rec = tx.run(
                    """
                    MATCH (b:Belief {name: $name, project_id: $pid})
                    RETURN b.confidence AS confidence, b.status AS status,
                           b.summary AS summary, b.valid_until AS valid_until,
                           b.active AS active, b.updated_at AS updated_at
                    """,
                    name=name, pid=project_id,
                ).single()

                if before_rec is None:
                    tx.rollback()
                    return {"updated": False, "name": name,
                            "error": f"Belief '{name}' not found in project '{project_id}'"}

                before_state = dict(before_rec)

                # Apply changes
                cypher = f"""
                MATCH (b:Belief {{name: $name, project_id: $pid}})
                SET {', '.join(set_clauses)}
                RETURN b.confidence AS confidence, b.status AS status,
                       b.summary AS summary, b.valid_until AS valid_until,
                       b.active AS active, b.updated_at AS updated_at
                """
                after_rec = tx.run(
                    cypher, name=name, pid=project_id, now=now,
                    confidence=confidence, status=status, summary=summary,
                    valid_until=valid_until,
                ).single()

                after_state = dict(after_rec) if after_rec else None

                # Log operation and history
                self._log_history(tx, op_id, name, project_id,
                                  "update_belief", before_state, actor)
                self._log_operation(tx, op_id, "update_belief", actor,
                                    project_id, payload=changes,
                                    before_state=before_state,
                                    after_state=after_state)
                tx.commit()
                return {"updated": True, "name": name, "changes": changes,
                        "operation_id": op_id}
            except Exception as exc:
                tx.rollback()
                return {"updated": False, "name": name, "error": str(exc)}

    def consolidate_beliefs(self, belief_a_name, belief_b_name,
                            resolution_text, project_id="default",
                            new_name=None, actor="api"):
        """Resolve a contradiction between two beliefs with a user-provided explanation.

        Creates a new synthesis belief from the resolution text, marks both
        originals as 'consolidated', removes the CONTRADICTS relationship,
        and creates SUPERSEDES links from the new belief to both originals.

        Args:
            belief_a_name: Name of first contradicting belief.
            belief_b_name: Name of second contradicting belief.
            resolution_text: User's free-text explanation of the real situation.
            project_id: Project scope.
            new_name: Optional name for synthesis belief. Auto-generated if omitted.
            actor: Who performed the consolidation.

        Returns:
            {"ok": bool, "synthesis": dict, "consolidated": list, "operation_id": str}
        """
        now = datetime.now(timezone.utc).isoformat()
        op_id = self._operation_id()

        with self._driver.session() as session:
            tx = session.begin_transaction()
            try:
                # 1. Find both beliefs
                rec_a = tx.run(
                    "MATCH (b:Belief {name: $name, project_id: $pid}) "
                    "RETURN b.name AS name, b.summary AS summary, "
                    "b.confidence AS confidence, b.status AS status, "
                    "b.source_cycle AS source_cycle, b.node_id AS node_id",
                    name=belief_a_name, pid=project_id,
                ).single()
                rec_b = tx.run(
                    "MATCH (b:Belief {name: $name, project_id: $pid}) "
                    "RETURN b.name AS name, b.summary AS summary, "
                    "b.confidence AS confidence, b.status AS status, "
                    "b.source_cycle AS source_cycle, b.node_id AS node_id",
                    name=belief_b_name, pid=project_id,
                ).single()

                if not rec_a:
                    tx.rollback()
                    return {"ok": False,
                            "error": f"Belief '{belief_a_name}' not found"}
                if not rec_b:
                    tx.rollback()
                    return {"ok": False,
                            "error": f"Belief '{belief_b_name}' not found"}

                before_a = dict(rec_a)
                before_b = dict(rec_b)

                # 2. Verify CONTRADICTS relationship exists (either direction)
                contra = tx.run(
                    "MATCH (a:Belief {name: $a, project_id: $pid})"
                    "-[r:CONTRADICTS]-"
                    "(b:Belief {name: $b, project_id: $pid}) "
                    "RETURN r.reason AS reason",
                    a=belief_a_name, b=belief_b_name, pid=project_id,
                ).single()
                if not contra:
                    tx.rollback()
                    return {"ok": False,
                            "error": f"No CONTRADICTS relationship between "
                                     f"'{belief_a_name}' and '{belief_b_name}'"}

                # 3. Generate synthesis belief name
                synthesis_name = new_name or (
                    f"Consolidated: {belief_a_name[:40]} + {belief_b_name[:40]}"
                )

                # 4. Create synthesis belief node
                synthesis_id = self._operation_id()  # unique node_id
                tx.run(
                    """
                    CREATE (b:Belief {
                        name: $name,
                        summary: $summary,
                        confidence: 0.8,
                        status: 'active',
                        active: true,
                        project_id: $pid,
                        node_id: $nid,
                        created_at: $now,
                        updated_at: $now,
                        source_cycle: $source,
                        consolidated_from: $from_names
                    })
                    """,
                    name=synthesis_name,
                    summary=resolution_text,
                    pid=project_id,
                    nid=synthesis_id,
                    now=now,
                    source=f"consolidation:{op_id}",
                    from_names=f"{belief_a_name} | {belief_b_name}",
                )

                # 5. Mark both originals as consolidated
                for bname in (belief_a_name, belief_b_name):
                    tx.run(
                        "MATCH (b:Belief {name: $name, project_id: $pid}) "
                        "SET b.status = 'consolidated', b.active = false, "
                        "b.updated_at = $now, "
                        "b.consolidated_into = $synth",
                        name=bname, pid=project_id, now=now,
                        synth=synthesis_name,
                    )

                # 6. Remove CONTRADICTS relationship (both directions for symmetric)
                tx.run(
                    "MATCH (a:Belief {name: $a, project_id: $pid})"
                    "-[r:CONTRADICTS]-"
                    "(b:Belief {name: $b, project_id: $pid}) "
                    "DELETE r",
                    a=belief_a_name, b=belief_b_name, pid=project_id,
                )

                # 7. Create SUPERSEDES from synthesis to both originals
                for bname in (belief_a_name, belief_b_name):
                    tx.run(
                        """
                        MATCH (s:Belief {name: $synth, project_id: $pid})
                        MATCH (o:Belief {name: $orig, project_id: $pid})
                        CREATE (s)-[:SUPERSEDES {
                            created_at: $now,
                            reason: 'Consolidation resolution',
                            confidence: 0.9,
                            active: true
                        }]->(o)
                        """,
                        synth=synthesis_name, orig=bname,
                        pid=project_id, now=now,
                    )

                # 8. Audit trail
                payload = {
                    "belief_a": belief_a_name,
                    "belief_b": belief_b_name,
                    "resolution": resolution_text,
                    "synthesis_name": synthesis_name,
                    "original_contradiction_reason": contra["reason"],
                }
                before_state = {
                    "belief_a": before_a,
                    "belief_b": before_b,
                    "relationship": "CONTRADICTS",
                }
                after_state = {
                    "synthesis": synthesis_name,
                    "belief_a_status": "consolidated",
                    "belief_b_status": "consolidated",
                    "relationship": "SUPERSEDES",
                }
                self._log_history(tx, op_id, synthesis_name, project_id,
                                  "consolidate_beliefs", before_state, actor)
                self._log_operation(tx, op_id, "consolidate_beliefs", actor,
                                    project_id, payload=payload,
                                    before_state=before_state,
                                    after_state=after_state)
                tx.commit()

                return {
                    "ok": True,
                    "operation_id": op_id,
                    "synthesis": {
                        "name": synthesis_name,
                        "summary": resolution_text,
                        "confidence": 0.8,
                        "status": "active",
                    },
                    "consolidated": [belief_a_name, belief_b_name],
                }
            except Exception as exc:
                tx.rollback()
                return {"ok": False, "error": str(exc)}

    def add_relationship(self, source, target, rel_type, project_id="default",
                         reason="", confidence=0.7,
                         source_type=None, target_type=None,
                         actor="api"):
        if rel_type not in RELATIONSHIP_TYPES:
            return {"ok": False, "error": f"Invalid relationship type: {rel_type}"}
        now = datetime.now(timezone.utc).isoformat()
        op_id = self._operation_id()
        with self._driver.session() as session:
            tx = session.begin_transaction()
            try:
                source_label = source_type if source_type in NODE_TYPES else None
                target_label = target_type if target_type in NODE_TYPES else None
                source_clause = f":{source_label}" if source_label else ""
                target_clause = f":{target_label}" if target_label else ""
                cypher = f"""
                MATCH (a{source_clause} {{name: $source, project_id: $pid}})
                MATCH (b{target_clause} {{name: $target, project_id: $pid}})
                MERGE (a)-[r:{rel_type}]->(b)
                ON CREATE SET
                    r.created_at = $now,
                    r.updated_at = $now,
                    r.reason = $reason,
                    r.confidence = $confidence,
                    r.active = true
                ON MATCH SET
                    r.updated_at = $now,
                    r.reason = CASE WHEN size($reason) > 0 THEN $reason ELSE r.reason END,
                    r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END
                RETURN a.name AS source, b.name AS target, properties(r) AS rel_props
                """
                rec = tx.run(
                    cypher,
                    source=source,
                    target=target,
                    pid=project_id,
                    now=now,
                    reason=reason,
                    confidence=max(0.0, min(1.0, float(confidence))),
                ).single()
                if not rec:
                    tx.rollback()
                    return {"ok": False, "error": "Source or target node not found"}

                payload = {
                    "source": source,
                    "target": target,
                    "type": rel_type,
                    "reason": reason,
                    "confidence": confidence,
                }
                self._log_history(tx, op_id, f"{source}->{target}:{rel_type}", project_id,
                                  "add_relationship", payload, actor)
                self._log_operation(tx, op_id, "add_relationship", actor, project_id,
                                    payload=payload, after_state=payload)
                tx.commit()
                return {"ok": True, "operation_id": op_id, "relationship": payload}
            except Exception as exc:
                tx.rollback()
                return {"ok": False, "error": str(exc)}

    def delete_relationship(self, source, target, rel_type, project_id="default",
                            source_type=None, target_type=None, actor="api"):
        if rel_type not in RELATIONSHIP_TYPES:
            return {"ok": False, "error": f"Invalid relationship type: {rel_type}"}
        op_id = self._operation_id()
        with self._driver.session() as session:
            tx = session.begin_transaction()
            try:
                source_label = source_type if source_type in NODE_TYPES else None
                target_label = target_type if target_type in NODE_TYPES else None
                source_clause = f":{source_label}" if source_label else ""
                target_clause = f":{target_label}" if target_label else ""
                if rel_type in SYMMETRIC_TYPES:
                    cypher = f"""
                    MATCH (a{source_clause} {{project_id: $pid}})-[r:{rel_type}]-(b{target_clause} {{project_id: $pid}})
                    WHERE ((a.name = $source AND b.name = $target)
                        OR (a.name = $target AND b.name = $source))
                    WITH collect({{source: startNode(r).name, target: endNode(r).name,
                                   type: type(r), props: properties(r)}}) AS rels,
                         collect(r) AS rel_edges
                    FOREACH (x IN rel_edges | DELETE x)
                    RETURN rels
                    """
                else:
                    cypher = f"""
                    MATCH (a{source_clause} {{name: $source, project_id: $pid}})-[r:{rel_type}]->
                          (b{target_clause} {{name: $target, project_id: $pid}})
                    WITH collect({{source: startNode(r).name, target: endNode(r).name,
                                   type: type(r), props: properties(r)}}) AS rels,
                         collect(r) AS rel_edges
                    FOREACH (x IN rel_edges | DELETE x)
                    RETURN rels
                    """
                rec = tx.run(cypher, source=source, target=target, pid=project_id).single()
                rels = rec["rels"] if rec else []
                if not rels:
                    tx.rollback()
                    return {"ok": False, "error": "Relationship not found"}

                payload = {"source": source, "target": target, "type": rel_type}
                self._log_history(tx, op_id, f"{source}->{target}:{rel_type}", project_id,
                                  "delete_relationship", {"deleted": rels}, actor)
                self._log_operation(tx, op_id, "delete_relationship", actor, project_id,
                                    payload=payload, before_state={"relationships": rels})
                tx.commit()
                return {"ok": True, "operation_id": op_id, "deleted": len(rels)}
            except Exception as exc:
                tx.rollback()
                return {"ok": False, "error": str(exc)}

    def delete_node(self, name, project_id="default", node_type=None, actor="api"):
        op_id = self._operation_id()
        vector_deleted = False
        before_state = None
        node_type = node_type if node_type in NODE_TYPES else None
        with self._driver.session() as session:
            tx = session.begin_transaction()
            try:
                label = node_type if node_type in NODE_TYPES else None
                label_clause = f":{label}" if label else ""
                rec = tx.run(
                    f"""
                    MATCH (n{label_clause} {{name: $name, project_id: $pid}})
                    OPTIONAL MATCH (n)-[r]-(m {{project_id: $pid}})
                    RETURN labels(n)[0] AS node_type,
                           properties(n) AS node_props,
                           collect(CASE WHEN r IS NULL THEN NULL ELSE {{
                               source: startNode(r).name,
                               target: endNode(r).name,
                               type: type(r),
                               props: properties(r)
                           }} END) AS rels
                    """,
                    name=name,
                    pid=project_id,
                ).single()
                if not rec or rec["node_props"] is None:
                    tx.rollback()
                    return {"ok": False, "error": f"Node '{name}' not found"}

                node_type = rec["node_type"] or "Concept"
                before_state = {
                    "node": rec["node_props"],
                    "node_type": node_type,
                    "relationships": [x for x in (rec["rels"] or []) if x],
                }

                tx.run(
                    f"MATCH (n{label_clause} {{name: $name, project_id: $pid}}) DETACH DELETE n",
                    name=name,
                    pid=project_id,
                )

                node_id = before_state.get("node", {}).get("node_id")
                vector_id = self._vector_id_for_node(project_id, name, node_type=node_type, node_id=node_id)
                if not self.vector_delete(vector_id, project_id=project_id):
                    tx.rollback()
                    return {"ok": False, "error": "Vector delete failed; graph rollback applied"}
                vector_deleted = True

                self._log_history(tx, op_id, name, project_id, "delete_node", before_state, actor)
                self._log_operation(tx, op_id, "delete_node", actor, project_id,
                                    payload={"name": name}, before_state=before_state)
                tx.commit()
                return {"ok": True, "operation_id": op_id, "deleted": name}
            except Exception as exc:
                tx.rollback()
                if vector_deleted and before_state:
                    node_props = before_state.get("node", {})
                    old_type = before_state.get("node_type", node_type)
                    self.vector_upsert(
                        self._vector_id_for_node(project_id, name, node_type=old_type,
                                                 node_id=node_props.get("node_id")),
                        self._vector_text_for_node(name, node_props.get("summary", "")),
                        {
                            "name": name,
                            "node_type": old_type,
                            "source": "api_rollback",
                            "node_id": node_props.get("node_id"),
                        },
                        project_id=project_id,
                    )
                return {"ok": False, "error": str(exc)}

    def update_node(self, name, project_id="default", updates=None, new_name=None,
                    node_type=None, actor="api"):
        updates = dict(updates or {})
        updates.pop("project_id", None)
        updates.pop("created_at", None)
        op_id = self._operation_id()
        before_state = None
        after_name = name
        node_type = node_type if node_type in NODE_TYPES else None
        vector_upserted = False
        old_vector_deleted = False
        with self._driver.session() as session:
            tx = session.begin_transaction()
            try:
                label = node_type if node_type in NODE_TYPES else None
                label_clause = f":{label}" if label else ""
                before = tx.run(
                    f"""
                    MATCH (n{label_clause} {{name: $name, project_id: $pid}})
                    RETURN labels(n)[0] AS node_type, properties(n) AS node_props
                    """,
                    name=name,
                    pid=project_id,
                ).single()
                if not before or before["node_props"] is None:
                    tx.rollback()
                    return {"ok": False, "error": f"Node '{name}' not found"}

                before_state = {"node_type": before["node_type"], "node": before["node_props"]}

                if new_name:
                    exists = tx.run(
                        f"MATCH (n{label_clause} {{name: $new_name, project_id: $pid}}) RETURN n LIMIT 1",
                        new_name=new_name,
                        pid=project_id,
                    ).single()
                    if exists:
                        tx.rollback()
                        return {"ok": False, "error": f"Node '{new_name}' already exists"}
                    updates["name"] = new_name

                updates["updated_at"] = datetime.now(timezone.utc).isoformat()
                rec = tx.run(
                    f"""
                    MATCH (n{label_clause} {{name: $name, project_id: $pid}})
                    SET n += $updates
                    RETURN labels(n)[0] AS node_type, properties(n) AS node_props
                    """,
                    name=name,
                    pid=project_id,
                    updates=updates,
                ).single()

                after_props = rec["node_props"]
                after_name = after_props.get("name", name)
                node_type = rec["node_type"] or "Concept"
                summary = after_props.get("summary", "")
                after_node_id = after_props.get("node_id")
                before_node_id = before_state.get("node", {}).get("node_id") if before_state else None
                vector_id_new = self._vector_id_for_node(project_id, after_name, node_type=node_type,
                                                        node_id=after_node_id)
                metadata = {
                    "name": after_name,
                    "node_type": node_type,
                    "source": "api_update",
                    "node_id": after_node_id,
                }
                ok = self.vector_upsert(
                    vector_id_new,
                    self._vector_text_for_node(after_name, summary),
                    metadata,
                    project_id=project_id,
                )
                if not ok:
                    tx.rollback()
                    return {"ok": False, "error": "Vector upsert failed; graph rollback applied"}
                vector_upserted = True

                if after_name != name:
                    vector_id_old = self._vector_id_for_node(project_id, name, node_type=node_type,
                                                            node_id=before_node_id)
                    if not self.vector_delete(vector_id_old, project_id=project_id):
                        tx.rollback()
                        return {"ok": False, "error": "Vector rename cleanup failed; graph rollback applied"}
                    old_vector_deleted = True

                after_state = {"node_type": node_type, "node": after_props}
                self._log_history(tx, op_id, after_name, project_id, "update_node", {
                    "before": before_state,
                    "after": after_state,
                }, actor)
                self._log_operation(tx, op_id, "update_node", actor, project_id,
                                    payload={"name": name, "new_name": new_name, "updates": updates},
                                    before_state=before_state,
                                    after_state=after_state)
                tx.commit()
                return {"ok": True, "operation_id": op_id, "node": after_state}
            except Exception as exc:
                tx.rollback()
                if vector_upserted:
                    self.vector_delete(
                        self._vector_id_for_node(project_id, after_name, node_type=node_type,
                                                 node_id=after_node_id),
                        project_id=project_id,
                    )
                if old_vector_deleted and before_state:
                    old_node = before_state.get("node", {})
                    old_type = before_state.get("node_type", "Concept")
                    self.vector_upsert(
                        self._vector_id_for_node(project_id, name, node_type=old_type,
                                                 node_id=old_node.get("node_id")),
                        self._vector_text_for_node(name, old_node.get("summary", "")),
                        {
                            "name": name,
                            "node_type": old_type,
                            "source": "api_rollback",
                            "node_id": old_node.get("node_id"),
                        },
                        project_id=project_id,
                    )
                return {"ok": False, "error": str(exc)}

    def merge_nodes(self, keep_name, remove_name, project_id="default",
                    keep_type=None, remove_type=None,
                    keep_node_id=None, remove_node_id=None,
                    actor="api"):
        # With IDs provided, same-name merges are valid. Without IDs, require type disambiguation.
        if keep_node_id and remove_node_id and keep_node_id == remove_node_id:
            return {"ok": False, "error": "keep_node_id and remove_node_id must differ"}
        if (not keep_node_id and not remove_node_id and keep_name == remove_name and
                (not keep_type or not remove_type or keep_type == remove_type)):
            return {
                "ok": False,
                "error": "keep_name and remove_name must differ unless keep_type/remove_type disambiguate or node IDs are provided",
            }

        op_id = self._operation_id()
        now = datetime.now(timezone.utc).isoformat()
        before_state = None
        resolved_keep_type = keep_type or "Concept"
        resolved_remove_type = remove_type or "Concept"
        keep_vector_upserted = False
        remove_vector_deleted = False
        keep_props = None
        keep_id = keep_node_id
        remove_id = remove_node_id

        self._ensure_project_node_ids(project_id)
        with self._driver.session() as session:
            tx = session.begin_transaction()
            try:
                def _resolve_merge_node(name, node_type, node_id, role):
                    label = node_type if node_type in NODE_TYPES else None
                    clause = f":{label}" if label else ""
                    if node_id:
                        rec = tx.run(
                            f"MATCH (n{clause} {{node_id: $node_id, project_id: $pid}}) "
                            "RETURN labels(n)[0] AS node_type, properties(n) AS node_props",
                            node_id=node_id,
                            pid=project_id,
                        ).single()
                        if not rec:
                            return None, f"{role} node not found for node_id={node_id}"
                        return rec, None

                    rows = tx.run(
                        f"MATCH (n{clause} {{name: $name, project_id: $pid}}) "
                        "RETURN labels(n)[0] AS node_type, properties(n) AS node_props "
                        "LIMIT 2",
                        name=name,
                        pid=project_id,
                    ).data()
                    if not rows:
                        return None, f"{role} node not found"
                    if len(rows) > 1:
                        return None, (
                            f"{role} node lookup is ambiguous for name='{name}'. "
                            "Provide node_id (and optionally node_type)."
                        )
                    return rows[0], None

                keep_rec, keep_err = _resolve_merge_node(keep_name, keep_type, keep_node_id, "keep")
                remove_rec, remove_err = _resolve_merge_node(remove_name, remove_type, remove_node_id, "remove")
                if not keep_rec or not remove_rec:
                    tx.rollback()
                    return {"ok": False, "error": keep_err or remove_err or "Both nodes must exist to merge"}

                resolved_keep_type = keep_rec["node_type"] or "Concept"
                resolved_remove_type = remove_rec["node_type"] or "Concept"
                keep_props = keep_rec["node_props"]
                remove_props = remove_rec["node_props"]
                keep_id = keep_props.get("node_id")
                remove_id = remove_props.get("node_id")
                if not keep_id or not remove_id:
                    tx.rollback()
                    return {"ok": False, "error": "Both nodes must have node_id to merge"}
                if keep_id == remove_id:
                    tx.rollback()
                    return {"ok": False, "error": "Cannot merge a node into itself"}

                before_state = {
                    "keep": keep_props,
                    "remove": remove_props,
                }

                keep_clause_resolved = f":{resolved_keep_type}"
                remove_clause_resolved = f":{resolved_remove_type}"

                for rel_type in RELATIONSHIP_TYPES:
                    tx.run(
                        f"""
                        MATCH (drop{remove_clause_resolved} {{node_id: $remove_id, project_id: $pid}})-[r:{rel_type}]->
                              (dst {{project_id: $pid}})
                        WHERE dst.node_id <> $keep_id
                        MATCH (keep{keep_clause_resolved} {{node_id: $keep_id, project_id: $pid}})
                        MERGE (keep)-[nr:{rel_type}]->(dst)
                        ON CREATE SET nr += properties(r), nr.created_at = coalesce(r.created_at, $now), nr.updated_at = $now
                        ON MATCH SET nr += properties(r), nr.updated_at = $now
                        """,
                        keep_id=keep_id,
                        remove_id=remove_id,
                        pid=project_id,
                        now=now,
                    )
                    tx.run(
                        f"""
                        MATCH (src {{project_id: $pid}})-[r:{rel_type}]->
                              (drop{remove_clause_resolved} {{node_id: $remove_id, project_id: $pid}})
                        WHERE src.node_id <> $keep_id
                        MATCH (keep{keep_clause_resolved} {{node_id: $keep_id, project_id: $pid}})
                        MERGE (src)-[nr:{rel_type}]->(keep)
                        ON CREATE SET nr += properties(r), nr.created_at = coalesce(r.created_at, $now), nr.updated_at = $now
                        ON MATCH SET nr += properties(r), nr.updated_at = $now
                        """,
                        keep_id=keep_id,
                        remove_id=remove_id,
                        pid=project_id,
                        now=now,
                    )

                tx.run(
                    f"""
                    MATCH (keep{keep_clause_resolved} {{node_id: $keep_id, project_id: $pid}})
                    MATCH (drop{remove_clause_resolved} {{node_id: $remove_id, project_id: $pid}})
                    SET keep.summary = CASE
                        WHEN size(coalesce(drop.summary, '')) > size(coalesce(keep.summary, ''))
                        THEN drop.summary ELSE keep.summary END,
                        keep.updated_at = $now
                    """,
                    keep_id=keep_id,
                    remove_id=remove_id,
                    pid=project_id,
                    now=now,
                )

                tx.run(
                    f"MATCH (n{remove_clause_resolved} {{node_id: $remove_id, project_id: $pid}}) DETACH DELETE n",
                    remove_id=remove_id,
                    pid=project_id,
                )

                keep_after = tx.run(
                    f"MATCH (n{keep_clause_resolved} {{node_id: $keep_id, project_id: $pid}}) "
                    "RETURN labels(n)[0] AS node_type, properties(n) AS node_props",
                    keep_id=keep_id,
                    pid=project_id,
                ).single()

                keep_props = keep_after["node_props"]
                resolved_keep_type = keep_after["node_type"] or resolved_keep_type
                keep_name = keep_props.get("name", keep_name)

                vec_ok = self.vector_upsert(
                    self._vector_id_for_node(project_id, keep_name,
                                             node_type=resolved_keep_type,
                                             node_id=keep_props.get("node_id")),
                    self._vector_text_for_node(keep_name, keep_props.get("summary", "")),
                    {
                        "name": keep_name,
                        "node_type": resolved_keep_type,
                        "source": "api_merge",
                        "node_id": keep_props.get("node_id"),
                    },
                    project_id=project_id,
                )
                if not vec_ok:
                    tx.rollback()
                    return {"ok": False, "error": "Vector upsert failed; graph rollback applied"}
                keep_vector_upserted = True

                if not self.vector_delete(
                    self._vector_id_for_node(project_id, remove_name,
                                             node_type=resolved_remove_type,
                                             node_id=before_state.get("remove", {}).get("node_id")),
                    project_id=project_id,
                ):
                    tx.rollback()
                    return {"ok": False, "error": "Vector delete for merged node failed; graph rollback applied"}
                remove_vector_deleted = True

                after_state = {"keep": keep_props, "removed": remove_name}
                self._log_history(tx, op_id, keep_name, project_id, "merge_nodes", {
                    "before": before_state,
                    "after": after_state,
                }, actor)
                self._log_operation(tx, op_id, "merge_nodes", actor, project_id,
                                    payload={
                                        "keep_name": keep_name,
                                        "remove_name": remove_name,
                                        "keep_node_id": keep_id,
                                        "remove_node_id": remove_id,
                                        "keep_type": resolved_keep_type,
                                        "remove_type": resolved_remove_type,
                                    },
                                    before_state=before_state,
                                    after_state=after_state)

                tx.commit()
                return {
                    "ok": True,
                    "operation_id": op_id,
                    "merged_into": keep_name,
                    "removed": remove_name,
                    "keep_node_id": keep_id,
                    "remove_node_id": remove_id,
                }
            except Exception as exc:
                tx.rollback()
                if keep_vector_upserted and before_state:
                    old_keep = before_state.get("keep", {})
                    self.vector_upsert(
                        self._vector_id_for_node(project_id, keep_name,
                                                 node_type=resolved_keep_type,
                                                 node_id=old_keep.get("node_id")),
                        self._vector_text_for_node(keep_name, old_keep.get("summary", "")),
                        {
                            "name": keep_name,
                            "node_type": resolved_keep_type,
                            "source": "api_rollback",
                            "node_id": old_keep.get("node_id"),
                        },
                        project_id=project_id,
                    )
                if remove_vector_deleted and before_state:
                    old_remove = before_state.get("remove", {})
                    self.vector_upsert(
                        self._vector_id_for_node(project_id, remove_name,
                                                 node_type=resolved_remove_type,
                                                 node_id=old_remove.get("node_id")),
                        self._vector_text_for_node(remove_name, old_remove.get("summary", "")),
                        {
                            "name": remove_name,
                            "node_type": resolved_remove_type,
                            "source": "api_rollback",
                            "node_id": old_remove.get("node_id"),
                        },
                        project_id=project_id,
                    )
                return {"ok": False, "error": str(exc)}


    # --- Audit History Implementation ---

    def get_history(self, project_id="default",
                    entity_name=None, operation_type=None,
                    since=None, until=None,
                    limit=50, offset=0):
        """Retrieve audit trail from OperationLog + HistoryLog nodes."""
        limit = min(max(1, limit), 200)
        offset = max(0, offset)

        with self._driver.session() as session:
            # Build dynamic WHERE clauses
            where_clauses = ["op.project_id = $project_id"]
            params: dict = {"project_id": project_id, "limit": limit, "offset": offset}

            if operation_type:
                where_clauses.append("op.type = $operation_type")
                params["operation_type"] = operation_type

            if since:
                where_clauses.append("op.created_at >= $since")
                params["since"] = since

            if until:
                where_clauses.append("op.created_at <= $until")
                params["until"] = until

            where_str = " AND ".join(where_clauses)

            if entity_name:
                # Join with HistoryLog to filter by entity
                count_query = f"""
                    MATCH (op:OperationLog)
                    WHERE {where_str}
                    WITH op
                    MATCH (h:HistoryLog {{operation_id: op.id, project_id: $project_id}})
                    WHERE h.entity_name = $entity_name
                    RETURN count(DISTINCT op) AS total
                """
                params["entity_name"] = entity_name

                data_query = f"""
                    MATCH (op:OperationLog)
                    WHERE {where_str}
                    WITH op
                    MATCH (h:HistoryLog {{operation_id: op.id, project_id: $project_id}})
                    WHERE h.entity_name = $entity_name
                    WITH DISTINCT op, collect(h {{
                        .entity_name, .action, .actor, .snapshot_json, .created_at
                    }}) AS history_entries
                    ORDER BY op.created_at DESC
                    SKIP $offset LIMIT $limit
                    RETURN op {{
                        .id, .type, .actor, .project_id, .payload_json,
                        .before_json, .after_json, .status, .created_at
                    }} AS operation, history_entries
                """
            else:
                # No entity filter — just OperationLog with optional HistoryLog
                count_query = f"""
                    MATCH (op:OperationLog)
                    WHERE {where_str}
                    RETURN count(op) AS total
                """

                data_query = f"""
                    MATCH (op:OperationLog)
                    WHERE {where_str}
                    WITH op ORDER BY op.created_at DESC
                    SKIP $offset LIMIT $limit
                    OPTIONAL MATCH (h:HistoryLog {{operation_id: op.id, project_id: $project_id}})
                    WITH op, collect(
                        CASE WHEN h IS NOT NULL THEN h {{
                            .entity_name, .action, .actor, .snapshot_json, .created_at
                        }} ELSE null END
                    ) AS raw_history
                    WITH op, [x IN raw_history WHERE x IS NOT NULL] AS history_entries
                    RETURN op {{
                        .id, .type, .actor, .project_id, .payload_json,
                        .before_json, .after_json, .status, .created_at
                    }} AS operation, history_entries
                """

            # Execute count
            count_result = session.run(count_query, **params)
            total = count_result.single()["total"]

            # Execute data query
            records = session.run(data_query, **params)
            entries = []
            for rec in records:
                op = dict(rec["operation"])
                # Parse JSON fields for readability
                for json_field in ("payload_json", "before_json", "after_json"):
                    val = op.get(json_field)
                    if val:
                        try:
                            op[json_field.replace("_json", "")] = json.loads(val)
                        except (json.JSONDecodeError, TypeError):
                            op[json_field.replace("_json", "")] = val
                    else:
                        op[json_field.replace("_json", "")] = None
                    del op[json_field]

                # Parse history entry snapshots
                history = []
                for h in rec["history_entries"]:
                    h_dict = dict(h)
                    snap = h_dict.get("snapshot_json")
                    if snap:
                        try:
                            h_dict["snapshot"] = json.loads(snap)
                        except (json.JSONDecodeError, TypeError):
                            h_dict["snapshot"] = snap
                    else:
                        h_dict["snapshot"] = None
                    h_dict.pop("snapshot_json", None)
                    history.append(h_dict)

                entries.append({
                    "operation": op,
                    "history": history,
                })

            return {
                "entries": entries,
                "total": total,
                "limit": limit,
                "offset": offset,
            }


class Neo4jQdrantAdapter(Neo4jBaseAdapter):
    """Concrete adapter for local/open-source deployment: Neo4j CE + Qdrant.

    Uses Qdrant (self-hosted) for vector storage and FastEmbed for local
    embedding generation — no external API calls needed for vector operations.

    v1.0 — Z1142 (2026-03-07)
    v1.1 — Z1336: Inherits shared graph ops from Neo4jBaseAdapter.
    """

    DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, neo4j_uri=None, neo4j_user=None, neo4j_password=None,
                 qdrant_url=None, qdrant_api_key=None,
                 embed_model=None):
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._qdrant_url = qdrant_url or "http://localhost:6333"
        self._qdrant_api_key = qdrant_api_key
        self._embed_model_name = embed_model or self.DEFAULT_EMBED_MODEL
        self._driver = None
        self._qdrant = None
        self._embedder = None

    def connect(self):
        self._load_credentials()
        self._connect_neo4j()
        self._connect_qdrant()
        self._init_embedder()

    def _load_credentials(self):
        env = self._load_neo4j_credentials()
        self._qdrant_url = env.get("QDRANT_URL", self._qdrant_url)
        self._qdrant_api_key = env.get("QDRANT_API_KEY", self._qdrant_api_key)

    def _connect_qdrant(self):
        try:
            from qdrant_client import QdrantClient
            kwargs = {"url": self._qdrant_url, "timeout": 15}
            if self._qdrant_api_key:
                kwargs["api_key"] = self._qdrant_api_key
            self._qdrant = QdrantClient(**kwargs)
            self._qdrant.get_collections()
        except Exception as e:
            logger.error("Qdrant connection failed: %s", e)
            raise

    def _init_embedder(self):
        try:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding(model_name=self._embed_model_name)
            logger.info("FastEmbed initialized: %s", self._embed_model_name)
        except ImportError:
            logger.warning("fastembed not installed — vector operations will fail. "
                           "Install with: pip install fastembed")
        except Exception as e:
            logger.warning("FastEmbed init failed: %s", e)

    def _get_collection_name(self, project_id, namespace=None):
        # Qdrant collection names cannot contain ':'
        safe_id = project_id.replace(':', '_')
        if namespace:
            return f"{safe_id}_{namespace}"
        return safe_id

    def _ensure_collection(self, collection_name, vector_size=384):
        from qdrant_client import models
        try:
            self._qdrant.get_collection(collection_name)
        except Exception:
            self._qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", collection_name)

    def _embed_text(self, text):
        if not self._embedder:
            logger.warning("No embedder available")
            return None
        try:
            embeddings = list(self._embedder.embed([text]))
            return embeddings[0].tolist()
        except Exception as e:
            logger.warning("Embedding failed: %s", e)
            return None

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None
        if self._qdrant:
            self._qdrant.close()
            self._qdrant = None
        self._embedder = None

    def is_healthy(self) -> bool:
        neo4j_ok = False
        qdrant_ok = False
        try:
            if self._driver:
                self._driver.verify_connectivity()
                neo4j_ok = True
        except Exception:
            pass
        try:
            if self._qdrant:
                self._qdrant.get_collections()
                qdrant_ok = True
        except Exception:
            pass
        return neo4j_ok and qdrant_ok

    # --- Vector Operations (Qdrant-specific) ---

    def vector_search(self, query_text, top_k=5, project_id="default",
                      namespace=None):
        if not self._qdrant:
            return []
        collection = self._get_collection_name(project_id, namespace)
        try:
            from qdrant_client import models
            embedding = self._embed_text(query_text)
            if not embedding:
                return []
            self._ensure_collection(collection, vector_size=len(embedding))
            effective_top_k = max(top_k, min(top_k * 3, 100))
            project_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="project_id",
                        match=models.MatchValue(value=project_id),
                    )
                ]
            )
            hits = self._qdrant.query_points(
                collection_name=collection,
                query=embedding,
                query_filter=project_filter,
                limit=effective_top_k,
                with_payload=True,
            ).points
            results = []
            for hit in hits:
                payload = hit.payload or {}
                results.append({
                    "id": str(hit.id),
                    "score": hit.score,
                    "content": payload.get("content", ""),
                    "metadata": payload,
                })
            deduped = self._dedupe_vector_results(results, project_id=project_id, top_k=top_k)
            return self._filter_expired_results(deduped, project_id=project_id)
        except Exception as e:
            logger.warning("Qdrant search failed: %s", e)
            return []

    def vector_upsert(self, vector_id, text, metadata,
                      project_id="default", namespace=None):
        if not self._qdrant:
            return False
        collection = self._get_collection_name(project_id, namespace)
        try:
            from qdrant_client import models
            embedding = self._embed_text(text)
            if not embedding:
                return False
            self._ensure_collection(collection, vector_size=len(embedding))
            metadata["project_id"] = project_id
            metadata["content"] = text
            point_id = _string_to_uuid(vector_id)
            self._qdrant.upsert(
                collection_name=collection,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=metadata,
                    )
                ],
            )
            return True
        except Exception as e:
            logger.warning("Qdrant upsert failed: %s", e)
            return False

    def vector_delete(self, vector_id, project_id="default", namespace=None):
        if not self._qdrant:
            return False
        collection = self._get_collection_name(project_id, namespace)
        try:
            from qdrant_client import models
            point_id = _string_to_uuid(vector_id)
            self._qdrant.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=[point_id]),
            )
            return True
        except Exception as e:
            logger.warning("Qdrant delete failed: %s", e)
            return False


class Neo4jPineconeAdapter(Neo4jBaseAdapter):
    """Concrete adapter wrapping Neo4j + Pinecone (managed/cloud).

    This is the reference implementation for production Merkraum deployments
    using Pinecone's managed vector service.

    v1.0 — Z1134 (2026-03-07)
    v1.1 — Z1336: Inherits shared graph ops from Neo4jBaseAdapter.
    """

    def __init__(self, neo4j_uri=None, neo4j_user=None, neo4j_password=None,
                 pinecone_api_key=None, pinecone_index="vsg-memory"):
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._pinecone_api_key = pinecone_api_key
        self._pinecone_index = pinecone_index
        self._pinecone_host = None
        self._driver = None

    def connect(self):
        self._load_credentials()
        self._connect_neo4j()
        self._resolve_pinecone_host()

    def _load_credentials(self):
        env = self._load_neo4j_credentials()
        if not self._pinecone_api_key:
            self._pinecone_api_key = env.get("PINECONE_API_KEY", "")

    def _resolve_pinecone_host(self):
        if not self._pinecone_api_key:
            return
        try:
            headers = {
                "Api-Key": self._pinecone_api_key,
                "X-Pinecone-API-Version": "2025-04",
            }
            req = urllib.request.Request(
                "https://api.pinecone.io/indexes", headers=headers
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                indexes = json.loads(resp.read()).get("indexes", [])
            for idx in indexes:
                if idx.get("name") == self._pinecone_index:
                    self._pinecone_host = idx.get("host")
                    break
        except Exception as e:
            logger.warning("Pinecone host resolution failed: %s", e)

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    def is_healthy(self) -> bool:
        neo4j_ok = False
        pinecone_ok = False
        try:
            if self._driver:
                self._driver.verify_connectivity()
                neo4j_ok = True
        except Exception:
            pass
        pinecone_ok = self._pinecone_host is not None
        return neo4j_ok and pinecone_ok

    # --- Vector Operations (Pinecone-specific) ---

    def vector_search(self, query_text, top_k=5, project_id="default",
                      namespace=None):
        if not self._pinecone_host or not self._pinecone_api_key:
            return []
        ns = namespace or "knowledge"
        try:
            embedding = self._embed_text(query_text)
            if not embedding:
                return []
            effective_top_k = max(top_k, min(top_k * 3, 100))
            headers = {
                "Api-Key": self._pinecone_api_key,
                "Content-Type": "application/json",
            }
            body = json.dumps({
                "vector": embedding,
                "topK": effective_top_k,
                "namespace": ns,
                "filter": {
                    "project_id": {"$eq": project_id},
                },
                "includeMetadata": True,
            })
            req = urllib.request.Request(
                f"https://{self._pinecone_host}/query",
                data=body.encode(),
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            results = []
            for m in data.get("matches", []):
                results.append({
                    "id": m.get("id", ""),
                    "score": m.get("score", 0),
                    "content": m.get("metadata", {}).get("content", ""),
                    "metadata": m.get("metadata", {}),
                })
            deduped = self._dedupe_vector_results(results, project_id=project_id, top_k=top_k)
            return self._filter_expired_results(deduped, project_id=project_id)
        except Exception as e:
            logger.warning("Pinecone search failed: %s", e)
            return []

    def vector_upsert(self, vector_id, text, metadata,
                      project_id="default", namespace=None):
        if not self._pinecone_host or not self._pinecone_api_key:
            return False
        ns = namespace or "knowledge"
        try:
            embedding = self._embed_text(text)
            if not embedding:
                return False
            metadata["project_id"] = project_id
            headers = {
                "Api-Key": self._pinecone_api_key,
                "Content-Type": "application/json",
            }
            body = json.dumps({
                "vectors": [{
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata,
                }],
                "namespace": ns,
            })
            req = urllib.request.Request(
                f"https://{self._pinecone_host}/vectors/upsert",
                data=body.encode(),
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                json.loads(resp.read())
            return True
        except Exception as e:
            logger.warning("Pinecone upsert failed: %s", e)
            return False

    def vector_delete(self, vector_id, project_id="default", namespace=None):
        if not self._pinecone_host or not self._pinecone_api_key:
            return False
        ns = namespace or "knowledge"
        try:
            headers = {
                "Api-Key": self._pinecone_api_key,
                "Content-Type": "application/json",
            }
            body = json.dumps({
                "ids": [vector_id],
                "namespace": ns,
            })
            req = urllib.request.Request(
                f"https://{self._pinecone_host}/vectors/delete",
                data=body.encode(),
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                json.loads(resp.read())
            return True
        except Exception as e:
            logger.warning("Pinecone delete failed: %s", e)
            return False

    def _embed_text(self, text):
        """Embed text using Pinecone's inference API (llama-text-embed-v2)."""
        try:
            headers = {
                "Api-Key": self._pinecone_api_key,
                "X-Pinecone-API-Version": "2025-04",
                "Content-Type": "application/json",
            }
            body = json.dumps({
                "model": "llama-text-embed-v2",
                "inputs": [{"text": text}],
                "parameters": {"input_type": "query", "truncate": "END"},
            })
            req = urllib.request.Request(
                "https://api.pinecone.io/embed",
                data=body.encode(),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            return data["data"][0]["values"]
        except Exception as e:
            logger.warning("Embedding failed: %s", e)
            return None


# --- Factory ---

def create_adapter(backend_type=None, **kwargs) -> BackendAdapter:
    """Factory function to create the appropriate backend adapter.

    Args:
        backend_type: One of "neo4j_pinecone", "neo4j_qdrant".
            If None, auto-detects from MERKRAUM_BACKEND env var,
            defaulting to "neo4j_pinecone".
        **kwargs: Backend-specific configuration.

    Returns:
        BackendAdapter instance (not yet connected — call .connect()).
    """
    if backend_type is None:
        backend_type = os.environ.get("MERKRAUM_BACKEND", "neo4j_pinecone")
    if backend_type == "neo4j_pinecone":
        return Neo4jPineconeAdapter(**kwargs)
    if backend_type == "neo4j_qdrant":
        return Neo4jQdrantAdapter(**kwargs)
    raise ValueError(f"Unknown backend type: {backend_type}. "
                     f"Valid: neo4j_pinecone, neo4j_qdrant")


# --- Utility ---

def _string_to_uuid(s: str) -> str:
    """Convert a string to a deterministic UUID (v5, DNS namespace)."""
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


def _load_env():
    """Load .env file from script directory."""
    env = dict(os.environ)
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    env[key.strip()] = val.strip().strip('"').strip("'")
    return env


# --- CLI (testing) ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merkraum BackendAdapter CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("test", help="Test connectivity")
    sub.add_parser("stats", help="Show graph stats")

    sp_query = sub.add_parser("query", help="Query nodes")
    sp_query.add_argument("--type", help="Node type filter")
    sp_query.add_argument("--project", default="default")
    sp_query.add_argument("--limit", type=int, default=20)

    sp_search = sub.add_parser("search", help="Vector search")
    sp_search.add_argument("query", help="Search query text")
    sp_search.add_argument("--top", type=int, default=5)
    sp_search.add_argument("--project", default="default")

    sp_traverse = sub.add_parser("traverse", help="Graph traversal")
    sp_traverse.add_argument("entity", help="Entity name to traverse from")
    sp_traverse.add_argument("--project", default="default")
    sp_traverse.add_argument("--depth", type=int, default=2)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        exit(0)

    adapter = create_adapter()
    adapter.connect()

    try:
        if args.command == "test":
            healthy = adapter.is_healthy()
            print(f"Backend health: {'OK' if healthy else 'DEGRADED'}")
            if adapter._driver:
                print("  Neo4j: connected")
            else:
                print("  Neo4j: FAILED")
            adapter_type = type(adapter).__name__
            if adapter_type == "Neo4jPineconeAdapter":
                if adapter._pinecone_host:
                    print(f"  Pinecone: {adapter._pinecone_host}")
                else:
                    print("  Pinecone: NOT RESOLVED")
            elif adapter_type == "Neo4jQdrantAdapter":
                if adapter._qdrant:
                    print("  Qdrant: connected")
                else:
                    print("  Qdrant: NOT CONNECTED")

        elif args.command == "stats":
            stats = adapter.get_stats()
            print(f"Nodes: {stats['total_nodes']}")
            for nt, c in sorted(stats["nodes"].items()):
                print(f"  {nt}: {c}")
            print(f"Edges: {stats['total_edges']}")
            for rt, c in sorted(stats["edges"].items()):
                print(f"  {rt}: {c}")

        elif args.command == "query":
            nodes = adapter.query_nodes(
                node_type=args.type, project_id=args.project, limit=args.limit
            )
            for n in nodes:
                print(f"  [{n['type']}] {n['name']}: {(n['summary'] or '')[:80]}")
            print(f"Total: {len(nodes)}")

        elif args.command == "search":
            results = adapter.vector_search(
                args.query, top_k=args.top, project_id=args.project
            )
            for r in results:
                print(f"  [{r['score']:.3f}] {r['content'][:100]}")
            print(f"Total: {len(results)}")

        elif args.command == "traverse":
            data = adapter.traverse(
                args.entity, project_id=args.project, max_depth=args.depth
            )
            print(f"Nodes: {len(data['nodes'])}")
            for n in data["nodes"][:10]:
                print(f"  [{n['type']}] {n['name']}")
            print(f"Edges: {len(data['edges'])}")
            for e in data["edges"][:10]:
                print(f"  {e['source']} -[{e['type']}]-> {e['target']}")
    finally:
        adapter.close()
