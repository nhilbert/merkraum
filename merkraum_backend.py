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
from datetime import datetime, timezone
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
                    limit: int = 100) -> list:
        """Query nodes, optionally filtered by type and project."""

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
    def get_usage(self, project_id: str = "default") -> dict:
        """Get usage metrics for a project.

        Returns:
            {"nodes": int, "edges": int}
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

    def write_entities(self, entities, source_cycle, source_type="extraction",
                       project_id="default", node_limit=None):
        if node_limit is not None and entities:
            usage = self.get_usage(project_id)
            if usage["nodes"] + len(entities) > node_limit:
                raise NodeLimitExceeded(
                    current=usage["nodes"],
                    limit=node_limit,
                    attempted=len(entities),
                )
        now = datetime.now().isoformat()
        written = 0
        with self._driver.session() as session:
            for ent in entities:
                try:
                    node_type = ent.get("node_type", "Concept")
                    if node_type not in NODE_TYPES:
                        continue
                    params = dict(
                        name=ent["name"],
                        summary=ent.get("summary", ""),
                        now=now,
                        source_cycle=source_cycle,
                        source_type=source_type,
                        project_id=project_id,
                    )
                    if node_type == "Belief":
                        params["confidence"] = ent.get("confidence", 0.7)
                        session.run(
                            f"""
                            MERGE (n:Belief {{name: $name, project_id: $project_id}})
                            ON CREATE SET
                                n.summary = $summary, n.created_at = $now,
                                n.updated_at = $now, n.source_cycle = $source_cycle,
                                n.source_type = $source_type, n.active = true,
                                n.status = 'active', n.confidence = $confidence,
                                n.project_id = $project_id
                            ON MATCH SET
                                n.summary = $summary, n.updated_at = $now,
                                n.confidence = CASE WHEN $confidence > n.confidence
                                    THEN $confidence ELSE n.confidence END
                            """,
                            **params,
                        )
                    else:
                        session.run(
                            f"""
                            MERGE (n:{node_type} {{name: $name, project_id: $project_id}})
                            ON CREATE SET
                                n.summary = $summary, n.created_at = $now,
                                n.updated_at = $now, n.source_cycle = $source_cycle,
                                n.source_type = $source_type,
                                n.project_id = $project_id
                            ON MATCH SET
                                n.summary = CASE WHEN size($summary) > size(coalesce(n.summary, ''))
                                    THEN $summary ELSE n.summary END,
                                n.updated_at = $now
                            """,
                            **params,
                        )
                    written += 1
                except Exception as e:
                    logger.warning("Entity write failed for %s: %s", ent.get("name"), e)
        return written

    def write_relationships(self, relationships, source_cycle,
                            source_type="extraction",
                            project_id="default"):
        now = datetime.now().isoformat()
        written = 0
        with self._driver.session() as session:
            for rel in relationships:
                try:
                    rel_type = rel.get("type", "REFERENCES")
                    if rel_type not in RELATIONSHIP_TYPES:
                        continue
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
                    )
                    cypher = f"""
                    MATCH (a {{name: $source, project_id: $project_id}})
                    MATCH (b {{name: $target, project_id: $project_id}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    ON CREATE SET
                        r.confidence = $confidence, r.reason = $reason,
                        r.source_cycle = $source_cycle, r.source_type = $source_type,
                        r.created_at = $now, r.active = true,
                        r.weight = $confidence, r.valid_from = $valid_from,
                        r.valid_until = null, r.expired_at = null
                    ON MATCH SET
                        r.confidence = CASE WHEN $confidence > r.confidence
                            THEN $confidence ELSE r.confidence END,
                        r.updated_at = $now,
                        r.weight = CASE WHEN $confidence > r.weight
                            THEN $confidence ELSE r.weight END
                    """
                    result = session.run(cypher, **params)
                    summary = result.consume()
                    if summary.counters.relationships_created > 0:
                        written += 1
                        if rel_type in SYMMETRIC_TYPES:
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
                            session.run(inverse, **params)
                    elif summary.counters.properties_set > 0:
                        written += 1
                except Exception as e:
                    logger.warning("Relationship write failed %s->%s: %s",
                                   rel.get("source"), rel.get("target"), e)
        return written

    def query_nodes(self, node_type=None, project_id="default", limit=100):
        results = []
        with self._driver.session() as session:
            if node_type and node_type in NODE_TYPES:
                cypher = f"""
                MATCH (n:{node_type} {{project_id: $pid}})
                RETURN n.name AS name, n.summary AS summary,
                       labels(n)[0] AS type, n.created_at AS created
                ORDER BY n.updated_at DESC LIMIT $limit
                """
            else:
                cypher = """
                MATCH (n {project_id: $pid})
                RETURN n.name AS name, n.summary AS summary,
                       labels(n)[0] AS type, n.created_at AS created
                ORDER BY n.updated_at DESC LIMIT $limit
                """
            records = session.run(cypher, pid=project_id, limit=limit)
            for rec in records:
                results.append({
                    "name": rec["name"],
                    "summary": rec["summary"],
                    "type": rec["type"],
                    "created": rec["created"],
                })
        return results

    def traverse(self, entity_name, project_id="default", max_depth=2):
        nodes = {}
        edges = []
        with self._driver.session() as session:
            cypher = """
            MATCH path = (start {name: $name, project_id: $pid})-[*1..""" + str(max_depth) + """]->(end)
            WHERE end.project_id = $pid
            RETURN path
            LIMIT 100
            """
            records = session.run(cypher, name=entity_name, pid=project_id)
            for rec in records:
                path = rec["path"]
                for node in path.nodes:
                    nid = node.element_id
                    if nid not in nodes:
                        nodes[nid] = {
                            "name": node.get("name", ""),
                            "type": list(node.labels)[0] if node.labels else "Unknown",
                            "summary": node.get("summary", ""),
                        }
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
            if status == "uncertain":
                cypher = """
                MATCH (b:Belief {project_id: $pid})
                WHERE b.active = true AND b.confidence < 0.5
                RETURN b.name AS name, b.summary AS summary,
                       b.confidence AS confidence, b.source_cycle AS cycle
                ORDER BY b.confidence ASC
                """
            elif status == "contradicted":
                cypher = """
                MATCH (b1:Belief {project_id: $pid})-[:CONTRADICTS]-(b2:Belief {project_id: $pid})
                WHERE b1.active = true AND b2.active = true
                RETURN DISTINCT b1.name AS name, b1.summary AS summary,
                       b1.confidence AS confidence, b1.source_cycle AS cycle
                """
            elif status == "superseded":
                cypher = """
                MATCH (b:Belief {project_id: $pid})
                WHERE b.status = 'superseded'
                RETURN b.name AS name, b.summary AS summary,
                       b.confidence AS confidence, b.source_cycle AS cycle
                ORDER BY b.updated_at DESC LIMIT 20
                """
            else:
                cypher = """
                MATCH (b:Belief {project_id: $pid})
                WHERE b.active = true
                RETURN b.name AS name, b.summary AS summary,
                       b.confidence AS confidence, b.source_cycle AS cycle
                ORDER BY b.confidence DESC
                """
            records = session.run(cypher, pid=project_id)
            for rec in records:
                beliefs.append({
                    "name": rec["name"],
                    "summary": rec["summary"],
                    "confidence": rec["confidence"],
                    "cycle": rec["cycle"],
                })
        return beliefs

    def get_stats(self, project_id="default"):
        stats = {"nodes": {}, "edges": {}, "total_nodes": 0, "total_edges": 0}
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
        return counts

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

    def update_belief(self, name, project_id="default", confidence=None,
                      status=None, summary=None):
        valid_statuses = ("active", "superseded")
        if status is not None and status not in valid_statuses:
            return {"updated": False, "name": name,
                    "error": f"Invalid status: {status}. Valid: {valid_statuses}"}
        if confidence is not None:
            confidence = max(0.0, min(1.0, float(confidence)))

        now = datetime.now().isoformat()
        changes = {}
        set_clauses = ["b.updated_at = $now"]

        if confidence is not None:
            set_clauses.append("b.confidence = $confidence")
            changes["confidence"] = confidence
        if status is not None:
            set_clauses.append("b.status = $status")
            if status == "superseded":
                set_clauses.append("b.active = false")
            elif status == "active":
                set_clauses.append("b.active = true")
            changes["status"] = status
        if summary is not None:
            set_clauses.append("b.summary = $summary")
            changes["summary"] = summary

        if not changes:
            return {"updated": False, "name": name, "error": "No changes specified"}

        cypher = f"""
        MATCH (b:Belief {{name: $name, project_id: $pid}})
        SET {', '.join(set_clauses)}
        RETURN b.name AS name
        """
        with self._driver.session() as session:
            result = session.run(
                cypher, name=name, pid=project_id, now=now,
                confidence=confidence, status=status, summary=summary,
            )
            record = result.single()

        if record is None:
            return {"updated": False, "name": name,
                    "error": f"Belief '{name}' not found in project '{project_id}'"}

        return {"updated": True, "name": name, "changes": changes}


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
        if namespace:
            return f"{project_id}_{namespace}"
        return project_id

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
            embedding = self._embed_text(query_text)
            if not embedding:
                return []
            self._ensure_collection(collection, vector_size=len(embedding))
            hits = self._qdrant.query_points(
                collection_name=collection,
                query=embedding,
                limit=top_k,
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
            return results
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
            headers = {
                "Api-Key": self._pinecone_api_key,
                "Content-Type": "application/json",
            }
            body = json.dumps({
                "vector": embedding,
                "topK": top_k,
                "namespace": ns,
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
            return results
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
