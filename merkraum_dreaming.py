#!/usr/bin/env python3
"""
Merkraum Dreaming Engine v2.0 — neuroscience-inspired memory consolidation.

Universal logic module: no Flask, no hosting dependencies.
Designed for open-source sharing — all hosting/user logic stays in the API layer.

Architecture v2.0 (Norman-directed, 2026-03-23):
Four phases inspired by biological sleep research (sharp-wave ripples,
CLS theory, REM pruning, Ebbinghaus spaced repetition):

1. Walk (hippocampal replay + free association): Unified graph walk with
   three movement types — 70% graph edges, 20% semantic jumps, 10% random
   teleportation. Entry at high-activation nodes. Creates provisional edges
   from dream observations. Replaces separate replay + bridging phases.
2. Consolidation (hippocampus-to-neocortex transfer): Episodic beliefs
   merged into abstractions (textual similarity) + entity-based topic
   compression. S5/S4-aware importance scoring guards near-duplicates.
3. Reflection (Default Mode Network): Structural health analysis —
   orphan detection, hub over-centralization, schema discipline.
4. Maintenance (REM synaptic pruning): Confidence decay, TTL enforcement
   with linear retention scaling (~10 activations ≈ 1 year), orphan
   pruning, edge deduplication.

All operations yield progress messages via generators, enabling
async monitoring and live visualization in the frontend.

v1.0 — Initial replay + consolidation
v1.1 — Bridging phase added (2026-03-23)
v1.2 — Compression phase added (2026-03-23)
v2.0 — Architecture v2.0: Walk-based 4-phase (Norman-directed, 2026-03-23)
"""

import json
import os
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Generator

from merkraum_backend import Neo4jBaseAdapter, NODE_TYPES, RELATIONSHIP_TYPES
from merkraum_llm import llm_call, get_provider_info

logger = logging.getLogger("merkraum-dreaming")


# ---------------------------------------------------------------------------
# Model configuration — SUP-159 + v2.0 architecture
# ---------------------------------------------------------------------------
# Walk: lightweight analysis of graph walks → Haiku (cost-efficient, many calls)
# Consolidation: quality abstraction + near-duplicate discrimination → Sonnet
# Reflection: no LLM (pure structural analysis)
# Maintenance: no LLM (pure Cypher operations)

_DEFAULT_WALK_MODEL = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
_DEFAULT_CONSOLIDATION_MODEL = "eu.anthropic.claude-sonnet-4-6"

# Legacy model aliases for backward compatibility
_DEFAULT_REPLAY_MODEL = _DEFAULT_WALK_MODEL
_DEFAULT_BRIDGING_MODEL = _DEFAULT_WALK_MODEL
_DEFAULT_COMPRESSION_MODEL = _DEFAULT_CONSOLIDATION_MODEL

# TTL scaling: linear growth, calibrated so ~10 activations ≈ 1 year (Norman's directive)
# Formula: TTL_days = base_ttl * (1 + access_count * TTL_SCALE_FACTOR)
# With base_ttl=30: 30 * (1 + 10 * 1.12) = 366 days ≈ 1 year
TTL_SCALE_FACTOR = 1.12

# Reconsolidation: importance-weighted confidence strengthening (Proposal #5)
# Biological basis: frequently recalled memories undergo reconsolidation,
# strengthening synaptic traces. This counters confidence decay for actively
# used knowledge. Boost = min(access_count * RECONSOLIDATION_FACTOR, MAX_BOOST).
# Threshold: only beliefs accessed >= MIN_ACCESS_FOR_RECONSOLIDATION get boosted.
RECONSOLIDATION_FACTOR = 0.01  # +0.01 confidence per access
RECONSOLIDATION_MAX_BOOST = 0.15  # Cap: max +0.15 confidence per maintenance cycle
RECONSOLIDATION_MIN_ACCESS = 3  # Minimum access_count to qualify


def _get_walk_model() -> str | None:
    """Return the model for walk phase (env override or default)."""
    return (os.environ.get("MERKRAUM_DREAMING_WALK_MODEL")
            or os.environ.get("MERKRAUM_DREAMING_REPLAY_MODEL")
            or _DEFAULT_WALK_MODEL)


def _get_consolidation_model() -> str | None:
    """Return the model for consolidation phase (env override or default)."""
    return os.environ.get("MERKRAUM_DREAMING_CONSOLIDATION_MODEL") or _DEFAULT_CONSOLIDATION_MODEL


# Legacy accessors for backward compatibility
def _get_replay_model() -> str | None:
    return _get_walk_model()


def _get_bridging_model() -> str | None:
    return _get_walk_model()


def _get_compression_model() -> str | None:
    return (os.environ.get("MERKRAUM_DREAMING_COMPRESSION_MODEL")
            or _get_consolidation_model())


def get_dreaming_config() -> dict:
    """Return current dreaming model configuration for diagnostics."""
    return {
        "walk_model": _get_walk_model(),
        "consolidation_model": _get_consolidation_model(),
        "reflection_model": None,
        "maintenance_model": None,
        "walk_model_source": "env" if os.environ.get("MERKRAUM_DREAMING_WALK_MODEL") or os.environ.get("MERKRAUM_DREAMING_REPLAY_MODEL") else "default",
        "consolidation_model_source": "env" if os.environ.get("MERKRAUM_DREAMING_CONSOLIDATION_MODEL") else "default",
        "ttl_scale_factor": TTL_SCALE_FACTOR,
        "architecture_version": "2.0",
        # Legacy keys for backward compat
        "replay_model": _get_walk_model(),
        "bridging_model": _get_walk_model(),
        "compression_model": _get_consolidation_model(),
        "replay_model_source": "env" if os.environ.get("MERKRAUM_DREAMING_WALK_MODEL") or os.environ.get("MERKRAUM_DREAMING_REPLAY_MODEL") else "default",
        "bridging_model_source": "env" if os.environ.get("MERKRAUM_DREAMING_WALK_MODEL") or os.environ.get("MERKRAUM_DREAMING_REPLAY_MODEL") else "default",
        "compression_model_source": "env" if os.environ.get("MERKRAUM_DREAMING_CONSOLIDATION_MODEL") else "default",
    }


# ---------------------------------------------------------------------------
# Progress message types — structured for frontend consumption
# ---------------------------------------------------------------------------

def _msg(phase: str, step: str, detail: str, data: dict | None = None) -> dict:
    """Create a structured progress message."""
    msg = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "step": step,
        "detail": detail,
    }
    if data:
        msg["data"] = data
    return msg


# ---------------------------------------------------------------------------
# Access Logging — increment access_count on node reads (v2.0 infrastructure)
# ---------------------------------------------------------------------------

def _log_access(session, project_id: str, node_names: list[str]):
    """Increment access_count and update last_accessed for visited nodes.

    Called during walk to track activation frequency. This data drives:
    - Walk entry point selection (high-activation nodes)
    - TTL extension (more activations = longer retention)
    - Importance scoring (activation_score component)
    """
    if not node_names:
        return
    now = datetime.now(timezone.utc).isoformat()
    session.run(
        "UNWIND $names AS name "
        "MATCH (n {project_id: $pid, name: name}) "
        "SET n.access_count = COALESCE(n.access_count, 0) + 1, "
        "    n.last_accessed = $now",
        names=node_names, pid=project_id, now=now,
    )


def calculate_ttl_days(base_ttl_days: int, access_count: int,
                       scale_factor: float = TTL_SCALE_FACTOR) -> int:
    """Calculate effective TTL using linear scaling.

    Norman's directive: ~10 activations ≈ 1 year (with base_ttl=30 days).
    Formula: TTL = base_ttl * (1 + access_count * scale_factor)

    Examples (base_ttl=30, scale_factor=1.12):
      0 activations: 30 days
      1 activation:  64 days
      5 activations: 198 days
      10 activations: 366 days ≈ 1 year
      20 activations: 702 days ≈ 2 years
    """
    return int(base_ttl_days * (1 + access_count * scale_factor))


# ---------------------------------------------------------------------------
# Phase 1 (v2.0): Walk — unified graph walk with 70/20/10 movement
# ---------------------------------------------------------------------------

def walk(
    adapter: Neo4jBaseAdapter,
    project_id: str = "default",
    steps: int = 20,
    walks: int = 5,
    p_graph: float = 0.70,
    p_semantic: float = 0.20,
    p_random: float = 0.10,
    create_edges: bool = True,
    seed: str | None = None,
) -> Generator[dict, None, dict]:
    """Neuroscience-inspired graph walk with three movement types.

    Replaces separate replay + bridging phases (v2.0 architecture).

    Entry: high-activation nodes (access_count * recency_weight).
    Movement at each step:
      - p_graph (70%): follow a random edge to a neighbor
      - p_semantic (20%): semantic jump via vector search
      - p_random (10%): random teleportation to any node

    At each walk completion, LLM analyzes the path for:
      - Missing relationships → provisional edges
      - Surprising connections → provisional edges
      - Contradictions with existing beliefs
      - Cross-domain patterns (the bridging function)

    Access logging: every visited node gets access_count incremented.

    Args:
        adapter: Backend adapter (Neo4j + vector store)
        project_id: Project to walk
        steps: Steps per walk (default 20, max 50)
        walks: Number of walks per session (default 5, max 15)
        p_graph: Probability of graph walk (default 0.70)
        p_semantic: Probability of semantic jump (default 0.20)
        p_random: Probability of random teleportation (default 0.10)
        create_edges: Write provisional edges from observations (default True)
        seed: Optional starting entity for first walk

    Yields progress messages. Returns final result dict.
    """
    yield _msg("walk", "start",
               f"Starting {walks} walk(s), {steps} steps each "
               f"(graph {p_graph:.0%}/semantic {p_semantic:.0%}/random {p_random:.0%})")
    now = datetime.now(timezone.utc).isoformat()
    all_walks = []

    for walk_num in range(walks):
        yield _msg("walk", "walk_start",
                   f"Walk {walk_num + 1}/{walks}: selecting entry point")

        with adapter._driver.session() as session:
            # Entry point selection: high-activation nodes (access_count * recency)
            # Norman's feedback: "steigen wir dort ein, wo haeufig aktiviert wird"
            if seed and walk_num == 0:
                seeds = session.run(
                    "MATCH (n {project_id: $pid}) "
                    "WHERE toLower(n.name) CONTAINS toLower($q) "
                    "AND any(lbl IN labels(n) WHERE lbl IN $types) "
                    "RETURN n.name AS name, labels(n)[0] AS type, "
                    "COALESCE(n.access_count, 0) AS access_count LIMIT 1",
                    pid=project_id, q=seed, types=NODE_TYPES,
                ).data()
                if not seeds:
                    yield _msg("walk", "seed_not_found",
                               f"Entity '{seed}' not found, using activation-based entry")
                    seeds = None
                else:
                    seed_node = seeds[0]
            else:
                seeds = None

            if not seeds:
                # Select entry by activation frequency weighted by recency
                candidates = session.run(
                    "MATCH (n {project_id: $pid}) "
                    "WHERE any(lbl IN labels(n) WHERE lbl IN $types) "
                    "WITH n, labels(n)[0] AS type, "
                    "COALESCE(n.access_count, 0) AS ac, "
                    "CASE WHEN n.last_accessed IS NOT NULL "
                    "     THEN 1.0 / (duration.inDays(datetime(n.last_accessed), datetime()).days + 1) "
                    "     ELSE 0.01 END AS recency "
                    "RETURN n.name AS name, type, ac, recency, "
                    "(ac + 1) * recency AS activation_score "
                    "ORDER BY activation_score DESC LIMIT 50",
                    pid=project_id, types=NODE_TYPES,
                ).data()

                if not candidates:
                    yield _msg("walk", "empty_graph",
                               "No nodes in project — skipping walk")
                    return {"walks": [], "observations": [], "phase": "walk",
                            "status": "empty"}

                # Weighted random selection from top candidates
                weights = [max(c.get("activation_score", 0.01), 0.01)
                           for c in candidates]
                seed_node = random.choices(candidates, weights=weights, k=1)[0]

            yield _msg("walk", "entry_selected",
                       f"Entry: {seed_node['name']} [{seed_node['type']}] "
                       f"(access_count: {seed_node.get('access_count', 0)})",
                       {"entry": seed_node})

            # Execute walk with 70/20/10 movement
            current = seed_node["name"]
            walk_path = [{"name": current, "type": seed_node["type"],
                          "hop": "entry"}]
            visited = {current}
            subgraph_data = []
            movement_stats = {"graph": 0, "semantic": 0, "random": 0,
                              "dead_end": 0}

            for step in range(steps):
                # Collect neighborhood for LLM context
                neighbors = session.run(
                    "MATCH (n {project_id: $pid})-[r]-(neighbor {project_id: $pid}) "
                    "WHERE n.name = $name "
                    "AND any(lbl IN labels(neighbor) WHERE lbl IN $types) "
                    "RETURN neighbor.name AS name, labels(neighbor)[0] AS type, "
                    "type(r) AS rel, r.confidence AS conf",
                    pid=project_id, name=current, types=NODE_TYPES,
                ).data()
                subgraph_data.append({"node": current, "neighbors": neighbors})

                # Roll for movement type
                roll = random.random()
                moved = False

                if roll < p_graph:
                    # Graph walk: follow a random edge
                    unvisited = [n for n in neighbors
                                 if n["name"] not in visited]
                    pool = unvisited if unvisited else neighbors
                    if pool:
                        target = random.choice(pool)
                        current = target["name"]
                        visited.add(current)
                        walk_path.append({
                            "name": current, "type": target["type"],
                            "hop": "graph", "rel": target["rel"]})
                        movement_stats["graph"] += 1
                        moved = True

                elif roll < p_graph + p_semantic:
                    # Semantic jump: vector search for similar node
                    try:
                        results = adapter.vector_search(
                            current, top_k=8, project_id=project_id)
                        if results:
                            jump_candidates = [
                                r for r in results
                                if r.get("name") and r["name"] not in visited
                            ]
                            if jump_candidates:
                                target_name = random.choice(
                                    jump_candidates[:5])["name"]
                                target_info = session.run(
                                    "MATCH (n {project_id: $pid, name: $name}) "
                                    "RETURN n.name AS name, labels(n)[0] AS type "
                                    "LIMIT 1",
                                    pid=project_id, name=target_name,
                                ).data()
                                if target_info:
                                    current = target_info[0]["name"]
                                    visited.add(current)
                                    walk_path.append({
                                        "name": current,
                                        "type": target_info[0]["type"],
                                        "hop": "semantic"})
                                    movement_stats["semantic"] += 1
                                    moved = True
                    except Exception:
                        pass  # Semantic jump is optional — fall through

                else:
                    # Random teleportation: jump to any random node
                    random_nodes = session.run(
                        "MATCH (n {project_id: $pid}) "
                        "WHERE any(lbl IN labels(n) WHERE lbl IN $types) "
                        "AND n.name <> $current "
                        "WITH n, rand() AS r ORDER BY r LIMIT 1 "
                        "RETURN n.name AS name, labels(n)[0] AS type",
                        pid=project_id, types=NODE_TYPES, current=current,
                    ).data()
                    if random_nodes:
                        target = random_nodes[0]
                        current = target["name"]
                        visited.add(current)
                        walk_path.append({
                            "name": current, "type": target["type"],
                            "hop": "random"})
                        movement_stats["random"] += 1
                        moved = True

                if not moved:
                    # Fallback: try graph walk, then random teleport
                    unvisited = [n for n in neighbors
                                 if n["name"] not in visited]
                    pool = unvisited if unvisited else neighbors
                    if pool:
                        target = random.choice(pool)
                        current = target["name"]
                        visited.add(current)
                        walk_path.append({
                            "name": current, "type": target["type"],
                            "hop": "fallback", "rel": target["rel"]})
                        movement_stats["graph"] += 1
                        moved = True
                    else:
                        movement_stats["dead_end"] += 1
                        yield _msg("walk", "dead_end",
                                   f"Dead end at '{current}' (step {step + 1})")
                        break

                if step % 5 == 4:  # Progress every 5 steps
                    yield _msg("walk", "progress",
                               f"Walk {walk_num + 1}, step {step + 1}/{steps}: "
                               f"{walk_path[-1]['name']} [{walk_path[-1]['hop']}]",
                               {"path_length": len(walk_path)})

            # Log access for all visited nodes
            _log_access(session, project_id, list(visited))

            # Update last_dreamed
            for v in visited:
                session.run(
                    "MATCH (n {project_id: $pid, name: $name}) "
                    "SET n.last_dreamed = $now",
                    pid=project_id, name=v, now=now,
                )

        # LLM analysis of walk
        yield _msg("walk", "analyzing",
                   f"Analyzing walk {walk_num + 1} ({len(walk_path)} nodes, "
                   f"g:{movement_stats['graph']}/s:{movement_stats['semantic']}/"
                   f"r:{movement_stats['random']})...")

        walk_text = " → ".join(
            f"{s['name']}{'*' if s['hop'] == 'semantic' else '†' if s['hop'] == 'random' else ''}"
            for s in walk_path
        )
        subgraph_text = ""
        for sg in subgraph_data:
            subgraph_text += f"\nNode: {sg['node']}\n"
            for n in sg["neighbors"][:10]:
                subgraph_text += (
                    f"  --[{n['rel']}]--> {n['name']} [{n['type']}] "
                    f"(conf: {n['conf']})\n"
                )

        _walk_schema = {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": [
                                "surprising_connection", "missing_relationship",
                                "cross_domain_bridge", "redundancy", "insight",
                                "contradiction"]},
                            "description": {"type": "string"},
                            "entities_involved": {"type": "array",
                                                  "items": {"type": "string"},
                                                  "minItems": 2},
                            "suggested_relationship": {
                                "type": "string",
                                "enum": list(RELATIONSHIP_TYPES),
                            },
                            "suggested_action": {"type": "string"},
                        },
                        "required": ["type", "description",
                                     "entities_involved"],
                    },
                },
                "walk_quality": {"type": "string",
                                 "enum": ["productive", "routine", "dead_end"]},
                "summary": {"type": "string"},
            },
            "required": ["observations", "walk_quality", "summary"],
        }
        analysis = llm_call(
            "You analyze knowledge graph walks to find surprising patterns, "
            "missing relationships, cross-domain bridges, contradictions, "
            "and insights. The walk path shows nodes visited with movement "
            "types: plain=graph edge, *=semantic jump, †=random teleportation. "
            "For observations of type 'missing_relationship', "
            "'surprising_connection', or 'cross_domain_bridge', always include "
            "'suggested_relationship' — a relationship type from the enum.",
            f"Walk path: {walk_text}\n\nSubgraph:\n{subgraph_text}",
            temperature=0.4,
            json_schema=_walk_schema,
            model=_get_walk_model(),
        )

        walk_result = {
            "path": walk_path,
            "analysis": analysis,
            "nodes_visited": len(visited),
            "movement_stats": movement_stats,
        }
        all_walks.append(walk_result)

        if analysis:
            yield _msg("walk", "walk_complete",
                       f"Walk {walk_num + 1}: {analysis.get('walk_quality', '?')} — "
                       f"{analysis.get('summary', '')[:120]}",
                       {"walk_quality": analysis.get("walk_quality"),
                        "observations_count": len(analysis.get("observations", [])),
                        "movement_stats": movement_stats})
        else:
            yield _msg("walk", "walk_complete",
                       f"Walk {walk_num + 1}: analysis unavailable",
                       {"walk_quality": "unknown",
                        "movement_stats": movement_stats})

    observations = []
    for w in all_walks:
        if w.get("analysis"):
            observations.extend(w["analysis"].get("observations", []))

    # --- Provisional Edge Creation ---
    edges_created = 0
    edge_details = []

    if create_edges:
        from datetime import timedelta
        ttl_days = 7
        valid_until = (datetime.now(timezone.utc)
                       + timedelta(days=ttl_days)).isoformat()

        actionable_types = ("missing_relationship", "surprising_connection",
                            "cross_domain_bridge")
        actionable = [
            obs for obs in observations
            if obs.get("type") in actionable_types
            and len(obs.get("entities_involved", [])) >= 2
        ]

        if actionable:
            yield _msg("walk", "edge_creation_start",
                       f"Creating {len(actionable)} provisional edge(s)")

            for obs in actionable:
                entities = obs["entities_involved"]
                rel_type = obs.get("suggested_relationship", "RELATES_TO")
                if rel_type not in RELATIONSHIP_TYPES:
                    rel_type = "RELATES_TO"

                rel = {
                    "source": entities[0],
                    "target": entities[1],
                    "type": rel_type,
                    "confidence": 0.3,
                    "reason": f"Dream walk: {obs.get('description', '')[:200]}",
                    "valid_until": valid_until,
                }

                try:
                    count = adapter.write_relationships(
                        [rel],
                        source_cycle="dream",
                        source_type="dream",
                        project_id=project_id,
                        actor="dreaming-walk",
                    )
                    if count > 0:
                        edges_created += 1
                        detail = {
                            "source": entities[0],
                            "target": entities[1],
                            "rel_type": rel_type,
                            "observation_type": obs["type"],
                            "valid_until": valid_until,
                        }
                        edge_details.append(detail)
                        yield _msg("walk", "edge_created",
                                   f"  {entities[0]} --[{rel_type}]--> "
                                   f"{entities[1]} (conf: 0.3, TTL: {ttl_days}d)",
                                   detail)
                except Exception as e:
                    logger.warning("Walk edge creation failed: %s -> %s: %s",
                                   entities[0], entities[1], e)
                    yield _msg("walk", "edge_failed",
                               f"  Failed: {entities[0]} -> {entities[1]}: {e}")

    total_movement = sum(v for k, v in movement_stats.items()
                         if k != "dead_end")
    edge_summary = f", {edges_created} edge(s) created" if edges_created else ""
    yield _msg("walk", "done",
               f"Walk complete: {walks} walk(s), {len(observations)} "
               f"observation(s){edge_summary}",
               {"walks": walks, "total_steps": total_movement,
                "observations": len(observations),
                "edges_created": edges_created})

    return {
        "phase": "walk",
        "status": "completed",
        "walks": all_walks,
        "observations": observations,
        "total_observations": len(observations),
        "edges_created": edges_created,
        "edge_details": edge_details,
    }


# ---------------------------------------------------------------------------
# Legacy Phase 1: Replay — kept for backward compatibility
# ---------------------------------------------------------------------------

def replay(
    adapter: Neo4jBaseAdapter,
    project_id: str = "default",
    hops: int = 5,
    walks: int = 3,
    seed: str | None = None,
    create_edges: bool = True,
) -> Generator[dict, None, dict]:
    """Random walk through the graph, surfacing unexpected connections.

    When create_edges=True (default), observations of type 'missing_relationship'
    or 'surprising_connection' are written as provisional edges with low confidence
    (0.3) and a 7-day TTL. This implements associative edge creation — dreaming
    doesn't just observe, it writes new connections for later review.

    Yields progress messages. Returns final result dict.
    """
    yield _msg("replay", "start", f"Starting {walks} replay walk(s), {hops} hops each")
    now = datetime.now(timezone.utc).isoformat()
    all_walks = []

    for walk_num in range(walks):
        yield _msg("replay", "walk_start", f"Walk {walk_num + 1}/{walks}: selecting seed node")

        with adapter._driver.session() as session:
            # Seed selection: staleness-weighted random
            if seed and walk_num == 0:
                seeds = session.run(
                    "MATCH (n {project_id: $pid}) "
                    "WHERE toLower(n.name) CONTAINS toLower($q) "
                    "RETURN n.name AS name, labels(n)[0] AS type LIMIT 1",
                    pid=project_id, q=seed,
                ).data()
                if not seeds:
                    yield _msg("replay", "seed_not_found", f"Entity '{seed}' not found, using random")
                    seeds = None
                else:
                    seed_node = seeds[0]
            else:
                seeds = None

            if not seeds:
                candidates = session.run(
                    "MATCH (n {project_id: $pid}) "
                    "WHERE any(lbl IN labels(n) WHERE lbl IN $types) "
                    "WITH n, labels(n)[0] AS type, "
                    "CASE WHEN n.last_dreamed IS NULL THEN 100 "
                    "     ELSE duration.inDays(datetime(n.last_dreamed), datetime()).days + 1 "
                    "END AS staleness "
                    "RETURN n.name AS name, type, staleness "
                    "ORDER BY staleness DESC LIMIT 50",
                    pid=project_id, types=NODE_TYPES,
                ).data()
                if not candidates:
                    yield _msg("replay", "empty_graph", "No nodes in project — skipping replay")
                    return {"walks": [], "observations": [], "phase": "replay", "status": "empty"}
                weights = [c["staleness"] for c in candidates]
                seed_node = random.choices(candidates, weights=weights, k=1)[0]

            yield _msg("replay", "seed_selected",
                       f"Seed: {seed_node['name']} [{seed_node['type']}]",
                       {"seed": seed_node})

            # Walk
            current = seed_node["name"]
            walk_path = [{"name": current, "type": seed_node["type"], "hop": "seed"}]
            visited = {current}
            subgraph_data = []

            for hop in range(hops):
                neighbors = session.run(
                    "MATCH (n {project_id: $pid})-[r]-(neighbor {project_id: $pid}) "
                    "WHERE n.name = $name "
                    "AND any(lbl IN labels(neighbor) WHERE lbl IN $types) "
                    "RETURN neighbor.name AS name, labels(neighbor)[0] AS type, "
                    "type(r) AS rel, r.confidence AS conf",
                    pid=project_id, name=current, types=NODE_TYPES,
                ).data()
                subgraph_data.append({"node": current, "neighbors": neighbors})

                # Semantic jump (30% chance after first hop)
                do_semantic = (random.random() >= 0.7 and len(visited) > 1)
                jumped = False

                if do_semantic:
                    try:
                        results = adapter.vector_search(current, top_k=8, project_id=project_id)
                        if results:
                            jump_names = [r.get("name", "") for r in results
                                          if r.get("name") and r["name"] not in visited]
                            if jump_names:
                                target_name = random.choice(jump_names[:5])
                                target_info = session.run(
                                    "MATCH (n {project_id: $pid, name: $name}) "
                                    "RETURN n.name AS name, labels(n)[0] AS type LIMIT 1",
                                    pid=project_id, name=target_name,
                                ).data()
                                if target_info:
                                    current = target_info[0]["name"]
                                    visited.add(current)
                                    walk_path.append({"name": current,
                                                      "type": target_info[0]["type"],
                                                      "hop": "semantic_jump"})
                                    jumped = True
                    except Exception:
                        pass  # Semantic jump is optional

                if not jumped:
                    unvisited = [n for n in neighbors if n["name"] not in visited]
                    pool = unvisited if unvisited else neighbors
                    if not pool:
                        yield _msg("replay", "dead_end",
                                   f"Dead end at '{current}' (hop {hop + 1})")
                        break
                    target = random.choice(pool)
                    current = target["name"]
                    visited.add(current)
                    walk_path.append({"name": current, "type": target["type"],
                                      "hop": "graph", "rel": target["rel"]})

                yield _msg("replay", "hop",
                           f"Hop {hop + 1}: {walk_path[-1]['name']} "
                           f"[{walk_path[-1].get('hop', 'graph')}]",
                           {"path_length": len(walk_path)})

            # Update last_dreamed
            for v in visited:
                session.run(
                    "MATCH (n {project_id: $pid, name: $name}) "
                    "SET n.last_dreamed = $now",
                    pid=project_id, name=v, now=now,
                )

        # LLM analysis of walk
        yield _msg("replay", "analyzing",
                   f"Analyzing walk {walk_num + 1} ({len(walk_path)} nodes)...")

        walk_text = " → ".join(
            f"{s['name']}{'*' if s['hop'] == 'semantic_jump' else ''}"
            for s in walk_path
        )
        subgraph_text = ""
        for sg in subgraph_data:
            subgraph_text += f"\nNode: {sg['node']}\n"
            for n in sg["neighbors"][:10]:
                subgraph_text += (
                    f"  --[{n['rel']}]--> {n['name']} [{n['type']}] "
                    f"(conf: {n['conf']})\n"
                )

        _replay_schema = {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["surprising_connection", "missing_relationship", "redundancy", "insight"]},
                            "description": {"type": "string"},
                            "entities_involved": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                            "suggested_relationship": {
                                "type": "string",
                                "enum": list(RELATIONSHIP_TYPES),
                                "description": "For missing_relationship or surprising_connection: the relationship type to create between the first two entities_involved",
                            },
                            "suggested_action": {"type": "string"},
                        },
                        "required": ["type", "description", "entities_involved"],
                    },
                },
                "walk_quality": {"type": "string", "enum": ["productive", "routine", "dead_end"]},
                "summary": {"type": "string", "description": "One-paragraph synthesis of the walk"},
            },
            "required": ["observations", "walk_quality", "summary"],
        }
        analysis = llm_call(
            "You analyze knowledge graph walks to find surprising patterns, "
            "missing relationships, redundancies, and insights. "
            "For observations of type 'missing_relationship' or 'surprising_connection', "
            "always include 'suggested_relationship' — a relationship type from the enum "
            "that should connect the first two entities in 'entities_involved'.",
            f"Walk path: {walk_text}\n\nSubgraph:\n{subgraph_text}",
            temperature=0.4,
            json_schema=_replay_schema,
            model=_get_replay_model(),
        )

        walk_result = {
            "path": walk_path,
            "analysis": analysis,
            "nodes_visited": len(visited),
        }
        all_walks.append(walk_result)

        if analysis:
            yield _msg("replay", "walk_complete",
                       f"Walk {walk_num + 1}: {analysis.get('walk_quality', '?')} — "
                       f"{analysis.get('summary', '')[:120]}",
                       {"walk_quality": analysis.get("walk_quality"),
                        "observations_count": len(analysis.get("observations", []))})
        else:
            yield _msg("replay", "walk_complete",
                       f"Walk {walk_num + 1}: analysis unavailable",
                       {"walk_quality": "unknown"})

    observations = []
    for w in all_walks:
        if w.get("analysis"):
            observations.extend(w["analysis"].get("observations", []))

    # --- Associative Edge Creation (Proposal #1, Z1839) ---
    # Write provisional edges for missing_relationship and surprising_connection
    # observations. Low confidence (0.3), 7-day TTL, source_type="dream".
    edges_created = 0
    edge_details = []

    if create_edges:
        from datetime import timedelta
        ttl_days = 7
        valid_until = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()

        actionable = [
            obs for obs in observations
            if obs.get("type") in ("missing_relationship", "surprising_connection")
            and len(obs.get("entities_involved", [])) >= 2
        ]

        if actionable:
            yield _msg("replay", "edge_creation_start",
                       f"Creating {len(actionable)} provisional edge(s) from dream observations")

            for obs in actionable:
                entities = obs["entities_involved"]
                rel_type = obs.get("suggested_relationship", "RELATES_TO")
                if rel_type not in RELATIONSHIP_TYPES:
                    rel_type = "RELATES_TO"

                rel = {
                    "source": entities[0],
                    "target": entities[1],
                    "type": rel_type,
                    "confidence": 0.3,
                    "reason": f"Dream observation: {obs.get('description', '')[:200]}",
                    "valid_until": valid_until,
                }

                try:
                    count = adapter.write_relationships(
                        [rel],
                        source_cycle="dream",
                        source_type="dream",
                        project_id=project_id,
                        actor="dreaming-replay",
                    )
                    if count > 0:
                        edges_created += 1
                        detail = {
                            "source": entities[0],
                            "target": entities[1],
                            "rel_type": rel_type,
                            "observation_type": obs["type"],
                            "valid_until": valid_until,
                        }
                        edge_details.append(detail)
                        yield _msg("replay", "edge_created",
                                   f"  {entities[0]} --[{rel_type}]--> {entities[1]} "
                                   f"(conf: 0.3, TTL: {ttl_days}d)",
                                   detail)
                except Exception as e:
                    logger.warning("Dream edge creation failed: %s -> %s: %s",
                                   entities[0], entities[1], e)
                    yield _msg("replay", "edge_failed",
                               f"  Failed: {entities[0]} -> {entities[1]}: {e}")

    edge_summary = f", {edges_created} edge(s) created" if edges_created else ""
    yield _msg("replay", "done",
               f"Replay complete: {walks} walk(s), "
               f"{len(observations)} observation(s){edge_summary}")

    return {
        "phase": "replay",
        "status": "completed",
        "walks": all_walks,
        "observations": observations,
        "total_observations": len(observations),
        "edges_created": edges_created,
        "edge_details": edge_details,
    }


# ---------------------------------------------------------------------------
# Phase 2: Consolidation — merge episodic beliefs into abstractions
# ---------------------------------------------------------------------------

def consolidate(
    adapter: Neo4jBaseAdapter,
    project_id: str = "default",
    similarity_threshold: float = 0.75,
    min_cluster_size: int = 2,
    dry_run: bool = False,
) -> Generator[dict, None, dict]:
    """Merge similar beliefs into abstract, higher-confidence beliefs.

    Yields progress messages. Returns final result dict.
    """
    yield _msg("consolidation", "start",
               f"Scanning for similar beliefs (threshold: {similarity_threshold})")
    now = datetime.now(timezone.utc).isoformat()

    # Fetch active beliefs, then compute similarity in Python (no APOC dependency)
    with adapter._driver.session() as session:
        all_beliefs = session.run(
            "MATCH (b:Belief {project_id: $pid}) "
            "WHERE (b.status IS NULL OR b.status = 'active') "
            "AND (b.active = true OR b.active IS NULL) "
            "RETURN b.name AS name, b.confidence AS confidence",
            pid=project_id,
        ).data()

    def _strip_prefix(name):
        return name[8:] if name.startswith("Belief: ") else name

    from difflib import SequenceMatcher
    pairs = []
    for i in range(len(all_beliefs)):
        na = _strip_prefix(all_beliefs[i]["name"])
        for j in range(i + 1, len(all_beliefs)):
            nb = _strip_prefix(all_beliefs[j]["name"])
            sim = SequenceMatcher(None, na.lower(), nb.lower()).ratio()
            if sim >= similarity_threshold:
                pairs.append({
                    "name_a": all_beliefs[i]["name"],
                    "conf_a": all_beliefs[i]["confidence"],
                    "name_b": all_beliefs[j]["name"],
                    "conf_b": all_beliefs[j]["confidence"],
                    "similarity": round(sim, 3),
                })
    pairs.sort(key=lambda p: p["similarity"], reverse=True)

    if not pairs:
        yield _msg("consolidation", "done", "No consolidation candidates found")
        return {
            "phase": "consolidation",
            "status": "completed",
            "clusters_found": 0,
            "clusters_consolidated": 0,
        }

    yield _msg("consolidation", "pairs_found",
               f"Found {len(pairs)} similar belief pair(s)",
               {"pairs_count": len(pairs)})

    # Build clusters via union-find
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for p in pairs:
        union(p["name_a"], p["name_b"])

    clusters = {}
    all_names = set()
    for p in pairs:
        all_names.update([p["name_a"], p["name_b"]])
    for name in all_names:
        root = find(name)
        clusters.setdefault(root, set()).add(name)

    valid_clusters = {k: sorted(v) for k, v in clusters.items()
                      if len(v) >= min_cluster_size}

    if not valid_clusters:
        yield _msg("consolidation", "done",
                   f"No clusters with >= {min_cluster_size} members")
        return {
            "phase": "consolidation",
            "status": "completed",
            "clusters_found": len(clusters),
            "clusters_consolidated": 0,
        }

    yield _msg("consolidation", "clustering",
               f"{len(valid_clusters)} cluster(s) ready for consolidation",
               {"clusters_count": len(valid_clusters)})

    consolidated_count = 0
    results = []

    for i, (root, members) in enumerate(valid_clusters.items()):
        yield _msg("consolidation", "cluster_start",
                   f"Cluster {i + 1}/{len(valid_clusters)}: "
                   f"{len(members)} beliefs",
                   {"cluster_index": i, "members": list(members)})

        # LLM consolidation
        beliefs_text = "\n".join(f"  - {m}" for m in members)
        _consolidation_schema = {
            "type": "object",
            "properties": {
                "abstract_belief": {"type": "string", "maxLength": 200, "description": "Single falsifiable proposition"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reasoning": {"type": "string", "description": "Brief explanation"},
            },
            "required": ["abstract_belief", "confidence", "reasoning"],
        }
        result = llm_call(
            "Convert episodic beliefs (specific observations) into one "
            "abstract belief (generalizable knowledge). The abstract belief "
            "should be a single falsifiable proposition, max 200 characters, "
            "faithful to the evidence.",
            f"Consolidate these related beliefs:\n{beliefs_text}",
            temperature=0.3,
            json_schema=_consolidation_schema,
            model=_get_consolidation_model(),
        )

        if not result or "abstract_belief" not in result:
            yield _msg("consolidation", "cluster_skip",
                       f"Cluster {i + 1}: LLM analysis unavailable, skipping")
            continue

        abstract = result["abstract_belief"]
        confidence = result.get("confidence", 0.7)
        reasoning = result.get("reasoning", "")

        cluster_result = {
            "original_beliefs": list(members),
            "abstract_belief": abstract,
            "confidence": confidence,
            "reasoning": reasoning,
            "applied": not dry_run,
        }
        results.append(cluster_result)

        if dry_run:
            yield _msg("consolidation", "cluster_preview",
                       f"Would create: \"{abstract}\" (conf: {confidence})",
                       cluster_result)
            continue

        # Write abstract belief and link originals
        with adapter._driver.session() as session:
            session.run(
                "MERGE (b:Belief {name: $name, project_id: $pid}) "
                "ON CREATE SET b.summary = $reasoning, "
                "b.created_at = $now, b.source_type = 'consolidation', "
                "b.active = true, b.status = 'active', "
                "b.confidence = $confidence, b.mentions = 1 "
                "ON MATCH SET b.confidence = CASE "
                "WHEN $confidence > b.confidence THEN $confidence "
                "ELSE b.confidence END, b.updated_at = $now",
                name=abstract, pid=project_id,
                reasoning=reasoning, now=now, confidence=confidence,
            )
            for member in members:
                session.run(
                    "MATCH (orig:Belief {name: $orig_name, project_id: $pid}) "
                    "MATCH (abst:Belief {name: $abst_name, project_id: $pid}) "
                    "MERGE (orig)-[r:SUPPORTS]->(abst) "
                    "ON CREATE SET r.confidence = 0.9, "
                    "r.reason = 'consolidated from episodic cluster', "
                    "r.source_type = 'consolidation', r.created_at = $now, "
                    "r.active = true, r.weight = 0.9 "
                    "SET orig.status = 'consolidated', orig.active = false, "
                    "orig.consolidated_into = $abst_name, "
                    "orig.consolidated_at = $now",
                    orig_name=member, abst_name=abstract,
                    pid=project_id, now=now,
                )

        consolidated_count += 1
        yield _msg("consolidation", "cluster_done",
                   f"Consolidated: \"{abstract}\" (conf: {confidence})",
                   cluster_result)

    mode = "preview" if dry_run else "completed"
    yield _msg("consolidation", "done",
               f"Consolidation {mode}: {consolidated_count} cluster(s) processed",
               {"consolidated": consolidated_count})

    return {
        "phase": "consolidation",
        "status": mode,
        "clusters_found": len(valid_clusters),
        "clusters_consolidated": consolidated_count,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Phase 3: Compression — entity-based topic abstraction (hippocampus→neocortex)
# ---------------------------------------------------------------------------

def compress(
    adapter: Neo4jBaseAdapter,
    project_id: str = "default",
    min_cluster_size: int = 3,
    max_clusters: int = 10,
    dry_run: bool = False,
) -> Generator[dict, None, dict]:
    """Compress episodic beliefs into higher-level abstractions by topic.

    Unlike consolidation (which merges textually similar beliefs), compression
    identifies beliefs clustered around the same graph entities and creates
    abstract generalizations — the hippocampus-to-neocortex transfer.

    Strategy: Beliefs sharing connections to the same entities form natural
    topic groups. For each group with enough members, an LLM synthesizes a
    higher-level abstract belief. Original beliefs are marked as compressed
    and linked to the abstraction.

    Args:
        adapter: Backend adapter (Neo4j + vector store)
        project_id: Project to compress
        min_cluster_size: Minimum beliefs per entity to form a cluster (default 3)
        max_clusters: Maximum clusters to process per session (default 10)
        dry_run: If True, preview compressions without applying

    Yields progress messages. Returns final result dict.
    """
    mode = "preview" if dry_run else "apply"
    yield _msg("compression", "start",
               f"Topic compression ({mode}): grouping beliefs by shared entities "
               f"(min cluster size: {min_cluster_size})")

    # Find entity-based belief clusters: beliefs connected to the same entity
    with adapter._driver.session() as session:
        clusters_raw = session.run(
            "MATCH (b:Belief {project_id: $pid})-[r]-(e {project_id: $pid}) "
            "WHERE (b.status IS NULL OR b.status = 'active') "
            "AND (b.active = true OR b.active IS NULL) "
            "AND any(lbl IN labels(e) WHERE lbl IN $entity_types) "
            "WITH e.name AS entity, labels(e)[0] AS entity_type, "
            "collect(DISTINCT b.name) AS beliefs, "
            "collect(DISTINCT type(r)) AS rel_types "
            "WHERE size(beliefs) >= $min_size "
            "RETURN entity, entity_type, beliefs, rel_types "
            "ORDER BY size(beliefs) DESC "
            "LIMIT $max_clusters",
            pid=project_id,
            entity_types=[t for t in NODE_TYPES if t != "Belief"],
            min_size=min_cluster_size,
            max_clusters=max_clusters,
        ).data()

    if not clusters_raw:
        yield _msg("compression", "done",
                   f"No entity clusters with >= {min_cluster_size} beliefs found")
        return {
            "phase": "compression",
            "status": "completed",
            "clusters_found": 0,
            "clusters_compressed": 0,
            "abstractions_created": 0,
        }

    yield _msg("compression", "clusters_found",
               f"Found {len(clusters_raw)} entity cluster(s) with "
               f">= {min_cluster_size} beliefs",
               {"clusters_count": len(clusters_raw)})

    # Filter out beliefs already compressed in a previous session
    with adapter._driver.session() as session:
        already_compressed = session.run(
            "MATCH (b:Belief {project_id: $pid}) "
            "WHERE b.status = 'compressed' "
            "RETURN collect(b.name) AS names",
            pid=project_id,
        ).data()
    compressed_names = set(already_compressed[0]["names"]) if already_compressed else set()

    compressed_count = 0
    abstractions_created = 0
    results = []
    now = datetime.now(timezone.utc).isoformat()

    for i, cluster in enumerate(clusters_raw):
        entity = cluster["entity"]
        entity_type = cluster["entity_type"]
        beliefs = [b for b in cluster["beliefs"] if b not in compressed_names]

        if len(beliefs) < min_cluster_size:
            yield _msg("compression", "cluster_skip",
                       f"Cluster {i + 1} ({entity}): {len(beliefs)} uncompressed "
                       f"beliefs after filtering (< {min_cluster_size}), skipping")
            continue

        yield _msg("compression", "cluster_start",
                   f"Cluster {i + 1}/{len(clusters_raw)}: {entity} [{entity_type}] "
                   f"— {len(beliefs)} belief(s)",
                   {"entity": entity, "entity_type": entity_type,
                    "belief_count": len(beliefs)})

        # Fetch belief details for LLM
        with adapter._driver.session() as session:
            belief_details = session.run(
                "MATCH (b:Belief {project_id: $pid}) "
                "WHERE b.name IN $names "
                "RETURN b.name AS name, b.confidence AS confidence, "
                "b.summary AS summary",
                pid=project_id, names=beliefs,
            ).data()

        beliefs_text = "\n".join(
            f"  - {bd['name']} (confidence: {bd['confidence']}, "
            f"summary: {bd.get('summary') or 'none'})"
            for bd in belief_details
        )

        _compression_schema = {
            "type": "object",
            "properties": {
                "abstract_belief": {
                    "type": "string",
                    "maxLength": 200,
                    "description": "A single higher-level proposition that captures "
                                   "the collective meaning of these beliefs about "
                                   "the entity. Must be falsifiable.",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in the abstract belief (weighted by "
                                   "source belief confidences)",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of the abstraction logic",
                },
                "beliefs_used": {
                    "type": "integer",
                    "description": "Number of source beliefs that contributed to "
                                   "the abstraction (may be fewer than total if "
                                   "some are irrelevant)",
                },
            },
            "required": ["abstract_belief", "confidence", "reasoning", "beliefs_used"],
        }

        result = llm_call(
            "You compress multiple specific beliefs about the same entity/topic "
            "into a single higher-level abstraction. This is hippocampus-to-"
            "neocortex transfer: converting many episodic memories into one "
            "generalizable understanding. The abstract belief should be:\n"
            "  1. Higher-level than any individual source belief\n"
            "  2. Falsifiable — a testable proposition, not a vague summary\n"
            "  3. Faithful to the evidence — don't overstate\n"
            "  4. Max 200 characters\n"
            "If the beliefs are too diverse to meaningfully compress, set "
            "beliefs_used to 0 and explain why in reasoning.",
            f"Entity: {entity} [{entity_type}]\n\n"
            f"Beliefs about this entity:\n{beliefs_text}",
            temperature=0.3,
            json_schema=_compression_schema,
            model=_get_compression_model(),
        )

        if not result or "abstract_belief" not in result:
            yield _msg("compression", "cluster_skip",
                       f"Cluster {i + 1} ({entity}): LLM analysis unavailable")
            continue

        beliefs_used = result.get("beliefs_used", len(beliefs))
        if beliefs_used == 0:
            yield _msg("compression", "cluster_skip",
                       f"Cluster {i + 1} ({entity}): beliefs too diverse to "
                       f"compress — {result.get('reasoning', '')[:120]}")
            results.append({
                "entity": entity,
                "entity_type": entity_type,
                "original_beliefs": beliefs,
                "abstract_belief": None,
                "skipped": True,
                "reason": result.get("reasoning", ""),
            })
            continue

        abstract = result["abstract_belief"]
        confidence = result.get("confidence", 0.7)
        reasoning = result.get("reasoning", "")

        cluster_result = {
            "entity": entity,
            "entity_type": entity_type,
            "original_beliefs": beliefs,
            "abstract_belief": abstract,
            "confidence": confidence,
            "reasoning": reasoning,
            "beliefs_used": beliefs_used,
            "applied": not dry_run,
        }
        results.append(cluster_result)

        if dry_run:
            yield _msg("compression", "cluster_preview",
                       f"Would create: \"{abstract}\" (conf: {confidence:.2f}) "
                       f"from {beliefs_used} beliefs about {entity}",
                       cluster_result)
            compressed_count += 1
            continue

        # Write abstract belief and mark originals as compressed
        with adapter._driver.session() as session:
            # Create the abstract belief
            session.run(
                "MERGE (b:Belief {name: $name, project_id: $pid}) "
                "ON CREATE SET b.summary = $reasoning, "
                "b.created_at = $now, b.source_type = 'compression', "
                "b.active = true, b.status = 'active', "
                "b.confidence = $confidence, b.mentions = 1 "
                "ON MATCH SET b.confidence = CASE "
                "WHEN $confidence > b.confidence THEN $confidence "
                "ELSE b.confidence END, b.updated_at = $now",
                name=abstract, pid=project_id,
                reasoning=reasoning, now=now, confidence=confidence,
            )

            # Link abstract belief to the entity
            session.run(
                "MATCH (b:Belief {name: $belief_name, project_id: $pid}), "
                "(e {name: $entity_name, project_id: $pid}) "
                "WHERE any(lbl IN labels(e) WHERE lbl IN $entity_types) "
                "MERGE (b)-[r:APPLIES]->(e) "
                "ON CREATE SET r.confidence = $confidence, "
                "r.reason = 'compressed abstraction', "
                "r.source_type = 'compression', r.created_at = $now, "
                "r.active = true, r.weight = 0.9",
                belief_name=abstract, entity_name=entity,
                pid=project_id, now=now, confidence=confidence,
                entity_types=[t for t in NODE_TYPES if t != "Belief"],
            )

            # Mark originals as compressed
            for member in beliefs:
                session.run(
                    "MATCH (orig:Belief {name: $orig_name, project_id: $pid}) "
                    "MATCH (abst:Belief {name: $abst_name, project_id: $pid}) "
                    "MERGE (orig)-[r:SUPPORTS]->(abst) "
                    "ON CREATE SET r.confidence = 0.9, "
                    "r.reason = 'compressed into topic abstraction', "
                    "r.source_type = 'compression', r.created_at = $now, "
                    "r.active = true, r.weight = 0.9 "
                    "SET orig.status = 'compressed', orig.active = false, "
                    "orig.compressed_into = $abst_name, "
                    "orig.compressed_at = $now",
                    orig_name=member, abst_name=abstract,
                    pid=project_id, now=now,
                )

        compressed_count += 1
        abstractions_created += 1
        yield _msg("compression", "cluster_done",
                   f"Compressed: \"{abstract}\" (conf: {confidence:.2f}) "
                   f"← {len(beliefs)} beliefs about {entity}",
                   cluster_result)

    yield _msg("compression", "done",
               f"Compression {mode}: {compressed_count} cluster(s) processed, "
               f"{abstractions_created} abstraction(s) created",
               {"compressed": compressed_count,
                "abstractions_created": abstractions_created})

    return {
        "phase": "compression",
        "status": mode,
        "clusters_found": len(clusters_raw),
        "clusters_compressed": compressed_count,
        "abstractions_created": abstractions_created,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Phase 4: Bridging — cross-domain connection discovery (free association)
# ---------------------------------------------------------------------------

def bridge(
    adapter: Neo4jBaseAdapter,
    project_id: str = "default",
    pairs: int = 15,
    min_distance: int = 4,
    create_edges: bool = True,
) -> Generator[dict, None, dict]:
    """Pick random belief pairs from distant graph clusters and evaluate
    whether a meaningful relationship exists between them.

    Biologically inspired by free association — the "random dream" effect
    that connects concepts from completely different mental domains.

    Args:
        adapter: Backend adapter (Neo4j + vector store)
        project_id: Project to bridge
        pairs: Number of belief pairs to evaluate (default 15)
        min_distance: Minimum shortest-path hops to consider "distant" (default 4)
        create_edges: If True, write provisional edges for discovered connections

    Yields progress messages. Returns final result dict.
    """
    yield _msg("bridging", "start",
               f"Cross-domain bridging: evaluating up to {pairs} distant belief pairs "
               f"(min distance: {min_distance} hops)")

    # Fetch all active beliefs with their names
    with adapter._driver.session() as session:
        all_beliefs = session.run(
            "MATCH (b:Belief {project_id: $pid}) "
            "WHERE (b.status IS NULL OR b.status = 'active') "
            "AND (b.active = true OR b.active IS NULL) "
            "RETURN b.name AS name, b.confidence AS confidence",
            pid=project_id,
        ).data()

    if len(all_beliefs) < 2:
        yield _msg("bridging", "done",
                   f"Too few beliefs ({len(all_beliefs)}) for cross-domain bridging")
        return {
            "phase": "bridging",
            "status": "completed",
            "pairs_evaluated": 0,
            "connections_found": 0,
            "edges_created": 0,
        }

    yield _msg("bridging", "beliefs_loaded",
               f"Loaded {len(all_beliefs)} active beliefs",
               {"belief_count": len(all_beliefs)})

    # Sample candidate pairs and check graph distance
    # Strategy: random sample pairs, then verify distance via shortest path query
    candidate_pairs = []
    max_attempts = pairs * 5  # Allow some failures to find distant pairs
    attempts = 0

    yield _msg("bridging", "sampling",
               f"Sampling distant pairs (shortest path >= {min_distance} hops)...")

    with adapter._driver.session() as session:
        while len(candidate_pairs) < pairs and attempts < max_attempts:
            attempts += 1
            pair = random.sample(all_beliefs, 2)
            a_name, b_name = pair[0]["name"], pair[1]["name"]

            # Check shortest path length between the two beliefs
            result = session.run(
                "MATCH (a:Belief {name: $a_name, project_id: $pid}), "
                "      (b:Belief {name: $b_name, project_id: $pid}) "
                "OPTIONAL MATCH path = shortestPath((a)-[*..10]-(b)) "
                "RETURN CASE WHEN path IS NULL THEN -1 "
                "       ELSE length(path) END AS distance",
                a_name=a_name, b_name=b_name, pid=project_id,
            ).data()

            distance = result[0]["distance"] if result else -1

            # Accept if distant enough: path >= min_distance, or disconnected (-1)
            if distance >= min_distance or distance == -1:
                candidate_pairs.append({
                    "belief_a": a_name,
                    "confidence_a": pair[0]["confidence"],
                    "belief_b": b_name,
                    "confidence_b": pair[1]["confidence"],
                    "distance": distance,
                })

    if not candidate_pairs:
        yield _msg("bridging", "done",
                   f"No distant pairs found after {attempts} attempts "
                   f"(graph may be too densely connected)")
        return {
            "phase": "bridging",
            "status": "completed",
            "pairs_evaluated": 0,
            "connections_found": 0,
            "edges_created": 0,
        }

    yield _msg("bridging", "pairs_found",
               f"Found {len(candidate_pairs)} distant pair(s) "
               f"(from {attempts} attempts)",
               {"pairs_count": len(candidate_pairs), "attempts": attempts})

    # Evaluate each pair with LLM
    connections_found = 0
    edges_created = 0
    edge_details = []
    evaluated_pairs = []

    _bridging_schema = {
        "type": "object",
        "properties": {
            "relationship_exists": {
                "type": "boolean",
                "description": "Is there a meaningful, non-trivial relationship?",
            },
            "relationship_type": {
                "type": "string",
                "enum": list(RELATIONSHIP_TYPES),
                "description": "The type of relationship, if one exists",
            },
            "description": {
                "type": "string",
                "description": "Brief explanation of the connection (max 200 chars)",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "How confident are you in this connection?",
            },
        },
        "required": ["relationship_exists"],
    }

    for i, pair in enumerate(candidate_pairs):
        dist_label = "disconnected" if pair["distance"] == -1 else f"{pair['distance']} hops"
        yield _msg("bridging", "evaluating",
                   f"Pair {i + 1}/{len(candidate_pairs)}: "
                   f"'{pair['belief_a'][:60]}' ↔ '{pair['belief_b'][:60]}' "
                   f"({dist_label})")

        analysis = llm_call(
            "You evaluate whether two beliefs from different domains of a "
            "knowledge graph have a meaningful, non-trivial relationship. "
            "Most pairs will NOT be related — only say yes if there is a "
            "genuine conceptual connection worth recording. "
            "Be selective: a ~10-20% hit rate is expected.",
            f"Belief A: {pair['belief_a']}\n"
            f"  (confidence: {pair['confidence_a']})\n\n"
            f"Belief B: {pair['belief_b']}\n"
            f"  (confidence: {pair['confidence_b']})\n\n"
            f"Graph distance: {dist_label}\n\n"
            f"Is there a meaningful relationship between these beliefs?",
            temperature=0.3,
            json_schema=_bridging_schema,
            model=_get_bridging_model(),
        )

        pair_result = {
            "belief_a": pair["belief_a"],
            "belief_b": pair["belief_b"],
            "distance": pair["distance"],
            "analysis": analysis,
        }

        if analysis and analysis.get("relationship_exists"):
            connections_found += 1
            rel_type = analysis.get("relationship_type", "COMPLEMENTS")
            if rel_type not in RELATIONSHIP_TYPES:
                rel_type = "COMPLEMENTS"
            description = analysis.get("description", "Cross-domain bridge")
            confidence = analysis.get("confidence", 0.3)

            pair_result["connected"] = True
            pair_result["rel_type"] = rel_type

            yield _msg("bridging", "connection_found",
                       f"  CONNECTION: {pair['belief_a'][:50]} "
                       f"--[{rel_type}]--> {pair['belief_b'][:50]} "
                       f"({description[:80]})",
                       {"rel_type": rel_type, "description": description,
                        "confidence": confidence})

            if create_edges:
                from datetime import timedelta
                ttl_days = 7
                valid_until = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()

                rel = {
                    "source": pair["belief_a"],
                    "target": pair["belief_b"],
                    "type": rel_type,
                    "confidence": min(confidence, 0.3),  # Cap at 0.3 for provisional
                    "reason": f"Dream bridging: {description[:200]}",
                    "valid_until": valid_until,
                }

                try:
                    count = adapter.write_relationships(
                        [rel],
                        source_cycle="dream",
                        source_type="dream",
                        project_id=project_id,
                        actor="dreaming-bridging",
                    )
                    if count > 0:
                        edges_created += 1
                        detail = {
                            "source": pair["belief_a"],
                            "target": pair["belief_b"],
                            "rel_type": rel_type,
                            "description": description,
                            "valid_until": valid_until,
                        }
                        edge_details.append(detail)
                        yield _msg("bridging", "edge_created",
                                   f"  Edge written: {rel_type} "
                                   f"(conf: 0.3, TTL: {ttl_days}d)",
                                   detail)
                except Exception as e:
                    logger.warning("Bridge edge creation failed: %s -> %s: %s",
                                   pair["belief_a"], pair["belief_b"], e)
                    yield _msg("bridging", "edge_failed",
                               f"  Failed: {e}")
        else:
            pair_result["connected"] = False
            yield _msg("bridging", "no_connection",
                       f"  No connection (pair {i + 1})")

        evaluated_pairs.append(pair_result)

    hit_rate = round(100 * connections_found / max(len(candidate_pairs), 1), 1)
    edge_summary = f", {edges_created} edge(s) created" if edges_created else ""
    yield _msg("bridging", "done",
               f"Bridging complete: {len(candidate_pairs)} pairs evaluated, "
               f"{connections_found} connection(s) found ({hit_rate}% hit rate)"
               f"{edge_summary}",
               {"pairs_evaluated": len(candidate_pairs),
                "connections_found": connections_found,
                "hit_rate": hit_rate,
                "edges_created": edges_created})

    return {
        "phase": "bridging",
        "status": "completed",
        "pairs_evaluated": len(candidate_pairs),
        "connections_found": connections_found,
        "hit_rate": hit_rate,
        "edges_created": edges_created,
        "edge_details": edge_details,
        "evaluated_pairs": evaluated_pairs,
    }


# ---------------------------------------------------------------------------
# Phase 5: Reflection — structural health analysis
# ---------------------------------------------------------------------------

def reflect(
    adapter: Neo4jBaseAdapter,
    project_id: str = "default",
) -> Generator[dict, None, dict]:
    """Structural pattern analysis — meta-cognition over the knowledge graph.

    Yields progress messages. Returns final result dict.
    """
    yield _msg("reflection", "start", "Analyzing graph structure...")

    with adapter._driver.session() as session:
        # Hub analysis
        yield _msg("reflection", "analyzing_hubs", "Identifying hub nodes...")
        hubs = session.run(
            "MATCH (n {project_id: $pid})-[r]-() "
            "WHERE any(lbl IN labels(n) WHERE lbl IN $types) "
            "WITH n, labels(n)[0] AS type, count(r) AS degree "
            "ORDER BY degree DESC LIMIT 10 "
            "RETURN n.name AS name, type, degree",
            pid=project_id, types=NODE_TYPES,
        ).data()

        # Relationship distribution
        yield _msg("reflection", "analyzing_rels", "Checking relationship distribution...")
        rel_dist = session.run(
            "MATCH (a {project_id: $pid})-[r]->(b {project_id: $pid}) "
            "RETURN type(r) AS rel, count(r) AS cnt ORDER BY cnt DESC",
            pid=project_id,
        ).data()

        # Node type distribution
        type_counts = session.run(
            "MATCH (n {project_id: $pid}) "
            "WHERE any(lbl IN labels(n) WHERE lbl IN $types) "
            "WITH labels(n)[0] AS type, count(n) AS cnt "
            "RETURN type, cnt ORDER BY cnt DESC",
            pid=project_id, types=NODE_TYPES,
        ).data()

        total_nodes = sum(t["cnt"] for t in type_counts)
        total_edges = sum(r["cnt"] for r in rel_dist)

        yield _msg("reflection", "topology",
                   f"Graph: {total_nodes} nodes, {total_edges} edges",
                   {"nodes": total_nodes, "edges": total_edges,
                    "edge_node_ratio": round(total_edges / max(total_nodes, 1), 2)})

        # Low-confidence beliefs
        yield _msg("reflection", "analyzing_beliefs", "Checking belief health...")
        low_conf = session.run(
            "MATCH (b:Belief {project_id: $pid}) "
            "WHERE (b.active = true OR b.active IS NULL) "
            "AND b.confidence <= 0.5 "
            "RETURN b.name AS name, b.confidence AS conf "
            "ORDER BY b.confidence ASC LIMIT 10",
            pid=project_id,
        ).data()

        # Orphan nodes (degree <= 1)
        yield _msg("reflection", "analyzing_orphans", "Detecting orphan nodes...")
        orphans = session.run(
            "MATCH (n {project_id: $pid}) "
            "WHERE any(lbl IN labels(n) WHERE lbl IN $types) "
            "OPTIONAL MATCH (n)-[r]-() "
            "WITH n, labels(n)[0] AS type, count(r) AS degree "
            "WHERE degree <= 1 "
            "RETURN type, count(*) AS cnt ORDER BY cnt DESC",
            pid=project_id, types=NODE_TYPES,
        ).data()

        total_orphans = sum(o["cnt"] for o in orphans)
        orphan_pct = round(100 * total_orphans / max(total_nodes, 1), 1)

        yield _msg("reflection", "orphan_report",
                   f"Orphans: {total_orphans}/{total_nodes} ({orphan_pct}%)",
                   {"orphans": total_orphans, "orphan_pct": orphan_pct,
                    "by_type": orphans})

        # Belief statistics
        belief_stats = session.run(
            "MATCH (b:Belief {project_id: $pid}) "
            "RETURN b.status AS status, count(b) AS cnt "
            "ORDER BY cnt DESC",
            pid=project_id,
        ).data()

        contradictions = session.run(
            "MATCH (a:Belief {project_id: $pid})-[r:CONTRADICTS]->"
            "(b:Belief {project_id: $pid}) "
            "WHERE a.active = true AND b.active = true "
            "RETURN count(r) AS cnt",
            pid=project_id,
        ).data()
        contradiction_count = contradictions[0]["cnt"] if contradictions else 0

    # Build health assessment
    issues = []
    if orphan_pct > 20:
        issues.append(f"High orphan rate ({orphan_pct}%) — many nodes lack connections")
    if total_edges / max(total_nodes, 1) < 1.5:
        issues.append(f"Low edge/node ratio ({total_edges / max(total_nodes, 1):.1f}) — graph is sparse")
    if hubs and hubs[0]["degree"] > total_edges * 0.3:
        issues.append(f"Hub over-centralization: '{hubs[0]['name']}' has {hubs[0]['degree']}/{total_edges} edges")
    if low_conf:
        issues.append(f"{len(low_conf)} low-confidence belief(s) need evidence or review")
    if contradiction_count > 0:
        issues.append(f"{contradiction_count} unresolved contradiction(s)")

    health_score = max(0.0, 1.0 - len(issues) * 0.15)

    result = {
        "phase": "reflection",
        "status": "completed",
        "health_score": round(health_score, 2),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "edge_node_ratio": round(total_edges / max(total_nodes, 1), 2),
        "hubs": hubs[:5],
        "orphans": {"total": total_orphans, "pct": orphan_pct, "by_type": orphans},
        "belief_stats": belief_stats,
        "low_confidence_beliefs": low_conf,
        "contradictions": contradiction_count,
        "relationship_distribution": rel_dist,
        "type_distribution": type_counts,
        "issues": issues,
    }

    for issue in issues:
        yield _msg("reflection", "issue", issue)

    yield _msg("reflection", "done",
               f"Health score: {health_score:.0%} — {len(issues)} issue(s)",
               {"health_score": health_score, "issues_count": len(issues)})

    return result


# ---------------------------------------------------------------------------
# Reconsolidation — importance-weighted confidence strengthening (Proposal #5)
# ---------------------------------------------------------------------------

def reconsolidate(
    adapter: Neo4jBaseAdapter,
    project_id: str = "default",
    min_access: int = RECONSOLIDATION_MIN_ACCESS,
    factor: float = RECONSOLIDATION_FACTOR,
    max_boost: float = RECONSOLIDATION_MAX_BOOST,
    dry_run: bool = False,
) -> Generator[dict, None, dict]:
    """Strengthen confidence of frequently-accessed beliefs (reconsolidation).

    Biological basis: memories that are frequently recalled undergo
    reconsolidation, strengthening synaptic traces. This counters
    confidence decay for actively used knowledge — the inverse of
    forgetting. Only beliefs with access_count >= min_access qualify.

    The boost is proportional to access frequency but capped:
      boost = min(access_count * factor, max_boost)
      new_confidence = min(old_confidence + boost, 1.0)

    S5 beliefs and facts (knowledge_type='fact') are exempt — they have
    their own stability mechanisms.

    Args:
        adapter: Backend adapter (Neo4j + vector store)
        project_id: Project scope
        min_access: Minimum access_count to qualify (default 3)
        factor: Confidence boost per access (default 0.01)
        max_boost: Maximum boost per cycle (default 0.15)
        dry_run: If True, preview without applying

    Yields progress messages. Returns result dict.
    """
    mode = "preview" if dry_run else "apply"
    yield _msg("reconsolidation", "start",
               f"Importance-weighted strengthening ({mode} mode, "
               f"min_access={min_access}, factor={factor}, max_boost={max_boost})")

    strengthened = []
    skipped = 0

    try:
        with adapter._driver.session() as session:
            # Find active beliefs with sufficient access_count
            candidates = session.run(
                "MATCH (b:Belief {project_id: $pid}) "
                "WHERE b.active = true "
                "AND b.confidence IS NOT NULL "
                "AND b.access_count IS NOT NULL "
                "AND b.access_count >= $min_access "
                "AND b.confidence < 1.0 "
                "AND COALESCE(b.vsm_level, 'S1') <> 'S5' "
                "AND COALESCE(b.knowledge_type, 'unclassified') <> 'fact' "
                "RETURN b.name AS name, b.confidence AS confidence, "
                "b.access_count AS access_count, "
                "COALESCE(b.knowledge_type, 'unclassified') AS knowledge_type, "
                "COALESCE(b.vsm_level, 'S1') AS vsm_level",
                pid=project_id, min_access=min_access,
            ).data()

            yield _msg("reconsolidation", "candidates",
                       f"Found {len(candidates)} belief(s) with access_count >= {min_access}")

            now_iso = datetime.now(timezone.utc).isoformat()

            for c in candidates:
                ac = c["access_count"]
                old_conf = c["confidence"]
                boost = min(ac * factor, max_boost)
                new_conf = round(min(old_conf + boost, 1.0), 4)

                # Skip if change is negligible (< 0.001)
                if abs(new_conf - old_conf) < 0.001:
                    skipped += 1
                    continue

                entry = {
                    "name": c["name"],
                    "knowledge_type": c["knowledge_type"],
                    "access_count": ac,
                    "old_confidence": old_conf,
                    "new_confidence": new_conf,
                    "boost": round(boost, 4),
                }
                strengthened.append(entry)

                if not dry_run:
                    session.run(
                        "MATCH (b:Belief {project_id: $pid, name: $name}) "
                        "SET b.confidence = $new_conf, "
                        "    b.reconsolidated_at = $now, "
                        "    b.last_boost = $boost",
                        pid=project_id, name=c["name"],
                        new_conf=new_conf, now=now_iso,
                        boost=round(boost, 4),
                    )

    except Exception as e:
        logger.warning("Reconsolidation failed: %s", e)
        yield _msg("reconsolidation", "error", f"Reconsolidation error: {e}")
        return {
            "phase": "reconsolidation", "status": "error",
            "error": str(e), "strengthened": [], "count": 0,
        }

    # Progress: show top strengthened beliefs
    for entry in strengthened[:10]:
        yield _msg("reconsolidation", "strengthened",
                   f"  {entry['name']}: {entry['old_confidence']:.3f} → "
                   f"{entry['new_confidence']:.3f} "
                   f"(+{entry['boost']:.3f}, {entry['access_count']} accesses)")

    if len(strengthened) > 10:
        yield _msg("reconsolidation", "truncated",
                   f"  ... and {len(strengthened) - 10} more")

    yield _msg("reconsolidation", "done",
               f"Reconsolidation: {len(strengthened)} belief(s) "
               f"{'would be ' if dry_run else ''}strengthened, {skipped} skipped",
               {"strengthened": len(strengthened), "skipped": skipped,
                "dry_run": dry_run})

    return {
        "phase": "reconsolidation",
        "status": "completed",
        "mode": mode,
        "count": len(strengthened),
        "skipped": skipped,
        "items": strengthened,
        "params": {
            "min_access": min_access,
            "factor": factor,
            "max_boost": max_boost,
        },
    }


# ---------------------------------------------------------------------------
# Phase 6: Maintenance — confidence decay and knowledge hygiene
# ---------------------------------------------------------------------------

def maintain(
    adapter: Neo4jBaseAdapter,
    project_id: str = "default",
    dry_run: bool = False,
) -> Generator[dict, None, dict]:
    """Synaptic pruning — apply confidence decay, expire TTLs, prune
    orphans, deduplicate edges, and surface review needs.

    Biologically inspired: during sleep, synapses are pruned and
    strengthened based on recent use. This phase performs active graph
    maintenance — the knowledge equivalent of garbage collection.

    Steps:
        1. Confidence decay (existing — stale beliefs lose confidence)
        2. TTL enforcement (expire_nodes — provisional edges and timed nodes)
        3. Orphan pruning (prune_orphan_nodes — 0-edge stale nodes)
        4. Edge deduplication (deduplicate_edges — Cartesian product cleanup)
        5. Review queue (surface items needing human review)
        6. Governance statistics summary

    Args:
        adapter: Backend adapter (Neo4j + vector store)
        project_id: Project to maintain
        dry_run: If True, preview changes without applying

    Yields progress messages. Returns final result dict.
    """
    mode = "preview" if dry_run else "apply"
    yield _msg("maintenance", "start",
               f"Knowledge maintenance ({mode} mode)...")

    # Step 0 (v2.0): TTL extension based on access_count
    # Nodes with access_count > 0 get their valid_until extended using
    # linear scaling: TTL_days = base_ttl * (1 + access_count * TTL_SCALE_FACTOR)
    # Norman's directive: ~10 activations ≈ 1 year
    yield _msg("maintenance", "ttl_extend_start",
               "Extending TTLs based on access frequency...")
    ttl_extended = 0
    try:
        with adapter._driver.session() as session:
            # Find nodes with access_count > 0 AND a valid_until that could expire
            candidates = session.run(
                "MATCH (n {project_id: $pid}) "
                "WHERE n.access_count IS NOT NULL AND n.access_count > 0 "
                "AND n.valid_until IS NOT NULL "
                "AND (n.expired_at IS NULL) "
                "AND any(lbl IN labels(n) WHERE lbl IN $types) "
                "RETURN n.name AS name, labels(n)[0] AS type, "
                "n.access_count AS access_count, "
                "n.valid_until AS valid_until, "
                "n.created_at AS created_at, "
                "COALESCE(n.vsm_level, 'S1') AS vsm_level",
                pid=project_id, types=NODE_TYPES,
            ).data()

            from merkraum_backend import VSM_DEFAULT_TTL_DAYS

            for c in candidates:
                ac = c["access_count"]
                vsm = c.get("vsm_level", "S1")
                base_ttl = VSM_DEFAULT_TTL_DAYS.get(vsm) or 30

                # S5 nodes are never expired (importance shield)
                if vsm == "S5":
                    continue

                new_ttl_days = calculate_ttl_days(base_ttl, ac)

                # Calculate new valid_until from created_at + extended TTL
                created_at = c.get("created_at")
                if not created_at:
                    continue

                try:
                    from datetime import timedelta
                    created_dt = datetime.fromisoformat(
                        created_at.replace("Z", "+00:00"))
                    new_valid = (created_dt
                                + timedelta(days=new_ttl_days)).isoformat()

                    # Only extend, never shorten
                    current_valid = c.get("valid_until", "")
                    if new_valid > current_valid and not dry_run:
                        session.run(
                            "MATCH (n {project_id: $pid, name: $name}) "
                            "SET n.valid_until = $new_valid, "
                            "    n.ttl_extended_at = $now, "
                            "    n.ttl_effective_days = $ttl_days",
                            pid=project_id, name=c["name"],
                            new_valid=new_valid,
                            now=datetime.now(timezone.utc).isoformat(),
                            ttl_days=new_ttl_days,
                        )
                        ttl_extended += 1
                    elif new_valid > current_valid:
                        ttl_extended += 1  # Count for dry_run
                except (ValueError, TypeError):
                    continue

    except Exception as e:
        logger.warning("TTL extension failed: %s", e)
        yield _msg("maintenance", "ttl_extend_error", f"TTL extension error: {e}")

    yield _msg("maintenance", "ttl_extend_done",
               f"TTL: {ttl_extended} node(s) {'would be ' if dry_run else ''}"
               f"extended based on access frequency",
               {"extended": ttl_extended, "scale_factor": TTL_SCALE_FACTOR})

    # Step 1: Apply confidence decay
    yield _msg("maintenance", "decay_start", "Applying confidence decay...")
    decay_result = adapter.apply_confidence_decay(
        project_id=project_id, dry_run=dry_run, actor="dreaming-maintenance")

    decayed_count = decay_result.get("total", 0)
    unchanged_count = decay_result.get("unchanged", 0)

    yield _msg("maintenance", "decay_done",
               f"Decay: {decayed_count} belief(s) {'would be ' if dry_run else ''}adjusted, "
               f"{unchanged_count} unchanged",
               {"decayed": decayed_count, "unchanged": unchanged_count,
                "dry_run": dry_run})

    # Log individual decays for transparency
    for entry in decay_result.get("decayed", [])[:10]:  # Cap at 10 for progress
        yield _msg("maintenance", "decay_item",
                   f"  {entry['name']}: {entry['old_confidence']:.3f} → "
                   f"{entry['new_confidence']:.3f} "
                   f"({entry['days_since_update']:.0f}d, {entry['knowledge_type'] or 'unclassified'})")

    if decayed_count > 10:
        yield _msg("maintenance", "decay_truncated",
                   f"  ... and {decayed_count - 10} more")

    # Step 1b: Reconsolidation — importance-weighted strengthening (Proposal #5)
    # Counters decay for frequently-accessed beliefs. Runs after decay so
    # both forces apply: stale beliefs decay, actively-used beliefs strengthen.
    gen = reconsolidate(adapter, project_id, dry_run=dry_run)
    reconsolidation_result = None
    try:
        while True:
            progress = next(gen)
            yield progress
    except StopIteration as e:
        reconsolidation_result = e.value
    reconsolidated_count = (reconsolidation_result or {}).get("count", 0)

    # Step 2: TTL enforcement — expire nodes past their valid_until
    yield _msg("maintenance", "expire_start",
               "Enforcing TTL expirations...")
    expire_result = adapter.expire_nodes(
        project_id=project_id, dry_run=dry_run, actor="dreaming-maintenance")

    expired_count = expire_result.get("total", 0)
    yield _msg("maintenance", "expire_done",
               f"TTL: {expired_count} node(s) {'would be ' if dry_run else ''}expired",
               {"expired": expired_count, "dry_run": dry_run})

    for entry in expire_result.get("expired", [])[:5]:
        yield _msg("maintenance", "expire_item",
                   f"  {entry['name']} ({entry['type']}) — "
                   f"valid_until: {entry.get('valid_until', 'unknown')}")

    if expired_count > 5:
        yield _msg("maintenance", "expire_truncated",
                   f"  ... and {expired_count - 5} more")

    # Step 3: Orphan pruning — deactivate 0-edge stale nodes
    yield _msg("maintenance", "prune_start",
               "Pruning orphan nodes (0 edges, 30+ days stale)...")
    prune_result = adapter.prune_orphan_nodes(
        project_id=project_id, stale_days=30,
        dry_run=dry_run, actor="dreaming-maintenance")

    pruned_count = prune_result.get("total", 0)
    yield _msg("maintenance", "prune_done",
               f"Orphans: {pruned_count} node(s) {'would be ' if dry_run else ''}pruned",
               {"pruned": pruned_count, "dry_run": dry_run})

    for entry in prune_result.get("pruned", [])[:5]:
        yield _msg("maintenance", "prune_item",
                   f"  {entry['name']} ({entry['type']}) — "
                   f"last update: {entry.get('last_update', 'unknown')}")

    if pruned_count > 5:
        yield _msg("maintenance", "prune_truncated",
                   f"  ... and {pruned_count - 5} more")

    # Step 4: Edge deduplication — clean up Cartesian product artifacts
    yield _msg("maintenance", "dedup_start",
               "Checking for duplicate edges...")
    dedup_result = adapter.deduplicate_edges(
        project_id=project_id, dry_run=dry_run,
        actor="dreaming-maintenance")

    dedup_found = dedup_result.get("duplicates_found", 0)
    dedup_removed = dedup_result.get("edges_removed", 0)
    yield _msg("maintenance", "dedup_done",
               f"Dedup: {dedup_found} duplicate(s) found, "
               f"{dedup_removed} {'would be ' if dry_run else ''}removed",
               {"duplicates_found": dedup_found,
                "edges_removed": dedup_removed, "dry_run": dry_run})

    # Step 5: Surface review queue
    yield _msg("maintenance", "review_start", "Checking certainty review queue...")
    review_result = adapter.get_certainty_review_queue(
        project_id=project_id, limit=20)

    review_categories = review_result.get("categories", {})
    total_review = sum(
        len(items) for items in review_categories.values()
    )

    category_summary = {
        cat: len(items)
        for cat, items in review_categories.items()
        if items
    }

    yield _msg("maintenance", "review_done",
               f"Review queue: {total_review} item(s) across "
               f"{len(category_summary)} categor{'y' if len(category_summary) == 1 else 'ies'}",
               {"total": total_review, "categories": category_summary})

    # Step 6: Certainty statistics summary
    yield _msg("maintenance", "stats_start", "Computing governance statistics...")
    stats_result = adapter.get_certainty_stats(project_id=project_id)

    governance = stats_result.get("governance", {})
    health_status = governance.get("status", "unknown")

    yield _msg("maintenance", "stats_done",
               f"Governance health: {health_status}",
               {"governance": governance})

    result = {
        "phase": "maintenance",
        "status": "completed",
        "mode": mode,
        "ttl_extended": {
            "count": ttl_extended,
            "scale_factor": TTL_SCALE_FACTOR,
        },
        "decay": {
            "applied": not dry_run,
            "decayed_count": decayed_count,
            "unchanged_count": unchanged_count,
            "items": decay_result.get("decayed", []),
        },
        "reconsolidation": reconsolidation_result or {
            "count": 0, "skipped": 0,
        },
        "expired": {
            "count": expired_count,
            "items": expire_result.get("expired", []),
        },
        "pruned": {
            "count": pruned_count,
            "items": prune_result.get("pruned", []),
        },
        "dedup": {
            "duplicates_found": dedup_found,
            "edges_removed": dedup_removed,
        },
        "review_queue": {
            "total": total_review,
            "categories": category_summary,
        },
        "governance_health": health_status,
    }

    yield _msg("maintenance", "done",
               f"Maintenance complete — {ttl_extended} TTL extended, "
               f"{decayed_count} decay(s), {reconsolidated_count} reconsolidated, "
               f"{expired_count} expired, {pruned_count} pruned, "
               f"{dedup_removed} deduped, "
               f"{total_review} review item(s), health: {health_status}",
               {"ttl_extended": ttl_extended,
                "decayed": decayed_count,
                "reconsolidated": reconsolidated_count,
                "expired": expired_count,
                "pruned": pruned_count, "deduped": dedup_removed,
                "review_items": total_review, "health": health_status})

    return result


# ---------------------------------------------------------------------------
# Full Dream Session v2.0 — 4-phase orchestrator
# ---------------------------------------------------------------------------

def _run_phase(gen):
    """Helper: consume a generator, yielding progress, returning result."""
    result = None
    try:
        while True:
            yield next(gen)
    except StopIteration as e:
        result = e.value
    return result


def dream(
    adapter: Neo4jBaseAdapter,
    project_id: str = "default",
    phases: list[str] | None = None,
    # Walk parameters (v2.0)
    walk_steps: int = 20,
    walk_count: int = 5,
    walk_p_graph: float = 0.70,
    walk_p_semantic: float = 0.20,
    walk_p_random: float = 0.10,
    walk_create_edges: bool = True,
    seed: str | None = None,
    # Consolidation parameters
    consolidation_threshold: float = 0.75,
    consolidation_dry_run: bool = False,
    compression_min_cluster: int = 3,
    compression_max_clusters: int = 10,
    compression_dry_run: bool = False,
    # Maintenance parameters
    maintenance_dry_run: bool = False,
    # Legacy parameters (mapped to v2.0 equivalents)
    replay_hops: int | None = None,
    replay_walks: int | None = None,
    replay_create_edges: bool | None = None,
    bridging_pairs: int = 15,
    bridging_create_edges: bool = True,
) -> Generator[dict, None, dict]:
    """Run a full dreaming session.

    v2.0 architecture: Walk → Consolidate → Reflect → Maintain

    Legacy phase names ('replay', 'bridging', 'compression') are accepted
    and mapped to v2.0 equivalents for backward compatibility.

    Args:
        adapter: Backend adapter (Neo4j + vector store)
        project_id: Project to dream about
        phases: Which phases to run (default: all four v2.0 phases)
        walk_steps: Steps per walk (default 20)
        walk_count: Number of walks (default 5)
        walk_p_graph/walk_p_semantic/walk_p_random: Movement probabilities
        walk_create_edges: Create provisional edges from walk observations
        seed: Optional starting entity for first walk
        consolidation_threshold: Similarity threshold for belief clustering
        consolidation_dry_run: Preview consolidation without applying
        compression_min_cluster: Min beliefs per entity for compression
        compression_max_clusters: Max clusters to compress
        compression_dry_run: Preview compression without applying
        maintenance_dry_run: Preview maintenance without applying
        replay_hops/replay_walks/replay_create_edges: Legacy aliases
        bridging_pairs/bridging_create_edges: Legacy (bridging in Walk now)

    Yields progress messages. Returns combined result dict.
    """
    # Map legacy parameters
    if replay_hops is not None:
        walk_steps = replay_hops
    if replay_walks is not None:
        walk_count = replay_walks
    if replay_create_edges is not None:
        walk_create_edges = replay_create_edges

    # Map legacy phase names to v2.0
    v2_default = ["walk", "consolidation", "reflection", "maintenance"]
    if phases is None:
        active_phases = v2_default
    else:
        active_phases = []
        for p in phases:
            if p in ("replay", "bridging"):
                if "walk" not in active_phases:
                    active_phases.append("walk")
            elif p == "compression":
                if "consolidation" not in active_phases:
                    active_phases.append("consolidation")
            elif p in ("walk", "consolidation", "reflection", "maintenance"):
                if p not in active_phases:
                    active_phases.append(p)

    session_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    yield _msg("dream", "session_start",
               f"Dream session {session_id} (v2.0) — "
               f"phases: {', '.join(active_phases)}",
               {"session_id": session_id, "phases": active_phases,
                "architecture": "v2.0"})

    results = {"session_id": session_id, "phases": {},
               "architecture": "v2.0"}

    # Phase 1: Walk (replaces replay + bridging)
    if "walk" in active_phases:
        yield _msg("dream", "phase_start",
                   "Phase 1: Walk (hippocampal replay + free association)")
        gen = walk(adapter, project_id,
                   steps=walk_steps, walks=walk_count,
                   p_graph=walk_p_graph, p_semantic=walk_p_semantic,
                   p_random=walk_p_random,
                   create_edges=walk_create_edges, seed=seed)
        walk_result = None
        try:
            while True:
                progress = next(gen)
                yield progress
        except StopIteration as e:
            walk_result = e.value
        results["phases"]["walk"] = walk_result or {}

    # Phase 2: Consolidation (merging + compression)
    if "consolidation" in active_phases:
        yield _msg("dream", "phase_start",
                   "Phase 2: Consolidation (memory transfer + compression)")

        # 2a: Textual similarity consolidation
        gen = consolidate(adapter, project_id,
                          similarity_threshold=consolidation_threshold,
                          dry_run=consolidation_dry_run)
        consolidation_result = None
        try:
            while True:
                progress = next(gen)
                yield progress
        except StopIteration as e:
            consolidation_result = e.value
        results["phases"]["consolidation"] = consolidation_result or {}

        # 2b: Entity-based compression
        gen = compress(adapter, project_id,
                       min_cluster_size=compression_min_cluster,
                       max_clusters=compression_max_clusters,
                       dry_run=compression_dry_run)
        compression_result = None
        try:
            while True:
                progress = next(gen)
                yield progress
        except StopIteration as e:
            compression_result = e.value
        results["phases"]["compression"] = compression_result or {}

    # Phase 3: Reflection (structural health)
    if "reflection" in active_phases:
        yield _msg("dream", "phase_start",
                   "Phase 3: Reflection (structural health)")
        gen = reflect(adapter, project_id)
        reflection_result = None
        try:
            while True:
                progress = next(gen)
                yield progress
        except StopIteration as e:
            reflection_result = e.value
        results["phases"]["reflection"] = reflection_result or {}

    # Phase 4: Maintenance (synaptic pruning)
    if "maintenance" in active_phases:
        yield _msg("dream", "phase_start",
                   "Phase 4: Maintenance (synaptic pruning)")
        gen = maintain(adapter, project_id, dry_run=maintenance_dry_run)
        maintenance_result = None
        try:
            while True:
                progress = next(gen)
                yield progress
        except StopIteration as e:
            maintenance_result = e.value
        results["phases"]["maintenance"] = maintenance_result or {}

    elapsed = round(time.time() - start_time, 1)
    results["duration_seconds"] = elapsed
    results["status"] = "completed"

    yield _msg("dream", "session_complete",
               f"Dream session {session_id} complete ({elapsed}s)",
               {"duration_seconds": elapsed})

    return results
