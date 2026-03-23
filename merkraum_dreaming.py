#!/usr/bin/env python3
"""
Merkraum Dreaming Engine — neuroscience-inspired memory consolidation.

Universal logic module: no Flask, no hosting dependencies.
Designed for open-source sharing — all hosting/user logic stays in the API layer.

Five operations inspired by memory consolidation research
(McClelland et al., O'Reilly & Frank):

1. Replay (hippocampal replay): Random walks through the graph surface
   unexpected connections between distant knowledge clusters.
2. Consolidation (hippocampus-to-neocortex transfer): Episodic beliefs
   are merged into abstract, generalizable beliefs.
3. Bridging (free association): Cross-domain connection discovery between
   distant belief clusters — picks random belief pairs with high graph
   distance and evaluates whether a meaningful relationship exists.
4. Reflection (Default Mode Network): Structural health analysis —
   orphan detection, hub over-centralization, schema discipline.
5. Maintenance (synaptic pruning): Confidence decay on stale beliefs,
   certainty review queue, governance health assessment.

All operations yield progress messages via generators, enabling
async monitoring and live visualization in the frontend.

v1.1 — Bridging phase added (2026-03-23)
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
# Model configuration — SUP-159: Dream Analysis Model Upgrade
# ---------------------------------------------------------------------------
# Replay: lightweight analysis of graph walks → Haiku (cost-efficient, many calls)
# Consolidation: quality abstraction of episodic beliefs → Sonnet (accuracy matters)
# Reflection: no LLM (pure structural analysis)

_DEFAULT_REPLAY_MODEL = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
_DEFAULT_CONSOLIDATION_MODEL = "eu.anthropic.claude-sonnet-4-6"
_DEFAULT_BRIDGING_MODEL = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"


def _get_replay_model() -> str | None:
    """Return the model for replay phase (env override or default)."""
    return os.environ.get("MERKRAUM_DREAMING_REPLAY_MODEL") or _DEFAULT_REPLAY_MODEL


def _get_consolidation_model() -> str | None:
    """Return the model for consolidation phase (env override or default)."""
    return os.environ.get("MERKRAUM_DREAMING_CONSOLIDATION_MODEL") or _DEFAULT_CONSOLIDATION_MODEL


def _get_bridging_model() -> str | None:
    """Return the model for bridging phase (env override or default)."""
    return os.environ.get("MERKRAUM_DREAMING_BRIDGING_MODEL") or _DEFAULT_BRIDGING_MODEL


def get_dreaming_config() -> dict:
    """Return current dreaming model configuration for diagnostics."""
    return {
        "replay_model": _get_replay_model(),
        "consolidation_model": _get_consolidation_model(),
        "bridging_model": _get_bridging_model(),
        "reflection_model": None,  # No LLM used
        "replay_model_source": "env" if os.environ.get("MERKRAUM_DREAMING_REPLAY_MODEL") else "default",
        "consolidation_model_source": "env" if os.environ.get("MERKRAUM_DREAMING_CONSOLIDATION_MODEL") else "default",
        "bridging_model_source": "env" if os.environ.get("MERKRAUM_DREAMING_BRIDGING_MODEL") else "default",
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
# Phase 1: Replay — random walks with serendipitous discovery
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
# Phase 3: Bridging — cross-domain connection discovery (free association)
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
# Phase 4: Reflection — structural health analysis
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
# Phase 4: Maintenance — confidence decay and knowledge hygiene
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
        "decay": {
            "applied": not dry_run,
            "decayed_count": decayed_count,
            "unchanged_count": unchanged_count,
            "items": decay_result.get("decayed", []),
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
               f"Maintenance complete — {decayed_count} decay(s), "
               f"{expired_count} expired, {pruned_count} pruned, "
               f"{dedup_removed} deduped, "
               f"{total_review} review item(s), health: {health_status}",
               {"decayed": decayed_count, "expired": expired_count,
                "pruned": pruned_count, "deduped": dedup_removed,
                "review_items": total_review, "health": health_status})

    return result


# ---------------------------------------------------------------------------
# Full Dream Session — orchestrates all four phases
# ---------------------------------------------------------------------------

def dream(
    adapter: Neo4jBaseAdapter,
    project_id: str = "default",
    phases: list[str] | None = None,
    replay_hops: int = 5,
    replay_walks: int = 3,
    consolidation_threshold: float = 0.75,
    consolidation_dry_run: bool = False,
    seed: str | None = None,
    maintenance_dry_run: bool = False,
    replay_create_edges: bool = True,
    bridging_pairs: int = 15,
    bridging_create_edges: bool = True,
) -> Generator[dict, None, dict]:
    """Run a full dreaming session (replay → consolidation → bridging → reflection → maintenance).

    Args:
        adapter: Backend adapter (Neo4j + vector store)
        project_id: Project to dream about
        phases: Which phases to run (default: all five)
        replay_hops: Steps per random walk
        replay_walks: Number of random walks
        consolidation_threshold: Similarity threshold for belief clustering
        consolidation_dry_run: If True, preview consolidation without applying
        seed: Optional starting entity for replay
        maintenance_dry_run: If True, preview confidence decay without applying
        replay_create_edges: If True, create provisional edges from dream observations
        bridging_pairs: Number of distant belief pairs to evaluate (default 15)
        bridging_create_edges: If True, create provisional edges from bridge discoveries

    Yields progress messages. Returns combined result dict.
    """
    active_phases = phases or ["replay", "consolidation", "bridging", "reflection", "maintenance"]
    session_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    yield _msg("dream", "session_start",
               f"Dream session {session_id} — phases: {', '.join(active_phases)}",
               {"session_id": session_id, "phases": active_phases})

    results = {"session_id": session_id, "phases": {}}

    if "replay" in active_phases:
        yield _msg("dream", "phase_start", "Phase 1: Replay (hippocampal)")
        gen = replay(adapter, project_id, hops=replay_hops,
                     walks=replay_walks, seed=seed,
                     create_edges=replay_create_edges)
        replay_result = None
        try:
            while True:
                progress = next(gen)
                yield progress
        except StopIteration as e:
            replay_result = e.value
        results["phases"]["replay"] = replay_result or {}

    if "consolidation" in active_phases:
        yield _msg("dream", "phase_start", "Phase 2: Consolidation (memory transfer)")
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

    if "bridging" in active_phases:
        yield _msg("dream", "phase_start", "Phase 3: Bridging (cross-domain free association)")
        gen = bridge(adapter, project_id, pairs=bridging_pairs,
                     create_edges=bridging_create_edges)
        bridging_result = None
        try:
            while True:
                progress = next(gen)
                yield progress
        except StopIteration as e:
            bridging_result = e.value
        results["phases"]["bridging"] = bridging_result or {}

    if "reflection" in active_phases:
        yield _msg("dream", "phase_start", "Phase 4: Reflection (structural health)")
        gen = reflect(adapter, project_id)
        reflection_result = None
        try:
            while True:
                progress = next(gen)
                yield progress
        except StopIteration as e:
            reflection_result = e.value
        results["phases"]["reflection"] = reflection_result or {}

    if "maintenance" in active_phases:
        yield _msg("dream", "phase_start", "Phase 5: Maintenance (synaptic pruning)")
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
