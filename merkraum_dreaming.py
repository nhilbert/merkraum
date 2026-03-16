#!/usr/bin/env python3
"""
Merkraum Dreaming Engine — neuroscience-inspired memory consolidation.

Universal logic module: no Flask, no hosting dependencies.
Designed for open-source sharing — all hosting/user logic stays in the API layer.

Three operations inspired by memory consolidation research
(McClelland et al., O'Reilly & Frank):

1. Replay (hippocampal replay): Random walks through the graph surface
   unexpected connections between distant knowledge clusters.
2. Consolidation (hippocampus-to-neocortex transfer): Episodic beliefs
   are merged into abstract, generalizable beliefs.
3. Reflection (Default Mode Network): Structural health analysis —
   orphan detection, hub over-centralization, schema discipline.

All operations yield progress messages via generators, enabling
async monitoring and live visualization in the frontend.

v1.0 — SUP-146/SUP-147 (2026-03-14)
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
) -> Generator[dict, None, dict]:
    """Random walk through the graph, surfacing unexpected connections.

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
                            "entities_involved": {"type": "array", "items": {"type": "string"}},
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
            "missing relationships, redundancies, and insights.",
            f"Walk path: {walk_text}\n\nSubgraph:\n{subgraph_text}",
            temperature=0.4,
            json_schema=_replay_schema,
            model=os.environ.get("MERKRAUM_DREAMING_REPLAY_MODEL"),
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

    yield _msg("replay", "done",
               f"Replay complete: {walks} walk(s), "
               f"{len(observations)} observation(s)")

    return {
        "phase": "replay",
        "status": "completed",
        "walks": all_walks,
        "observations": observations,
        "total_observations": len(observations),
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

    with adapter._driver.session() as session:
        # Find similar belief pairs using Jaro-Winkler distance
        pairs = session.run(
            "MATCH (a:Belief {project_id: $pid}), (b:Belief {project_id: $pid}) "
            "WHERE (a.status IS NULL OR a.status = 'active') "
            "AND (b.status IS NULL OR b.status = 'active') "
            "AND (a.active = true OR a.active IS NULL) "
            "AND (b.active = true OR b.active IS NULL) "
            "AND id(a) < id(b) "
            "WITH a, b, "
            "CASE WHEN a.name STARTS WITH 'Belief: ' "
            "THEN substring(a.name, 8) ELSE a.name END AS na, "
            "CASE WHEN b.name STARTS WITH 'Belief: ' "
            "THEN substring(b.name, 8) ELSE b.name END AS nb "
            "WITH a, b, 1.0 - apoc.text.jaroWinklerDistance(na, nb) AS sim "
            "WHERE sim >= $threshold "
            "RETURN a.name AS name_a, a.confidence AS conf_a, "
            "b.name AS name_b, b.confidence AS conf_b, "
            "round(sim * 1000) / 1000.0 AS similarity "
            "ORDER BY sim DESC",
            pid=project_id, threshold=similarity_threshold,
        ).data()

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
            model=os.environ.get("MERKRAUM_DREAMING_CONSOLIDATION_MODEL"),
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
# Phase 3: Reflection — structural health analysis
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
# Full Dream Session — orchestrates all three phases
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
) -> Generator[dict, None, dict]:
    """Run a full dreaming session (replay → consolidation → reflection).

    Args:
        adapter: Backend adapter (Neo4j + vector store)
        project_id: Project to dream about
        phases: Which phases to run (default: all three)
        replay_hops: Steps per random walk
        replay_walks: Number of random walks
        consolidation_threshold: Similarity threshold for belief clustering
        consolidation_dry_run: If True, preview consolidation without applying
        seed: Optional starting entity for replay

    Yields progress messages. Returns combined result dict.
    """
    active_phases = phases or ["replay", "consolidation", "reflection"]
    session_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    yield _msg("dream", "session_start",
               f"Dream session {session_id} — phases: {', '.join(active_phases)}",
               {"session_id": session_id, "phases": active_phases})

    results = {"session_id": session_id, "phases": {}}

    if "replay" in active_phases:
        yield _msg("dream", "phase_start", "Phase 1: Replay (hippocampal)")
        gen = replay(adapter, project_id, hops=replay_hops,
                     walks=replay_walks, seed=seed)
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

    if "reflection" in active_phases:
        yield _msg("dream", "phase_start", "Phase 3: Reflection (structural health)")
        gen = reflect(adapter, project_id)
        reflection_result = None
        try:
            while True:
                progress = next(gen)
                yield progress
        except StopIteration as e:
            reflection_result = e.value
        results["phases"]["reflection"] = reflection_result or {}

    elapsed = round(time.time() - start_time, 1)
    results["duration_seconds"] = elapsed
    results["status"] = "completed"

    yield _msg("dream", "session_complete",
               f"Dream session {session_id} complete ({elapsed}s)",
               {"duration_seconds": elapsed})

    return results
