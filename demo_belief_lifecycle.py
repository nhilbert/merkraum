#!/usr/bin/env python3
"""
Merkraum Demo: Belief Lifecycle — Contradiction, Supersession, Provenance.

Demonstrates Merkraum's core differentiator: structured belief objects that
track provenance, detect contradictions, and manage knowledge evolution.

Scenario: Organizational diagnostics using VSM (Viable System Model).
A consultant uses Merkraum to track evolving understanding of a client's
organization through interview data.

Usage:
    docker compose up -d          # start Neo4j + Qdrant
    python demo_belief_lifecycle.py
    python demo_belief_lifecycle.py --clean   # clean up after

Prerequisites:
    pip install neo4j qdrant-client fastembed

Z1401 — SUP-107 prep (Belief Tracking Demo Video).
"""

import sys
import os
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

DEMO_PROJECT = "demo_belief_lifecycle"

# ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def header(text):
    print(f"\n{'=' * 60}")
    print(f"{BOLD}{CYAN}{text}{RESET}")
    print(f"{'=' * 60}")


def step(num, text):
    print(f"\n{YELLOW}[Step {num}]{RESET} {BOLD}{text}{RESET}")


def info(text):
    print(f"  {DIM}{text}{RESET}")


def result(text):
    print(f"  {GREEN}→ {text}{RESET}")


def warn(text):
    print(f"  {RED}! {text}{RESET}")


def show_beliefs(adapter, status="active", label="Active"):
    beliefs = adapter.get_beliefs(DEMO_PROJECT, status)
    if beliefs:
        print(f"\n  {BOLD}{label} beliefs ({len(beliefs)}):{RESET}")
        for b in beliefs:
            conf = b.get("confidence", "?")
            conf_str = f"{conf:.0%}" if isinstance(conf, (int, float)) else str(conf)
            print(f"    [{conf_str}] {b['name']}")
            if b.get("summary"):
                print(f"         {DIM}{b['summary'][:80]}{RESET}")
    else:
        print(f"\n  {DIM}No {label.lower()} beliefs.{RESET}")
    return beliefs


def show_graph(adapter, entity_name):
    traversal = adapter.traverse(entity_name, DEMO_PROJECT, 2)
    nodes = traversal.get("nodes", [])
    edges = traversal.get("edges", [])
    if nodes or edges:
        print(f"\n  {BOLD}Graph from '{entity_name}':{RESET}")
        for edge in edges:
            conf = edge.get("confidence", 0)
            print(f"    {edge['source']} —[{edge['type']} {conf:.0%}]→ {edge['target']}")
    return traversal


def main():
    parser = argparse.ArgumentParser(description="Merkraum Belief Lifecycle Demo")
    parser.add_argument("--clean", action="store_true",
                        help="Delete demo data after run")
    parser.add_argument("--backend", default="neo4j_qdrant",
                        choices=["neo4j_qdrant", "neo4j_pinecone"],
                        help="Backend type (default: neo4j_qdrant)")
    args = parser.parse_args()

    header("Merkraum Demo: Belief Lifecycle")
    print("""
Scenario: A consultant diagnoses an organization using the Viable System
Model (VSM). Through successive interviews, beliefs about the organization
evolve — some get confirmed, others contradicted and superseded.

Merkraum tracks this evolution with full provenance.
""")

    # --- Connect ---
    step(1, "Connect to knowledge graph")
    try:
        from merkraum_backend import create_adapter
        adapter = create_adapter(args.backend)
        adapter.connect()
        if not adapter.is_healthy():
            warn("Backend unhealthy. Run: docker compose up -d")
            sys.exit(1)
        result("Connected to Neo4j + vector backend")
    except Exception as e:
        warn(f"Connection failed: {e}")
        print("\n  Run 'docker compose up -d' to start backends.")
        sys.exit(1)

    # Clean slate
    try:
        adapter.delete_project_data(DEMO_PROJECT)
    except ValueError:
        pass  # fresh project, nothing to delete

    # =====================================================================
    # PHASE 1: Initial assessment — management interviews
    # =====================================================================
    step(2, "Phase 1: Management interviews → initial beliefs")
    info("The consultant interviews 3 managers. They paint a positive picture.")

    # Create organizational entities
    entities_phase1 = [
        {"name": "Acme GmbH", "node_type": "Organization",
         "summary": "Mid-size manufacturing company, 450 employees, 3 divisions"},
        {"name": "CEO Interview (Day 1)", "node_type": "Interview",
         "summary": "CEO describes strong coordination and clear strategy"},
        {"name": "COO Interview (Day 1)", "node_type": "Interview",
         "summary": "COO emphasizes efficient production and low defect rates"},
        {"name": "CFO Interview (Day 1)", "node_type": "Interview",
         "summary": "CFO reports stable financials, on-budget operations"},
        # Initial beliefs from management interviews
        {"name": "Internal coordination is effective",
         "node_type": "Belief", "confidence": 0.75,
         "summary": "S2 function: management reports smooth cross-division coordination"},
        {"name": "Strategic planning is well-established",
         "node_type": "Belief", "confidence": 0.70,
         "summary": "S4 function: CEO describes regular strategy reviews and market analysis"},
        {"name": "Operational units have sufficient autonomy",
         "node_type": "Belief", "confidence": 0.70,
         "summary": "S1 function: divisions report adequate decision-making authority"},
        {"name": "Quality control is strong",
         "node_type": "Belief", "confidence": 0.80,
         "summary": "S3 function: COO reports systematic quality processes and audits"},
    ]

    count = adapter.write_entities(entities_phase1, "phase_1", "interview", DEMO_PROJECT)
    result(f"Created {count} entities (4 interviews + 4 beliefs)")

    # Link interviews to beliefs (provenance)
    rels_phase1 = [
        {"source": "CEO Interview (Day 1)", "target": "Internal coordination is effective",
         "type": "SUPPORTS", "confidence": 0.75, "reason": "CEO states: divisions work well together"},
        {"source": "CEO Interview (Day 1)", "target": "Strategic planning is well-established",
         "type": "SUPPORTS", "confidence": 0.70, "reason": "CEO describes quarterly strategy offsite"},
        {"source": "COO Interview (Day 1)", "target": "Quality control is strong",
         "type": "SUPPORTS", "confidence": 0.80, "reason": "COO: defect rate below 2%, ISO certified"},
        {"source": "COO Interview (Day 1)", "target": "Operational units have sufficient autonomy",
         "type": "SUPPORTS", "confidence": 0.65, "reason": "COO: divisions set own production targets"},
        {"source": "CFO Interview (Day 1)", "target": "Internal coordination is effective",
         "type": "SUPPORTS", "confidence": 0.60, "reason": "CFO: budget conflicts resolved quickly"},
    ]

    rel_count = adapter.write_relationships(rels_phase1, "phase_1", "interview", DEMO_PROJECT)
    result(f"Created {rel_count} provenance relationships")

    # Embed for semantic search
    for ent in entities_phase1:
        adapter.vector_upsert(
            f"{DEMO_PROJECT}:{ent['name']}",
            f"{ent['name']}: {ent.get('summary', '')}",
            {"name": ent["name"], "node_type": ent["node_type"], "phase": "1"},
            DEMO_PROJECT,
        )

    show_beliefs(adapter, "active", "Active")
    print(f"\n  {DIM}All beliefs are positive — management sees no problems.{RESET}")

    # =====================================================================
    # PHASE 2: Team-level interviews — contradictions emerge
    # =====================================================================
    step(3, "Phase 2: Team-level interviews → contradictions emerge")
    info("The consultant now interviews team leads and individual contributors.")
    info("A different picture emerges...")

    entities_phase2 = [
        {"name": "Team Lead Alpha Interview (Day 3)", "node_type": "Interview",
         "summary": "Division Alpha team lead describes constant firefighting and unclear priorities"},
        {"name": "Team Lead Beta Interview (Day 3)", "node_type": "Interview",
         "summary": "Division Beta lead reports being blindsided by strategy changes"},
        {"name": "Engineer Interview (Day 4)", "node_type": "Interview",
         "summary": "Senior engineer describes workarounds to bypass rigid approval processes"},
        # New beliefs from team-level data
        {"name": "Cross-division coordination relies on informal workarounds",
         "node_type": "Belief", "confidence": 0.80,
         "summary": "S2 dysfunction: formal coordination channels are slow; teams use Slack/hallway conversations instead"},
        {"name": "Strategy changes are not communicated to operational level",
         "node_type": "Belief", "confidence": 0.85,
         "summary": "S4-S3 gap: strategic decisions made at offsite are not systematically cascaded down"},
        {"name": "Operational autonomy is constrained by approval bureaucracy",
         "node_type": "Belief", "confidence": 0.75,
         "summary": "S1-S3 imbalance: formal autonomy exists on paper but approval processes negate it in practice"},
    ]

    count = adapter.write_entities(entities_phase2, "phase_2", "interview", DEMO_PROJECT)
    result(f"Created {count} entities (3 interviews + 3 new beliefs)")

    # These new beliefs CONTRADICT the management beliefs
    rels_phase2 = [
        # Provenance for new beliefs
        {"source": "Team Lead Alpha Interview (Day 3)",
         "target": "Cross-division coordination relies on informal workarounds",
         "type": "SUPPORTS", "confidence": 0.80,
         "reason": "Team lead: 'We bypass the official process because it takes 3 weeks'"},
        {"source": "Team Lead Beta Interview (Day 3)",
         "target": "Strategy changes are not communicated to operational level",
         "type": "SUPPORTS", "confidence": 0.85,
         "reason": "Team lead: 'I learned about the pivot from a customer, not my boss'"},
        {"source": "Engineer Interview (Day 4)",
         "target": "Operational autonomy is constrained by approval bureaucracy",
         "type": "SUPPORTS", "confidence": 0.75,
         "reason": "Engineer: 'I have the authority in theory, but need 4 signatures for any purchase'"},
        # CONTRADICTIONS — the key differentiator
        {"source": "Cross-division coordination relies on informal workarounds",
         "target": "Internal coordination is effective",
         "type": "CONTRADICTS", "confidence": 0.80,
         "reason": "Management sees coordination as smooth; teams see it as broken, relying on workarounds"},
        {"source": "Strategy changes are not communicated to operational level",
         "target": "Strategic planning is well-established",
         "type": "CONTRADICTS", "confidence": 0.85,
         "reason": "Strategy may exist at top level but fails at the S4→S3→S1 cascade — a planning process that doesn't reach operations isn't planning"},
        {"source": "Operational autonomy is constrained by approval bureaucracy",
         "target": "Operational units have sufficient autonomy",
         "type": "CONTRADICTS", "confidence": 0.75,
         "reason": "Formal autonomy contradicted by actual approval bottlenecks — S1 constrained by over-controlling S3"},
    ]

    rel_count = adapter.write_relationships(rels_phase2, "phase_2", "interview", DEMO_PROJECT)
    result(f"Created {rel_count} relationships (3 provenance + 3 contradictions)")

    for ent in entities_phase2:
        adapter.vector_upsert(
            f"{DEMO_PROJECT}:{ent['name']}",
            f"{ent['name']}: {ent.get('summary', '')}",
            {"name": ent["name"], "node_type": ent["node_type"], "phase": "2"},
            DEMO_PROJECT,
        )

    # Show contradicted beliefs
    show_beliefs(adapter, "contradicted", "Contradicted")
    print(f"\n  {BOLD}3 management beliefs now have contradicting evidence.{RESET}")
    print(f"  {DIM}This is what makes Merkraum unique: beliefs aren't just stored —{RESET}")
    print(f"  {DIM}contradictions are TRACKED with provenance back to source interviews.{RESET}")

    # =====================================================================
    # PHASE 3: Resolution — supersession
    # =====================================================================
    step(4, "Phase 3: Synthesis → beliefs superseded")
    info("The consultant synthesizes both perspectives into nuanced findings.")

    # Update old beliefs — supersede them with refined versions
    adapter.update_belief(
        "Internal coordination is effective", DEMO_PROJECT,
        status="superseded", confidence=0.30,
    )
    result("Superseded: 'Internal coordination is effective' (0.30)")

    adapter.update_belief(
        "Strategic planning is well-established", DEMO_PROJECT,
        status="superseded", confidence=0.25,
    )
    result("Superseded: 'Strategic planning is well-established' (0.25)")

    adapter.update_belief(
        "Operational units have sufficient autonomy", DEMO_PROJECT,
        status="superseded", confidence=0.35,
    )
    result("Superseded: 'Operational units have sufficient autonomy' (0.35)")

    # Create refined beliefs that supersede the old ones
    refined_beliefs = [
        {"name": "S2 coordination exists formally but is bypassed in practice",
         "node_type": "Belief", "confidence": 0.85,
         "summary": "Synthesis: coordination structures exist (management is not lying) but are too slow, "
                    "so teams route around them. The S2 function is present but inadequate for operational tempo."},
        {"name": "S4 strategy function is decoupled from S1 operations",
         "node_type": "Belief", "confidence": 0.90,
         "summary": "Synthesis: strategic planning occurs (CEO confirmed) but the S4→S3→S1 information cascade "
                    "is broken. Strategy exists at the top; execution happens at the bottom; the link is missing."},
        {"name": "S3 over-controls S1 through bureaucratic approval, undermining formal autonomy",
         "node_type": "Belief", "confidence": 0.80,
         "summary": "Synthesis: S1 divisions have formal authority (COO confirmed) but S3's approval processes "
                    "create a de facto bottleneck. The VSM diagnosis: S3 is over-controlling, not under-controlling."},
    ]

    count = adapter.write_entities(refined_beliefs, "phase_3", "synthesis", DEMO_PROJECT)
    result(f"Created {count} refined beliefs (synthesis)")

    # SUPERSEDES relationships — provenance chain continues
    supersedes_rels = [
        {"source": "S2 coordination exists formally but is bypassed in practice",
         "target": "Internal coordination is effective",
         "type": "SUPERSEDES", "confidence": 0.85,
         "reason": "Refined understanding: not 'good' or 'bad' but 'present but insufficient'"},
        {"source": "S2 coordination exists formally but is bypassed in practice",
         "target": "Cross-division coordination relies on informal workarounds",
         "type": "SUPERSEDES", "confidence": 0.85,
         "reason": "Both management and team perspectives are partially correct — synthesis captures both"},
        {"source": "S4 strategy function is decoupled from S1 operations",
         "target": "Strategic planning is well-established",
         "type": "SUPERSEDES", "confidence": 0.90,
         "reason": "Planning exists but doesn't cascade — refines both management and team views"},
        {"source": "S4 strategy function is decoupled from S1 operations",
         "target": "Strategy changes are not communicated to operational level",
         "type": "SUPERSEDES", "confidence": 0.90,
         "reason": "Communication failure is the mechanism; S4-S1 decoupling is the structural diagnosis"},
        {"source": "S3 over-controls S1 through bureaucratic approval, undermining formal autonomy",
         "target": "Operational units have sufficient autonomy",
         "type": "SUPERSEDES", "confidence": 0.80,
         "reason": "Formal autonomy is real but negated by approval processes — both sides are right"},
    ]

    rel_count = adapter.write_relationships(supersedes_rels, "phase_3", "synthesis", DEMO_PROJECT)
    result(f"Created {rel_count} SUPERSEDES relationships")

    for ent in refined_beliefs:
        adapter.vector_upsert(
            f"{DEMO_PROJECT}:{ent['name']}",
            f"{ent['name']}: {ent.get('summary', '')}",
            {"name": ent["name"], "node_type": ent["node_type"], "phase": "3"},
            DEMO_PROJECT,
        )

    # =====================================================================
    # PHASE 4: Query the evolved knowledge
    # =====================================================================
    step(5, "Phase 4: Query evolved knowledge state")

    print(f"\n  {BOLD}Final belief landscape:{RESET}")
    show_beliefs(adapter, "active", "Active (current understanding)")
    show_beliefs(adapter, "superseded", "Superseded (historical, with provenance)")

    # Semantic search
    step(6, "Semantic search: 'coordination problems between divisions'")
    time.sleep(1)  # allow indexing
    results = adapter.vector_search("coordination problems between divisions", 3, DEMO_PROJECT)
    if results:
        print(f"\n  {BOLD}Top {len(results)} semantic matches:{RESET}")
        for r in results:
            score = r.get("score", 0)
            content = r.get("content", r.get("id", "?"))[:80]
            print(f"    [{score:.2f}] {content}")

    # Graph traversal — show provenance chain
    step(7, "Graph traversal: provenance chain from synthesis belief")
    show_graph(adapter, "S2 coordination exists formally but is bypassed in practice")

    # =====================================================================
    # Summary
    # =====================================================================
    header("Demo Summary")
    stats = adapter.get_stats(DEMO_PROJECT)
    total_nodes = stats.get("total_nodes", 0)
    total_edges = stats.get("total_relationships", stats.get("total_edges", 0))

    print(f"""
  Graph: {total_nodes} nodes, {total_edges} relationships

  What Merkraum tracked:
  1. {GREEN}Beliefs with confidence scores{RESET} — not just facts, but degrees of certainty
  2. {GREEN}Contradiction detection{RESET} — team data contradicts management narrative
  3. {GREEN}Supersession chains{RESET} — refined beliefs replace outdated ones
  4. {GREEN}Full provenance{RESET} — every belief traces back to source interviews
  5. {GREEN}Semantic search{RESET} — find relevant knowledge by meaning, not keywords
  6. {GREEN}Knowledge evolution{RESET} — the graph shows HOW understanding changed

  This is what no other knowledge graph product does:
  Track not just WHAT you know, but HOW your knowledge evolved and WHY.
""")

    # --- Cleanup ---
    if args.clean:
        step(8, "Cleanup: removing demo data")
        counts = adapter.delete_project_data(DEMO_PROJECT)
        result(f"Deleted: {counts}")

    adapter.close()


if __name__ == "__main__":
    main()
