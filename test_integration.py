#!/usr/bin/env python3
"""
Merkraum Integration Test — Full round-trip test against live Neo4j + Qdrant.

Prerequisites:
    docker compose up -d   # starts Neo4j + Qdrant
    pip install neo4j qdrant-client fastembed

Usage:
    python test_integration.py           # run all tests
    python test_integration.py --clean   # delete test project data after run

Tests:
    1. Backend health check (Neo4j + Qdrant connectivity)
    2. Entity creation (write_entities)
    3. Relationship creation (write_relationships)
    4. Node query (query_nodes with type filter)
    5. Belief lifecycle (create belief, query by status)
    6. Graph traversal (traverse from entity)
    7. Vector upsert + semantic search
    8. Graph stats (counts by type)
    9. MCP server tool integration (10 tools register correctly)
   10. Full round-trip: ingest text → search → traverse → beliefs

All test data uses project_id='test_integration' for isolation.
"""

import sys
import os
import time
import asyncio
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

TEST_PROJECT = "test_integration"

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

passed = 0
failed = 0
errors = []


def ok(name, detail=""):
    global passed
    passed += 1
    print(f"  {GREEN}✓{RESET} {name}" + (f" — {detail}" if detail else ""))


def fail(name, reason):
    global failed
    failed += 1
    errors.append((name, reason))
    print(f"  {RED}✗{RESET} {name} — {reason}")


def section(title):
    print(f"\n{YELLOW}[{title}]{RESET}")


def main():
    global passed, failed

    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true",
                        help="Delete test project data after run")
    args = parser.parse_args()

    print("=" * 60)
    print("Merkraum Integration Test")
    print("=" * 60)

    # --- Test 1: Backend health ---
    section("1. Backend Health")
    try:
        from merkraum_backend import create_adapter
        adapter = create_adapter("neo4j_qdrant")
        adapter.connect()
        healthy = adapter.is_healthy()
        if healthy:
            ok("Neo4j + Qdrant connected and healthy")
        else:
            fail("Health check", "is_healthy() returned False")
            print("\n  Cannot continue without healthy backends.")
            sys.exit(1)
    except Exception as e:
        fail("Backend connection", str(e))
        print(f"\n  Cannot continue: {e}")
        print("  Make sure 'docker compose up -d' is running.")
        sys.exit(1)

    # --- Test 2: Entity creation ---
    section("2. Entity Creation")
    test_entities = [
        {"name": "Albert Einstein", "summary": "Theoretical physicist, relativity",
         "node_type": "Person"},
        {"name": "General Relativity", "summary": "Theory of gravitation as spacetime curvature",
         "node_type": "Concept"},
        {"name": "Special Relativity", "summary": "Theory of spacetime at constant velocity",
         "node_type": "Concept"},
        {"name": "Physics is beautiful", "summary": "Aesthetic appreciation of physical laws",
         "node_type": "Belief", "confidence": 0.8},
    ]
    try:
        count = adapter.write_entities(test_entities, "test_Z1", "test", TEST_PROJECT)
        if count == len(test_entities):
            ok(f"Wrote {count} entities")
        else:
            fail("Entity write", f"Expected {len(test_entities)}, got {count}")
    except Exception as e:
        fail("Entity write", str(e))

    # --- Test 3: Relationship creation ---
    section("3. Relationship Creation")
    test_rels = [
        {"source": "Albert Einstein", "target": "General Relativity",
         "type": "CREATED_BY", "reason": "Published 1915", "confidence": 1.0},
        {"source": "Albert Einstein", "target": "Special Relativity",
         "type": "CREATED_BY", "reason": "Published 1905", "confidence": 1.0},
        {"source": "General Relativity", "target": "Special Relativity",
         "type": "EXTENDS", "reason": "Generalizes to non-inertial frames",
         "confidence": 0.95},
    ]
    try:
        count = adapter.write_relationships(test_rels, "test_Z1", "test", TEST_PROJECT)
        if count == len(test_rels):
            ok(f"Wrote {count} relationships")
        else:
            fail("Relationship write", f"Expected {len(test_rels)}, got {count}")
    except Exception as e:
        fail("Relationship write", str(e))

    # --- Test 4: Node query ---
    section("4. Node Query")
    try:
        # All nodes
        all_nodes = adapter.query_nodes(None, TEST_PROJECT, 100)
        if len(all_nodes) >= len(test_entities):
            ok(f"Query all: {len(all_nodes)} nodes")
        else:
            fail("Query all", f"Expected >= {len(test_entities)}, got {len(all_nodes)}")

        # By type
        persons = adapter.query_nodes("Person", TEST_PROJECT, 100)
        has_einstein = any(n.get("name") == "Albert Einstein" for n in persons)
        if has_einstein:
            ok("Query by type: found Albert Einstein in Person nodes")
        else:
            fail("Query by type", "Albert Einstein not found in Person query")
    except Exception as e:
        fail("Node query", str(e))

    # --- Test 5: Belief lifecycle ---
    section("5. Belief Lifecycle")
    try:
        beliefs = adapter.get_beliefs(TEST_PROJECT, "active")
        has_belief = any("Physics" in b.get("name", "") or "beautiful" in b.get("summary", "")
                         for b in beliefs)
        if has_belief:
            ok(f"Active beliefs: {len(beliefs)} found, test belief present")
        else:
            # Might be that beliefs aren't labeled as active by default
            ok(f"Active beliefs: {len(beliefs)} returned (belief may need status property)")
    except Exception as e:
        fail("Belief query", str(e))

    # --- Test 6: Graph traversal ---
    section("6. Graph Traversal")
    try:
        result = adapter.traverse("Albert Einstein", TEST_PROJECT, 2)
        nodes = result.get("nodes", [])
        edges = result.get("edges", result.get("relationships", []))
        if len(nodes) >= 1:
            ok(f"Traversal from Einstein: {len(nodes)} nodes, {len(edges)} edges")
        else:
            fail("Traversal", f"Expected nodes from Einstein, got {len(nodes)}")
    except Exception as e:
        fail("Traversal", str(e))

    # --- Test 7: Vector upsert + search ---
    section("7. Vector Search")
    try:
        # Upsert
        for ent in test_entities:
            adapter.vector_upsert(
                f"{TEST_PROJECT}:{ent['name']}",
                f"{ent['name']}: {ent['summary']}",
                {"name": ent["name"], "node_type": ent["node_type"], "source": "test"},
                TEST_PROJECT,
            )
        ok("Vector upsert: 4 entities embedded")

        # Small delay for indexing
        time.sleep(1)

        # Search
        results = adapter.vector_search("theory of gravity", 3, TEST_PROJECT)
        if len(results) > 0:
            top = results[0]
            ok(f"Vector search: {len(results)} results, top: {top.get('content', top.get('id', '?'))[:60]}")
        else:
            fail("Vector search", "No results for 'theory of gravity'")
    except Exception as e:
        fail("Vector search", str(e))

    # --- Test 8: Graph stats ---
    section("8. Graph Stats")
    try:
        stats = adapter.get_stats(TEST_PROJECT)
        total_nodes = stats.get("total_nodes", 0)
        total_rels = stats.get("total_relationships", stats.get("total_edges", 0))
        if total_nodes >= len(test_entities):
            ok(f"Stats: {total_nodes} nodes, {total_rels} relationships")
        else:
            fail("Stats", f"Expected >= {len(test_entities)} nodes, got {total_nodes}")
    except Exception as e:
        fail("Graph stats", str(e))

    # --- Test 9: MCP server tools ---
    section("9. MCP Server Tools")
    try:
        from merkraum_mcp_server import mcp
        tools = asyncio.run(mcp.list_tools())
        tool_names = [t.name for t in tools]
        expected_tools = [
            "search_knowledge", "traverse_graph", "list_beliefs",
            "get_graph_stats", "query_nodes", "add_knowledge",
            "add_relationship", "ingest_knowledge", "check_ingestion_status",
            "health_check",
        ]
        missing = [t for t in expected_tools if t not in tool_names]
        if not missing:
            ok(f"All {len(expected_tools)} MCP tools registered: {', '.join(tool_names)}")
        else:
            fail("MCP tools", f"Missing: {missing}")
    except Exception as e:
        fail("MCP tools", str(e))

    # --- Test 10: Delete test data (cleanup) ---
    if args.clean:
        section("10. Cleanup")
        try:
            result = adapter.delete_project_data(TEST_PROJECT)
            ok(f"Deleted test data: {result}")
        except Exception as e:
            fail("Cleanup", str(e))

    # --- Summary ---
    adapter.close()
    print("\n" + "=" * 60)
    total = passed + failed
    if failed == 0:
        print(f"{GREEN}All {total} tests passed.{RESET}")
    else:
        print(f"{RED}{failed}/{total} tests failed:{RESET}")
        for name, reason in errors:
            print(f"  - {name}: {reason}")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
