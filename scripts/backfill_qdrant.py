#!/usr/bin/env python3
"""Backfill Qdrant vectors from existing Neo4j nodes.

Reads all nodes with a summary from Neo4j and upserts their embeddings
into Qdrant. Safe to re-run — uses deterministic vector IDs.

Usage:
    python3 scripts/backfill_qdrant.py                    # all projects
    python3 scripts/backfill_qdrant.py --project vsg_knowledge
    python3 scripts/backfill_qdrant.py --dry-run          # preview only
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# Load secrets from AWS Secrets Manager (same as API startup)
try:
    import boto3
    client = boto3.client("secretsmanager", region_name="eu-central-1")
    resp = client.get_secret_value(SecretId="merkraum/config")
    secrets = json.loads(resp["SecretString"])
    for key, value in secrets.items():
        os.environ.setdefault(key, value)
except Exception:
    pass

from merkraum_backend import create_adapter


def backfill(project_id=None, dry_run=False):
    adapter = create_adapter("neo4j_qdrant")
    adapter.connect()

    query = """
    MATCH (n)
    WHERE n.summary IS NOT NULL AND n.summary <> ''
    """
    if project_id:
        query += " AND n.project_id = $project_id"
    query += " RETURN n.name AS name, n.summary AS summary, n.project_id AS project_id, labels(n) AS labels, elementId(n) AS node_id"

    params = {"project_id": project_id} if project_id else {}

    with adapter._driver.session() as session:
        results = list(session.run(query, params))

    print(f"Found {len(results)} nodes with summaries")
    if not results:
        return

    success = 0
    failed = 0

    for i, record in enumerate(results):
        name = record["name"]
        summary = record["summary"]
        proj = record["project_id"] or "default"
        labels = [l for l in record["labels"] if l != "Node"]
        node_type = labels[0] if labels else None

        text = adapter._vector_text_for_node(name, summary)
        vector_id = adapter._vector_id_for_node(proj, name, node_type=node_type)

        if dry_run:
            if i < 5:
                print(f"  [{proj}] {node_type}:{name} -> {vector_id}")
            elif i == 5:
                print(f"  ... and {len(results) - 5} more")
            continue

        ok = adapter.vector_upsert(
            vector_id=vector_id,
            text=text,
            metadata={"name": name, "node_type": node_type or "unknown"},
            project_id=proj,
        )
        if ok:
            success += 1
        else:
            failed += 1
            print(f"  FAILED: {name} ({node_type})")

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(results)} (ok={success}, fail={failed})")

    if dry_run:
        print(f"Dry run — would upsert {len(results)} vectors")
    else:
        print(f"Done: {success} upserted, {failed} failed, {len(results)} total")

    adapter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill Qdrant vectors from Neo4j")
    parser.add_argument("--project", help="Only backfill a specific project")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()
    backfill(project_id=args.project, dry_run=args.dry_run)
