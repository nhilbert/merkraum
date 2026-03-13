#!/usr/bin/env python3
"""
Unit tests for Merkraum BackendAdapter hierarchy.

Tests the factory function, shared Neo4jBaseAdapter graph operations,
and adapter-specific vector/connection behavior — all with mocked backends.
No live Neo4j, Qdrant, or Pinecone required.

Z1337 — SUP-93 further work: adapter unit tests.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from merkraum_backend import (
    BackendAdapter,
    Neo4jBaseAdapter,
    Neo4jQdrantAdapter,
    Neo4jPineconeAdapter,
    create_adapter,
    _string_to_uuid,
    NODE_TYPES,
    RELATIONSHIP_TYPES,
    SYMMETRIC_TYPES,
    NodeLimitExceeded,
    TIER_LIMITS,
)


# --- Factory tests ---

class TestCreateAdapter(unittest.TestCase):
    """Test create_adapter factory function."""

    def test_explicit_neo4j_pinecone(self):
        adapter = create_adapter("neo4j_pinecone")
        self.assertIsInstance(adapter, Neo4jPineconeAdapter)

    def test_explicit_neo4j_qdrant(self):
        adapter = create_adapter("neo4j_qdrant")
        self.assertIsInstance(adapter, Neo4jQdrantAdapter)

    def test_invalid_backend_type(self):
        with self.assertRaises(ValueError) as ctx:
            create_adapter("mysql_redis")
        self.assertIn("Unknown backend type", str(ctx.exception))

    @patch.dict(os.environ, {"MERKRAUM_BACKEND": "neo4j_qdrant"})
    def test_env_var_auto_detection(self):
        adapter = create_adapter()
        self.assertIsInstance(adapter, Neo4jQdrantAdapter)

    @patch.dict(os.environ, {}, clear=True)
    def test_default_is_neo4j_pinecone(self):
        # Remove MERKRAUM_BACKEND if present
        os.environ.pop("MERKRAUM_BACKEND", None)
        adapter = create_adapter()
        self.assertIsInstance(adapter, Neo4jPineconeAdapter)

    def test_kwargs_forwarded_to_pinecone(self):
        adapter = create_adapter("neo4j_pinecone",
                                 neo4j_uri="bolt://test:7687",
                                 pinecone_api_key="test-key")
        self.assertEqual(adapter._neo4j_uri, "bolt://test:7687")
        self.assertEqual(adapter._pinecone_api_key, "test-key")

    def test_kwargs_forwarded_to_qdrant(self):
        adapter = create_adapter("neo4j_qdrant",
                                 neo4j_uri="bolt://test:7687",
                                 qdrant_url="http://test:6333")
        self.assertEqual(adapter._neo4j_uri, "bolt://test:7687")
        self.assertEqual(adapter._qdrant_url, "http://test:6333")


# --- Schema constants tests ---

class TestSchemaConstants(unittest.TestCase):
    """Verify schema constants are consistent."""

    def test_node_types_non_empty(self):
        self.assertGreater(len(NODE_TYPES), 0)

    def test_relationship_types_non_empty(self):
        self.assertGreater(len(RELATIONSHIP_TYPES), 0)

    def test_symmetric_types_subset_of_relationship_types(self):
        for st in SYMMETRIC_TYPES:
            self.assertIn(st, RELATIONSHIP_TYPES,
                          f"Symmetric type '{st}' not in RELATIONSHIP_TYPES")

    def test_expected_node_types(self):
        for expected in ["Belief", "Person", "Concept", "Organization"]:
            self.assertIn(expected, NODE_TYPES)

    def test_expected_relationship_types(self):
        for expected in ["SUPPORTS", "CONTRADICTS", "SUPERSEDES"]:
            self.assertIn(expected, RELATIONSHIP_TYPES)


# --- Utility tests ---

class TestStringToUuid(unittest.TestCase):

    def test_deterministic(self):
        a = _string_to_uuid("test-vector-id")
        b = _string_to_uuid("test-vector-id")
        self.assertEqual(a, b)

    def test_different_inputs_different_uuids(self):
        a = _string_to_uuid("alpha")
        b = _string_to_uuid("beta")
        self.assertNotEqual(a, b)

    def test_valid_uuid_format(self):
        import uuid
        result = _string_to_uuid("my-id")
        uuid.UUID(result)  # Should not raise


# --- Neo4jBaseAdapter shared graph operations ---

def _make_mock_adapter():
    """Create a Neo4jQdrantAdapter with mocked Neo4j driver (no connect())."""
    adapter = Neo4jQdrantAdapter.__new__(Neo4jQdrantAdapter)
    adapter._driver = MagicMock()
    adapter._qdrant = None
    adapter._embedder = None
    adapter._neo4j_uri = "bolt://mock:7687"
    adapter._neo4j_user = "neo4j"
    adapter._neo4j_password = "test"
    adapter._qdrant_url = "http://mock:6333"
    adapter._qdrant_api_key = None
    adapter._embed_model_name = "BAAI/bge-small-en-v1.5"
    return adapter


class TestWriteEntities(unittest.TestCase):
    """Test Neo4jBaseAdapter.write_entities via a concrete subclass."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)
        self.adapter.vector_upsert = MagicMock(return_value=True)

    def test_write_valid_entity(self):
        entities = [{"name": "TestConcept", "node_type": "Concept", "summary": "A test"}]
        count = self.adapter.write_entities(entities, "Z1337", project_id="test")
        self.assertEqual(count, 1)
        self.session.run.assert_called_once()

    def test_skip_invalid_node_type(self):
        entities = [{"name": "Bad", "node_type": "InvalidType"}]
        count = self.adapter.write_entities(entities, "Z1337", project_id="test")
        self.assertEqual(count, 0)
        self.session.run.assert_not_called()

    def test_default_node_type_is_concept(self):
        entities = [{"name": "NoType"}]  # No node_type field
        count = self.adapter.write_entities(entities, "Z1337", project_id="test")
        self.assertEqual(count, 1)
        # Verify the Cypher was for Concept (default)
        call_args = self.session.run.call_args
        self.assertIn("Concept", call_args[0][0])

    def test_belief_entity_uses_confidence(self):
        entities = [{"name": "TestBelief", "node_type": "Belief",
                     "confidence": 0.9, "summary": "High confidence"}]
        count = self.adapter.write_entities(entities, "Z1337", project_id="test")
        self.assertEqual(count, 1)
        call_args = self.session.run.call_args
        self.assertIn("Belief", call_args[0][0])
        self.assertIn("confidence", call_args[0][0])

    def test_belief_default_confidence(self):
        entities = [{"name": "LowConf", "node_type": "Belief"}]
        count = self.adapter.write_entities(entities, "Z1337", project_id="test")
        self.assertEqual(count, 1)
        # Default confidence is 0.7
        call_kwargs = self.session.run.call_args[1]
        self.assertEqual(call_kwargs["confidence"], 0.7)

    def test_multiple_entities(self):
        entities = [
            {"name": "A", "node_type": "Person"},
            {"name": "B", "node_type": "Organization"},
            {"name": "C", "node_type": "InvalidType"},  # Skipped
            {"name": "D", "node_type": "Event"},
        ]
        count = self.adapter.write_entities(entities, "Z1337", project_id="test")
        self.assertEqual(count, 3)  # A, B, D written; C skipped

    def test_entity_write_error_continues(self):
        """If one entity fails, others should still be written."""
        self.session.run.side_effect = [Exception("DB error"), None]
        entities = [
            {"name": "Fail", "node_type": "Concept"},
            {"name": "Succeed", "node_type": "Concept"},
        ]
        count = self.adapter.write_entities(entities, "Z1337", project_id="test")
        self.assertEqual(count, 1)  # Only second succeeds

    def test_project_id_passed_through(self):
        entities = [{"name": "X", "node_type": "Concept"}]
        self.adapter.write_entities(entities, "Z1337", project_id="my_project")
        call_kwargs = self.session.run.call_args[1]
        self.assertEqual(call_kwargs["project_id"], "my_project")

    def test_entity_write_triggers_vector_upsert(self):
        entities = [{"name": "Vectorized", "node_type": "Concept", "summary": "Embedding"}]
        self.adapter.write_entities(entities, "Z1337", project_id="my_project")
        self.adapter.vector_upsert.assert_called_once()
        call_kwargs = self.adapter.vector_upsert.call_args[1]
        self.assertEqual(call_kwargs["project_id"], "my_project")
        self.assertEqual(call_kwargs["metadata"]["name"], "Vectorized")

    def test_vector_upsert_failure_does_not_block_entity_write(self):
        self.adapter.vector_upsert.return_value = False
        entities = [{"name": "NoVector", "node_type": "Concept"}]
        count = self.adapter.write_entities(entities, "Z1337", project_id="test")
        self.assertEqual(count, 1)


class TestWriteRelationships(unittest.TestCase):

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)
        # Default: simulate a new relationship created
        mock_summary = MagicMock()
        mock_summary.counters.relationships_created = 1
        mock_summary.counters.properties_set = 0
        mock_result = MagicMock()
        mock_result.consume.return_value = mock_summary
        self.session.run.return_value = mock_result

    def test_write_valid_relationship(self):
        rels = [{"source": "A", "target": "B", "type": "SUPPORTS"}]
        count = self.adapter.write_relationships(rels, "Z1337", project_id="test")
        self.assertEqual(count, 1)

    def test_skip_invalid_relationship_type(self):
        rels = [{"source": "A", "target": "B", "type": "FAKE_REL"}]
        count = self.adapter.write_relationships(rels, "Z1337", project_id="test")
        self.assertEqual(count, 0)

    def test_default_type_is_references(self):
        rels = [{"source": "A", "target": "B"}]  # No type field
        count = self.adapter.write_relationships(rels, "Z1337", project_id="test")
        self.assertEqual(count, 1)
        call_args = self.session.run.call_args_list[0]
        self.assertIn("REFERENCES", call_args[0][0])

    def test_symmetric_type_creates_inverse(self):
        rels = [{"source": "A", "target": "B", "type": "CONTRADICTS"}]
        self.adapter.write_relationships(rels, "Z1337", project_id="test")
        # Should have 2 calls: forward + inverse
        self.assertEqual(self.session.run.call_count, 2)

    def test_non_symmetric_no_inverse(self):
        rels = [{"source": "A", "target": "B", "type": "SUPPORTS"}]
        self.adapter.write_relationships(rels, "Z1337", project_id="test")
        # Only 1 call (forward) + consume
        self.assertEqual(self.session.run.call_count, 1)

    def test_update_existing_counts(self):
        """When rel already exists (properties_set > 0, relationships_created = 0)."""
        mock_summary = MagicMock()
        mock_summary.counters.relationships_created = 0
        mock_summary.counters.properties_set = 3
        mock_result = MagicMock()
        mock_result.consume.return_value = mock_summary
        self.session.run.return_value = mock_result

        rels = [{"source": "A", "target": "B", "type": "SUPPORTS"}]
        count = self.adapter.write_relationships(rels, "Z1337", project_id="test")
        self.assertEqual(count, 1)  # Updated counts as written

    def test_relationship_error_continues(self):
        self.session.run.side_effect = [Exception("DB error"), self.session.run.return_value]
        rels = [
            {"source": "A", "target": "B", "type": "SUPPORTS"},
            {"source": "C", "target": "D", "type": "SUPPORTS"},
        ]
        count = self.adapter.write_relationships(rels, "Z1337", project_id="test")
        self.assertEqual(count, 1)


class TestQueryNodes(unittest.TestCase):

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def test_query_all_nodes(self):
        self.session.run.return_value = [
            {"name": "A", "summary": "desc", "type": "Concept", "created": "2026-01-01"},
        ]
        results = self.adapter.query_nodes(project_id="test")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "A")

    def test_query_filtered_by_type(self):
        self.session.run.return_value = []
        self.adapter.query_nodes(node_type="Belief", project_id="test")
        call_args = self.session.run.call_args[0][0]
        self.assertIn("Belief", call_args)

    def test_invalid_type_uses_unfiltered_query(self):
        self.session.run.return_value = []
        self.adapter.query_nodes(node_type="FakeType", project_id="test")
        call_args = self.session.run.call_args[0][0]
        # Should NOT contain FakeType in the Cypher
        self.assertNotIn("FakeType", call_args)


class TestGetBeliefs(unittest.TestCase):

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)
        self.session.run.return_value = []

    def test_active_status(self):
        self.adapter.get_beliefs(project_id="test", status="active")
        cypher = self.session.run.call_args[0][0]
        self.assertIn("b.active = true", cypher)

    def test_uncertain_status(self):
        self.adapter.get_beliefs(project_id="test", status="uncertain")
        cypher = self.session.run.call_args[0][0]
        self.assertIn("confidence < 0.5", cypher)

    def test_contradicted_status(self):
        self.adapter.get_beliefs(project_id="test", status="contradicted")
        cypher = self.session.run.call_args[0][0]
        self.assertIn("CONTRADICTS", cypher)

    def test_superseded_status(self):
        self.adapter.get_beliefs(project_id="test", status="superseded")
        cypher = self.session.run.call_args[0][0]
        self.assertIn("superseded", cypher)


class TestGetStats(unittest.TestCase):

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)
        # All counts return 0 by default
        mock_result = MagicMock()
        mock_result.single.return_value = {"c": 0}
        self.session.run.return_value = mock_result

    def test_stats_structure(self):
        stats = self.adapter.get_stats(project_id="test")
        self.assertIn("nodes", stats)
        self.assertIn("edges", stats)
        self.assertIn("total_nodes", stats)
        self.assertIn("total_edges", stats)

    def test_stats_queries_all_types(self):
        self.adapter.get_stats(project_id="test")
        # Should query each NODE_TYPE + each RELATIONSHIP_TYPE
        expected_calls = len(NODE_TYPES) + len(RELATIONSHIP_TYPES)
        self.assertEqual(self.session.run.call_count, expected_calls)


class TestDeleteProjectData(unittest.TestCase):

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def test_cannot_delete_default(self):
        with self.assertRaises(ValueError):
            self.adapter.delete_project_data("default")

    def test_delete_non_default(self):
        mock_result = MagicMock()
        mock_result.single.return_value = {"c": 5}
        self.session.run.return_value = mock_result
        counts = self.adapter.delete_project_data("test_project")
        self.assertEqual(counts["nodes_deleted"], 5)


# --- Adapter-specific tests ---

class TestNeo4jBaseAdapterCredentials(unittest.TestCase):
    """Test shared credential loading and Neo4j connection in base class."""

    @patch.dict(os.environ, {}, clear=True)
    def test_load_neo4j_credentials_from_env_file(self):
        """When no constructor args, credentials come from _load_env()."""
        adapter = Neo4jQdrantAdapter()  # No URI passed
        with patch("merkraum_backend._load_env", return_value={
            "NEO4J_URI": "bolt://envhost:7687",
            "NEO4J_USER": "envuser",
            "NEO4J_PASSWORD": "envpass",
        }):
            adapter._load_neo4j_credentials()
        self.assertEqual(adapter._neo4j_uri, "bolt://envhost:7687")
        self.assertEqual(adapter._neo4j_user, "envuser")
        self.assertEqual(adapter._neo4j_password, "envpass")

    def test_load_neo4j_credentials_skips_if_already_set(self):
        """Constructor-provided URI takes precedence over env."""
        adapter = Neo4jQdrantAdapter(neo4j_uri="bolt://explicit:7687",
                                     neo4j_user="explicit",
                                     neo4j_password="explicitpw")
        with patch("merkraum_backend._load_env", return_value={
            "NEO4J_URI": "bolt://envhost:7687",
        }):
            adapter._load_neo4j_credentials()
        self.assertEqual(adapter._neo4j_uri, "bolt://explicit:7687")
        self.assertEqual(adapter._neo4j_user, "explicit")

    def test_load_neo4j_credentials_returns_env_dict(self):
        """_load_neo4j_credentials returns env dict for subclass use."""
        adapter = Neo4jPineconeAdapter()
        with patch("merkraum_backend._load_env", return_value={
            "NEO4J_URI": "bolt://test:7687",
            "PINECONE_API_KEY": "pk-test",
        }):
            env = adapter._load_neo4j_credentials()
        self.assertEqual(env["PINECONE_API_KEY"], "pk-test")

    def test_connect_neo4j_uses_base_class(self):
        """Both adapters use the same _connect_neo4j from base class."""
        self.assertIs(Neo4jQdrantAdapter._connect_neo4j,
                      Neo4jPineconeAdapter._connect_neo4j)
        self.assertIs(Neo4jQdrantAdapter._connect_neo4j,
                      Neo4jBaseAdapter._connect_neo4j)

    def test_qdrant_load_credentials_loads_qdrant_keys(self):
        """Qdrant adapter loads vendor-specific keys via base env dict."""
        adapter = Neo4jQdrantAdapter()
        with patch("merkraum_backend._load_env", return_value={
            "NEO4J_URI": "bolt://test:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "pw",
            "QDRANT_URL": "http://qdrant:6333",
            "QDRANT_API_KEY": "qk-test",
        }):
            adapter._load_credentials()
        self.assertEqual(adapter._qdrant_url, "http://qdrant:6333")
        self.assertEqual(adapter._qdrant_api_key, "qk-test")
        self.assertEqual(adapter._neo4j_uri, "bolt://test:7687")

    def test_pinecone_load_credentials_loads_pinecone_key(self):
        """Pinecone adapter loads vendor-specific keys via base env dict."""
        adapter = Neo4jPineconeAdapter()
        with patch("merkraum_backend._load_env", return_value={
            "NEO4J_URI": "bolt://test:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "pw",
            "PINECONE_API_KEY": "pk-test",
        }):
            adapter._load_credentials()
        self.assertEqual(adapter._pinecone_api_key, "pk-test")
        self.assertEqual(adapter._neo4j_uri, "bolt://test:7687")


class TestNeo4jQdrantAdapterInit(unittest.TestCase):

    def test_default_qdrant_url(self):
        adapter = Neo4jQdrantAdapter()
        self.assertEqual(adapter._qdrant_url, "http://localhost:6333")

    def test_custom_qdrant_url(self):
        adapter = Neo4jQdrantAdapter(qdrant_url="http://custom:6333")
        self.assertEqual(adapter._qdrant_url, "http://custom:6333")

    def test_default_embed_model(self):
        adapter = Neo4jQdrantAdapter()
        self.assertEqual(adapter._embed_model_name, "BAAI/bge-small-en-v1.5")

    def test_collection_name_default(self):
        adapter = _make_mock_adapter()
        self.assertEqual(adapter._get_collection_name("proj1"), "proj1")

    def test_collection_name_with_namespace(self):
        adapter = _make_mock_adapter()
        self.assertEqual(adapter._get_collection_name("proj1", "beliefs"), "proj1_beliefs")

    def test_is_healthy_both_down(self):
        adapter = Neo4jQdrantAdapter()
        adapter._driver = None
        adapter._qdrant = None
        self.assertFalse(adapter.is_healthy())

    def test_vector_search_no_client_returns_empty(self):
        adapter = _make_mock_adapter()
        adapter._qdrant = None
        result = adapter.vector_search("test query")
        self.assertEqual(result, [])

    def test_vector_upsert_no_client_returns_false(self):
        adapter = _make_mock_adapter()
        adapter._qdrant = None
        result = adapter.vector_upsert("id1", "text", {})
        self.assertFalse(result)

    def test_close_clears_all(self):
        adapter = _make_mock_adapter()
        adapter._qdrant = MagicMock()
        adapter._embedder = MagicMock()
        adapter.close()
        self.assertIsNone(adapter._driver)
        self.assertIsNone(adapter._qdrant)
        self.assertIsNone(adapter._embedder)


class TestNeo4jPineconeAdapterInit(unittest.TestCase):

    def test_default_index_name(self):
        adapter = Neo4jPineconeAdapter()
        self.assertEqual(adapter._pinecone_index, "vsg-memory")

    def test_custom_index_name(self):
        adapter = Neo4jPineconeAdapter(pinecone_index="my-index")
        self.assertEqual(adapter._pinecone_index, "my-index")

    def test_is_healthy_no_driver_no_host(self):
        adapter = Neo4jPineconeAdapter()
        adapter._driver = None
        adapter._pinecone_host = None
        self.assertFalse(adapter.is_healthy())

    def test_is_healthy_with_host_no_driver(self):
        adapter = Neo4jPineconeAdapter()
        adapter._driver = None
        adapter._pinecone_host = "index.pinecone.io"
        self.assertFalse(adapter.is_healthy())

    def test_is_healthy_with_driver_no_host(self):
        adapter = Neo4jPineconeAdapter()
        adapter._driver = MagicMock()
        adapter._pinecone_host = None
        self.assertFalse(adapter.is_healthy())

    def test_vector_search_no_host_returns_empty(self):
        adapter = Neo4jPineconeAdapter()
        adapter._pinecone_host = None
        adapter._pinecone_api_key = "key"
        result = adapter.vector_search("test")
        self.assertEqual(result, [])

    def test_vector_upsert_no_host_returns_false(self):
        adapter = Neo4jPineconeAdapter()
        adapter._pinecone_host = None
        adapter._pinecone_api_key = "key"
        result = adapter.vector_upsert("id1", "text", {})
        self.assertFalse(result)

    def test_close_clears_driver(self):
        adapter = Neo4jPineconeAdapter()
        adapter._driver = MagicMock()
        adapter.close()
        self.assertIsNone(adapter._driver)


# --- Abstract interface compliance ---

class TestBackendAdapterInterface(unittest.TestCase):
    """Verify both concrete adapters implement the full abstract interface."""

    def test_qdrant_adapter_is_backend_adapter(self):
        self.assertTrue(issubclass(Neo4jQdrantAdapter, BackendAdapter))

    def test_pinecone_adapter_is_backend_adapter(self):
        self.assertTrue(issubclass(Neo4jPineconeAdapter, BackendAdapter))

    def test_all_abstract_methods_implemented(self):
        """Both adapters should be instantiable (all abstract methods implemented)."""
        # These should not raise TypeError
        Neo4jQdrantAdapter()
        Neo4jPineconeAdapter()

    def test_required_methods_exist(self):
        required = [
            "connect", "close", "is_healthy",
            "write_entities", "write_relationships", "query_nodes",
            "traverse", "get_beliefs", "get_stats", "delete_project_data",
            "update_node", "delete_node", "add_relationship", "delete_relationship",
            "merge_nodes", "vector_search", "vector_upsert", "vector_delete",
            "create_project", "get_project", "update_project", "list_projects",
            "reindex_project_vectors",
        ]
        for method in required:
            self.assertTrue(hasattr(Neo4jQdrantAdapter, method),
                            f"Neo4jQdrantAdapter missing {method}")
            self.assertTrue(hasattr(Neo4jPineconeAdapter, method),
                            f"Neo4jPineconeAdapter missing {method}")


# --- Node Limit Enforcement (SUP-97) ---

class TestNodeLimitExceeded(unittest.TestCase):
    """Test the NodeLimitExceeded exception."""

    def test_exception_attributes(self):
        exc = NodeLimitExceeded(current=90, limit=100, attempted=15)
        self.assertEqual(exc.current, 90)
        self.assertEqual(exc.limit, 100)
        self.assertEqual(exc.attempted, 15)

    def test_exception_message(self):
        exc = NodeLimitExceeded(current=90, limit=100, attempted=15)
        self.assertIn("90/100", str(exc))
        self.assertIn("15", str(exc))


class TestTierLimits(unittest.TestCase):
    """Test tier limit constants."""

    def test_free_tier_limit(self):
        self.assertEqual(TIER_LIMITS["free"], 100)

    def test_pro_tier_limit(self):
        self.assertEqual(TIER_LIMITS["pro"], 1_000)

    def test_team_tier_limit(self):
        self.assertEqual(TIER_LIMITS["team"], 5_000)

    def test_enterprise_tier_limit(self):
        self.assertEqual(TIER_LIMITS["enterprise"], 50_000)

    def test_all_tiers_present(self):
        self.assertEqual(set(TIER_LIMITS.keys()),
                         {"free", "pro", "team", "enterprise"})


class TestGetUsage(unittest.TestCase):
    """Test Neo4jBaseAdapter.get_usage."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def test_returns_node_and_edge_counts(self):
        node_result = MagicMock()
        node_result.single.return_value = {"c": 42}
        edge_result = MagicMock()
        edge_result.single.return_value = {"c": 87}
        self.session.run.side_effect = [node_result, edge_result]

        usage = self.adapter.get_usage(project_id="test")
        self.assertEqual(usage["nodes"], 42)
        self.assertEqual(usage["edges"], 87)

    def test_empty_project(self):
        node_result = MagicMock()
        node_result.single.return_value = {"c": 0}
        edge_result = MagicMock()
        edge_result.single.return_value = {"c": 0}
        self.session.run.side_effect = [node_result, edge_result]

        usage = self.adapter.get_usage(project_id="empty")
        self.assertEqual(usage["nodes"], 0)
        self.assertEqual(usage["edges"], 0)


class TestWriteEntitiesWithLimit(unittest.TestCase):
    """Test node limit enforcement in write_entities."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)
        self.adapter.vector_upsert = MagicMock(return_value=True)

    def _mock_usage(self, node_count, extra_writes=10):
        """Set up get_usage to return a specific node count, then allow writes."""
        node_result = MagicMock()
        node_result.single.return_value = {"c": node_count}
        edge_result = MagicMock()
        edge_result.single.return_value = {"c": 0}
        self.session.run.side_effect = (
            [node_result, edge_result] + [None] * extra_writes
        )

    def test_no_limit_allows_write(self):
        """Without node_limit, writes proceed normally."""
        entities = [{"name": "A", "node_type": "Concept"}]
        count = self.adapter.write_entities(entities, "Z1341", project_id="test")
        self.assertEqual(count, 1)

    def test_within_limit_allows_write(self):
        """Writes within limit proceed normally."""
        self._mock_usage(50)
        entities = [{"name": "A", "node_type": "Concept"}]
        count = self.adapter.write_entities(
            entities, "Z1341", project_id="test", node_limit=100
        )
        self.assertEqual(count, 1)

    def test_exceeding_limit_raises(self):
        """Writes that would exceed limit raise NodeLimitExceeded."""
        self._mock_usage(95)
        entities = [{"name": f"E{i}", "node_type": "Concept"} for i in range(10)]
        with self.assertRaises(NodeLimitExceeded) as ctx:
            self.adapter.write_entities(
                entities, "Z1341", project_id="test", node_limit=100
            )
        self.assertEqual(ctx.exception.current, 95)
        self.assertEqual(ctx.exception.limit, 100)
        self.assertEqual(ctx.exception.attempted, 10)

    def test_at_exact_limit_raises(self):
        """Already at limit, any new entity raises."""
        self._mock_usage(100)
        entities = [{"name": "One", "node_type": "Concept"}]
        with self.assertRaises(NodeLimitExceeded):
            self.adapter.write_entities(
                entities, "Z1341", project_id="test", node_limit=100
            )

    def test_empty_entities_with_limit_ok(self):
        """Empty entity list with limit set should not check usage."""
        count = self.adapter.write_entities(
            [], "Z1341", project_id="test", node_limit=100
        )
        self.assertEqual(count, 0)

    def test_exactly_filling_to_limit_ok(self):
        """Adding entities that exactly reach (not exceed) the limit succeeds."""
        self._mock_usage(95)
        entities = [{"name": f"E{i}", "node_type": "Concept"} for i in range(5)]
        count = self.adapter.write_entities(
            entities, "Z1341", project_id="test", node_limit=100
        )
        self.assertEqual(count, 5)


class TestReindexProjectVectors(unittest.TestCase):
    """Test vector reindexing for existing project nodes."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)
        self.adapter.vector_upsert = MagicMock(return_value=True)
        self.adapter._ensure_project_node_ids = MagicMock(return_value=None)

    def _total_result(self, count):
        result = MagicMock()
        result.single.return_value = {"c": count}
        return result

    def test_reindex_counts_successes(self):
        self.session.run.side_effect = [
            [
                {"name": "Albert Einstein", "node_type": "Person", "summary": "Physicist", "node_id": "id-1"},
                {"name": "Quantenphysik", "node_type": "Concept", "summary": "Topic", "node_id": "id-2"},
            ],
            self._total_result(2),
        ]
        result = self.adapter.reindex_project_vectors(project_id="proj", limit=100)
        self.assertEqual(result["project_id"], "proj")
        self.assertEqual(result["total_nodes"], 2)
        self.assertEqual(result["upserted"], 2)
        self.assertEqual(result["failed"], 0)
        self.assertFalse(result["truncated"])
        self.assertEqual(self.adapter.vector_upsert.call_count, 2)

    def test_reindex_marks_truncated(self):
        self.session.run.side_effect = [
            [
                {"name": "Node A", "node_type": "Concept", "summary": "", "node_id": "id-a"},
            ],
            self._total_result(5),
        ]
        result = self.adapter.reindex_project_vectors(project_id="proj", limit=1)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["total_nodes"], 1)

    def test_reindex_counts_failures(self):
        self.adapter.vector_upsert.side_effect = [True, False]
        self.session.run.side_effect = [
            [
                {"name": "Node A", "node_type": "Concept", "summary": "", "node_id": "id-a"},
                {"name": "Node B", "node_type": "Concept", "summary": "", "node_id": "id-b"},
            ],
            self._total_result(2),
        ]
        result = self.adapter.reindex_project_vectors(project_id="proj", limit=10)
        self.assertEqual(result["upserted"], 1)
        self.assertEqual(result["failed"], 1)


class TestBackendAdapterUsageInterface(unittest.TestCase):
    """Verify get_usage is in the abstract interface."""

    def test_get_usage_in_required_methods(self):
        self.assertTrue(hasattr(Neo4jQdrantAdapter, "get_usage"))
        self.assertTrue(hasattr(Neo4jPineconeAdapter, "get_usage"))


# --- update_belief tests ---

class TestUpdateBelief(unittest.TestCase):
    """Test update_belief method on Neo4jBaseAdapter."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def _mock_found(self, name="test belief"):
        """Mock a successful belief match."""
        mock_result = MagicMock()
        mock_result.single.return_value = {"name": name}
        self.session.run.return_value = mock_result

    def _mock_not_found(self):
        """Mock a belief not found."""
        mock_result = MagicMock()
        mock_result.single.return_value = None
        self.session.run.return_value = mock_result

    def test_update_confidence(self):
        self._mock_found()
        result = self.adapter.update_belief("test belief", "proj", confidence=0.9)
        self.assertTrue(result["updated"])
        self.assertEqual(result["changes"]["confidence"], 0.9)

    def test_update_status_superseded(self):
        self._mock_found()
        result = self.adapter.update_belief("test belief", "proj", status="superseded")
        self.assertTrue(result["updated"])
        self.assertEqual(result["changes"]["status"], "superseded")
        cypher = self.session.run.call_args[0][0]
        self.assertIn("b.active = false", cypher)

    def test_update_status_active(self):
        self._mock_found()
        result = self.adapter.update_belief("test belief", "proj", status="active")
        self.assertTrue(result["updated"])
        cypher = self.session.run.call_args[0][0]
        self.assertIn("b.active = true", cypher)

    def test_update_summary(self):
        self._mock_found()
        result = self.adapter.update_belief("test belief", "proj", summary="new text")
        self.assertTrue(result["updated"])
        self.assertEqual(result["changes"]["summary"], "new text")

    def test_multiple_changes(self):
        self._mock_found()
        result = self.adapter.update_belief(
            "test belief", "proj", confidence=0.3, status="superseded"
        )
        self.assertTrue(result["updated"])
        self.assertIn("confidence", result["changes"])
        self.assertIn("status", result["changes"])

    def test_invalid_status(self):
        result = self.adapter.update_belief("test belief", "proj", status="invalid")
        self.assertFalse(result["updated"])
        self.assertIn("error", result)

    def test_no_changes(self):
        result = self.adapter.update_belief("test belief", "proj")
        self.assertFalse(result["updated"])
        self.assertIn("error", result)

    def test_belief_not_found(self):
        self._mock_not_found()
        result = self.adapter.update_belief("nonexistent", "proj", confidence=0.5)
        self.assertFalse(result["updated"])
        self.assertIn("not found", result["error"])

    def test_confidence_clamped_high(self):
        self._mock_found()
        result = self.adapter.update_belief("test belief", "proj", confidence=1.5)
        self.assertEqual(result["changes"]["confidence"], 1.0)

    def test_confidence_clamped_low(self):
        self._mock_found()
        result = self.adapter.update_belief("test belief", "proj", confidence=-0.5)
        self.assertEqual(result["changes"]["confidence"], 0.0)


class TestUpdateBeliefInterface(unittest.TestCase):
    """Verify update_belief is in the abstract interface."""

    def test_update_belief_in_adapters(self):
        self.assertTrue(hasattr(Neo4jQdrantAdapter, "update_belief"))
        self.assertTrue(hasattr(Neo4jPineconeAdapter, "update_belief"))


class TestGraphMutationOperations(unittest.TestCase):
    """Smoke tests for new node/relationship mutation methods."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.tx = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)
        self.session.begin_transaction.return_value = self.tx
        self.adapter.vector_upsert = MagicMock(return_value=True)
        self.adapter.vector_delete = MagicMock(return_value=True)

    def test_add_relationship_invalid_type(self):
        result = self.adapter.add_relationship("A", "B", "BAD", project_id="p")
        self.assertFalse(result["ok"])

    def test_delete_relationship_not_found(self):
        rec = {"rels": []}
        mock_result = MagicMock()
        mock_result.single.return_value = rec
        self.tx.run.return_value = mock_result
        result = self.adapter.delete_relationship("A", "B", "SUPPORTS", project_id="p")
        self.assertFalse(result["ok"])

    def test_delete_node_not_found(self):
        mock_result = MagicMock()
        mock_result.single.return_value = None
        self.tx.run.return_value = mock_result
        result = self.adapter.delete_node("Missing", project_id="p")
        self.assertFalse(result["ok"])
        self.assertIn("not found", result.get("error", "").lower())

    def test_delete_node_uses_requested_type_label(self):
        mock_result = MagicMock()
        mock_result.single.return_value = None
        self.tx.run.return_value = mock_result
        self.adapter.delete_node("Missing", project_id="p", node_type="Belief")
        cypher = self.tx.run.call_args[0][0]
        self.assertIn("(n:Belief", cypher)

    def test_update_node_not_found(self):
        mock_result = MagicMock()
        mock_result.single.return_value = None
        self.tx.run.return_value = mock_result
        result = self.adapter.update_node("Missing", project_id="p", updates={"summary": "x"})
        self.assertFalse(result["ok"])

    def test_update_node_uses_requested_type_label(self):
        mock_result = MagicMock()
        mock_result.single.return_value = None
        self.tx.run.return_value = mock_result
        self.adapter.update_node("Missing", project_id="p", updates={"summary": "x"}, node_type="Belief")
        cypher = self.tx.run.call_args[0][0]
        self.assertIn("(n:Belief", cypher)

    def test_merge_nodes_same_name_rejected(self):
        result = self.adapter.merge_nodes("A", "A", project_id="p")
        self.assertFalse(result["ok"])

    def test_merge_nodes_same_name_with_node_ids_succeeds(self):
        self.adapter._ensure_project_node_ids = MagicMock(return_value=None)

        keep_props = {"name": "A", "summary": "keep", "node_id": "keep-1"}
        remove_props = {"name": "A", "summary": "remove", "node_id": "remove-1"}

        class _FakeResult:
            def __init__(self, single_value=None, data_value=None):
                self._single = single_value
                self._data = data_value

            def single(self):
                return self._single

            def data(self):
                return self._data if self._data is not None else ([] if self._single is None else [self._single])

        def _run(query, **kwargs):
            if "node_id: $node_id" in query and "RETURN labels(n)[0] AS node_type" in query:
                node_id = kwargs.get("node_id")
                if node_id == "keep-1":
                    return _FakeResult({"node_type": "Concept", "node_props": keep_props})
                if node_id == "remove-1":
                    return _FakeResult({"node_type": "Concept", "node_props": remove_props})
                return _FakeResult(None)
            if "node_id: $keep_id" in query and "RETURN labels(n)[0] AS node_type" in query:
                return _FakeResult({"node_type": "Concept", "node_props": keep_props})
            return _FakeResult(None)

        self.tx.run.side_effect = _run

        result = self.adapter.merge_nodes(
            "A",
            "A",
            project_id="p",
            keep_node_id="keep-1",
            remove_node_id="remove-1",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result.get("keep_node_id"), "keep-1")
        self.assertEqual(result.get("remove_node_id"), "remove-1")


# --- Project Management (SUP-134) ---

class TestCreateProject(unittest.TestCase):
    """Test Neo4jBaseAdapter.create_project."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def _mock_no_existing(self):
        mock_result = MagicMock()
        mock_result.single.return_value = None
        self.session.run.side_effect = [mock_result, None]

    def _mock_existing(self):
        mock_result = MagicMock()
        mock_result.single.return_value = {"pm": {"project_id": "test"}}
        self.session.run.return_value = mock_result

    def test_create_new_project(self):
        self._mock_no_existing()
        result = self.adapter.create_project("test", "Test Project", "user1")
        self.assertTrue(result["created"])
        self.assertEqual(result["project_id"], "test")
        self.assertEqual(result["name"], "Test Project")
        self.assertEqual(result["owner"], "user1")
        self.assertEqual(result["tier"], "free")

    def test_create_duplicate_raises(self):
        self._mock_existing()
        with self.assertRaises(ValueError) as ctx:
            self.adapter.create_project("test", "Test", "user1")
        self.assertIn("already exists", str(ctx.exception))

    def test_invalid_tier_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.adapter.create_project("test", "Test", "user1", tier="invalid")
        self.assertIn("Invalid tier", str(ctx.exception))


class TestGetProject(unittest.TestCase):

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def test_get_existing(self):
        mock_result = MagicMock()
        mock_pm = MagicMock()
        mock_pm.__iter__ = lambda s: iter([("project_id", "test"), ("name", "Test")])
        mock_pm.items = lambda: [("project_id", "test"), ("name", "Test")]
        mock_pm.keys = lambda: ["project_id", "name"]
        mock_pm.__getitem__ = lambda s, k: {"project_id": "test", "name": "Test"}[k]
        mock_result.single.return_value = {"pm": mock_pm}
        self.session.run.return_value = mock_result
        result = self.adapter.get_project("test")
        self.assertIsNotNone(result)

    def test_get_nonexistent(self):
        mock_result = MagicMock()
        mock_result.single.return_value = None
        self.session.run.return_value = mock_result
        result = self.adapter.get_project("nonexistent")
        self.assertIsNone(result)


class TestUpdateProject(unittest.TestCase):

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def test_update_not_found(self):
        mock_result = MagicMock()
        mock_result.single.return_value = None
        self.session.run.return_value = mock_result
        result = self.adapter.update_project("nonexistent", name="New Name")
        self.assertFalse(result["updated"])

    def test_invalid_tier_raises(self):
        with self.assertRaises(ValueError):
            self.adapter.update_project("test", tier="invalid")


class TestListProjects(unittest.TestCase):

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def test_list_empty(self):
        self.session.run.return_value = []
        result = self.adapter.list_projects()
        self.assertEqual(result, [])

    def test_list_with_owner_filter(self):
        self.session.run.return_value = []
        self.adapter.list_projects(owner="user1")
        cypher = self.session.run.call_args[0][0]
        self.assertIn("owner", cypher)


class TestProjectManagementInterface(unittest.TestCase):
    """Verify project management methods exist on adapters."""

    def test_methods_exist(self):
        for method in ["create_project", "get_project", "update_project", "list_projects"]:
            self.assertTrue(hasattr(Neo4jQdrantAdapter, method),
                            f"Neo4jQdrantAdapter missing {method}")
            self.assertTrue(hasattr(Neo4jPineconeAdapter, method),
                            f"Neo4jPineconeAdapter missing {method}")


# --- PAT (Personal Access Token) tests ---

import hashlib
from jwt_auth import PATValidator, PAT_PREFIX, PAT_SCOPES, require_scope
from merkraum_acl import is_project_allowed


class TestPATValidator(unittest.TestCase):
    """Test PATValidator class with mocked Neo4j driver."""

    def setUp(self):
        self.mock_driver = MagicMock()
        self.validator = PATValidator(self.mock_driver)

    def test_validate_rejects_non_pat_token(self):
        result = self.validator.validate("eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.xxx")
        self.assertIsNone(result)
        self.mock_driver.session.assert_not_called()

    def test_validate_accepts_pat_prefix(self):
        token = "mk_pat_" + "a1" * 32
        mock_session = MagicMock()
        mock_tx_result = MagicMock()
        mock_tx_result.__getitem__ = MagicMock(side_effect=lambda k: {
            "token_prefix": "mk_pat_a1b2",
            "name": "Test Token",
            "owner_id": "user-123",
            "scopes": ["read", "write"],
            "projects": ["proj-1"],
            "all_projects": False,
            "expires_at": None,
        }[k])
        # Make dict() work on the result
        mock_tx_result.keys = MagicMock(return_value=[
            "token_prefix", "name", "owner_id", "scopes", "projects", "all_projects", "expires_at"
        ])

        self.mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        self.mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Mock execute_write to call the static method
        def fake_execute_write(func, token_hash):
            return {
                "token_prefix": "mk_pat_a1b2",
                "name": "Test Token",
                "owner_id": "user-123",
                "scopes": ["read", "write"],
                "projects": ["proj-1"],
                "all_projects": False,
                "expires_at": None,
            }
        mock_session.execute_write = fake_execute_write

        result = self.validator.validate(token)
        self.assertIsNotNone(result)
        self.assertEqual(result["owner_id"], "user-123")
        self.assertEqual(result["scopes"], ["read", "write"])

    def test_validate_returns_none_for_invalid_token(self):
        token = "mk_pat_" + "ff" * 32
        mock_session = MagicMock()
        mock_session.execute_write = MagicMock(return_value=None)
        self.mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        self.mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        result = self.validator.validate(token)
        self.assertIsNone(result)

    def test_token_hash_is_sha256(self):
        token = "mk_pat_" + "ab" * 32
        expected_hash = hashlib.sha256(token.encode()).hexdigest()
        self.assertEqual(len(expected_hash), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in expected_hash))

    def test_create_token_validates_scopes(self):
        with self.assertRaises(ValueError) as ctx:
            self.validator.create_token(
                owner_id="user-1",
                name="Bad Token",
                scopes=["read", "destroy_everything"],
                projects=[],
            )
        self.assertIn("Invalid scopes", str(ctx.exception))


class TestPATConstants(unittest.TestCase):
    """Test PAT-related constants."""

    def test_pat_prefix(self):
        self.assertEqual(PAT_PREFIX, "mk_pat_")

    def test_known_scopes(self):
        expected = {"read", "write", "search", "ingest", "projects", "admin"}
        self.assertEqual(PAT_SCOPES, expected)


class TestACLWithPAT(unittest.TestCase):
    """Test is_project_allowed with PAT project restrictions."""

    @patch.dict(os.environ, {"AUTH_REQUIRED": "true"})
    def test_pat_empty_projects_no_all_projects_denies(self):
        """Empty projects list with all_projects=False = no access."""
        result = is_project_allowed(
            "some-project", "user-123",
            pat_projects=[], pat_all_projects=False,
        )
        self.assertFalse(result)

    @patch.dict(os.environ, {"AUTH_REQUIRED": "true"})
    def test_pat_project_not_in_list_denies(self):
        result = is_project_allowed(
            "project-b", "user-123",
            pat_projects=["project-a"], pat_all_projects=False,
        )
        self.assertFalse(result)

    @patch.dict(os.environ, {"AUTH_REQUIRED": "true"})
    def test_pat_project_in_list_allows(self):
        """PAT with matching project falls through to user-level ACL."""
        result = is_project_allowed(
            "user-123", "user-123",  # user namespace match
            pat_projects=["user-123"], pat_all_projects=False,
        )
        self.assertTrue(result)

    @patch.dict(os.environ, {"AUTH_REQUIRED": "true"})
    def test_pat_all_projects_allows(self):
        result = is_project_allowed(
            "user-123", "user-123",
            pat_projects=[], pat_all_projects=True,
        )
        self.assertTrue(result)

    @patch.dict(os.environ, {"AUTH_REQUIRED": "false"})
    def test_auth_not_required_allows_all(self):
        result = is_project_allowed(
            "any-project", None,
            pat_projects=[], pat_all_projects=False,
        )
        self.assertTrue(result)

    @patch.dict(os.environ, {"AUTH_REQUIRED": "true"})
    def test_no_pat_context_uses_standard_acl(self):
        """When pat_projects is None (Cognito auth), standard ACL applies."""
        result = is_project_allowed(
            "user-123", "user-123",
            pat_projects=None, pat_all_projects=None,
        )
        self.assertTrue(result)


class TestRequireScope(unittest.TestCase):
    """Test require_scope decorator."""

    def test_require_scope_allows_matching_scope(self):
        from flask import Flask
        test_app = Flask(__name__)

        @require_scope("read")
        def dummy():
            return "ok"

        with test_app.test_request_context():
            from flask import request as req
            req.pat_scopes = ["read", "search"]
            result = dummy()
            self.assertEqual(result, "ok")

    def test_require_scope_denies_missing_scope(self):
        from flask import Flask
        test_app = Flask(__name__)

        @require_scope("write")
        def dummy():
            return "ok"

        with test_app.test_request_context():
            from flask import request as req
            req.pat_scopes = ["read", "search"]
            result = dummy()
            # Should return (jsonify(...), 403)
            self.assertIsInstance(result, tuple)
            self.assertEqual(result[1], 403)

    def test_require_scope_admin_bypasses(self):
        from flask import Flask
        test_app = Flask(__name__)

        @require_scope("write")
        def dummy():
            return "ok"

        with test_app.test_request_context():
            from flask import request as req
            req.pat_scopes = ["admin"]
            result = dummy()
            self.assertEqual(result, "ok")

    def test_require_scope_cognito_user_allowed(self):
        from flask import Flask
        test_app = Flask(__name__)

        @require_scope("write")
        def dummy():
            return "ok"

        with test_app.test_request_context():
            from flask import request as req
            req.pat_scopes = None  # Cognito user
            result = dummy()
            self.assertEqual(result, "ok")


if __name__ == "__main__":
    unittest.main()
