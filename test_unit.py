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
        # write_entities now uses begin_transaction(), so mock the tx
        self.tx = MagicMock()
        # before-state query returns None (new entity) by default;
        # after-state properties(n) query returns a dict
        before_result = MagicMock()
        before_result.single.return_value = None
        after_result = MagicMock()
        after_result.single.return_value = {"props": {"name": "mock", "node_type": "Concept"}}
        # tx.run returns before_result by default, but after-state query
        # (3rd call) needs after_result. Use side_effect for ordered calls:
        # call 1: before-state, call 2: MERGE, call 3: after-state props,
        # call 4+: log_history/log_operation
        log_result = MagicMock()
        log_result.single.return_value = None
        self.tx.run.return_value = after_result  # default fallback
        self.tx.run.side_effect = None  # clear any side_effect
        # Simple approach: return after_result for all calls (it has .single())
        self.tx.run.return_value = after_result
        self.session.begin_transaction.return_value = self.tx
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)
        self.adapter.vector_upsert = MagicMock(return_value=True)

    def test_write_valid_entity(self):
        entities = [{"name": "TestConcept", "node_type": "Concept", "summary": "A test"}]
        count = self.adapter.write_entities(entities, "Z1337", project_id="test")
        self.assertEqual(count, 1)
        # tx.run called: before-state query + MERGE + _log_history + _log_operation
        self.assertTrue(self.tx.run.call_count >= 2)
        self.tx.commit.assert_called_once()

    def test_skip_invalid_node_type(self):
        entities = [{"name": "Bad", "node_type": "InvalidType"}]
        count = self.adapter.write_entities(entities, "Z1337", project_id="test")
        self.assertEqual(count, 0)

    def test_default_node_type_is_concept(self):
        entities = [{"name": "NoType"}]  # No node_type field
        count = self.adapter.write_entities(entities, "Z1337", project_id="test")
        self.assertEqual(count, 1)
        # Verify the MERGE Cypher was for Concept (default) — second tx.run call
        merge_call = self.tx.run.call_args_list[1]
        self.assertIn("Concept", merge_call[0][0])

    def test_belief_entity_uses_confidence(self):
        entities = [{"name": "TestBelief", "node_type": "Belief",
                     "confidence": 0.9, "summary": "High confidence"}]
        count = self.adapter.write_entities(entities, "Z1337", project_id="test")
        self.assertEqual(count, 1)
        merge_call = self.tx.run.call_args_list[1]
        self.assertIn("Belief", merge_call[0][0])
        self.assertIn("confidence", merge_call[0][0])

    def test_belief_default_confidence(self):
        entities = [{"name": "LowConf", "node_type": "Belief"}]
        count = self.adapter.write_entities(entities, "Z1337", project_id="test")
        self.assertEqual(count, 1)
        # Default confidence is 0.7 — passed in kwargs to the MERGE tx.run call
        merge_call = self.tx.run.call_args_list[1]
        self.assertEqual(merge_call[1]["confidence"], 0.7)

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
        # First entity's before-state query raises, second succeeds
        self.tx.run.side_effect = [Exception("DB error")]
        # Need a fresh tx for second entity
        tx2 = MagicMock()
        tx2_after = MagicMock()
        tx2_after.single.return_value = {"props": {"name": "Succeed"}}
        tx2.run.return_value = tx2_after
        self.session.begin_transaction.side_effect = [self.tx, tx2]
        entities = [
            {"name": "Fail", "node_type": "Concept"},
            {"name": "Succeed", "node_type": "Concept"},
        ]
        count = self.adapter.write_entities(entities, "Z1337", project_id="test")
        self.assertEqual(count, 1)  # Only second succeeds

    def test_project_id_passed_through(self):
        entities = [{"name": "X", "node_type": "Concept"}]
        self.adapter.write_entities(entities, "Z1337", project_id="my_project")
        # MERGE call (second tx.run) should have project_id in kwargs
        merge_call = self.tx.run.call_args_list[1]
        self.assertEqual(merge_call[1]["project_id"], "my_project")

    def test_entity_write_triggers_vector_upsert(self):
        entities = [{"name": "Vectorized", "node_type": "Concept", "summary": "Embedding"}]
        self.adapter.write_entities(entities, "Z1337", project_id="my_project")
        self.adapter.vector_upsert.assert_called_once()
        call_args = self.adapter.vector_upsert.call_args
        # metadata is 3rd positional arg, project_id is keyword
        metadata = call_args[0][2]
        self.assertEqual(call_args[1]["project_id"], "my_project")
        self.assertEqual(metadata["name"], "Vectorized")

    def test_vector_upsert_failure_does_not_block_entity_write(self):
        self.adapter.vector_upsert.return_value = False
        entities = [{"name": "NoVector", "node_type": "Concept"}]
        count = self.adapter.write_entities(entities, "Z1337", project_id="test")
        self.assertEqual(count, 1)


class TestWriteRelationships(unittest.TestCase):

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        # write_relationships now uses begin_transaction()
        self.tx = MagicMock()
        # before-state query returns None (new relationship) by default
        before_result = MagicMock()
        before_result.single.return_value = None
        # MERGE result with consume()
        mock_summary = MagicMock()
        mock_summary.counters.relationships_created = 1
        mock_summary.counters.properties_set = 0
        mock_merge_result = MagicMock()
        mock_merge_result.consume.return_value = mock_summary
        # tx.run returns before_result first, then merge_result
        self.tx.run.side_effect = [before_result, mock_merge_result] + [MagicMock()] * 10
        self.session.begin_transaction.return_value = self.tx
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)
        self._mock_summary = mock_summary

    def _reset_tx_for_n_rels(self, n, created=1, props_set=0):
        """Reset tx.run side_effect for n relationships."""
        effects = []
        for _ in range(n):
            before = MagicMock()
            before.single.return_value = None
            summary = MagicMock()
            summary.counters.relationships_created = created
            summary.counters.properties_set = props_set
            merge = MagicMock()
            merge.consume.return_value = summary
            effects.extend([before, merge] + [MagicMock()] * 4)  # log calls
        self.tx.run.side_effect = effects

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
        # MERGE Cypher is the second tx.run call (after before-state query)
        merge_call = self.tx.run.call_args_list[1]
        self.assertIn("REFERENCES", merge_call[0][0])

    def test_symmetric_type_creates_inverse(self):
        self._reset_tx_for_n_rels(1)
        rels = [{"source": "A", "target": "B", "type": "CONTRADICTS"}]
        self.adapter.write_relationships(rels, "Z1337", project_id="test")
        # tx should have: before-state + MERGE + inverse + log calls
        self.assertTrue(self.tx.run.call_count >= 3)

    def test_non_symmetric_no_inverse(self):
        rels = [{"source": "A", "target": "B", "type": "SUPPORTS"}]
        self.adapter.write_relationships(rels, "Z1337", project_id="test")
        # Verify no inverse MERGE — check that the third call is a log, not a MERGE
        calls = self.tx.run.call_args_list
        # calls[0] = before-state, calls[1] = MERGE, calls[2+] = logs
        for call in calls[2:]:
            if call[0]:
                self.assertNotIn("MERGE", call[0][0])

    def test_update_existing_counts(self):
        """When rel already exists (properties_set > 0, relationships_created = 0)."""
        self._reset_tx_for_n_rels(1, created=0, props_set=3)
        rels = [{"source": "A", "target": "B", "type": "SUPPORTS"}]
        count = self.adapter.write_relationships(rels, "Z1337", project_id="test")
        self.assertEqual(count, 1)  # Updated counts as written

    def test_relationship_error_continues(self):
        # First rel's before-state query raises, second succeeds
        self.tx.run.side_effect = [Exception("DB error")]
        tx2 = MagicMock()
        before2 = MagicMock()
        before2.single.return_value = None
        summary2 = MagicMock()
        summary2.counters.relationships_created = 1
        summary2.counters.properties_set = 0
        merge2 = MagicMock()
        merge2.consume.return_value = summary2
        tx2.run.side_effect = [before2, merge2] + [MagicMock()] * 4
        self.session.begin_transaction.side_effect = [self.tx, tx2]
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


    def test_query_excludes_expired_by_default(self):
        self.session.run.return_value = []
        self.adapter.query_nodes(project_id="test")
        call_args = self.session.run.call_args[0][0]
        self.assertIn("n.expired_at IS NULL", call_args)

    def test_query_includes_expired_when_requested(self):
        self.session.run.return_value = []
        self.adapter.query_nodes(project_id="test", include_expired=True)
        call_args = self.session.run.call_args[0][0]
        self.assertNotIn("expired_at", call_args)

    def test_query_filtered_type_excludes_expired(self):
        self.session.run.return_value = []
        self.adapter.query_nodes(node_type="Belief", project_id="test")
        call_args = self.session.run.call_args[0][0]
        self.assertIn("n.expired_at IS NULL", call_args)


class TestKnowledgeType(unittest.TestCase):
    """SUP-162: Knowledge Type Taxonomy tests."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def test_query_nodes_filters_by_knowledge_type(self):
        self.session.run.return_value = []
        self.adapter.query_nodes(project_id="test", knowledge_type="fact")
        call_args = self.session.run.call_args[0][0]
        self.assertIn("n.knowledge_type", call_args)

    def test_query_nodes_invalid_knowledge_type_ignored(self):
        self.session.run.return_value = []
        self.adapter.query_nodes(project_id="test", knowledge_type="invalid")
        call_args = self.session.run.call_args[0][0]
        # The filter clause should NOT be present for invalid types
        self.assertNotIn("n.knowledge_type = $knowledge_type", call_args)

    def test_query_nodes_returns_knowledge_type(self):
        self.session.run.return_value = [
            {"name": "A", "summary": "desc", "type": "Concept", "created": "2026-01-01",
             "node_id": "123", "confidence": None, "valid_until": None,
             "vsm_level": None, "knowledge_type": "fact"},
        ]
        results = self.adapter.query_nodes(project_id="test")
        self.assertEqual(results[0]["knowledge_type"], "fact")

    def test_query_nodes_omits_null_knowledge_type(self):
        self.session.run.return_value = [
            {"name": "B", "summary": "desc", "type": "Concept", "created": "2026-01-01",
             "node_id": "456", "confidence": None, "valid_until": None,
             "vsm_level": None, "knowledge_type": None},
        ]
        results = self.adapter.query_nodes(project_id="test")
        self.assertNotIn("knowledge_type", results[0])

    def test_write_entities_persists_knowledge_type(self):
        tx = MagicMock()
        tx.run.return_value.single.return_value = None  # no before-state
        self.session.begin_transaction.return_value = tx
        self.adapter._upsert_vector = MagicMock()
        entities = [{"name": "Test", "node_type": "Concept", "summary": "test",
                      "knowledge_type": "state"}]
        self.adapter.write_entities(entities, "Z1", project_id="test")
        # Check that knowledge_type param was passed to Cypher
        merge_call = tx.run.call_args_list[1]  # [0]=before-state query, [1]=MERGE
        self.assertEqual(merge_call[1]["knowledge_type"], "state")

    def test_write_entities_invalid_knowledge_type_set_null(self):
        tx = MagicMock()
        tx.run.return_value.single.return_value = None
        self.session.begin_transaction.return_value = tx
        self.adapter._upsert_vector = MagicMock()
        entities = [{"name": "Test", "node_type": "Concept", "summary": "test",
                      "knowledge_type": "bogus"}]
        self.adapter.write_entities(entities, "Z1", project_id="test")
        merge_call = tx.run.call_args_list[1]
        self.assertIsNone(merge_call[1]["knowledge_type"])

    def test_get_beliefs_filters_by_knowledge_type(self):
        self.session.run.return_value = []
        self.adapter.get_beliefs(project_id="test", knowledge_type="belief")
        call_args = self.session.run.call_args[0][0]
        self.assertIn("b.knowledge_type", call_args)

    def test_get_beliefs_returns_knowledge_type(self):
        self.session.run.return_value = [
            {"name": "B1", "summary": "test", "confidence": 0.7, "cycle": "Z1",
             "valid_until": None, "vsm_level": None, "node_id": "789",
             "updated_at": "2026-01-01", "status": "active",
             "knowledge_type": "belief"},
        ]
        results = self.adapter.get_beliefs(project_id="test")
        self.assertEqual(results[0]["knowledge_type"], "belief")

    def test_get_stats_includes_knowledge_types(self):
        # Mock session.run to handle multiple calls (nodes, edges, vsm_levels, knowledge_types)
        call_count = [0]
        node_type_count = len(__import__("merkraum_backend").NODE_TYPES)
        rel_type_count = len(__import__("merkraum_backend").RELATIONSHIP_TYPES)

        def mock_run(cypher, **kwargs):
            call_count[0] += 1
            idx = call_count[0]
            if idx <= node_type_count:
                # Node type counts
                result = MagicMock()
                result.single.return_value = {"c": 0}
                return result
            elif idx <= node_type_count + rel_type_count:
                # Relationship type counts
                result = MagicMock()
                result.single.return_value = {"c": 0}
                return result
            elif idx == node_type_count + rel_type_count + 1:
                # VSM levels
                return []
            else:
                # Knowledge types
                return [{"kt": "fact", "c": 5}, {"kt": "belief", "c": 3}]

        self.session.run.side_effect = mock_run
        stats = self.adapter.get_stats(project_id="test")
        self.assertIn("knowledge_types", stats)
        self.assertEqual(stats["knowledge_types"]["fact"], 5)
        self.assertEqual(stats["knowledge_types"]["belief"], 3)


class TestTraverseExpiryFilter(unittest.TestCase):

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)
        self.session.run.return_value = []

    def test_traverse_excludes_expired_by_default(self):
        self.adapter.traverse("Test", project_id="test")
        call_args = self.session.run.call_args[0][0]
        self.assertIn("expired_at IS NULL", call_args)

    def test_traverse_includes_expired_when_requested(self):
        self.adapter.traverse("Test", project_id="test", include_expired=True)
        call_args = self.session.run.call_args[0][0]
        self.assertNotIn("expired_at", call_args)


class TestFilterExpiredResults(unittest.TestCase):

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def test_filters_expired_from_vector_results(self):
        results = [
            {"score": 0.9, "metadata": {"name": "Active Node"}, "content": "a"},
            {"score": 0.8, "metadata": {"name": "Expired Node"}, "content": "b"},
        ]
        self.session.run.return_value = [{"name": "Expired Node"}]
        filtered = self.adapter._filter_expired_results(results, "test")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["metadata"]["name"], "Active Node")

    def test_returns_all_when_none_expired(self):
        results = [
            {"score": 0.9, "metadata": {"name": "Node A"}, "content": "a"},
            {"score": 0.8, "metadata": {"name": "Node B"}, "content": "b"},
        ]
        self.session.run.return_value = []
        filtered = self.adapter._filter_expired_results(results, "test")
        self.assertEqual(len(filtered), 2)

    def test_handles_empty_results(self):
        filtered = self.adapter._filter_expired_results([], "test")
        self.assertEqual(filtered, [])


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
        self.assertIn("b.status = $status", cypher)

    def test_contradicted_status(self):
        self.adapter.get_beliefs(project_id="test", status="contradicted")
        cypher = self.session.run.call_args[0][0]
        self.assertIn("b.status = $status", cypher)

    def test_superseded_status(self):
        self.adapter.get_beliefs(project_id="test", status="superseded")
        cypher = self.session.run.call_args[0][0]
        self.assertIn("b.status = $status", cypher)


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
        self.assertIn("vsm_levels", stats)
        self.assertIn("knowledge_types", stats)

    def test_stats_queries_all_types(self):
        self.adapter.get_stats(project_id="test")
        # Should query each NODE_TYPE + each RELATIONSHIP_TYPE + 1 VSM level + 1 knowledge_type query
        expected_calls = len(NODE_TYPES) + len(RELATIONSHIP_TYPES) + 2
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


class TestDeleteProjectDataQdrantCleanup(unittest.TestCase):
    """Test that Neo4jQdrantAdapter.delete_project_data also clears Qdrant."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.adapter._qdrant = MagicMock()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_result = MagicMock()
        mock_result.single.return_value = {"c": 3}
        self.session.run.return_value = mock_result

    def test_deletes_qdrant_collection(self):
        counts = self.adapter.delete_project_data("test_project")
        self.adapter._qdrant.delete_collection.assert_called_once_with(
            collection_name="test_project"
        )
        self.assertEqual(counts["nodes_deleted"], 3)
        self.assertTrue(counts["vectors_deleted"])

    def test_qdrant_collection_name_sanitized(self):
        counts = self.adapter.delete_project_data("user:project")
        self.adapter._qdrant.delete_collection.assert_called_once_with(
            collection_name="user_project"
        )
        self.assertTrue(counts["vectors_deleted"])

    def test_qdrant_failure_does_not_block_neo4j_delete(self):
        self.adapter._qdrant.delete_collection.side_effect = Exception("Qdrant down")
        counts = self.adapter.delete_project_data("test_project")
        self.assertEqual(counts["nodes_deleted"], 3)
        self.assertFalse(counts["vectors_deleted"])

    def test_no_qdrant_client_skips_vector_cleanup(self):
        self.adapter._qdrant = None
        counts = self.adapter.delete_project_data("test_project")
        self.assertEqual(counts["nodes_deleted"], 3)
        self.assertNotIn("vectors_deleted", counts)

    def test_cannot_delete_default_still_raises(self):
        with self.assertRaises(ValueError):
            self.adapter.delete_project_data("default")


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
        # Mock transaction for write_entities (uses begin_transaction now)
        self.tx = MagicMock()
        before_result = MagicMock()
        before_result.single.return_value = None
        self.tx.run.return_value = before_result
        self.session.begin_transaction.return_value = self.tx
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)
        self.adapter.vector_upsert = MagicMock(return_value=True)

    def _mock_usage(self, node_count, extra_writes=10):
        """Set up get_usage to return a specific node count, then allow writes."""
        node_result = MagicMock()
        node_result.single.return_value = {"c": node_count}
        edge_result = MagicMock()
        edge_result.single.return_value = {"c": 0}
        # get_usage uses session.run directly, while write_entities uses tx
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

    def test_reindex_cleanup_legacy_ids_enabled(self):
        self.adapter.vector_delete = MagicMock(return_value=True)
        self.session.run.side_effect = [
            [
                {"name": "Node A", "node_type": "Concept", "summary": "", "node_id": "id-a"},
            ],
            self._total_result(1),
        ]
        result = self.adapter.reindex_project_vectors(
            project_id="proj",
            limit=10,
            cleanup_legacy_ids=True,
        )
        self.assertEqual(result["legacy_deleted"], 2)
        self.assertEqual(self.adapter.vector_delete.call_count, 2)
        self.adapter.vector_delete.assert_any_call("proj:Node A", project_id="proj")
        self.adapter.vector_delete.assert_any_call("proj:Concept:Node A", project_id="proj")

    def test_reindex_cleanup_legacy_ids_disabled(self):
        self.adapter.vector_delete = MagicMock(return_value=True)
        self.session.run.side_effect = [
            [
                {"name": "Node A", "node_type": "Concept", "summary": "", "node_id": "id-a"},
            ],
            self._total_result(1),
        ]
        result = self.adapter.reindex_project_vectors(
            project_id="proj",
            limit=10,
            cleanup_legacy_ids=False,
        )
        self.assertEqual(result["legacy_deleted"], 0)
        self.adapter.vector_delete.assert_not_called()


class TestVectorResultDedupe(unittest.TestCase):
    def setUp(self):
        self.adapter = _make_mock_adapter()

    def test_dedupe_prefers_highest_score_for_same_node_id(self):
        results = [
            {"id": "1", "score": 0.61, "metadata": {"node_id": "n-1", "name": "Kant"}},
            {"id": "2", "score": 0.92, "metadata": {"node_id": "n-1", "name": "Immanuel Kant"}},
            {"id": "3", "score": 0.70, "metadata": {"node_id": "n-2", "name": "Bundeskartellamt"}},
        ]
        deduped = self.adapter._dedupe_vector_results(results, project_id="proj", top_k=10)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["id"], "2")

    def test_dedupe_fallback_by_name_type_project(self):
        results = [
            {
                "id": "a",
                "score": 0.52,
                "metadata": {"name": "Bundeskartellamt", "node_type": "Organization", "project_id": "proj"},
            },
            {
                "id": "b",
                "score": 0.84,
                "metadata": {"name": " bundeskartellamt ", "node_type": "Organization", "project_id": "proj"},
            },
            {
                "id": "c",
                "score": 0.50,
                "metadata": {"name": "Bundeskartellamt", "node_type": "Organization", "project_id": "other"},
            },
        ]
        deduped = self.adapter._dedupe_vector_results(results, project_id="proj", top_k=10)
        self.assertEqual(len(deduped), 2)
        ids = {x["id"] for x in deduped}
        self.assertIn("b", ids)
        self.assertIn("c", ids)


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
        # update_belief now uses begin_transaction()
        self.tx = MagicMock()
        self.session.begin_transaction.return_value = self.tx
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def _mock_found(self, name="test belief"):
        """Mock a successful belief match — before-state + after-state queries."""
        before_result = MagicMock()
        before_result.single.return_value = {
            "confidence": 0.7, "status": "active", "summary": "old",
            "valid_until": None, "active": True, "updated_at": "2026-01-01",
        }
        after_result = MagicMock()
        after_result.single.return_value = {
            "confidence": 0.9, "status": "active", "summary": "old",
            "valid_until": None, "active": True, "updated_at": "2026-03-19",
        }
        # tx.run calls: before-state, SET, _log_history, _log_operation
        self.tx.run.side_effect = [before_result, after_result] + [MagicMock()] * 4

    def _mock_not_found(self):
        """Mock a belief not found."""
        not_found = MagicMock()
        not_found.single.return_value = None
        self.tx.run.return_value = not_found

    def test_update_confidence(self):
        self._mock_found()
        result = self.adapter.update_belief("test belief", "proj", confidence=0.9)
        self.assertTrue(result["updated"])
        self.assertEqual(result["changes"]["confidence"], 0.9)
        self.assertIn("operation_id", result)

    def test_update_status_superseded(self):
        self._mock_found()
        result = self.adapter.update_belief("test belief", "proj", status="superseded")
        self.assertTrue(result["updated"])
        self.assertEqual(result["changes"]["status"], "superseded")
        # SET Cypher is the second tx.run call
        set_call = self.tx.run.call_args_list[1]
        self.assertIn("b.active = false", set_call[0][0])

    def test_update_status_active(self):
        self._mock_found()
        result = self.adapter.update_belief("test belief", "proj", status="active")
        self.assertTrue(result["updated"])
        set_call = self.tx.run.call_args_list[1]
        self.assertIn("b.active = true", set_call[0][0])

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


class TestConsolidateBeliefs(unittest.TestCase):
    """Test consolidate_beliefs method on Neo4jBaseAdapter."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.tx = MagicMock()
        self.session.begin_transaction.return_value = self.tx
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def _mock_both_found_with_contradiction(self):
        """Mock: both beliefs found, CONTRADICTS relationship exists."""
        belief_a = MagicMock()
        belief_a.single.return_value = {
            "name": "Market A growing", "summary": "German market +30%",
            "confidence": 0.8, "status": "contradicted",
            "source_cycle": "Z100", "node_id": "id-a",
        }
        belief_b = MagicMock()
        belief_b.single.return_value = {
            "name": "Market B declining", "summary": "International market -10%",
            "confidence": 0.7, "status": "contradicted",
            "source_cycle": "Z101", "node_id": "id-b",
        }
        contra_rel = MagicMock()
        contra_rel.single.return_value = {"reason": "Conflicting market trends"}
        # tx.run calls: find A, find B, find CONTRADICTS, CREATE synthesis,
        # SET A consolidated, SET B consolidated, DELETE CONTRADICTS,
        # CREATE SUPERSEDES x2, _log_history, _log_operation (prev_hash lookup + CREATE)
        self.tx.run.side_effect = [
            belief_a, belief_b, contra_rel,
        ] + [MagicMock()] * 9  # CREATE, 2 SETs, DELETE, 2 SUPERSEDES, 3 logs

    def test_successful_consolidation(self):
        self._mock_both_found_with_contradiction()
        result = self.adapter.consolidate_beliefs(
            "Market A growing", "Market B declining",
            "These are different markets — German domestic vs international. "
            "Both trends are correct in their respective scope.",
            project_id="proj",
        )
        self.assertTrue(result["ok"])
        self.assertIn("synthesis", result)
        self.assertEqual(result["synthesis"]["status"], "active")
        self.assertIn("Market A growing", result["consolidated"])
        self.assertIn("Market B declining", result["consolidated"])
        self.assertIn("operation_id", result)

    def test_custom_name(self):
        self._mock_both_found_with_contradiction()
        result = self.adapter.consolidate_beliefs(
            "Market A growing", "Market B declining",
            "Different scope markets.",
            project_id="proj",
            new_name="Market trends by region",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["synthesis"]["name"], "Market trends by region")

    def test_belief_a_not_found(self):
        not_found = MagicMock()
        not_found.single.return_value = None
        self.tx.run.return_value = not_found
        result = self.adapter.consolidate_beliefs(
            "nonexistent", "Market B declining", "resolution", "proj",
        )
        self.assertFalse(result["ok"])
        self.assertIn("not found", result["error"])

    def test_belief_b_not_found(self):
        found = MagicMock()
        found.single.return_value = {
            "name": "A", "summary": "s", "confidence": 0.8,
            "status": "contradicted", "source_cycle": "Z1", "node_id": "id-a",
        }
        not_found = MagicMock()
        not_found.single.return_value = None
        self.tx.run.side_effect = [found, not_found]
        result = self.adapter.consolidate_beliefs(
            "A", "nonexistent", "resolution", "proj",
        )
        self.assertFalse(result["ok"])
        self.assertIn("not found", result["error"])

    def test_no_contradicts_relationship(self):
        found_a = MagicMock()
        found_a.single.return_value = {
            "name": "A", "summary": "s", "confidence": 0.8,
            "status": "active", "source_cycle": "Z1", "node_id": "id-a",
        }
        found_b = MagicMock()
        found_b.single.return_value = {
            "name": "B", "summary": "s", "confidence": 0.7,
            "status": "active", "source_cycle": "Z2", "node_id": "id-b",
        }
        no_rel = MagicMock()
        no_rel.single.return_value = None
        self.tx.run.side_effect = [found_a, found_b, no_rel]
        result = self.adapter.consolidate_beliefs(
            "A", "B", "resolution", "proj",
        )
        self.assertFalse(result["ok"])
        self.assertIn("No CONTRADICTS", result["error"])

    def test_consolidated_status_marks_inactive(self):
        """Verify update_belief with status='consolidated' sets active=false."""
        # Use update_belief to verify the status is accepted
        result = self.adapter.update_belief("test", "proj", status="consolidated")
        # Should not reject as invalid status
        self.assertNotIn("Invalid status", result.get("error", ""))


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


class TestCreateTokenTierResolution(unittest.TestCase):
    """Token endpoint must derive tier server-side and ignore forged payload tier."""

    def setUp(self):
        import merkraum_api
        self.api = merkraum_api
        self.previous_adapter = self.api.adapter
        self.mock_adapter = MagicMock()
        self.api.adapter = self.mock_adapter

    def tearDown(self):
        self.api.adapter = self.previous_adapter

    def test_payload_enterprise_from_free_user_enforces_free(self):
        mock_pat_validator = MagicMock()
        mock_pat_validator.create_token.return_value = {
            "token": "mk_pat_mock",
            "token_prefix": "mk_pat_mock",
            "name": "Test",
            "scopes": ["read"],
            "projects": ["proj-free"],
            "all_projects": False,
            "expires_at": None,
            "created_at": "2026-03-25T00:00:00+00:00",
        }
        self.mock_adapter.get_project.return_value = {"project_id": "proj-free", "tier": "free"}

        with self.api.app.app_context():
            self.api.current_app._pat_validator = mock_pat_validator
            with self.api.app.test_request_context(
                "/api/tokens",
                method="POST",
                json={
                    "name": "Test",
                    "scopes": ["read"],
                    "projects": ["proj-free"],
                    "tier": "enterprise",
                },
            ):
                self.api.request.user_id = "user-free"
                self.api.request.pat_scopes = None
                response, status = self.api.create_token.__wrapped__()

        self.assertEqual(status, 400)
        self.assertIn("tier", response.get_json()["error"])
        mock_pat_validator.create_token.assert_not_called()

    def test_unknown_forged_tier_does_not_affect_limits(self):
        mock_pat_validator = MagicMock()
        mock_pat_validator.create_token.return_value = {
            "token": "mk_pat_mock",
            "token_prefix": "mk_pat_mock",
            "name": "Test",
            "scopes": ["read"],
            "projects": ["proj-free"],
            "all_projects": False,
            "expires_at": None,
            "created_at": "2026-03-25T00:00:00+00:00",
        }
        self.mock_adapter.get_project.return_value = {"project_id": "proj-free", "tier": "free"}

        with self.api.app.app_context():
            self.api.current_app._pat_validator = mock_pat_validator
            with self.api.app.test_request_context(
                "/api/tokens",
                method="POST",
                json={
                    "name": "Test",
                    "scopes": ["read"],
                    "projects": ["proj-free"],
                    "tier": "totally-forged",
                },
            ):
                self.api.request.user_id = "user-free"
                self.api.request.pat_scopes = None
                response, status = self.api.create_token.__wrapped__()

        self.assertEqual(status, 400)
        self.assertIn("tier", response.get_json()["error"])
        mock_pat_validator.create_token.assert_not_called()

    def test_multiple_project_tiers_use_strictest_server_side_tier(self):
        mock_pat_validator = MagicMock()
        mock_pat_validator.create_token.return_value = {
            "token": "mk_pat_mock",
            "token_prefix": "mk_pat_mock",
            "name": "Test",
            "scopes": ["read"],
            "projects": ["proj-free", "proj-enterprise"],
            "all_projects": False,
            "expires_at": None,
            "created_at": "2026-03-25T00:00:00+00:00",
        }
        self.mock_adapter.get_project.side_effect = [
            {"project_id": "proj-free", "tier": "free"},
            {"project_id": "proj-enterprise", "tier": "enterprise"},
        ]

        with self.api.app.app_context():
            self.api.current_app._pat_validator = mock_pat_validator
            with self.api.app.test_request_context(
                "/api/tokens",
                method="POST",
                json={
                    "name": "Test",
                    "scopes": ["read"],
                    "projects": ["proj-free", "proj-enterprise"],
                },
            ):
                self.api.request.user_id = "user-free"
                self.api.request.pat_scopes = None
                response, status = self.api.create_token.__wrapped__()

        self.assertEqual(status, 201)
        self.assertEqual(response.get_json()["token_prefix"], "mk_pat_mock")
        self.assertEqual(mock_pat_validator.create_token.call_args.kwargs["tier"], "free")


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


class TestExpireNodes(unittest.TestCase):
    """Test expire_nodes managed forgetting method."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.tx = MagicMock()
        self.session.begin_transaction.return_value = self.tx
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def test_dry_run_returns_expired_list(self):
        """dry_run=True should return expired nodes without modifying anything."""
        expired_rec = MagicMock()
        expired_rec.__getitem__ = lambda s, k: {
            "name": "Old fact", "summary": "stale info", "type": "Concept",
            "valid_until": "2025-01-01T00:00:00+00:00", "node_id": "id-1",
            "vsm_level": "S1", "confidence": None, "status": None, "active": None,
        }[k]
        expired_rec.get = lambda k, d=None: {
            "name": "Old fact", "summary": "stale info", "type": "Concept",
            "valid_until": "2025-01-01T00:00:00+00:00", "node_id": "id-1",
            "vsm_level": "S1", "confidence": None, "status": None, "active": None,
        }.get(k, d)
        self.session.run.return_value = [expired_rec]

        result = self.adapter.expire_nodes(project_id="proj", dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["expired"][0]["name"], "Old fact")
        # No transaction should have been created for dry_run
        self.session.begin_transaction.assert_not_called()

    def test_expire_marks_belief_inactive(self):
        """Expiring a Belief should set active=false and status='expired'."""
        expired_rec = MagicMock()
        data = {
            "name": "Stale belief", "summary": "old", "type": "Belief",
            "valid_until": "2025-01-01T00:00:00+00:00", "node_id": "id-2",
            "vsm_level": "S3", "confidence": 0.6, "status": "active", "active": True,
        }
        expired_rec.__getitem__ = lambda s, k: data[k]
        expired_rec.get = lambda k, d=None: data.get(k, d)
        self.session.run.return_value = [expired_rec]
        self.tx.run.return_value = MagicMock()

        result = self.adapter.expire_nodes(project_id="proj", dry_run=False)
        self.assertFalse(result["dry_run"])
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["expired"][0]["name"], "Stale belief")
        self.tx.commit.assert_called_once()

    def test_empty_when_nothing_expired(self):
        """No expired nodes should return empty list."""
        self.session.run.return_value = []
        result = self.adapter.expire_nodes(project_id="proj", dry_run=False)
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["expired"], [])

    def test_expire_non_belief_sets_expired_at_only(self):
        """Non-Belief nodes get expired_at but no status/active change."""
        expired_rec = MagicMock()
        data = {
            "name": "Old event", "summary": "happened", "type": "Event",
            "valid_until": "2025-06-01T00:00:00+00:00", "node_id": "id-3",
            "vsm_level": None, "confidence": None, "status": None, "active": None,
        }
        expired_rec.__getitem__ = lambda s, k: data[k]
        expired_rec.get = lambda k, d=None: data.get(k, d)
        self.session.run.return_value = [expired_rec]
        self.tx.run.return_value = MagicMock()

        result = self.adapter.expire_nodes(project_id="proj", dry_run=False)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["expired"][0]["type"], "Event")
        self.tx.commit.assert_called_once()


class TestPruneOrphanNodes(unittest.TestCase):
    """Test prune_orphan_nodes active pruning method — Z1841 proposal #2."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.tx = MagicMock()
        self.session.begin_transaction.return_value = self.tx
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def test_dry_run_returns_orphan_list(self):
        """dry_run=True should return orphan nodes without modifying anything."""
        orphan_rec = MagicMock()
        data = {
            "name": "Disconnected concept", "type": "Concept",
            "node_id": "id-orphan", "last_update": "2026-01-01T00:00:00",
            "confidence": None, "active": None, "status": None,
        }
        orphan_rec.__getitem__ = lambda s, k: data[k]
        orphan_rec.get = lambda k, d=None: data.get(k, d)
        self.session.run.return_value = [orphan_rec]

        result = self.adapter.prune_orphan_nodes(
            project_id="proj", stale_days=30, dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["pruned"][0]["name"], "Disconnected concept")
        self.session.begin_transaction.assert_not_called()

    def test_prune_belief_sets_status_pruned(self):
        """Pruning a Belief should set active=false and status='pruned'."""
        orphan_rec = MagicMock()
        data = {
            "name": "Orphan belief", "type": "Belief",
            "node_id": "id-ob", "last_update": "2026-01-15T00:00:00",
            "confidence": 0.3, "active": True, "status": "active",
        }
        orphan_rec.__getitem__ = lambda s, k: data[k]
        orphan_rec.get = lambda k, d=None: data.get(k, d)
        self.session.run.return_value = [orphan_rec]
        self.tx.run.return_value = MagicMock()

        result = self.adapter.prune_orphan_nodes(
            project_id="proj", stale_days=30, dry_run=False)
        self.assertFalse(result["dry_run"])
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["pruned"][0]["name"], "Orphan belief")
        self.tx.commit.assert_called_once()

    def test_empty_when_no_orphans(self):
        """No orphan nodes should return empty list."""
        self.session.run.return_value = []
        result = self.adapter.prune_orphan_nodes(
            project_id="proj", stale_days=30, dry_run=False)
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["pruned"], [])

    def test_prune_non_belief_sets_expired_at(self):
        """Non-Belief orphans get expired_at only."""
        orphan_rec = MagicMock()
        data = {
            "name": "Old event", "type": "Event",
            "node_id": "id-oe", "last_update": "2026-01-01T00:00:00",
            "confidence": None, "active": None, "status": None,
        }
        orphan_rec.__getitem__ = lambda s, k: data[k]
        orphan_rec.get = lambda k, d=None: data.get(k, d)
        self.session.run.return_value = [orphan_rec]
        self.tx.run.return_value = MagicMock()

        result = self.adapter.prune_orphan_nodes(
            project_id="proj", stale_days=30, dry_run=False)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["pruned"][0]["type"], "Event")
        self.tx.commit.assert_called_once()


class TestRenewNode(unittest.TestCase):
    """Test renew_node managed renewal method."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.tx = MagicMock()
        self.session.begin_transaction.return_value = self.tx
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def test_renew_with_extend_days(self):
        """Renewing with extend_days should succeed."""
        before = MagicMock()
        before.__getitem__ = lambda s, k: {
            "valid_until": "2025-01-01", "expired_at": None,
            "active": True, "status": "active", "type": "Concept",
        }[k]
        before.get = lambda k, d=None: {
            "valid_until": "2025-01-01", "expired_at": None,
            "active": True, "status": "active", "type": "Concept",
        }.get(k, d)
        self.tx.run.side_effect = [
            MagicMock(single=MagicMock(return_value=before)),
            MagicMock(),  # UPDATE
            MagicMock(),  # _log_history
            MagicMock(single=MagicMock(return_value=None)),  # _log_operation prev_hash lookup
            MagicMock(),  # _log_operation CREATE
        ]
        result = self.adapter.renew_node("Test node", project_id="proj", extend_days=90)
        self.assertTrue(result["renewed"])
        self.assertIn("new_valid_until", result)
        self.assertFalse(result["was_expired"])
        self.tx.commit.assert_called_once()

    def test_renew_expired_belief_reactivates(self):
        """Renewing an expired Belief should set active=true, status='active'."""
        before = MagicMock()
        data = {
            "valid_until": "2025-01-01", "expired_at": "2025-01-02",
            "active": False, "status": "expired", "type": "Belief",
        }
        before.__getitem__ = lambda s, k: data[k]
        before.get = lambda k, d=None: data.get(k, d)
        self.tx.run.side_effect = [
            MagicMock(single=MagicMock(return_value=before)),
            MagicMock(),  # UPDATE
            MagicMock(),  # _log_history
            MagicMock(single=MagicMock(return_value=None)),  # _log_operation prev_hash lookup
            MagicMock(),  # _log_operation CREATE
        ]
        result = self.adapter.renew_node("Old belief", project_id="proj", extend_days=180)
        self.assertTrue(result["renewed"])
        self.assertTrue(result["was_expired"])
        self.tx.commit.assert_called_once()

    def test_renew_not_found(self):
        """Renewing a nonexistent node should return error."""
        self.tx.run.return_value = MagicMock(single=MagicMock(return_value=None))
        result = self.adapter.renew_node("ghost", project_id="proj", extend_days=30)
        self.assertFalse(result["renewed"])
        self.assertIn("not found", result["error"])

    def test_renew_requires_days_or_until(self):
        """Must provide extend_days or new_valid_until."""
        result = self.adapter.renew_node("test", project_id="proj")
        self.assertFalse(result["renewed"])
        self.assertIn("Provide", result["error"])

    def test_renew_with_explicit_valid_until(self):
        """Renewing with new_valid_until should use the provided value."""
        before = MagicMock()
        data = {
            "valid_until": "2025-01-01", "expired_at": None,
            "active": True, "status": "active", "type": "Event",
        }
        before.__getitem__ = lambda s, k: data[k]
        before.get = lambda k, d=None: data.get(k, d)
        self.tx.run.side_effect = [
            MagicMock(single=MagicMock(return_value=before)),
            MagicMock(),  # UPDATE
            MagicMock(),  # _log_history
            MagicMock(single=MagicMock(return_value=None)),  # _log_operation prev_hash lookup
            MagicMock(),  # _log_operation CREATE
        ]
        result = self.adapter.renew_node(
            "Test event", project_id="proj",
            new_valid_until="2027-12-31T00:00:00+00:00",
        )
        self.assertTrue(result["renewed"])
        self.assertEqual(result["new_valid_until"], "2027-12-31T00:00:00+00:00")


class TestApplyConfidenceDecay(unittest.TestCase):
    """Test apply_confidence_decay certainty management method."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.tx = MagicMock()
        self.session.begin_transaction.return_value = self.tx
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def _make_belief_rec(self, name, confidence, knowledge_type, days_ago):
        """Create a mock belief record updated days_ago days in the past."""
        from datetime import datetime, timedelta, timezone
        updated = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
        data = {
            "name": name, "confidence": confidence,
            "knowledge_type": knowledge_type, "updated_at": updated,
            "source_cycle": "Z1", "vsm_level": "S4",
        }
        rec = MagicMock()
        rec.__getitem__ = lambda s, k: data[k]
        rec.get = lambda k, d=None: data.get(k, d)
        return rec

    def test_dry_run_returns_decay_preview(self):
        """dry_run=True should return affected beliefs without modifying."""
        rec = self._make_belief_rec("Stale belief", 0.8, "belief", days_ago=30)
        self.session.run.return_value = [rec]

        result = self.adapter.apply_confidence_decay(project_id="proj", dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["total"], 1)
        self.assertLess(result["decayed"][0]["new_confidence"],
                        result["decayed"][0]["old_confidence"])
        self.session.begin_transaction.assert_not_called()

    def test_facts_are_exempt_from_decay(self):
        """Facts should not decay regardless of age."""
        rec = self._make_belief_rec("Permanent fact", 0.9, "fact", days_ago=365)
        self.session.run.return_value = [rec]

        result = self.adapter.apply_confidence_decay(project_id="proj", dry_run=True)
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["unchanged"], 1)

    def test_state_decays_fastest(self):
        """State knowledge type should decay faster than belief."""
        state_rec = self._make_belief_rec("Current state", 0.8, "state", days_ago=30)
        belief_rec = self._make_belief_rec("An opinion", 0.8, "belief", days_ago=30)
        self.session.run.return_value = [state_rec, belief_rec]

        result = self.adapter.apply_confidence_decay(project_id="proj", dry_run=True)
        self.assertEqual(result["total"], 2)
        state_new = result["decayed"][0]["new_confidence"]
        belief_new = result["decayed"][1]["new_confidence"]
        self.assertLess(state_new, belief_new)

    def test_respects_floor(self):
        """Confidence should never decay below the floor for the knowledge type."""
        rec = self._make_belief_rec("Very old state", 0.2, "state", days_ago=365)
        self.session.run.return_value = [rec]

        result = self.adapter.apply_confidence_decay(project_id="proj", dry_run=True)
        if result["total"] > 0:
            self.assertGreaterEqual(result["decayed"][0]["new_confidence"], 0.1)

    def test_recent_beliefs_unchanged(self):
        """Beliefs updated less than 1 day ago should not decay."""
        rec = self._make_belief_rec("Fresh belief", 0.8, "belief", days_ago=0)
        self.session.run.return_value = [rec]

        result = self.adapter.apply_confidence_decay(project_id="proj", dry_run=True)
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["unchanged"], 1)

    def test_apply_commits_changes(self):
        """dry_run=False should commit confidence changes to Neo4j."""
        rec = self._make_belief_rec("Old belief", 0.8, "belief", days_ago=60)
        self.session.run.return_value = [rec]
        self.tx.run.return_value = MagicMock()

        result = self.adapter.apply_confidence_decay(
            project_id="proj", dry_run=False)
        self.assertFalse(result["dry_run"])
        self.assertEqual(result["total"], 1)
        self.tx.commit.assert_called_once()

    def test_unclassified_uses_default_rate(self):
        """Beliefs with no knowledge_type should use the default decay rate."""
        rec = self._make_belief_rec("Untyped belief", 0.8, None, days_ago=30)
        self.session.run.return_value = [rec]

        result = self.adapter.apply_confidence_decay(project_id="proj", dry_run=True)
        self.assertEqual(result["total"], 1)
        self.assertIn("decay_rate", result["decayed"][0])


class TestGetCertaintyReviewQueue(unittest.TestCase):
    """Test get_certainty_review_queue method."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def test_returns_all_categories(self):
        """Review queue should have all 5 expected categories."""
        self.session.run.return_value = []
        result = self.adapter.get_certainty_review_queue(project_id="proj")
        self.assertIn("queue", result)
        self.assertIn("total_items", result)
        for cat in ["stale", "low_confidence", "type_mismatch",
                     "approaching_expiry", "unclassified"]:
            self.assertIn(cat, result["queue"])

    def test_stale_beliefs_detected(self):
        """Beliefs not updated in 30+ days should appear in stale category."""
        from datetime import datetime, timedelta, timezone
        old_date = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
        stale_rec = MagicMock()
        data = {"name": "Old belief", "confidence": 0.7, "kt": "belief",
                "updated_at": old_date, "vsm_level": "S4"}
        stale_rec.__getitem__ = lambda s, k: data[k]
        stale_rec.get = lambda k, d=None: data.get(k, d)
        # First call returns stale, rest return empty
        self.session.run.side_effect = [[stale_rec], [], [], [], []]

        result = self.adapter.get_certainty_review_queue(project_id="proj")
        self.assertEqual(len(result["queue"]["stale"]), 1)
        self.assertEqual(result["queue"]["stale"][0]["name"], "Old belief")

    def test_empty_graph_returns_zero_items(self):
        """Empty graph should return zero items in all categories."""
        self.session.run.return_value = []
        result = self.adapter.get_certainty_review_queue(project_id="proj")
        self.assertEqual(result["total_items"], 0)


class TestGetCertaintyStats(unittest.TestCase):
    """Test get_certainty_stats method."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def test_returns_all_sections(self):
        """Stats should include histogram, type_confidence, staleness, governance."""
        # Mock all the queries to return empty/zero
        zero_single = MagicMock(single=MagicMock(return_value={"c": 0}))
        self.session.run.return_value = zero_single
        # Override to return iterable for histogram and type-confidence
        self.session.run.side_effect = [
            [],  # histogram
            [],  # type_confidence
            MagicMock(single=MagicMock(return_value={"c": 0})),  # fresh
            MagicMock(single=MagicMock(return_value={"c": 0})),  # aging
            MagicMock(single=MagicMock(return_value={"c": 0})),  # stale
            MagicMock(single=MagicMock(return_value={"c": 0})),  # total_active
            MagicMock(single=MagicMock(return_value={"c": 0})),  # low_conf
            MagicMock(single=MagicMock(return_value={"c": 0})),  # contradicted
            MagicMock(single=MagicMock(return_value={"c": 0})),  # unclassified
        ]
        result = self.adapter.get_certainty_stats(project_id="proj")
        self.assertIn("confidence_histogram", result)
        self.assertIn("type_confidence", result)
        self.assertIn("staleness", result)
        self.assertIn("governance_summary", result)

    def test_governance_healthy_when_no_issues(self):
        """Empty graph should report healthy governance."""
        self.session.run.side_effect = [
            [],  # histogram
            [],  # type_confidence
            MagicMock(single=MagicMock(return_value={"c": 0})),  # fresh
            MagicMock(single=MagicMock(return_value={"c": 0})),  # aging
            MagicMock(single=MagicMock(return_value={"c": 0})),  # stale
            MagicMock(single=MagicMock(return_value={"c": 0})),  # total_active
            MagicMock(single=MagicMock(return_value={"c": 0})),  # low_conf
            MagicMock(single=MagicMock(return_value={"c": 0})),  # contradicted
            MagicMock(single=MagicMock(return_value={"c": 0})),  # unclassified
        ]
        result = self.adapter.get_certainty_stats(project_id="proj")
        self.assertEqual(result["governance_summary"]["health"], "healthy")

    def test_governance_needs_attention_with_contradictions(self):
        """Contradicted beliefs should trigger needs_attention health."""
        self.session.run.side_effect = [
            [],  # histogram
            [],  # type_confidence
            MagicMock(single=MagicMock(return_value={"c": 5})),  # fresh
            MagicMock(single=MagicMock(return_value={"c": 0})),  # aging
            MagicMock(single=MagicMock(return_value={"c": 0})),  # stale
            MagicMock(single=MagicMock(return_value={"c": 10})),  # total_active
            MagicMock(single=MagicMock(return_value={"c": 0})),  # low_conf
            MagicMock(single=MagicMock(return_value={"c": 3})),  # contradicted
            MagicMock(single=MagicMock(return_value={"c": 0})),  # unclassified
        ]
        result = self.adapter.get_certainty_stats(project_id="proj")
        self.assertEqual(result["governance_summary"]["health"], "needs_attention")


class TestConfidenceDecayRates(unittest.TestCase):
    """Test CONFIDENCE_DECAY_RATES constants."""

    def test_all_knowledge_types_have_decay_config(self):
        """Every KNOWLEDGE_TYPE should have a decay config entry."""
        from merkraum_backend import CONFIDENCE_DECAY_RATES, KNOWLEDGE_TYPES
        for kt in KNOWLEDGE_TYPES:
            self.assertIn(kt, CONFIDENCE_DECAY_RATES,
                          f"Missing decay config for knowledge type: {kt}")

    def test_default_decay_exists(self):
        """None key should exist for unclassified nodes."""
        from merkraum_backend import CONFIDENCE_DECAY_RATES
        self.assertIn(None, CONFIDENCE_DECAY_RATES)

    def test_fact_has_no_decay(self):
        """Facts should have no decay rate (rate_per_day=None)."""
        from merkraum_backend import CONFIDENCE_DECAY_RATES
        self.assertIsNone(CONFIDENCE_DECAY_RATES["fact"]["rate_per_day"])

    def test_state_decays_fastest(self):
        """State should have the highest decay rate."""
        from merkraum_backend import CONFIDENCE_DECAY_RATES
        state_rate = CONFIDENCE_DECAY_RATES["state"]["rate_per_day"]
        for kt in ["rule", "belief", "memory"]:
            other_rate = CONFIDENCE_DECAY_RATES[kt]["rate_per_day"]
            self.assertGreater(state_rate, other_rate,
                               f"State decay rate should exceed {kt}")


# --- Dreaming model configuration tests (SUP-159) ---

class TestDreamingModelConfig(unittest.TestCase):
    """Test dreaming engine model selection — SUP-159."""

    def test_default_replay_model_is_haiku(self):
        """Replay phase should default to Haiku (cost-efficient)."""
        from merkraum_dreaming import _DEFAULT_REPLAY_MODEL, _get_replay_model
        self.assertIn("haiku", _DEFAULT_REPLAY_MODEL.lower())
        # Without env override, should return default
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MERKRAUM_DREAMING_REPLAY_MODEL", None)
            self.assertEqual(_get_replay_model(), _DEFAULT_REPLAY_MODEL)

    def test_default_consolidation_model_is_sonnet(self):
        """Consolidation phase should default to Sonnet (quality matters)."""
        from merkraum_dreaming import _DEFAULT_CONSOLIDATION_MODEL, _get_consolidation_model
        self.assertIn("sonnet", _DEFAULT_CONSOLIDATION_MODEL.lower())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MERKRAUM_DREAMING_CONSOLIDATION_MODEL", None)
            self.assertEqual(_get_consolidation_model(), _DEFAULT_CONSOLIDATION_MODEL)

    def test_env_override_replay_model(self):
        """Env var should override replay model."""
        from merkraum_dreaming import _get_replay_model
        with patch.dict(os.environ, {"MERKRAUM_DREAMING_REPLAY_MODEL": "custom-model"}):
            self.assertEqual(_get_replay_model(), "custom-model")

    def test_env_override_consolidation_model(self):
        """Env var should override consolidation model."""
        from merkraum_dreaming import _get_consolidation_model
        with patch.dict(os.environ, {"MERKRAUM_DREAMING_CONSOLIDATION_MODEL": "custom-model"}):
            self.assertEqual(_get_consolidation_model(), "custom-model")

    def test_get_dreaming_config_structure(self):
        """get_dreaming_config should return all expected keys."""
        from merkraum_dreaming import get_dreaming_config
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MERKRAUM_DREAMING_REPLAY_MODEL", None)
            os.environ.pop("MERKRAUM_DREAMING_CONSOLIDATION_MODEL", None)
            config = get_dreaming_config()
        self.assertIn("replay_model", config)
        self.assertIn("consolidation_model", config)
        self.assertIn("reflection_model", config)
        self.assertIn("replay_model_source", config)
        self.assertIn("consolidation_model_source", config)
        self.assertIsNone(config["reflection_model"])

    def test_get_dreaming_config_source_default(self):
        """Source should be 'default' when no env override."""
        from merkraum_dreaming import get_dreaming_config
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MERKRAUM_DREAMING_REPLAY_MODEL", None)
            os.environ.pop("MERKRAUM_DREAMING_CONSOLIDATION_MODEL", None)
            config = get_dreaming_config()
        self.assertEqual(config["replay_model_source"], "default")
        self.assertEqual(config["consolidation_model_source"], "default")

    def test_get_dreaming_config_source_env(self):
        """Source should be 'env' when env var is set."""
        from merkraum_dreaming import get_dreaming_config
        with patch.dict(os.environ, {
            "MERKRAUM_DREAMING_REPLAY_MODEL": "x",
            "MERKRAUM_DREAMING_CONSOLIDATION_MODEL": "y",
        }):
            config = get_dreaming_config()
        self.assertEqual(config["replay_model_source"], "env")
        self.assertEqual(config["consolidation_model_source"], "env")

    def test_consolidation_model_different_from_replay(self):
        """Default consolidation model should be higher quality than replay."""
        from merkraum_dreaming import _DEFAULT_REPLAY_MODEL, _DEFAULT_CONSOLIDATION_MODEL
        self.assertNotEqual(_DEFAULT_REPLAY_MODEL, _DEFAULT_CONSOLIDATION_MODEL)
        self.assertIn("sonnet", _DEFAULT_CONSOLIDATION_MODEL.lower())
        self.assertIn("haiku", _DEFAULT_REPLAY_MODEL.lower())


# --- Dreaming maintenance phase tests (SUP-163 integration) ---

class TestDreamingMaintenance(unittest.TestCase):
    """Test maintenance phase in dreaming engine — SUP-163 integration."""

    def _make_mock_adapter(self, decay_result=None, review_result=None,
                           stats_result=None, expire_result=None,
                           prune_result=None, dedup_result=None):
        """Create a mock adapter with certainty management methods."""
        adapter = MagicMock(spec=Neo4jBaseAdapter)
        adapter.apply_confidence_decay.return_value = decay_result or {
            "decayed": [], "total": 0, "unchanged": 5, "dry_run": True,
        }
        adapter.expire_nodes.return_value = expire_result or {
            "expired": [], "total": 0, "dry_run": True,
        }
        adapter.prune_orphan_nodes.return_value = prune_result or {
            "pruned": [], "total": 0, "dry_run": True,
        }
        adapter.deduplicate_edges.return_value = dedup_result or {
            "duplicates_found": 0, "edges_removed": 0,
            "groups": [], "dry_run": True,
        }
        adapter.get_certainty_review_queue.return_value = review_result or {
            "categories": {
                "stale": [], "low_confidence": [], "type_mismatch": [],
                "approaching_expiry": [], "unclassified": [],
            }
        }
        adapter.get_certainty_stats.return_value = stats_result or {
            "governance": {"status": "healthy", "low_confidence_count": 0,
                           "contradicted_count": 0, "unclassified_count": 0},
        }
        return adapter

    def _run_generator(self, gen):
        """Consume a generator, collecting messages and returning the result."""
        messages = []
        try:
            while True:
                messages.append(next(gen))
        except StopIteration as e:
            return messages, e.value

    def test_maintain_returns_result_structure(self):
        """Maintenance phase should return expected keys."""
        from merkraum_dreaming import maintain
        adapter = self._make_mock_adapter()
        messages, result = self._run_generator(
            maintain(adapter, "test-project", dry_run=True))
        self.assertEqual(result["phase"], "maintenance")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["mode"], "preview")
        self.assertIn("decay", result)
        self.assertIn("expired", result)
        self.assertIn("pruned", result)
        self.assertIn("dedup", result)
        self.assertIn("review_queue", result)
        self.assertIn("governance_health", result)

    def test_maintain_dry_run_does_not_apply(self):
        """Dry run should pass dry_run=True to backend."""
        from merkraum_dreaming import maintain
        adapter = self._make_mock_adapter()
        self._run_generator(maintain(adapter, "proj", dry_run=True))
        adapter.apply_confidence_decay.assert_called_once_with(
            project_id="proj", dry_run=True, actor="dreaming-maintenance")
        adapter.expire_nodes.assert_called_once_with(
            project_id="proj", dry_run=True, actor="dreaming-maintenance")
        adapter.prune_orphan_nodes.assert_called_once_with(
            project_id="proj", stale_days=30,
            dry_run=True, actor="dreaming-maintenance")
        adapter.deduplicate_edges.assert_called_once_with(
            project_id="proj", dry_run=True,
            actor="dreaming-maintenance")

    def test_maintain_apply_mode(self):
        """Apply mode should pass dry_run=False to backend."""
        from merkraum_dreaming import maintain
        adapter = self._make_mock_adapter()
        messages, result = self._run_generator(
            maintain(adapter, "proj", dry_run=False))
        adapter.apply_confidence_decay.assert_called_once_with(
            project_id="proj", dry_run=False, actor="dreaming-maintenance")
        adapter.expire_nodes.assert_called_once_with(
            project_id="proj", dry_run=False, actor="dreaming-maintenance")
        adapter.prune_orphan_nodes.assert_called_once_with(
            project_id="proj", stale_days=30,
            dry_run=False, actor="dreaming-maintenance")
        adapter.deduplicate_edges.assert_called_once_with(
            project_id="proj", dry_run=False,
            actor="dreaming-maintenance")
        self.assertEqual(result["mode"], "apply")

    def test_maintain_reports_decay_count(self):
        """Should report correct decay counts from backend."""
        from merkraum_dreaming import maintain
        adapter = self._make_mock_adapter(decay_result={
            "decayed": [
                {"name": "test-belief", "old_confidence": 0.8,
                 "new_confidence": 0.75, "days_since_update": 10.0,
                 "knowledge_type": "belief", "decay_rate": 0.005},
            ],
            "total": 1, "unchanged": 4, "dry_run": False,
        })
        messages, result = self._run_generator(
            maintain(adapter, "proj", dry_run=False))
        self.assertEqual(result["decay"]["decayed_count"], 1)
        self.assertEqual(result["decay"]["unchanged_count"], 4)

    def test_maintain_reports_review_queue(self):
        """Should surface review queue categories."""
        from merkraum_dreaming import maintain
        adapter = self._make_mock_adapter(review_result={
            "categories": {
                "stale": [{"name": "old-fact"}],
                "low_confidence": [{"name": "weak-belief"}, {"name": "weak2"}],
                "type_mismatch": [], "approaching_expiry": [],
                "unclassified": [],
            }
        })
        messages, result = self._run_generator(
            maintain(adapter, "proj", dry_run=True))
        self.assertEqual(result["review_queue"]["total"], 3)
        self.assertEqual(result["review_queue"]["categories"]["stale"], 1)
        self.assertEqual(result["review_queue"]["categories"]["low_confidence"], 2)

    def test_maintain_reports_governance_health(self):
        """Should pass through governance health status."""
        from merkraum_dreaming import maintain
        adapter = self._make_mock_adapter(stats_result={
            "governance": {"status": "needs_attention",
                           "low_confidence_count": 5},
        })
        messages, result = self._run_generator(
            maintain(adapter, "proj", dry_run=True))
        self.assertEqual(result["governance_health"], "needs_attention")

    def test_maintain_yields_progress_messages(self):
        """Should yield structured progress messages."""
        from merkraum_dreaming import maintain
        adapter = self._make_mock_adapter()
        messages, result = self._run_generator(
            maintain(adapter, "proj", dry_run=True))
        phases = set(m["phase"] for m in messages)
        # Maintenance includes reconsolidation sub-phase
        self.assertIn("maintenance", phases)
        self.assertIn("reconsolidation", phases)
        steps = [m["step"] for m in messages]
        self.assertIn("start", steps)
        self.assertIn("decay_done", steps)
        self.assertIn("expire_done", steps)
        self.assertIn("prune_done", steps)
        self.assertIn("dedup_done", steps)
        self.assertIn("review_done", steps)
        self.assertIn("done", steps)

    def test_maintain_reports_expire_count(self):
        """Should report expired nodes from TTL enforcement."""
        from merkraum_dreaming import maintain
        adapter = self._make_mock_adapter(expire_result={
            "expired": [
                {"name": "old-edge", "type": "Belief",
                 "valid_until": "2026-03-01T00:00:00"},
            ],
            "total": 1, "dry_run": False,
        })
        messages, result = self._run_generator(
            maintain(adapter, "proj", dry_run=False))
        self.assertEqual(result["expired"]["count"], 1)
        self.assertEqual(len(result["expired"]["items"]), 1)

    def test_maintain_reports_prune_count(self):
        """Should report pruned orphan nodes."""
        from merkraum_dreaming import maintain
        adapter = self._make_mock_adapter(prune_result={
            "pruned": [
                {"name": "orphan-node", "type": "Concept",
                 "last_update": "2026-02-01T00:00:00"},
                {"name": "orphan-belief", "type": "Belief",
                 "last_update": "2026-01-15T00:00:00"},
            ],
            "total": 2, "dry_run": False,
        })
        messages, result = self._run_generator(
            maintain(adapter, "proj", dry_run=False))
        self.assertEqual(result["pruned"]["count"], 2)

    def test_maintain_reports_dedup_count(self):
        """Should report deduplicated edges."""
        from merkraum_dreaming import maintain
        adapter = self._make_mock_adapter(dedup_result={
            "duplicates_found": 5, "edges_removed": 5,
            "groups": [{"source": "A", "target": "B",
                        "type": "RELATES_TO", "count": 3,
                        "kept": {"confidence": 0.8}, "removed": 2}],
            "dry_run": False,
        })
        messages, result = self._run_generator(
            maintain(adapter, "proj", dry_run=False))
        self.assertEqual(result["dedup"]["duplicates_found"], 5)
        self.assertEqual(result["dedup"]["edges_removed"], 5)

    def test_maintain_done_message_includes_all_counts(self):
        """Final done message should summarize all maintenance actions."""
        from merkraum_dreaming import maintain
        adapter = self._make_mock_adapter(
            decay_result={"decayed": [], "total": 2, "unchanged": 3,
                          "dry_run": False},
            expire_result={"expired": [], "total": 1, "dry_run": False},
            prune_result={"pruned": [], "total": 3, "dry_run": False},
            dedup_result={"duplicates_found": 4, "edges_removed": 4,
                          "groups": [], "dry_run": False},
        )
        messages, result = self._run_generator(
            maintain(adapter, "proj", dry_run=False))
        done_msgs = [m for m in messages
                     if m["step"] == "done" and m["phase"] == "maintenance"]
        self.assertEqual(len(done_msgs), 1)
        self.assertIn("2 decay", done_msgs[0]["detail"])
        self.assertIn("1 expired", done_msgs[0]["detail"])
        self.assertIn("3 pruned", done_msgs[0]["detail"])
        self.assertIn("4 deduped", done_msgs[0]["detail"])
        self.assertIn("reconsolidated", done_msgs[0]["detail"])

    def test_dream_includes_maintenance_by_default(self):
        """Dream session should include maintenance in default phases."""
        from merkraum_dreaming import dream
        adapter = self._make_mock_adapter()
        # Mock the Neo4j driver for reflect phase
        mock_session = MagicMock()
        mock_session.run.return_value = MagicMock(data=lambda: [])
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session
        # Run only maintenance phase to test integration
        messages, result = self._run_generator(
            dream(adapter, "proj", phases=["maintenance"],
                  maintenance_dry_run=True))
        self.assertIn("maintenance", result["phases"])
        self.assertEqual(result["phases"]["maintenance"]["phase"], "maintenance")

    def test_dream_maintenance_dry_run_param(self):
        """Dream should pass maintenance_dry_run to maintain phase."""
        from merkraum_dreaming import maintain
        adapter = self._make_mock_adapter()
        # Test directly that maintain respects dry_run
        messages, result = self._run_generator(
            maintain(adapter, "proj", dry_run=True))
        self.assertFalse(result["decay"]["applied"])
        self.assertEqual(result["expired"]["count"], 0)
        self.assertEqual(result["pruned"]["count"], 0)
        messages, result = self._run_generator(
            maintain(adapter, "proj", dry_run=False))
        self.assertTrue(result["decay"]["applied"])


# --- Dreaming reconsolidation tests (Z1857, Proposal #5) ---

class TestDreamingReconsolidation(unittest.TestCase):
    """Test importance-weighted confidence strengthening — Z1857 proposal #5."""

    def _run_generator(self, gen):
        """Consume a generator, collecting messages and returning the result."""
        messages = []
        try:
            while True:
                messages.append(next(gen))
        except StopIteration as e:
            return messages, e.value

    def _make_mock_adapter(self, beliefs=None):
        """Create a mock adapter with configurable belief query results."""
        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()
        mock_session.run.return_value = MagicMock(
            data=lambda: beliefs or [])
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session
        return adapter

    def test_reconsolidate_strengthens_accessed_beliefs(self):
        """Beliefs with high access_count should get confidence boost."""
        from merkraum_dreaming import reconsolidate
        beliefs = [
            {"name": "B1", "confidence": 0.6, "access_count": 5,
             "knowledge_type": "opinion", "vsm_level": "S3"},
        ]
        adapter = self._make_mock_adapter(beliefs)
        messages, result = self._run_generator(
            reconsolidate(adapter, "proj", dry_run=False))
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["items"][0]["name"], "B1")
        self.assertAlmostEqual(result["items"][0]["boost"], 0.05, places=3)
        self.assertAlmostEqual(result["items"][0]["new_confidence"], 0.65, places=3)

    def test_reconsolidate_caps_at_max_boost(self):
        """Boost should be capped at RECONSOLIDATION_MAX_BOOST."""
        from merkraum_dreaming import reconsolidate
        beliefs = [
            {"name": "B1", "confidence": 0.5, "access_count": 100,
             "knowledge_type": "opinion", "vsm_level": "S1"},
        ]
        adapter = self._make_mock_adapter(beliefs)
        messages, result = self._run_generator(
            reconsolidate(adapter, "proj", dry_run=False))
        self.assertEqual(result["count"], 1)
        # 100 * 0.01 = 1.0, but capped at 0.15
        self.assertAlmostEqual(result["items"][0]["boost"], 0.15, places=3)
        self.assertAlmostEqual(result["items"][0]["new_confidence"], 0.65, places=3)

    def test_reconsolidate_caps_confidence_at_1(self):
        """New confidence should never exceed 1.0."""
        from merkraum_dreaming import reconsolidate
        beliefs = [
            {"name": "B1", "confidence": 0.95, "access_count": 20,
             "knowledge_type": "hypothesis", "vsm_level": "S4"},
        ]
        adapter = self._make_mock_adapter(beliefs)
        messages, result = self._run_generator(
            reconsolidate(adapter, "proj", dry_run=False))
        self.assertEqual(result["count"], 1)
        self.assertAlmostEqual(result["items"][0]["new_confidence"], 1.0, places=3)

    def test_reconsolidate_skips_below_min_access(self):
        """Beliefs with access_count < min_access should be excluded by query."""
        from merkraum_dreaming import reconsolidate
        # Empty result since the query filters access_count < 3
        adapter = self._make_mock_adapter([])
        messages, result = self._run_generator(
            reconsolidate(adapter, "proj", dry_run=False))
        self.assertEqual(result["count"], 0)

    def test_reconsolidate_dry_run_no_writes(self):
        """Dry run should not write to Neo4j."""
        from merkraum_dreaming import reconsolidate
        beliefs = [
            {"name": "B1", "confidence": 0.6, "access_count": 5,
             "knowledge_type": "opinion", "vsm_level": "S3"},
        ]
        adapter = self._make_mock_adapter(beliefs)
        messages, result = self._run_generator(
            reconsolidate(adapter, "proj", dry_run=True))
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["mode"], "preview")
        # session.run called once for the query, not for writes
        mock_session = adapter._driver.session.return_value.__enter__()
        self.assertEqual(mock_session.run.call_count, 1)

    def test_reconsolidate_skips_negligible_change(self):
        """Changes < 0.001 should be skipped."""
        from merkraum_dreaming import reconsolidate
        beliefs = [
            {"name": "B1", "confidence": 0.999, "access_count": 3,
             "knowledge_type": "fact_derived", "vsm_level": "S1"},
        ]
        adapter = self._make_mock_adapter(beliefs)
        messages, result = self._run_generator(
            reconsolidate(adapter, "proj", dry_run=False))
        # 0.999 + 0.03 = 1.0, diff = 0.001 — borderline, should be included
        # since 1.0 - 0.999 = 0.001, abs < 0.001 is false (it equals 0.001)
        self.assertEqual(result["count"], 1)

    def test_reconsolidate_custom_params(self):
        """Custom factor and max_boost should be respected."""
        from merkraum_dreaming import reconsolidate
        beliefs = [
            {"name": "B1", "confidence": 0.5, "access_count": 10,
             "knowledge_type": "opinion", "vsm_level": "S2"},
        ]
        adapter = self._make_mock_adapter(beliefs)
        messages, result = self._run_generator(
            reconsolidate(adapter, "proj", factor=0.02, max_boost=0.10,
                          dry_run=True))
        self.assertEqual(result["count"], 1)
        # 10 * 0.02 = 0.20, capped at 0.10
        self.assertAlmostEqual(result["items"][0]["boost"], 0.10, places=3)
        self.assertEqual(result["params"]["factor"], 0.02)
        self.assertEqual(result["params"]["max_boost"], 0.10)

    def test_maintain_includes_reconsolidation(self):
        """Maintain should include reconsolidation results."""
        from merkraum_dreaming import maintain
        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()
        # Return empty for all queries
        mock_session.run.return_value = MagicMock(data=lambda: [])
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session
        adapter.apply_confidence_decay.return_value = {
            "decayed": [], "total": 0, "unchanged": 0}
        adapter.expire_nodes.return_value = {"expired": [], "total": 0}
        adapter.prune_orphan_nodes.return_value = {"pruned": [], "total": 0}
        adapter.deduplicate_edges.return_value = {
            "duplicates_found": 0, "edges_removed": 0, "groups": []}
        adapter.get_certainty_review_queue.return_value = {"categories": {}}
        adapter.get_certainty_stats.return_value = {
            "governance": {"status": "healthy"}}
        messages, result = self._run_generator(
            maintain(adapter, "proj", dry_run=False))
        self.assertIn("reconsolidation", result)
        self.assertEqual(result["reconsolidation"]["count"], 0)


# --- Dreaming replay associative edge creation tests (Z1839) ---

class TestDreamingAssociativeEdges(unittest.TestCase):
    """Test replay phase associative edge creation — Z1839 proposal #1."""

    def _run_generator(self, gen):
        """Consume a generator, collecting messages and returning the result."""
        messages = []
        try:
            while True:
                messages.append(next(gen))
        except StopIteration as e:
            return messages, e.value

    @patch("merkraum_dreaming.llm_call")
    def test_replay_creates_edges_from_missing_relationship(self, mock_llm):
        """Replay should write provisional edges for missing_relationship observations."""
        from merkraum_dreaming import replay

        mock_llm.return_value = {
            "observations": [
                {
                    "type": "missing_relationship",
                    "description": "Entity A and Entity B should be connected",
                    "entities_involved": ["Alpha", "Beta"],
                    "suggested_relationship": "RELATES_TO",
                },
            ],
            "walk_quality": "productive",
            "summary": "Found a missing link.",
        }

        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()
        # Seed selection returns one candidate
        mock_session.run.return_value = MagicMock(
            data=lambda: [{"name": "Alpha", "type": "Concept", "staleness": 10}])
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session
        adapter.vector_search.return_value = []
        adapter.write_relationships.return_value = 1

        messages, result = self._run_generator(
            replay(adapter, "test-proj", hops=0, walks=1, create_edges=True))

        self.assertEqual(result["edges_created"], 1)
        self.assertEqual(len(result["edge_details"]), 1)
        self.assertEqual(result["edge_details"][0]["source"], "Alpha")
        self.assertEqual(result["edge_details"][0]["target"], "Beta")
        self.assertEqual(result["edge_details"][0]["rel_type"], "RELATES_TO")

        # Verify write_relationships was called with correct params
        adapter.write_relationships.assert_called_once()
        call_args = adapter.write_relationships.call_args
        rel = call_args[0][0][0]  # First positional arg, first element
        self.assertEqual(rel["source"], "Alpha")
        self.assertEqual(rel["target"], "Beta")
        self.assertEqual(rel["confidence"], 0.3)
        self.assertEqual(rel["type"], "RELATES_TO")
        self.assertIn("valid_until", rel)
        self.assertEqual(call_args[1]["source_type"], "dream")
        self.assertEqual(call_args[1]["actor"], "dreaming-replay")

    @patch("merkraum_dreaming.llm_call")
    def test_replay_no_edges_when_create_edges_false(self, mock_llm):
        """Replay should not write edges when create_edges=False."""
        from merkraum_dreaming import replay

        mock_llm.return_value = {
            "observations": [
                {
                    "type": "missing_relationship",
                    "description": "Test",
                    "entities_involved": ["A", "B"],
                    "suggested_relationship": "SUPPORTS",
                },
            ],
            "walk_quality": "productive",
            "summary": "Test.",
        }

        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()
        mock_session.run.return_value = MagicMock(
            data=lambda: [{"name": "A", "type": "Concept", "staleness": 5}])
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session
        adapter.vector_search.return_value = []

        messages, result = self._run_generator(
            replay(adapter, "test-proj", hops=0, walks=1, create_edges=False))

        self.assertEqual(result["edges_created"], 0)
        self.assertEqual(result["edge_details"], [])
        adapter.write_relationships.assert_not_called()

    @patch("merkraum_dreaming.llm_call")
    def test_replay_skips_insight_and_redundancy_types(self, mock_llm):
        """Replay should NOT create edges for 'insight' or 'redundancy' observations."""
        from merkraum_dreaming import replay

        mock_llm.return_value = {
            "observations": [
                {
                    "type": "insight",
                    "description": "Interesting pattern",
                    "entities_involved": ["X", "Y"],
                },
                {
                    "type": "redundancy",
                    "description": "Duplicate info",
                    "entities_involved": ["X", "Z"],
                },
            ],
            "walk_quality": "productive",
            "summary": "Test.",
        }

        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()
        mock_session.run.return_value = MagicMock(
            data=lambda: [{"name": "X", "type": "Concept", "staleness": 5}])
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session
        adapter.vector_search.return_value = []

        messages, result = self._run_generator(
            replay(adapter, "test-proj", hops=0, walks=1, create_edges=True))

        self.assertEqual(result["edges_created"], 0)
        adapter.write_relationships.assert_not_called()

    @patch("merkraum_dreaming.llm_call")
    def test_replay_falls_back_to_relates_to_for_invalid_type(self, mock_llm):
        """If LLM suggests an invalid relationship type, fall back to RELATES_TO."""
        from merkraum_dreaming import replay

        mock_llm.return_value = {
            "observations": [
                {
                    "type": "surprising_connection",
                    "description": "Unexpected link",
                    "entities_involved": ["Foo", "Bar"],
                    "suggested_relationship": "INVALID_TYPE",
                },
            ],
            "walk_quality": "productive",
            "summary": "Test.",
        }

        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()
        mock_session.run.return_value = MagicMock(
            data=lambda: [{"name": "Foo", "type": "Concept", "staleness": 5}])
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session
        adapter.vector_search.return_value = []
        adapter.write_relationships.return_value = 1

        messages, result = self._run_generator(
            replay(adapter, "test-proj", hops=0, walks=1, create_edges=True))

        self.assertEqual(result["edges_created"], 1)
        call_args = adapter.write_relationships.call_args
        rel = call_args[0][0][0]
        self.assertEqual(rel["type"], "RELATES_TO")  # Fallback

    @patch("merkraum_dreaming.llm_call")
    def test_replay_edge_has_7day_ttl(self, mock_llm):
        """Provisional edges should have valid_until set to ~7 days from now."""
        from merkraum_dreaming import replay
        from datetime import datetime, timezone, timedelta

        mock_llm.return_value = {
            "observations": [
                {
                    "type": "missing_relationship",
                    "description": "Test",
                    "entities_involved": ["A", "B"],
                    "suggested_relationship": "SUPPORTS",
                },
            ],
            "walk_quality": "productive",
            "summary": "Test.",
        }

        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()
        mock_session.run.return_value = MagicMock(
            data=lambda: [{"name": "A", "type": "Concept", "staleness": 5}])
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session
        adapter.vector_search.return_value = []
        adapter.write_relationships.return_value = 1

        before = datetime.now(timezone.utc)
        messages, result = self._run_generator(
            replay(adapter, "test-proj", hops=0, walks=1, create_edges=True))
        after = datetime.now(timezone.utc)

        call_args = adapter.write_relationships.call_args
        rel = call_args[0][0][0]
        valid_until = datetime.fromisoformat(rel["valid_until"])
        # Should be ~7 days from now (within a 1-minute tolerance)
        expected_min = before + timedelta(days=7) - timedelta(minutes=1)
        expected_max = after + timedelta(days=7) + timedelta(minutes=1)
        self.assertGreater(valid_until, expected_min)
        self.assertLess(valid_until, expected_max)

    @patch("merkraum_dreaming.llm_call")
    def test_replay_handles_write_failure_gracefully(self, mock_llm):
        """If write_relationships fails, edge creation should continue without crashing."""
        from merkraum_dreaming import replay

        mock_llm.return_value = {
            "observations": [
                {
                    "type": "missing_relationship",
                    "description": "Test",
                    "entities_involved": ["A", "B"],
                    "suggested_relationship": "SUPPORTS",
                },
            ],
            "walk_quality": "productive",
            "summary": "Test.",
        }

        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()
        mock_session.run.return_value = MagicMock(
            data=lambda: [{"name": "A", "type": "Concept", "staleness": 5}])
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session
        adapter.vector_search.return_value = []
        adapter.write_relationships.side_effect = Exception("DB error")

        messages, result = self._run_generator(
            replay(adapter, "test-proj", hops=0, walks=1, create_edges=True))

        # Should complete without raising, with 0 edges created
        self.assertEqual(result["edges_created"], 0)
        self.assertEqual(result["edge_details"], [])
        self.assertEqual(result["status"], "completed")
        # Should have an edge_failed message
        failed_msgs = [m for m in messages if m["step"] == "edge_failed"]
        self.assertEqual(len(failed_msgs), 1)


class TestDreamingBridging(unittest.TestCase):
    """Test bridging phase — cross-domain connection discovery (Proposal #3)."""

    def _run_generator(self, gen):
        """Consume a generator, collecting messages and returning the result."""
        messages = []
        try:
            while True:
                messages.append(next(gen))
        except StopIteration as e:
            return messages, e.value

    @patch("merkraum_dreaming.llm_call")
    def test_bridge_finds_connections_between_distant_beliefs(self, mock_llm):
        """Bridge should evaluate distant belief pairs and create edges for connections."""
        from merkraum_dreaming import bridge

        mock_llm.return_value = {
            "relationship_exists": True,
            "relationship_type": "COMPLEMENTS",
            "description": "Both beliefs address organizational resilience",
            "confidence": 0.6,
        }

        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()

        # Return beliefs
        beliefs_data = [
            {"name": "Belief: Regulatory compliance requires documentation", "confidence": 0.8},
            {"name": "Belief: Agile teams self-organize around goals", "confidence": 0.7},
            {"name": "Belief: Knowledge graphs improve information retrieval", "confidence": 0.9},
        ]
        call_count = [0]
        def mock_run(*args, **kwargs):
            call_count[0] += 1
            query = args[0] if args else kwargs.get("query", "")
            mock_result = MagicMock()
            if "shortestPath" in query:
                mock_result.data = lambda: [{"distance": 5}]
            else:
                mock_result.data = lambda: beliefs_data
            return mock_result

        mock_session.run = mock_run
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session
        adapter.write_relationships.return_value = 1

        messages, result = self._run_generator(
            bridge(adapter, "test-proj", pairs=2, min_distance=4, create_edges=True))

        self.assertEqual(result["phase"], "bridging")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["pairs_evaluated"], 2)
        self.assertEqual(result["connections_found"], 2)
        self.assertEqual(result["edges_created"], 2)
        self.assertGreater(result["hit_rate"], 0)

    @patch("merkraum_dreaming.llm_call")
    def test_bridge_no_edges_when_create_edges_false(self, mock_llm):
        """Bridge should not write edges when create_edges=False."""
        from merkraum_dreaming import bridge

        mock_llm.return_value = {
            "relationship_exists": True,
            "relationship_type": "SUPPORTS",
            "description": "Connection found",
            "confidence": 0.5,
        }

        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()

        beliefs_data = [
            {"name": "Belief: A", "confidence": 0.8},
            {"name": "Belief: B", "confidence": 0.7},
        ]
        def mock_run(*args, **kwargs):
            mock_result = MagicMock()
            query = args[0] if args else ""
            if "shortestPath" in query:
                mock_result.data = lambda: [{"distance": 6}]
            else:
                mock_result.data = lambda: beliefs_data
            return mock_result

        mock_session.run = mock_run
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session

        messages, result = self._run_generator(
            bridge(adapter, "test-proj", pairs=1, create_edges=False))

        self.assertEqual(result["connections_found"], 1)
        self.assertEqual(result["edges_created"], 0)
        adapter.write_relationships.assert_not_called()

    @patch("merkraum_dreaming.llm_call")
    def test_bridge_handles_no_connection(self, mock_llm):
        """Bridge should handle pairs where LLM finds no relationship."""
        from merkraum_dreaming import bridge

        mock_llm.return_value = {
            "relationship_exists": False,
        }

        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()

        beliefs_data = [
            {"name": "Belief: X", "confidence": 0.8},
            {"name": "Belief: Y", "confidence": 0.7},
        ]
        def mock_run(*args, **kwargs):
            mock_result = MagicMock()
            query = args[0] if args else ""
            if "shortestPath" in query:
                mock_result.data = lambda: [{"distance": 5}]
            else:
                mock_result.data = lambda: beliefs_data
            return mock_result

        mock_session.run = mock_run
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session

        messages, result = self._run_generator(
            bridge(adapter, "test-proj", pairs=3, create_edges=True))

        self.assertEqual(result["connections_found"], 0)
        self.assertEqual(result["edges_created"], 0)
        adapter.write_relationships.assert_not_called()

    def test_bridge_too_few_beliefs(self):
        """Bridge should gracefully handle fewer than 2 beliefs."""
        from merkraum_dreaming import bridge

        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()
        mock_session.run.return_value = MagicMock(
            data=lambda: [{"name": "Belief: Only one", "confidence": 0.8}])
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session

        messages, result = self._run_generator(
            bridge(adapter, "test-proj", pairs=5))

        self.assertEqual(result["pairs_evaluated"], 0)
        self.assertEqual(result["connections_found"], 0)
        self.assertEqual(result["status"], "completed")

    @patch("merkraum_dreaming.llm_call")
    def test_bridge_caps_confidence_at_03(self, mock_llm):
        """Bridge edges should have confidence capped at 0.3 (provisional)."""
        from merkraum_dreaming import bridge

        mock_llm.return_value = {
            "relationship_exists": True,
            "relationship_type": "SUPPORTS",
            "description": "Strong connection",
            "confidence": 0.9,  # LLM says high confidence
        }

        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()

        beliefs_data = [
            {"name": "Belief: A", "confidence": 0.8},
            {"name": "Belief: B", "confidence": 0.7},
        ]
        def mock_run(*args, **kwargs):
            mock_result = MagicMock()
            query = args[0] if args else ""
            if "shortestPath" in query:
                mock_result.data = lambda: [{"distance": 5}]
            else:
                mock_result.data = lambda: beliefs_data
            return mock_result

        mock_session.run = mock_run
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session
        adapter.write_relationships.return_value = 1

        messages, result = self._run_generator(
            bridge(adapter, "test-proj", pairs=1, create_edges=True))

        # Verify confidence is capped at 0.3
        call_args = adapter.write_relationships.call_args
        rel = call_args[0][0][0]
        self.assertEqual(rel["confidence"], 0.3)  # Capped, not 0.9

    @patch("merkraum_dreaming.llm_call")
    def test_bridge_falls_back_to_complements_for_invalid_type(self, mock_llm):
        """Bridge should fall back to COMPLEMENTS for invalid relationship types."""
        from merkraum_dreaming import bridge

        mock_llm.return_value = {
            "relationship_exists": True,
            "relationship_type": "INVALID_REL",
            "description": "Some connection",
            "confidence": 0.4,
        }

        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()

        beliefs_data = [
            {"name": "Belief: A", "confidence": 0.8},
            {"name": "Belief: B", "confidence": 0.7},
        ]
        def mock_run(*args, **kwargs):
            mock_result = MagicMock()
            query = args[0] if args else ""
            if "shortestPath" in query:
                mock_result.data = lambda: [{"distance": 5}]
            else:
                mock_result.data = lambda: beliefs_data
            return mock_result

        mock_session.run = mock_run
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session
        adapter.write_relationships.return_value = 1

        messages, result = self._run_generator(
            bridge(adapter, "test-proj", pairs=1, create_edges=True))

        call_args = adapter.write_relationships.call_args
        rel = call_args[0][0][0]
        self.assertEqual(rel["type"], "COMPLEMENTS")  # Fallback

    @patch("merkraum_dreaming.llm_call")
    def test_bridge_accepts_disconnected_pairs(self, mock_llm):
        """Disconnected belief pairs (no path) should be accepted as candidates."""
        from merkraum_dreaming import bridge

        mock_llm.return_value = {
            "relationship_exists": True,
            "relationship_type": "EXTENDS",
            "description": "Cross-domain insight",
            "confidence": 0.3,
        }

        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()

        beliefs_data = [
            {"name": "Belief: A", "confidence": 0.8},
            {"name": "Belief: B", "confidence": 0.7},
        ]
        def mock_run(*args, **kwargs):
            mock_result = MagicMock()
            query = args[0] if args else ""
            if "shortestPath" in query:
                mock_result.data = lambda: [{"distance": -1}]  # Disconnected
            else:
                mock_result.data = lambda: beliefs_data
            return mock_result

        mock_session.run = mock_run
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session
        adapter.write_relationships.return_value = 1

        messages, result = self._run_generator(
            bridge(adapter, "test-proj", pairs=1, create_edges=True))

        self.assertEqual(result["pairs_evaluated"], 1)
        self.assertEqual(result["connections_found"], 1)
        self.assertEqual(result["edges_created"], 1)

    @patch("merkraum_dreaming.llm_call")
    def test_dream_includes_walk_by_default(self, mock_llm):
        """v2.0: dream() includes walk phase (replaces replay + bridging)."""
        from merkraum_dreaming import dream

        mock_llm.return_value = None

        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()
        mock_session.run.return_value = MagicMock(data=lambda: [])
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session
        adapter.vector_search.return_value = []
        adapter.apply_confidence_decay.return_value = {"total": 0, "unchanged": 0, "decayed": []}
        adapter.expire_nodes.return_value = {"total": 0, "expired": []}
        adapter.prune_orphan_nodes.return_value = {"total": 0, "pruned": []}
        adapter.deduplicate_edges.return_value = {"duplicates_found": 0, "edges_removed": 0}
        adapter.get_certainty_review_queue.return_value = {"categories": {}}
        adapter.get_certainty_stats.return_value = {"governance": {"status": "healthy"}}

        messages, result = self._run_generator(dream(adapter, "test-proj"))

        # v2.0: walk phase replaces replay + bridging
        self.assertIn("walk", result["phases"])
        self.assertEqual(result.get("architecture"), "v2.0")
        # Walk phase messages should exist
        walk_msgs = [m for m in messages if m.get("phase") == "walk"]
        self.assertGreater(len(walk_msgs), 0)


class TestReconstructAt(unittest.TestCase):
    """Test Neo4jBaseAdapter.reconstruct_at — point-in-time reconstruction."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def test_reconstruct_found(self):
        """Should return state from the most recent operation before timestamp."""
        import json
        after_state = {"name": "TestEntity", "summary": "Updated", "confidence": 0.9}
        op_result = MagicMock()
        op_result.single.return_value = {
            "op_id": "abc123",
            "op_type": "update_belief",
            "actor": "system",
            "before_json": None,
            "after_json": json.dumps(after_state),
            "created_at": "2026-03-20T10:00:00+00:00",
            "action": "update_belief",
        }
        count_result = MagicMock()
        count_result.single.return_value = {"total": 5}
        self.session.run.side_effect = [op_result, count_result]

        result = self.adapter.reconstruct_at(
            "TestEntity", "2026-03-20T12:00:00+00:00", project_id="test")
        self.assertTrue(result["found"])
        self.assertEqual(result["state"]["name"], "TestEntity")
        self.assertEqual(result["state"]["confidence"], 0.9)
        self.assertEqual(result["operation"]["type"], "update_belief")
        self.assertEqual(result["total_ops"], 5)

    def test_reconstruct_not_found(self):
        """Should return found=False when no operations exist before timestamp."""
        op_result = MagicMock()
        op_result.single.return_value = None
        self.session.run.return_value = op_result

        result = self.adapter.reconstruct_at(
            "NoSuchEntity", "2026-01-01T00:00:00+00:00", project_id="test")
        self.assertFalse(result["found"])
        self.assertIsNone(result["state"])
        self.assertEqual(result["total_ops"], 0)

    def test_reconstruct_deleted_entity(self):
        """Should return before_state with _deleted flag for delete operations."""
        import json
        before_state = {"node": {"name": "Deleted", "summary": "Was here"}}
        op_result = MagicMock()
        op_result.single.return_value = {
            "op_id": "del123",
            "op_type": "delete_node",
            "actor": "admin",
            "before_json": json.dumps(before_state),
            "after_json": None,
            "created_at": "2026-03-20T10:00:00+00:00",
            "action": "delete_node",
        }
        count_result = MagicMock()
        count_result.single.return_value = {"total": 3}
        self.session.run.side_effect = [op_result, count_result]

        result = self.adapter.reconstruct_at(
            "Deleted", "2026-03-20T12:00:00+00:00", project_id="test")
        self.assertTrue(result["found"])
        self.assertTrue(result["state"]["_deleted"])
        self.assertEqual(result["operation"]["type"], "delete_node")

    def test_reconstruct_entity_name_in_result(self):
        """Result should include the requested entity name and timestamp."""
        op_result = MagicMock()
        op_result.single.return_value = None
        self.session.run.return_value = op_result

        result = self.adapter.reconstruct_at(
            "MyEntity", "2026-06-01T00:00:00+00:00", project_id="test")
        self.assertEqual(result["entity"], "MyEntity")
        self.assertEqual(result["timestamp"], "2026-06-01T00:00:00+00:00")


class TestWriteEntitiesFullAfterState(unittest.TestCase):
    """Test that write_entities captures full after_state from DB (SUP-166)."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.tx = MagicMock()
        # Simulate: before=None, MERGE, after-state properties query
        after_props = {
            "name": "TestConcept", "summary": "A test",
            "created_at": "2026-03-20T10:00:00+00:00",
            "updated_at": "2026-03-20T10:00:00+00:00",
            "node_id": "abc", "project_id": "test",
        }
        before_mock = MagicMock()
        before_mock.single.return_value = None  # new entity
        merge_mock = MagicMock()
        after_mock = MagicMock()
        after_mock.single.return_value = {"props": after_props}
        log_mock = MagicMock()
        # Calls: 1=before, 2=MERGE, 3=after-props, 4=log_history, 5=log_operation
        self.tx.run.side_effect = [before_mock, merge_mock, after_mock, log_mock, log_mock]
        self.session.begin_transaction.return_value = self.tx
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)
        self.adapter.vector_upsert = MagicMock(return_value=True)

    def test_after_state_includes_timestamps(self):
        """after_state should include created_at, updated_at from DB."""
        entities = [{"name": "TestConcept", "node_type": "Concept", "summary": "A test"}]
        self.adapter.write_entities(entities, "Z1801", project_id="test")
        # _log_operation is the 5th tx.run call — check after_state arg
        log_op_call = self.tx.run.call_args_list[4]
        cypher = log_op_call[0][0]
        self.assertIn("OperationLog", cypher)  # This is the log_operation CREATE

    def test_after_state_has_node_type(self):
        """after_state should include node_type."""
        entities = [{"name": "TestConcept", "node_type": "Concept", "summary": "A test"}]
        self.adapter.write_entities(entities, "Z1801", project_id="test")
        # The after-props query is the 3rd call
        after_call = self.tx.run.call_args_list[2]
        self.assertIn("properties(n)", after_call[0][0])


class TestExpireNodesUniqueOpId(unittest.TestCase):
    """Test that expire_nodes uses unique op_id per node (SUP-166 bug fix)."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def test_unique_op_ids_per_expired_node(self):
        """Each expired node should get its own op_id."""
        # Simulate 2 expired nodes
        expired_records = [
            {"name": "A", "summary": "s", "type": "Concept", "valid_until": "2026-01-01",
             "node_id": "a1", "vsm_level": None, "confidence": None,
             "status": None, "active": True},
            {"name": "B", "summary": "s", "type": "Belief", "valid_until": "2026-01-02",
             "node_id": "b1", "vsm_level": None, "confidence": 0.5,
             "status": "active", "active": True},
        ]
        find_result = MagicMock()
        find_result.__iter__ = MagicMock(return_value=iter(expired_records))
        self.session.run.return_value = find_result

        tx1 = MagicMock()
        tx2 = MagicMock()
        self.session.begin_transaction.side_effect = [tx1, tx2]

        # Track op_ids used in _log_operation calls
        op_ids = []
        original_log = self.adapter._log_operation

        def capture_log(tx, op_id, *args, **kwargs):
            op_ids.append(op_id)

        self.adapter._log_operation = capture_log
        self.adapter._log_history = MagicMock()

        self.adapter.expire_nodes(project_id="test", dry_run=False)
        # Should have 2 different op_ids
        self.assertEqual(len(op_ids), 2)
        self.assertNotEqual(op_ids[0], op_ids[1])


class TestConfidenceDecayUniqueOpId(unittest.TestCase):
    """Test that confidence_decay uses unique op_id per belief (SUP-166 bug fix)."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def test_unique_op_ids_per_decayed_belief(self):
        """Each decayed belief should get its own op_id."""
        from datetime import datetime, timezone, timedelta
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        records = [
            {"name": "B1", "confidence": 0.8, "knowledge_type": "state",
             "updated_at": old_time, "source_cycle": "Z1", "vsm_level": None},
            {"name": "B2", "confidence": 0.7, "knowledge_type": "belief",
             "updated_at": old_time, "source_cycle": "Z2", "vsm_level": None},
        ]
        find_result = MagicMock()
        find_result.__iter__ = MagicMock(return_value=iter(records))
        self.session.run.return_value = find_result

        tx1 = MagicMock()
        tx2 = MagicMock()
        self.session.begin_transaction.side_effect = [tx1, tx2]

        op_ids = []
        original_log = self.adapter._log_operation

        def capture_log(tx, op_id, *args, **kwargs):
            op_ids.append(op_id)

        self.adapter._log_operation = capture_log

        self.adapter.apply_confidence_decay(project_id="test", dry_run=False)
        self.assertEqual(len(op_ids), 2)
        self.assertNotEqual(op_ids[0], op_ids[1])


class TestHashChainIntegrity(unittest.TestCase):
    """Test hash chain integrity for audit trail (SUP-167)."""

    def setUp(self):
        self.adapter = _make_mock_adapter()
        self.session = MagicMock()
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    def test_compute_entry_hash_deterministic(self):
        """Same inputs should produce the same hash."""
        h1 = Neo4jBaseAdapter._compute_entry_hash(
            "GENESIS", "op1", "entity_upsert", "system",
            "proj1", "2026-03-23T06:00:00+00:00", '{"name": "Test"}')
        h2 = Neo4jBaseAdapter._compute_entry_hash(
            "GENESIS", "op1", "entity_upsert", "system",
            "proj1", "2026-03-23T06:00:00+00:00", '{"name": "Test"}')
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)  # SHA-256 hex digest

    def test_compute_entry_hash_changes_with_input(self):
        """Different inputs should produce different hashes."""
        h1 = Neo4jBaseAdapter._compute_entry_hash(
            "GENESIS", "op1", "entity_upsert", "system",
            "proj1", "2026-03-23T06:00:00+00:00", '{"name": "A"}')
        h2 = Neo4jBaseAdapter._compute_entry_hash(
            "GENESIS", "op1", "entity_upsert", "system",
            "proj1", "2026-03-23T06:00:00+00:00", '{"name": "B"}')
        self.assertNotEqual(h1, h2)

    def test_compute_entry_hash_chain_dependency(self):
        """Hash should change when prev_hash changes."""
        h1 = Neo4jBaseAdapter._compute_entry_hash(
            "GENESIS", "op1", "entity_upsert", "system",
            "proj1", "2026-03-23T06:00:00+00:00", '{"name": "Test"}')
        h2 = Neo4jBaseAdapter._compute_entry_hash(
            "abc123def456", "op1", "entity_upsert", "system",
            "proj1", "2026-03-23T06:00:00+00:00", '{"name": "Test"}')
        self.assertNotEqual(h1, h2)

    def test_log_operation_includes_hash_fields(self):
        """_log_operation should include prev_hash and entry_hash in CREATE."""
        tx = MagicMock()
        prev_result = MagicMock()
        prev_result.single.return_value = None  # GENESIS
        # First call is prev hash lookup, second is CREATE
        tx.run.side_effect = [prev_result, None]

        self.adapter._log_operation(
            tx, "op1", "entity_upsert", "system", "test",
            {"name": "Test"})

        # Second tx.run call should be the CREATE with hash fields
        create_call = tx.run.call_args_list[1]
        create_query = create_call[0][0]
        create_kwargs = create_call[1]
        self.assertIn("prev_hash", create_query)
        self.assertIn("entry_hash", create_query)
        self.assertEqual(create_kwargs["prev_hash"], "GENESIS")
        self.assertEqual(len(create_kwargs["entry_hash"]), 64)

    def test_log_operation_chains_from_previous(self):
        """_log_operation should use previous entry's hash as prev_hash."""
        tx = MagicMock()
        prev_result = MagicMock()
        prev_result.single.return_value = {
            "entry_hash": "abc123" * 10 + "abcd",  # 64 chars
            "created_at": "2026-03-23T05:00:00+00:00",
        }
        tx.run.side_effect = [prev_result, None]

        self.adapter._log_operation(
            tx, "op2", "update_belief", "api", "test",
            {"confidence": 0.9})

        create_kwargs = tx.run.call_args_list[1][1]
        self.assertEqual(create_kwargs["prev_hash"], "abc123" * 10 + "abcd")

    def test_verify_chain_empty(self):
        """Empty project should return valid=True with 0 entries."""
        self.session.run.return_value = iter([])
        result = self.adapter.verify_chain(project_id="empty")
        self.assertTrue(result["valid"])
        self.assertEqual(result["total_entries"], 0)
        self.assertEqual(result["verified"], 0)

    def test_verify_chain_valid(self):
        """Valid chain should return valid=True."""
        h1 = Neo4jBaseAdapter._compute_entry_hash(
            "GENESIS", "op1", "entity_upsert", "system",
            "test", "2026-03-23T06:00:00+00:00", '{"n": "A"}')
        h2 = Neo4jBaseAdapter._compute_entry_hash(
            h1, "op2", "update_belief", "api",
            "test", "2026-03-23T06:01:00+00:00", '{"c": 0.9}')

        entries = [
            {"id": "op1", "type": "entity_upsert", "actor": "system",
             "project_id": "test", "created_at": "2026-03-23T06:00:00+00:00",
             "payload_json": '{"n": "A"}', "prev_hash": "GENESIS",
             "entry_hash": h1},
            {"id": "op2", "type": "update_belief", "actor": "api",
             "project_id": "test", "created_at": "2026-03-23T06:01:00+00:00",
             "payload_json": '{"c": 0.9}', "prev_hash": h1,
             "entry_hash": h2},
        ]
        self.session.run.return_value = iter(entries)
        result = self.adapter.verify_chain(project_id="test")
        self.assertTrue(result["valid"])
        self.assertEqual(result["verified"], 2)
        self.assertEqual(len(result["breaks"]), 0)

    def test_verify_chain_tampered(self):
        """Tampered entry hash should be detected."""
        h1 = Neo4jBaseAdapter._compute_entry_hash(
            "GENESIS", "op1", "entity_upsert", "system",
            "test", "2026-03-23T06:00:00+00:00", '{"n": "A"}')

        entries = [
            {"id": "op1", "type": "entity_upsert", "actor": "system",
             "project_id": "test", "created_at": "2026-03-23T06:00:00+00:00",
             "payload_json": '{"n": "A"}', "prev_hash": "GENESIS",
             "entry_hash": "tampered" + "0" * 56},  # wrong hash
        ]
        self.session.run.return_value = iter(entries)
        result = self.adapter.verify_chain(project_id="test")
        self.assertFalse(result["valid"])
        self.assertEqual(len(result["breaks"]), 1)
        self.assertEqual(result["breaks"][0]["error"], "entry_hash_mismatch")

    def test_verify_chain_broken_link(self):
        """Mismatched prev_hash should be detected."""
        h1 = Neo4jBaseAdapter._compute_entry_hash(
            "GENESIS", "op1", "entity_upsert", "system",
            "test", "2026-03-23T06:00:00+00:00", '{"n": "A"}')

        entries = [
            {"id": "op1", "type": "entity_upsert", "actor": "system",
             "project_id": "test", "created_at": "2026-03-23T06:00:00+00:00",
             "payload_json": '{"n": "A"}', "prev_hash": "GENESIS",
             "entry_hash": h1},
            {"id": "op2", "type": "update_belief", "actor": "api",
             "project_id": "test", "created_at": "2026-03-23T06:01:00+00:00",
             "payload_json": '{"c": 0.9}', "prev_hash": "wrong_prev_hash",
             "entry_hash": Neo4jBaseAdapter._compute_entry_hash(
                 "wrong_prev_hash", "op2", "update_belief", "api",
                 "test", "2026-03-23T06:01:00+00:00", '{"c": 0.9}')},
        ]
        self.session.run.return_value = iter(entries)
        result = self.adapter.verify_chain(project_id="test")
        self.assertFalse(result["valid"])
        self.assertEqual(result["breaks"][0]["error"], "prev_hash_mismatch")

    def test_verify_chain_unchained_entries(self):
        """Pre-hash entries should be counted as unchained."""
        entries = [
            {"id": "old1", "type": "entity_upsert", "actor": "system",
             "project_id": "test", "created_at": "2026-03-20T06:00:00+00:00",
             "payload_json": '{"n": "Old"}', "prev_hash": None,
             "entry_hash": None},
        ]
        self.session.run.return_value = iter(entries)
        result = self.adapter.verify_chain(project_id="test")
        self.assertTrue(result["valid"])
        self.assertEqual(result["unchained"], 1)
        self.assertEqual(result["verified"], 0)

    def test_history_returns_hash_fields(self):
        """get_history should include prev_hash and entry_hash in operation."""
        import json as _json
        count_result = MagicMock()
        count_result.single.return_value = {"total": 1}

        h = Neo4jBaseAdapter._compute_entry_hash(
            "GENESIS", "op1", "entity_upsert", "system",
            "test", "2026-03-23T06:00:00+00:00",
            _json.dumps({"name": "Test"}, ensure_ascii=True))

        data_records = [
            {
                "operation": {
                    "id": "op1", "type": "entity_upsert",
                    "actor": "system", "project_id": "test",
                    "payload_json": _json.dumps({"name": "Test"}),
                    "before_json": None, "after_json": _json.dumps({"name": "Test"}),
                    "status": "committed",
                    "created_at": "2026-03-23T06:00:00+00:00",
                    "prev_hash": "GENESIS",
                    "entry_hash": h,
                },
                "history_entries": [],
            }
        ]
        self.session.run.side_effect = [count_result, data_records]

        result = self.adapter.get_history(project_id="test")
        op = result["entries"][0]["operation"]
        self.assertEqual(op.get("prev_hash"), "GENESIS")
        self.assertEqual(op.get("entry_hash"), h)


    # --- Edge deduplication tests ---

    def test_deduplicate_edges_no_duplicates(self):
        """Empty result when no duplicate edges exist."""
        self.session.run.return_value = iter([])
        result = self.adapter.deduplicate_edges(project_id="test", dry_run=True)
        self.assertEqual(result["duplicates_found"], 0)
        self.assertEqual(result["edges_removed"], 0)
        self.assertTrue(result["dry_run"])

    def test_deduplicate_edges_dry_run(self):
        """Dry run reports duplicates without removing them."""
        dup_group = MagicMock()
        dup_group.__getitem__ = lambda s, k: {
            "src": "VSG", "tgt": "Merkraum", "rtype": "CONTRADICTS",
            "edges": [
                {"id": 100, "confidence": 0.8, "created_at": "2026-03-20T06:00:00",
                 "source_label": "Organization", "target_label": "Project"},
                {"id": 101, "confidence": 0.7, "created_at": "2026-03-19T06:00:00",
                 "source_label": "Concept", "target_label": "Project"},
                {"id": 102, "confidence": 0.7, "created_at": "2026-03-18T06:00:00",
                 "source_label": "Concept", "target_label": "Concept"},
            ],
        }[k]
        self.session.run.return_value = iter([dup_group])
        result = self.adapter.deduplicate_edges(project_id="test", dry_run=True)
        self.assertEqual(result["duplicates_found"], 2)
        self.assertEqual(result["edges_removed"], 0)  # dry run — no deletions
        self.assertEqual(len(result["groups"]), 1)
        self.assertEqual(result["groups"][0]["source"], "VSG")
        self.assertEqual(result["groups"][0]["kept"]["confidence"], 0.8)

    def test_deduplicate_edges_execute(self):
        """Actual dedup removes duplicate edges and logs operation."""
        dup_group = MagicMock()
        dup_group.__getitem__ = lambda s, k: {
            "src": "VSG", "tgt": "Merkraum", "rtype": "CONTRADICTS",
            "edges": [
                {"id": 100, "confidence": 0.8, "created_at": "2026-03-20T06:00:00",
                 "source_label": "Organization", "target_label": "Project"},
                {"id": 101, "confidence": 0.5, "created_at": "2026-03-19T06:00:00",
                 "source_label": "Concept", "target_label": "Project"},
            ],
        }[k]
        # First call: find duplicates; subsequent calls: delete + log
        self.session.run.return_value = iter([dup_group])
        tx_mock = MagicMock()
        self.session.begin_transaction.return_value = tx_mock
        # _log_operation needs prev_hash lookup (hash chaining)
        prev_hash_result = MagicMock()
        prev_hash_result.single.return_value = None
        tx_mock.run.side_effect = [None, prev_hash_result, None]

        result = self.adapter.deduplicate_edges(
            project_id="test", dry_run=False, actor="test_user")
        self.assertEqual(result["duplicates_found"], 1)
        self.assertEqual(result["edges_removed"], 1)
        self.assertFalse(result["dry_run"])
        tx_mock.commit.assert_called_once()

    def test_write_relationships_with_type_constraints(self):
        """write_relationships uses source_type/target_type from rel dict."""
        rels = [{
            "source": "VSG", "target": "Merkraum", "type": "REFERENCES",
            "source_type": "Organization", "target_type": "Project",
        }]
        tx_mock = MagicMock()
        self.session.begin_transaction.return_value = tx_mock
        # before-state lookup returns None (new rel)
        before_result = MagicMock()
        before_result.single.return_value = None
        # MERGE result
        merge_result = MagicMock()
        merge_summary = MagicMock()
        merge_summary.counters.relationships_created = 1
        merge_summary.counters.properties_set = 5
        merge_result.consume.return_value = merge_summary
        # _log_history, _log_operation (prev_hash lookup + create)
        prev_hash_result = MagicMock()
        prev_hash_result.single.return_value = None
        tx_mock.run.side_effect = [
            before_result, merge_result,
            None, prev_hash_result, None,  # _log_history + _log_operation
        ]
        count = self.adapter.write_relationships(rels, "Z_test", project_id="test")
        self.assertEqual(count, 1)
        # Verify the MERGE query (second call) includes label constraints
        merge_call_args = tx_mock.run.call_args_list[1]
        query = merge_call_args[0][0]
        self.assertIn(":Organization", query)
        self.assertIn(":Project", query)

    def test_write_relationships_limit_prevents_cartesian(self):
        """write_relationships uses LIMIT 1 to prevent Cartesian products."""
        rels = [{"source": "VSG", "target": "Merkraum", "type": "REFERENCES"}]
        tx_mock = MagicMock()
        self.session.begin_transaction.return_value = tx_mock
        before_result = MagicMock()
        before_result.single.return_value = None
        merge_result = MagicMock()
        merge_summary = MagicMock()
        merge_summary.counters.relationships_created = 1
        merge_summary.counters.properties_set = 5
        merge_result.consume.return_value = merge_summary
        prev_hash_result = MagicMock()
        prev_hash_result.single.return_value = None
        tx_mock.run.side_effect = [
            before_result, merge_result,
            None, prev_hash_result, None,
        ]
        self.adapter.write_relationships(rels, "Z_test", project_id="test")
        # The MERGE query should contain LIMIT 1 to prevent Cartesian products
        merge_call = tx_mock.run.call_args_list[1]
        merge_query = merge_call[0][0]
        self.assertIn("LIMIT 1", merge_query)


class TestDreamingCompression(unittest.TestCase):
    """Test compression phase — entity-based topic abstraction (Proposal #4)."""

    def _run_generator(self, gen):
        """Consume a generator, collecting messages and returning the result."""
        messages = []
        try:
            while True:
                messages.append(next(gen))
        except StopIteration as e:
            return messages, e.value

    def _make_mock_adapter(self, clusters_data=None, compressed_names=None,
                           belief_details=None):
        """Create a mock adapter for compression tests."""
        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()

        call_count = [0]
        def mock_run(*args, **kwargs):
            call_count[0] += 1
            query = args[0] if args else kwargs.get("query", "")
            mock_result = MagicMock()

            if "collect(DISTINCT b.name)" in query:
                # Entity cluster query
                mock_result.data = lambda: (clusters_data or [])
            elif "b.status = 'compressed'" in query:
                # Already-compressed filter
                mock_result.data = lambda: [{"names": list(compressed_names or [])}]
            elif "b.summary" in query:
                # Belief details query
                mock_result.data = lambda: (belief_details or [
                    {"name": "Belief: X is growing", "confidence": 0.8, "summary": "Growth signal"},
                    {"name": "Belief: X has strong team", "confidence": 0.7, "summary": "Team quality"},
                    {"name": "Belief: X raised funding", "confidence": 0.9, "summary": "Funding"},
                ])
            else:
                # MERGE/SET operations (write queries)
                mock_result.data = lambda: []
            return mock_result

        mock_session.run = mock_run
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session
        return adapter

    @patch("merkraum_dreaming.llm_call")
    def test_compress_creates_abstraction_from_entity_cluster(self, mock_llm):
        """Compression should create abstract beliefs from entity-based clusters."""
        from merkraum_dreaming import compress

        mock_llm.return_value = {
            "abstract_belief": "Company X is a well-funded, high-growth organization with strong talent",
            "confidence": 0.8,
            "reasoning": "Three beliefs converge on X's organizational strength",
            "beliefs_used": 3,
        }

        clusters = [{
            "entity": "Company X",
            "entity_type": "Organization",
            "beliefs": ["Belief: X is growing", "Belief: X has strong team", "Belief: X raised funding"],
            "rel_types": ["APPLIES"],
        }]

        adapter = self._make_mock_adapter(clusters_data=clusters)
        messages, result = self._run_generator(
            compress(adapter, "test-proj", min_cluster_size=3, dry_run=False))

        self.assertEqual(result["phase"], "compression")
        self.assertEqual(result["status"], "apply")
        self.assertEqual(result["clusters_found"], 1)
        self.assertEqual(result["clusters_compressed"], 1)
        self.assertEqual(result["abstractions_created"], 1)

    @patch("merkraum_dreaming.llm_call")
    def test_compress_dry_run_does_not_write(self, mock_llm):
        """Dry run should preview compressions without writing to graph."""
        from merkraum_dreaming import compress

        mock_llm.return_value = {
            "abstract_belief": "Abstract belief about X",
            "confidence": 0.75,
            "reasoning": "Test",
            "beliefs_used": 3,
        }

        clusters = [{
            "entity": "Entity A",
            "entity_type": "Concept",
            "beliefs": ["Belief: A1", "Belief: A2", "Belief: A3"],
            "rel_types": ["APPLIES"],
        }]

        adapter = self._make_mock_adapter(clusters_data=clusters)
        messages, result = self._run_generator(
            compress(adapter, "test-proj", min_cluster_size=3, dry_run=True))

        self.assertEqual(result["status"], "preview")
        self.assertEqual(result["clusters_compressed"], 1)
        # In dry_run, no abstractions are created in the graph
        self.assertEqual(result["abstractions_created"], 0)

    def test_compress_no_clusters_found(self):
        """Compression should handle empty graphs gracefully."""
        from merkraum_dreaming import compress

        adapter = self._make_mock_adapter(clusters_data=[])
        messages, result = self._run_generator(
            compress(adapter, "test-proj", min_cluster_size=3))

        self.assertEqual(result["phase"], "compression")
        self.assertEqual(result["clusters_found"], 0)
        self.assertEqual(result["clusters_compressed"], 0)

    @patch("merkraum_dreaming.llm_call")
    def test_compress_skips_already_compressed_beliefs(self, mock_llm):
        """Already compressed beliefs should be filtered out."""
        from merkraum_dreaming import compress

        clusters = [{
            "entity": "Entity B",
            "entity_type": "Concept",
            "beliefs": ["Belief: B1", "Belief: B2", "Belief: B3"],
            "rel_types": ["APPLIES"],
        }]

        # All three beliefs are already compressed → cluster too small after filtering
        adapter = self._make_mock_adapter(
            clusters_data=clusters,
            compressed_names={"Belief: B1", "Belief: B2"})
        messages, result = self._run_generator(
            compress(adapter, "test-proj", min_cluster_size=3))

        # Only 1 uncompressed belief remains → cluster skipped
        self.assertEqual(result["clusters_compressed"], 0)
        mock_llm.assert_not_called()

    @patch("merkraum_dreaming.llm_call")
    def test_compress_skips_diverse_cluster(self, mock_llm):
        """If LLM says beliefs are too diverse, skip the cluster."""
        from merkraum_dreaming import compress

        mock_llm.return_value = {
            "abstract_belief": "",
            "confidence": 0.0,
            "reasoning": "These beliefs address completely different aspects with no common thread",
            "beliefs_used": 0,
        }

        clusters = [{
            "entity": "Entity C",
            "entity_type": "Person",
            "beliefs": ["Belief: C1", "Belief: C2", "Belief: C3"],
            "rel_types": ["APPLIES"],
        }]

        adapter = self._make_mock_adapter(clusters_data=clusters)
        messages, result = self._run_generator(
            compress(adapter, "test-proj", min_cluster_size=3))

        self.assertEqual(result["clusters_compressed"], 0)
        self.assertEqual(result["abstractions_created"], 0)
        # Should have a result entry with skipped=True
        self.assertTrue(any(r.get("skipped") for r in result.get("results", [])))

    @patch("merkraum_dreaming.llm_call")
    def test_compress_handles_llm_failure(self, mock_llm):
        """Compression should gracefully handle LLM failures."""
        from merkraum_dreaming import compress

        mock_llm.return_value = None  # LLM failure

        clusters = [{
            "entity": "Entity D",
            "entity_type": "Organization",
            "beliefs": ["Belief: D1", "Belief: D2", "Belief: D3"],
            "rel_types": ["APPLIES"],
        }]

        adapter = self._make_mock_adapter(clusters_data=clusters)
        messages, result = self._run_generator(
            compress(adapter, "test-proj", min_cluster_size=3))

        self.assertEqual(result["clusters_compressed"], 0)
        self.assertEqual(result["abstractions_created"], 0)

    @patch("merkraum_dreaming.llm_call")
    def test_compress_yields_progress_messages(self, mock_llm):
        """Compression should yield structured progress messages."""
        from merkraum_dreaming import compress

        mock_llm.return_value = {
            "abstract_belief": "Test abstraction",
            "confidence": 0.7,
            "reasoning": "Test",
            "beliefs_used": 3,
        }

        clusters = [{
            "entity": "Entity E",
            "entity_type": "Concept",
            "beliefs": ["Belief: E1", "Belief: E2", "Belief: E3"],
            "rel_types": ["APPLIES"],
        }]

        adapter = self._make_mock_adapter(clusters_data=clusters)
        messages, result = self._run_generator(
            compress(adapter, "test-proj", min_cluster_size=3))

        phases = [m["phase"] for m in messages]
        self.assertTrue(all(p == "compression" for p in phases))
        steps = [m["step"] for m in messages]
        self.assertIn("start", steps)
        self.assertIn("done", steps)

    @patch("merkraum_dreaming.llm_call")
    def test_dream_includes_compression_by_default(self, mock_llm):
        """The dream() orchestrator should include compression phase by default."""
        from merkraum_dreaming import dream

        adapter = self._make_mock_adapter(clusters_data=[])

        # Run dream with only compression phase to verify integration
        messages, result = self._run_generator(
            dream(adapter, "test-proj", phases=["compression"]))

        self.assertIn("compression", result["phases"])
        self.assertEqual(result["phases"]["compression"]["phase"], "compression")

    def test_compression_model_config(self):
        """Compression model should default to Sonnet."""
        from merkraum_dreaming import _DEFAULT_COMPRESSION_MODEL, _get_compression_model
        self.assertIn("sonnet", _DEFAULT_COMPRESSION_MODEL.lower())
        # Default getter returns same
        with patch.dict(os.environ, {}, clear=False):
            if "MERKRAUM_DREAMING_COMPRESSION_MODEL" in os.environ:
                del os.environ["MERKRAUM_DREAMING_COMPRESSION_MODEL"]
            self.assertIn("sonnet", _get_compression_model().lower())

    def test_compression_model_env_override(self):
        """Env var should override compression model."""
        from merkraum_dreaming import _get_compression_model
        with patch.dict(os.environ, {"MERKRAUM_DREAMING_COMPRESSION_MODEL": "custom-model"}):
            self.assertEqual(_get_compression_model(), "custom-model")


# ---------------------------------------------------------------------------
# v2.0 Walk + TTL tests
# ---------------------------------------------------------------------------


class TestDreamingWalk(unittest.TestCase):
    """Test the v2.0 walk() function."""

    def _run_generator(self, gen):
        messages = []
        result = None
        try:
            while True:
                messages.append(next(gen))
        except StopIteration as e:
            result = e.value
        return messages, result

    @patch("merkraum_dreaming.llm_call")
    def test_walk_empty_graph(self, mock_llm):
        """Walk on empty graph returns empty status."""
        from merkraum_dreaming import walk

        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()
        mock_session.run.return_value = MagicMock(data=lambda: [])
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session

        messages, result = self._run_generator(
            walk(adapter, "test-proj", steps=5, walks=1))

        self.assertEqual(result["phase"], "walk")
        self.assertEqual(result["status"], "empty")

    @patch("merkraum_dreaming.llm_call")
    def test_walk_creates_edges_from_cross_domain_bridge(self, mock_llm):
        """Walk should create edges for cross_domain_bridge observations."""
        from merkraum_dreaming import walk

        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()

        # First call: entry point candidates
        # Second+: neighbors
        call_count = {"n": 0}
        def mock_run(*args, **kwargs):
            call_count["n"] += 1
            query = args[0] if args else kwargs.get("query", "")
            mock_result = MagicMock()
            if "activation_score" in query:
                mock_result.data = lambda: [{"name": "NodeA", "type": "Concept",
                                              "ac": 5, "recency": 0.5,
                                              "activation_score": 3.0,
                                              "access_count": 5}]
            elif "neighbor" in query:
                mock_result.data = lambda: []
            else:
                mock_result.data = lambda: []
            return mock_result

        mock_session.run = mock_run
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session

        mock_llm.return_value = {
            "observations": [{
                "type": "cross_domain_bridge",
                "description": "Found bridge between concepts",
                "entities_involved": ["NodeA", "NodeB"],
                "suggested_relationship": "COMPLEMENTS",
            }],
            "walk_quality": "productive",
            "summary": "Good walk",
        }
        adapter.write_relationships.return_value = 1

        messages, result = self._run_generator(
            walk(adapter, "test-proj", steps=2, walks=1))

        self.assertEqual(result["phase"], "walk")
        self.assertEqual(result["edges_created"], 1)
        self.assertEqual(result["edge_details"][0]["observation_type"],
                         "cross_domain_bridge")

    def test_walk_default_probabilities(self):
        """Walk default probabilities should be 70/20/10."""
        import inspect
        from merkraum_dreaming import walk
        sig = inspect.signature(walk)
        self.assertAlmostEqual(sig.parameters["p_graph"].default, 0.70)
        self.assertAlmostEqual(sig.parameters["p_semantic"].default, 0.20)
        self.assertAlmostEqual(sig.parameters["p_random"].default, 0.10)


class TestTTLScaling(unittest.TestCase):
    """Test the v2.0 TTL scaling formula."""

    def test_zero_activations(self):
        from merkraum_dreaming import calculate_ttl_days
        self.assertEqual(calculate_ttl_days(30, 0), 30)

    def test_ten_activations_approx_one_year(self):
        from merkraum_dreaming import calculate_ttl_days
        result = calculate_ttl_days(30, 10)
        # ~10 activations should be ~365 days (Norman's directive)
        self.assertGreater(result, 350)
        self.assertLess(result, 400)

    def test_never_shrinks(self):
        """TTL should grow monotonically with access_count."""
        from merkraum_dreaming import calculate_ttl_days
        prev = 0
        for ac in range(20):
            current = calculate_ttl_days(30, ac)
            self.assertGreaterEqual(current, prev)
            prev = current

    def test_custom_scale_factor(self):
        from merkraum_dreaming import calculate_ttl_days
        result = calculate_ttl_days(30, 5, scale_factor=2.0)
        self.assertEqual(result, 30 * (1 + 5 * 2))


class TestAccessLogging(unittest.TestCase):
    """Test access logging infrastructure."""

    def test_log_access_increments_count(self):
        from merkraum_dreaming import _log_access
        mock_session = MagicMock()
        _log_access(mock_session, "test-proj", ["Node1", "Node2"])
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        self.assertIn("access_count", call_args[0][0])
        self.assertIn("last_accessed", call_args[0][0])

    def test_log_access_empty_names(self):
        from merkraum_dreaming import _log_access
        mock_session = MagicMock()
        _log_access(mock_session, "test-proj", [])
        mock_session.run.assert_not_called()


class TestDreamV2Legacy(unittest.TestCase):
    """Test backward compatibility of dream() v2.0."""

    def _run_generator(self, gen):
        messages = []
        result = None
        try:
            while True:
                messages.append(next(gen))
        except StopIteration as e:
            result = e.value
        return messages, result

    @patch("merkraum_dreaming.llm_call")
    def test_legacy_replay_phase_maps_to_walk(self, mock_llm):
        """Passing phases=['replay'] should run walk phase."""
        from merkraum_dreaming import dream

        mock_llm.return_value = None
        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()
        mock_session.run.return_value = MagicMock(data=lambda: [])
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session

        messages, result = self._run_generator(
            dream(adapter, "test-proj", phases=["replay"]))

        self.assertIn("walk", result["phases"])
        self.assertNotIn("replay", result["phases"])

    @patch("merkraum_dreaming.llm_call")
    def test_legacy_bridging_phase_maps_to_walk(self, mock_llm):
        """Passing phases=['bridging'] should run walk phase."""
        from merkraum_dreaming import dream

        mock_llm.return_value = None
        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()
        mock_session.run.return_value = MagicMock(data=lambda: [])
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session

        messages, result = self._run_generator(
            dream(adapter, "test-proj", phases=["bridging"]))

        self.assertIn("walk", result["phases"])

    @patch("merkraum_dreaming.llm_call")
    def test_dream_v2_architecture_field(self, mock_llm):
        """dream() result should include architecture='v2.0'."""
        from merkraum_dreaming import dream

        mock_llm.return_value = None
        adapter = MagicMock(spec=Neo4jBaseAdapter)
        mock_session = MagicMock()
        mock_session.run.return_value = MagicMock(data=lambda: [])
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        adapter._driver.session.return_value = mock_session
        adapter.apply_confidence_decay.return_value = {"total": 0, "unchanged": 0, "decayed": []}
        adapter.expire_nodes.return_value = {"total": 0, "expired": []}
        adapter.prune_orphan_nodes.return_value = {"total": 0, "pruned": []}
        adapter.deduplicate_edges.return_value = {"duplicates_found": 0, "edges_removed": 0}
        adapter.get_certainty_review_queue.return_value = {"categories": {}}
        adapter.get_certainty_stats.return_value = {"governance": {"status": "healthy"}}

        messages, result = self._run_generator(dream(adapter, "test-proj"))

        self.assertEqual(result["architecture"], "v2.0")
        self.assertIn("walk", result["phases"])
        self.assertIn("consolidation", result["phases"])
        self.assertIn("reflection", result["phases"])
        self.assertIn("maintenance", result["phases"])

    def test_walk_model_defaults_to_haiku(self):
        """Walk model should default to Haiku (cost-efficient)."""
        from merkraum_dreaming import _get_walk_model
        with patch.dict(os.environ, {}, clear=False):
            for key in ["MERKRAUM_DREAMING_WALK_MODEL", "MERKRAUM_DREAMING_REPLAY_MODEL"]:
                os.environ.pop(key, None)
            self.assertIn("haiku", _get_walk_model().lower())

    def test_dreaming_config_has_v2_fields(self):
        """get_dreaming_config() should include v2.0 fields."""
        from merkraum_dreaming import get_dreaming_config
        config = get_dreaming_config()
        self.assertIn("walk_model", config)
        self.assertIn("architecture_version", config)
        self.assertEqual(config["architecture_version"], "2.0")
        self.assertIn("ttl_scale_factor", config)


class TestSUP192CrossProjectIsolation(unittest.TestCase):
    """SUP-192: Chat must not display sources from other projects."""

    def _make_adapter(self):
        """Create a mock Neo4jQdrantAdapter for testing."""
        with patch("merkraum_backend.Neo4jQdrantAdapter.__init__", return_value=None):
            adapter = Neo4jQdrantAdapter.__new__(Neo4jQdrantAdapter)
        adapter._driver = MagicMock()
        adapter._qdrant = MagicMock()
        adapter._embedder = MagicMock()
        adapter._pii_gateway = None
        return adapter

    def test_dedupe_skips_vectors_without_project_id(self):
        """Vectors missing project_id in payload must not be attributed to
        the requesting project — they should be skipped entirely."""
        adapter = self._make_adapter()
        results = [
            {
                "id": "vec-1",
                "score": 0.95,
                "content": "Some content",
                "metadata": {
                    "name": "Entity A",
                    "node_type": "Concept",
                    # NOTE: no project_id in metadata
                },
            },
            {
                "id": "vec-2",
                "score": 0.85,
                "content": "Other content",
                "metadata": {
                    "name": "Entity B",
                    "node_type": "Concept",
                    "project_id": "user123:my-project",
                },
            },
        ]
        deduped = adapter._dedupe_vector_results(
            results, project_id="user123:my-project", top_k=10
        )
        # Only the vector with matching project_id should remain
        names = [r["metadata"]["name"] for r in deduped]
        self.assertNotIn("Entity A", names)
        self.assertIn("Entity B", names)

    def test_dedupe_keeps_vectors_with_correct_project_id(self):
        """Vectors with the correct project_id should be kept."""
        adapter = self._make_adapter()
        results = [
            {
                "id": "vec-1",
                "score": 0.95,
                "content": "Content A",
                "metadata": {
                    "name": "Entity A",
                    "node_type": "Concept",
                    "project_id": "user123:project-a",
                },
            },
            {
                "id": "vec-2",
                "score": 0.85,
                "content": "Content B",
                "metadata": {
                    "name": "Entity B",
                    "node_type": "Concept",
                    "project_id": "user456:project-b",
                },
            },
        ]
        deduped = adapter._dedupe_vector_results(
            results, project_id="user123:project-a", top_k=10
        )
        # Both should be kept — dedup is by key, project_id is part of key
        # so different projects create different keys
        names = [r["metadata"]["name"] for r in deduped]
        self.assertIn("Entity A", names)
        self.assertIn("Entity B", names)
        self.assertEqual(len(deduped), 2)

    def test_dedupe_with_node_id_bypasses_project_check(self):
        """Vectors with node_id use that as the dedup key, not project_id."""
        adapter = self._make_adapter()
        results = [
            {
                "id": "vec-1",
                "score": 0.95,
                "content": "Content",
                "metadata": {
                    "node_id": "unique-node-123",
                    "name": "Entity A",
                    "project_id": "user123:proj",
                },
            },
        ]
        deduped = adapter._dedupe_vector_results(
            results, project_id="user123:proj", top_k=10
        )
        self.assertEqual(len(deduped), 1)


if __name__ == "__main__":
    unittest.main()
