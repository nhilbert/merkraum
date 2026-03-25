#!/usr/bin/env python3
"""
Unit tests for Merkraum PII Gateway (merkraum_pii.py).

Tests the PII detection, gating logic, language detection,
and project settings extraction — all without requiring Presidio
installation (uses mocked analyzer for most tests).

Z1931 — SUP-182
"""

import unittest
from unittest.mock import MagicMock, patch

from merkraum_pii import (
    PIIDetected,
    PIIFinding,
    PII_MODES,
    DEFAULT_MODE,
    DEFAULT_LANGUAGE,
    DEFAULT_PII_ENTITIES,
    detect_language,
    scan_text,
    scan_entity,
    gate_entities,
    get_pii_settings,
)


class TestPIIFinding(unittest.TestCase):
    """Test PIIFinding data class."""

    def test_to_dict(self):
        f = PIIFinding("PERSON", "John Doe", 0.95, "name", 0, 8)
        d = f.to_dict()
        self.assertEqual(d["entity_type"], "PERSON")
        self.assertEqual(d["text"], "John Doe")
        self.assertEqual(d["score"], 0.95)
        self.assertEqual(d["field"], "name")
        self.assertEqual(d["start"], 0)
        self.assertEqual(d["end"], 8)

    def test_to_dict_truncates_long_text(self):
        long_text = "A" * 50
        f = PIIFinding("PERSON", long_text, 0.9, "summary", 0, 50)
        d = f.to_dict()
        self.assertTrue(d["text"].endswith("..."))
        self.assertEqual(len(d["text"]), 23)  # 20 + "..."

    def test_to_dict_short_text_no_truncation(self):
        f = PIIFinding("PERSON", "short", 0.9, "name", 0, 5)
        d = f.to_dict()
        self.assertEqual(d["text"], "short")


class TestPIIDetected(unittest.TestCase):
    """Test PIIDetected exception."""

    def test_exception_message(self):
        findings = [{"entity_type": "PERSON"}, {"entity_type": "EMAIL_ADDRESS"}]
        exc = PIIDetected(findings)
        self.assertIn("PERSON", str(exc))
        self.assertIn("EMAIL_ADDRESS", str(exc))
        self.assertEqual(exc.findings, findings)

    def test_exception_truncates_at_5(self):
        findings = [{"entity_type": f"TYPE_{i}"} for i in range(10)]
        exc = PIIDetected(findings)
        # Should only list first 5 in message
        self.assertIn("TYPE_0", str(exc))
        self.assertIn("TYPE_4", str(exc))
        self.assertNotIn("TYPE_5", str(exc))


class TestDetectLanguage(unittest.TestCase):
    """Test heuristic language detection."""

    def test_english_text(self):
        self.assertEqual(detect_language("This is a test of the English language detection system"), "en")

    def test_german_text(self):
        self.assertEqual(detect_language("Das ist ein Test der deutschen Spracherkennung für das System"), "de")

    def test_german_umlauts(self):
        self.assertEqual(detect_language("Die Überwachung der Qualität ist für uns wichtig"), "de")

    def test_short_text_defaults_to_english(self):
        self.assertEqual(detect_language("Hi"), "en")

    def test_empty_text_defaults_to_english(self):
        self.assertEqual(detect_language(""), "en")

    def test_none_defaults_to_english(self):
        self.assertEqual(detect_language(None), "en")


class TestScanTextMocked(unittest.TestCase):
    """Test scan_text with mocked Presidio analyzer."""

    def _make_mock_result(self, entity_type, start, end, score=0.9):
        r = MagicMock()
        r.entity_type = entity_type
        r.start = start
        r.end = end
        r.score = score
        return r

    @patch("merkraum_pii._get_analyzer")
    def test_finds_person(self, mock_get):
        analyzer = MagicMock()
        mock_get.return_value = analyzer
        analyzer.analyze.return_value = [
            self._make_mock_result("PERSON", 0, 8),
        ]
        findings = scan_text("John Doe works here", language="en", field="name")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].entity_type, "PERSON")
        self.assertEqual(findings[0].text, "John Doe")
        self.assertEqual(findings[0].field, "name")

    @patch("merkraum_pii._get_analyzer")
    def test_empty_text_returns_empty(self, mock_get):
        findings = scan_text("", language="en")
        self.assertEqual(findings, [])
        mock_get.assert_not_called()

    @patch("merkraum_pii._get_analyzer")
    def test_no_analyzer_returns_empty(self, mock_get):
        mock_get.return_value = None
        findings = scan_text("John Doe", language="en")
        self.assertEqual(findings, [])

    @patch("merkraum_pii._get_analyzer")
    def test_auto_language_detection(self, mock_get):
        analyzer = MagicMock()
        mock_get.return_value = analyzer
        analyzer.analyze.return_value = []
        scan_text("Das ist ein Test der Erkennung", language="auto")
        # Should detect German and call analyze with de
        call_kwargs = analyzer.analyze.call_args
        self.assertEqual(call_kwargs[1]["language"], "de")

    @patch("merkraum_pii._get_analyzer")
    def test_analyzer_exception_returns_empty(self, mock_get):
        analyzer = MagicMock()
        mock_get.return_value = analyzer
        analyzer.analyze.side_effect = RuntimeError("spaCy crashed")
        findings = scan_text("John Doe", language="en")
        self.assertEqual(findings, [])


class TestScanEntity(unittest.TestCase):
    """Test scan_entity scans both name and summary."""

    @patch("merkraum_pii._get_analyzer")
    def test_scans_name_and_summary(self, mock_get):
        analyzer = MagicMock()
        mock_get.return_value = analyzer

        def analyze_side_effect(text, **kwargs):
            r = MagicMock()
            r.entity_type = "PERSON"
            r.start = 0
            r.end = len(text)
            r.score = 0.9
            return [r]

        analyzer.analyze.side_effect = analyze_side_effect
        entity = {"name": "Max Müller", "summary": "Max Müller is a consultant"}
        findings = scan_entity(entity, language="de")
        # Should have findings from both name and summary
        fields = {f.field for f in findings}
        self.assertIn("name", fields)
        self.assertIn("summary", fields)

    @patch("merkraum_pii._get_analyzer")
    def test_entity_without_summary(self, mock_get):
        analyzer = MagicMock()
        mock_get.return_value = analyzer
        analyzer.analyze.return_value = []
        entity = {"name": "TestConcept"}
        findings = scan_entity(entity, language="en")
        self.assertEqual(findings, [])
        # Only one call (for name), not two
        self.assertEqual(analyzer.analyze.call_count, 1)


class TestGateEntities(unittest.TestCase):
    """Test gate_entities with different modes."""

    def _make_entities(self):
        return [
            {"name": "John Doe", "node_type": "Person", "summary": "A person"},
            {"name": "GDPR Regulation", "node_type": "Regulation", "summary": "EU data protection"},
            {"name": "Max Müller", "node_type": "Person", "summary": "German consultant"},
        ]

    @patch("merkraum_pii.scan_entity")
    def test_off_mode_skips_scanning(self, mock_scan):
        entities = self._make_entities()
        result = gate_entities(entities, mode="off")
        self.assertEqual(len(result["entities"]), 3)
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["mode"], "off")
        mock_scan.assert_not_called()

    @patch("merkraum_pii.scan_entity")
    def test_warn_mode_passes_all_entities(self, mock_scan):
        # First and third entities have PII, second is clean
        mock_scan.side_effect = [
            [PIIFinding("PERSON", "John Doe", 0.95, "name", 0, 8)],
            [],
            [PIIFinding("PERSON", "Max Müller", 0.9, "name", 0, 10)],
        ]
        entities = self._make_entities()
        result = gate_entities(entities, mode="warn")
        self.assertEqual(len(result["entities"]), 3)  # All pass through
        self.assertEqual(len(result["findings"]), 2)   # Two findings reported
        self.assertEqual(result["blocked"], [])
        self.assertEqual(result["mode"], "warn")

    @patch("merkraum_pii.scan_entity")
    def test_log_mode_passes_all_entities(self, mock_scan):
        mock_scan.side_effect = [
            [PIIFinding("PERSON", "John Doe", 0.95, "name", 0, 8)],
            [],
            [],
        ]
        entities = self._make_entities()
        result = gate_entities(entities, mode="log")
        self.assertEqual(len(result["entities"]), 3)
        self.assertEqual(len(result["findings"]), 1)
        self.assertEqual(result["mode"], "log")

    @patch("merkraum_pii.scan_entity")
    def test_block_mode_raises_on_pii(self, mock_scan):
        mock_scan.side_effect = [
            [PIIFinding("PERSON", "John Doe", 0.95, "name", 0, 8)],
            [],
            [],
        ]
        entities = self._make_entities()
        with self.assertRaises(PIIDetected) as ctx:
            gate_entities(entities, mode="block")
        self.assertEqual(len(ctx.exception.findings), 1)

    @patch("merkraum_pii.scan_entity")
    def test_block_mode_no_pii_passes(self, mock_scan):
        mock_scan.return_value = []
        entities = self._make_entities()
        result = gate_entities(entities, mode="block")
        self.assertEqual(len(result["entities"]), 3)
        self.assertEqual(result["findings"], [])

    def test_empty_entities_returns_immediately(self):
        result = gate_entities([], mode="block")
        self.assertEqual(result["entities"], [])
        self.assertEqual(result["mode"], "block")

    @patch("merkraum_pii.scan_entity")
    def test_invalid_mode_falls_back_to_default(self, mock_scan):
        mock_scan.return_value = []
        entities = [{"name": "Test", "node_type": "Concept"}]
        result = gate_entities(entities, mode="invalid_mode")
        self.assertEqual(result["mode"], DEFAULT_MODE)


class TestGetPIISettings(unittest.TestCase):
    """Test extraction of PII settings from project metadata."""

    def test_none_returns_defaults(self):
        settings = get_pii_settings(None)
        self.assertEqual(settings["mode"], DEFAULT_MODE)
        self.assertEqual(settings["language"], DEFAULT_LANGUAGE)

    def test_reads_from_project_meta(self):
        meta = {"pii_mode": "block", "pii_language": "de"}
        settings = get_pii_settings(meta)
        self.assertEqual(settings["mode"], "block")
        self.assertEqual(settings["language"], "de")

    def test_invalid_mode_falls_back(self):
        meta = {"pii_mode": "invalid"}
        settings = get_pii_settings(meta)
        self.assertEqual(settings["mode"], DEFAULT_MODE)

    def test_invalid_language_falls_back(self):
        meta = {"pii_language": "fr"}
        settings = get_pii_settings(meta)
        self.assertEqual(settings["language"], DEFAULT_LANGUAGE)

    def test_missing_keys_use_defaults(self):
        meta = {"tier": "pro"}  # No PII keys
        settings = get_pii_settings(meta)
        self.assertEqual(settings["mode"], DEFAULT_MODE)
        self.assertEqual(settings["language"], DEFAULT_LANGUAGE)

    def test_off_mode_accepted(self):
        meta = {"pii_mode": "off"}
        settings = get_pii_settings(meta)
        self.assertEqual(settings["mode"], "off")

    def test_auto_language_accepted(self):
        meta = {"pii_language": "auto"}
        settings = get_pii_settings(meta)
        self.assertEqual(settings["language"], "auto")


class TestWriteEntitiesPIIIntegration(unittest.TestCase):
    """Test PII gateway integration at write_entities level."""

    def setUp(self):
        from merkraum_backend import Neo4jQdrantAdapter
        self.adapter = Neo4jQdrantAdapter.__new__(Neo4jQdrantAdapter)
        self.adapter._driver = MagicMock()
        self.adapter._qdrant = None
        self.adapter._embedder = None
        self.adapter._neo4j_uri = "bolt://mock:7687"
        self.adapter._neo4j_user = "neo4j"
        self.adapter._neo4j_password = "test"
        self.adapter._qdrant_url = "http://mock:6333"
        self.adapter._qdrant_api_key = None
        self.adapter._embed_model_name = "BAAI/bge-small-en-v1.5"
        # Mock session/tx
        self.session = MagicMock()
        self.tx = MagicMock()
        after_result = MagicMock()
        after_result.single.return_value = {"props": {"name": "mock", "node_type": "Concept"}}
        self.tx.run.return_value = after_result
        self.session.begin_transaction.return_value = self.tx
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)
        self.adapter.vector_upsert = MagicMock(return_value=True)
        # Mock get_project to avoid DB calls
        self.adapter.get_project = MagicMock(return_value=None)

    @patch("merkraum_pii.gate_entities")
    def test_write_entities_calls_pii_gateway(self, mock_gate):
        """Verify write_entities invokes PII gateway."""
        self.adapter.get_project.return_value = {"pii_mode": "warn", "pii_language": "auto"}
        entities = [{"name": "TestConcept", "node_type": "Concept", "summary": "clean"}]
        mock_gate.return_value = {
            "entities": entities, "findings": [], "blocked": [], "mode": "warn"
        }
        count = self.adapter.write_entities(entities, "Z1931", project_id="test")
        mock_gate.assert_called_once()
        self.assertEqual(count, 1)

    @patch("merkraum_pii.gate_entities")
    def test_write_entities_block_raises(self, mock_gate):
        """Verify write_entities raises PIIDetected in block mode."""
        self.adapter.get_project.return_value = {"pii_mode": "block"}
        mock_gate.side_effect = PIIDetected([{"entity_type": "PERSON"}])
        entities = [{"name": "John Doe", "node_type": "Person"}]
        with self.assertRaises(PIIDetected):
            self.adapter.write_entities(entities, "Z1931", project_id="test")

    @patch("merkraum_pii.gate_entities")
    def test_write_entities_off_mode_skips(self, mock_gate):
        """Verify write_entities skips PII gateway when mode is off."""
        self.adapter.get_project.return_value = {"pii_mode": "off"}
        entities = [{"name": "TestConcept", "node_type": "Concept"}]
        count = self.adapter.write_entities(entities, "Z1931", project_id="test")
        mock_gate.assert_not_called()
        self.assertEqual(count, 1)

    @patch("merkraum_pii.gate_entities")
    def test_write_entities_default_warn_when_no_project(self, mock_gate):
        """Verify default warn mode when project metadata not found."""
        self.adapter.get_project.return_value = None
        entities = [{"name": "Test", "node_type": "Concept"}]
        mock_gate.return_value = {
            "entities": entities, "findings": [], "blocked": [], "mode": "warn"
        }
        count = self.adapter.write_entities(entities, "Z1931", project_id="test")
        mock_gate.assert_called_once()
        # Check it was called with warn mode (default)
        call_kwargs = mock_gate.call_args[1]
        self.assertEqual(call_kwargs["mode"], "warn")


class TestWriteEntitiesPIIAuditLogging(unittest.TestCase):
    """Test PII gateway audit trail logging in write_entities."""

    def setUp(self):
        from merkraum_backend import Neo4jQdrantAdapter
        self.adapter = Neo4jQdrantAdapter.__new__(Neo4jQdrantAdapter)
        self.adapter._driver = MagicMock()
        self.adapter._qdrant = None
        self.adapter._embedder = None
        self.adapter._neo4j_uri = "bolt://mock:7687"
        self.adapter._neo4j_user = "neo4j"
        self.adapter._neo4j_password = "test"
        self.adapter._qdrant_url = "http://mock:6333"
        self.adapter._qdrant_api_key = None
        self.adapter._embed_model_name = "BAAI/bge-small-en-v1.5"
        # Mock session/tx for entity writes
        self.session = MagicMock()
        self.tx = MagicMock()
        after_result = MagicMock()
        after_result.single.return_value = {"props": {"name": "mock", "node_type": "Concept"}}
        self.tx.run.return_value = after_result
        self.session.begin_transaction.return_value = self.tx
        self.adapter._driver.session.return_value.__enter__ = lambda s: self.session
        self.adapter._driver.session.return_value.__exit__ = MagicMock(return_value=False)
        self.adapter.vector_upsert = MagicMock(return_value=True)
        self.adapter.get_project = MagicMock(return_value=None)

    @patch("merkraum_pii.gate_entities")
    def test_last_pii_result_set_on_findings(self, mock_gate):
        """Verify _last_pii_result is set when PII findings exist."""
        self.adapter.get_project.return_value = {"pii_mode": "warn"}
        findings = [{"entity_type": "PERSON", "text": "John", "score": 0.9, "field": "name", "start": 0, "end": 4}]
        entities = [{"name": "Test", "node_type": "Concept"}]
        mock_gate.return_value = {
            "entities": entities, "findings": findings, "blocked": [], "mode": "warn"
        }
        self.adapter.write_entities(entities, "Z1936", project_id="test")
        self.assertIsNotNone(self.adapter._last_pii_result)
        self.assertEqual(len(self.adapter._last_pii_result["findings"]), 1)
        self.assertEqual(self.adapter._last_pii_result["mode"], "warn")

    @patch("merkraum_pii.gate_entities")
    def test_last_pii_result_none_when_no_findings(self, mock_gate):
        """Verify _last_pii_result is None when no PII found."""
        self.adapter.get_project.return_value = {"pii_mode": "warn"}
        entities = [{"name": "Test", "node_type": "Concept"}]
        mock_gate.return_value = {
            "entities": entities, "findings": [], "blocked": [], "mode": "warn"
        }
        self.adapter.write_entities(entities, "Z1936", project_id="test")
        self.assertIsNone(self.adapter._last_pii_result)

    @patch("merkraum_pii.gate_entities")
    def test_last_pii_result_none_when_off(self, mock_gate):
        """Verify _last_pii_result is None when mode is off."""
        self.adapter.get_project.return_value = {"pii_mode": "off"}
        entities = [{"name": "Test", "node_type": "Concept"}]
        self.adapter.write_entities(entities, "Z1936", project_id="test")
        self.assertIsNone(self.adapter._last_pii_result)
        mock_gate.assert_not_called()

    @patch("merkraum_pii.gate_entities")
    def test_pii_audit_log_created_on_findings(self, mock_gate):
        """Verify pii_scan OperationLog entry is created when findings exist."""
        self.adapter.get_project.return_value = {"pii_mode": "warn"}
        findings = [{"entity_type": "PERSON", "text": "John", "score": 0.9, "field": "name", "start": 0, "end": 4}]
        entities = [{"name": "Test", "node_type": "Concept"}]
        mock_gate.return_value = {
            "entities": entities, "findings": findings, "blocked": [], "mode": "warn"
        }
        # Track _log_operation calls
        self.adapter._log_operation = MagicMock()
        self.adapter._operation_id = MagicMock(return_value="test-op-id")
        self.adapter.write_entities(entities, "Z1936", project_id="test")
        # _log_operation should be called at least once for pii_scan
        pii_calls = [
            c for c in self.adapter._log_operation.call_args_list
            if len(c[0]) >= 3 and c[0][2] == "pii_scan"
        ]
        self.assertEqual(len(pii_calls), 1)
        # Verify payload structure
        payload = pii_calls[0][1].get("payload") if pii_calls[0][1] else pii_calls[0][0][5]
        self.assertIn("findings", payload)
        self.assertIn("mode", payload)
        self.assertEqual(payload["mode"], "warn")

    @patch("merkraum_pii.gate_entities")
    def test_pii_audit_log_not_created_when_no_findings(self, mock_gate):
        """Verify no pii_scan entry when no findings."""
        self.adapter.get_project.return_value = {"pii_mode": "warn"}
        entities = [{"name": "Test", "node_type": "Concept"}]
        mock_gate.return_value = {
            "entities": entities, "findings": [], "blocked": [], "mode": "warn"
        }
        self.adapter._log_operation = MagicMock()
        self.adapter.write_entities(entities, "Z1936", project_id="test")
        pii_calls = [
            c for c in self.adapter._log_operation.call_args_list
            if len(c[0]) >= 3 and c[0][2] == "pii_scan"
        ]
        self.assertEqual(len(pii_calls), 0)

    @patch("merkraum_pii.gate_entities")
    def test_log_mode_records_findings(self, mock_gate):
        """Verify log mode also records PII findings in audit trail."""
        self.adapter.get_project.return_value = {"pii_mode": "log"}
        findings = [{"entity_type": "EMAIL_ADDRESS", "text": "test@...", "score": 0.85, "field": "summary", "start": 0, "end": 7}]
        entities = [{"name": "Test", "node_type": "Concept"}]
        mock_gate.return_value = {
            "entities": entities, "findings": findings, "blocked": [], "mode": "log"
        }
        self.adapter._log_operation = MagicMock()
        self.adapter._operation_id = MagicMock(return_value="test-op-id")
        self.adapter.write_entities(entities, "Z1936", project_id="test")
        pii_calls = [
            c for c in self.adapter._log_operation.call_args_list
            if len(c[0]) >= 3 and c[0][2] == "pii_scan"
        ]
        self.assertEqual(len(pii_calls), 1)


class TestPIIModes(unittest.TestCase):
    """Verify mode constants."""

    def test_all_modes_present(self):
        self.assertIn("block", PII_MODES)
        self.assertIn("warn", PII_MODES)
        self.assertIn("log", PII_MODES)
        self.assertIn("off", PII_MODES)

    def test_default_mode_is_warn(self):
        self.assertEqual(DEFAULT_MODE, "warn")

    def test_default_language_is_auto(self):
        self.assertEqual(DEFAULT_LANGUAGE, "auto")


if __name__ == "__main__":
    unittest.main()
