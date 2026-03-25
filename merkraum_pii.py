#!/usr/bin/env python3
"""
Merkraum PII Gateway — Privacy-preserving entity ingestion filter.

SUP-182: Detects and handles personally identifiable information (PII) in
entities before they reach the knowledge graph. Uses Microsoft Presidio
for detection with spaCy NER models.

Modes:
  - block:  Reject entities containing PII (raises PIIDetected)
  - warn:   Allow entities but return PII findings in response
  - log:    Allow entities, log PII findings silently
  - off:    No PII detection

Supports English and German dynamically via language detection or
explicit project-level configuration.

Standalone module — no dependency on Merkraum internals beyond the entity
dict format ({name, summary, node_type, ...}).

v1.0 — Z1931 (2026-03-25), Norman-approved
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# --- Presidio lazy loading ---
# Presidio + spaCy are optional dependencies. The gateway degrades
# gracefully if they are not installed (logs a warning, passes through).

_analyzer = None
_spacy_loaded = set()

PII_MODES = {"block", "warn", "log", "off"}
SUPPORTED_LANGUAGES = {"en", "de"}
DEFAULT_MODE = "warn"
DEFAULT_LANGUAGE = "auto"

# Default PII entity types to detect (GDPR-critical categories).
DEFAULT_PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IBAN_CODE",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "LOCATION",
    "DATE_OF_BIRTH",
    "NRP",              # Nationality, religious, political group
    "MEDICAL_LICENSE",
]


class PIIDetected(Exception):
    """Raised when PII is detected in block mode."""

    def __init__(self, findings: list):
        self.findings = findings
        entities = ", ".join(f["entity_type"] for f in findings[:5])
        super().__init__(f"PII detected: {entities}")


class PIIFinding:
    """A single PII detection result."""

    __slots__ = ("entity_type", "text", "score", "field", "start", "end")

    def __init__(self, entity_type: str, text: str, score: float,
                 field: str, start: int, end: int):
        self.entity_type = entity_type
        self.text = text
        self.score = score
        self.field = field
        self.start = start
        self.end = end

    def to_dict(self) -> dict:
        return {
            "entity_type": self.entity_type,
            "text": self.text[:20] + "..." if len(self.text) > 20 else self.text,
            "score": round(self.score, 3),
            "field": self.field,
            "start": self.start,
            "end": self.end,
        }


def _get_analyzer():
    """Lazy-initialize the Presidio AnalyzerEngine with spaCy models."""
    global _analyzer
    if _analyzer is not None:
        return _analyzer
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        # Configure spaCy models for both languages
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "en", "model_name": "en_core_web_lg"},
                {"lang_code": "de", "model_name": "de_core_news_lg"},
            ],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        nlp_engine = provider.create_engine()
        _analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["en", "de"],
        )
        logger.info("PII Gateway: Presidio AnalyzerEngine initialized (en + de)")
        return _analyzer
    except ImportError:
        logger.warning(
            "PII Gateway: presidio-analyzer or spaCy models not installed. "
            "PII detection disabled. Install with: "
            "pip install presidio-analyzer spacy && "
            "python -m spacy download en_core_web_lg && "
            "python -m spacy download de_core_news_lg"
        )
        return None
    except Exception as e:
        logger.warning(f"PII Gateway: Failed to initialize Presidio: {e}")
        return None


def detect_language(text: str) -> str:
    """Simple heuristic language detection for en/de.

    Uses character patterns and common words. Not a full language detector
    but sufficient for PII gateway purposes where the choice is binary.
    """
    if not text or len(text) < 10:
        return "en"  # Default to English for very short text

    # German-specific indicators
    german_patterns = [
        r'\b(?:der|die|das|und|ist|ein|eine|für|mit|auf|den|dem|des)\b',
        r'[äöüßÄÖÜ]',
        r'\b(?:nicht|auch|sich|wird|oder|nach|bei|nur|noch|kann)\b',
    ]
    german_score = 0
    for pattern in german_patterns:
        german_score += len(re.findall(pattern, text, re.IGNORECASE))

    # If significant German indicators found, classify as German
    words = text.split()
    if words and german_score / max(len(words), 1) > 0.1:
        return "de"
    return "en"


def scan_text(text: str, language: str = "auto",
              entity_types: Optional[list] = None,
              score_threshold: float = 0.5,
              field: str = "text") -> list:
    """Scan a text string for PII entities.

    Args:
        text: The text to scan.
        language: Language code ("en", "de", or "auto" for detection).
        entity_types: List of Presidio entity types to detect.
                      Defaults to DEFAULT_PII_ENTITIES.
        score_threshold: Minimum confidence score (0.0-1.0).
        field: Name of the field being scanned (for reporting).

    Returns:
        List of PIIFinding objects.
    """
    if not text or not text.strip():
        return []

    analyzer = _get_analyzer()
    if analyzer is None:
        return []  # Presidio not available — pass through

    if language == "auto":
        language = detect_language(text)
    if language not in SUPPORTED_LANGUAGES:
        language = "en"

    if entity_types is None:
        entity_types = DEFAULT_PII_ENTITIES

    try:
        results = analyzer.analyze(
            text=text,
            language=language,
            entities=entity_types,
            score_threshold=score_threshold,
        )
    except Exception as e:
        logger.warning(f"PII Gateway: Analysis error: {e}")
        return []

    findings = []
    for result in results:
        findings.append(PIIFinding(
            entity_type=result.entity_type,
            text=text[result.start:result.end],
            score=result.score,
            field=field,
            start=result.start,
            end=result.end,
        ))

    return findings


def scan_entity(entity: dict, language: str = "auto",
                entity_types: Optional[list] = None,
                score_threshold: float = 0.5) -> list:
    """Scan a single entity dict for PII in name and summary fields.

    Args:
        entity: Entity dict with at least 'name' key, optionally 'summary'.
        language: Language code or "auto".
        entity_types: Presidio entity types to detect.
        score_threshold: Minimum confidence score.

    Returns:
        List of PIIFinding objects across all scanned fields.
    """
    findings = []

    name = entity.get("name", "")
    if name:
        findings.extend(scan_text(
            name, language=language, entity_types=entity_types,
            score_threshold=score_threshold, field="name",
        ))

    summary = entity.get("summary", "")
    if summary:
        findings.extend(scan_text(
            summary, language=language, entity_types=entity_types,
            score_threshold=score_threshold, field="summary",
        ))

    return findings


def gate_entities(entities: list, mode: str = DEFAULT_MODE,
                  language: str = DEFAULT_LANGUAGE,
                  entity_types: Optional[list] = None,
                  score_threshold: float = 0.5) -> dict:
    """Apply PII gateway to a list of entities before ingestion.

    This is the main entry point called from write_entities().

    Args:
        entities: List of entity dicts.
        mode: PII mode — "block", "warn", "log", or "off".
        language: Language code or "auto".
        entity_types: Presidio entity types to detect.
        score_threshold: Minimum confidence score.

    Returns:
        dict with keys:
          - "entities": The (possibly filtered) entity list to proceed with.
          - "findings": List of finding dicts (for warn mode response).
          - "blocked": List of entity names that were blocked.
          - "mode": The mode that was applied.

    Raises:
        PIIDetected: In "block" mode when any entity contains PII.
    """
    if mode == "off" or not entities:
        return {
            "entities": entities,
            "findings": [],
            "blocked": [],
            "mode": mode,
        }

    if mode not in PII_MODES:
        logger.warning(f"PII Gateway: Invalid mode '{mode}', using '{DEFAULT_MODE}'")
        mode = DEFAULT_MODE

    all_findings = []
    blocked_names = []
    passed_entities = []

    for entity in entities:
        findings = scan_entity(
            entity, language=language, entity_types=entity_types,
            score_threshold=score_threshold,
        )

        if findings:
            finding_dicts = [f.to_dict() for f in findings]
            entity_name = entity.get("name", "<unnamed>")

            if mode == "block":
                # Collect all findings first, then raise
                all_findings.extend(finding_dicts)
                blocked_names.append(entity_name)
                logger.warning(
                    f"PII Gateway [block]: Entity '{entity_name}' blocked — "
                    f"{len(findings)} PII finding(s)"
                )
            elif mode == "warn":
                all_findings.extend(finding_dicts)
                passed_entities.append(entity)
                logger.info(
                    f"PII Gateway [warn]: Entity '{entity_name}' — "
                    f"{len(findings)} PII finding(s)"
                )
            elif mode == "log":
                all_findings.extend(finding_dicts)
                passed_entities.append(entity)
                logger.info(
                    f"PII Gateway [log]: Entity '{entity_name}' — "
                    f"{len(findings)} PII finding(s)"
                )
        else:
            passed_entities.append(entity)

    if mode == "block" and blocked_names:
        raise PIIDetected(all_findings)

    return {
        "entities": passed_entities,
        "findings": all_findings,
        "blocked": blocked_names,
        "mode": mode,
    }


def get_pii_settings(project_meta: Optional[dict] = None) -> dict:
    """Extract PII gateway settings from project metadata.

    Settings are stored on the ProjectMeta node as:
      - pii_mode: "block" | "warn" | "log" | "off" (default: "warn")
      - pii_language: "en" | "de" | "auto" (default: "auto")

    Args:
        project_meta: ProjectMeta node properties dict, or None.

    Returns:
        dict with "mode" and "language" keys.
    """
    if project_meta is None:
        return {"mode": DEFAULT_MODE, "language": DEFAULT_LANGUAGE}

    mode = project_meta.get("pii_mode", DEFAULT_MODE)
    if mode not in PII_MODES:
        mode = DEFAULT_MODE

    language = project_meta.get("pii_language", DEFAULT_LANGUAGE)
    if language not in SUPPORTED_LANGUAGES and language != "auto":
        language = DEFAULT_LANGUAGE

    return {"mode": mode, "language": language}
