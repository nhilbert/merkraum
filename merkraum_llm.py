#!/usr/bin/env python3
"""
Merkraum LLM Extraction — configurable provider (AWS Bedrock or OpenAI).

Supports:
  - AWS Bedrock (default): Mistral, Anthropic, Amazon models via Converse API
  - OpenAI: gpt-4o-mini and other OpenAI models via REST API

Configuration via environment variables:
  MERKRAUM_LLM_PROVIDER  = bedrock | openai   (default: bedrock)
  MERKRAUM_LLM_MODEL     = model ID           (default: provider-specific)
  MERKRAUM_LLM_REGION    = AWS region          (default: eu-central-1)

For Bedrock: uses boto3 + Converse API. No API key needed (uses IAM/instance role).
For OpenAI: requires OPENAI_API_KEY in environment.

v1.0 — SUP-149 (2026-03-14). EU-compliant model support via AWS Bedrock.
"""

import json
import logging
import os
import urllib.request
import urllib.error

logger = logging.getLogger("merkraum-llm")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.environ.get("MERKRAUM_LLM_PROVIDER", "bedrock")
LLM_REGION = os.environ.get("MERKRAUM_LLM_REGION", "eu-central-1")

# Default models per provider
_DEFAULT_MODELS = {
    "bedrock": "eu.mistral.mistral-large-2411-v1:0",
    "openai": "gpt-4o-mini",
}


def _get_model() -> str:
    """Return configured model or provider default."""
    return os.environ.get("MERKRAUM_LLM_MODEL") or _DEFAULT_MODELS.get(
        LLM_PROVIDER, _DEFAULT_MODELS["bedrock"]
    )


# ---------------------------------------------------------------------------
# Bedrock Converse API
# ---------------------------------------------------------------------------

def _llm_call_bedrock(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 8000,
) -> str:
    """Call AWS Bedrock Converse API and return the text response."""
    import boto3

    model_id = model or _get_model()
    region = LLM_REGION

    client = boto3.client("bedrock-runtime", region_name=region)

    messages = [{"role": "user", "content": [{"text": user_prompt}]}]

    kwargs = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": {
            "temperature": temperature,
            "maxTokens": max_tokens,
        },
    }
    # System prompt — Bedrock Converse API uses top-level system parameter
    if system_prompt:
        kwargs["system"] = [{"text": system_prompt}]

    response = client.converse(**kwargs)
    output = response.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])

    for block in content_blocks:
        if "text" in block:
            return block["text"]

    return ""


# ---------------------------------------------------------------------------
# OpenAI REST API
# ---------------------------------------------------------------------------

def _llm_call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 8000,
    api_key: str | None = None,
) -> str:
    """Call OpenAI chat completions API and return the text response."""
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY not set")

    model_id = model or _get_model()
    url = "https://api.openai.com/v1/chat/completions"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model_id,
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "messages": messages,
    }

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
        return data["choices"][0]["message"].get("content", "")


# ---------------------------------------------------------------------------
# Unified extraction interface
# ---------------------------------------------------------------------------

def llm_extract(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 8000,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Extract entities and relationships from text using configured LLM.

    Returns parsed JSON dict with 'entities' and 'relationships' keys.
    Falls back to empty arrays on parse failure.
    """
    prov = (provider or LLM_PROVIDER).lower()
    logger.info("LLM extraction: provider=%s, model=%s", prov, model or _get_model())

    try:
        if prov == "bedrock":
            raw = _llm_call_bedrock(
                system_prompt, user_prompt,
                model=model, temperature=temperature, max_tokens=max_tokens,
            )
        elif prov == "openai":
            raw = _llm_call_openai(
                system_prompt, user_prompt,
                model=model, temperature=temperature, max_tokens=max_tokens,
                api_key=api_key,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {prov}")
    except Exception as e:
        logger.error("LLM call failed (provider=%s): %s", prov, e)
        raise

    if not raw:
        return {"entities": [], "relationships": []}

    # Strip markdown fences if present (Bedrock models don't use json_object mode)
    text = raw.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response as JSON: %s", e)
        logger.debug("Raw response: %s", raw[:500])
        return {"entities": [], "relationships": []}

    return {
        "entities": result.get("entities", []),
        "relationships": result.get("relationships", []),
    }


def llm_call(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 4000,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict | None:
    """General-purpose LLM call that returns parsed JSON.

    Unlike llm_extract(), this does not assume entities/relationships schema.
    Returns the parsed JSON dict, or None on failure.
    """
    prov = (provider or LLM_PROVIDER).lower()

    try:
        if prov == "bedrock":
            raw = _llm_call_bedrock(
                system_prompt, user_prompt,
                model=model, temperature=temperature, max_tokens=max_tokens,
            )
        elif prov == "openai":
            raw = _llm_call_openai(
                system_prompt, user_prompt,
                model=model, temperature=temperature, max_tokens=max_tokens,
                api_key=api_key,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {prov}")
    except Exception as e:
        logger.error("LLM call failed (provider=%s): %s", prov, e)
        return None

    if not raw:
        return None

    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM JSON response: %s", e)
        return None


def get_provider_info() -> dict:
    """Return current LLM configuration for diagnostics."""
    return {
        "provider": LLM_PROVIDER,
        "model": _get_model(),
        "region": LLM_REGION if LLM_PROVIDER == "bedrock" else None,
    }
