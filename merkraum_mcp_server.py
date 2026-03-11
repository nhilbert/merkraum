#!/usr/bin/env python3
"""
Merkraum MCP Server — Knowledge Graph for Claude, Cursor, and any MCP client.

Open-source standalone server using Neo4j + Qdrant/Pinecone with
Cognito JWT authentication and OAuth 2.0 discovery for MCP clients.

Security:
    1. Cognito JWT validation on every MCP request
    2. OAuth discovery via /.well-known/oauth-authorization-server
    3. Dynamic client registration stub (RFC 7591)
    4. Audit logging for every tool call

Architecture:
    Claude Desktop/Code → HTTPS (CloudFront) → nginx → this server (127.0.0.1:8090)

v1.0 — Z1144 (2026-03-07). Extracted from vsg_mcp_server.py.
v2.0 — (2026-03-11). Added Cognito OAuth for MCP client authentication.
"""

import os
import sys
import json
import time
import uuid
import queue
import logging
import urllib.request
import urllib.error
import threading
import asyncio
import argparse
from functools import partial
from typing import Optional

# Ensure project root is on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from fastmcp import FastMCP
from fastmcp.server.auth import AccessToken, TokenVerifier

from merkraum_backend import (
    create_adapter, BackendAdapter, NODE_TYPES, RELATIONSHIP_TYPES, TIER_LIMITS,
)

# --- Configuration ---

COGNITO_REGION = os.environ.get("COGNITO_AWS_REGION", "eu-central-1")
COGNITO_POOL_ID = os.environ.get("COGNITO_USER_POOL_ID", "eu-central-1_JhyAYVWGl")
COGNITO_ISSUER = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_POOL_ID}"
COGNITO_JWKS_URL = f"{COGNITO_ISSUER}/.well-known/jwks.json"
COGNITO_APP_CLIENT_ID = os.environ.get("MCP_COGNITO_CLIENT_ID", "4moil9kjh8ufgvqs5nv8chq64n")
COGNITO_APP_CLIENT_SECRET = os.environ.get("MCP_COGNITO_CLIENT_SECRET", "")
COGNITO_AUTH_DOMAIN = os.environ.get("COGNITO_AUTH_DOMAIN", "https://merkraum.auth.eu-central-1.amazoncognito.com")
COGNITO_TOKEN_URL = f"{COGNITO_AUTH_DOMAIN}/oauth2/token"
COGNITO_AUTHORIZE_URL = f"{COGNITO_AUTH_DOMAIN}/oauth2/authorize"
COGNITO_ALLOWED_SCOPES = {"openid", "email", "phone", "profile"}

MCP_BASE_URL = os.environ.get("MCP_BASE_URL", "https://www.agent.nhilbert.de/mcp/merkraum")

DEFAULT_BACKEND = os.environ.get("MERKRAUM_BACKEND", "neo4j_qdrant")
DEFAULT_PROJECT = os.environ.get("MERKRAUM_PROJECT", "default")
HTTP_PORT = int(os.environ.get("MERKRAUM_MCP_PORT", "8090"))
HTTP_HOST = os.environ.get("MERKRAUM_MCP_HOST", "127.0.0.1")

# LLM for knowledge extraction (optional)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EXTRACTION_MODEL = os.environ.get("MERKRAUM_EXTRACTION_MODEL", "gpt-4o-mini")

# --- Logging ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("merkraum-mcp")

# --- Secrets Manager ---

def _load_secrets():
    """Load secrets from AWS Secrets Manager into environment variables."""
    try:
        import boto3
        client = boto3.client("secretsmanager", region_name="eu-central-1")
        resp = client.get_secret_value(SecretId="merkraum/config")
        secrets = json.loads(resp["SecretString"])
        for key, value in secrets.items():
            os.environ.setdefault(key, value)
        logger.info("Loaded %d secrets from AWS Secrets Manager", len(secrets))
    except Exception as exc:
        logger.info("Secrets Manager not available (%s), using env vars", exc)


# --- JWT Validation ---

_jwks_cache = None
_jwks_cache_time = 0
_jwks_lock = threading.Lock()
JWKS_CACHE_TTL = 3600  # 1 hour


def _fetch_jwks():
    """Fetch Cognito JWKS (cached for 1 hour)."""
    global _jwks_cache, _jwks_cache_time
    now = time.time()
    if _jwks_cache and (now - _jwks_cache_time) < JWKS_CACHE_TTL:
        return _jwks_cache
    with _jwks_lock:
        if _jwks_cache and (now - _jwks_cache_time) < JWKS_CACHE_TTL:
            return _jwks_cache
        try:
            req = urllib.request.Request(COGNITO_JWKS_URL)
            with urllib.request.urlopen(req, timeout=10) as resp:
                _jwks_cache = json.loads(resp.read())
                _jwks_cache_time = time.time()
                logger.info("JWKS refreshed from Cognito")
                return _jwks_cache
        except Exception as e:
            logger.error("Failed to fetch JWKS: %s", e)
            if _jwks_cache:
                return _jwks_cache  # Use stale cache
            raise


def validate_jwt(token: str) -> dict:
    """Validate a Cognito JWT token. Returns decoded claims or raises."""
    import jwt as pyjwt
    from jwt.algorithms import RSAAlgorithm

    jwks = _fetch_jwks()

    # Get the key ID from the token header
    header = pyjwt.get_unverified_header(token)
    kid = header.get("kid")
    if not kid:
        raise ValueError("Token missing kid header")

    # Find matching key in JWKS
    matching_key = None
    for key_data in jwks.get("keys", []):
        if key_data.get("kid") == kid:
            matching_key = key_data
            break

    if not matching_key:
        # Force refresh JWKS in case of key rotation
        global _jwks_cache_time
        _jwks_cache_time = 0
        jwks = _fetch_jwks()
        for key_data in jwks.get("keys", []):
            if key_data.get("kid") == kid:
                matching_key = key_data
                break

    if not matching_key:
        raise ValueError("Token signed with unknown key")

    # Construct public key from JWK
    public_key = RSAAlgorithm.from_jwk(json.dumps(matching_key))

    # Decode and validate
    claims = pyjwt.decode(
        token,
        public_key,
        algorithms=["RS256"],
        issuer=COGNITO_ISSUER,
        options={
            "verify_aud": False,  # Cognito access tokens don't have 'aud'
            "verify_exp": True,
            "verify_iss": True,
        },
    )
    return claims


# --- Audit Logging ---

def audit_log(tool_name: str, sub: str, params: dict,
              duration_ms: int, status: str, error: str = None):
    """Log every tool call for security audit."""
    entry = {
        "tool": tool_name,
        "sub": sub,
        "params": {k: (v[:100] if isinstance(v, str) and len(v) > 100 else v)
                   for k, v in params.items()},
        "duration_ms": duration_ms,
        "status": status,
    }
    if error:
        entry["error"] = error[:200]
    logger.info("AUDIT %s", json.dumps(entry))


# --- Token Verifier ---

class CognitoTokenVerifier(TokenVerifier):
    """Validates Cognito JWT access tokens."""

    async def verify_token(self, token: str) -> AccessToken | None:
        """Called by FastMCP for every authenticated request."""
        if not token:
            return None
        try:
            claims = validate_jwt(token)
            return AccessToken(
                token=token,
                client_id=claims.get("client_id", "unknown"),
                scopes=claims.get("scope", "").split() if claims.get("scope") else [],
                expires_at=claims.get("exp"),
                claims=claims,
            )
        except Exception as e:
            logger.warning("JWT validation failed: %s", e)
            return None


# --- MCP Server Setup ---

mcp = FastMCP(
    "Merkraum Knowledge Graph",
    auth=CognitoTokenVerifier(),
)


# --- OAuth Discovery + Proxy Endpoints ---

from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from urllib.parse import urlencode


@mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
async def oauth_authorization_server_metadata(request: Request) -> JSONResponse:
    """Serve OAuth metadata with endpoints pointing to our proxy routes."""
    return JSONResponse({
        "issuer": COGNITO_ISSUER,
        "authorization_endpoint": f"{MCP_BASE_URL}/authorize",
        "token_endpoint": f"{MCP_BASE_URL}/token",
        "revocation_endpoint": f"{COGNITO_AUTH_DOMAIN}/oauth2/revoke",
        "jwks_uri": COGNITO_JWKS_URL,
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
        "scopes_supported": ["openid", "email", "profile"],
        "registration_endpoint": f"{MCP_BASE_URL}/register",
    })


# --- OAuth Authorize Proxy ---

AUTHORIZE_ALLOWED_PARAMS = {
    "response_type", "client_id", "redirect_uri", "state",
    "scope", "code_challenge", "code_challenge_method", "nonce",
}


@mcp.custom_route("/authorize", methods=["GET"])
async def authorize_proxy(request: Request) -> RedirectResponse:
    """Proxy authorize request to Cognito, stripping unsupported params."""
    params = dict(request.query_params)
    logger.info("AUDIT /authorize incoming params: %s", list(params.keys()))

    # Filter to only Cognito-supported params (strips 'resource' etc.)
    filtered = {k: v for k, v in params.items() if k in AUTHORIZE_ALLOWED_PARAMS}

    # Fix scope: keep only Cognito-supported scopes
    if "scope" in filtered:
        requested_scopes = filtered["scope"].split()
        valid_scopes = [s for s in requested_scopes if s in COGNITO_ALLOWED_SCOPES]
        if not valid_scopes:
            valid_scopes = ["openid"]
        filtered["scope"] = " ".join(valid_scopes)

    target_url = f"{COGNITO_AUTHORIZE_URL}?{urlencode(filtered)}"
    logger.info("AUDIT /authorize redirecting to Cognito (filtered %d params)",
                len(params) - len(filtered))
    return RedirectResponse(url=target_url, status_code=302)


# --- OAuth Token Proxy ---

TOKEN_ALLOWED_PARAMS = {
    "grant_type", "code", "redirect_uri", "client_id",
    "code_verifier", "refresh_token", "scope",
}


@mcp.custom_route("/token", methods=["POST"])
async def token_proxy(request: Request) -> Response:
    """Proxy token request to Cognito's token endpoint."""
    body = await request.body()
    content_type = request.headers.get("content-type", "")

    # Parse form body
    if "application/x-www-form-urlencoded" in content_type:
        from urllib.parse import parse_qs
        parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
        params = {k: v[0] for k, v in parsed.items()}
    else:
        try:
            params = json.loads(body)
        except Exception:
            params = {}

    logger.info("AUDIT /token grant_type=%s", params.get("grant_type", "unknown"))

    # Filter to only Cognito-supported params
    filtered = {k: v for k, v in params.items() if k in TOKEN_ALLOWED_PARAMS}

    # Inject client secret if configured
    if COGNITO_APP_CLIENT_SECRET:
        filtered["client_secret"] = COGNITO_APP_CLIENT_SECRET

    # Fix scope if present
    if "scope" in filtered:
        requested_scopes = filtered["scope"].split()
        valid_scopes = [s for s in requested_scopes if s in COGNITO_ALLOWED_SCOPES]
        if valid_scopes:
            filtered["scope"] = " ".join(valid_scopes)
        else:
            del filtered["scope"]

    # POST to Cognito token endpoint
    encoded_body = urlencode(filtered).encode("utf-8")
    req = urllib.request.Request(
        COGNITO_TOKEN_URL,
        data=encoded_body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp_body = resp.read()
            resp_status = resp.status
            resp_headers = dict(resp.headers)
    except urllib.error.HTTPError as e:
        resp_body = e.read()
        resp_status = e.code
        resp_headers = dict(e.headers)
        logger.warning("AUDIT /token Cognito returned %d: %s", resp_status, resp_body[:200])
    except Exception as e:
        logger.error("AUDIT /token proxy error: %s", e)
        return JSONResponse({"error": "token_proxy_error", "detail": str(e)}, status_code=502)

    return Response(
        content=resp_body,
        status_code=resp_status,
        media_type=resp_headers.get("Content-Type", "application/json"),
    )


# --- Dynamic Client Registration Stub ---

@mcp.custom_route("/register", methods=["POST"])
async def register_client(request: Request) -> JSONResponse:
    """Return pre-created Cognito app client for MCP OAuth bootstrap."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    logger.info("AUDIT register client_name=%s", body.get("client_name", "unknown"))

    return JSONResponse({
        "client_id": COGNITO_APP_CLIENT_ID,
        "client_name": body.get("client_name", "mcp-client"),
        "redirect_uris": body.get("redirect_uris", []),
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }, status_code=201)


# --- Backend Singleton ---

_adapter: Optional[BackendAdapter] = None
_adapter_lock = threading.Lock()


def _get_adapter() -> BackendAdapter:
    """Lazy-initialize the backend adapter."""
    global _adapter
    if _adapter is not None:
        return _adapter
    with _adapter_lock:
        if _adapter is not None:
            return _adapter
        logger.info("Initializing backend: %s", DEFAULT_BACKEND)
        _adapter = create_adapter(DEFAULT_BACKEND)
        _adapter.connect()
        logger.info("Backend connected: %s", DEFAULT_BACKEND)
        return _adapter


# --- Async Helper ---

def _run_sync(func, *args, **kwargs):
    """Run a synchronous function in a thread executor."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, partial(func, *args, **kwargs))


# --- Tools ---

@mcp.tool()
async def search_knowledge(
    query: str,
    top_k: int = 5,
    project: str = None,
) -> dict:
    """Search the knowledge graph using semantic similarity.
    Returns ranked results with content, scores, and metadata.

    Args:
        query: What to search for (natural language)
        top_k: Number of results to return (1-20, default 5)
        project: Project ID (default: server default)
    """
    project = project or DEFAULT_PROJECT
    top_k = max(1, min(20, top_k))
    adapter = _get_adapter()
    start = time.time()
    try:
        results = await _run_sync(adapter.vector_search, query, top_k, project)
        audit_log("search_knowledge", "authed",
                  {"query": query, "top_k": top_k, "project": project},
                  int((time.time() - start) * 1000), "ok")
        return {"results": results, "count": len(results), "query": query,
                "duration_ms": int((time.time() - start) * 1000)}
    except Exception as e:
        audit_log("search_knowledge", "authed",
                  {"query": query}, int((time.time() - start) * 1000), "error", str(e))
        return {"error": str(e)}


@mcp.tool()
async def traverse_graph(
    entity: str,
    depth: int = 2,
    project: str = None,
) -> dict:
    """Walk the knowledge graph from an entity, following relationships.

    Args:
        entity: Entity name to start from
        depth: How many hops to traverse (1-4, default 2)
        project: Project ID (default: server default)
    """
    project = project or DEFAULT_PROJECT
    depth = max(1, min(4, depth))
    adapter = _get_adapter()
    start = time.time()
    try:
        result = await _run_sync(adapter.traverse, entity, project, depth)
        result["duration_ms"] = int((time.time() - start) * 1000)
        audit_log("traverse_graph", "authed",
                  {"entity": entity, "depth": depth, "project": project},
                  int((time.time() - start) * 1000), "ok")
        return result
    except Exception as e:
        audit_log("traverse_graph", "authed",
                  {"entity": entity}, int((time.time() - start) * 1000), "error", str(e))
        return {"error": str(e)}


@mcp.tool()
async def list_beliefs(
    status: str = "active",
    project: str = None,
) -> dict:
    """List beliefs in the knowledge graph.

    Args:
        status: Filter — active, uncertain, superseded, contradicted
        project: Project ID (default: server default)
    """
    project = project or DEFAULT_PROJECT
    if status not in ("active", "uncertain", "superseded", "contradicted"):
        status = "active"
    adapter = _get_adapter()
    start = time.time()
    try:
        beliefs = await _run_sync(adapter.get_beliefs, project, status)
        audit_log("list_beliefs", "authed",
                  {"status": status, "project": project},
                  int((time.time() - start) * 1000), "ok")
        return {"beliefs": beliefs, "count": len(beliefs), "status_filter": status,
                "duration_ms": int((time.time() - start) * 1000)}
    except Exception as e:
        audit_log("list_beliefs", "authed",
                  {"status": status}, int((time.time() - start) * 1000), "error", str(e))
        return {"error": str(e)}


@mcp.tool()
async def get_graph_stats(project: str = None) -> dict:
    """Get knowledge graph statistics: node counts by type, edge counts, totals.

    Args:
        project: Project ID (default: server default)
    """
    project = project or DEFAULT_PROJECT
    adapter = _get_adapter()
    start = time.time()
    try:
        stats = await _run_sync(adapter.get_stats, project)
        stats["duration_ms"] = int((time.time() - start) * 1000)
        audit_log("get_graph_stats", "authed",
                  {"project": project}, int((time.time() - start) * 1000), "ok")
        return stats
    except Exception as e:
        audit_log("get_graph_stats", "authed",
                  {}, int((time.time() - start) * 1000), "error", str(e))
        return {"error": str(e)}


@mcp.tool()
async def query_nodes(
    node_type: str = None,
    limit: int = 50,
    project: str = None,
) -> dict:
    """Query entities in the knowledge graph, optionally filtered by type.

    Args:
        node_type: Filter by type (Person, Organization, Project, Concept,
            Regulation, Event, Belief, Artifact, Interview, Quote). None = all.
        limit: Max results (1-200, default 50)
        project: Project ID (default: server default)
    """
    project = project or DEFAULT_PROJECT
    if node_type and node_type not in NODE_TYPES:
        return {"error": f"Unknown node_type: {node_type}. Valid: {NODE_TYPES}"}
    limit = max(1, min(200, limit))
    adapter = _get_adapter()
    start = time.time()
    try:
        nodes = await _run_sync(adapter.query_nodes, node_type, project, limit)
        audit_log("query_nodes", "authed",
                  {"node_type": node_type, "limit": limit, "project": project},
                  int((time.time() - start) * 1000), "ok")
        return {"nodes": nodes, "count": len(nodes), "node_type_filter": node_type,
                "duration_ms": int((time.time() - start) * 1000)}
    except Exception as e:
        audit_log("query_nodes", "authed",
                  {"node_type": node_type}, int((time.time() - start) * 1000), "error", str(e))
        return {"error": str(e)}


@mcp.tool()
async def add_knowledge(
    name: str,
    summary: str,
    node_type: str = "Concept",
    confidence: float = 0.7,
    project: str = None,
) -> dict:
    """Add a single entity to the knowledge graph.

    Args:
        name: Entity name
        summary: Brief description
        node_type: One of Person, Organization, Project, Concept, Regulation,
            Event, Belief, Artifact, Interview, Quote
        confidence: For Beliefs only (0.0-1.0, default 0.7)
        project: Project ID (default: server default)
    """
    project = project or DEFAULT_PROJECT
    if node_type not in NODE_TYPES:
        return {"error": f"Unknown node_type: {node_type}. Valid: {NODE_TYPES}"}
    adapter = _get_adapter()
    entity = {"name": name, "summary": summary, "node_type": node_type}
    if node_type == "Belief":
        entity["confidence"] = max(0.0, min(1.0, confidence))
    start = time.time()
    try:
        written = await _run_sync(
            adapter.write_entities, [entity], "manual", "user", project
        )
        await _run_sync(
            adapter.vector_upsert,
            f"{project}:{name}", f"{name}: {summary}",
            {"name": name, "node_type": node_type, "source": "manual"}, project,
        )
        audit_log("add_knowledge", "authed",
                  {"name": name, "node_type": node_type, "project": project},
                  int((time.time() - start) * 1000), "ok")
        return {"created": written > 0, "name": name, "node_type": node_type,
                "duration_ms": int((time.time() - start) * 1000)}
    except Exception as e:
        audit_log("add_knowledge", "authed",
                  {"name": name}, int((time.time() - start) * 1000), "error", str(e))
        return {"error": str(e)}


@mcp.tool()
async def add_relationship(
    entity_a: str,
    entity_b: str,
    relationship_type: str,
    reason: str = "",
    project: str = None,
) -> dict:
    """Link two entities with a typed relationship.

    Args:
        entity_a: Source entity name
        entity_b: Target entity name
        relationship_type: One of SUPPORTS, CONTRADICTS, COMPLEMENTS, SUPERSEDES,
            EXTENDS, REFINES, CREATED_BY, AFFILIATED_WITH, APPLIES, IMPLEMENTS,
            PARTICIPATED_IN, PRODUCES, REFERENCES, TEMPORAL, MENTIONS, PART_OF
        reason: Why this relationship exists (optional)
        project: Project ID (default: server default)
    """
    project = project or DEFAULT_PROJECT
    if relationship_type not in RELATIONSHIP_TYPES:
        return {"error": f"Unknown type: {relationship_type}. Valid: {RELATIONSHIP_TYPES}"}
    adapter = _get_adapter()
    rel = {
        "source": entity_a, "target": entity_b, "type": relationship_type,
        "reason": reason[:200] if reason else "", "confidence": 0.9,
    }
    start = time.time()
    try:
        written = await _run_sync(
            adapter.write_relationships, [rel], "manual", "user", project
        )
        audit_log("add_relationship", "authed",
                  {"a": entity_a, "b": entity_b, "type": relationship_type, "project": project},
                  int((time.time() - start) * 1000), "ok" if written else "noop")
        return {"created": written > 0, "source": entity_a, "target": entity_b,
                "type": relationship_type, "duration_ms": int((time.time() - start) * 1000)}
    except Exception as e:
        audit_log("add_relationship", "authed",
                  {"a": entity_a, "b": entity_b}, int((time.time() - start) * 1000), "error", str(e))
        return {"error": str(e)}


@mcp.tool()
async def update_belief(
    name: str,
    confidence: float = None,
    status: str = None,
    summary: str = None,
    project: str = None,
) -> dict:
    """Update an existing belief's confidence, status, or summary.

    Args:
        name: Belief name (must already exist)
        confidence: New confidence score (0.0-1.0). None = no change.
        status: New status — active or superseded. None = no change.
        summary: New summary text. None = no change.
        project: Project ID (default: server default)
    """
    project = project or DEFAULT_PROJECT
    adapter = _get_adapter()
    start = time.time()
    try:
        result = await _run_sync(
            adapter.update_belief, name, project,
            confidence=confidence, status=status, summary=summary,
        )
        result["duration_ms"] = int((time.time() - start) * 1000)
        audit_log("update_belief", "authed",
                  {"name": name, "project": project},
                  int((time.time() - start) * 1000), "ok")
        return result
    except Exception as e:
        audit_log("update_belief", "authed",
                  {"name": name}, int((time.time() - start) * 1000), "error", str(e))
        return {"error": str(e)}


@mcp.tool()
async def get_usage(
    tier: str = "free",
    project: str = None,
) -> dict:
    """Get usage metrics and tier limits.

    Args:
        tier: Pricing tier — free (100), pro (1000), team (5000), enterprise (50000)
        project: Project ID (default: server default)
    """
    project = project or DEFAULT_PROJECT
    tier = tier.lower()
    if tier not in TIER_LIMITS:
        tier = "free"
    adapter = _get_adapter()
    start = time.time()
    try:
        usage = await _run_sync(adapter.get_usage, project)
        node_limit = TIER_LIMITS[tier]
        usage_pct = round(usage["nodes"] / node_limit * 100, 1) if node_limit else 0
        return {"nodes": usage["nodes"], "edges": usage["edges"],
                "node_limit": node_limit, "usage_pct": usage_pct, "tier": tier,
                "duration_ms": int((time.time() - start) * 1000)}
    except Exception as e:
        audit_log("get_usage", "authed", {}, int((time.time() - start) * 1000), "error", str(e))
        return {"error": str(e)}


# --- Async Ingestion ---

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_job_queue: queue.Queue = None
MAX_JOBS = 50
MAX_INGEST_LENGTH = 10240


def _ensure_worker():
    global _job_queue
    if _job_queue is not None:
        return
    _job_queue = queue.Queue()

    def _worker():
        while True:
            job_id, text = _job_queue.get()
            _run_ingestion_job(job_id, text)
            _job_queue.task_done()

    t = threading.Thread(target=_worker, daemon=True, name="ingestion-worker")
    t.start()


def _run_ingestion_job(job_id: str, text: str):
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    start = time.time()
    try:
        result = _extract_and_write(text)
        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = result
            _jobs[job_id]["duration_ms"] = int((time.time() - start) * 1000)
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["duration_ms"] = int((time.time() - start) * 1000)


def _extract_and_write(text: str) -> dict:
    """Extract entities/relationships from text via LLM, write to graph."""
    api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}

    prompt = f"""Extract structured knowledge from this text. Return JSON:
{{
  "entities": [
    {{"name": "...", "node_type": "...", "summary": "..."}}
  ],
  "relationships": [
    {{"source": "...", "target": "...", "type": "...", "reason": "..."}}
  ]
}}

Valid node_types: {json.dumps(NODE_TYPES)}
Valid relationship types: {json.dumps(RELATIONSHIP_TYPES)}

Text:
{text[:8000]}"""

    req_body = json.dumps({
        "model": EXTRACTION_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=req_body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    content = data["choices"][0]["message"]["content"]
    extracted = json.loads(content)

    adapter = _get_adapter()
    entities = extracted.get("entities", [])
    relationships = extracted.get("relationships", [])

    ent_count = adapter.write_entities(entities, "ingest", "extraction", DEFAULT_PROJECT)
    rel_count = adapter.write_relationships(relationships, "ingest", "extraction", DEFAULT_PROJECT)

    for ent in entities:
        name = ent.get("name", "")
        summary = ent.get("summary", "")
        if name:
            adapter.vector_upsert(
                f"{DEFAULT_PROJECT}:{name}", f"{name}: {summary}",
                {"name": name, "node_type": ent.get("node_type", "Concept"),
                 "source": "extraction"}, DEFAULT_PROJECT,
            )

    return {
        "entities_written": ent_count, "relationships_written": rel_count,
        "entities_extracted": len(entities), "relationships_extracted": len(relationships),
    }


@mcp.tool()
async def ingest_knowledge(
    text: str,
    project: str = None,
) -> dict:
    """Ingest free text into the knowledge graph using LLM extraction.
    Async — returns a job_id. Use check_ingestion_status to poll.

    Args:
        text: Text to extract knowledge from (max 10KB)
        project: Project ID (default: server default)
    """
    if len(text) > MAX_INGEST_LENGTH:
        return {"error": f"Text too long ({len(text)} bytes). Max: {MAX_INGEST_LENGTH}"}
    if not text.strip():
        return {"error": "Empty text"}

    _ensure_worker()
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        if len(_jobs) >= MAX_JOBS:
            completed = [k for k, v in _jobs.items()
                         if v["status"] in ("completed", "failed")]
            for k in completed[:len(_jobs) - MAX_JOBS + 1]:
                del _jobs[k]
        _jobs[job_id] = {
            "status": "queued", "created": time.time(),
            "text_len": len(text), "result": None, "error": None,
        }
    _job_queue.put((job_id, text))
    return {"job_id": job_id, "status": "queued",
            "message": "Ingestion queued. Use check_ingestion_status to poll."}


@mcp.tool()
async def check_ingestion_status(job_id: str) -> dict:
    """Check the status of an async ingestion job.

    Args:
        job_id: The job ID returned by ingest_knowledge
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return {"error": f"Unknown job_id: {job_id}"}
    response = {
        "job_id": job_id, "status": job["status"],
        "text_len": job["text_len"],
        "elapsed_s": int(time.time() - job["created"]),
    }
    if job["status"] == "completed":
        response["result"] = job["result"]
        response["duration_ms"] = job.get("duration_ms")
    elif job["status"] == "failed":
        response["error"] = job["error"]
        response["duration_ms"] = job.get("duration_ms")
    return response


@mcp.tool()
async def health_check() -> dict:
    """Check if the knowledge graph backends are healthy."""
    adapter = _get_adapter()
    start = time.time()
    healthy = adapter.is_healthy()
    return {
        "healthy": healthy, "backend": DEFAULT_BACKEND,
        "project": DEFAULT_PROJECT,
        "extraction_available": bool(OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")),
        "duration_ms": int((time.time() - start) * 1000),
    }


# --- Main ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merkraum MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"],
                        default="http",
                        help="Transport mode (default: http)")
    parser.add_argument("--port", type=int, default=HTTP_PORT,
                        help=f"HTTP port (default: {HTTP_PORT})")
    parser.add_argument("--host", default=HTTP_HOST,
                        help=f"HTTP host (default: {HTTP_HOST})")
    args = parser.parse_args()

    # Load secrets from AWS Secrets Manager
    _load_secrets()

    # OPENAI_API_KEY is read from os.environ at extraction time

    tool_count = len(asyncio.run(mcp.list_tools()))
    logger.info("Merkraum MCP Server starting")
    logger.info("Backend: %s | Project: %s | Tools: %d",
                DEFAULT_BACKEND, DEFAULT_PROJECT, tool_count)
    logger.info("Cognito pool: %s | Auth domain: %s", COGNITO_POOL_ID, COGNITO_AUTH_DOMAIN)
    logger.info("MCP base URL: %s", MCP_BASE_URL)
    logger.info("LLM extraction: %s",
                "available" if OPENAI_API_KEY else "unavailable (set OPENAI_API_KEY)")

    # Pre-warm JWKS cache
    try:
        _fetch_jwks()
        logger.info("JWKS pre-warmed successfully")
    except Exception as e:
        logger.warning("JWKS pre-warm failed (will retry on first request): %s", e)

    if args.transport == "http":
        logger.info("Transport: HTTP on %s:%d", args.host, args.port)
        mcp.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
            path="/mcp",
            stateless_http=True,
            json_response=True,
        )
    else:
        logger.info("Transport: stdio")
        mcp.run(transport="stdio")
