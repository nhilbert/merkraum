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
v2.1 — Z1439 (2026-03-12). Default project = Cognito sub (Norman directive). Auto-provision on first MCP connection.
"""

import os
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
from urllib.parse import urlencode, urlparse

from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response

from fastmcp import FastMCP
from fastmcp.server.auth import AccessToken, TokenVerifier

from merkraum_acl import is_project_allowed
from fastmcp.server.dependencies import get_access_token

from merkraum_backend import (
    create_adapter, BackendAdapter, NODE_TYPES, RELATIONSHIP_TYPES, TIER_LIMITS,
    VSM_LEVELS, KNOWLEDGE_TYPES,
)
from merkraum_llm import llm_extract, get_provider_info

# --- Configuration ---

_dev_mode_env = os.environ.get("DEV_MODE")
if _dev_mode_env is None:
    _app_env = (os.environ.get("APP_ENV") or os.environ.get("FLASK_ENV") or "").strip().lower()
    DEV_MODE = _app_env not in {"prod", "production"}
else:
    DEV_MODE = _dev_mode_env.lower() in ("true", "1", "yes")

COGNITO_REGION = os.environ.get("COGNITO_AWS_REGION", "")
COGNITO_POOL_ID = os.environ.get("COGNITO_USER_POOL_ID", "")
COGNITO_ISSUER = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_POOL_ID}"
COGNITO_JWKS_URL = f"{COGNITO_ISSUER}/.well-known/jwks.json"
COGNITO_APP_CLIENT_ID = os.environ.get("MCP_COGNITO_CLIENT_ID", "")
COGNITO_APP_CLIENT_SECRET = os.environ.get("MCP_COGNITO_CLIENT_SECRET", "")
COGNITO_AUTH_DOMAIN = os.environ.get("COGNITO_AUTH_DOMAIN", "")
COGNITO_TOKEN_URL = f"{COGNITO_AUTH_DOMAIN}/oauth2/token"
COGNITO_AUTHORIZE_URL = f"{COGNITO_AUTH_DOMAIN}/oauth2/authorize"
COGNITO_ALLOWED_SCOPES = {"openid", "email", "phone", "profile"}

MCP_ALLOWED_CLIENT_IDS = {
    x.strip() for x in os.environ.get("MCP_ALLOWED_CLIENT_IDS", "").split(",") if x.strip()
}
if not MCP_ALLOWED_CLIENT_IDS and COGNITO_APP_CLIENT_ID:
    MCP_ALLOWED_CLIENT_IDS = {COGNITO_APP_CLIENT_ID}

MCP_BASE_URL = os.environ.get("MCP_BASE_URL", "")
MCP_ENABLE_DYNAMIC_CLIENT_REGISTRATION = os.environ.get(
    "MCP_ENABLE_DYNAMIC_CLIENT_REGISTRATION", "false"
).lower() in ("true", "1", "yes")

DEFAULT_BACKEND = os.environ.get("MERKRAUM_BACKEND", "neo4j_qdrant")
DEFAULT_PROJECT = os.environ.get("MERKRAUM_PROJECT", "default")
HTTP_PORT = int(os.environ.get("MERKRAUM_MCP_PORT", "8090"))
HTTP_HOST = os.environ.get("MERKRAUM_MCP_HOST", "127.0.0.1")

# LLM for knowledge extraction — now configurable via merkraum_llm module
# Legacy OPENAI_API_KEY still read for backward compatibility when provider=openai
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# --- Logging ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("merkraum-mcp")


def _validate_https_url(url: str, label: str):
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{label} must be a valid https URL")


def _validate_auth_config():
    required = {
        "COGNITO_AWS_REGION": COGNITO_REGION,
        "COGNITO_USER_POOL_ID": COGNITO_POOL_ID,
        "COGNITO_AUTH_DOMAIN": COGNITO_AUTH_DOMAIN,
        "MCP_BASE_URL": MCP_BASE_URL,
    }
    missing = [name for name, value in required.items() if not value]

    if missing and not DEV_MODE:
        raise RuntimeError(
            "Missing required auth configuration for MCP server: " + ", ".join(missing)
        )

    if DEV_MODE and missing:
        logger.warning("DEV_MODE=true, allowing missing auth configuration: %s", ", ".join(missing))

    if COGNITO_AUTH_DOMAIN:
        _validate_https_url(COGNITO_AUTH_DOMAIN, "COGNITO_AUTH_DOMAIN")
    if COGNITO_JWKS_URL:
        _validate_https_url(COGNITO_JWKS_URL, "COGNITO_JWKS_URL")
    if MCP_BASE_URL:
        _validate_https_url(MCP_BASE_URL, "MCP_BASE_URL")

    if not MCP_ALLOWED_CLIENT_IDS and not DEV_MODE:
        raise RuntimeError("MCP_ALLOWED_CLIENT_IDS (or MCP_COGNITO_CLIENT_ID) is required in non-dev mode")

    logger.info(
        "Auth config loaded: issuer=%s auth_domain=%s base_url=%s allowed_clients=%d dev_mode=%s",
        COGNITO_ISSUER,
        COGNITO_AUTH_DOMAIN,
        MCP_BASE_URL,
        len(MCP_ALLOWED_CLIENT_IDS),
        DEV_MODE,
    )


_validate_auth_config()

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
            "verify_aud": False,
            "verify_exp": True,
            "verify_iss": True,
        },
    )

    token_use = claims.get("token_use")
    if token_use == "id":
        aud = claims.get("aud")
        if aud not in MCP_ALLOWED_CLIENT_IDS:
            raise ValueError("Token aud is not an allowed client id")
    else:
        client_id = claims.get("client_id")
        if client_id not in MCP_ALLOWED_CLIENT_IDS:
            raise ValueError("Token client_id is not an allowed client id")

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


# --- PAT Validation ---

PAT_PREFIX = "mk_pat_"

def _validate_pat(token: str) -> dict | None:
    """Validate a Personal Access Token against Neo4j. Returns metadata or None."""
    import hashlib
    adapter = _get_adapter()
    driver = getattr(adapter, "_driver", None)
    if not driver:
        logger.warning("PAT validation: no Neo4j driver available")
        return None
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (t:PersonalAccessToken {token_hash: $hash})
            WHERE t.revoked = false
              AND (t.expires_at IS NULL OR datetime(t.expires_at) > datetime())
            SET t.last_used_at = toString(datetime())
            RETURN t.token_prefix AS token_prefix,
                   t.name AS name,
                   t.owner_id AS owner_id,
                   t.scopes AS scopes,
                   t.projects AS projects,
                   t.all_projects AS all_projects,
                   t.expires_at AS expires_at
            """,
            hash=token_hash,
        ).single()
        if not result:
            return None
        return dict(result)


# --- Token Verifier ---

class CognitoTokenVerifier(TokenVerifier):
    """Validates Cognito JWT access tokens and Personal Access Tokens (mk_pat_)."""

    async def verify_token(self, token: str) -> AccessToken | None:
        """Called by FastMCP for every authenticated request."""
        if not token:
            return None

        # PAT tokens: validate against Neo4j
        if token.startswith(PAT_PREFIX):
            try:
                pat_meta = await asyncio.get_event_loop().run_in_executor(
                    None, _validate_pat, token
                )
                if not pat_meta:
                    logger.warning("PAT validation failed: token not found or expired")
                    return None
                scopes = pat_meta.get("scopes") or []
                if isinstance(scopes, str):
                    scopes = scopes.split(",")
                logger.info("PAT authenticated: owner=%s name=%s",
                            pat_meta.get("owner_id", "?"), pat_meta.get("name", "?"))
                return AccessToken(
                    token=token,
                    client_id=f"pat:{pat_meta.get('owner_id', 'unknown')}",
                    scopes=scopes,
                    expires_at=None,
                    claims={
                        "sub": pat_meta.get("owner_id", "unknown"),
                        "token_type": "pat",
                        "pat_name": pat_meta.get("name", ""),
                        "projects": pat_meta.get("projects") or [],
                        "all_projects": pat_meta.get("all_projects", False),
                    },
                )
            except Exception as e:
                logger.warning("PAT validation error: %s", e)
                return None

        # Cognito JWT tokens
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
        return JSONResponse({"error": "token_proxy_error", "detail": "token proxy failed"}, status_code=502)

    return Response(
        content=resp_body,
        status_code=resp_status,
        media_type=resp_headers.get("Content-Type", "application/json"),
    )


# --- Dynamic Client Registration Stub ---

@mcp.custom_route("/register", methods=["POST"])
async def register_client(request: Request) -> JSONResponse:
    """Return pre-created Cognito app client for MCP OAuth bootstrap."""
    if not MCP_ENABLE_DYNAMIC_CLIENT_REGISTRATION:
        return JSONResponse({"error": "registration_disabled"}, status_code=404)

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



def _normalize_pat_projects(raw_projects) -> list[str] | None:
    if raw_projects is None:
        return None
    if isinstance(raw_projects, list):
        return [str(p).strip() for p in raw_projects if str(p).strip()]
    if isinstance(raw_projects, str):
        return [p.strip() for p in raw_projects.split(",") if p.strip()]
    return None


def _get_auth_context() -> dict:
    """Extract auth context from FastMCP access token (JWT or PAT)."""
    token = get_access_token()
    if token is None:
        return {
            "user_id": None,
            "groups": set(),
            "token_type": None,
            "scopes": set(),
            "pat_projects": None,
            "pat_all_projects": None,
        }

    claims = getattr(token, "claims", {}) or {}
    token_type = claims.get("token_type") or "jwt"
    scopes = set(getattr(token, "scopes", []) or [])
    if not scopes and claims.get("scope"):
        scopes = {s for s in str(claims.get("scope", "")).split() if s}
    context = {
        "user_id": claims.get("sub"),
        "groups": set(claims.get("cognito:groups", []) or []),
        "token_type": token_type,
        "scopes": scopes,
        "pat_projects": None,
        "pat_all_projects": None,
    }
    if token_type == "pat":
        context["pat_projects"] = _normalize_pat_projects(claims.get("projects"))
        context["pat_all_projects"] = bool(claims.get("all_projects", False))
    return context


def _get_authenticated_user() -> tuple[str | None, set[str]]:
    """Extract user_id and groups from the auth context."""
    auth_ctx = _get_auth_context()
    return auth_ctx["user_id"], auth_ctx["groups"]


def _require_pat_scope(required_scope: str | None, auth_ctx: dict) -> str | None:
    """Enforce PAT scopes only; Cognito/JWT tokens are treated as fully scoped."""
    if not required_scope:
        return None
    if auth_ctx.get("token_type") != "pat":
        return None
    scopes = auth_ctx.get("scopes", set()) or set()
    if "admin" in scopes or required_scope in scopes:
        return None
    return f"Token lacks required scope: {required_scope}"


def _resolve_project(project: str | None, auth_ctx: dict | None = None) -> str:
    """Resolve project ID: explicit value > authenticated user's Cognito sub > DEFAULT_PROJECT.

    Norman's directive (Z1439): When connecting via MCP, the default project
    should be the user's Cognito sub. Users don't need to configure or pass
    a project_id — the auth already tells us who they are.
    """
    if project and project.strip():
        return project.strip()
    if auth_ctx is None:
        auth_ctx = _get_auth_context()
    user_id = auth_ctx.get("user_id")
    if user_id:
        return user_id
    return DEFAULT_PROJECT


def _auto_provision_project(project: str, user_id: str) -> None:
    """Auto-create a personal project if it matches the user's sub and doesn't exist yet.

    Mirrors the REST API auto-provision logic from Z1437. Only triggers when
    project == user_id (personal namespace), avoiding unintended creation.
    """
    if project != user_id:
        return
    adapter = _get_adapter()
    try:
        existing = adapter.list_projects(owner=user_id)
        if any(p.get("project_id") == project for p in existing):
            return
    except Exception:
        return
    try:
        adapter.create_project(
            project_id=user_id,
            name="My Knowledge Space",
            owner=user_id,
            description="Auto-created on first MCP connection",
            tier="free",
        )
        logger.info("Auto-provisioned project %s for user", user_id[:8])
    except (ValueError, Exception) as e:
        logger.debug("Auto-provision skipped (may already exist): %s", e)


def _check_project_access(project: str,
                          auth_ctx: dict | None = None) -> tuple[str | None, set[str], str | None]:
    """Check project ACL for the authenticated user. Auto-provisions if needed.

    Returns (user_id, groups, error_message).
    error_message is None if access is allowed.
    """
    if auth_ctx is None:
        auth_ctx = _get_auth_context()
    user_id = auth_ctx.get("user_id")
    groups = auth_ctx.get("groups", set()) or set()
    pat_projects = auth_ctx.get("pat_projects")
    pat_all_projects = auth_ctx.get("pat_all_projects")

    # Auto-provision personal project on first access
    if user_id and project == user_id:
        _auto_provision_project(project, user_id)

    if not is_project_allowed(
        project,
        user_id,
        groups,
        pat_projects=pat_projects,
        pat_all_projects=pat_all_projects,
    ):
        return user_id, groups, f"Forbidden: no access to project '{project}'"
    return user_id, groups, None

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
        project: Project ID (default: your personal space)
    """
    auth_ctx = _get_auth_context()
    scope_err = _require_pat_scope("search", auth_ctx)
    if scope_err:
        return {"error": scope_err}
    project = _resolve_project(project, auth_ctx)
    _uid, _grp, _err = _check_project_access(project, auth_ctx)
    if _err:
        return {"error": _err}
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
        project: Project ID (default: your personal space)
    """
    auth_ctx = _get_auth_context()
    scope_err = _require_pat_scope("read", auth_ctx)
    if scope_err:
        return {"error": scope_err}
    project = _resolve_project(project, auth_ctx)
    _uid, _grp, _err = _check_project_access(project, auth_ctx)
    if _err:
        return {"error": _err}
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
    knowledge_type: str = None,
    project: str = None,
) -> dict:
    """List beliefs in the knowledge graph.

    Args:
        status: Filter — active, uncertain, superseded, contradicted
        knowledge_type: Filter by knowledge type (fact=permanent truths,
            state=temporary current facts, rule=policies/procedures,
            belief=subjective assessments, memory=episodic events). None = all.
        project: Project ID (default: your personal space)
    """
    auth_ctx = _get_auth_context()
    scope_err = _require_pat_scope("read", auth_ctx)
    if scope_err:
        return {"error": scope_err}
    project = _resolve_project(project, auth_ctx)
    _uid, _grp, _err = _check_project_access(project, auth_ctx)
    if _err:
        return {"error": _err}
    if status not in ("active", "uncertain", "superseded", "contradicted"):
        status = "active"
    if knowledge_type and knowledge_type not in KNOWLEDGE_TYPES:
        return {"error": f"Unknown knowledge_type: {knowledge_type}. Valid: {KNOWLEDGE_TYPES}"}
    adapter = _get_adapter()
    start = time.time()
    try:
        beliefs = await _run_sync(adapter.get_beliefs, project, status,
                                  knowledge_type)
        audit_log("list_beliefs", "authed",
                  {"status": status, "knowledge_type": knowledge_type,
                   "project": project},
                  int((time.time() - start) * 1000), "ok")
        return {"beliefs": beliefs, "count": len(beliefs), "status_filter": status,
                "knowledge_type_filter": knowledge_type,
                "duration_ms": int((time.time() - start) * 1000)}
    except Exception as e:
        audit_log("list_beliefs", "authed",
                  {"status": status}, int((time.time() - start) * 1000), "error", str(e))
        return {"error": str(e)}


@mcp.tool()
async def get_graph_stats(project: str = None) -> dict:
    """Get knowledge graph statistics: node counts by type, edge counts, totals.

    Args:
        project: Project ID (default: your personal space)
    """
    auth_ctx = _get_auth_context()
    scope_err = _require_pat_scope("read", auth_ctx)
    if scope_err:
        return {"error": scope_err}
    project = _resolve_project(project, auth_ctx)
    _uid, _grp, _err = _check_project_access(project, auth_ctx)
    if _err:
        return {"error": _err}
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
    vsm_level: str = None,
    knowledge_type: str = None,
    limit: int = 50,
    project: str = None,
) -> dict:
    """Query entities in the knowledge graph, optionally filtered by type, VSM level, and knowledge type.

    Args:
        node_type: Filter by type (Person, Organization, Project, Concept,
            Regulation, Event, Belief, Artifact, Interview, Quote). None = all.
        vsm_level: Filter by VSM system level (S1=operational, S2=coordination,
            S3=control, S4=strategic, S5=identity). None = all.
        knowledge_type: Filter by knowledge type (fact=permanent truths,
            state=temporary current facts, rule=policies/procedures,
            belief=subjective assessments, memory=episodic events). None = all.
        limit: Max results (1-200, default 50)
        project: Project ID (default: your personal space)
    """
    auth_ctx = _get_auth_context()
    scope_err = _require_pat_scope("read", auth_ctx)
    if scope_err:
        return {"error": scope_err}
    project = _resolve_project(project, auth_ctx)
    _uid, _grp, _err = _check_project_access(project, auth_ctx)
    if _err:
        return {"error": _err}
    if node_type and node_type not in NODE_TYPES:
        return {"error": f"Unknown node_type: {node_type}. Valid: {NODE_TYPES}"}
    if vsm_level and vsm_level not in VSM_LEVELS:
        return {"error": f"Unknown vsm_level: {vsm_level}. Valid: {VSM_LEVELS}"}
    if knowledge_type and knowledge_type not in KNOWLEDGE_TYPES:
        return {"error": f"Unknown knowledge_type: {knowledge_type}. Valid: {KNOWLEDGE_TYPES}"}
    limit = max(1, min(200, limit))
    adapter = _get_adapter()
    start = time.time()
    try:
        nodes = await _run_sync(adapter.query_nodes, node_type, project, limit,
                                vsm_level, False, knowledge_type)
        audit_log("query_nodes", "authed",
                  {"node_type": node_type, "vsm_level": vsm_level,
                   "knowledge_type": knowledge_type,
                   "limit": limit, "project": project},
                  int((time.time() - start) * 1000), "ok")
        return {"nodes": nodes, "count": len(nodes), "node_type_filter": node_type,
                "vsm_level_filter": vsm_level,
                "knowledge_type_filter": knowledge_type,
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
    valid_until: str = None,
    vsm_level: str = None,
    knowledge_type: str = None,
    project: str = None,
) -> dict:
    """Add a single entity to the knowledge graph.

    Args:
        name: Entity name
        summary: Brief description
        node_type: One of Person, Organization, Project, Concept, Regulation,
            Event, Belief, Artifact, Interview, Quote
        confidence: For Beliefs only (0.0-1.0, default 0.7)
        valid_until: ISO 8601 date when this knowledge expires (optional, null = no expiration)
        vsm_level: VSM system level (S1=operational, S2=coordination, S3=control,
            S4=strategic, S5=identity). Determines default TTL if valid_until not set.
        knowledge_type: Epistemological classification (fact=permanent truths,
            state=temporary current facts, rule=policies/procedures,
            belief=subjective assessments, memory=episodic events). Optional.
        project: Project ID (default: your personal space)
    """
    auth_ctx = _get_auth_context()
    scope_err = _require_pat_scope("write", auth_ctx)
    if scope_err:
        return {"error": scope_err}
    project = _resolve_project(project, auth_ctx)
    _uid, _grp, _err = _check_project_access(project, auth_ctx)
    if _err:
        return {"error": _err}
    if node_type not in NODE_TYPES:
        return {"error": f"Unknown node_type: {node_type}. Valid: {NODE_TYPES}"}
    if vsm_level and vsm_level not in VSM_LEVELS:
        return {"error": f"Unknown vsm_level: {vsm_level}. Valid: {VSM_LEVELS}"}
    if knowledge_type and knowledge_type not in KNOWLEDGE_TYPES:
        return {"error": f"Unknown knowledge_type: {knowledge_type}. Valid: {KNOWLEDGE_TYPES}"}
    adapter = _get_adapter()
    entity = {"name": name, "summary": summary, "node_type": node_type}
    if node_type == "Belief":
        entity["confidence"] = max(0.0, min(1.0, confidence))
    if valid_until:
        entity["valid_until"] = valid_until
    if vsm_level:
        entity["vsm_level"] = vsm_level
    if knowledge_type:
        entity["knowledge_type"] = knowledge_type
    start = time.time()
    try:
        written = await _run_sync(
            adapter.write_entities, [entity], "manual", "user", project
        )
        audit_log("add_knowledge", "authed",
                  {"name": name, "node_type": node_type,
                   "knowledge_type": knowledge_type, "project": project},
                  int((time.time() - start) * 1000), "ok")
        return {"created": written > 0, "name": name, "node_type": node_type,
                "knowledge_type": knowledge_type,
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
        project: Project ID (default: your personal space)
    """
    auth_ctx = _get_auth_context()
    scope_err = _require_pat_scope("write", auth_ctx)
    if scope_err:
        return {"error": scope_err}
    project = _resolve_project(project, auth_ctx)
    _uid, _grp, _err = _check_project_access(project, auth_ctx)
    if _err:
        return {"error": _err}
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
    valid_until: str = None,
    project: str = None,
) -> dict:
    """Update an existing belief's confidence, status, summary, or temporal validity.

    Args:
        name: Belief name (must already exist)
        confidence: New confidence score (0.0-1.0). None = no change.
        status: New status — active, uncertain, contradicted, or superseded. None = no change.
        summary: New summary text. None = no change.
        valid_until: ISO 8601 date when this belief expires (e.g. "2026-06-30"). None = no change.
        project: Project ID (default: your personal space)
    """
    auth_ctx = _get_auth_context()
    scope_err = _require_pat_scope("write", auth_ctx)
    if scope_err:
        return {"error": scope_err}
    project = _resolve_project(project, auth_ctx)
    _uid, _grp, _err = _check_project_access(project, auth_ctx)
    if _err:
        return {"error": _err}
    adapter = _get_adapter()
    start = time.time()
    try:
        result = await _run_sync(
            adapter.update_belief, name, project,
            confidence=confidence, status=status, summary=summary,
            valid_until=valid_until, actor=_uid or "mcp",
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
async def consolidate_beliefs(
    belief_a: str,
    belief_b: str,
    resolution: str,
    new_name: str = None,
    project: str = None,
) -> dict:
    """Resolve a false contradiction between two beliefs with a user explanation.

    When two beliefs are marked as contradicting but actually represent
    complementary information (e.g., different markets, different time periods),
    use this tool to consolidate them with a free-text resolution.

    Creates a synthesis belief from the resolution, marks both originals as
    'consolidated', removes the CONTRADICTS relationship, and adds SUPERSEDES
    links from the synthesis to both originals.

    Args:
        belief_a: Name of the first contradicting belief
        belief_b: Name of the second contradicting belief
        resolution: Free-text explanation of the real situation — why the beliefs
            are not actually contradictory, or what the synthesized truth is
        new_name: Optional name for the new synthesis belief (auto-generated if omitted)
        project: Project ID (default: your personal space)
    """
    auth_ctx = _get_auth_context()
    scope_err = _require_pat_scope("write", auth_ctx)
    if scope_err:
        return {"error": scope_err}
    project = _resolve_project(project, auth_ctx)
    _uid, _grp, _err = _check_project_access(project, auth_ctx)
    if _err:
        return {"error": _err}
    adapter = _get_adapter()
    start = time.time()
    try:
        result = await _run_sync(
            adapter.consolidate_beliefs,
            belief_a_name=belief_a,
            belief_b_name=belief_b,
            resolution_text=resolution,
            project_id=project,
            new_name=new_name,
            actor=_uid or "mcp",
        )
        result["duration_ms"] = int((time.time() - start) * 1000)
        audit_log("consolidate_beliefs", "authed",
                  {"belief_a": belief_a, "belief_b": belief_b, "project": project},
                  int((time.time() - start) * 1000),
                  "ok" if result.get("ok") else "error")
        return result
    except Exception as e:
        audit_log("consolidate_beliefs", "authed",
                  {"belief_a": belief_a, "belief_b": belief_b},
                  int((time.time() - start) * 1000), "error", str(e))
        return {"error": str(e)}


@mcp.tool()
async def expire_nodes(
    dry_run: bool = True,
    project: str = None,
) -> dict:
    """Enforce managed forgetting: expire nodes past their valid_until date.

    Finds all nodes whose valid_until timestamp has passed and marks them
    as expired (sets expired_at, deactivates Beliefs). Non-destructive —
    nodes remain in the graph with full audit trail.

    Use dry_run=True (default) to preview what would be expired.
    Set dry_run=False to actually expire the nodes.

    Args:
        dry_run: If True, only show what would be expired without changing anything.
        project: Project ID (uses authenticated user's default if omitted).
    """
    start = time.time()
    auth = await _require_auth()
    if isinstance(auth, dict) and "error" in auth:
        return auth
    project = project or auth.get("project_id", "default")
    try:
        result = adapter.expire_nodes(
            project_id=project, dry_run=dry_run,
            actor=auth.get("sub", "mcp"),
        )
        audit_log("expire_nodes", "authed",
                  {"project": project, "dry_run": dry_run},
                  int((time.time() - start) * 1000),
                  "ok" if result.get("total", 0) >= 0 else "error")
        return result
    except Exception as e:
        audit_log("expire_nodes", "authed",
                  {"project": project, "dry_run": dry_run},
                  int((time.time() - start) * 1000), "error", str(e))
        return {"error": str(e)}


@mcp.tool()
async def renew_node(
    name: str,
    extend_days: int = None,
    new_valid_until: str = None,
    node_type: str = None,
    project: str = None,
) -> dict:
    """Renew a node's validity — extend or reset its valid_until date.

    Use this when a node is expiring but is still relevant. Either provide
    extend_days (adds N days from now) or new_valid_until (explicit ISO date).
    Also clears expired_at and reactivates the node if it was previously expired.

    Args:
        name: The node name to renew.
        extend_days: Number of days to extend from now (1-3650).
        new_valid_until: Explicit ISO datetime for the new valid_until.
        node_type: Optional node type to narrow the match (e.g. 'Concept', 'Belief').
        project: Project ID (uses authenticated user's default if omitted).
    """
    start = time.time()
    auth = await _require_auth()
    if isinstance(auth, dict) and "error" in auth:
        return auth
    project = project or auth.get("project_id", "default")
    try:
        result = adapter.renew_node(
            name=name, project_id=project, extend_days=extend_days,
            new_valid_until=new_valid_until, node_type=node_type,
            actor=auth.get("sub", "mcp"),
        )
        audit_log("renew_node", "authed",
                  {"name": name, "project": project, "extend_days": extend_days},
                  int((time.time() - start) * 1000),
                  "ok" if result.get("renewed") else "error")
        return result
    except Exception as e:
        audit_log("renew_node", "authed",
                  {"name": name, "project": project},
                  int((time.time() - start) * 1000), "error", str(e))
        return {"error": str(e)}


# --- Certainty Management (SUP-163) ---

@mcp.tool()
async def certainty_decay(
    dry_run: bool = True,
    project: str = None,
) -> dict:
    """Apply time-based confidence decay to active beliefs.

    Confidence decays based on knowledge_type: 'state' decays fastest,
    'memory' slowest, 'fact' is exempt. Use dry_run=True (default) to
    preview which beliefs would be affected and by how much.

    Args:
        dry_run: If True, only show what would change without applying. Default True.
        project: Project ID (uses authenticated user's default if omitted).
    """
    start = time.time()
    auth = await _require_auth()
    if isinstance(auth, dict) and "error" in auth:
        return auth
    project = project or auth.get("project_id", "default")
    try:
        result = adapter.apply_confidence_decay(
            project_id=project, dry_run=dry_run,
            actor=auth.get("sub", "mcp"),
        )
        audit_log("certainty_decay", "authed",
                  {"project": project, "dry_run": dry_run},
                  int((time.time() - start) * 1000),
                  "ok" if result.get("total", 0) >= 0 else "error")
        return result
    except Exception as e:
        audit_log("certainty_decay", "authed",
                  {"project": project, "dry_run": dry_run},
                  int((time.time() - start) * 1000), "error", str(e))
        return {"error": str(e)}


@mcp.tool()
async def certainty_review(
    limit: int = 50,
    project: str = None,
) -> dict:
    """Get beliefs needing review based on certainty governance rules.

    Surfaces beliefs that need attention: stale (not updated in 30+ days),
    low confidence (<=0.3), type-confidence mismatches (e.g. 'fact' at low
    confidence), approaching expiry (within 7 days), and unclassified
    (no knowledge_type set).

    Args:
        limit: Maximum items per category (1-200). Default 50.
        project: Project ID (uses authenticated user's default if omitted).
    """
    start = time.time()
    auth = await _require_auth()
    if isinstance(auth, dict) and "error" in auth:
        return auth
    project = project or auth.get("project_id", "default")
    limit = max(1, min(limit, 200))
    try:
        result = adapter.get_certainty_review_queue(
            project_id=project, limit=limit,
        )
        audit_log("certainty_review", "authed",
                  {"project": project, "limit": limit},
                  int((time.time() - start) * 1000), "ok")
        return result
    except Exception as e:
        audit_log("certainty_review", "authed",
                  {"project": project},
                  int((time.time() - start) * 1000), "error", str(e))
        return {"error": str(e)}


@mcp.tool()
async def certainty_stats(
    project: str = None,
) -> dict:
    """Confidence distribution statistics for certainty governance.

    Returns confidence histogram, average confidence per knowledge_type,
    staleness distribution (fresh/aging/stale), and a governance health
    summary. Use this to understand the overall epistemic health of a
    knowledge graph.

    Args:
        project: Project ID (uses authenticated user's default if omitted).
    """
    start = time.time()
    auth = await _require_auth()
    if isinstance(auth, dict) and "error" in auth:
        return auth
    project = project or auth.get("project_id", "default")
    try:
        result = adapter.get_certainty_stats(project_id=project)
        audit_log("certainty_stats", "authed",
                  {"project": project},
                  int((time.time() - start) * 1000), "ok")
        return result
    except Exception as e:
        audit_log("certainty_stats", "authed",
                  {"project": project},
                  int((time.time() - start) * 1000), "error", str(e))
        return {"error": str(e)}


@mcp.tool()
async def get_usage(
    tier: str = "free",
    project: str = None,
) -> dict:
    """Get usage metrics and tier limits.

    Args:
        tier: Pricing tier — free (100), pro (1000), team (5000), enterprise (50000)
        project: Project ID (default: your personal space)
    """
    auth_ctx = _get_auth_context()
    scope_err = _require_pat_scope("read", auth_ctx)
    if scope_err:
        return {"error": scope_err}
    project = _resolve_project(project, auth_ctx)
    _uid, _grp, _err = _check_project_access(project, auth_ctx)
    if _err:
        return {"error": _err}
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
            job_id, text, project = _job_queue.get()
            _run_ingestion_job(job_id, text, project)
            _job_queue.task_done()

    t = threading.Thread(target=_worker, daemon=True, name="ingestion-worker")
    t.start()


def _run_ingestion_job(job_id: str, text: str, project: str):
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    start = time.time()
    try:
        result = _extract_and_write(text, project)
        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = result
            _jobs[job_id]["duration_ms"] = int((time.time() - start) * 1000)
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["duration_ms"] = int((time.time() - start) * 1000)


def _extract_and_write(text: str, project: str = None) -> dict:
    """Extract entities/relationships from text via LLM, write to graph.

    Uses merkraum_llm module for provider-agnostic extraction (Bedrock or OpenAI).
    """
    provider_info = get_provider_info()
    if provider_info["provider"] == "openai":
        api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return {"error": "OPENAI_API_KEY not set (or switch to MERKRAUM_LLM_PROVIDER=bedrock)"}
    else:
        api_key = None

    system_prompt = f"""Extract structured knowledge from this text. Return JSON:
{{
  "entities": [
    {{"name": "...", "node_type": "...", "summary": "...", "vsm_level": "S1|S2|S3|S4|S5|null"}}
  ],
  "relationships": [
    {{"source": "...", "target": "...", "type": "...", "reason": "..."}}
  ]
}}

Valid node_types: {json.dumps(NODE_TYPES)}
Valid relationship types: {json.dumps(RELATIONSHIP_TYPES)}
VSM levels (optional, classify by organizational function):
- S1 (Operational): task data, current values, in-progress notes
- S2 (Coordination): rules, procedures, process descriptions
- S3 (Control): metrics, quality assessments, priorities
- S4 (Strategic): environmental models, competitive intel, research
- S5 (Identity): core values, policies, identity claims"""

    user_prompt = f"Extract structured knowledge from this text:\n\n{text[:8000]}"

    extracted = llm_extract(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
        api_key=api_key,
    )

    adapter = _get_adapter()
    entities = extracted.get("entities", [])
    relationships = extracted.get("relationships", [])

    project = project or DEFAULT_PROJECT
    ent_count = adapter.write_entities(entities, "ingest", "extraction", project)
    rel_count = adapter.write_relationships(relationships, "ingest", "extraction", project)

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
        project: Project ID (default: your personal space)
    """
    if len(text) > MAX_INGEST_LENGTH:
        return {"error": f"Text too long ({len(text)} bytes). Max: {MAX_INGEST_LENGTH}"}
    if not text.strip():
        return {"error": "Empty text"}

    # Resolve project now (in auth context) before dispatching to background thread
    auth_ctx = _get_auth_context()
    scope_err = _require_pat_scope("ingest", auth_ctx)
    if scope_err:
        return {"error": scope_err}
    project = _resolve_project(project, auth_ctx)
    _uid, _grp, _err = _check_project_access(project, auth_ctx)
    if _err:
        return {"error": _err}

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
            "owner_id": auth_ctx.get("user_id"),
            "project": project,
        }
    _job_queue.put((job_id, text, project))
    return {"job_id": job_id, "status": "queued",
            "message": "Ingestion queued. Use check_ingestion_status to poll."}


@mcp.tool()
async def check_ingestion_status(job_id: str) -> dict:
    """Check the status of an async ingestion job.

    Args:
        job_id: The job ID returned by ingest_knowledge
    """
    auth_ctx = _get_auth_context()
    scope_err = _require_pat_scope("read", auth_ctx)
    if scope_err:
        return {"error": scope_err}

    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return {"error": f"Unknown job_id: {job_id}"}
    owner_id = job.get("owner_id")
    user_id = auth_ctx.get("user_id")
    if owner_id:
        if user_id is None:
            return {"error": "Forbidden: authentication required for this job"}
        if owner_id != user_id:
            return {"error": "Forbidden: no access to this ingestion job"}
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
async def get_history(
    entity_name: str = None,
    operation_type: str = None,
    since: str = None,
    until: str = None,
    limit: int = 20,
    project: str = None,
) -> dict:
    """Retrieve the audit trail of mutations to the knowledge graph.

    Returns chronological history of all changes (entity creates/updates,
    belief updates, relationship changes, node merges/deletes) with
    before/after snapshots.

    Args:
        entity_name: Filter to a specific entity (optional)
        operation_type: Filter by type: 'update_belief', 'entity_upsert',
            'relationship_upsert', 'add_relationship', 'delete_relationship',
            'delete_node', 'update_node', 'merge_nodes' (optional)
        since: ISO 8601 timestamp — only entries after this time (optional)
        until: ISO 8601 timestamp — only entries before this time (optional)
        limit: Max entries to return (default 20, max 200)
        project: Project ID (default: your personal space)
    """
    auth_ctx = _get_auth_context()
    scope_err = _require_pat_scope("read", auth_ctx)
    if scope_err:
        return {"error": scope_err}
    project = _resolve_project(project, auth_ctx)
    _uid, _grp, _err = _check_project_access(project, auth_ctx)
    if _err:
        return {"error": _err}
    adapter = _get_adapter()
    start = time.time()
    try:
        result = await _run_sync(
            adapter.get_history,
            project_id=project,
            entity_name=entity_name,
            operation_type=operation_type,
            since=since,
            until=until,
            limit=min(limit, 200),
        )
        result["duration_ms"] = int((time.time() - start) * 1000)
        audit_log("get_history", "authed",
                  {"entity": entity_name or "", "type": operation_type or "",
                   "project": project},
                  int((time.time() - start) * 1000), "ok")
        return result
    except Exception as e:
        audit_log("get_history", "authed",
                  {"entity": entity_name or "", "project": project},
                  int((time.time() - start) * 1000), "error", str(e))
        return {"error": str(e)}


@mcp.tool()
async def health_check() -> dict:
    """Check if the knowledge graph backends are healthy."""
    auth_ctx = _get_auth_context()
    scope_err = _require_pat_scope("read", auth_ctx)
    if scope_err:
        return {"error": scope_err}
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
