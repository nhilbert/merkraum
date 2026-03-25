# Security Review (2026-03-11)

## Scope

This review covers the Python services in this repository:

- `merkraum_api.py` (Flask REST API)
- `merkraum_mcp_server.py` (FastMCP server + OAuth proxies)
- `jwt_auth.py` (Cognito JWT validation middleware)
- `merkraum_backend.py` (data adapters + external service calls)

Methodology:

1. Manual code review of authentication, authorization, input handling, secrets handling, and outbound HTTP usage.
2. Static analysis with Bandit.
3. Validation checks run in this repo (`pytest`, `ruff`, `mypy`, `npm run build`).

---

## Executive summary

The repository has solid baseline controls (JWT signature validation, issuer checks, allowlisted CORS origins, parameterized Cypher query values, and production guardrails in the Flask entrypoint). However, several high-impact misconfiguration and multi-tenant isolation risks remain, especially in the MCP server authentication model and project authorization defaults.

### Overall risk rating: **Medium-High**

- **High findings**: 2
- **Medium findings**: 3
- **Low findings**: 3

Top priorities:

1. Enforce audience/client binding for JWTs in MCP auth path.
2. Remove insecure hardcoded Cognito defaults and fail closed when auth config is missing.
3. Tighten project authorization defaults (`ALLOW_DEFAULT_PROJECT=false`) for multi-tenant deployments.

---

## Findings

### 1) JWT audience is not verified in MCP authentication (**High**)

**Evidence**

- `validate_jwt()` explicitly sets `"verify_aud": False` and does not check `aud` or `client_id` claims against an allowlist. `COGNITO_APP_CLIENT_ID` exists but is not used in token validation logic.【F:merkraum_mcp_server.py†L53-L54】【F:merkraum_mcp_server.py†L164-L175】

**Impact**

Any valid token issued by the configured Cognito User Pool may be accepted, even if minted for a different app client. In shared user pool designs, this can widen trust boundaries and enable unauthorized client access.

**Recommendation**

- Enforce client binding by validating `aud` (ID tokens) and/or `client_id` (access tokens) against explicit allowed app client IDs.
- Add config `MCP_ALLOWED_CLIENT_IDS` and fail authentication when missing in non-dev environments.

---

### 2) MCP server has security-sensitive hardcoded Cognito defaults (**High**)

**Evidence**

- Default pool ID, default client ID, default auth domain, and default public MCP base URL are hardcoded at import time.【F:merkraum_mcp_server.py†L49-L60】

**Impact**

If environment variables are missing or partially configured, the server can silently trust unintended identity infrastructure. This increases risk of accidental exposure and operational drift between environments.

**Recommendation**

- Remove hardcoded identity defaults for production paths.
- Fail startup if required auth vars are missing, except in explicit `DEV_MODE=true`.
- Log effective auth config safely (without secrets) at boot.

---

### 3) Shared dynamic client registration response without authentication controls (**Medium**)

**Evidence**

- `/register` returns a pre-created `client_id` for all callers and sets `token_endpoint_auth_method` to `none` without caller authentication, policy checks, or rate limiting at application layer.【F:merkraum_mcp_server.py†L358-L375】

**Impact**

Untrusted callers can repeatedly obtain bootstrap metadata and abuse client registration assumptions. Even if Cognito enforces controls downstream, this endpoint can become an abuse or reconnaissance surface.

**Recommendation**

- Require authenticated/admin bootstrap for registration or remove endpoint if not required.
- At minimum, add IP/rate limits and strict redirect URI validation.
- Consider per-client registration rather than global static client metadata.

---

### 4) Authorization default allows broad access to `default` project (**Medium**)

**Evidence**

- `_is_project_allowed()` grants access to project `default` whenever `ALLOW_DEFAULT_PROJECT` is truthy, and it defaults to `"true"`.【F:merkraum_api.py†L163-L173】

**Impact**

In multi-tenant deployments, authenticated users can potentially access shared data in `default` unless hardening env vars are set. This is a cross-tenant data exposure risk.

**Recommendation**

- Change default to `ALLOW_DEFAULT_PROJECT=false`.
- Make tenancy explicit: user-owned namespace by default, shared project only via ACL.
- Add startup warning/error if auth is enabled and default project remains open.

---

### 5) REST API auth defaults to disabled and relies on environment tagging for enforcement (**Medium**)

**Evidence**

- `require_auth` bypasses authentication when `AUTH_REQUIRED` is unset/false.【F:jwt_auth.py†L229-L240】
- Production hardening (`AUTH_REQUIRED` must be true) only executes when app detects production env labels (`APP_ENV`/`FLASK_ENV`).【F:merkraum_api.py†L1146-L1154】

**Impact**

Mis-set environment labels can cause accidental unauthenticated deployments.

**Recommendation**

- Invert default for server runtime (`AUTH_REQUIRED=true` by default).
- Add explicit `DEV_MODE=true` gate for auth bypass.
- Emit startup error when bound to non-loopback and auth is disabled.

---

### 6) Outbound URL handling allows generic `urlopen` usage on configurable endpoints (**Low**)

**Evidence**

- Bandit reports multiple B310 occurrences where `urllib.request.urlopen` is used against config-derived URLs (Cognito endpoints, Qdrant/Pinecone/OpenAI paths).【F:merkraum_mcp_server.py†L55-L57】【F:merkraum_mcp_server.py†L335-L337】

**Impact**

If environment is attacker-controlled, SSRF-style misuse becomes easier.

**Recommendation**

- Validate URL scheme/host against strict allowlists before outbound calls.
- Prefer typed client SDKs with host pinning where possible.

---

### 7) Error detail leakage from token proxy (**Low**)

**Evidence**

- Token proxy returns raw exception string in API response body (`"detail": str(e)`).【F:merkraum_mcp_server.py†L345-L347】

**Impact**

May expose internal network or parsing details useful for reconnaissance.

**Recommendation**

- Return generic error messages to clients; keep detailed diagnostics only in logs.

---

### 8) Broad network bind default in Flask API entrypoint (**Low**)

**Evidence**

- API CLI default host is `0.0.0.0`.【F:merkraum_api.py†L1127-L1129】

**Impact**

Increases accidental exposure likelihood in local/dev environments.

**Recommendation**

- Default to `127.0.0.1`; require explicit opt-in for external bind.

---

## Positive controls observed

- JWT validation checks signature and issuer; expiration is enforced by JWT library defaults in the Flask auth path.【F:jwt_auth.py†L129-L140】【F:jwt_auth.py†L155-L166】
- REST endpoints consistently use `@require_auth` decorators (subject to `AUTH_REQUIRED`).【F:merkraum_api.py†L420-L421】【F:merkraum_api.py†L1034-L1035】
- CORS is origin-allowlisted and preflight handling denies disallowed origins for API paths.【F:merkraum_api.py†L59-L76】【F:merkraum_api.py†L96-L102】
- Neo4j query values are generally passed as parameters rather than string-interpolated untrusted values.

---

## Tool output summary

- **Bandit**: reported medium/low issues (notably B310 URL handling and B104 broad bind).
- **Pytest**: all tests passed.
- **Ruff/Mypy**: existing lint/type issues present in repository (not introduced by this review).
- **npm run build**: not applicable for this repo layout (`package.json` missing).

---

## Prioritized remediation roadmap

## Remediation status update (implemented in this repository)

Implemented in code:

1. JWT client binding is now enforced in MCP token validation via `MCP_ALLOWED_CLIENT_IDS` (with `aud` checks for ID tokens and `client_id` checks for access tokens).
2. MCP auth configuration now fails closed in non-dev mode when required Cognito/MCP env vars are missing; insecure hardcoded Cognito defaults were removed.
3. Default project access is now hardened: `ALLOW_DEFAULT_PROJECT` defaults to `false`.
4. API auth default is now secure-by-default (`AUTH_REQUIRED=true` unless explicitly disabled or `DEV_MODE=true`).
5. Token proxy no longer returns raw exception text to clients.
6. Dynamic client registration (`/register`) is now disabled by default and must be explicitly enabled.
7. `/register` now has non-dev hard gates (admin secret header or signed registration token), strict redirect URI allowlisting, approved-only response metadata, and in-process rate limiting.
8. MCP startup now emits an explicit warning (dev) or startup error (non-dev) when dynamic registration is enabled without strict controls configured.
9. API default bind host is now loopback (`127.0.0.1`) and startup blocks non-loopback binds when auth is disabled.

Partially addressed / deferred:

1. URL egress hardening remains partial (basic HTTPS URL validation for configured auth endpoints added; full host allowlisting across all adapters still pending).
2. Gateway rate-limiting policy for `/register` still must be configured per deployment (nginx/CloudFront/WAF), even though app-level throttling now exists.

### Immediate (0–7 days)

1. Enforce JWT audience/client checks in MCP verifier.
2. Remove hardcoded Cognito defaults and fail closed on missing auth config.
3. Disable open default project in auth-enabled environments.

### Near-term (1–3 weeks)

1. Harden `/register` endpoint (authN/authZ + rate limiting) or remove if unnecessary.
2. Replace verbose token proxy error details with generic messages.
3. Add secure startup assertions for non-loopback bind + auth disabled combinations.

### Ongoing (quarterly)

1. Add CI security gates: Bandit + dependency audit + secret scanning.
2. Establish deployment profiles (dev/stage/prod) with locked security defaults.
3. Add tenant-isolation integration tests covering project ACL edge cases.
