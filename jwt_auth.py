#!/usr/bin/env python3
"""
Merkraum JWT Authentication — Cognito JWT + PAT validation middleware for Flask.

Validates Authorization: Bearer <token> headers against:
1. Personal Access Tokens (mk_pat_ prefix) — validated against Neo4j
2. AWS Cognito User Pool JWTs — validated against JWKS

v1.0 — SUP-95 (2026-03-11)
v2.0 — PAT support (2026-03-13): PATValidator, require_scope decorator
"""

import hashlib
import json
import logging
import os
import secrets
from functools import wraps
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import jwt
import requests
from flask import request, jsonify, current_app

logger = logging.getLogger(__name__)


def _is_dev_mode() -> bool:
    dev_mode = os.environ.get("DEV_MODE")
    if dev_mode is not None:
        return dev_mode.lower() in ("true", "1", "yes")
    return False


def _auth_required() -> bool:
    raw = os.environ.get("AUTH_REQUIRED")
    if raw is None:
        return not _is_dev_mode()
    return raw.lower() in ("true", "1", "yes")


class CognitoJWTValidator:
    """Validates JWT tokens from AWS Cognito User Pool."""

    def __init__(
        self,
        user_pool_id: str,
        aws_region: str,
        client_id: Optional[str] = None,
        allowed_token_use: Optional[set[str]] = None,
    ):
        """
        Initialize Cognito JWT validator.

        Args:
            user_pool_id: Cognito User Pool ID (e.g., 'eu-central-1_B6owQ6if4')
            aws_region: AWS region (e.g., 'eu-central-1')
            client_id: Optional Cognito App Client ID for additional validation
        """
        self.user_pool_id = user_pool_id
        self.aws_region = aws_region
        self.client_id = client_id
        self.allowed_token_use = allowed_token_use or {"id"}
        self.cognito_domain = f"https://cognito-idp.{aws_region}.amazonaws.com/{user_pool_id}"
        self.jwks_url = f"{self.cognito_domain}/.well-known/jwks.json"
        self._jwks_cache: Optional[Dict[str, Any]] = None
        self._jwks_cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 3600  # Cache public keys for 1 hour

    def _get_jwks(self) -> Dict[str, Any]:
        """Fetch and cache the Cognito public keys (JWKS)."""
        now = datetime.now(timezone.utc)

        # Return cached keys if still valid
        if self._jwks_cache and self._jwks_cache_time:
            age = (now - self._jwks_cache_time).total_seconds()
            if age < self._cache_ttl_seconds:
                return self._jwks_cache

        # Fetch fresh JWKS from Cognito
        try:
            logger.info("Fetching JWKS from %s", self.jwks_url)
            response = requests.get(self.jwks_url, timeout=5)
            response.raise_for_status()
            self._jwks_cache = response.json()
            self._jwks_cache_time = now
            logger.info("JWKS fetched and cached successfully")
            return self._jwks_cache
        except requests.RequestException as exc:
            logger.error("Failed to fetch JWKS: %s", exc)
            raise

    def _get_public_key(self, token: str) -> Optional[str]:
        """Extract the public key for a given JWT token."""
        try:
            # Decode header without verification to get kid
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")

            if not kid:
                logger.error("JWT token has no 'kid' in header")
                return None

            jwks = self._get_jwks()
            keys = jwks.get("keys", [])

            # Find the key with matching kid
            matching_key = None
            for key in keys:
                if key.get("kid") == kid:
                    matching_key = key
                    break

            if not matching_key:
                logger.error("No key found for kid: %s", kid)
                return None

            # Convert JWKS key to PEM format
            from jwt.algorithms import RSAAlgorithm

            public_key = RSAAlgorithm.from_jwk(json.dumps(matching_key))
            return public_key

        except Exception as exc:
            logger.error("Error extracting public key: %s", exc)
            return None

    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate and decode a JWT token.

        Args:
            token: The JWT token string (without 'Bearer ' prefix)

        Returns:
            Decoded token claims if valid, None otherwise
        """
        try:
            public_key = self._get_public_key(token)
            if not public_key:
                logger.error("Could not retrieve public key for token")
                return None

            # Decode and validate the token
            decoded = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience=self.client_id,  # Can be None, which skips audience validation
                options={"verify_aud": bool(self.client_id)},
            )

            # Verify token is from expected Cognito User Pool
            token_iss = decoded.get("iss")
            expected_iss = self.cognito_domain
            if token_iss != expected_iss:
                logger.error(
                    "Invalid issuer: expected %s, got %s", expected_iss, token_iss
                )
                return None

            token_use = decoded.get("token_use")
            if token_use not in self.allowed_token_use:
                logger.error(
                    "Invalid token_use: expected one of %s, got %s",
                    sorted(self.allowed_token_use),
                    token_use,
                )
                return None

            # Verify token is not expired (jwt.decode handles this by default)
            logger.info(
                "Token validated successfully for user: %s",
                decoded.get("cognito:username"),
            )
            return decoded

        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as exc:
            logger.warning("Invalid token: %s", exc)
            return None
        except Exception as exc:
            logger.error("Error validating token: %s", exc)
            return None


PAT_PREFIX = "mk_pat_"
PAT_TOKEN_BYTES = 32  # 256 bits of entropy
PAT_PREFIX_LENGTH = 12  # chars shown in UI for identification

# Scope definitions
PAT_SCOPES = {"read", "write", "search", "ingest", "projects", "admin"}

# Per-tier token limits
PAT_TIER_LIMITS = {
    "free": 10,
    "pro": 50,
    "team": None,       # unlimited
    "enterprise": None,  # unlimited
}

# Per-project hard limit on active tokens (across all users)
PAT_PER_PROJECT_LIMIT = 20


class PATValidator:
    """Validates Personal Access Tokens against Neo4j storage."""

    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver

    def validate(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a PAT and return its metadata, or None if invalid.

        Uses a single atomic transaction for validation + last_used_at update
        (Norman review fix: prevents race condition under load).
        """
        if not token.startswith(PAT_PREFIX):
            return None

        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with self.driver.session() as session:
            result = session.execute_write(self._validate_and_touch, token_hash)
            return result

    @staticmethod
    def _validate_and_touch(tx, token_hash: str) -> Optional[Dict[str, Any]]:
        """Single transaction: match, validate, update last_used_at, return."""
        result = tx.run(
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

    def create_token(
        self,
        owner_id: str,
        name: str,
        scopes: list[str],
        projects: list[str],
        all_projects: bool = False,
        expires_at: Optional[str] = None,
        tier: str = "free",
    ) -> Optional[Dict[str, Any]]:
        """Create a new PAT. Returns token metadata including the plaintext (shown once)."""
        # Validate scopes
        invalid_scopes = set(scopes) - PAT_SCOPES
        if invalid_scopes:
            raise ValueError(f"Invalid scopes: {invalid_scopes}")

        # Check per-user token limit
        limit = PAT_TIER_LIMITS.get(tier)
        if limit is not None:
            count = self._count_user_tokens(owner_id)
            if count >= limit:
                raise ValueError(
                    f"Token limit reached: {count}/{limit} for tier '{tier}'"
                )

        # Check per-project limits
        if projects and not all_projects:
            for project in projects:
                proj_count = self._count_project_tokens(project)
                if proj_count >= PAT_PER_PROJECT_LIMIT:
                    raise ValueError(
                        f"Per-project token limit reached for '{project}': "
                        f"{proj_count}/{PAT_PER_PROJECT_LIMIT}"
                    )

        # Generate token
        raw_token = PAT_PREFIX + secrets.token_hex(PAT_TOKEN_BYTES)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        token_prefix = raw_token[:PAT_PREFIX_LENGTH]

        with self.driver.session() as session:
            session.execute_write(
                self._create_token_tx,
                token_hash=token_hash,
                token_prefix=token_prefix,
                name=name,
                owner_id=owner_id,
                scopes=scopes,
                projects=projects,
                all_projects=all_projects,
                expires_at=expires_at,
            )

        return {
            "token": raw_token,  # shown ONCE
            "token_prefix": token_prefix,
            "name": name,
            "scopes": scopes,
            "projects": projects,
            "all_projects": all_projects,
            "expires_at": expires_at,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _create_token_tx(
        tx,
        token_hash: str,
        token_prefix: str,
        name: str,
        owner_id: str,
        scopes: list[str],
        projects: list[str],
        all_projects: bool,
        expires_at: Optional[str],
    ):
        tx.run(
            """
            CREATE (t:PersonalAccessToken {
                token_hash: $hash,
                token_prefix: $prefix,
                name: $name,
                owner_id: $owner_id,
                scopes: $scopes,
                projects: $projects,
                all_projects: $all_projects,
                expires_at: $expires_at,
                created_at: toString(datetime()),
                last_used_at: null,
                revoked: false,
                revoked_at: null
            })
            """,
            hash=token_hash,
            prefix=token_prefix,
            name=name,
            owner_id=owner_id,
            scopes=scopes,
            projects=projects,
            all_projects=all_projects,
            expires_at=expires_at,
        )

    def list_tokens(self, owner_id: str) -> list[Dict[str, Any]]:
        """List all tokens for a user (never returns plaintext)."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (t:PersonalAccessToken {owner_id: $owner_id})
                RETURN t.token_prefix AS token_prefix,
                       t.name AS name,
                       t.scopes AS scopes,
                       t.projects AS projects,
                       t.all_projects AS all_projects,
                       t.expires_at AS expires_at,
                       t.created_at AS created_at,
                       t.last_used_at AS last_used_at,
                       t.revoked AS revoked
                ORDER BY t.created_at DESC
                """,
                owner_id=owner_id,
            )
            return [dict(r) for r in result]

    def revoke_token(self, owner_id: str, token_prefix: str) -> bool:
        """Revoke a token by prefix. Returns True if found and revoked."""
        with self.driver.session() as session:
            result = session.execute_write(
                self._revoke_token_tx, owner_id, token_prefix
            )
            return result

    @staticmethod
    def _revoke_token_tx(tx, owner_id: str, token_prefix: str) -> bool:
        result = tx.run(
            """
            MATCH (t:PersonalAccessToken {
                owner_id: $owner_id,
                token_prefix: $prefix,
                revoked: false
            })
            SET t.revoked = true, t.revoked_at = toString(datetime())
            RETURN count(t) AS revoked_count
            """,
            owner_id=owner_id,
            prefix=token_prefix,
        ).single()
        return result and result["revoked_count"] > 0

    def _count_user_tokens(self, owner_id: str) -> int:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (t:PersonalAccessToken {owner_id: $owner_id, revoked: false})
                RETURN count(t) AS cnt
                """,
                owner_id=owner_id,
            ).single()
            return result["cnt"] if result else 0

    def _count_project_tokens(self, project: str) -> int:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (t:PersonalAccessToken {revoked: false})
                WHERE $project IN t.projects
                RETURN count(t) AS cnt
                """,
                project=project,
            ).single()
            return result["cnt"] if result else 0

    @staticmethod
    def ensure_constraints(driver):
        """Create Neo4j constraint and index for PAT nodes. Call once at startup."""
        with driver.session() as session:
            session.run(
                "CREATE CONSTRAINT pat_hash_unique IF NOT EXISTS "
                "FOR (t:PersonalAccessToken) REQUIRE t.token_hash IS UNIQUE"
            )
            session.run(
                "CREATE INDEX pat_owner_index IF NOT EXISTS "
                "FOR (t:PersonalAccessToken) ON (t.owner_id)"
            )
        logger.info("PAT Neo4j constraints and indexes ensured")


def get_cognito_validator() -> Optional[CognitoJWTValidator]:
    """Create and return a CognitoJWTValidator from environment variables."""
    user_pool_id = os.environ.get("COGNITO_USER_POOL_ID")
    aws_region = os.environ.get("COGNITO_AWS_REGION")
    client_id = os.environ.get("COGNITO_CLIENT_ID")
    auth_required = _auth_required()
    allowed_token_use = {
        x.strip() for x in os.environ.get("COGNITO_TOKEN_USE", "id").split(",") if x.strip()
    } or {"id"}

    if not user_pool_id or not aws_region:
        logger.warning(
            "Cognito configuration incomplete: USER_POOL_ID=%s, AWS_REGION=%s",
            "set" if user_pool_id else "unset",
            "set" if aws_region else "unset",
        )
        return None

    if auth_required and not client_id:
        logger.warning(
            "COGNITO_CLIENT_ID is required when AUTH_REQUIRED=true"
        )
        return None

    return CognitoJWTValidator(
        user_pool_id=user_pool_id,
        aws_region=aws_region,
        client_id=client_id,
        allowed_token_use=allowed_token_use,
    )


def require_auth(f):
    """
    Flask decorator to require valid Cognito JWT authentication.

    Respects AUTH_REQUIRED environment variable:
    - AUTH_REQUIRED=true (or any truthy value): Requires valid token for all requests
    - AUTH_REQUIRED=false/unset: Allows unauthenticated access (local development mode)

    Usage:
        @app.route('/api/protected')
        @require_auth
        def protected_endpoint():
            ...

    Extracts user info from validated token and stores in request attributes:
    - request.user: Full token claims dict
    - request.user_id: Cognito subject (unique user ID)
    - request.username: Cognito username
    - request.groups: List of Cognito groups user belongs to
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if authentication is enabled
        auth_required = _auth_required()

        # Initialize request attributes
        request.user = None
        request.user_id = None
        request.username = None
        request.groups = []

        if not auth_required:
            # Auth disabled (local development mode) — allow unauthenticated access
            logger.debug("AUTH_REQUIRED=false: skipping authentication check")
            return f(*args, **kwargs)

        # Get validator from app config
        validator = getattr(current_app, "_cognito_validator", None)

        if not validator:
            logger.warning("Cognito validator not configured, but AUTH_REQUIRED=true. Rejecting request.")
            return (
                jsonify({"error": "Authentication required but not configured"}),
                503,
            )

        # Try to extract Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning(
                "Missing or invalid Authorization header from %s",
                request.remote_addr,
            )
            return (
                jsonify({"error": "Missing or invalid Authorization header"}),
                401,
            )

        token = auth_header[7:]  # Remove "Bearer " prefix

        # PAT authentication path
        if token.startswith(PAT_PREFIX):
            pat_validator = getattr(current_app, "_pat_validator", None)
            if not pat_validator:
                return jsonify({"error": "PAT authentication not configured"}), 503

            pat_data = pat_validator.validate(token)
            if not pat_data:
                logger.warning(
                    "PAT validation failed for request from %s",
                    request.remote_addr,
                )
                return jsonify({"error": "Invalid, expired, or revoked token"}), 401

            # Set request context to look like Cognito auth
            request.user_id = pat_data["owner_id"]
            request.username = f"pat:{pat_data.get('name', 'unknown')}"
            request.groups = []
            request.user = {"sub": pat_data["owner_id"], "auth_type": "pat"}
            request.pat_scopes = pat_data.get("scopes") or []
            request.pat_projects = pat_data.get("projects") or []
            request.pat_all_projects = pat_data.get("all_projects", False)

            logger.info(
                "PAT authenticated request from owner: %s (token: %s, scopes: %s)",
                pat_data["owner_id"],
                pat_data.get("token_prefix", "?"),
                pat_data.get("scopes", []),
            )
            return f(*args, **kwargs)

        # Cognito JWT authentication path
        claims = validator.validate_token(token)
        if not claims:
            logger.warning(
                "Token validation failed for request from %s", request.remote_addr
            )
            return jsonify({"error": "Invalid or expired token"}), 401

        # Attach user info to request context
        request.user = claims
        request.user_id = claims.get("sub")  # Cognito subject (unique user ID)
        request.username = claims.get("cognito:username")
        request.groups = claims.get("cognito:groups", [])
        # Cognito users have no PAT restrictions
        request.pat_scopes = None
        request.pat_projects = None
        request.pat_all_projects = None

        logger.info(
            "Authenticated request from user: %s (groups: %s)",
            request.username,
            ", ".join(request.groups) or "none",
        )

        return f(*args, **kwargs)

    return decorated_function


def require_scope(scope: str):
    """Decorator: require specific scope for PAT-authenticated requests.

    Cognito JWT users implicitly have all scopes. PAT users must have the
    scope explicitly listed, or 'admin' scope which grants all.
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            pat_scopes = getattr(request, "pat_scopes", None)
            if pat_scopes is not None:  # PAT auth (not Cognito)
                if scope not in pat_scopes and "admin" not in pat_scopes:
                    return (
                        jsonify({"error": f"Token lacks required scope: {scope}"}),
                        403,
                    )
            return f(*args, **kwargs)
        return decorated
    return decorator


# ============================================================================
# Optional: Optional auth decorator (allows both authenticated and unauthenticated)
# ============================================================================


def optional_auth(f):
    """
    Flask decorator for optional Cognito JWT authentication.

    If a valid Authorization: Bearer header is present, validates it and
    attaches user info to flask.g.user. If not present or invalid, continues
    anyway (but request.user will be None).

    Usage:
        @app.route('/api/maybe-protected')
        @optional_auth
        def endpoint():
            if request.user:
                # User is authenticated
            else:
                # User is not authenticated
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        request.user = None
        request.user_id = None
        request.username = None
        request.groups = []

        validator = getattr(current_app, "_cognito_validator", None)
        if not validator:
            return f(*args, **kwargs)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return f(*args, **kwargs)

        token = auth_header[7:]
        claims = validator.validate_token(token)

        if claims:
            request.user = claims
            request.user_id = claims.get("sub")
            request.username = claims.get("cognito:username")
            request.groups = claims.get("cognito:groups", [])
            logger.info(
                "Optional auth: authenticated as %s (groups: %s)",
                request.username,
                ", ".join(request.groups) or "none",
            )
        else:
            logger.debug("Optional auth: could not validate token, continuing unauthenticated")

        return f(*args, **kwargs)

    return decorated_function
