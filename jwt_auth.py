#!/usr/bin/env python3
"""
Merkraum JWT Authentication — Cognito JWT validation middleware for Flask.

Validates Authorization: Bearer <token> headers against AWS Cognito User Pool.
Caches public keys to avoid repeated JWKS downloads.

v1.0 — SUP-95 (2026-03-11)
"""

import json
import logging
import os
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

        # Validate token
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

        logger.info(
            "Authenticated request from user: %s (groups: %s)",
            request.username,
            ", ".join(request.groups) or "none",
        )

        return f(*args, **kwargs)

    return decorated_function


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
