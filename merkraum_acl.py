"""Shared project-level ACL logic for Merkraum API and MCP servers.

Centralizes authorization rules so both entry points enforce identical policies.
"""

import json
import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger("merkraum-acl")


def is_auth_required() -> bool:
    """Check if authentication is required based on environment."""
    raw = os.environ.get("AUTH_REQUIRED")
    if raw is not None:
        return raw.lower() in ("true", "1", "yes")
    dev_mode = os.environ.get("DEV_MODE")
    if dev_mode is not None:
        return dev_mode.lower() not in ("true", "1", "yes")
    return True


def split_csv_env(name: str) -> set[str]:
    """Parse a comma-separated environment variable into a set."""
    raw = os.environ.get(name, "")
    return {x.strip() for x in raw.split(",") if x.strip()}


@lru_cache(maxsize=1)
def project_group_acl() -> dict[str, set[str]]:
    """Parse PROJECT_GROUP_ACL_JSON into {project: {groups}}."""
    raw = os.environ.get("PROJECT_GROUP_ACL_JSON", "{}")
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}
        out: dict[str, set[str]] = {}
        for project, groups in parsed.items():
            if isinstance(project, str) and isinstance(groups, list):
                out[project] = {str(g).strip() for g in groups if str(g).strip()}
        return out
    except Exception:
        logger.warning("Invalid PROJECT_GROUP_ACL_JSON; ignoring")
        return {}


@lru_cache(maxsize=1)
def project_user_acl() -> dict[str, set[str]]:
    """Parse PROJECT_USER_ACL_JSON into {project: {user_ids}}."""
    raw = os.environ.get("PROJECT_USER_ACL_JSON", "{}")
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}
        out: dict[str, set[str]] = {}
        for project, users in parsed.items():
            if isinstance(project, str) and isinstance(users, list):
                out[project] = {str(u).strip() for u in users if str(u).strip()}
        return out
    except Exception:
        logger.warning("Invalid PROJECT_USER_ACL_JSON; ignoring")
        return {}


def is_project_allowed(
    project: str,
    user_id: Optional[str],
    groups: Optional[set[str]] = None,
) -> bool:
    """Check if a user is allowed to access a project.

    Authorization rules (in order):
    1. AUTH_REQUIRED=false → allow all
    2. No user_id → deny
    3. project=="default" and ALLOW_DEFAULT_PROJECT=true → allow
    4. User in ADMIN_GROUPS → allow all projects
    5. project == user_id or project starts with "{user_id}:" → allow (user namespace)
    6. user_id in PROJECT_USER_ACL_JSON[project] → allow
    7. Any user group in PROJECT_GROUP_ACL_JSON[project] → allow
    8. Deny
    """
    if not is_auth_required():
        return True

    if not user_id:
        return False

    groups = groups or set()

    if project == "default" and os.environ.get(
        "ALLOW_DEFAULT_PROJECT", "false"
    ).lower() in ("true", "1", "yes"):
        return True

    admin_groups = split_csv_env("ADMIN_GROUPS")
    if admin_groups and groups.intersection(admin_groups):
        return True

    if project == user_id or project.startswith(f"{user_id}:"):
        return True

    if user_id in project_user_acl().get(project, set()):
        return True

    if groups.intersection(project_group_acl().get(project, set())):
        return True

    return False
