#!/usr/bin/env python3
import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestAuthDefaults(unittest.TestCase):
    def test_jwt_auth_required_defaults_to_true(self):
        with patch.dict(os.environ, {}, clear=True):
            import jwt_auth
            self.assertTrue(jwt_auth._auth_required())

    def test_jwt_auth_required_dev_mode_false(self):
        with patch.dict(os.environ, {"DEV_MODE": "true"}, clear=True):
            import jwt_auth
            self.assertFalse(jwt_auth._auth_required())

    def test_api_auth_required_respects_dev_mode(self):
        with patch.dict(os.environ, {"DEV_MODE": "true"}, clear=True):
            import merkraum_api
            self.assertFalse(merkraum_api._is_auth_required())


class TestProjectAclHardening(unittest.TestCase):
    def test_default_project_denied_by_default_when_auth_enabled(self):
        with patch.dict(os.environ, {}, clear=True):
            import merkraum_api
            with merkraum_api.app.test_request_context("/api/graph?project=default"):
                from flask import request
                request.user_id = "user-123"
                request.groups = []
                self.assertFalse(merkraum_api._is_project_allowed("default"))

    def test_default_project_can_be_enabled_explicitly(self):
        with patch.dict(os.environ, {"ALLOW_DEFAULT_PROJECT": "true"}, clear=True):
            import merkraum_api
            with merkraum_api.app.test_request_context("/api/graph?project=default"):
                from flask import request
                request.user_id = "user-123"
                request.groups = []
                self.assertTrue(merkraum_api._is_project_allowed("default"))


class TestMcpClientBinding(unittest.TestCase):
    @patch("merkraum_mcp_server._fetch_jwks", return_value={"keys": [{"kid": "k1"}]})
    @patch("jwt.algorithms.RSAAlgorithm.from_jwk", return_value="public_key")
    @patch("jwt.decode")
    @patch("jwt.get_unverified_header", return_value={"kid": "k1"})
    def test_validate_jwt_rejects_non_allowlisted_client(
        self,
        _mock_header,
        mock_decode,
        _mock_key,
        _mock_jwks,
    ):
        import merkraum_mcp_server

        mock_decode.return_value = {
            "iss": merkraum_mcp_server.COGNITO_ISSUER,
            "token_use": "access",
            "client_id": "unexpected-client",
            "exp": 9999999999,
        }

        old_allowed = set(merkraum_mcp_server.MCP_ALLOWED_CLIENT_IDS)
        merkraum_mcp_server.MCP_ALLOWED_CLIENT_IDS = {"allowed-client"}
        try:
            with self.assertRaises(ValueError):
                merkraum_mcp_server.validate_jwt("dummy-token")
        finally:
            merkraum_mcp_server.MCP_ALLOWED_CLIENT_IDS = old_allowed


class TestGraphQueryHardening(unittest.TestCase):
    def test_semantic_graph_query_requires_search_scope_for_pat(self):
        with patch.dict(os.environ, {"AUTH_REQUIRED": "false"}, clear=True):
            import merkraum_api
            previous_adapter = merkraum_api.adapter
            try:
                merkraum_api.adapter = object()
                with merkraum_api.app.test_request_context("/api/graph?project=test-proj&q=risk"):
                    from flask import request
                    request.user_id = "user-123"
                    request.groups = []
                    request.pat_scopes = ["read"]
                    request.pat_projects = ["test-proj"]
                    request.pat_all_projects = False

                    response, status = merkraum_api.graph()
                    self.assertEqual(status, 403)
                    self.assertEqual(response.get_json().get("error"), "Token lacks required scope: search")
            finally:
                merkraum_api.adapter = previous_adapter

    def test_semantic_graph_query_clamps_params_and_calls_subgraph_builder(self):
        with patch.dict(os.environ, {"AUTH_REQUIRED": "false"}, clear=True):
            import merkraum_api
            previous_adapter = merkraum_api.adapter
            try:
                merkraum_api.adapter = object()
                with patch.object(
                    merkraum_api,
                    "_get_semantic_subgraph",
                    return_value={
                        "nodes": [],
                        "links": [],
                        "meta": {"mode": "semantic_subgraph", "truncated": False},
                    },
                ) as subgraph_mock:
                    with merkraum_api.app.test_request_context(
                        "/api/graph?project=test-proj&q=delta&hops=99&top=999&limit=99999"
                    ):
                        from flask import request
                        request.user_id = "user-123"
                        request.groups = []
                        request.pat_scopes = ["read", "search"]
                        request.pat_projects = ["test-proj"]
                        request.pat_all_projects = False

                        response = merkraum_api.graph()
                        self.assertEqual(response.status_code, 200)
                        payload = response.get_json()
                        self.assertEqual(payload.get("meta", {}).get("mode"), "semantic_subgraph")

                        subgraph_mock.assert_called_once_with(
                            merkraum_api.adapter,
                            "test-proj",
                            "delta",
                            limit=merkraum_api.MAX_GRAPH_LIMIT,
                            hops=merkraum_api.MAX_GRAPH_HOPS,
                            top=merkraum_api.MAX_SEARCH_TOP,
                        )
            finally:
                merkraum_api.adapter = previous_adapter


class TestMcpTenantHardening(unittest.TestCase):
    @patch.dict(os.environ, {"AUTH_REQUIRED": "true"})
    def test_mcp_check_project_access_honors_pat_project_restrictions(self):
        import merkraum_mcp_server

        token = SimpleNamespace(
            claims={
                "sub": "user-123",
                "token_type": "pat",
                "projects": ["user-123"],
                "all_projects": False,
            },
            scopes=["read"],
        )
        with patch.object(merkraum_mcp_server, "get_access_token", return_value=token):
            _uid, _groups, err = merkraum_mcp_server._check_project_access("other-project")
            self.assertEqual(err, "Forbidden: no access to project 'other-project'")

    def test_mcp_require_pat_scope_denies_missing_scope(self):
        import merkraum_mcp_server

        token = SimpleNamespace(
            claims={"sub": "user-123", "token_type": "pat"},
            scopes=["read"],
        )
        with patch.object(merkraum_mcp_server, "get_access_token", return_value=token):
            auth_ctx = merkraum_mcp_server._get_auth_context()
            self.assertEqual(
                merkraum_mcp_server._require_pat_scope("write", auth_ctx),
                "Token lacks required scope: write",
            )
            self.assertIsNone(merkraum_mcp_server._require_pat_scope("read", auth_ctx))

    def test_check_ingestion_status_is_owner_bound(self):
        import merkraum_mcp_server

        original_jobs = None
        with merkraum_mcp_server._jobs_lock:
            original_jobs = dict(merkraum_mcp_server._jobs)
            merkraum_mcp_server._jobs.clear()
            merkraum_mcp_server._jobs["job-1"] = {
                "status": "queued",
                "created": 0,
                "text_len": 12,
                "result": None,
                "error": None,
                "owner_id": "owner-user",
            }

        token = SimpleNamespace(
            claims={"sub": "other-user", "token_type": "pat"},
            scopes=["read"],
        )
        try:
            with patch.object(merkraum_mcp_server, "get_access_token", return_value=token):
                result = asyncio.run(merkraum_mcp_server.check_ingestion_status("job-1"))
                self.assertEqual(result.get("error"), "Forbidden: no access to this ingestion job")
        finally:
            with merkraum_mcp_server._jobs_lock:
                merkraum_mcp_server._jobs.clear()
                merkraum_mcp_server._jobs.update(original_jobs)


class TestReindexApiHardening(unittest.TestCase):
    def test_reindex_api_forwards_cleanup_flag(self):
        with patch.dict(os.environ, {"AUTH_REQUIRED": "false"}, clear=True):
            import merkraum_api

            previous_adapter = merkraum_api.adapter
            mock_adapter = MagicMock()
            mock_adapter.reindex_project_vectors.return_value = {
                "project_id": "proj-1",
                "total_nodes": 0,
                "upserted": 0,
                "failed": 0,
                "legacy_deleted": 0,
                "limit": 5000,
                "truncated": False,
            }
            try:
                merkraum_api.adapter = mock_adapter
                with merkraum_api.app.test_request_context(
                    "/api/projects/proj-1/vectors/reindex",
                    method="POST",
                    json={"cleanup_legacy_ids": True},
                ):
                    response = merkraum_api.reindex_project_vectors("proj-1")
                    status = response[1] if isinstance(response, tuple) else response.status_code
                    self.assertEqual(status, 200)
                    mock_adapter.reindex_project_vectors.assert_called_once_with(
                        project_id="proj-1",
                        limit=5000,
                        cleanup_legacy_ids=True,
                    )
            finally:
                merkraum_api.adapter = previous_adapter


class TestMcpCertaintyToolAuthorization(unittest.TestCase):
    """PAT scope + ACL checks for certainty/expiry MCP tools."""

    def _make_token(self, scopes, projects=None, all_projects=False):
        return SimpleNamespace(
            claims={
                "sub": "user-123",
                "token_type": "pat",
                "projects": projects if projects is not None else ["user-123"],
                "all_projects": all_projects,
            },
            scopes=scopes,
        )

    def test_pat_no_scope_rejected(self):
        import merkraum_mcp_server

        adapter = MagicMock()
        token = self._make_token(scopes=[])
        cases = [
            (merkraum_mcp_server.expire_nodes, {"dry_run": True, "project": "user-123"}),
            (merkraum_mcp_server.renew_node, {"name": "belief-1", "extend_days": 30, "project": "user-123"}),
            (merkraum_mcp_server.certainty_decay, {"dry_run": True, "project": "user-123"}),
            (merkraum_mcp_server.certainty_review, {"limit": 10, "project": "user-123"}),
            (merkraum_mcp_server.certainty_stats, {"project": "user-123"}),
        ]

        with patch.object(merkraum_mcp_server, "get_access_token", return_value=token), \
                patch.object(merkraum_mcp_server, "_get_adapter", return_value=adapter):
            for tool_fn, kwargs in cases:
                with self.subTest(tool=tool_fn.__name__):
                    result = asyncio.run(tool_fn(**kwargs))
                    self.assertIn("error", result)
                    self.assertIn("Token lacks required scope", result["error"])

        adapter.expire_nodes.assert_not_called()
        adapter.renew_node.assert_not_called()
        adapter.apply_confidence_decay.assert_not_called()
        adapter.get_certainty_review_queue.assert_not_called()
        adapter.get_certainty_stats.assert_not_called()

    def test_pat_correct_scope_succeeds(self):
        import merkraum_mcp_server

        adapter = MagicMock()
        adapter.expire_nodes.return_value = {"total": 0, "expired": [], "dry_run": True}
        adapter.renew_node.return_value = {"renewed": True, "name": "belief-1"}
        adapter.apply_confidence_decay.return_value = {"total": 0, "decayed": [], "dry_run": True}
        adapter.get_certainty_review_queue.return_value = {"categories": {}}
        adapter.get_certainty_stats.return_value = {"governance": {"status": "healthy"}}

        write_token = self._make_token(scopes=["write"], projects=["user-123"])
        read_token = self._make_token(scopes=["read"], projects=["user-123"])

        with patch.object(merkraum_mcp_server, "_get_adapter", return_value=adapter):
            with patch.object(merkraum_mcp_server, "get_access_token", return_value=write_token):
                expire_result = asyncio.run(merkraum_mcp_server.expire_nodes(project="user-123"))
                renew_result = asyncio.run(merkraum_mcp_server.renew_node(
                    name="belief-1", extend_days=30, project="user-123"))
                decay_result = asyncio.run(merkraum_mcp_server.certainty_decay(project="user-123"))

            with patch.object(merkraum_mcp_server, "get_access_token", return_value=read_token):
                review_result = asyncio.run(merkraum_mcp_server.certainty_review(project="user-123"))
                stats_result = asyncio.run(merkraum_mcp_server.certainty_stats(project="user-123"))

        self.assertNotIn("error", expire_result)
        self.assertNotIn("error", renew_result)
        self.assertNotIn("error", decay_result)
        self.assertNotIn("error", review_result)
        self.assertNotIn("error", stats_result)
        adapter.expire_nodes.assert_called_once()
        adapter.renew_node.assert_called_once()
        adapter.apply_confidence_decay.assert_called_once()
        adapter.get_certainty_review_queue.assert_called_once()
        adapter.get_certainty_stats.assert_called_once()

    def test_project_acl_denial(self):
        import merkraum_mcp_server

        adapter = MagicMock()
        write_token = self._make_token(scopes=["write"], projects=[], all_projects=True)
        read_token = self._make_token(scopes=["read"], projects=[], all_projects=True)

        with patch.object(merkraum_mcp_server, "_get_adapter", return_value=adapter):
            with patch.object(merkraum_mcp_server, "get_access_token", return_value=write_token):
                expire_result = asyncio.run(merkraum_mcp_server.expire_nodes(project="project-b"))
                renew_result = asyncio.run(merkraum_mcp_server.renew_node(
                    name="belief-1", extend_days=30, project="project-b"))
                decay_result = asyncio.run(merkraum_mcp_server.certainty_decay(project="project-b"))

            with patch.object(merkraum_mcp_server, "get_access_token", return_value=read_token):
                review_result = asyncio.run(merkraum_mcp_server.certainty_review(project="project-b"))
                stats_result = asyncio.run(merkraum_mcp_server.certainty_stats(project="project-b"))

        for result in [expire_result, renew_result, decay_result, review_result, stats_result]:
            self.assertEqual(result.get("error"), "Forbidden: no access to project 'project-b'")

        adapter.expire_nodes.assert_not_called()
        adapter.renew_node.assert_not_called()
        adapter.apply_confidence_decay.assert_not_called()
        adapter.get_certainty_review_queue.assert_not_called()
        adapter.get_certainty_stats.assert_not_called()

if __name__ == "__main__":
    unittest.main()
