#!/usr/bin/env python3
import os
import unittest
from unittest.mock import patch


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

if __name__ == "__main__":
    unittest.main()
