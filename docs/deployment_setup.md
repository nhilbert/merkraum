# Merkraum Backend Deployment Setup

## Current Status (Z1349)
The merkraum REST API (`merkraum_api.py`) runs on `localhost:8083` but is NOT accessible from the internet. The frontend at `app.merkraum.de` cannot reach it.

## What Norman Needs to Do (2 steps, ~3 minutes)

### Step 1: Add nginx proxy (requires sudo)

Add this location block to `/etc/nginx/sites-available/default` inside the `agent.nhilbert.de` server block (after the `/mcp` block):

```nginx
    # Merkraum REST API
    location /api/merkraum/ {
        proxy_pass http://127.0.0.1:8083/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
```

Then reload nginx:
```bash
sudo nginx -t && sudo systemctl reload nginx
```

### Step 2: Set VITE_API_URL in Amplify

```bash
aws amplify update-branch \
  --app-id d3mbzq4y4p60mx \
  --branch-name main \
  --environment-variables VITE_API_URL=https://www.agent.nhilbert.de/api/merkraum
```

Then trigger a redeploy (push any commit to merkraum-front, or start a build in the Amplify console).

### Step 3 (optional): Make API persistent

Add to crontab:
```
@reboot /home/vsg/merkraum/start_api.sh
```

## Architecture

```
Browser (app.merkraum.de)
  → CloudFront (Amplify)
  → React app calls VITE_API_URL + /api/ingest/text
  → https://www.agent.nhilbert.de/api/merkraum/ingest/text
  → nginx (port 80) proxy_pass
  → localhost:8083/api/ingest/text
  → merkraum_api.py → Neo4j + OpenAI
```

## API Endpoints
- `GET /api/health` — health check
- `GET /api/stats` — graph statistics
- `GET /api/beliefs` — list beliefs
- `GET /api/graph` — graph data (nodes + edges)
- `POST /api/ingest/text` — LLM-based text ingestion (requires OpenAI key)
- `GET /api/search?q=...` — semantic search (PAT scope: `search`)
- `POST /api/chat` — chat over graph context (PAT scope: `search`)
- `POST /api/feedback` — feedback ticket submission (PAT scope: `write`)
- `GET /api/projects` — list projects

## Required reverse-proxy / WAF controls for OAuth token proxy

When exposing MCP OAuth endpoints (`/authorize`, `/token`, `/register`) behind nginx, CloudFront, or another edge gateway, enforce the controls below in the edge layer **in addition to** application-level checks:

1. **Body size limit on `/token`**: cap request body to `16k` (or stricter), and reject larger requests before they reach the app.
2. **Rate limit on `/token` and `/register`**:
   - Per client IP and path (example: burst 10, sustained 30/min).
   - Return HTTP `429` on exceed.
3. **Method and content-type allowlist**:
   - `/token` must be `POST`.
   - Allow only `application/x-www-form-urlencoded` and `application/json`.
4. **Forwarded client IP integrity**:
   - Ensure only trusted proxies can set `X-Forwarded-For`.
   - Strip user-supplied forwarding headers at the edge.
5. **TLS + host allowlist**:
   - Require HTTPS at the edge.
   - Restrict Host headers to expected deployment hostnames.

Example nginx controls:

```nginx
client_max_body_size 16k;

limit_req_zone $binary_remote_addr zone=oauth_token_ip:10m rate=30r/m;

location = /token {
    limit_req zone=oauth_token_ip burst=10 nodelay;
    if ($request_method != POST) { return 405; }
    proxy_pass http://127.0.0.1:8090/token;
}
```

## Safe enablement: MCP dynamic client registration (`/register`)

Dynamic client registration is **disabled by default** (`MCP_ENABLE_DYNAMIC_CLIENT_REGISTRATION=false`).  
Only enable it when all controls below are configured.

### Required environment variables

```bash
# Explicit opt-in
MCP_ENABLE_DYNAMIC_CLIENT_REGISTRATION=true

# Redirect allowlist (comma-separated exact URIs)
MCP_ALLOWED_REDIRECT_URIS=https://claude.ai/oauth/callback,https://example.com/mcp/callback

# App-level throttle (requests per IP per minute)
MCP_REGISTER_RATE_LIMIT_PER_MINUTE=10

# At least ONE non-dev registration gate:
# Option A: shared admin bootstrap secret header (X-MCP-Admin-Secret)
MCP_REGISTRATION_ADMIN_SECRET=<long-random-secret>

# Option B: signed registration token validation (HS256)
MCP_REGISTRATION_TOKEN_SIGNING_KEY=<long-random-hmac-key>
MCP_REGISTRATION_TOKEN_AUDIENCE=mcp-register
```

### Startup behavior

- If `MCP_ENABLE_DYNAMIC_CLIENT_REGISTRATION=true` **without strict controls**:
  - **non-dev**: server startup fails fast with an explicit error.
  - **dev**: server startup logs an explicit warning.

### Gateway-level rate limiting (nginx)

In addition to app-level throttling, configure edge/gateway limits:

```nginx
limit_req_zone $binary_remote_addr zone=mcp_register_limit:10m rate=10r/m;

location = /register {
    limit_req zone=mcp_register_limit burst=5 nodelay;
    proxy_pass http://127.0.0.1:8090/register;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

### Operational notes

- Response metadata from `/register` is server-approved only (no arbitrary echo of request fields).
- `redirect_uris` are accepted only when every value is exactly allowlisted.
- In non-dev, registration requires either `X-MCP-Admin-Secret` or a valid signed `registration_token`.
