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
