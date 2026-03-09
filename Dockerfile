# Merkraum — Personal Knowledge Memory for AI Agents
# Dockerfile for the MCP server container.
# Uses multi-stage build to keep the image small.
#
# v1.0 — Z1150 (2026-03-08)

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY merkraum_backend.py merkraum_mcp_server.py ./

# Default environment (overridable via docker-compose or .env)
ENV MERKRAUM_BACKEND=neo4j_qdrant \
    MERKRAUM_PROJECT=default \
    MERKRAUM_PORT=8090 \
    MERKRAUM_HOST=0.0.0.0 \
    NEO4J_URI=bolt://neo4j:7687 \
    NEO4J_USER=neo4j \
    NEO4J_PASSWORD=merkraum-local \
    QDRANT_URL=http://qdrant:6333

EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8090/mcp', timeout=5)" || exit 1

# Run MCP server in HTTP mode (container-to-container)
CMD ["python", "merkraum_mcp_server.py", "--transport", "http"]
