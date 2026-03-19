#!/bin/bash
# Start merkraum API server
# Connects to local Neo4j (bolt://localhost:7687) and serves REST API on port 8083
# CORS configured for: app.merkraum.de, localhost:3000, localhost:5173

cd /home/vsg/merkraum

# Check if already running
if lsof -ti:8083 >/dev/null 2>&1; then
    echo "Merkraum API already running on port 8083"
    curl -s http://localhost:8083/api/health
    exit 0
fi

# Start the API server
nohup python3 merkraum_api.py --host 0.0.0.0 --port 8083 > /tmp/merkraum_api.log 2>&1 &
echo "Started merkraum API (PID: $!)"
sleep 2
curl -s http://localhost:8083/api/health
