#!/bin/bash
# Smoke tests for the NMN public API.
# Usage: BASE_URL=https://norskmeteornettverk.no/api/v1 API_KEY=ak_live_... ./test_api.sh

set -u
BASE_URL=${BASE_URL:-http://localhost/api/v1}
API_KEY=${API_KEY:-}

echo "Testing $BASE_URL"

echo "--- GET /stations (anonymous) ---"
curl -s -o /tmp/api_stations.json -w "HTTP %{http_code}\n" "$BASE_URL/stations" | tail -1
head -c 200 /tmp/api_stations.json; echo

echo "--- GET /cameras/fovs (anonymous) ---"
curl -s -o /tmp/api_fovs.json -w "HTTP %{http_code}\n" "$BASE_URL/cameras/fovs"

echo "--- GET /kp (anonymous) ---"
curl -s -o /tmp/api_kp.json -w "HTTP %{http_code}\n" "$BASE_URL/kp"

echo "--- POST /predict/passes (anonymous, queued) ---"
PASSES=$(curl -s -X POST -w "\nHTTP %{http_code}\n" "$BASE_URL/predict/passes?station=ams173&days=1")
echo "$PASSES" | tail -1
echo "$PASSES" | head -1 | tee /tmp/api_pass_task.json
PASS_TASK=$(python3 -c "import json,sys; print(json.loads(sys.stdin.read())['task_id'])" < /tmp/api_pass_task.json 2>/dev/null || echo '')

echo "--- GET /predict/passes/{task_id} ---"
if [ -n "$PASS_TASK" ]; then
  curl -s -o /tmp/api_pass_status.json -w "HTTP %{http_code}\n" "$BASE_URL/predict/passes/$PASS_TASK"
  head -c 200 /tmp/api_pass_status.json; echo
fi

echo "--- POST /downloads without key (expect 401) ---"
curl -s -X POST -w "HTTP %{http_code}\n" "$BASE_URL/downloads" -H 'Content-Type: application/json' -d '{}'

if [ -n "$API_KEY" ]; then
  echo "--- POST /downloads with key (small payload) ---"
  curl -s -X POST -o /tmp/api_download.json -w "HTTP %{http_code}\n" \
    "$BASE_URL/downloads" \
    -H 'Content-Type: application/json' \
    -H "X-API-Key: $API_KEY" \
    -d '{"files":[{"station_id":"ams173","cam":1,"time":"2026-08-31_20:00","file_type":"image"}]}'
  cat /tmp/api_download.json; echo
fi

echo "--- GET /docs/openapi.yaml ---"
curl -s -o /tmp/api_openapi.yaml -w "HTTP %{http_code}\n" "$BASE_URL/docs/openapi.yaml"
head -5 /tmp/api_openapi.yaml

echo "Done."
