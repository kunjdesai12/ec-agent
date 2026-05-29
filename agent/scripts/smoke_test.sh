#!/usr/bin/env bash
# Quick smoke test against a running API. Assumes API at localhost:8000.
set -euo pipefail

API="${API:-http://localhost:8000}"
SID="smoke-$(date +%s)"

echo "== health =="
curl -s "$API/v1/health" | jq .

echo
echo "== sync chat: discovery =="
curl -s -X POST "$API/v1/chat/sync" \
  -H 'content-type: application/json' \
  -d "{\"session_id\":\"$SID\",\"message\":\"Show me something spicy and non-veg\"}" | jq .

echo
echo "== sync chat: follow-up =="
curl -s -X POST "$API/v1/chat/sync" \
  -H 'content-type: application/json' \
  -d "{\"session_id\":\"$SID\",\"message\":\"Order 2 of the first one to my home address\"}" | jq .

echo
echo "== streaming chat =="
curl -N -X POST "$API/v1/chat" \
  -H 'content-type: application/json' \
  -d "{\"session_id\":\"$SID-stream\",\"message\":\"Get me biryani under 400 rupees\"}"
