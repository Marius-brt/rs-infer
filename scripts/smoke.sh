#!/usr/bin/env bash
# Smoke-test a running ortinfer server: SMOKE=http://host:port ./scripts/smoke.sh
set -euo pipefail
SMOKE=${SMOKE:-http://127.0.0.1:8080}
echo "== health";    curl -sf $SMOKE/health | jq -c
echo "== models";    curl -sf $SMOKE/v1/models | jq -r '.data[] | [.id,.kind,(.execution_providers|join("+"))] | @tsv'
echo "== embed";     curl -sf $SMOKE/v1/embeddings -H 'content-type: application/json' \
  -d '{"input":["The quick brown fox","A fast reddish fox","Quantum chromodynamics"]}' | jq -c '{n:(.data|length),dim:(.data[0].embedding|length),usage}'
echo "== embed b64"; curl -sf $SMOKE/v1/embeddings -H 'content-type: application/json' -d '{"input":"hi","encoding_format":"base64"}' | jq -c '.data[0].embedding | length'
echo "== rerank";    curl -sf $SMOKE/v1/rerank -H 'content-type: application/json' \
  -d '{"query":"capital of France","documents":["Paris is the capital of France.","Bananas are yellow."],"top_n":2}' | jq -c '.results'
echo "== score";     curl -sf $SMOKE/v1/score -H 'content-type: application/json' \
  -d '{"text_1":"capital of France","text_2":"Paris is the capital of France."}' | jq -c '.data'
echo "== pii";       curl -sf $SMOKE/pii/detect -H 'content-type: application/json' \
  -d '{"text":"Email john.smith@example.com to reach John Smith."}' | jq -c '.results[0].entities'
echo "== redact";    curl -sf $SMOKE/pii/redact -H 'content-type: application/json' \
  -d '{"text":"Email john.smith@example.com now."}' | jq -r '.results[0].text'
echo "== classify";  curl -sf $SMOKE/classify/zero-shot -H 'content-type: application/json' \
  -d '{"input":"The player scored a hat-trick","candidate_labels":["sports","politics","cooking"]}' | jq -c '{labels,scores}'
echo OK
