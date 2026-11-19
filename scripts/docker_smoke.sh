#!/usr/bin/env bash
# Smoke test of the Docker image: does it start, and does it check tokens? No model is
# needed: these routes do not call it. Used by CI after `docker build`.
#
#   docker build -t brindle-agent . && scripts/docker_smoke.sh brindle-agent
set -euo pipefail

image="${1:-brindle-agent}"
port=8000
name="brindle-smoke-$$"

docker run -d --name "$name" -p "$port:8000" \
    -e BRINDLE_TOKENS="smoke-c21=customer:C0021,smoke-rev=reviewer" "$image" >/dev/null
trap 'docker rm -f "$name" >/dev/null' EXIT

# Wait for the server (up to 30 s).
for _ in $(seq 30); do
    curl -sf "localhost:$port/health" >/dev/null && break
    sleep 1
done

check() {  # check <expected status> <description> <curl arguments...>
    local expected="$1" what="$2"
    shift 2
    local got
    got=$(curl -s -o /dev/null -w "%{http_code}" "$@")
    if [[ "$got" != "$expected" ]]; then
        echo "FAIL $what: expected $expected, got $got"
        docker logs "$name"
        exit 1
    fi
    echo "ok   $what ($got)"
}

check 200 "health, no token" "localhost:$port/health"
check 401 "no token is refused" -X POST "localhost:$port/conversations"
check 201 "a customer starts a conversation" \
    -X POST -H "Authorization: Bearer smoke-c21" "localhost:$port/conversations"
check 403 "a customer cannot list approvals" \
    -H "Authorization: Bearer smoke-c21" "localhost:$port/approvals"
check 200 "the reviewer lists approvals" \
    -H "Authorization: Bearer smoke-rev" "localhost:$port/approvals"
