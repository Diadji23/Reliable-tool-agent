#!/usr/bin/env bash
# Build the folder pushed to the Hugging Face Space: the files the Docker build needs and
# nothing else, plus a README whose YAML header configures the Space.
#
#   scripts/space_bundle.sh <out dir>
#
# The file list is the one .dockerignore lets in (its "!" lines): one list for "what the
# image needs", whether it is built here, in CI or by the Space. Hugging Face refuses
# plain-git binary files, and none of these is binary (the shop database is generated
# inside the image).
set -euo pipefail

out="${1:?usage: space_bundle.sh <out dir>}"
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$out"

cd "$root"
cp Dockerfile .dockerignore "$out/"
grep '^!' .dockerignore | cut -c2- | while read -r path; do
    mkdir -p "$out/$(dirname "$path")"
    cp -r "${path%/}" "$out/$(dirname "$path")/"
done
find "$out" \( -name __pycache__ -o -name '*.pyc' \) -prune -exec rm -rf {} +

# The Space's own README (replaces the project's, which the image only needs as a file).
cat > "$out/README.md" <<'EOF'
---
title: Brindle Home support agent
emoji: 📦
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: A tool-using support agent with human approval of refunds
---

# Brindle Home support agent

The HTTP API of a customer-support agent for a fictional shop: it answers order and
return questions with tools, and every refund waits for a human reviewer. Calls need a
bearer token (the demo's model runs on a free quota). Interactive API page: `/docs`.

Code, design, evaluation and limits: https://github.com/Diadji23/Reliable-tool-agent
EOF
