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
return questions with tools, and every refund waits for a human reviewer.

## Try it

Open `/docs` on this Space (the interactive API page), click **Authorize** and paste a
token, then **Try it out** on a route:

| Token | Role |
|---|---|
| `demo-customer` | customer C0021: start a conversation, ask about orders, ask for a refund |
| `demo-reviewer` | the shop's reviewer: see pending refunds, approve or reject them |

A conversation to try, as `demo-customer`: `POST /conversations`, then `POST
/conversations/{id}/messages` with `{"text": "My pillow from O10040 arrived torn, please
refund it."}`. The turn stops at `awaiting_approval`. Switch to `demo-reviewer`: `GET
/approvals` shows the quote and your words, `POST /approvals/{id}` decides.

**This is a shared demo.** The data is fictional and resets when the Space restarts; the
reviewer token sees every visitor's pending messages, so type nothing personal. The model
runs on a free quota: when it is used up, the API answers 503 and the demo is back the
next day. An answer can take up to a minute.

Code, design, evaluation and limits: https://github.com/Diadji23/Reliable-tool-agent
EOF
