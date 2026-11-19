# The support agent's HTTP API in one image. The model is not inside: the API reaches an
# OpenAI-compatible server through MODEL_BASE_URL (llama-server on the host, or a hosted API).
#
#   docker build -t brindle-agent .
#   docker run --rm --network host \
#     -e BRINDLE_TOKENS="demo-c21=customer:C0021,demo-rev=reviewer" brindle-agent
#
# --network host (Linux): the container shares the machine's network, so it reaches a
# llama-server listening on 127.0.0.1:8080 (the default MODEL_BASE_URL) and serves on :8000.

FROM python:3.12-slim

# uv, at the version used to write uv.lock, copied from its official image.
COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /bin/uv

WORKDIR /app
# Compile to bytecode once at build time (faster start); copy files instead of linking
# them from uv's cache, which is not kept in the image.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# The libraries first, from the lock file only: this layer is rebuilt when the
# dependencies change, not at every code change. Only the runtime libraries: no tests,
# notebooks or MLflow.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-default-groups --no-install-project

# Then the project itself, and the shop database (generated, not in git).
COPY README.md LICENSE ./
COPY src/ src/
COPY scripts/build_db.py scripts/
COPY data/policies/ data/policies/
RUN uv sync --locked --no-default-groups \
    && .venv/bin/python scripts/build_db.py

# Not root: a flaw in the app would not own the container.
RUN useradd --create-home agent
USER agent

ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
EXPOSE 8000
# The port: $PORT when the platform sets one (Render does), else 8000. A shell is needed to
# read it; `exec` makes uvicorn replace that shell, so it gets the platform's stop signal.
ENV PORT=8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "exec uvicorn reliable_tool_agent.api:app_from_env --factory --host 0.0.0.0 --port $PORT"]
