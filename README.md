# reliable-tool-agent

A customer-support agent for a fictional home-goods shop, built as a complete system:
tools served over MCP, LangGraph orchestration, guardrails against prompt injection and
out-of-scope requests, human approval before refunds and cancellations, tracing, and an
evaluation that checks the **final state of the database** rather than the wording of
the answer.

> **Status: work in progress.** Only the foundations exist so far (step 0). Sections
> below describe what is built, not what is planned.

## The shop

"Brindle Home" is invented on purpose: a well-known sample database could have been
memorised by the model, and the evaluation would then measure recall instead of tool use.

- `src/reliable_tool_agent/world.py` generates the SQLite database deterministically
  (fixed seed, fixed "today" = 2026-10-15 so that the 30-day return window always gives
  the same answer). Money is stored in integer cents.
- `data/policies/` holds the shop policies the agent must follow (returns, refunds,
  cancellations, privacy, scope).
- `src/reliable_tool_agent/db.py` opens the database read-only, with two independent
  locks (`mode=ro` and `PRAGMA query_only`).

## Setup

```bash
uv sync
uv run python scripts/build_db.py   # writes data/shop.db
uv run pytest
```

## Repository layout

```
src/reliable_tool_agent/
  world.py        schema and deterministic data generator of the fictional shop
  db.py           read-only database access
scripts/          build_db.py
data/policies/    shop policies (Markdown)
tests/            unit tests (no model loaded)
```

## License

MIT
