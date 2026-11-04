# reliable-tool-agent

A customer-support agent for a fictional home-goods shop, built as a complete system:
tools served over MCP, LangGraph orchestration, guardrails against prompt injection and
out-of-scope requests, human approval before refunds and cancellations, tracing, and an
evaluation that checks the **final state of the database** rather than the wording of
the answer.

> **Status: work in progress.** Built so far: the fictional shop, read-only access,
> three read-only tools served over MCP, and a hand-written agent loop tested with a scripted
> model (no LLM plugged in yet). Sections below describe what is built, not what is planned.

![Architecture](docs/architecture.png)

*Solid, coloured boxes are built; dashed grey boxes are planned. The customer identity
travels from the authenticated session straight to the tool server, never through the
model. Source: [`docs/architecture.tex`](docs/architecture.tex) (TikZ).*

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

## Tools over MCP

The tools live in `tools.py` as plain Python functions with Pydantic outputs, and
`mcp_server.py` exposes them through the Model Context Protocol, so any MCP client can
use them.

- **The customer identity never comes from the model.** The application starts the
  server for one authenticated customer (`--customer-id`); no tool schema contains a
  customer id. Another customer's order gets the same "not found" answer as a missing
  one, so its existence is not revealed.
- **Expected failures reach the model, crashes do not.** "Order not found" is raised as an
  MCP `ToolError`, whose message the model receives so it can recover. Any other exception
  is treated as a crash: the client gets a generic message and the details stay in the
  server log.

| Tool | Arguments | Returns |
|---|---|---|
| `get_order` | `order_id` | status, dates, items and amounts (integer cents) |
| `list_my_orders` | `status` (optional, one of 4 values) | the customer's orders, newest first |
| `search_policies` | `query` | up to 3 numbered policy rules, or an empty list |

Arguments are validated by Pydantic before a tool runs: `list_my_orders(status="lost")`
is rejected with the list of allowed values, so the model can correct itself.

`search_policies` is deliberately simple: overlap of crudely stemmed keywords
(returning / returned → return), ties broken in favour of the policy the query names.
It has no notion of synonyms ("send it back" does not find the returns policy; a test
documents this limit). The policies are short enough (~600 tokens) to fit in the prompt;
a search tool keeps the prompt small for local models on CPU, where every prompt token is
paid again at each step. Which option works better is a question for the evaluation.

```bash
uv run python -m reliable_tool_agent.mcp_server --customer-id C0001   # stdio server
```

## Agent loop

`agent_loop.py` is a deliberately minimal loop, written without a framework: ask the model
with the tool descriptions; if it requests tool calls, run them through the MCP client and
append each result as a `tool` message; stop when it answers with text, or after
`max_steps` model calls. Messages use the OpenAI chat-completions format, which both
llama.cpp's `llama-server` and most hosted APIs accept.

Every mistake the model can make (unknown tool, invalid JSON, arguments of the wrong
shape or value, an order that does not exist) is returned to it as text so it can
correct itself; only bugs on our side raise. Tests drive the loop with a scripted model
(`fake_model.py`) against the real MCP server.

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
  tools.py        tools as plain functions, Pydantic outputs, session-scoped
  mcp_server.py   MCP server exposing the tools (stdio)
  agent_loop.py   hand-written agent loop (model <-> MCP tools)
  fake_model.py   scripted model for tests
scripts/          build_db.py
docs/             architecture diagram (TikZ source + PNG)
data/policies/    shop policies (Markdown)
tests/            unit tests (no model loaded)
```

## License

MIT
