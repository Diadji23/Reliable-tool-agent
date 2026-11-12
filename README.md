# reliable-tool-agent

A customer-support agent for a fictional home-goods shop, built as a complete system
rather than a demo: tools served over MCP, LangGraph orchestration, human approval before
any refund, guardrails measured against attacks, and an evaluation that checks the
**final state of the database**, not only the wording of the answer. Runs on a 4B local
model on a laptop CPU, at zero cost.

> **Status: work in progress.** Built: the shop, six tools over MCP, the agent (hand-written
> loop and LangGraph graph), conversation memory, human approval, the first guardrails.
> Planned: tracing, the full evaluation with a judge, web API, Docker, CI.

![Architecture](docs/architecture.png)

*Solid boxes are built, dashed grey boxes are planned. Source: [`docs/architecture.tex`](docs/architecture.tex) (TikZ).*

## What makes it reliable

- **The customer identity never goes through the model.** It comes from the session; no
  tool takes a customer id, and another customer's order looks missing.
- **Deterministic work is done in code.** Dates, return eligibility and refund amounts are
  computed in Python; the model chooses the tool and words the answer.
- **No refund without a human, for an exact amount.** The graph pauses before
  `refund_order`; the reviewer sees the amount computed by the code and the customer's own
  words, and the tool re-checks that amount in the same transaction as the write.
- **Least privilege in the database.** Reads are read-only; the refund tool's connection
  can insert into one table and do nothing else.
- **Untrusted text is kept away from the model.** A carrier note planted with prompt
  injections never reaches it ([threat model](docs/threat-model.md)).
- **Tested without a model.** 152 tests drive the agent with a scripted model against the
  real MCP server; deliberately breaking the code makes them fail.

Details, reasons and known limits: [`docs/design.md`](docs/design.md).

## Measured results

| Experiment | Finding |
|---|---|
| [01 — first real run](experiments/01-first-real-run/README.md) | 8/9 questions correct; 20–41 s per question on CPU; prompt cache cuts total time by 3.1x |
| [02 — return eligibility](experiments/02-return-eligibility/README.md) | Date error that appeared only mid-conversation, never on the question alone; fixed by moving the rule into code: 10/10 |
| [03 — guardrails](experiments/03-guardrail-baseline/README.md) | Indirect prompt injections resisted: 2/7 before, 7/7 after keeping the carrier note away from the model; 20/23 attack and borderline cases overall |

Every experiment states its hypotheses before the run and reads the answers by hand.

## Run it

```bash
uv sync
uv run python scripts/build_db.py   # writes data/shop.db
uv run pytest
```

Talk to the agent as one customer (needs [llama.cpp](https://github.com/ggml-org/llama.cpp)
and a GGUF model, here Qwen3.5-4B Q4_K_M):

```bash
llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --jinja -c 8192 --port 8080 -t 4   # terminal 1
uv run python scripts/chat.py --customer C0001                                   # terminal 2
```

The chat shows each tool call, the time and the tokens of every turn, and asks you, as the
support team, to approve or reject each refund. It works on a temporary copy of the database.

## Repository layout

```
src/reliable_tool_agent/   world, database access, tools, MCP server, agent loop and graph
scripts/                   build the database, chat, run experiments
experiments/               one folder per experiment: hypotheses, protocol, traces, results
evaluation/                attack cases and cases found in chat sessions
docs/                      design, threat model, architecture diagram
tests/                     unit tests (no model loaded)
```

## License

MIT
