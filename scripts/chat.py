"""Talk to the agent in the terminal, as one customer, and see what it does behind the scenes.

Needs a running llama-server (in another terminal):

    llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --jinja -c 8192 --port 8080 -t 4

Then:

    uv run python scripts/chat.py --customer C0001

Commands: /orders (the customer's orders, read directly, without the model), /new (new
conversation), /trace (whole conversation as the model sees it), /quit.

Before a refund is issued, the chat stops and you answer as the support team reviewer:
approve, or reject with a reason. The chat works on a temporary copy of the shop
database, so refunds you approve never change data/shop.db (used by the experiments).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import tempfile
import time
from pathlib import Path

import httpx
from langgraph.checkpoint.memory import InMemorySaver
from mcp import Client

from reliable_tool_agent import tools
from reliable_tool_agent.agent_graph import GraphAgent
from reliable_tool_agent.agent_loop import AgentResult, McpToolbox
from reliable_tool_agent.db import DEFAULT_DB_PATH
from reliable_tool_agent.llama_server import LlamaServerModel
from reliable_tool_agent.mcp_server import build_server
from reliable_tool_agent.prompts import system_prompt

GREY, CYAN, YELLOW, BOLD, RESET = "\033[90m", "\033[36m", "\033[33m", "\033[1m", "\033[0m"
RED = "\033[31m"


class CountingModel:
    """Wraps the model to add up the tokens of each turn."""

    def __init__(self, inner):
        self.inner = inner
        self.prompt_tokens = self.completion_tokens = 0

    async def complete(self, messages, tool_specs):
        turn = await self.inner.complete(messages, tool_specs)
        if turn.usage:
            self.prompt_tokens += turn.usage.get("prompt_tokens", 0)
            self.completion_tokens += turn.usage.get("completion_tokens", 0)
        return turn


def short(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " ..."


def show_turn(new_messages: list[dict]) -> None:
    """Print the tool calls and results of one turn, in grey."""
    for m in new_messages:
        for call in m.get("tool_calls", []) if m["role"] == "assistant" else []:
            f = call["function"]
            print(f"{GREY}  -> {f['name']}({f['arguments']}){RESET}")
        if m["role"] == "tool":
            print(f"{GREY}  <- {short(m['content'])}{RESET}")


def show_trace(messages: list[dict]) -> None:
    for m in messages:
        role = m["role"]
        if m.get("content"):
            print(f"{YELLOW}[{role}]{RESET} {m['content']}")
        for call in m.get("tool_calls", []):
            f = call["function"]
            print(f"{YELLOW}[{role}]{RESET} (tool call) {f['name']} {f['arguments']}")


def euros(cents: int) -> str:
    return f"{cents / 100:.2f} EUR"


def review(pending: list[dict]) -> dict[str, dict]:
    """Ask the person at the terminal, as the support team, to decide each request."""
    decisions = {}
    for request in pending:
        q = request["preview"]
        print(f"\n{RED}{BOLD}APPROVAL NEEDED{RESET} {RED}(you are the support team){RESET}")
        print(f"  {request['tool']}: order {q['order_id']}, {q['quantity']} x {q['product_name']}")
        print(f"  condition: {q['condition']}")
        print(
            f"  amount:    {euros(q['amount_cents'])} = {euros(q['item_cents'])} item(s) "
            f"+ {euros(q['shipping_cents'])} shipping"
        )
        if q["already_refunded_cents"]:
            print(f"  already refunded on this order: {euros(q['already_refunded_cents'])}")
        try:
            answer = input(f"{BOLD}approve? [y/N]{RESET} ").strip().lower()
            reason = "" if answer in ("y", "yes") else input(f"{BOLD}reason:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{GREY}(no decision: the remaining requests count as rejected){RESET}")
            break  # the agent treats a request without a decision as rejected
        if answer in ("y", "yes"):
            decisions[request["tool_call_id"]] = {"approved": True}
        else:
            decisions[request["tool_call_id"]] = {
                "approved": False,
                "reason": reason or "not approved by the support team",
            }
    return decisions


def show_orders(session: tools.Session) -> None:
    for o in tools.list_my_orders(session):
        print(
            f"{GREY}  {o.id}  {o.status:<10} ordered {o.ordered_at}  "
            f"{euros(o.total_cents)}  {o.n_items} item(s){RESET}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--customer", default="C0001", help="authenticated customer id")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--thinking", action="store_true", help="let the model think aloud")
    args = parser.parse_args()
    logging.getLogger("httpx").setLevel(logging.WARNING)  # no line per HTTP request
    # The in-process MCP server logs each refused tool call at INFO; the chat already shows
    # the refusal as the tool result.
    logging.getLogger("mcp").setLevel(logging.WARNING)

    try:
        httpx.get(args.base_url.removesuffix("/v1") + "/health", timeout=3).raise_for_status()
    except httpx.HTTPError:
        print(
            "llama-server is not reachable. Start it in another terminal:\n"
            "  llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --jinja -c 8192 --port 8080 -t 4"
        )
        return

    model = CountingModel(LlamaServerModel(base_url=args.base_url, thinking=args.thinking))
    workdir = Path(tempfile.mkdtemp(prefix="brindle-chat-"))
    try:
        await chat(args, model, workdir)
    finally:
        shutil.rmtree(workdir)


async def chat(args: argparse.Namespace, model: CountingModel, workdir: Path) -> None:
    session = tools.Session(args.customer, shutil.copyfile(DEFAULT_DB_PATH, workdir / "shop.db"))
    async with Client(build_server(session)) as client:
        agent = GraphAgent(
            model, McpToolbox(client), system_prompt(), InMemorySaver(), args.max_steps
        )
        thread, n_thread, seen = "1", 1, 0
        print(
            f"{BOLD}Brindle Home support{RESET} - customer {args.customer}. "
            "Commands: /orders /new /trace /quit"
        )
        print(f"{GREY}(working on a copy of the shop database: {session.db_path}){RESET}")
        while True:
            try:
                text = input(f"\n{BOLD}you>{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text == "/quit":
                break
            if text == "/orders":
                show_orders(session)
                continue
            if text == "/new":
                n_thread += 1
                thread, seen = str(n_thread), 0
                print(f"{GREY}(new conversation){RESET}")
                continue
            if text == "/trace":
                show_trace(await agent.history(thread))
                continue

            model.prompt_tokens = model.completion_tokens = 0
            seconds = 0.0  # agent time only: the time you spend deciding is not counted
            start = time.perf_counter()
            result: AgentResult = await agent.ask(thread, text)
            seconds += time.perf_counter() - start
            while result.stop_reason == "awaiting_approval":
                show_turn(result.messages[seen:])
                seen = len(result.messages)
                decisions = review(result.pending)
                start = time.perf_counter()
                result = await agent.resume(thread, decisions)
                seconds += time.perf_counter() - start
            show_turn(result.messages[seen:])
            seen = len(result.messages)
            print(f"\n{CYAN}agent>{RESET} {result.answer or '(no answer: step budget used up)'}")
            print(
                f"{GREY}  [{result.steps} model call(s), {seconds:.1f} s, "
                f"{model.prompt_tokens} prompt + {model.completion_tokens} generated tokens"
                f"{', stopped by max_steps' if result.stop_reason == 'max_steps' else ''}]"
                f"{RESET}"
            )
        await model.inner.aclose()


if __name__ == "__main__":
    asyncio.run(main())
