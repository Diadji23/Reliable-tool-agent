"""Talk to the agent in the terminal, as one customer, and see what it does behind the scenes.

Needs a running llama-server (in another terminal):

    llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --jinja -c 8192 --port 8080 -t 4

Then:

    uv run python scripts/chat.py --customer C0001

Commands: /orders (the customer's orders, read directly, without the model), /new (new
conversation), /trace (whole conversation as the model sees it), /quit.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

import httpx
from langgraph.checkpoint.memory import InMemorySaver
from mcp import Client

from reliable_tool_agent import tools
from reliable_tool_agent.agent_graph import GraphAgent
from reliable_tool_agent.agent_loop import McpToolbox
from reliable_tool_agent.llama_server import LlamaServerModel
from reliable_tool_agent.mcp_server import build_server
from reliable_tool_agent.prompts import system_prompt

GREY, CYAN, YELLOW, BOLD, RESET = "\033[90m", "\033[36m", "\033[33m", "\033[1m", "\033[0m"


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


def show_orders(customer_id: str) -> None:
    for o in tools.list_my_orders(tools.Session(customer_id)):
        print(
            f"{GREY}  {o.id}  {o.status:<10} ordered {o.ordered_at}  "
            f"{o.total_cents / 100:.2f} EUR  {o.n_items} item(s){RESET}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--customer", default="C0001", help="authenticated customer id")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--thinking", action="store_true", help="let the model think aloud")
    args = parser.parse_args()
    logging.getLogger("httpx").setLevel(logging.WARNING)  # no line per HTTP request

    try:
        httpx.get(args.base_url.removesuffix("/v1") + "/health", timeout=3).raise_for_status()
    except httpx.HTTPError:
        print(
            "llama-server is not reachable. Start it in another terminal:\n"
            "  llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --jinja -c 8192 --port 8080 -t 4"
        )
        return

    model = CountingModel(LlamaServerModel(base_url=args.base_url, thinking=args.thinking))
    async with Client(build_server(tools.Session(args.customer))) as client:
        agent = GraphAgent(
            model, McpToolbox(client), system_prompt(), InMemorySaver(), args.max_steps
        )
        thread, n_thread, seen = "1", 1, 0
        print(
            f"{BOLD}Brindle Home support{RESET} - customer {args.customer}. "
            "Commands: /orders /new /trace /quit"
        )
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
                show_orders(args.customer)
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
            start = time.perf_counter()
            result = await agent.ask(thread, text)
            seconds = time.perf_counter() - start
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
