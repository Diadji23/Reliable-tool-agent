"""Replay every model call of traces.jsonl with llama-server's prompt cache off.

Measures what the run would cost if the whole conversation were processed again at each
step, to check the estimate made from the fitted latency model.

    uv run python experiments/01-first-real-run/replay_no_cache.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from reliable_tool_agent.prompts import system_prompt

HERE = Path(__file__).parent
URL = "http://127.0.0.1:8080/v1/chat/completions"


def main() -> None:
    tools = json.loads((HERE / "tools.json").read_text())
    rows = []
    for line in (HERE / "traces.jsonl").open():
        record = json.loads(line)
        messages = [{"role": "system", "content": system_prompt()}, *record["messages"]]
        # The k-th model call saw every message before the k-th assistant message.
        cuts = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        for k, cut in enumerate(cuts):
            payload = {
                "messages": messages[:cut],
                "tools": tools,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
                "cache_prompt": False,
            }
            start = time.perf_counter()
            body = httpx.post(URL, json=payload, timeout=600).json()
            seconds = time.perf_counter() - start
            cached = record["model_calls"][k]["seconds"]
            rows.append(
                {
                    "id": record["id"],
                    "call": k,
                    "no_cache_s": round(seconds, 2),
                    "cached_s": cached,
                    "timings": body.get("timings"),
                }
            )
            print(f"{record['id']} call {k}: no cache {seconds:6.1f}s  (run: {cached:5.1f}s)")
    (HERE / "replay_no_cache.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    total_no, total_c = sum(r["no_cache_s"] for r in rows), sum(r["cached_s"] for r in rows)
    print(f"total: no cache {total_no:.0f}s, with cache {total_c:.0f}s, x{total_no / total_c:.1f}")


if __name__ == "__main__":
    main()
