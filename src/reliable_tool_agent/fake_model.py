"""A scripted stand-in for an LLM: replays prepared turns, loads nothing.

Used by the tests (and to demonstrate the loop) so that the agent mechanics can be
checked exactly, without a model's randomness, cost or latency.
"""

from __future__ import annotations

import copy

from reliable_tool_agent.agent_loop import AssistantTurn, Message


class ScriptedModel:
    def __init__(self, turns: list[AssistantTurn]):
        self.turns = list(turns)
        self.seen: list[list[Message]] = []  # the messages received at each call

    async def complete(self, messages: list[Message], tools: list[dict]) -> AssistantTurn:
        self.seen.append(copy.deepcopy(messages))
        if not self.turns:
            raise AssertionError("ScriptedModel ran out of turns: the loop called it too often.")
        return self.turns.pop(0)
