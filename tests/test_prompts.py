"""The system prompt gives the model what it cannot know on its own."""

from __future__ import annotations

from datetime import date

from reliable_tool_agent.prompts import system_prompt


def test_prompt_states_today_so_the_return_window_can_be_applied():
    assert "Today is 2026-10-15." in system_prompt()
    assert "Today is 2027-01-02." in system_prompt(date(2027, 1, 2))


def test_prompt_explains_that_amounts_are_cents():
    assert "integer cents" in system_prompt()


def test_prompt_sends_return_questions_to_the_eligibility_tool():
    assert "check_return_eligibility" in system_prompt()
