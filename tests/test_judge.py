"""The judge's mechanics, with a scripted model: what it is shown, how its reply is read."""

import pytest

from reliable_tool_agent.agent_loop import AssistantTurn
from reliable_tool_agent.fake_model import ScriptedModel
from reliable_tool_agent.judge import (
    JudgeOutputError,
    judge,
    judge_messages,
    load_policies,
    parse_verdict,
    render_transcript,
)

pytestmark = pytest.mark.anyio

CONVERSATION = [
    {"role": "user", "content": "Where is O10004?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "function": {"name": "get_order", "arguments": '{"order_id":"O10004"}'}}
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": '{"status": "delivered"}'},
    {"role": "assistant", "content": "It was delivered."},
]


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_transcript_shows_customer_tool_calls_results_and_answers_in_order():
    assert render_transcript(CONVERSATION).splitlines() == [
        "CUSTOMER: Where is O10004?",
        'ASSISTANT calls get_order({"order_id":"O10004"})',
        'TOOL RESULT: {"status": "delivered"}',
        "ASSISTANT: It was delivered.",
    ]


def test_judge_is_shown_the_policies_the_expected_behaviour_and_the_conversation():
    policies = load_policies()
    system, user = judge_messages("Delivered on 2026-08-07.", CONVERSATION, policies)
    assert "within **30 days of delivery**" in policies  # the real policy files were read
    assert policies in system["content"]
    assert "Today is 2026-10-15" in system["content"]
    assert "Delivered on 2026-08-07." in user["content"]
    assert "ASSISTANT: It was delivered." in user["content"]


def test_verdict_is_read_with_or_without_a_code_fence():
    fenced = '```json\n{"critique": "Fine.", "verdict": "pass"}\n```'
    assert parse_verdict(fenced).verdict == "pass"
    assert parse_verdict('{"critique": "Invents a date.", "verdict": "fail"}').verdict == "fail"


async def test_malformed_reply_gets_one_retry_that_quotes_the_error():
    model = ScriptedModel(
        [
            AssistantTurn('{"critique": "Fine.", "verdict": "maybe"}'),
            AssistantTurn('{"critique": "Fine.", "verdict": "pass"}'),
        ]
    )
    verdict = await judge(model, "Delivered.", CONVERSATION, "policies")
    assert verdict.verdict == "pass"
    retry = model.seen[1]
    assert retry[-2]["content"] == '{"critique": "Fine.", "verdict": "maybe"}'
    assert "not a valid verdict" in retry[-1]["content"]


async def test_second_malformed_reply_is_an_error_not_a_verdict():
    model = ScriptedModel([AssistantTurn("pass"), AssistantTurn("I think it passes.")])
    with pytest.raises(JudgeOutputError):
        await judge(model, "Delivered.", CONVERSATION, "policies")
