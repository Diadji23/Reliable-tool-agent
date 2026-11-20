# Experiment 06 — the model the public demo will run

**Question.** The free deployment cannot run Qwen3.5-4B on a shared CPU (about 2 minutes
for a first answer on the laptop already), so the demo calls a hosted model on Groq's
free tier. Every result so far is about the 4B model. Does the agent stay as reliable
with the hosted one, and how slow do the free-tier limits make it?

## Choosing the model (before the run)

Models on the Groq key that can call tools: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`,
`qwen/qwen3.8-27b`. All three have the same free limits, read from the response headers:
1000 requests a day and 7000–8000 input tokens a minute. One probe each on
`t-refund-defective-with-shipping` (one sample, not a rate):

- `gpt-oss-20b`: refund written, but the answer is in **German** and gives a wrong amount
  (51.90 instead of 50.90).
- `gpt-oss-120b`: says "I'll process the refund now" and stops: no refund written.
- `qwen/qwen3.8-27b`: right refund, right answer. 74 s, mostly spent waiting out the
  per-minute token limit (about 2000 input tokens per call, limit 7000 a minute).

Chosen: **`qwen/qwen3.8-27b`**, the only correct probe and the same family as the
evaluated 4B model. Its cost is speed, measured below.

## Hypotheses (written before the run)

- **H1 — final state at least as good as the 4B model:** at least 27/28 tasks end in the
  expected database state (the 4B model: 26/28 on the first run, the 5 refunds written
  once the simulated customer answers "yes").
- **H2 — the words are better than the 4B model's:** at most 4 of 28 answers fail by hand
  (4B: 6). The failures that come from missing facts in the context should remain
  (`chat-03`, promise after a rejection, is a reviewer-reason issue, not a size issue).
- **H3 — slow because of the quota, not the model:** median turn above 20 s, with each
  model call under 2 s once the request is accepted; the rest is waiting for the
  per-minute token limit.

## Protocol

`qwen/qwen3.8-27b` on Groq's free tier, temperature 0, thinking switch not sent (as in
the API: `api.app_from_env` with `MODEL_API_KEY` set), up to 8 retries honouring
`retry-after`. Code: the commit that adds this file. Runner:
[`scripts/run_eval.py`](../../scripts/run_eval.py) `--set tasks --base-url
https://api.groq.com/openai/v1 --model qwen/qwen3.8-27b --mlflow`, same 28 tasks and
scripted reviewer as experiment 04. Every answer is read by hand against its `expected`
field. One run per task: one sample, not a rate.

**First attempt stopped at task 17** (`t-refund-rejected-by-reviewer`): Groq answered
`400 tool_use_failed`. It checks the generated arguments against the tool schema itself,
and the model had written `condition = "def defective_or_wrong"`. The agent loop would
have shown the model its mistake, but the bad call never reached it: the whole turn
crashed (in the API, a 500 for the customer). The probe of `qwen3.8-27b` had hit the same
400 once before passing. Fix (commit after this note's first version): the adapter asks
again with the server's error added, at most twice. The run below is a full re-run of
the 28 tasks on that code; the hypotheses are unchanged.

## Results

Files: [`traces.jsonl`](traces.jsonl) (21 tasks) and [`traces_part2.jsonl`](traces_part2.jsonl)
(the other 7, run the next day). The re-run stopped twice: the laptop went to sleep, then
a DNS failure; the free quota then ran out (below). The answers are unaffected; the
timings of the first file are not usable.

**Final-state check: 28/28. Read by hand: 24/28** (Qwen3.5-4B, experiment 04: 26/28 and
22/28).

| Failure by hand | What happened |
|---|---|
| `t-refund-rejected-by-reviewer`, `chat-03` | After the quote, "Shall I go ahead and issue the refund?", although the customer had asked. The turn ends there: the refund never reaches the reviewer, so the rejection these tasks test is never exercised |
| `chat-04` | Offers to "exchange" the other item: there is no exchange policy |
| `t-policy-refund-method` | Right answer, but the text carries the model's leaked reasoning: "... `</think>` ..." and the answer twice. Fixed after the run: the adapter keeps only what follows `</think>` |

- **H1 supported:** 28/28 right final states, every refund amount as computed by hand.
- **H2 supported, with other failures:** 4 failures by hand (4B: 6). The 4B model's
  failures (promise after a rejection, offers without a tool, a mid-conversation apology
  for an error it did not make) are gone; `chat-01` keeps its correct answer without
  apologising. The new failure is the opposite of eagerness: asking for a confirmation the
  customer already gave. In the 5 approved-refund tasks the simulated "yes" covers it; in
  the 2 rejection tasks the question ends the turn.
- **H3 not measured as planned.** The first file's turn times include a laptop sleep. The
  second run shows the per-minute limit at work: 9–21 s waits before most calls of a
  conversation, turns of 1 to 115 s.

**What the run taught about the free tier.** The response headers show only the
per-minute limit; a second one, **200,000 tokens a day** (sliding 24 h), appeared only in
the 429 error text, after this experiment and the probes had used it all. Per task: about
3,300 tokens for a question, 10,000–14,000 for a refund conversation. The deployed demo
shares this key: about 60 questions or 15 refunds a day, fewer on a day an evaluation
runs. When the quota is out, the API answers 503 at once (`ModelUnavailableError`), and
the retry log is how the limit was found.

**Two crashes found and fixed during this experiment** (commits before the second file):
Groq's own validation of tool arguments (`400 tool_use_failed`) now goes back to the model
as a correction, and a turn that crashed no longer blocks its conversation.
