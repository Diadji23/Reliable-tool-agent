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
